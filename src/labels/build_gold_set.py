"""
Draw a stratified sample of characters for ground-truth annotation.

The sample is built to answer three specific questions, not to be
representative of the corpus:

  1. Are the rows the pipeline drops (is_dead=None) actually alive?
     If they are, the fix is to count them as alive rather than delete them.
     Weighted toward the franchises that lose the most rows, since that is
     where the answer changes the numbers.

  2. On dod_only wikis, does "has a date of death" mean the character dies
     during the story, or only that they are dead somewhere in the fictional
     chronology? Lord of the Rings reports 72% mortality, which is the shape a
     corpus takes when ancestors and historical figures count as deaths.

  3. How accurate is the status-based label when it does fire?

Output: data/gold/gold_sample.csv, one row per character, with the text an
annotator needs to decide. Deterministic given --seed.

Run:  python -m src.labels.build_gold_set
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re

import pandas as pd

from src.labels.audit_labels import load_regimes
from src.paths import GOLD_DIR, RAW_DIR

# How many characters to draw from each stratum. Weighted toward the dropped
# rows and the dod_only deaths because those are the two unknowns that actually
# change the corrected label; the status-based cells are there to measure
# precision of a label we already broadly trust.
STRATA_TARGETS = {
    ("status", "dropped"): 200,
    ("dod_only", "dead"): 150,
    ("status", "dead"): 100,
    ("dod_only", "alive"): 75,
    ("status", "alive"): 75,
}

DEATH_WORD_RE = re.compile(
    r"\b(died|dies|death|killed|kills|murdered|executed|slain|perished|"
    r"assassinated|deceased|sacrificed)\b",
    re.IGNORECASE,
)

# Infobox fields worth showing the annotator verbatim.
INTERESTING_KEYS = {
    "status", "death", "died", "dod", "date of death", "cause of death",
    "causeofdeath", "actor", "portrayed", "portrayed by", "played by",
    "species", "gender", "born", "birth", "first", "last", "appearances",
    "seasons", "episode", "deathep",
}


def char_id(page_url: str) -> str:
    """Stable id so annotations survive re-sampling and stay joinable."""
    return hashlib.sha1(page_url.encode("utf-8")).hexdigest()[:12]


def label_state(is_dead) -> str:
    if is_dead == 1:
        return "dead"
    if is_dead == 0:
        return "alive"
    return "dropped"


def death_sentences(text: str, limit: int = 3) -> str:
    """
    Up to `limit` sentences mentioning death.

    Extracted for every character, including those with none -- an empty result
    is itself evidence, and pulling these only for likely-dead characters would
    hand the annotator the answer.
    """
    hits = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if DEATH_WORD_RE.search(s)]
    return " ".join(hits[:limit])[:800]


def summarize_infobox(infobox: dict) -> str:
    parts = [f"{k}: {v}" for k, v in infobox.items() if k.lower() in INTERESTING_KEYS]
    return " | ".join(parts)[:600]


def collect(regimes: dict[str, str]) -> pd.DataFrame:
    rows = []
    files = sorted(RAW_DIR.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"No JSONL files in {RAW_DIR}.")

    for path in files:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("is_redirect"):
                    continue
                franchise = rec.get("franchise", "")
                regime = regimes.get(franchise)
                if regime is None:
                    continue
                text = (rec.get("page_text") or "").strip()
                if len(text) < 80:
                    continue  # stub pages carry no evidence to annotate
                infobox = rec.get("infobox") or {}
                rows.append(
                    {
                        "char_id": char_id(rec.get("page_url", "")),
                        "name": rec.get("name", ""),
                        "franchise": franchise,
                        "regime": regime,
                        "page_url": rec.get("page_url", ""),
                        "auto_is_dead": rec.get("is_dead"),
                        "state": label_state(rec.get("is_dead")),
                        "infobox_fields": summarize_infobox(infobox),
                        "death_sentences": death_sentences(text),
                        "text_excerpt": re.sub(r"\s+", " ", text[:1000]),
                    }
                )
    return pd.DataFrame(rows)


def stratified_sample(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """
    Sample each stratum, spreading draws across franchises.

    Within a stratum the franchises are visited round-robin so that Star Wars
    (56% of all pages) cannot swallow the cell -- a sample dominated by one wiki
    would measure that wiki's conventions rather than the corpus's.
    """
    rng = random.Random(seed)
    picked = []

    for (regime, state), target in STRATA_TARGETS.items():
        pool = df[(df["regime"] == regime) & (df["state"] == state)]
        if pool.empty:
            print(f"  WARNING: stratum ({regime}, {state}) is empty")
            continue

        by_franchise = {}
        for franchise, grp in pool.groupby("franchise"):
            ids = grp.index.tolist()
            rng.shuffle(ids)
            by_franchise[franchise] = ids

        # Visit franchises most-affected-first so a short pool still covers them.
        order = sorted(by_franchise, key=lambda f: -len(by_franchise[f]))
        chosen: list[int] = []
        while len(chosen) < target and any(by_franchise.values()):
            for franchise in order:
                if not by_franchise[franchise]:
                    continue
                chosen.append(by_franchise[franchise].pop())
                if len(chosen) >= target:
                    break

        got = pool.loc[chosen].copy()
        got["stratum"] = f"{regime}:{state}"
        picked.append(got)
        print(f"  {regime:9s} {state:8s}  target {target:4d}  got {len(got):4d}  "
              f"across {got['franchise'].nunique()} franchises")

    return pd.concat(picked, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260814,
                    help="RNG seed; the sample is deterministic given this")
    args = ap.parse_args()

    regimes = load_regimes()
    print("Collecting candidates from data/raw ...")
    df = collect(regimes)
    print(f"  {len(df):,} annotatable pages\n")

    print("Sampling strata:")
    sample = stratified_sample(df, args.seed)

    # A page can only be sampled once even if strata overlap.
    before = len(sample)
    sample = sample.drop_duplicates(subset="char_id").reset_index(drop=True)
    if len(sample) != before:
        print(f"\n  dropped {before - len(sample)} duplicate char_ids")

    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    out = GOLD_DIR / "gold_sample.csv"
    sample.to_csv(out, index=False, encoding="utf-8")

    print(f"\nSampled {len(sample)} characters from "
          f"{sample['franchise'].nunique()} franchises")
    print(sample["stratum"].value_counts().to_string())
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
