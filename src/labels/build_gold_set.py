import argparse
import hashlib
import json
import random
import re

import pandas as pd

from src.constants import (
    GOLD_EXCERPT_CHARS,
    GOLD_MIN_TEXT_CHARS,
    GOLD_SAMPLE_SEED,
    INTERESTING_INFOBOX_KEYS,
    STRATA_TARGETS,
)
from src.labels.audit_labels import LabelAuditor
from src.paths import GOLD_DIR, RAW_DIR

DEATH_WORD_RE = re.compile(
    r"\b(died|dies|death|killed|kills|murdered|executed|slain|perished|"
    r"assassinated|deceased|sacrificed)\b",
    re.IGNORECASE,
)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
WHITESPACE_RE = re.compile(r"\s+")


class GoldSetBuilder:
    """Draw a stratified sample of characters for annotation."""

    @staticmethod
    def char_id(page_url: str) -> str:
        """Stable id, so annotations survive a re-sample."""
        return hashlib.sha1(page_url.encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def label_state(is_dead) -> str:
        if is_dead == 1:
            return "dead"
        if is_dead == 0:
            return "alive"
        return "dropped"

    @staticmethod
    def death_sentences(text: str, limit: int = 3) -> str:
        """Up to `limit` death-mentioning sentences. Run for every character."""
        hits = [
            s.strip() for s in SENTENCE_SPLIT_RE.split(text)
            if DEATH_WORD_RE.search(s)
        ]
        return " ".join(hits[:limit])[:800]

    @staticmethod
    def summarize_infobox(infobox: dict) -> str:
        parts = [
            f"{k}: {v}" for k, v in infobox.items()
            if k.lower() in INTERESTING_INFOBOX_KEYS
        ]
        return " | ".join(parts)[:600]

    @staticmethod
    def collect(regimes: dict[str, str]) -> pd.DataFrame:
        files = sorted(RAW_DIR.glob("*.jsonl"))
        if not files:
            raise SystemExit(f"No JSONL files in {RAW_DIR}.")

        rows = []
        for path in files:
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if record.get("is_redirect"):
                        continue

                    franchise = record.get("franchise", "")
                    regime = regimes.get(franchise)
                    if regime is None:
                        continue

                    text = (record.get("page_text") or "").strip()
                    if len(text) < GOLD_MIN_TEXT_CHARS:
                        continue

                    infobox = record.get("infobox") or {}
                    rows.append({
                        "char_id": GoldSetBuilder.char_id(record.get("page_url", "")),
                        "name": record.get("name", ""),
                        "franchise": franchise,
                        "regime": regime,
                        "page_url": record.get("page_url", ""),
                        "auto_is_dead": record.get("is_dead"),
                        "state": GoldSetBuilder.label_state(record.get("is_dead")),
                        "infobox_fields": GoldSetBuilder.summarize_infobox(infobox),
                        "death_sentences": GoldSetBuilder.death_sentences(text),
                        "text_excerpt": WHITESPACE_RE.sub(
                            " ", text[:GOLD_EXCERPT_CHARS]
                        ),
                    })
        return pd.DataFrame(rows)

    @staticmethod
    def sample(df: pd.DataFrame, seed: int) -> pd.DataFrame:
        """Sample each stratum round-robin over franchises, so Star Wars can't fill a cell."""
        rng = random.Random(seed)
        picked = []

        for (regime, state), target in STRATA_TARGETS.items():
            pool = df[(df["regime"] == regime) & (df["state"] == state)]
            if pool.empty:
                print(f"  stratum ({regime}, {state}) is empty")
                continue

            by_franchise = {}
            for franchise, group in pool.groupby("franchise"):
                ids = group.index.tolist()
                rng.shuffle(ids)
                by_franchise[franchise] = ids

            order = sorted(by_franchise, key=lambda f: -len(by_franchise[f]))
            chosen: list[int] = []
            while len(chosen) < target and any(by_franchise.values()):
                for franchise in order:
                    if not by_franchise[franchise]:
                        continue
                    chosen.append(by_franchise[franchise].pop())
                    if len(chosen) >= target:
                        break

            selected = pool.loc[chosen].copy()
            selected["stratum"] = f"{regime}:{state}"
            picked.append(selected)
            print(f"  {regime:9s} {state:8s} target {target:4d}  got {len(selected):4d}  "
                  f"across {selected['franchise'].nunique()} franchises")

        return pd.concat(picked, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=GOLD_SAMPLE_SEED)
    args = parser.parse_args()

    regimes = LabelAuditor.load_regimes()
    print("Collecting candidates from data/raw ...")
    df = GoldSetBuilder.collect(regimes)
    print(f"  {len(df):,} annotatable pages\n")

    print("Sampling strata:")
    sample = GoldSetBuilder.sample(df, args.seed)
    sample = sample.drop_duplicates(subset="char_id").reset_index(drop=True)

    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    out_path = GOLD_DIR / "gold_sample.csv"
    sample.to_csv(out_path, index=False, encoding="utf-8")

    print(f"\nSampled {len(sample)} characters from "
          f"{sample['franchise'].nunique()} franchises")
    print(sample["stratum"].value_counts().to_string())
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    from src import setup_run_log
    setup_run_log(__spec__)
    main()
