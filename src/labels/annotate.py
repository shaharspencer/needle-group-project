"""
Split the gold sample into chunks for annotation, and merge the results back.

Annotation is done by Haiku 4.5 subagents, two independent passes with
different prompt framings. Each pass writes one JSONL file per chunk into
data/gold/annotations/pass{N}/. This module only prepares the input and
consolidates the output -- it does not call any model itself.

Three labels per character:

  is_character      Is this an individual character, as opposed to a species,
                    creature type, vehicle, organisation, location or concept?
                    The Jurassic Park wiki is full of game-only dinosaur
                    hybrids, and they are not people who can die in a story.

  appears_onscreen  Does this character appear in a film or TV episode, as
                    opposed to existing only in novels, comics or games? Star
                    Wars Legends contributes tens of thousands of characters
                    who were never filmed.

  dies_in_narrative Does the character die during the events of a filmed work?
                    Dying of old age generations before the story starts does
                    not count; that is what inflates the dod_only franchises.

Each is 1, 0, or -1 for "cannot tell from the text given".

Run:
  python -m src.labels.annotate --split --chunk-size 100
  python -m src.labels.annotate --merge --pass-id 1
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from src.paths import GOLD_DIR

CHUNK_DIR = GOLD_DIR / "chunks"
ANNOT_DIR = GOLD_DIR / "annotations"

LABELS = ("is_character", "appears_onscreen", "dies_in_narrative")


def split(chunk_size: int) -> None:
    sample = pd.read_csv(GOLD_DIR / "gold_sample.csv")
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)

    # Strip the columns that would let an annotator read the answer off the
    # pipeline instead of the evidence.
    cols = ["char_id", "name", "franchise", "infobox_fields",
            "death_sentences", "text_excerpt"]

    for i in range(0, len(sample), chunk_size):
        chunk = sample.iloc[i : i + chunk_size][cols]
        path = CHUNK_DIR / f"chunk_{i // chunk_size:02d}.csv"
        chunk.to_csv(path, index=False, encoding="utf-8")
        print(f"  wrote {path.name}  ({len(chunk)} rows)")

    n_chunks = (len(sample) + chunk_size - 1) // chunk_size
    print(f"\n{len(sample)} characters -> {n_chunks} chunks of up to {chunk_size}")


def merge(pass_id: int) -> pd.DataFrame:
    src_dir = ANNOT_DIR / f"pass{pass_id}"
    files = sorted(src_dir.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"No annotation files in {src_dir}")

    records, bad = [], 0
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if "char_id" not in rec or any(k not in rec for k in LABELS):
                bad += 1
                continue
            rec["pass_id"] = pass_id
            records.append(rec)

    df = pd.DataFrame(records)
    before = len(df)
    df = df.drop_duplicates(subset="char_id", keep="first")

    sample = pd.read_csv(GOLD_DIR / "gold_sample.csv")
    missing = set(sample["char_id"]) - set(df["char_id"])

    print(f"pass {pass_id}: {before} annotations from {len(files)} files")
    if bad:
        print(f"  {bad} malformed lines skipped")
    if before != len(df):
        print(f"  {before - len(df)} duplicate char_ids collapsed")
    if missing:
        print(f"  {len(missing)} characters still unannotated")

    out = ANNOT_DIR / f"pass{pass_id}_merged.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"  wrote {out}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--chunk-size", type=int, default=100)
    ap.add_argument("--pass-id", type=int, default=1)
    args = ap.parse_args()

    if args.split:
        split(args.chunk_size)
    if args.merge:
        merge(args.pass_id)
    if not (args.split or args.merge):
        ap.error("pass --split or --merge")


if __name__ == "__main__":
    main()
