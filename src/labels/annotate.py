"""Split the gold sample into chunks for annotation and merge the results back.

Annotation runs as Haiku 4.5 subagents over the chunk files, two independent
passes, each writing one JSONL per chunk under data/gold/annotations/passN/.

  python -m src.labels.annotate --split --chunk-size 100
  python -m src.labels.annotate --merge --pass-id 1
"""

import argparse
import json

import pandas as pd

from src.constants import ANNOTATION_LABELS
from src.paths import GOLD_DIR

CHUNK_DIR = GOLD_DIR / "chunks"
ANNOT_DIR = GOLD_DIR / "annotations"

# Columns the annotator sees. auto_is_dead and state are withheld so the
# judgement comes from the evidence rather than the existing label.
CHUNK_COLUMNS = [
    "char_id", "name", "franchise", "infobox_fields",
    "death_sentences", "text_excerpt",
]


class AnnotationChunker:
    """Prepare annotation input and consolidate annotation output."""

    @staticmethod
    def split(chunk_size: int) -> None:
        sample = pd.read_csv(GOLD_DIR / "gold_sample.csv")
        CHUNK_DIR.mkdir(parents=True, exist_ok=True)

        for start in range(0, len(sample), chunk_size):
            chunk = sample.iloc[start : start + chunk_size][CHUNK_COLUMNS]
            path = CHUNK_DIR / f"chunk_{start // chunk_size:02d}.csv"
            chunk.to_csv(path, index=False, encoding="utf-8")
            print(f"  wrote {path.name}  ({len(chunk)} rows)")

        n_chunks = (len(sample) + chunk_size - 1) // chunk_size
        print(f"\n{len(sample)} characters -> {n_chunks} chunks of up to {chunk_size}")

    @staticmethod
    def merge(pass_id: int) -> pd.DataFrame:
        source = ANNOT_DIR / f"pass{pass_id}"
        files = sorted(source.glob("*.jsonl"))
        if not files:
            raise SystemExit(f"No annotation files in {source}")

        records, malformed = [], 0
        for path in files:
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    continue
                if "char_id" not in record or any(k not in record for k in ANNOTATION_LABELS):
                    malformed += 1
                    continue
                record["pass_id"] = pass_id
                records.append(record)

        df = pd.DataFrame(records)
        before = len(df)
        df = df.drop_duplicates(subset="char_id", keep="first")

        sample = pd.read_csv(GOLD_DIR / "gold_sample.csv")
        missing = set(sample["char_id"]) - set(df["char_id"])

        print(f"pass {pass_id}: {before} annotations from {len(files)} files")
        if malformed:
            print(f"  {malformed} malformed lines skipped")
        if before != len(df):
            print(f"  {before - len(df)} duplicate char_ids collapsed")
        if missing:
            print(f"  {len(missing)} characters still unannotated")

        out_path = ANNOT_DIR / f"pass{pass_id}_merged.csv"
        df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"  wrote {out_path}")
        return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", action="store_true")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--pass-id", type=int, default=1)
    args = parser.parse_args()

    if args.split:
        AnnotationChunker.split(args.chunk_size)
    if args.merge:
        AnnotationChunker.merge(args.pass_id)
    if not (args.split or args.merge):
        parser.error("pass --split or --merge")


if __name__ == "__main__":
    main()
