"""Collect the characters no other source can label, for Haiku to annotate.

Labels are preferred in this order (see README, "Where labels come from"):

  1. a human-curated death registry, where one exists
  2. the infobox status / date-of-death field -- also human-written
  3. Claude Haiku 4.5, only for what neither of the above can settle

This script is step 3, and only step 3: it collects exactly the rows that
reach this point with no answer, so Haiku fills gaps rather than overrules
anyone. A row belongs here only if it is in the analysis population (on
screen, named), its infobox status could not be parsed, and no death
registry names it either -- registries only list deaths, so a registry
checks off the "dead" rows here but can never resolve the rest to "alive".

Output goes through the same chunk -> Haiku subagent -> merge pipeline as
the gold set (see annotate.py), into its own directory. It is never merged
into is_dead or is_dead_v2. rebuild_label.py reads it back as a clearly
separate is_dead_haiku column.

  python -m src.labels.fill_unlabelled --collect
  python -m src.labels.fill_unlabelled --merge --pass-id 1
"""

import argparse
import json
import re

import pandas as pd

from src.constants import ANNOTATION_LABELS, INTERESTING_INFOBOX_KEYS, analysis_population
from src.labels.build_gold_set import GoldSetBuilder
from src.labels.validate_registry import RegistryValidation
from src.paths import CLEAN_DIR, GOLD_DIR, RAW_DIR

FILL_DIR = GOLD_DIR / "fill_unlabelled"
CHUNK_DIR = FILL_DIR / "chunks"
ANNOT_DIR = FILL_DIR / "annotations"

CHUNK_COLUMNS = [
    "char_id", "name", "franchise", "infobox_fields",
    "death_sentences", "text_excerpt",
]

WHITESPACE_RE = re.compile(r"\s+")


class UnlabelledCollector:

    @staticmethod
    def registry_covered_names() -> dict[str, set[str]]:
        """franchise -> set of normalised names a death registry accounts for.

        Only franchises with a registry on disk. Absence from the registry
        does not mean alive, so this is used only to skip rows a registry
        already resolves to dead -- never to infer "alive".
        """
        return {
            "Game of Thrones": {
                RegistryValidation.normalise(n)
                for n in RegistryValidation.got_registry()
            },
            "The 100": {
                RegistryValidation.normalise(n)
                for n in RegistryValidation.the_100_registry()
            },
        }

    @staticmethod
    def collect() -> tuple[pd.DataFrame, pd.DataFrame]:
        # characters_model.csv cannot answer this question: build_pipeline.py
        # drops every row whose status could not be parsed before that table
        # exists (`df[df["is_dead"].notna()]`), so the rows this script needs
        # to find are never in it. onscreen_flags.csv is built straight from
        # data/raw/*.jsonl and keeps them, with raw_is_dead left null.
        flags = pd.read_csv(CLEAN_DIR / "onscreen_flags.csv")
        population = analysis_population(flags)
        unlabelled = population[population["raw_is_dead"].isna()]

        target_urls = set(unlabelled["page_url"])
        covered = UnlabelledCollector.registry_covered_names()

        print(f"analysis population: {len(population):,}")
        print(f"unlabelled (no parseable status): {len(unlabelled):,}")

        rows, registry_rows = [], []
        for path in sorted(RAW_DIR.glob("*.jsonl")):
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if record.get("page_url") not in target_urls:
                        continue

                    franchise = record.get("franchise", "")
                    name = record.get("name", "")
                    page_url = record.get("page_url", "")
                    names = covered.get(franchise)
                    if names is not None and RegistryValidation.normalise(name) in names:
                        # A registry already resolves this one to dead. Written
                        # out here rather than merely skipped, so the answer
                        # isn't silently discarded -- it still needs to reach
                        # mortality.py, and it must never be sent to Haiku.
                        registry_rows.append({
                            "page_url": page_url, "name": name, "franchise": franchise,
                        })
                        continue

                    text = (record.get("page_text") or "").strip()
                    infobox = record.get("infobox") or {}
                    rows.append({
                        "char_id": GoldSetBuilder.char_id(page_url),
                        "name": name,
                        "franchise": franchise,
                        "page_url": page_url,
                        "infobox_fields": GoldSetBuilder.summarize_infobox(infobox),
                        "death_sentences": GoldSetBuilder.death_sentences(text),
                        "text_excerpt": WHITESPACE_RE.sub(" ", text[:1000]),
                    })

        # The raw scrape can carry duplicate page_url rows (rebuild_label.py
        # dedupes for the same reason when it builds onscreen_flags.csv).
        result = pd.DataFrame(rows).drop_duplicates(subset="page_url")
        registry_filled = pd.DataFrame(
            registry_rows, columns=["page_url", "name", "franchise"]
        ).drop_duplicates(subset="page_url")
        print(f"resolved by a death registry: {len(registry_filled):,}")
        print(f"remaining for Haiku: {len(result):,}")
        return result, registry_filled

    @staticmethod
    def split(df: pd.DataFrame, chunk_size: int) -> None:
        CHUNK_DIR.mkdir(parents=True, exist_ok=True)
        for start in range(0, len(df), chunk_size):
            chunk = df.iloc[start:start + chunk_size][CHUNK_COLUMNS]
            path = CHUNK_DIR / f"chunk_{start // chunk_size:03d}.csv"
            chunk.to_csv(path, index=False, encoding="utf-8")
        n_chunks = (len(df) + chunk_size - 1) // chunk_size
        print(f"\n{len(df)} characters -> {n_chunks} chunks of up to {chunk_size} "
              f"in {CHUNK_DIR}")

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
                records.append(record)

        df = pd.DataFrame(records).drop_duplicates(subset="char_id", keep="first")
        if malformed:
            print(f"  {malformed} malformed lines skipped")

        out_path = FILL_DIR / "haiku_fill.csv"
        df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"wrote {out_path} ({len(df)} labelled)")
        return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--pass-id", type=int, default=1)
    args = parser.parse_args()

    if args.collect:
        df, registry_filled = UnlabelledCollector.collect()
        FILL_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(FILL_DIR / "to_annotate.csv", index=False, encoding="utf-8")
        registry_filled.to_csv(
            FILL_DIR / "registry_filled.csv", index=False, encoding="utf-8"
        )
        UnlabelledCollector.split(df, args.chunk_size)
    if args.merge:
        UnlabelledCollector.merge(args.pass_id)
    if not (args.collect or args.merge):
        parser.error("pass --collect or --merge")


if __name__ == "__main__":
    from src import setup_run_log
    setup_run_log(__spec__)
    main()
