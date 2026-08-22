"""Build the modelling table: characters_raw plus is_onscreen and is_dead_v2.

is_dead_v2 differs from is_dead by keeping unparsed statuses (as alive) and 
restricting to on-screen characters. is_onscreen is `has_actor OR tmdb_match`.

  python -m src.labels.rebuild_label
"""

import json

import pandas as pd

from src.constants import (
    ACTOR_INFOBOX_KEYS,
    clean_infobox_field_count,
    is_non_character,
    is_unnamed_role,
)
from src.enrich.name_match import NameMatcher
from src.paths import CLEAN_DIR, RAW_DIR


class LabelRebuilder:

    @staticmethod
    def scan_flags() -> pd.DataFrame:
        """One pass over the raw scrape for on-screen signals and prominence."""
        matcher = NameMatcher.from_disk()

        rows = []
        for path in sorted(RAW_DIR.glob("*.jsonl")):
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if record.get("is_redirect") or is_non_character(record):
                        continue

                    infobox = record.get("infobox") or {}
                    credit = matcher.lookup(
                        record.get("name", ""), record.get("franchise", ""), infobox
                    )
                    rows.append({
                        "page_url": record.get("page_url", ""),
                        "franchise": record.get("franchise", ""),
                        "is_named": int(not is_unnamed_role(record.get("name", ""))),
                        # Field count with the death-only fields removed, so the
                        # feature stops partly counting the label.
                        "infobox_field_count_clean": clean_infobox_field_count(infobox),
                        # Kept nullable: rows the pipeline could not label are
                        # exactly what the corrected denominator needs.
                        "raw_is_dead": record.get("is_dead"),
                        "has_actor": int(
                            any(k.lower() in ACTOR_INFOBOX_KEYS for k in infobox)
                        ),
                        **credit,
                    })

        flags = pd.DataFrame(rows).drop_duplicates(subset="page_url")
        flags["is_onscreen"] = (
            (flags["has_actor"] == 1) | (flags["tmdb_match"] == 1)
        ).astype(int)
        return flags


def main() -> None:
    raw = pd.read_csv(CLEAN_DIR / "characters_raw.csv", low_memory=False)
    keys = pd.read_csv(CLEAN_DIR / "characters_keys.csv")
    raw = raw.reset_index(drop=True)
    raw["page_url"] = keys["page_url"].values
    raw["name"] = keys["name"].values

    print("Scanning raw scrape for on-screen signals ...")
    flags = LabelRebuilder.scan_flags()
    flags.to_csv(CLEAN_DIR / "onscreen_flags.csv", index=False, encoding="utf-8")

    df = raw.merge(flags.drop(columns=["franchise", "raw_is_dead"]),
                   on="page_url", how="left")
    for col in ("has_actor", "tmdb_match", "is_onscreen"):
        df[col] = df[col].fillna(0).astype(int)
    # Recomputed from the merged name so the column is right even for rows the
    # flags scan missed.
    df["is_named"] = (~df["name"].map(is_unnamed_role)).astype(int)

    # is_dead is already 0/1 here; characters_raw dropped the unparsed rows
    # upstream, so counting them as alive happens in the audit rather than now.
    df["is_dead_v2"] = df["is_dead"].astype(int)

    onscreen = df[df["is_onscreen"] == 1]
    named = onscreen[onscreen["is_named"] == 1]
    unnamed = onscreen[onscreen["is_named"] == 0]
    print(f"\n{len(df):,} characters, {len(onscreen):,} on screen "
          f"({len(onscreen) / len(df):.1%})")
    print(f"mortality all rows   {df['is_dead'].mean():.1%}")
    print(f"mortality on screen  {onscreen['is_dead_v2'].mean():.1%}")
    print(f"billing_order known  {onscreen['billing_order'].notna().mean():.1%}")

    # The analysis population, and what excluding unnamed background roles costs.
    print(f"\nanalysis population (on screen and named): {len(named):,}")
    print(f"  mortality          {named['is_dead_v2'].mean():.1%}")
    print(f"excluded unnamed roles: {len(unnamed):,} "
          f"({len(unnamed) / len(onscreen):.1%} of on screen)")
    print(f"  mortality          {unnamed['is_dead_v2'].mean():.1%}")

    out_path = CLEAN_DIR / "characters_model.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\nWrote {out_path}")

    # Franchise-level mortality belongs in src/q2_franchise/mortality.py, not
    # here: this table's population is characters_model.csv, which never
    # contains a row with no parseable infobox status (build_pipeline.py
    # drops those upstream), so any per-franchise breakdown computed from it
    # is silently biased for the franchises where most rows fail to parse.
    # q2_franchise_mortality.csv accounts for those rows properly.


if __name__ == "__main__":
    from src import setup_run_log
    setup_run_log(__spec__)
    main()
