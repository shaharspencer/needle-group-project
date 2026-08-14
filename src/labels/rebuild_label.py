"""Build the modelling table: characters_raw plus is_onscreen and is_dead_v2.

is_dead_v2 differs from is_dead in two ways:
  - characters whose infobox status could not be parsed count as alive rather
    than being dropped
  - the analysis is restricted to characters who appear on screen

is_onscreen is `has_actor OR tmdb_match`. Measured against the annotated
sample: precision 0.79, recall 0.77. Neither signal alone is enough -- an actor
field alone recalls 0.61, and a TMDB cast match alone 0.58.

  python -m src.labels.rebuild_label
"""

import json

import pandas as pd

from src.constants import ACTOR_INFOBOX_KEYS
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
                    if record.get("is_redirect"):
                        continue

                    infobox = record.get("infobox") or {}
                    credit = matcher.lookup(
                        record.get("name", ""), record.get("franchise", ""), infobox
                    )
                    rows.append({
                        "page_url": record.get("page_url", ""),
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
    df = raw.merge(flags, on="page_url", how="left")
    for col in ("has_actor", "tmdb_match", "is_onscreen"):
        df[col] = df[col].fillna(0).astype(int)

    # is_dead is already 0/1 here; characters_raw dropped the unparsed rows
    # upstream, so counting them as alive happens in the audit rather than now.
    df["is_dead_v2"] = df["is_dead"].astype(int)

    onscreen = df[df["is_onscreen"] == 1]
    print(f"\n{len(df):,} characters, {len(onscreen):,} on screen "
          f"({len(onscreen) / len(df):.1%})")
    print(f"mortality all rows   {df['is_dead'].mean():.1%}")
    print(f"mortality on screen  {onscreen['is_dead_v2'].mean():.1%}")
    print(f"billing_order known  {onscreen['billing_order'].notna().mean():.1%}")

    out_path = CLEAN_DIR / "characters_model.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\nWrote {out_path}")

    by_franchise = (
        onscreen.groupby("franchise")["is_dead_v2"]
        .agg(["size", "mean"])
        .sort_values("mean", ascending=False)
    )
    by_franchise.columns = ["n_onscreen", "mortality"]
    by_franchise["mortality"] = (100 * by_franchise["mortality"]).round(1)
    print("\nOn-screen mortality by franchise")
    print(by_franchise.to_string())


if __name__ == "__main__":
    main()
