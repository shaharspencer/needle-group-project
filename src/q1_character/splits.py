"""Assign every character to trainval or test, once, and write it down.

Two regimes, because they answer different questions:
  random   rows split at random, stratified on the outcome. Franchises appear in
           both splits, so the model may use franchise base rates.
  grouped  whole franchises go to one split. Nothing about a test franchise is
           visible during training, to see if the model transfers to unseen stories.

  python -m src.q1_character.splits
"""

import pandas as pd

from src.constants import analysis_population
from src.paths import CLEAN_DIR

TARGET = "is_dead_v2"
SEED = 20260814

TEST_FRACTION = 0.2

SPLIT_NAMES = ("trainval", "test")


class Splits:

    @staticmethod
    def random_split(df: pd.DataFrame) -> pd.Series:
        """Stratified on the outcome so both splits share a base rate."""
        assignment = pd.Series("trainval", index=df.index, dtype=object)

        for _, group in df.groupby(TARGET):
            shuffled = group.sample(frac=1.0, random_state=SEED).index
            n_test = int(round(TEST_FRACTION * len(shuffled)))
            assignment.loc[shuffled[:n_test]] = "test"

        return assignment

    @staticmethod
    def grouped_split(df: pd.DataFrame) -> pd.Series:
        """Whole franchises to one split, greedily balancing size and mortality."""
        overall_rate = df[TARGET].mean()
        stats = (
            df.groupby("franchise")[TARGET]
            .agg(n="size", deaths="sum")
            .sort_values("n", ascending=False)
        )

        targets = {"trainval": 1 - TEST_FRACTION, "test": TEST_FRACTION}
        total_n = stats["n"].sum()
        running = {name: {"n": 0, "deaths": 0} for name in SPLIT_NAMES}

        # Largest-deficit-first bin packing: send each franchise to whichever
        # split is furthest below its row-count quota, not whichever split it
        # would individually fit best. Comparing a single franchise's own share
        # against trainval's 80% target -- what the first version of this
        # function did -- never favoured trainval, since no one franchise is
        # 80% of the corpus, so trainval never got anything. Ties on deficit
        # (there are many, early on, when every split is still empty) go to
        # whichever split the franchise's death rate pulls closest to the
        # corpus average.
        franchise_split: dict[str, str] = {}
        for franchise, row in stats.iterrows():
            deficits = {
                name: targets[name] * total_n - running[name]["n"]
                for name in SPLIT_NAMES
            }
            max_deficit = max(deficits.values())
            tied = [n for n, d in deficits.items() if d >= max_deficit - 1e-9]

            if len(tied) == 1:
                best_split = tied[0]
            else:
                def rate_gap(name):
                    new_n = running[name]["n"] + row["n"]
                    new_deaths = running[name]["deaths"] + row["deaths"]
                    return abs(new_deaths / new_n - overall_rate)
                best_split = min(tied, key=rate_gap)

            franchise_split[franchise] = best_split
            running[best_split]["n"] += row["n"]
            running[best_split]["deaths"] += row["deaths"]

        return df["franchise"].map(franchise_split)

    @staticmethod
    def build() -> pd.DataFrame:
        characters = pd.read_csv(
            CLEAN_DIR / "characters_model.csv", low_memory=False
        )
        population = analysis_population(characters)

        out = population[["page_url", "franchise", TARGET]].copy()
        out["split_random"] = Splits.random_split(population).values
        out["split_grouped"] = Splits.grouped_split(population).values
        return out

    @staticmethod
    def report(splits: pd.DataFrame) -> None:
        for regime in ("split_random", "split_grouped"):
            table = (
                splits.groupby(regime)
                .agg(n=(TARGET, "size"), mortality=(TARGET, "mean"),
                     franchises=("franchise", "nunique"))
                .reindex(list(SPLIT_NAMES))
            )
            table["share"] = (100 * table["n"] / table["n"].sum()).round(1)
            table["mortality"] = (100 * table["mortality"]).round(1)
            print(f"\n{regime}")
            print(table.to_string())


def main() -> None:
    splits = Splits.build()
    Splits.report(splits)

    out_path = CLEAN_DIR / "split_assignment.csv"
    splits.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\nWrote {out_path} ({len(splits):,} characters)")


if __name__ == "__main__":
    from src import setup_run_log
    setup_run_log(__spec__)
    main()
