"""Assign every character to trainval or test, once, and write it down.

Until this existed the project reported cross-validated scores for 21
configurations (seven feature sets by three models) and then quoted the best one
per feature set. Those numbers are chosen on the same folds they are measured on,
so they are optimistic: with 21 attempts, the maximum of a noisy estimate is
above its mean.

The fix is nested, not a third fixed split: `evaluate.py` runs k-fold CV *inside*
trainval to pick a winner -- reusing the same cross-validation regime `models.py`
already uses, so selection isn't at the mercy of one arbitrarily-drawn dev set --
then refits the winner on all of trainval and scores it on test exactly once.
A fixed train/dev/test split was tried first and dropped: with only 38
franchises, one held-out dev slice landed at 50.7% mortality against 32.6% for
train, purely from which handful of franchises it happened to contain. Folding
dev into a k-fold selection loop removes that single-draw noise; test still
needs to be a fixed, disjoint slice, since scoring it more than once is exactly
the problem this file exists to avoid.

Two regimes, because they answer different questions and neither is a substitute
for the other:

  random   rows split at random, stratified on the outcome. Franchises appear in
           both splits, so the model may use franchise base rates.
  grouped  whole franchises go to one split. Nothing about a test franchise is
           visible during training, which is the harder and more honest question:
           does this transfer to a story we have never seen?

Assigning franchises to the grouped split is not a coin flip, because franchise
sizes span three orders of magnitude -- Star Wars has 2,785 on-screen named
characters and The Matrix 4. Franchises are dealt by largest-remaining-deficit
(see `grouped_split`) so trainval and test land close to their target share and
close to the corpus death rate, without needing every stratum to be enormous.

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
        """Whole franchises to one split, greedily balancing size and mortality.

        Round-robin-by-size alone can deal an unlucky hand: with only 38
        franchises to distribute, whichever 7-8 land in dev by chance might
        happen to be unusually deadly ones. Assigning largest-first to
        whichever split most needs both more rows and a mortality rate closer
        to the corpus average keeps train/dev/test comparable on both axes,
        not just on row count.
        """
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
