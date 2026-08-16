"""Are the gaps between Q1 feature sets larger than sampling noise?

src/q1_character/models.py ranks feature sets by a single pooled PR-AUC per
configuration. The evaluation lecture's opening example is a table of eight
classifiers ranked eight different ways by eight different metrics, and its
point is that a ranking of point estimates does not by itself support a claim
that one method beats another. A gap of 0.007 between two configurations may be
a real difference or may be the fold assignment.

Two things are computed here.

**Per-fold scores.** cross_val_predict pools out-of-fold predictions and scores
them once, which gives one number and no spread. Scoring each fold separately
gives five, from which a paired comparison can be made. Pairing matters: fold 3
is harder than fold 1 for every model, so comparing per-fold differences removes
the fold effect that would otherwise dominate an unpaired test.

**A paired test on the differences.** With five folds the sample is far too
small for a t-test's normality assumption, so the Wilcoxon signed-rank test is
used, which asks only that the differences be symmetric about their median.
Even so, five paired observations cannot produce a p-value below 0.0625, so the
test is reported alongside the observed per-fold win count rather than in place
of it. Under the grouped rule the folds are whole franchises, and the spread
across them is itself the quantity of interest: a feature set that wins on four
folds and loses badly on the fifth has not transferred.

The comparison is run against the character-only baseline, since the question
Q1 asks is what each family adds to character attributes.

  python -m src.q1_character.significance
"""

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

from src.paths import CLEAN_DIR
from src.q1_character.models import (
    N_SPLITS, RANDOM_STATE, Q1Models, TARGET, build_feature_sets,
)

REFERENCE = "character"

# Which model to compare feature sets with. Held fixed so the comparison is
# about the features rather than about which model happened to suit them; the
# model-by-regime interaction is reported separately in models.py.
MODEL = "forest"


class FoldScores:

    @staticmethod
    def splitter(regime: str, df: pd.DataFrame):
        if regime == "random":
            return StratifiedKFold(
                n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
            ), None
        return GroupKFold(n_splits=N_SPLITS), df["franchise"].to_numpy()

    @staticmethod
    def per_fold(df: pd.DataFrame, spec, regime: str) -> np.ndarray:
        """PR-AUC within each held-out fold.

        Scored per fold rather than on pooled out-of-fold predictions. Pooling
        would mix folds with different base rates into one PR curve, and PR-AUC
        is not invariant to the base rate, so the pooled number is not the mean
        of the fold numbers.
        """
        splitter, groups = FoldScores.splitter(regime, df)
        y = df[TARGET].astype(int).to_numpy()
        X = df[spec.columns]

        scores = []
        for train_idx, test_idx in splitter.split(X, y, groups=groups):
            pipeline = Q1Models.make_pipeline(
                Q1Models.models()[MODEL], spec.categorical, spec.numeric, spec.text
            )
            pipeline.fit(X.iloc[train_idx], y[train_idx])
            probability = pipeline.predict_proba(X.iloc[test_idx])[:, 1]
            scores.append(average_precision_score(y[test_idx], probability))
        return np.array(scores)

    @staticmethod
    def compare(reference: np.ndarray, candidate: np.ndarray) -> dict:
        difference = candidate - reference
        wins = int((difference > 0).sum())

        if np.allclose(difference, 0):
            p_value = 1.0
        else:
            p_value = float(wilcoxon(candidate, reference).pvalue)

        # Standard error of the paired mean difference across folds. Folds share
        # training data, so this understates the true variance; it is reported
        # as a spread rather than used to build a confidence interval.
        spread = difference.std(ddof=1) / np.sqrt(len(difference))
        return {
            "mean_difference": float(difference.mean()),
            "se_difference": float(spread),
            "folds_won": wins,
            "n_folds": len(difference),
            "p_wilcoxon": p_value,
        }


def main() -> None:
    df = Q1Models.load()
    feature_sets = build_feature_sets(df)
    print(f"{len(df):,} characters, {df[TARGET].mean():.1%} dead")
    print(f"Comparing feature sets against '{REFERENCE}' using the {MODEL}, "
          f"{N_SPLITS} folds per regime.\n")

    fold_rows, comparison_rows = [], []
    for regime in ("random", "grouped"):
        scores = {}
        for name, spec in feature_sets.items():
            scores[name] = FoldScores.per_fold(df, spec, regime)
            for fold, value in enumerate(scores[name]):
                fold_rows.append({
                    "regime": regime, "features": name,
                    "fold": fold, "pr_auc": value,
                })
            print(f"  {regime:8s} {name:22s} "
                  f"mean {scores[name].mean():.3f}  "
                  f"sd {scores[name].std(ddof=1):.3f}")

        reference = scores[REFERENCE]
        for name, values in scores.items():
            if name == REFERENCE:
                continue
            comparison_rows.append({
                "regime": regime,
                "features": name,
                "reference": REFERENCE,
                "mean_pr_auc": float(values.mean()),
                "reference_pr_auc": float(reference.mean()),
                **FoldScores.compare(reference, values),
            })
        print()

    folds = pd.DataFrame(fold_rows)
    comparisons = pd.DataFrame(comparison_rows)

    shown = ["regime", "features", "mean_pr_auc", "mean_difference",
             "se_difference", "folds_won", "p_wilcoxon"]
    print(f"Paired per-fold comparison against '{REFERENCE}'")
    print(comparisons.sort_values(["regime", "mean_difference"], ascending=[True, False])
          [shown].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print(f"\nWith {N_SPLITS} paired folds the smallest attainable Wilcoxon "
          f"p-value is {2 / 2 ** N_SPLITS:.4f}, so no single comparison here can")
    print("reach p < 0.05. Read the fold win count and the spread alongside it.")

    folds.to_csv(CLEAN_DIR / "q1_fold_scores.csv", index=False, encoding="utf-8")
    comparisons.to_csv(
        CLEAN_DIR / "q1_significance.csv", index=False, encoding="utf-8"
    )
    print("\nWrote q1_fold_scores.csv, q1_significance.csv")


if __name__ == "__main__":
    from src import setup_run_log
    setup_run_log(__spec__)
    main()
