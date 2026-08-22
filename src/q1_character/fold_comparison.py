"""Compare Q1 feature sets with the character-only baseline.

Computes PR-AUC separately for each fold and reports the mean difference,
spread, and the number of folds won. The five-fold comparisons are descriptive.

  python -m src.q1_character.fold_comparison
"""

import numpy as np
import pandas as pd
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
            model = Q1Models.models()[MODEL]
            if hasattr(model, "n_jobs"):
                model.set_params(n_jobs=1)
            pipeline = Q1Models.make_pipeline(
                model, spec.categorical, spec.numeric, spec.text
            )
            pipeline.fit(X.iloc[train_idx], y[train_idx])
            probability = pipeline.predict_proba(X.iloc[test_idx])[:, 1]
            scores.append(average_precision_score(y[test_idx], probability))
        return np.array(scores)

    @staticmethod
    def compare(reference: np.ndarray, candidate: np.ndarray) -> dict:
        difference = candidate - reference
        wins = int((difference > 0).sum())

        spread = difference.std(ddof=1)
        return {
            "mean_difference": float(difference.mean()),
            "sd_difference": float(spread),
            "folds_won": wins,
            "n_folds": len(difference),
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
             "sd_difference", "folds_won"]
    print(f"Descriptive per-fold comparison against '{REFERENCE}'")
    print(comparisons.sort_values(["regime", "mean_difference"], ascending=[True, False])
          [shown].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    folds.to_csv(CLEAN_DIR / "q1_fold_scores.csv", index=False, encoding="utf-8")
    comparisons.to_csv(
        CLEAN_DIR / "q1_fold_comparison.csv", index=False, encoding="utf-8"
    )
    print("\nWrote q1_fold_scores.csv, q1_fold_comparison.csv")


if __name__ == "__main__":
    from src import setup_run_log
    setup_run_log(__spec__)
    main()
