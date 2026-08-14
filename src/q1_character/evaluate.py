"""Pick a model with k-fold CV inside trainval, then score it once on test.

`models.py` compares every feature set against every model under cross
validation and reports the best cell per feature set. That is the right tool
for "which family of features carries the signal" and the wrong tool for "how
well does this actually do": the winner of 21 comparisons is chosen on the same
numbers it is reported with, so a single reported PR-AUC from that table is
optimistic.

This fixes that without giving up the averaging that makes cross-validated
comparison reliable in the first place. A single fixed dev split was tried and
dropped -- with only 38 franchises, one held-out slice landed at 50.7%
mortality against 32.6% for train, purely from which franchises it happened to
contain (see splits.py). k-fold selection *inside* trainval keeps the averaging;
test is still a fixed slice, touched exactly once, because scoring it more than
once is exactly the problem this file exists to avoid.

  python -m src.q1_character.evaluate
"""

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict

from src.paths import CLEAN_DIR
from src.q1_character.models import N_SPLITS, RANDOM_STATE, Q1Models, TARGET, build_feature_sets

N_DEV_ERRORS = 40


class HoldoutEvaluation:

    @staticmethod
    def load() -> tuple[pd.DataFrame, dict]:
        df = Q1Models.load()
        splits = pd.read_csv(CLEAN_DIR / "split_assignment.csv")
        df = df.merge(
            splits[["page_url", "split_random", "split_grouped"]],
            on="page_url", how="inner",
        )
        return df, build_feature_sets(df)

    @staticmethod
    def cv_scores(trainval: pd.DataFrame, columns: list[str],
                  pipeline, regime: str) -> dict:
        """Mean cross-validated PR-AUC/ROC-AUC over trainval only."""
        X, y = trainval[columns], trainval[TARGET]
        if regime == "random":
            splitter = StratifiedKFold(
                n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
            )
            groups = None
        else:
            splitter = GroupKFold(n_splits=N_SPLITS)
            groups = trainval["franchise"]

        probability = cross_val_predict(
            pipeline, X, y, cv=splitter, groups=groups, method="predict_proba"
        )[:, 1]
        return {
            "pr_auc": average_precision_score(y, probability),
            "roc_auc": roc_auc_score(y, probability),
        }

    @staticmethod
    def run(df: pd.DataFrame, feature_sets: dict, regime: str) -> tuple[pd.DataFrame, dict]:
        column = f"split_{regime}"
        trainval = df[df[column] == "trainval"]
        test = df[df[column] == "test"]

        rows = []
        for set_name, spec in feature_sets.items():
            for model_name, model in Q1Models.models().items():
                pipeline = Q1Models.make_pipeline(
                    model, spec.categorical, spec.numeric, spec.text
                )
                scored = HoldoutEvaluation.cv_scores(
                    trainval, spec.columns, pipeline, regime
                )
                rows.append({
                    "regime": regime, "features": set_name, "model": model_name,
                    "cv_pr_auc": scored["pr_auc"], "cv_roc_auc": scored["roc_auc"],
                })
                print(f"  {regime:8s} {set_name:22s} {model_name:9s} "
                      f"trainval CV PR-AUC {scored['pr_auc']:.3f}")

        selection = pd.DataFrame(rows).sort_values("cv_pr_auc", ascending=False)
        best = selection.iloc[0]

        # Winner refit on all of trainval, then test exactly once.
        spec = feature_sets[best["features"]]
        final = Q1Models.make_pipeline(
            Q1Models.models()[best["model"]], spec.categorical, spec.numeric, spec.text
        )
        final.fit(trainval[spec.columns], trainval[TARGET])
        probability = final.predict_proba(test[spec.columns])[:, 1]
        test_scored = {
            "pr_auc": average_precision_score(test[TARGET], probability),
            "roc_auc": roc_auc_score(test[TARGET], probability),
        }

        summary = {
            "regime": regime,
            "chosen_features": best["features"],
            "chosen_model": best["model"],
            "trainval_cv_pr_auc": best["cv_pr_auc"],
            "test_pr_auc": test_scored["pr_auc"],
            "test_roc_auc": test_scored["roc_auc"],
            "n_trainval": len(trainval), "n_test": len(test),
            "test_base_rate": test[TARGET].mean(),
        }
        return selection, summary

    @staticmethod
    def dev_errors(df: pd.DataFrame, feature_sets: dict, summary: dict) -> pd.DataFrame:
        """Out-of-fold trainval predictions for the chosen model, worst first.

        Uses cross_val_predict over trainval so every row is scored by a model
        that never saw it, without touching test.
        """
        column = f"split_{summary['regime']}"
        trainval = df[df[column] == "trainval"].copy()

        spec = feature_sets[summary["chosen_features"]]
        columns = spec.columns
        pipeline = Q1Models.make_pipeline(
            Q1Models.models()[summary["chosen_model"]],
            spec.categorical, spec.numeric, spec.text,
        )
        if summary["regime"] == "random":
            splitter = StratifiedKFold(
                n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
            )
            groups = None
        else:
            splitter = GroupKFold(n_splits=N_SPLITS)
            groups = trainval["franchise"]

        trainval["predicted"] = cross_val_predict(
            pipeline, trainval[columns], trainval[TARGET],
            cv=splitter, groups=groups, method="predict_proba",
        )[:, 1]
        trainval["error"] = (trainval["predicted"] - trainval[TARGET]).abs()

        keep = [c for c in [
            "name", "franchise", TARGET, "predicted", "error", "gender",
            "archetype", "affiliation_alignment", "prominence_tier",
            "billing_order", "pagerank_rank", "page_text_length",
        ] if c in trainval.columns]
        return trainval.sort_values("error", ascending=False)[keep]


def main() -> None:
    df, feature_sets = HoldoutEvaluation.load()

    selections, summaries = [], []
    for regime in ("random", "grouped"):
        selection, summary = HoldoutEvaluation.run(df, feature_sets, regime)
        selections.append(selection)
        summaries.append(summary)
        print()

    pd.concat(selections).to_csv(
        CLEAN_DIR / "q1_dev_selection.csv", index=False, encoding="utf-8"
    )
    summary_table = pd.DataFrame(summaries)
    summary_table.to_csv(
        CLEAN_DIR / "q1_holdout_results.csv", index=False, encoding="utf-8"
    )

    print("Chosen by k-fold CV inside trainval, scored once on test")
    print(summary_table.round(3).to_string(index=False))

    # Error dump for the random regime, where every franchise is represented.
    errors = HoldoutEvaluation.dev_errors(df, feature_sets, summaries[0])
    errors.to_csv(CLEAN_DIR / "q1_dev_errors.csv", index=False, encoding="utf-8")
    print(f"\nWorst {N_DEV_ERRORS} out-of-fold trainval errors -> q1_dev_errors.csv")
    print(errors.head(N_DEV_ERRORS).to_string(index=False))


if __name__ == "__main__":
    from src import setup_run_log
    setup_run_log(__spec__)
    main()
