"""Cost-sensitive thresholds, and the wider metric table behind the headline.

Two things the evaluation lecture asks for that a PR-AUC on its own does not
supply.

**A decision rule.** PR-AUC scores a ranking over every threshold at once.
Using the model means picking one threshold, and 0.5 is the right pick only when
a false positive and a false negative cost the same. The lecture puts the
question directly, asking whether a good email in the spam folder is as bad as
spam in the inbox, and gives the answer as a weighted loss,

    L = w1 * FN + w2 * FP,   with w1 >> w2 when positives are rarer.

Deaths are the rarer class here at 35.6%, so w1 > w2 is the natural default. The
threshold that minimises L is swept rather than assumed, and reported across a
range of cost ratios so a reader who disagrees with our ratio can read off
theirs.

**A second look at more measures.** The lecture recommends leading with a few
metrics chosen for the problem and then showing a broader set, so that a method
which looks good on one measure and bad on the rest is visible as such. The
headline metric stays PR-AUC, for the class-imbalance reason given in the
writeup; accuracy, precision, recall, F1, the two error rates and the Brier
score follow underneath.

Everything here is computed on out-of-fold predictions over the trainval split,
so no threshold is chosen on the test set.

  python -m src.q1_character.costs
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict

from src.paths import CLEAN_DIR
from src.q1_character.models import (
    N_SPLITS, RANDOM_STATE, Q1Models, TARGET, build_feature_sets,
)

# Cost of missing a death, relative to a false alarm. 1.0 is the symmetric case
# that the default 0.5 threshold implicitly assumes.
COST_RATIOS = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]

THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.01), 2)

# The two franchises the error analysis found failing in opposite directions.
WATCHED = ["Spartacus", "Grey's Anatomy"]


class CostAnalysis:

    @staticmethod
    def out_of_fold(df: pd.DataFrame, spec, regime: str) -> np.ndarray:
        splitter, groups = (
            (StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                             random_state=RANDOM_STATE), None)
            if regime == "random"
            else (GroupKFold(n_splits=N_SPLITS), df["franchise"].to_numpy())
        )
        return cross_val_predict(
            Q1Models.make_pipeline(
                Q1Models.models()["forest"], spec.categorical, spec.numeric, spec.text
            ),
            df[spec.columns], df[TARGET].astype(int),
            cv=splitter, groups=groups, method="predict_proba",
        )[:, 1]

    @staticmethod
    def metrics(y: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
        pred = (proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

        precision = tp / (tp + fp) if tp + fp else float("nan")
        recall = tp / (tp + fn) if tp + fn else float("nan")
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else float("nan")
        )
        return {
            "threshold": threshold,
            "accuracy": (tp + tn) / len(y),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp_rate": recall,
            "fp_rate": fp / (fp + tn) if fp + tn else float("nan"),
            "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        }

    @staticmethod
    def cost_curve(y: np.ndarray, proba: np.ndarray, ratio: float) -> pd.DataFrame:
        """Weighted loss at every threshold, normalised per character.

        Normalising by n makes the loss readable as an expected cost per
        character and comparable between franchises of very different sizes,
        which the raw count is not.
        """
        rows = []
        for threshold in THRESHOLDS:
            pred = (proba >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
            rows.append({
                "cost_ratio": ratio,
                "threshold": threshold,
                "loss": (ratio * fn + fp) / len(y),
                "FN": int(fn), "FP": int(fp),
            })
        return pd.DataFrame(rows)

    @staticmethod
    def sweep(y: np.ndarray, proba: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
        curves = pd.concat(
            [CostAnalysis.cost_curve(y, proba, r) for r in COST_RATIOS],
            ignore_index=True,
        )
        best = (
            curves.loc[curves.groupby("cost_ratio")["loss"].idxmin()]
            .reset_index(drop=True)
        )
        return curves, best

    @staticmethod
    def per_franchise(df: pd.DataFrame, y: np.ndarray, proba: np.ndarray,
                      ratio: float) -> pd.DataFrame:
        """The cost-minimising threshold each franchise would choose alone.

        A single global threshold is what the model actually ships with. This
        asks what each franchise would have picked for itself, which measures
        how far apart their optima are and therefore how much a global
        threshold costs.
        """
        rows = []
        for franchise, index in df.groupby("franchise").groups.items():
            positions = df.index.get_indexer(index)
            y_f, p_f = y[positions], proba[positions]
            if len(y_f) < 30 or len(set(y_f)) < 2:
                continue
            curve = CostAnalysis.cost_curve(y_f, p_f, ratio)
            best = curve.loc[curve["loss"].idxmin()]
            rows.append({
                "franchise": franchise,
                "n": len(y_f),
                "mortality": float(y_f.mean()),
                "best_threshold": float(best["threshold"]),
                "loss_at_best": float(best["loss"]),
                "loss_at_global": float(
                    curve.loc[curve["threshold"] == GLOBAL_REFERENCE, "loss"].iloc[0]
                ),
            })
        out = pd.DataFrame(rows)
        out["cost_of_global"] = out["loss_at_global"] - out["loss_at_best"]
        return out.sort_values("best_threshold")


GLOBAL_REFERENCE = 0.5


def main() -> None:
    # Restricted to trainval. The docstring has always said thresholds are
    # chosen without touching the test split; until now main() loaded the whole
    # population, so that claim was false and every threshold here had seen the
    # test rows.
    df = Q1Models.load()
    splits = pd.read_csv(CLEAN_DIR / "split_assignment.csv")
    df = df.merge(splits[["page_url", "split_grouped"]], on="page_url", how="inner")
    df = df[df["split_grouped"] == "trainval"].reset_index(drop=True)
    feature_sets = build_feature_sets(df)
    spec = feature_sets["character+text"]

    print(f"{len(df):,} trainval characters, {df[TARGET].mean():.1%} dead")
    print("Out-of-fold predictions, forest on character+text, grouped folds.\n")

    y = df[TARGET].astype(int).to_numpy()
    proba = CostAnalysis.out_of_fold(df, spec, "grouped")

    print("Threshold-free summary")
    print(f"  PR-AUC   {average_precision_score(y, proba):.3f}   "
          f"(chance {y.mean():.3f})")
    print(f"  ROC-AUC  {roc_auc_score(y, proba):.3f}")
    print(f"  Brier    {brier_score_loss(y, proba):.3f}\n")

    table = pd.DataFrame([
        CostAnalysis.metrics(y, proba, t) for t in (0.3, 0.4, 0.5, 0.6, 0.7)
    ])
    print("The wider metric set, at a range of thresholds")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    curves, best = CostAnalysis.sweep(y, proba)
    print("\nCost-minimising threshold by cost ratio")
    print("  w1 is the cost of missing a death; a false alarm costs 1.")
    print(best[["cost_ratio", "threshold", "loss", "FN", "FP"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    ratio = 3.0
    franchises = CostAnalysis.per_franchise(df, y, proba, ratio)
    print(f"\nPer-franchise optimum at w1 = {ratio:g}, against the global "
          f"{GLOBAL_REFERENCE} threshold")
    extremes = pd.concat([franchises.head(5), franchises.tail(5)])
    print(extremes.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    watched = franchises[franchises["franchise"].isin(WATCHED)]
    if len(watched):
        print("\nThe two franchises the error analysis singled out")
        print(watched.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        spread = watched["best_threshold"].max() - watched["best_threshold"].min()
        print(f"  their preferred thresholds differ by {spread:.2f}")

    curves.to_csv(CLEAN_DIR / "q1_cost_curves.csv", index=False, encoding="utf-8")
    best.to_csv(CLEAN_DIR / "q1_cost_optima.csv", index=False, encoding="utf-8")
    table.to_csv(CLEAN_DIR / "q1_metric_table.csv", index=False, encoding="utf-8")
    franchises.to_csv(
        CLEAN_DIR / "q1_cost_by_franchise.csv", index=False, encoding="utf-8"
    )
    print("\nWrote q1_cost_curves.csv, q1_cost_optima.csv, q1_metric_table.csv, "
          "q1_cost_by_franchise.csv")


if __name__ == "__main__":
    from src import setup_run_log
    setup_run_log(__spec__)
    main()
