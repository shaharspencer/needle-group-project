"""Export the demo model and character table as JSON.

The demo runs entirely in the browser, so the model has to be small enough to
evaluate in JavaScript. That rules out the random forest and the 369 category
features, and it rules out the wiki-metadata features too: page length and
infobox field count are not things a person can choose for an invented
character, and they are the features we least want to lean on.

What is left is a logistic regression over attributes a user can actually pick.
It scores lower than the full model, and the demo says so.

  python -m src.app.export_demo
"""

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.paths import CLEAN_DIR

TARGET = "is_dead_v2"

CATEGORICAL = ["gender", "archetype", "affiliation_alignment", "franchise"]
NUMERIC = ["billing_order", "pagerank_rank", "is_human", "n_relatives"]

MAX_CHARACTERS = 4000


class DemoExport:

    @staticmethod
    def load() -> pd.DataFrame:
        df = pd.read_csv(CLEAN_DIR / "characters_model.csv", low_memory=False)
        df = df[df["is_onscreen"] == 1].copy()

        features = pd.read_csv(CLEAN_DIR / "character_features.csv", low_memory=False)
        features = features.drop_duplicates(subset="page_url")
        df = df.merge(
            features[["page_url", "pagerank_rank", "n_relatives"]],
            on="page_url", how="left",
        )

        for col in CATEGORICAL:
            df[col] = df[col].fillna("Unknown").astype(str)
        # Characters absent from the TMDB cast get the bottom of the billing
        # order rather than a median, which is what "uncredited" means.
        df["billing_order"] = df["billing_order"].fillna(40).clip(0, 40)
        df["pagerank_rank"] = df["pagerank_rank"].fillna(0.5)
        df["is_human"] = df["is_human"].fillna(0)
        df["n_relatives"] = df["n_relatives"].fillna(0).clip(0, 20)
        return df

    @staticmethod
    def design(df: pd.DataFrame, levels: dict) -> np.ndarray:
        """One-hot the categoricals by hand so JavaScript can repeat it exactly."""
        columns = []
        for col in CATEGORICAL:
            for level in levels[col]:
                columns.append((df[col] == level).astype(float).to_numpy())
        for col in NUMERIC:
            columns.append(df[col].astype(float).to_numpy())
        return np.column_stack(columns)

    @staticmethod
    def build() -> dict:
        df = DemoExport.load()
        levels = {
            col: sorted(df[col].value_counts().loc[lambda s: s >= 20].index)
            for col in CATEGORICAL
        }

        X = DemoExport.design(df, levels)
        y = df[TARGET].astype(int).to_numpy()

        means = X[:, -len(NUMERIC):].mean(axis=0)
        scales = X[:, -len(NUMERIC):].std(axis=0)
        scales[scales == 0] = 1.0
        X[:, -len(NUMERIC):] = (X[:, -len(NUMERIC):] - means) / scales

        model = LogisticRegression(max_iter=3000, C=1.0)
        model.fit(X, y)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        proba = cross_val_predict(
            LogisticRegression(max_iter=3000, C=1.0), X, y,
            cv=cv, method="predict_proba",
        )[:, 1]

        names = []
        for col in CATEGORICAL:
            names += [f"{col}={level}" for level in levels[col]]
        names += NUMERIC

        df["risk"] = model.predict_proba(X)[:, 1]

        # A manageable sample for the browser, keeping the most central
        # characters in each franchise so the browse list is recognisable.
        shown = (
            df.sort_values("pagerank_rank")
            .groupby("franchise", group_keys=False)
            .head(max(MAX_CHARACTERS // df["franchise"].nunique(), 40))
        )

        characters = [{
            "n": row["name"],
            "f": row["franchise"],
            "g": row["gender"],
            "a": row["archetype"],
            "al": row["affiliation_alignment"],
            "b": None if pd.isna(row["billing_order"]) else int(row["billing_order"]),
            "p": round(float(row["pagerank_rank"]), 3),
            "h": int(row["is_human"]),
            "r": int(row["n_relatives"]),
            "d": int(row[TARGET]),
            "risk": round(float(row["risk"]), 4),
            "u": row["page_url"],
        } for _, row in shown.iterrows()]

        return {
            "model": {
                "intercept": float(model.intercept_[0]),
                "coefficients": dict(zip(names, model.coef_[0].round(5).tolist())),
                "numeric": NUMERIC,
                "means": means.round(5).tolist(),
                "scales": scales.round(5).tolist(),
                "levels": levels,
            },
            "performance": {
                "pr_auc": round(float(average_precision_score(y, proba)), 3),
                "roc_auc": round(float(roc_auc_score(y, proba)), 3),
                "base_rate": round(float(y.mean()), 3),
                "n_train": int(len(y)),
            },
            "characters": characters,
        }


def main() -> None:
    payload = DemoExport.build()
    out_path = CLEAN_DIR / "demo_data.json"
    out_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")

    size_mb = out_path.stat().st_size / 1e6
    print(f"model: {len(payload['model']['coefficients'])} coefficients")
    print(f"performance: PR-AUC {payload['performance']['pr_auc']}, "
          f"ROC-AUC {payload['performance']['roc_auc']}, "
          f"base rate {payload['performance']['base_rate']}")
    print(f"characters: {len(payload['characters'])}")
    print(f"wrote {out_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
