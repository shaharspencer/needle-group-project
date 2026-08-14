"""Export the demo model and character table as JSON.

The demo runs entirely in the browser, so the model has to be small enough to
evaluate in JavaScript. That is the first cut: no random forest, no 369 one-hot
category features, no TF-IDF vocabulary.

The second cut is the one that matters, and it is why the demo exposes nine
attributes rather than the hundred-odd the full model sees. An attribute earns
a place here only if a user can meaningfully choose it for an invented
character. That excludes:

  wiki-attention features   page_text_length and infobox_field_count are our
                            two strongest predictors, and both measure how much
                            fans wrote, not who the character is. prominence_tier
                            is out for the same reason -- it is page length in
                            four buckets, so a user picking "Lead" would really
                            be picking "long wiki page", which is the circularity
                            the writeup spends its Impediments section on.
  sparse attributes         the power_* flags, age_group and hero_alignment are
                            all under 2% populated, so a control for them would
                            be inert for almost every character.
  franchise restatements    medium, genre and content_rating are properties of
                            the franchise, which is already a control.

What survives is a logistic regression over attributes that are either intrinsic
to the character (gender, role, alignment, species, named relatives) or describe
their position in the story rather than the wiki (billing order, PageRank rank,
in-degree). It scores below the full model, and the demo says so.

  python -m src.app.export_demo
"""

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from src.constants import analysis_population
from src.paths import CLEAN_DIR, DOCS_DIR

TARGET = "is_dead_v2"

CATEGORICAL = ["gender", "archetype", "affiliation_alignment", "franchise"]
NUMERIC = [
    "billing_order", "pagerank_rank", "in_degree", "is_human", "n_relatives",
]

MAX_CHARACTERS = 4000


class DemoExport:

    @staticmethod
    def load() -> pd.DataFrame:
        df = pd.read_csv(CLEAN_DIR / "characters_model.csv", low_memory=False)
        df = analysis_population(df)

        features = pd.read_csv(CLEAN_DIR / "character_features.csv", low_memory=False)
        features = features.drop_duplicates(subset="page_url")
        df = df.merge(
            features[["page_url", "pagerank_rank", "n_relatives", "in_degree"]],
            on="page_url", how="left",
        )

        # Same random split src.q1_character.evaluate uses, so "the model got
        # this one right/wrong" in the demo means what it says: this row was
        # never in the data the model shown here was fit on.
        splits = pd.read_csv(CLEAN_DIR / "split_assignment.csv")
        df = df.merge(splits[["page_url", "split_random"]], on="page_url", how="inner")

        for col in CATEGORICAL:
            df[col] = df[col].fillna("Unknown").astype(str)
        # Characters absent from the TMDB cast get the bottom of the billing
        # order rather than a median, which is what "uncredited" means.
        df["billing_order"] = df["billing_order"].fillna(40).clip(0, 40)
        df["pagerank_rank"] = df["pagerank_rank"].fillna(0.5)
        # In-degree is heavily right-skewed (median 4, max 5,444), so it is
        # clipped at roughly its 95th percentile. Without that, one Star Wars
        # hub character sets the scale and every ordinary character is squashed
        # into the bottom of the slider.
        df["in_degree"] = df["in_degree"].fillna(0).clip(0, 60)
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
        is_trainval = (df["split_random"] == "trainval").to_numpy()

        # Standardisation stats and the model itself are both fit on trainval
        # only. Fitting the scaler on the full population -- test included --
        # would be a smaller version of the same leak the rest of this
        # project spent a lot of effort removing: test rows would have
        # shaped the very features used to score them.
        means = X[is_trainval][:, -len(NUMERIC):].mean(axis=0)
        scales = X[is_trainval][:, -len(NUMERIC):].std(axis=0)
        scales[scales == 0] = 1.0
        X[:, -len(NUMERIC):] = (X[:, -len(NUMERIC):] - means) / scales

        model = LogisticRegression(max_iter=3000, C=1.0)
        model.fit(X[is_trainval], y[is_trainval])

        names = []
        for col in CATEGORICAL:
            names += [f"{col}={level}" for level in levels[col]]
        names += NUMERIC

        df["risk"] = model.predict_proba(X)[:, 1]

        # The demo only ever shows test characters: every risk score and
        # "the model called this right/wrong" verdict a viewer sees is for a
        # row the model above never trained on.
        df = df[~is_trainval].copy()
        test_proba = df["risk"].to_numpy()
        test_y = df[TARGET].astype(int).to_numpy()

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
            "i": int(row["in_degree"]),
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
                # Scored on test only, never seen during fitting -- not a
                # cross-validated average over rows the model trained on.
                "pr_auc": round(float(average_precision_score(test_y, test_proba)), 3),
                "roc_auc": round(float(roc_auc_score(test_y, test_proba)), 3),
                "base_rate": round(float(test_y.mean()), 3),
                "n_train": int(is_trainval.sum()),
                "n_test": int(len(test_y)),
            },
            "characters": characters,
        }


def main() -> None:
    payload = DemoExport.build()
    blob = json.dumps(payload, separators=(",", ":"))

    # Written to both places on purpose. data/clean is where every other build
    # output lands; docs/ is what GitHub Pages serves. Keeping the copy here
    # rather than in a shell step means the page can never quietly serve an
    # older model than the one this script just fit.
    out_paths = [CLEAN_DIR / "demo_data.json", DOCS_DIR / "demo_data.json"]
    for path in out_paths:
        path.write_text(blob, encoding="utf-8")

    print(f"model: {len(payload['model']['coefficients'])} coefficients")
    print(f"performance: PR-AUC {payload['performance']['pr_auc']}, "
          f"ROC-AUC {payload['performance']['roc_auc']}, "
          f"base rate {payload['performance']['base_rate']}")
    print(f"characters: {len(payload['characters'])}")
    for path in out_paths:
        print(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    from src import setup_run_log
    setup_run_log(__spec__)
    main()
