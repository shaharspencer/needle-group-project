"""Q1: do character attributes or the franchise predict death better?

Three feature sets are trained and compared:
  franchise  the franchise and its metadata only
  character  who the character is, with no franchise information
  both

The gap between them answers the question. Random forest importances then say
which character attributes carry the signal.

Scored under two splitting regimes, because they answer different questions.
A random split lets the model learn each franchise's base rate, which is the
only regime in which "a GoT character versus a Harry Potter character" is a
fair comparison. A grouped split holds out whole franchises, so franchise
identity is useless by construction and what is measured is whether character
attributes transfer to a franchise never seen in training.

  python -m src.q1_character.models
"""

import re
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.constants import analysis_population
from src.paths import CLEAN_DIR

TARGET = "is_dead_v2"

CHARACTER_CATEGORICAL = ["gender", "archetype", "affiliation_alignment", "prominence_tier"]
CHARACTER_NUMERIC = [
    "is_human", "billing_order", "episode_count", "page_text_length",
    # Deliberately the cleaned count, not infobox_field_count: a dead character
    # acquires "Cause of death" and "Date of death" fields, so the raw count
    # partly counts the label. See clean_infobox_field_count in src/constants.py.
    "infobox_field_count_clean", "appearance_count", "has_dob", "has_family",
    "has_image", "has_alias",
]
FRANCHISE_CATEGORICAL = ["franchise", "genre", "medium", "content_rating"]
FRANCHISE_NUMERIC = ["franchise_release_year"]

# Columns that encode the outcome directly or are derived from it.
LEAKY = {
    "is_dead", "has_dod", "dod", "has_death_section", "has_causeofdeath",
    "info_completeness", "franchise_mortality_rate", "franchise_size",
    "franchise_female_ratio", "hg_winner",
}

# How much fans wrote about a character, rather than anything about the
# character. Dead characters attract death sections and cause-of-death fields,
# so these plausibly encode the outcome instead of predicting it. Scored as a
# separate feature set to see how much of the result rests on them.
WIKI_METADATA = ["page_text_length", "infobox_field_count_clean", "appearance_count"]

NETWORK_NUMERIC = [
    "pagerank", "pagerank_rank", "hub", "authority",
    "in_degree", "out_degree", "component_size", "n_relatives", "n_categories",
]

# Column names naming death would restate the label.
DEATH_TERM_RE = re.compile(
    r"death|deceas|died|dead|kill|murder|victim|casualt|fatal|corpse|posthum|"
    r"execut|assassinat|suicide|martyr|sacrific|slain|slay|perish|massacre|"
    r"memorial|remembr|undead|ghost|revenant|resurrect|alive|living|surviv|"
    r"fallen|fate|_status|_dod|_dob",
    re.IGNORECASE,
)

class FeatureSet(NamedTuple):
    categorical: list[str]
    numeric: list[str]
    text: str | None = None      # one column name, TF-IDF'd inside the pipeline

    @property
    def columns(self) -> list[str]:
        return self.categorical + self.numeric + ([self.text] if self.text else [])


# TF-IDF over the death-stripped opening paragraph. Capped because the models
# are compared against sets of ~15 numeric features; an uncapped vocabulary
# would swamp them and make the comparison about dimensionality.
TEXT_COLUMN = "intro_clean"
TEXT_MAX_FEATURES = 300
TEXT_MIN_DF = 20

FEATURE_SETS = {
    "franchise": FeatureSet(FRANCHISE_CATEGORICAL, FRANCHISE_NUMERIC),
    "character": FeatureSet(CHARACTER_CATEGORICAL, CHARACTER_NUMERIC),
    "character_no_wiki_meta": FeatureSet(
        CHARACTER_CATEGORICAL,
        [c for c in CHARACTER_NUMERIC if c not in WIKI_METADATA],
    ),
    "both": FeatureSet(
        CHARACTER_CATEGORICAL + FRANCHISE_CATEGORICAL,
        CHARACTER_NUMERIC + FRANCHISE_NUMERIC,
    ),
}


def build_feature_sets(df: pd.DataFrame) -> dict:
    """Add the wide sets, whose columns are only known once the data is read."""
    wide = [c for c in df.columns if c.startswith(("cat_", "ib_"))]
    leaky = [c for c in wide + NETWORK_NUMERIC if DEATH_TERM_RE.search(c)]
    if leaky:
        raise SystemExit(f"death-related feature columns reached the matrix: {leaky}")

    sets = dict(FEATURE_SETS)
    sets["network"] = FeatureSet(FRANCHISE_CATEGORICAL, NETWORK_NUMERIC)
    sets["character+network"] = FeatureSet(
        CHARACTER_CATEGORICAL, CHARACTER_NUMERIC + NETWORK_NUMERIC
    )
    if TEXT_COLUMN in df.columns:
        sets["text"] = FeatureSet([], [], TEXT_COLUMN)
        sets["character+text"] = FeatureSet(
            CHARACTER_CATEGORICAL, CHARACTER_NUMERIC, TEXT_COLUMN
        )
    sets["rich"] = FeatureSet(
        CHARACTER_CATEGORICAL + FRANCHISE_CATEGORICAL,
        CHARACTER_NUMERIC + FRANCHISE_NUMERIC + NETWORK_NUMERIC + wide,
        TEXT_COLUMN if TEXT_COLUMN in df.columns else None,
    )
    return sets

N_SPLITS = 5
RANDOM_STATE = 0


class Q1Models:

    @staticmethod
    def load() -> pd.DataFrame:
        df = pd.read_csv(CLEAN_DIR / "characters_model.csv", low_memory=False)
        df = analysis_population(df)

        features = pd.read_csv(CLEAN_DIR / "character_features.csv", low_memory=False)
        features = features.drop_duplicates(subset="page_url", keep="first")
        df = df.merge(
            features.drop(columns=["franchise", "intro"], errors="ignore"),
            on="page_url", how="left",
        )

        assert not (LEAKY & set(FEATURE_SETS["both"][0] + FEATURE_SETS["both"][1])), \
            "a leaky column reached the feature list"

        for col in CHARACTER_CATEGORICAL + FRANCHISE_CATEGORICAL:
            df[col] = df[col].fillna("Unknown").astype(str)
        # TfidfVectorizer raises on NaN, and a page whose whole opening
        # paragraph was death-stripped away legitimately has no text left.
        if TEXT_COLUMN in df.columns:
            df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str)
        return df

    @staticmethod
    def make_pipeline(model, categorical: list[str], numeric: list[str],
                      text: str | None = None) -> Pipeline:
        encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=10)
        # add_indicator: billing_order is missing for two thirds of characters,
        # and "not in the TMDB cast" is itself a prominence signal.
        numeric_steps = Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ])
        branches = [
            ("cat", encoder, categorical),
            ("num", numeric_steps, numeric),
        ]
        if text is not None:
            # Inside the ColumnTransformer on purpose. Vectorising the corpus
            # once up front would fit the vocabulary and the IDF weights on
            # test documents as well as training ones, which is preprocessing
            # across the split -- the thing every other step here avoids.
            branches.append((
                "text",
                TfidfVectorizer(
                    max_features=TEXT_MAX_FEATURES,
                    min_df=TEXT_MIN_DF,
                    stop_words="english",
                    strip_accents="unicode",
                    sublinear_tf=True,
                ),
                text,
            ))
        pre = ColumnTransformer(branches)
        return Pipeline([("pre", pre), ("model", model)])

    @staticmethod
    def models() -> dict:
        return {
            "logistic": LogisticRegression(max_iter=2000, class_weight="balanced"),
            "tree": DecisionTreeClassifier(
                max_depth=6, min_samples_leaf=50,
                class_weight="balanced", random_state=RANDOM_STATE,
            ),
            "forest": RandomForestClassifier(
                n_estimators=400, min_samples_leaf=5, n_jobs=-1,
                class_weight="balanced_subsample", random_state=RANDOM_STATE,
            ),
        }

    @staticmethod
    def evaluate(df: pd.DataFrame, feature_sets: dict) -> pd.DataFrame:
        """
        Score every feature set under both splitting regimes.

        random   a held-out character from a franchise the model has seen.
                 Franchise identity is usable, so this is the regime in which
                 "GoT versus Harry Potter" is a fair question.
        grouped  a character from a franchise held out entirely. Franchise
                 identity cannot help by construction, so this measures whether
                 character attributes transfer to an unseen franchise.
        """
        y = df[TARGET].astype(int).to_numpy()
        groups = df["franchise"].to_numpy()

        regimes = {
            "random": (StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                                       random_state=RANDOM_STATE), None),
            "grouped": (GroupKFold(n_splits=N_SPLITS), groups),
        }

        rows = []
        for regime, (splitter, split_groups) in regimes.items():
            for name, strategy in [("majority", "most_frequent"), ("prior", "prior")]:
                dummy = DummyClassifier(strategy=strategy)
                proba = cross_val_predict(
                    dummy, df, y, cv=splitter, groups=split_groups,
                    method="predict_proba",
                )[:, 1]
                rows.append(Q1Models._score(regime, f"baseline:{name}", "-", y, proba))

            for feature_set, spec in feature_sets.items():
                X = df[spec.columns]
                for model_name, model in Q1Models.models().items():
                    pipeline = Q1Models.make_pipeline(
                        model, spec.categorical, spec.numeric, spec.text
                    )
                    proba = cross_val_predict(
                        pipeline, X, y, cv=splitter, groups=split_groups,
                        method="predict_proba", n_jobs=1,
                    )[:, 1]
                    rows.append(
                        Q1Models._score(regime, feature_set, model_name, y, proba)
                    )
                    print(f"  {regime:8s} {feature_set:10s} {model_name:9s} done")

        return pd.DataFrame(rows)

    @staticmethod
    def _score(regime: str, feature_set: str, model: str,
               y: np.ndarray, proba: np.ndarray) -> dict:
        pred = (proba >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        return {
            "regime": regime,
            "features": feature_set,
            "model": model,
            "pr_auc": average_precision_score(y, proba),
            "roc_auc": roc_auc_score(y, proba) if len(set(y)) > 1 else float("nan"),
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "recall": tp / (tp + fn) if tp + fn else float("nan"),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        }

    @staticmethod
    def importances(df: pd.DataFrame) -> pd.DataFrame:
        """Permutation importance on the character-only forest."""
        spec = FEATURE_SETS["character"]
        X, y = df[spec.columns], df[TARGET].astype(int)

        pipeline = Q1Models.make_pipeline(
            Q1Models.models()["forest"], spec.categorical, spec.numeric, spec.text
        )
        pipeline.fit(X, y)

        result = permutation_importance(
            pipeline, X, y, n_repeats=5,
            random_state=RANDOM_STATE, scoring="average_precision", n_jobs=1,
        )
        return (
            pd.DataFrame({
                "feature": spec.columns,
                "importance": result.importances_mean,
                "std": result.importances_std,
            })
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )


def main() -> None:
    df = Q1Models.load()
    print(f"{len(df):,} on-screen characters, {df[TARGET].mean():.1%} dead\n")

    print("Cross-validating (GroupKFold on franchise):")
    feature_sets = build_feature_sets(df)
    report = Q1Models.evaluate(df, feature_sets)

    print("\nResults, ranked by PR-AUC")
    print(report.sort_values("pr_auc", ascending=False)
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    report.to_csv(CLEAN_DIR / "q1_model_comparison.csv", index=False)

    print("\nPermutation importance, character-only forest")
    importance = Q1Models.importances(df)
    print(importance.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    importance.to_csv(CLEAN_DIR / "q1_feature_importance.csv", index=False)


if __name__ == "__main__":
    from src import setup_run_log
    setup_run_log(__spec__)
    main()
