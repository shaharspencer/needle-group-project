"""Extract and display the strongest TF-IDF coefficients predicting death/survival.

Fits a single logistic regression on trainval using the death-stripped `intro_clean` 
text. The output is read for its ranking to see which words push toward death or 
away from it, not quoted as a performance estimate.

  python -m src.q2_text.coefficients
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.constants import analysis_population
from src.paths import CLEAN_DIR
from src.q1_character.features import DEATH_TERMS
from src.q1_character.models import (
    TEXT_COLUMN, TEXT_MAX_FEATURES, TEXT_MIN_DF,
)

TARGET = "is_dead_v2"
TOP_N = 20

# The check runs the same pattern that did the stripping. It used to run a
# separate hand-written list, which is how the two drifted apart: the stripper
# was missing `die`, `dies` and `dying` while this list contained them, so the
# check passed on a vocabulary the stripper had never actually protected. Any
# term this flags is a term `strip_death_text` should have removed, by
# construction, so the assertion now tests the thing it claims to test.
LEAK_RE = DEATH_TERMS


def leaked_terms(vocabulary) -> list[str]:
    """Vocabulary entries the stripper should have removed."""
    return [term for term in vocabulary if LEAK_RE.search(str(term))]


class TextCoefficients:

    @staticmethod
    def load() -> pd.DataFrame:
        df = pd.read_csv(CLEAN_DIR / "characters_model.csv", low_memory=False)
        df = analysis_population(df)

        features = pd.read_csv(CLEAN_DIR / "character_features.csv", low_memory=False)
        features = features.drop_duplicates(subset="page_url")
        df = df.merge(features[["page_url", TEXT_COLUMN]], on="page_url", how="left")

        splits = pd.read_csv(CLEAN_DIR / "split_assignment.csv")
        df = df.merge(splits[["page_url", "split_random"]], on="page_url", how="inner")

        # A page whose opening paragraph was entirely death-stripped has no text
        # left, which is legitimate and must not become a NaN the vectoriser
        # rejects.
        df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str)
        return df

    @staticmethod
    def fit(df: pd.DataFrame) -> pd.DataFrame:
        trainval = df[df["split_random"] == "trainval"]

        vectoriser = TfidfVectorizer(
            max_features=TEXT_MAX_FEATURES,
            min_df=TEXT_MIN_DF,
            stop_words="english",
            strip_accents="unicode",
            sublinear_tf=True,
        )
        X = vectoriser.fit_transform(trainval[TEXT_COLUMN])
        y = trainval[TARGET].astype(int)

        model = LogisticRegression(max_iter=3000)
        model.fit(X, y)

        return pd.DataFrame({
            "term": vectoriser.get_feature_names_out(),
            "coefficient": model.coef_[0],
        }).sort_values("coefficient", ascending=False).reset_index(drop=True)


def main() -> None:
    df = TextCoefficients.load()
    coefficients = TextCoefficients.fit(df)

    out_path = CLEAN_DIR / "q2_text_coefficients.csv"
    coefficients.to_csv(out_path, index=False)

    n_trainval = int((df["split_random"] == "trainval").sum())
    print(f"Fit on {n_trainval:,} trainval characters, "
          f"{len(coefficients)} terms in the vocabulary.\n")

    print(f"Top {TOP_N} pushing toward death:")
    print(coefficients.head(TOP_N).to_string(index=False))
    print(f"\nTop {TOP_N} pushing toward survival:")
    print(coefficients.tail(TOP_N).iloc[::-1].to_string(index=False))

    leaked = leaked_terms(coefficients["term"])
    if leaked:
        raise SystemExit(
            f"death or survival tokens reached the vocabulary: {leaked}"
        )
    print(f"\nNo death or survival token in the {len(coefficients)}-term "
          "vocabulary, checked against the same pattern that stripped the text.")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    from src import setup_run_log
    setup_run_log(__spec__)
    main()
