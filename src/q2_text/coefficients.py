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
from src.q1_character.models import (
    TEXT_COLUMN, TEXT_MAX_FEATURES, TEXT_MIN_DF,
)

TARGET = "is_dead_v2"
TOP_N = 20

# Anchored on word boundaries, unlike the substring rule used when stripping the
# text itself. The vocabulary here is single tokens, and a substring test flags
# "soldier" for containing "die" -- an occupation, not a death word. Over-broad
# checks that cry wolf get switched off, which is worse than no check.
LEAK_RE = (
    r"^(death|deaths|dead|die|dies|died|dying|kill|kills|killed|killing|"
    r"murder|murders|murdered|deceased|corpse|corpses|grave|graves|funeral|"
    r"posthumous|posthumously|surviv\w*|alive|living)$"
)


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

    leaked = coefficients[coefficients["term"].str.contains(LEAK_RE, regex=True)]
    if len(leaked):
        raise SystemExit(
            f"death or survival tokens reached the vocabulary: "
            f"{leaked['term'].tolist()}"
        )
    print("\nNo death or survival token in the vocabulary.")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    from src import setup_run_log
    setup_run_log(__spec__)
    main()
