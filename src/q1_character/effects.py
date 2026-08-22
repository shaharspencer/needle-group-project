"""Describe mortality rates by gender, overall and within franchises.

This is a descriptive comparison: it reports male and female mortality rates
and their percentage-point difference without an adjusted statistical model.

  python -m src.q1_character.effects
"""

import pandas as pd

from src.constants import analysis_population
from src.paths import CLEAN_DIR

TARGET = "is_dead_v2"
MIN_FRANCHISE_N = 100
MIN_PER_GENDER = 20


class GenderEffect:

    @staticmethod
    def load() -> pd.DataFrame:
        df = pd.read_csv(CLEAN_DIR / "characters_model.csv", low_memory=False)
        df = analysis_population(df)
        df = df[df["gender"].isin(["Male", "Female"])].copy()
        df[TARGET] = df[TARGET].astype(int)
        return df

    @staticmethod
    def summarise(group: pd.DataFrame, label: str) -> dict:
        male = group.loc[group["gender"] == "Male", TARGET]
        female = group.loc[group["gender"] == "Female", TARGET]
        male_rate = 100 * male.mean()
        female_rate = 100 * female.mean()
        return {
            "estimate": label,
            "n": len(group),
            "n_male": len(male),
            "n_female": len(female),
            "male_mortality": male_rate,
            "female_mortality": female_rate,
            "pp_change": female_rate - male_rate,
        }

    @staticmethod
    def per_franchise(df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for franchise, group in df.groupby("franchise"):
            counts = group["gender"].value_counts()
            if (len(group) < MIN_FRANCHISE_N
                    or counts.get("Male", 0) < MIN_PER_GENDER
                    or counts.get("Female", 0) < MIN_PER_GENDER):
                continue
            rows.append(GenderEffect.summarise(group, franchise))
        return pd.DataFrame(rows).sort_values("pp_change")


def main() -> None:
    df = GenderEffect.load()
    pooled = pd.DataFrame([GenderEffect.summarise(df, "all franchises")])
    per = GenderEffect.per_franchise(df)

    print("Raw mortality rates by gender")
    print(pooled.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\nPer franchise (n>={MIN_FRANCHISE_N}, "
          f">={MIN_PER_GENDER} of each gender)")
    print(per.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    pooled.to_csv(CLEAN_DIR / "q1_gender_pooled.csv", index=False)
    per.to_csv(CLEAN_DIR / "q1_gender_by_franchise.csv", index=False)
    print("\nWrote q1_gender_pooled.csv and q1_gender_by_franchise.csv")


if __name__ == "__main__":
    from src import setup_run_log
    setup_run_log(__spec__)
    main()
