"""Compute gender effect sizes on mortality odds.

Estimates unadjusted, adjusted (conditioned on prominence, archetype, alignment, 
species, and franchise), and per-franchise effects. Reported as odds ratios with 
95% intervals and percentage-point differences in predicted death probability.

  python -m src.q1_character.effects
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from src.constants import analysis_population
from src.paths import CLEAN_DIR

TARGET = "is_dead_v2"

# Everything the gender estimate is conditioned on. Prominence is included
# because it is the strongest non-metadata predictor, and role because men and
# women are not evenly distributed across archetypes.
CONTROLS = [
    "billing_order_filled", "billing_missing", "archetype",
    "affiliation_alignment", "is_human",
]

MIN_FRANCHISE_N = 100
MIN_PER_GENDER = 20

# Franchise fixed effects separate perfectly when a franchise is almost all
# dead or almost all alive, and the fit then fails to converge. Such franchises
# are excluded from the pooled model and reported.
SEPARATION_MARGIN = 0.03
MIN_LEVEL_COUNT = 10
MIN_POOLED_FRANCHISE_N = 50


class GenderEffect:

    @staticmethod
    def load() -> pd.DataFrame:
        df = pd.read_csv(CLEAN_DIR / "characters_model.csv", low_memory=False)
        df = analysis_population(df)
        df = df[df["gender"].isin(["Male", "Female"])].copy()

        df["female"] = (df["gender"] == "Female").astype(int)
        df["billing_missing"] = df["billing_order"].isna().astype(int)
        df["billing_order_filled"] = df["billing_order"].fillna(
            df["billing_order"].median()
        )
        df["is_human"] = df["is_human"].fillna(0).astype(int)
        for col in ("archetype", "affiliation_alignment"):
            df[col] = df[col].fillna("Unknown").astype(str)

        df[TARGET] = df[TARGET].astype(int)
        return df

    @staticmethod
    def poolable(df: pd.DataFrame) -> pd.DataFrame:
        """Drop franchises that separate, so the fixed-effects fit converges."""
        rates = df.groupby("franchise")[TARGET].agg(["size", "mean"])
        keep = rates[
            (rates["size"] >= MIN_POOLED_FRANCHISE_N)
            & (rates["mean"] > SEPARATION_MARGIN)
            & (rates["mean"] < 1 - SEPARATION_MARGIN)
        ].index
        dropped = sorted(set(rates.index) - set(keep))
        if dropped:
            print(f"Excluded from the pooled model ({len(dropped)}): "
                  f"{', '.join(dropped)}")
        return df[df["franchise"].isin(keep)].copy()

    @staticmethod
    def fit(df: pd.DataFrame, controls: list[str], franchise_effects: bool):
        terms = ["female"] + [
            f"C({c})" if df[c].dtype == object else c for c in controls
        ]
        if franchise_effects:
            terms.append("C(franchise)")
        formula = f"{TARGET} ~ " + " + ".join(terms)
        model = smf.logit(formula, data=df).fit(disp=False, maxiter=200)
        if not model.mle_retvals.get("converged", True):
            raise RuntimeError(f"logit did not converge: {formula[:80]}")
        return model

    @staticmethod
    def marginal_effect(model, df: pd.DataFrame) -> float:
        """Average change in predicted death probability from male to female."""
        as_female = df.assign(female=1)
        as_male = df.assign(female=0)
        return float(
            (model.predict(as_female) - model.predict(as_male)).mean() * 100
        )

    @staticmethod
    def summarise(model, df: pd.DataFrame, label: str, n: int) -> dict:
        coef = model.params["female"]
        low, high = model.conf_int().loc["female"]
        return {
            "estimate": label,
            "n": n,
            "odds_ratio": float(np.exp(coef)),
            "ci_low": float(np.exp(low)),
            "ci_high": float(np.exp(high)),
            "p_value": float(model.pvalues["female"]),
            "pp_change": GenderEffect.marginal_effect(model, df),
        }

    @staticmethod
    def collapse_rare(group: pd.DataFrame, min_count: int = 10) -> pd.DataFrame:
        """Fold thin categorical levels into Other, which is what separates."""
        group = group.copy()
        for col in ("archetype", "affiliation_alignment"):
            counts = group[col].value_counts()
            rare = counts[counts < min_count].index
            group[col] = group[col].where(~group[col].isin(rare), "Other")
        return group

    @staticmethod
    def per_franchise(df: pd.DataFrame) -> pd.DataFrame:
        rows, failed = [], []
        for franchise, group in df.groupby("franchise"):
            counts = group["female"].value_counts()
            if (len(group) < MIN_FRANCHISE_N
                    or counts.get(0, 0) < MIN_PER_GENDER
                    or counts.get(1, 0) < MIN_PER_GENDER
                    or group[TARGET].nunique() < 2):
                continue

            group = GenderEffect.collapse_rare(group)
            # Controls that are constant inside a franchise would break the fit.
            controls = [c for c in CONTROLS if group[c].nunique() > 1]

            try:
                model = GenderEffect.fit(group, controls, franchise_effects=False)
            except Exception as exc:
                failed.append(f"{franchise} ({type(exc).__name__})")
                continue
            if not np.isfinite(model.params.get("female", np.nan)):
                failed.append(f"{franchise} (non-finite coefficient)")
                continue

            row = GenderEffect.summarise(model, group, franchise, len(group))
            row["raw_male"] = 100 * group.loc[group.female == 0, TARGET].mean()
            row["raw_female"] = 100 * group.loc[group.female == 1, TARGET].mean()
            rows.append(row)

        if failed:
            print(f"Per-franchise fit failed for {len(failed)}: {', '.join(failed)}")
        return pd.DataFrame(rows).sort_values("pp_change")


def main() -> None:
    df = GenderEffect.load()
    male, female = df[df.female == 0], df[df.female == 1]
    print(f"{len(df):,} on-screen characters with known gender "
          f"({len(female):,} female, {len(male):,} male)")
    print(f"raw death rate: male {male[TARGET].mean():.1%}, "
          f"female {female[TARGET].mean():.1%} "
          f"({100 * (female[TARGET].mean() - male[TARGET].mean()):+.1f} pp)\n")

    unadjusted = GenderEffect.fit(df, [], franchise_effects=False)

    poolable = GenderEffect.poolable(df)
    adjusted = GenderEffect.fit(poolable, CONTROLS, franchise_effects=True)

    pooled = pd.DataFrame([
        GenderEffect.summarise(unadjusted, df, "unadjusted", len(df)),
        GenderEffect.summarise(
            adjusted, poolable, "adjusted + franchise effects", len(poolable)
        ),
    ])
    print("Pooled effect of being female on dying")
    print(pooled.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    per = GenderEffect.per_franchise(df)
    print(f"\nPer franchise (n>={MIN_FRANCHISE_N}, >={MIN_PER_GENDER} of each gender)")
    print(per[["estimate", "n", "raw_male", "raw_female",
               "odds_ratio", "ci_low", "ci_high", "pp_change", "p_value"]]
          .to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    significant = per[per.p_value < 0.05]
    print(f"\n{len(significant)} of {len(per)} franchises significant at p<0.05: "
          f"{significant[significant.pp_change < 0].shape[0]} where women die less, "
          f"{significant[significant.pp_change > 0].shape[0]} where women die more")

    pooled.to_csv(CLEAN_DIR / "q1_gender_pooled.csv", index=False)
    per.to_csv(CLEAN_DIR / "q1_gender_by_franchise.csv", index=False)
    print(f"\nWrote q1_gender_pooled.csv and q1_gender_by_franchise.csv")


if __name__ == "__main__":
    from src import setup_run_log
    setup_run_log(__spec__)
    main()
