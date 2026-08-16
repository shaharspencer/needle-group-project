"""Test whether characters in the same narrative community share fates.

src/q1_character/features.py partitions each franchise's co-mention graph with
the Louvain method and records which community every character falls into. That
partition is built from topology alone. This module brings the label back and
asks the question the partition was built to answer: given that two characters
sit in the same narrative cluster, are they more likely to meet the same end?

The test statistic follows the pairwise-membership accuracy measure from the
community detection lecture, applied to fate rather than to a reference
partition. For every pair of characters within a franchise, ask two questions:
are they in the same community, and do they share a fate. The statistic is

    D = P(same fate | same community) - P(same fate | different community)

D is zero when community membership carries no information about fate, and
positive when communities are fate-homogeneous.

A raw D above zero proves nothing on its own, because communities differ in size
and the death rate differs across a franchise for reasons that have nothing to
do with narrative clustering. The reference distribution is a permutation null
in the spirit of the configuration model: hold the graph, the partition and the
number of deaths fixed, shuffle which characters carry the deaths, and recompute
D. The observed D is then read against the spread of that null, which is what
the lecture means by needing a null model before modularity means anything.

  python -m src.q1_character.community
"""

import numpy as np
import pandas as pd

from src.constants import analysis_population
from src.paths import CLEAN_DIR

TARGET = "is_dead_v2"

N_PERMUTATIONS = 2000
PERMUTATION_SEED = 0

# Below this many labelled characters a franchise's pair counts are too thin for
# the permutation null to have a usable spread.
MIN_CHARACTERS = 50

# A franchise whose partition is one giant community plus singletons has no
# cross-community pairs to compare against, so D is undefined in practice.
MIN_COMMUNITIES = 2


class CommunityFate:

    @staticmethod
    def load() -> pd.DataFrame:
        characters = pd.read_csv(
            CLEAN_DIR / "characters_model.csv", low_memory=False
        )
        characters = analysis_population(characters)

        features = pd.read_csv(
            CLEAN_DIR / "character_features.csv",
            usecols=["page_url", "community_id", "community_size", "embeddedness"],
            low_memory=False,
        ).drop_duplicates(subset="page_url", keep="first")

        df = characters.merge(features, on="page_url", how="inner")
        return df.dropna(subset=["community_id", TARGET])

    @staticmethod
    def fate_difference(community: np.ndarray, dead: np.ndarray) -> float:
        """P(same fate | same community) - P(same fate | different community).

        Counted in closed form rather than by enumerating pairs. For a franchise
        of n characters the pair count is n(n-1)/2, which is 6.1 million for the
        MCU alone and would make the permutation loop unusable.

        Writing d_c and a_c for the dead and alive counts in community c, the
        same-community pairs are the within-community pairs summed over c, and
        the same-fate ones among them are C(d_c,2) + C(a_c,2). Subtracting both
        from the franchise totals gives the cross-community counts.
        """
        n = len(community)
        total_pairs = n * (n - 1) / 2
        if total_pairs == 0:
            return float("nan")

        total_dead = int(dead.sum())
        total_alive = n - total_dead
        total_same_fate = (
            total_dead * (total_dead - 1) / 2 + total_alive * (total_alive - 1) / 2
        )

        # bincount over both the partition and the partition restricted to the
        # dead gives d_c and the community sizes in one pass each.
        sizes = np.bincount(community)
        dead_counts = np.bincount(community, weights=dead)
        alive_counts = sizes - dead_counts

        within_pairs = float((sizes * (sizes - 1) / 2).sum())
        within_same_fate = float(
            (dead_counts * (dead_counts - 1) / 2).sum()
            + (alive_counts * (alive_counts - 1) / 2).sum()
        )

        between_pairs = total_pairs - within_pairs
        between_same_fate = total_same_fate - within_same_fate
        if within_pairs == 0 or between_pairs == 0:
            return float("nan")

        return within_same_fate / within_pairs - between_same_fate / between_pairs

    @staticmethod
    def test_franchise(community: np.ndarray, dead: np.ndarray,
                       rng: np.random.Generator) -> dict:
        """Observed D against a label-shuffling null of the same size."""
        observed = CommunityFate.fate_difference(community, dead)
        if not np.isfinite(observed):
            return {}

        null = np.empty(N_PERMUTATIONS)
        shuffled = dead.copy()
        for i in range(N_PERMUTATIONS):
            rng.shuffle(shuffled)
            null[i] = CommunityFate.fate_difference(community, shuffled)

        spread = null.std()
        # One-sided: the hypothesis is that communities are more fate-homogeneous
        # than chance, not merely different from chance. The +1 keeps the
        # estimate from reaching exactly zero on a finite number of draws.
        p_value = (1 + int((null >= observed).sum())) / (N_PERMUTATIONS + 1)
        return {
            "d_observed": observed,
            "d_null_mean": float(null.mean()),
            "d_null_sd": float(spread),
            "z": float((observed - null.mean()) / spread) if spread else float("nan"),
            "p_permutation": p_value,
        }

    @staticmethod
    def run(df: pd.DataFrame) -> pd.DataFrame:
        rng = np.random.default_rng(PERMUTATION_SEED)
        rows = []

        for franchise, group in df.groupby("franchise"):
            # Community ids are per-franchise already, but they are not dense
            # after the analysis population filter drops rows, and bincount
            # needs contiguous codes.
            codes = pd.factorize(group["community_id"])[0]
            dead = group[TARGET].astype(float).to_numpy()

            if len(group) < MIN_CHARACTERS or codes.max() + 1 < MIN_COMMUNITIES:
                continue

            result = CommunityFate.test_franchise(codes, dead, rng)
            if not result:
                continue
            rows.append({
                "franchise": franchise,
                "n_characters": len(group),
                "n_communities": int(codes.max() + 1),
                "mortality": float(dead.mean()),
                **result,
            })

        return pd.DataFrame(rows).sort_values("z", ascending=False)


def main() -> None:
    df = CommunityFate.load()
    structure = pd.read_csv(CLEAN_DIR / "q1_community_structure.csv")

    print(f"{len(df):,} characters with a community assignment and a label\n")

    results = CommunityFate.run(df)
    results = results.merge(
        structure[["franchise", "modularity"]], on="franchise", how="left"
    )

    shown = ["franchise", "n_characters", "n_communities", "modularity",
             "mortality", "d_observed", "d_null_sd", "z", "p_permutation"]
    print("Do characters in the same community share fates?")
    print("  D = P(same fate | same community) - P(same fate | different community)")
    print(f"  null: {N_PERMUTATIONS} within-franchise label shuffles\n")
    print(results[shown].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    significant = results[results["p_permutation"] < 0.05]
    positive = significant[significant["d_observed"] > 0]
    print(f"\n{len(significant)} of {len(results)} franchises reject the null at "
          f"p < 0.05; {len(positive)} of those in the fate-homogeneous direction.")

    # Weighted by cast size, so the MCU is not outvoted by Peaky Blinders.
    weights = results["n_characters"]
    pooled = float((results["d_observed"] * weights).sum() / weights.sum())
    print(f"Cast-weighted mean D across {len(results)} franchises: {pooled:+.3f}")

    out_path = CLEAN_DIR / "q1_community_fate.csv"
    results.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    from src import setup_run_log
    setup_run_log(__spec__)
    main()
