"""Describe whether characters in the same Louvain community share fates.

For each franchise, computes
    D = P(same fate | same community) - P(same fate | different community).
A positive value means same-community pairs share a fate more often.

  python -m src.q1_character.community
"""

import numpy as np
import pandas as pd

from src.constants import analysis_population
from src.paths import CLEAN_DIR

TARGET = "is_dead_v2"
MIN_CHARACTERS = 50
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
        """Difference in same-fate rates within and between communities."""
        n = len(community)
        total_pairs = n * (n - 1) / 2
        if total_pairs == 0:
            return float("nan")

        total_dead = int(dead.sum())
        total_alive = n - total_dead
        total_same_fate = (
            total_dead * (total_dead - 1) / 2
            + total_alive * (total_alive - 1) / 2
        )

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
    def run(df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for franchise, group in df.groupby("franchise"):
            codes = pd.factorize(group["community_id"])[0]
            dead = group[TARGET].astype(float).to_numpy()
            if len(group) < MIN_CHARACTERS or codes.max() + 1 < MIN_COMMUNITIES:
                continue
            difference = CommunityFate.fate_difference(codes, dead)
            if not np.isfinite(difference):
                continue
            rows.append({
                "franchise": franchise,
                "n_characters": len(group),
                "n_communities": int(codes.max() + 1),
                "mortality": float(dead.mean()),
                "d_observed": difference,
            })
        return pd.DataFrame(rows).sort_values("d_observed", ascending=False)


def main() -> None:
    df = CommunityFate.load()
    structure = pd.read_csv(CLEAN_DIR / "q1_community_structure.csv")
    results = CommunityFate.run(df).merge(
        structure[["franchise", "modularity"]], on="franchise", how="left"
    )

    shown = ["franchise", "n_characters", "n_communities", "modularity",
             "mortality", "d_observed"]
    print("D = P(same fate | same community) - "
          "P(same fate | different community)\n")
    print(results[shown].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    weights = results["n_characters"]
    pooled = float((results["d_observed"] * weights).sum() / weights.sum())
    print(f"\nCast-weighted mean D across {len(results)} franchises: {pooled:+.3f}")

    out_path = CLEAN_DIR / "q1_community_fate.csv"
    results.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    from src import setup_run_log
    setup_run_log(__spec__)
    main()
