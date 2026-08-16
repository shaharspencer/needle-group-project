"""Analyze franchise-level mortality rates.

Computes mortality with Wilson intervals for each franchise, regresses
logit(mortality) on franchise properties weighted by cast size, and clusters
the franchises.

There are 38 franchises and 8 standardised properties to cluster them on. At
that size the failure modes the clustering lecture lists for k-means are all
live, so the clustering is validated along the lines the lecture sets out:

  1. k-means++ seeding, repeated over ten seeds, which addresses the lecture's
     warning that k-means results vary with the initial centroids.
  2. k chosen from the silhouette and from the average distance to the centroid
     across k. The lecture presents the second as the elbow method. They can
     disagree, so both are printed.
  3. Internal validation of the chosen partition: cohesion, separation, and the
     correlation between the distance matrix and the same-cluster adjacency
     matrix.
  4. External validation against a franchise attribute withheld from the
     clustering, scored by entropy and purity.
  5. Ward agglomerative clustering as a cross-check, since 38 points is few
     enough to read off a dendrogram.

  python -m src.q3_franchise.mortality
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.constants import analysis_population
from src.paths import CLEAN_DIR, GOLD_DIR

TARGET_COUNT = "n_dead"
K_RANGE = range(2, 9)
SEEDS = range(10)
MIN_FRANCHISE_N = 30

# Franchise properties the clustering runs on. Mortality is deliberately absent:
# it is the outcome, and clustering on it would guarantee clusters that differ
# in mortality.
CLUSTER_FEATURES = [
    "n_characters", "female_ratio", "n_titles", "n_tv", "first_year",
    "run_span", "mean_rating", "mean_popularity",
]

# Attribute held out of the clustering and used to score it from outside.
EXTERNAL_LABEL = "medium"

# Share of on-screen characters with no parsed status that the annotated sample
# found actually die. Used to place a point estimate between the bounds.
UNLABELLED_DEATH_RATE = 0.085


class FranchiseMortality:

    @staticmethod
    def registry_resolved() -> set:
        """page_urls a death registry already resolves to dead.

        Takes priority over Haiku by construction: fill_unlabelled.py excludes
        these from the rows it ever sends to Haiku. Checked again here so the
        priority holds even if haiku_fill.csv is ever regenerated separately
        from registry_filled.csv.
        """
        path = GOLD_DIR / "fill_unlabelled" / "registry_filled.csv"
        if not path.exists() or path.stat().st_size == 0:
            return set()
        try:
            return set(pd.read_csv(path)["page_url"])
        except pd.errors.EmptyDataError:
            return set()

    @staticmethod
    def haiku_resolved() -> pd.Series:
        """page_url -> dies_in_narrative, Haiku labels for rows no other
        source could settle (see src/labels/fill_unlabelled.py). -1
        (genuinely couldn't tell) is dropped rather than guessed at; those
        rows fall back to the flat UNLABELLED_DEATH_RATE assumption, same as
        every franchise this wasn't run for.
        """
        fill_path = GOLD_DIR / "fill_unlabelled" / "haiku_fill.csv"
        bridge_path = GOLD_DIR / "fill_unlabelled" / "to_annotate.csv"
        if not fill_path.exists() or not bridge_path.exists():
            return pd.Series(dtype="float64")

        fill = pd.read_csv(fill_path)
        bridge = pd.read_csv(bridge_path)[["char_id", "page_url"]]
        merged = fill.merge(bridge, on="char_id", how="inner")
        merged = merged[merged["dies_in_narrative"].isin([0, 1])]
        return merged.set_index("page_url")["dies_in_narrative"]

    @staticmethod
    def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
        """Wilson score interval. Behaves sensibly when n is small or p near 0/1."""
        if total == 0:
            return (float("nan"), float("nan"))
        p = successes / total
        denom = 1 + z**2 / total
        centre = (p + z**2 / (2 * total)) / denom
        half = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
        return (100 * max(0.0, centre - half), 100 * min(1.0, centre + half))

    @staticmethod
    def build() -> pd.DataFrame:
        # onscreen_flags keeps every scraped page, including the rows whose
        # infobox status could not be parsed. Those are on-screen characters
        # with no recorded death, so they belong in the denominator as alive.
        flags = pd.read_csv(CLEAN_DIR / "onscreen_flags.csv")
        onscreen = analysis_population(flags)
        registry_dead_urls = FranchiseMortality.registry_resolved()
        haiku = FranchiseMortality.haiku_resolved()

        rows = []
        for franchise, group in onscreen.groupby("franchise"):
            n = len(group)
            dead = int((group["raw_is_dead"] == 1).sum())
            unlabelled = group[group["raw_is_dead"].isna()]
            n_unlabelled = len(unlabelled)

            # Priority: a death registry beats Haiku, and Haiku is only ever
            # consulted for rows a registry didn't already resolve.
            is_registry_dead = unlabelled["page_url"].isin(registry_dead_urls)
            n_registry_dead = int(is_registry_dead.sum())
            still_open = unlabelled.loc[~is_registry_dead, "page_url"]

            resolved = haiku.reindex(still_open)
            n_haiku_dead = int((resolved == 1).sum())
            n_haiku_alive = int((resolved == 0).sum())
            n_still_unknown = len(still_open) - n_haiku_dead - n_haiku_alive

            low, high = FranchiseMortality.wilson(dead, n)
            rows.append({
                "franchise": franchise,
                "n_characters": n,
                "n_dead": dead,
                "n_unlabelled": n_unlabelled,
                "n_resolved_by_registry": n_registry_dead,
                "n_resolved_by_haiku": n_haiku_dead + n_haiku_alive,
                "mortality": 100 * dead / n,
                "ci_low": low,
                "ci_high": high,
                # Unlabelled rows counted as alive give a lower bound; counting
                # them all as dead gives an upper bound. The annotated sample
                # found 8.5% of them actually die, which sets a point estimate
                # for whatever neither a registry nor Haiku could resolve.
                "mortality_upper": 100 * (dead + n_unlabelled) / n,
                "mortality_adjusted": 100 * (
                    dead + UNLABELLED_DEATH_RATE * n_unlabelled
                ) / n,
                # Same point estimate, but rows a registry or Haiku actually
                # resolved use that answer instead of the flat rate. Only
                # differs from mortality_adjusted where either count is > 0.
                "mortality_haiku_filled": 100 * (
                    dead + n_registry_dead + n_haiku_dead
                    + UNLABELLED_DEATH_RATE * n_still_unknown
                ) / n,
            })
        df = pd.DataFrame(rows)

        characters = pd.read_csv(CLEAN_DIR / "characters_model.csv", low_memory=False)
        labelled = analysis_population(characters)
        extra = labelled.groupby("franchise").agg(
            female_ratio=("gender", lambda s: (s == "Female").mean()),
            median_billing=("billing_order", "median"),
        ).reset_index()
        df = df.merge(extra, on="franchise", how="left")

        titles = pd.read_csv(CLEAN_DIR / "tmdb_titles.csv")
        titles["is_tv"] = (titles["media_type"] == "tv").astype(int)
        aggregates = titles.groupby("franchise").agg(
            n_titles=("tmdb_id", "nunique"),
            n_tv=("is_tv", "sum"),
            first_year=("release_year", "min"),
            last_year=("end_year", "max"),
            mean_rating=("vote_average", "mean"),
            mean_popularity=("popularity", "mean"),
            total_seasons=("n_seasons", "sum"),
            total_episodes=("n_episodes", "sum"),
        ).reset_index()
        # last_year is the end of the last title, not the start of it. For a
        # film that is the release date; for a series it is the last air date.
        # Taking the max over release_year instead made every franchise with a
        # single long-running series look instantaneous -- Grey's Anatomy ran
        # 2005-2026 and scored a run span of 0, as did Lost, The Wire and
        # thirteen others. Run span is the one franchise property that comes
        # close to significance in the regression below, so this mattered.
        aggregates["run_span"] = aggregates["last_year"] - aggregates["first_year"]

        df = df.merge(aggregates, on="franchise", how="left")

        meta = characters.groupby("franchise").agg(
            genre=("genre", "first"),
            medium=("medium", "first"),
            content_rating=("content_rating", "first"),
        ).reset_index()
        df = df.merge(meta, on="franchise", how="left")

        return df.sort_values("mortality", ascending=False).reset_index(drop=True)

    @staticmethod
    def regress(df: pd.DataFrame):
        """
        Weighted least squares on logit(mortality), one row per franchise.

        The unit of analysis is the franchise, not the character. A binomial
        GLM over character counts reports Pearson chi-square over df of 36.6,
        meaning characters within a franchise are nowhere near independent
        draws, and it returns p < 1e-10 for every term on 36 data points.
        Weighting by cast size keeps Peaky Blinders (11 characters) from
        counting as much as the MCU (3,647) without pretending n is 17,503.
        """
        usable = df[df["n_characters"] >= MIN_FRANCHISE_N].copy()
        rate = np.clip(usable["n_dead"] / usable["n_characters"], 0.01, 0.99)
        usable["logit_mortality"] = np.log(rate / (1 - rate))

        formula = (
            "logit_mortality ~ C(medium) + first_year + n_titles + "
            "run_span + mean_rating"
        )
        model = smf.wls(
            formula, data=usable, weights=usable["n_characters"]
        ).fit()
        return model, usable

    @staticmethod
    def cluster(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        """k-means over franchise properties, excluding mortality itself.

        Mortality is the outcome under study, so including it would produce
        clusters that differ in mortality by construction and tell us nothing.
        Standardising first is required rather than cosmetic: first_year runs
        in the thousands and female_ratio in [0,1], and Euclidean distance on
        the raw columns would be a distance on release year alone.
        """
        usable = df.dropna(subset=CLUSTER_FEATURES).copy()
        X = StandardScaler().fit_transform(usable[CLUSTER_FEATURES])

        diagnostics = []
        for k in K_RANGE:
            scores, inertias, labellings = [], [], []
            for seed in SEEDS:
                # init="k-means++" is sklearn's default and is named here
                # because it is the specific remedy the lecture gives for
                # k-means landing in different places on different seeds.
                model = KMeans(
                    n_clusters=k, init="k-means++", n_init=10, random_state=seed
                )
                labels = model.fit_predict(X)
                scores.append(silhouette_score(X, labels))
                # Average distance to the assigned centroid, which is the
                # quantity the elbow method plots. inertia_ is the sum of
                # squared distances, so the root of its mean is the average
                # distance in the original standardised units.
                inertias.append(np.sqrt(model.inertia_ / len(X)))
                labellings.append(labels)

            # Stability: how often two franchises land together across seeds.
            agreements = []
            for a in range(len(labellings)):
                for b in range(a + 1, len(labellings)):
                    same_a = labellings[a][:, None] == labellings[a][None, :]
                    same_b = labellings[b][:, None] == labellings[b][None, :]
                    agreements.append((same_a == same_b).mean())

            diagnostics.append({
                "k": k,
                "silhouette": float(np.mean(scores)),
                "mean_distance_to_centroid": float(np.mean(inertias)),
                "stability": float(np.mean(agreements)),
            })

        diag = pd.DataFrame(diagnostics)
        best_k = int(diag.loc[diag["silhouette"].idxmax(), "k"])
        usable["cluster"] = KMeans(
            n_clusters=best_k, init="k-means++", n_init=10, random_state=0
        ).fit_predict(X)

        # Ward linkage on the same standardised matrix, cut at the same k so the
        # two partitions are comparable. Ward merges the pair whose union adds
        # least to within-cluster variance, which is the agglomerative analogue
        # of what k-means minimises, so a disagreement between them is about the
        # search rather than about the objective.
        usable["cluster_ward"] = fcluster(
            linkage(X, method="ward"), t=best_k, criterion="maxclust"
        ) - 1
        return usable, diag, X

    @staticmethod
    def internal_validation(X: np.ndarray, labels: np.ndarray) -> dict:
        """Cohesion, separation, and the distance-adjacency correlation.

        Cohesion is the mean distance from a point to others in its own
        cluster; separation the mean distance to points outside it. The
        correlation is between the pairwise distance matrix and the 0/1
        same-cluster matrix, and it should come out negative for a good
        partition, because pairs in the same cluster ought to be the close
        ones. The lecture states it as a positive correlation with similarity,
        which is the same statement with the sign flipped.
        """
        distances = squareform(pdist(X))
        same = (labels[:, None] == labels[None, :])
        off_diagonal = ~np.eye(len(X), dtype=bool)

        within = distances[same & off_diagonal]
        between = distances[~same]

        flat_distance = distances[off_diagonal]
        flat_same = same[off_diagonal].astype(float)
        correlation = float(np.corrcoef(flat_distance, flat_same)[0, 1])

        return {
            "cohesion": float(within.mean()),
            "separation": float(between.mean()),
            "separation_over_cohesion": float(between.mean() / within.mean()),
            "distance_adjacency_corr": correlation,
        }

    @staticmethod
    def external_validation(labels: np.ndarray, truth: pd.Series) -> pd.DataFrame:
        """Entropy and purity of the clusters against an attribute held out.

        `medium` is not among the columns clustered on, so this asks whether a
        partition built from cast size, title count, year, run span, rating and
        popularity happens to recover the film/TV distinction. Entropy is the
        weighted mean of each cluster's label entropy, so lower is better;
        purity the weighted mean of each cluster's largest label share, so
        higher is better.
        """
        rows = []
        total = len(labels)
        for cluster_id in sorted(set(labels)):
            members = truth[labels == cluster_id]
            shares = members.value_counts(normalize=True)
            entropy = float(-(shares * np.log2(shares)).sum())
            rows.append({
                "cluster": cluster_id,
                "n": len(members),
                "entropy": entropy,
                "purity": float(shares.max()),
                "dominant": shares.index[0],
                "weight": len(members) / total,
            })

        table = pd.DataFrame(rows)
        overall = pd.DataFrame([{
            "cluster": "weighted overall",
            "n": total,
            "entropy": float((table["entropy"] * table["weight"]).sum()),
            "purity": float((table["purity"] * table["weight"]).sum()),
            "dominant": "",
            "weight": 1.0,
        }])
        return pd.concat([table, overall], ignore_index=True)


def main() -> None:
    df = FranchiseMortality.build()
    print(f"{len(df)} franchises\n")

    shown = ["franchise", "n_characters", "n_unlabelled", "mortality",
             "mortality_adjusted", "mortality_upper", "ci_low", "ci_high",
             "medium", "genre"]
    print(df[shown].to_string(index=False, float_format=lambda v: f"{v:.1f}"))

    n_registry = int(df["n_resolved_by_registry"].sum())
    if n_registry:
        print(f"\n{n_registry} previously-unlabelled rows resolved by a death "
              f"registry (never sent to Haiku).")

    resolved = df[df["n_resolved_by_haiku"] > 0]
    if len(resolved):
        print(f"\n{int(resolved['n_resolved_by_haiku'].sum())} previously-unlabelled "
              f"rows resolved by Haiku across {len(resolved)} franchises -- only "
              "rows a death registry and the infobox both had nothing to say "
              "about (src/labels/fill_unlabelled.py). mortality_haiku_filled uses "
              "those answers in place of the flat-rate assumption:")
        print(resolved[["franchise", "n_unlabelled", "n_resolved_by_haiku",
                        "mortality_adjusted", "mortality_haiku_filled"]]
              .to_string(index=False, float_format=lambda v: f"{v:.1f}"))

    model, usable = FranchiseMortality.regress(df)
    print(f"\nBinomial GLM on {len(usable)} franchises with >= {MIN_FRANCHISE_N} characters")
    summary = pd.DataFrame({
        "coef": model.params,
        "ci_low": model.conf_int()[0],
        "ci_high": model.conf_int()[1],
        "p": model.pvalues,
    })
    print(summary.to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"R2 {model.rsquared:.3f}   adjusted R2 {model.rsquared_adj:.3f}")

    # Persisted, not just printed, so the figure captions and the writeup can
    # read these numbers instead of restating them from memory. Three captions
    # had gone stale against this model before it was written out.
    summary = summary.reset_index(names="term")
    summary["r_squared"] = model.rsquared
    summary["r_squared_adj"] = model.rsquared_adj
    summary["n_franchises"] = len(usable)
    summary.to_csv(CLEAN_DIR / "q3_glm_summary.csv", index=False)

    clustered, diag, X = FranchiseMortality.cluster(df)
    print(f"\nk-means over {len(CLUSTER_FEATURES)} standardised franchise "
          f"properties, {len(clustered)} franchises, k-means++ seeding")
    print("  silhouette picks k; mean distance to centroid is the elbow curve;")
    print("  stability is how often two franchises co-cluster across 10 seeds.")
    print(diag.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    best_k = clustered["cluster"].nunique()
    labels = clustered["cluster"].to_numpy()

    print(f"\nClusters at k={best_k}")
    for cluster_id, group in clustered.groupby("cluster"):
        names = ", ".join(sorted(group["franchise"]))
        print(f"  {cluster_id}: mortality {group['mortality'].mean():5.1f}%  "
              f"n={len(group)}  {names}")

    internal = FranchiseMortality.internal_validation(X, labels)
    print("\nInternal validation of the k-means partition")
    for name, value in internal.items():
        print(f"  {name:28s} {value:+.3f}")

    external = FranchiseMortality.external_validation(
        labels, clustered[EXTERNAL_LABEL].fillna("Unknown")
    )
    print(f"\nExternal validation against '{EXTERNAL_LABEL}', which the "
          f"clustering never saw")
    print(external.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # How far the two search strategies agree, as the share of franchise pairs
    # the two partitions place the same way. This is the pairwise-membership
    # accuracy measure from the community detection lecture, used here to
    # compare two clusterings rather than to score one against ground truth.
    ward = clustered["cluster_ward"].to_numpy()
    same_kmeans = labels[:, None] == labels[None, :]
    same_ward = ward[:, None] == ward[None, :]
    off = ~np.eye(len(labels), dtype=bool)
    agreement = float((same_kmeans[off] == same_ward[off]).mean())
    moved = clustered.loc[
        clustered.groupby("cluster")["cluster_ward"].transform("nunique") > 1,
        "franchise",
    ]
    print(f"\nWard agglomerative clustering cut at k={best_k} agrees with "
          f"k-means on {agreement:.1%} of franchise pairs")
    if len(moved):
        print(f"  k-means clusters that Ward splits: {len(moved)} franchises")

    df.to_csv(CLEAN_DIR / "q3_franchise_mortality.csv", index=False)
    diag.to_csv(CLEAN_DIR / "q3_kmeans_diagnostics.csv", index=False)
    external.to_csv(CLEAN_DIR / "q3_cluster_external.csv", index=False)
    pd.DataFrame([{**internal, "k": best_k, "ward_pair_agreement": agreement}]).to_csv(
        CLEAN_DIR / "q3_cluster_internal.csv", index=False
    )
    clustered[["franchise", "cluster", "cluster_ward"]].to_csv(
        CLEAN_DIR / "q3_franchise_clusters.csv", index=False
    )
    np.savetxt(CLEAN_DIR / "q3_cluster_matrix.csv", X, delimiter=",")
    print("\nWrote q3_franchise_mortality.csv, q3_glm_summary.csv, "
          "q3_kmeans_diagnostics.csv, q3_franchise_clusters.csv, "
          "q3_cluster_internal.csv, q3_cluster_external.csv")


if __name__ == "__main__":
    from src import setup_run_log
    setup_run_log(__spec__)
    main()
