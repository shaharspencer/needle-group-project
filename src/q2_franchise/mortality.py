"""Analyze franchise-level mortality rates.

Computes a mortality estimate for each franchise, fits an ordinary linear
regression on franchise properties, and clusters standardized properties using
k-means++. Silhouette, elbow, course validation measures, and Ward clustering
are used to check the clustering.

  python -m src.q2_franchise.mortality
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
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

            rows.append({
                "franchise": franchise,
                "n_characters": n,
                "n_dead": dead,
                "n_unlabelled": n_unlabelled,
                "n_resolved_by_registry": n_registry_dead,
                "n_resolved_by_haiku": n_haiku_dead + n_haiku_alive,
                "mortality": 100 * dead / n,
                # Unlabelled rows counted as alive give a lower bound; counting
                # them all as dead gives an upper bound. The annotated sample
                # found 8.5% of them actually die, which sets a point estimate
                # for whatever neither a registry nor Haiku could resolve.
                "mortality_upper": 100 * (dead + n_unlabelled) / n,
                # Rows resolved by a registry or Haiku use that answer. The
                # remaining unknown rows use the 8.5% rate from the labelled
                # sample of unparseable pages.
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
        # Taking the max over release_year instead would give a run span of 0
        # to a franchise containing one long-running series.
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
        """Ordinary linear regression with one row per franchise."""
        usable = df[df["n_characters"] >= MIN_FRANCHISE_N].copy()
        columns = ["medium", "first_year", "n_titles", "run_span", "mean_rating"]
        X = pd.get_dummies(usable[columns], columns=["medium"], drop_first=True,
                           dtype=float)
        y = usable["mortality_haiku_filled"]
        model = LinearRegression().fit(X, y)
        return model, usable, X

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
            scores, mean_distances, labellings = [], [], []
            for seed in SEEDS:
                # init="k-means++" is sklearn's default and is named here
                # because it is the specific remedy the lecture gives for
                # k-means landing in different places on different seeds.
                model = KMeans(
                    n_clusters=k, init="k-means++", n_init=10, random_state=seed
                )
                labels = model.fit_predict(X)
                scores.append(silhouette_score(X, labels))
                distances = np.linalg.norm(X - model.cluster_centers_[labels], axis=1)
                mean_distances.append(float(distances.mean()))
                labellings.append(labels)

            # Stability: agreement on whether each pair is together or apart.
            agreements = []
            off_diagonal = ~np.eye(len(X), dtype=bool)
            for a in range(len(labellings)):
                for b in range(a + 1, len(labellings)):
                    same_a = labellings[a][:, None] == labellings[a][None, :]
                    same_b = labellings[b][:, None] == labellings[b][None, :]
                    agreements.append(
                        (same_a[off_diagonal] == same_b[off_diagonal]).mean()
                    )

            diagnostics.append({
                "k": k,
                "silhouette": float(np.mean(scores)),
                "mean_distance_to_centroid": float(np.mean(mean_distances)),
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
             "mortality_haiku_filled", "mortality_upper", "medium", "genre"]
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
              "about (src/labels/fill_unlabelled.py):")
        print(resolved[["franchise", "n_unlabelled", "n_resolved_by_haiku",
                        "mortality_haiku_filled"]]
              .to_string(index=False, float_format=lambda v: f"{v:.1f}"))

    model, usable, X = FranchiseMortality.regress(df)
    print(f"\nLinear regression on {len(usable)} franchises with "
          f">= {MIN_FRANCHISE_N} characters")
    summary = pd.DataFrame({
        "term": ["intercept", *X.columns],
        "coefficient": [model.intercept_, *model.coef_],
    })
    print(summary.to_string(float_format=lambda v: f"{v:.4f}"))
    r_squared = model.score(X, usable["mortality_haiku_filled"])
    print(f"R2 {r_squared:.3f}")

    # Persisted, not just printed, so the figure captions and the writeup can
    # read these numbers instead of restating them from memory. Three captions
    # had gone stale against this model before it was written out.
    summary["r_squared"] = r_squared
    summary["n_franchises"] = len(usable)
    summary.to_csv(CLEAN_DIR / "q2_regression_summary.csv", index=False)

    clustered, diag, X = FranchiseMortality.cluster(df)
    print(f"\nk-means over {len(CLUSTER_FEATURES)} standardised franchise "
          f"properties, {len(clustered)} franchises, k-means++ seeding")
    print("  silhouette picks k; mean distance to centroid is the elbow curve;")
    print("  stability is agreement on whether pairs stay together or apart across seeds.")
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

    df.to_csv(CLEAN_DIR / "q2_franchise_mortality.csv", index=False)
    diag.to_csv(CLEAN_DIR / "q2_kmeans_diagnostics.csv", index=False)
    external.to_csv(CLEAN_DIR / "q2_cluster_external.csv", index=False)
    pd.DataFrame([{**internal, "k": best_k, "ward_pair_agreement": agreement}]).to_csv(
        CLEAN_DIR / "q2_cluster_internal.csv", index=False
    )
    clustered[["franchise", "cluster", "cluster_ward"]].to_csv(
        CLEAN_DIR / "q2_franchise_clusters.csv", index=False
    )
    np.savetxt(CLEAN_DIR / "q2_cluster_matrix.csv", X, delimiter=",")
    print("\nWrote q2_franchise_mortality.csv, q2_regression_summary.csv, "
          "q2_kmeans_diagnostics.csv, q2_franchise_clusters.csv, "
          "q2_cluster_internal.csv, q2_cluster_external.csv")


if __name__ == "__main__":
    from src import setup_run_log
    setup_run_log(__spec__)
    main()
