"""Render the writeup figures.

  python -m src.viz.figures
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.paths import CLEAN_DIR, FIG_DIR
from src.viz import style
from src.viz.style import BASELINE, INK, INK_SECONDARY, MUTED, SERIES, SURFACE


# One display name per feature set, so the comparison figure and the fold-spread
# figure label the same thing the same way.
FEATURE_LABELS = {
    "franchise": "Franchise only",
    "character_no_wiki_meta": "Character (no wiki metadata)",
    "character": "Character",
    "network": "Network centrality",
    "community": "Community position",
    "character+network": "Character + network",
    "character+community": "Character + community",
    "character+network+community": "Character + network + community",
    "text": "Text only",
    "character+text": "Character + text",
    "both": "Character + franchise",
    "rich": "All features",
}


def fig_model_comparison() -> None:
    """Which predicts death better, the character or the franchise?"""
    report = pd.read_csv(CLEAN_DIR / "q1_model_comparison.csv")
    # All 12 sets, ordered by grouped score. The writeup cites this figure in
    # place of a results table, so every set the models were run on belongs here.
    scored = report[report.model != "-"]
    order = (
        scored[scored.regime == "grouped"]
        .groupby("features")["pr_auc"].max().sort_values().index.tolist()
    )
    labels = FEATURE_LABELS

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 5.4), sharex=True)
    for ax, regime, title in zip(
        axes, ["random", "grouped"],
        ["Held-out character\n(franchise seen in training)",
         "Held-out franchise\n(never seen in training)"],
    ):
        subset = scored[(scored.regime == regime) & (scored.features.isin(order))]
        best = subset.groupby("features")["pr_auc"].max().reindex(order)
        baseline = report[(report.regime == regime)
                          & (report.features == "baseline:majority")]["pr_auc"].iloc[0]

        y = np.arange(len(best))
        # Dots, not bars. PR-AUC has a meaningful zero and a meaningful chance
        # level, and we want the axis to start below the chance level rather than
        # at zero, so that the interesting range is readable. A bar chart on a
        # truncated axis would misstate the ratios between feature sets; a dot
        # plot carries no such implication.
        colors = [SERIES[0] if v == best.max() else "#7fb3ea" for v in best]
        ax.hlines(y, 0.30, best.values, color="#dce8f7", linewidth=1.4, zorder=1)
        ax.scatter(best.values, y, s=44, color=colors, zorder=3,
                   edgecolor=SURFACE, linewidth=0.8)

        ax.axvline(baseline, color=MUTED, linewidth=1)
        ax.text(baseline, len(best) - 0.35, "  baseline", color=MUTED,
                fontsize=7.5, va="center")

        for i, v in enumerate(best.values):
            ax.text(v + 0.008, i, f"{v:.3f}", va="center", fontsize=7.5,
                    color=INK_SECONDARY)

        ax.set_yticks(y, [labels.get(f, f) for f in best.index], fontsize=7.5)
        ax.set_xlim(0.30, 0.85)
        ax.set_title(title, fontsize=9, color=INK_SECONDARY, pad=6)
        ax.set_xlabel("PR-AUC (best model per feature set)")
        ax.xaxis.grid(True)
        ax.yaxis.grid(False)
        style.strip_spines(ax, keep=("bottom",))
        ax.invert_yaxis()

    axes[1].tick_params(labelleft=False)
    fig.tight_layout()
    fig.suptitle("PR-AUC by feature set",
                 x=0.0, y=1.10, ha="left", fontsize=12, fontweight="bold")
    style.caption(fig,
        "Best score among the three models for each feature set. Left: random character folds. "
        "Right: folds holding out whole franchises. The vertical line is the 0.356 prevalence baseline.")
    fig.savefig(FIG_DIR / "q1_model_comparison.png")
    plt.close(fig)


def fig_gender_forest() -> None:
    """Raw male and female mortality rates by franchise."""
    per = pd.read_csv(CLEAN_DIR / "q1_gender_by_franchise.csv")
    pooled = pd.read_csv(CLEAN_DIR / "q1_gender_pooled.csv")
    per = per.sort_values("pp_change")

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    y = np.arange(len(per))

    for i, (_, row) in enumerate(per.iterrows()):
        ax.plot([row.male_mortality, row.female_mortality], [i, i],
                color="#c9d6e4", linewidth=1.4, zorder=1)
    ax.scatter(per.male_mortality, y, s=32, color=SERIES[1],
               zorder=3, label="Men")
    ax.scatter(per.female_mortality, y, s=32, color=SERIES[0],
               zorder=3, label="Women")

    row = pooled.iloc[0]
    ax.text(0.99, 0.02,
            f"Overall: men {row.male_mortality:.1f}%, women {row.female_mortality:.1f}%",
            transform=ax.transAxes, ha="right", va="bottom",
            color=INK_SECONDARY, fontsize=8)

    ax.set_yticks(y, per.estimate)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Mortality rate (%)")
    ax.set_ylim(-1, len(per))
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    style.strip_spines(ax, keep=("bottom",))
    ax.legend(loc="upper left")

    ax.set_title("Gender gap in mortality, by franchise")
    style.caption(fig,
        "Raw mortality rates for franchises with at least 100 labelled characters "
        "and at least 20 men and 20 women. Lines join the two rates within each franchise.")
    fig.savefig(FIG_DIR / "q1_gender_forest.png")
    plt.close(fig)


# One display name per raw feature column, so the importance figure doesn't
# leak internal column names into the writeup.
IMPORTANCE_LABELS = {
    "page_text_length": "Page length",
    "archetype": "Archetype",
    "gender": "Gender",
    "infobox_field_count_clean": "Infobox field count",
    "prominence_tier": "Prominence tier",
    "billing_order": "Billing order",
    "affiliation_alignment": "Alignment",
    "appearance_count": "Appearance count",
    "episode_count": "Episode count",
    "has_family": "Has family listed",
    "is_human": "Is human",
    "has_alias": "Has alias",
    "has_dob": "Has date of birth",
    "has_image": "Has image",
}


def fig_importance() -> None:
    """Which character attributes carry the signal."""
    imp = pd.read_csv(CLEAN_DIR / "q1_feature_importance.csv").head(12)
    wiki_meta = {"infobox_field_count_clean", "page_text_length", "appearance_count"}
    imp = imp.iloc[::-1]

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    colors = [SERIES[1] if f in wiki_meta else SERIES[0] for f in imp.feature]
    y = np.arange(len(imp))
    ax.barh(y, imp.importance, height=0.62, color=colors)

    ax.set_yticks(y, [IMPORTANCE_LABELS.get(f, f) for f in imp.feature])
    ax.set_xlabel("Training PR-AUC drop after shuffling feature")
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    style.strip_spines(ax, keep=("bottom",))

    handles = [
        plt.Line2D([], [], color=SERIES[0], linewidth=6, label="Character attribute"),
        plt.Line2D([], [], color=SERIES[1], linewidth=6, label="Wiki metadata"),
    ]
    ax.legend(handles=handles, loc="lower right")
    ax.set_title("Which features the model uses")
    style.caption(fig,
        "Training PR-AUC drop after shuffling each feature. Orange bars mark wiki-derived metadata.")
    fig.savefig(FIG_DIR / "q1_feature_importance.png")
    plt.close(fig)


def fig_franchise_mortality() -> None:
    """Franchise mortality with the range caused by missing labels."""
    df = pd.read_csv(CLEAN_DIR / "q2_franchise_mortality.csv")
    df = df[df.n_characters >= 30].sort_values("mortality_haiku_filled")

    fig, ax = plt.subplots(figsize=(7.6, 7.6))
    y = np.arange(len(df))

    # Wide light band: how far the estimate could move if every unlabelled
    # character turned out to be dead.
    ax.barh(y, df.mortality_upper - df.mortality, left=df.mortality,
            height=0.5, color="#cde2fb", label="range if unlabelled characters died")

    ax.scatter(df.mortality_haiku_filled, y, s=28, color=SERIES[0], zorder=3,
               label="estimate")

    ax.set_yticks(y, df.franchise)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of on-screen characters who die (%)")
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    style.strip_spines(ax, keep=("bottom",))
    ax.legend(loc="lower right")

    ax.set_title("Mortality rate by franchise")
    style.caption(fig,
        "Dots use the Haiku-resolved point estimate. Blue bands show how high the rate "
        "would be if every remaining unlabelled character died.")
    fig.savefig(FIG_DIR / "q2_franchise_mortality.png")
    plt.close(fig)


def fig_runspan() -> None:
    """Franchise properties against mortality for Q2."""
    df = pd.read_csv(CLEAN_DIR / "q2_franchise_mortality.csv")
    df = df[(df.n_characters >= 30) & df.run_span.notna()]

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    sizes = 18 + 120 * (df.n_characters / df.n_characters.max()) ** 0.5
    ax.scatter(df.run_span, df.mortality_haiku_filled, s=sizes,
               color=SERIES[0], alpha=0.75, edgecolor=SURFACE, linewidth=1.5)

    fit = np.polyfit(df.run_span, df.mortality_haiku_filled, 1)
    xs = np.linspace(df.run_span.min(), df.run_span.max(), 50)
    ax.plot(xs, np.polyval(fit, xs), color=SERIES[1], linewidth=2)

    # Label only the extremes, never every point.
    for _, row in pd.concat([
        df.nlargest(3, "mortality_haiku_filled"), df.nsmallest(3, "mortality_haiku_filled"),
        df.nlargest(2, "run_span"),
    ]).drop_duplicates("franchise").iterrows():
        ax.annotate(row.franchise, (row.run_span, row.mortality_haiku_filled),
                    textcoords="offset points", xytext=(7, 4),
                    fontsize=7.5, color=INK_SECONDARY)

    ax.set_xlabel("Years between first and last release")
    ax.set_ylabel("Mortality (%)")
    style.strip_spines(ax)

    ax.set_title("Does a longer run mean more deaths?")
    style.caption(fig,
        "Marker area is cast size. The line is a linear trend fit.")
    fig.savefig(FIG_DIR / "q2_run_span.png")
    plt.close(fig)


def fig_corpus() -> None:
    """What the scraped corpus actually contains."""
    comp = pd.read_csv(CLEAN_DIR / "corpus_composition.csv").set_index("category")
    share = comp["share_weighted"]
    parts = [
        ("On-screen characters", share["on-screen character"], SERIES[0]),
        ("Characters never filmed", share["character, never on screen"], "#9ec5f4"),
        ("Not a character", share["not an individual character"], BASELINE),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 1.7))
    left = 0.0
    for label, value, color in parts:
        ax.barh([0], [value], left=left, height=0.5, color=color)
        ax.text(left + value / 2, 0, f"{value:.0f}%", ha="center", va="center",
                fontsize=9, color=SURFACE if color == SERIES[0] else INK,
                fontweight="bold")
        ax.text(left + value / 2, -0.45, label, ha="center", va="top",
                fontsize=8, color=INK_SECONDARY)
        left += value + 0.4  # 2px-equivalent surface gap between segments

    ax.set_xlim(0, 101)
    ax.set_ylim(-1.1, 0.5)
    ax.axis("off")
    ax.set_title("Page type in the raw scrape")
    style.caption(fig,
        f"Estimated from {comp['n_annotated'].iloc[0]} annotated pages and reweighted to the "
        f"{comp['n_corpus_represented'].iloc[0]:,} pages they represent.")
    fig.savefig(FIG_DIR / "corpus_composition.png")
    plt.close(fig)


def fig_community_fate() -> None:
    """Do characters in the same Louvain community share fates?"""
    fate = pd.read_csv(CLEAN_DIR / "q1_community_fate.csv")
    fate = fate.sort_values("d_observed")
    display_names = fate["franchise"].replace({
        "Breaking Bad / Better Call Saul": "Breaking Bad universe",
    })

    fig, axes = plt.subplots(
        1, 2, figsize=(10.5, 6.2), gridspec_kw={"width_ratios": [1.35, 1]}
    )

    ax = axes[0]
    y = np.arange(len(fate))
    colors = [SERIES[0] if value > 0 else "#c9d6e4"
              for value in fate["d_observed"]]
    ax.barh(y, fate["d_observed"], height=0.7, color=colors)
    ax.axvline(0, color=INK, linewidth=0.9)

    ax.set_yticks(y, display_names, fontsize=7)
    ax.set_xlabel("D: within-community minus between-community same-fate rate")
    ax.set_title("Observed difference by franchise", fontsize=9,
                 color=INK_SECONDARY, pad=6)
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    style.strip_spines(ax, keep=("bottom",))

    # Right panel: the partition has to be worth interpreting before its
    # relationship with fate is worth reading, so modularity goes on one axis.
    ax = axes[1]
    ax.scatter(fate["modularity"], fate["d_observed"], s=30,
               color=colors,
               edgecolor=SURFACE, linewidth=0.6, zorder=3)
    ax.axhline(0, color=INK, linewidth=0.9)

    # Labelling only the extremes keeps the panel readable; the left panel
    # already names every franchise.
    notable = fate[(fate["d_observed"] > 0.05) | (fate["d_observed"] < -0.02)]
    for _, row in notable.iterrows():
        label = ("Breaking Bad universe"
                 if row["franchise"] == "Breaking Bad / Better Call Saul"
                 else row["franchise"])
        ax.annotate(label, (row["modularity"], row["d_observed"]),
                    fontsize=6.5, color=INK_SECONDARY,
                    xytext=(5, 3), textcoords="offset points")

    ax.set_xlabel("Modularity Q of the franchise partition")
    ax.set_ylabel("D, excess same-fate rate within communities")
    ax.set_title("Fate concentration vs. partition strength", fontsize=9,
                 color=INK_SECONDARY, pad=6)
    style.strip_spines(ax)

    fig.tight_layout()
    fig.suptitle("Do deaths cluster within communities?",
                 x=0.0, y=1.04, ha="left", fontsize=12, fontweight="bold")
    style.caption(fig,
        "D is the difference between the same-fate rate for character pairs in the same "
        "community and pairs in different communities. Positive values indicate more "
        "same-fate pairs within communities.")
    fig.savefig(FIG_DIR / "q1_community_fate.png")
    plt.close(fig)


def fig_cost_curves() -> None:
    """Where the cost-minimising threshold sits, for each assumed cost ratio."""
    curves = pd.read_csv(CLEAN_DIR / "q1_cost_curves.csv")
    optima = pd.read_csv(CLEAN_DIR / "q1_cost_optima.csv")

    fig, ax = plt.subplots(figsize=(7.0, 4.3))

    ratios = sorted(curves["cost_ratio"].unique())
    for i, ratio in enumerate(ratios):
        subset = curves[curves["cost_ratio"] == ratio]
        ax.plot(subset["threshold"], subset["loss"],
                color=SERIES[i % len(SERIES)], linewidth=1.5,
                label=f"w1 = {ratio:g}")
    for i, ratio in enumerate(ratios):
        best = optima[optima["cost_ratio"] == ratio].iloc[0]
        ax.scatter([best["threshold"]], [best["loss"]],
                   color=SERIES[i % len(SERIES)], s=26, zorder=4,
                   edgecolor=SURFACE, linewidth=0.8)

    ax.axvline(0.5, color=MUTED, linewidth=1)
    ax.text(0.5, ax.get_ylim()[1], " default 0.5", color=MUTED, fontsize=7.5,
            va="top")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Weighted loss per character")
    ax.legend(loc="upper left", ncol=2, bbox_to_anchor=(0.02, 0.98))
    ax.set_ylim(0, None)
    style.strip_spines(ax)

    fig.tight_layout()
    fig.suptitle("Choosing a decision threshold",
                 x=0.0, y=1.06, ha="left", fontsize=12, fontweight="bold")
    style.caption(fig,
        "All-feature forest: loss is w1*FN + FP on grouped out-of-fold predictions. Dots mark each minimum.")
    fig.savefig(FIG_DIR / "q1_cost_curves.png")
    plt.close(fig)


def fig_fold_spread() -> None:
    """Per-fold PR-AUC, so the gaps between feature sets can be read with a spread."""
    folds = pd.read_csv(CLEAN_DIR / "q1_fold_scores.csv")

    order = (
        folds[folds.regime == "grouped"]
        .groupby("features")["pr_auc"].mean().sort_values().index.tolist()
    )

    # sharex as well as sharey. With independent x-axes each panel rescales to
    # its own range, which hides the thing the figure exists to show: every
    # feature set scores lower when whole franchises are held out.
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.6), sharey=True, sharex=True)
    for ax, regime, title in zip(
        axes, ["random", "grouped"],
        ["Held-out character", "Held-out franchise"],
    ):
        subset = folds[folds.regime == regime]
        for i, name in enumerate(order):
            values = subset[subset.features == name]["pr_auc"].to_numpy()
            ax.scatter(values, np.full(len(values), i), s=14,
                       color="#c9d6e4", zorder=2)
            ax.scatter([values.mean()], [i], s=42, color=SERIES[0], zorder=3,
                       edgecolor=SURFACE, linewidth=0.8)
            ax.plot([values.min(), values.max()], [i, i],
                    color="#c9d6e4", linewidth=1.2, zorder=1)

        ax.set_yticks(np.arange(len(order)),
                      [FEATURE_LABELS.get(f, f) for f in order], fontsize=7.5)
        ax.set_xlabel("PR-AUC per fold")
        ax.set_title(title, fontsize=9, color=INK_SECONDARY, pad=6)
        ax.xaxis.grid(True)
        ax.yaxis.grid(False)
        style.strip_spines(ax, keep=("bottom",))

    fig.tight_layout()
    fig.suptitle("PR-AUC spread across folds",
                 x=0.0, y=1.05, ha="left", fontsize=12, fontweight="bold")
    style.caption(fig,
        "Random forest: small points show individual folds; large points show their mean. Both panels use the same scale.")
    fig.savefig(FIG_DIR / "q1_fold_spread.png")
    plt.close(fig)


def fig_cluster_diagnostics() -> None:
    """How k was chosen, and what the resulting partition looks like."""
    from scipy.cluster.hierarchy import dendrogram, linkage

    diag = pd.read_csv(CLEAN_DIR / "q2_kmeans_diagnostics.csv")
    clusters = pd.read_csv(CLEAN_DIR / "q2_franchise_clusters.csv")
    X = np.loadtxt(CLEAN_DIR / "q2_cluster_matrix.csv", delimiter=",")

    # Match the report's text width so labels are not shrunk when embedded.
    fig = plt.figure(figsize=(7.2, 4.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.5], height_ratios=[1, 1],
                          hspace=0.55, wspace=0.28,
                          top=0.85, bottom=0.32, left=0.08, right=0.98)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(diag["k"], diag["mean_distance_to_centroid"], color=SERIES[0],
            marker="o", markersize=4, linewidth=1.6)
    ax.set_ylabel("Mean distance\nto centroid", fontsize=8)
    ax.set_title("Elbow curve", fontsize=9, color=INK_SECONDARY, pad=4)
    ax.tick_params(labelbottom=False)
    style.strip_spines(ax)

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(diag["k"], diag["silhouette"], color=SERIES[1],
            marker="s", markersize=4, linewidth=1.6)
    ax.set_xlabel("k")
    ax.set_ylabel("Silhouette", fontsize=8)
    ax.set_title("Silhouette score", fontsize=9, color=INK_SECONDARY, pad=4)
    style.strip_spines(ax)

    ax = fig.add_subplot(gs[:, 1])
    linkage_matrix = linkage(X, method="ward")
    dendrogram(
        linkage_matrix, ax=ax, labels=clusters["franchise"].tolist(),
        leaf_font_size=6.5, color_threshold=0,
        above_threshold_color=INK_SECONDARY,
    )
    ax.set_ylabel("Ward merge distance")
    ax.set_title("Ward dendrogram over the same standardised properties",
                 fontsize=9, color=INK_SECONDARY, pad=6)
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)
    style.strip_spines(ax, keep=("left",))
    ax.tick_params(axis="x", rotation=90)

    fig.suptitle("How many franchise clusters?",
                 x=0.0, y=0.97, ha="left", fontsize=12, fontweight="bold")
    style.caption(fig,
        "Left: two internal diagnostics across k. Right: Ward dendrogram using the same "
        "standardised franchise properties.")
    fig.savefig(FIG_DIR / "q2_cluster_diagnostics.png")
    plt.close(fig)


def fig_reliability() -> None:
    """Chance-corrected agreement, before and after the rubric was adjudicated."""
    report = pd.read_csv(CLEAN_DIR / "label_reliability.csv")
    inter = report[report["comparison"].str.startswith("pass")]

    labels = ["is_character", "appears_onscreen", "dies_in_narrative"]
    v1 = [inter[(inter.comparison.str.contains("v1")) & (inter.label == l)]["kappa"].iloc[0]
          for l in labels]
    v2 = [inter[(inter.comparison.str.contains("v2")) & (inter.label == l)]["kappa"].iloc[0]
          for l in labels]

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    y = np.arange(len(labels))
    height = 0.36
    ax.barh(y + height / 2, v1, height=height, color="#c9d6e4",
            label="v1 rubric (passes 1 and 2)")
    ax.barh(y - height / 2, v2, height=height, color=SERIES[0],
            label="v2 rubric (passes 3 and 4)")

    # Landis and Koch bands, as given in the evaluation lecture. Rules go on the
    # band boundaries, names in the middle of the band they describe.
    for edge in (0.2, 0.4, 0.6, 0.8):
        ax.axvline(edge, color=BASELINE, linewidth=0.8)
    for centre, name in [(0.1, "slight"), (0.3, "fair"), (0.5, "moderate"),
                         (0.7, "substantial")]:
        ax.text(centre, -0.72, name, color=MUTED, fontsize=7,
                va="bottom", ha="center")

    for position, value in zip(y + height / 2, v1):
        ax.text(value + 0.012, position, f"{value:.2f}", va="center",
                fontsize=7.5, color=INK_SECONDARY)
    for position, value in zip(y - height / 2, v2):
        ax.text(value + 0.012, position, f"{value:.2f}", va="center",
                fontsize=7.5, color=INK_SECONDARY)

    ax.set_yticks(y, ["Is it a character?", "Does it appear on screen?",
                      "Does it die?"])
    ax.set_xlim(0, 0.9)
    ax.set_xlabel("Cohen's kappa between two annotation runs")
    ax.legend(loc="lower right")
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    style.strip_spines(ax, keep=("bottom",))
    ax.invert_yaxis()

    fig.tight_layout()
    fig.suptitle("Label and agreement by rubric version",
                 x=0.0, y=1.04, ha="left", fontsize=12, fontweight="bold")
    style.caption(fig,
        "Cohen's kappa from two annotation runs under each rubric version. "
        "Vertical lines mark the Landis and Koch agreement bands.")
    fig.savefig(FIG_DIR / "label_reliability.png")
    plt.close(fig)


def main() -> None:
    style.apply()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    for name, fn in [
        ("corpus_composition", fig_corpus),
        ("label_reliability", fig_reliability),
        ("q1_model_comparison", fig_model_comparison),
        ("q1_fold_spread", fig_fold_spread),
        ("q1_gender_forest", fig_gender_forest),
        ("q1_feature_importance", fig_importance),
        ("q1_community_fate", fig_community_fate),
        ("q1_cost_curves", fig_cost_curves),
        ("q2_franchise_mortality", fig_franchise_mortality),
        ("q2_run_span", fig_runspan),
        ("q2_cluster_diagnostics", fig_cluster_diagnostics),
    ]:
        fn()
        print(f"  wrote {name}.png")

    print(f"\n{len(list(FIG_DIR.glob('*.png')))} figures in {FIG_DIR}")


if __name__ == "__main__":
    from src import setup_run_log
    setup_run_log(__spec__)
    main()
