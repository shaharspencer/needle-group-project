"""Render the writeup figures.

  python -m src.viz.figures
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.paths import CLEAN_DIR, FIG_DIR
from src.q3_franchise.mortality import FranchiseMortality
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
    fig.suptitle("Character attributes predict death better than the franchise does",
                 x=0.0, y=1.10, ha="left", fontsize=12, fontweight="bold")
    style.caption(fig,
        "Best of three models per feature set, ranked by the grouped score. "
        "Axis starts at 0.30, just below the 0.356 chance level; dots rather than bars, so the truncation does not "
        "misstate ratios. Left: franchise identity is available to the model. Right: whole franchises are held out. "
        "Every set loses ground between the panels, "
        "and franchise-only drops to 0.398, close to the 0.356 baseline, since its one-hot column is then always zero.")
    fig.savefig(FIG_DIR / "q1_model_comparison.png")
    plt.close(fig)


def fig_gender_forest() -> None:
    """Effect of being female on the odds of dying, per franchise."""
    per = pd.read_csv(CLEAN_DIR / "q1_gender_by_franchise.csv")
    pooled = pd.read_csv(CLEAN_DIR / "q1_gender_pooled.csv")
    per = per.sort_values("odds_ratio")

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    y = np.arange(len(per))

    significant = per.p_value < 0.05
    for i, (_, row) in enumerate(per.iterrows()):
        color = SERIES[0] if row.p_value < 0.05 else MUTED
        ax.plot([row.ci_low, row.ci_high], [i, i], color=color,
                linewidth=1.4, solid_capstyle="round", alpha=0.9)
    ax.scatter(per.odds_ratio[significant], y[significant.values],
               s=34, color=SERIES[0], zorder=3, label="p < 0.05")
    ax.scatter(per.odds_ratio[~significant], y[~significant.values],
               s=34, facecolor=SURFACE, edgecolor=MUTED, linewidth=1.2,
               zorder=3, label="not significant")

    ax.axvline(1.0, color=INK_SECONDARY, linewidth=1.1)
    ax.text(1.05, len(per) - 0.6, "no effect", color=INK_SECONDARY, fontsize=7.5)

    row = pooled[pooled.estimate.str.startswith("adjusted")].iloc[0]
    ax.axvline(row.odds_ratio, color=SERIES[1], linewidth=1.2)
    ax.text(row.odds_ratio * 0.97, len(per) - 0.6,
            f"pooled {row.odds_ratio:.2f}", color=SERIES[1], fontsize=7.5, ha="right")

    ax.set_yticks(y, per.estimate)
    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.25, 0.5, 1, 2])
    ax.set_xticklabels(["0.1", "0.25", "0.5", "1", "2"])
    ax.set_xlim(0.05, 3.0)
    ax.set_xlabel("Odds ratio for dying, women vs men (log scale)")
    ax.set_ylim(-1, len(per))
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    style.strip_spines(ax, keep=("bottom",))
    # Upper left: the franchises are sorted by odds ratio, so the top rows sit
    # nearest 1.0 and leave the left corner empty. A lower-left legend lands on
    # top of Ozark and Breaking Bad, which have the widest intervals.
    ax.legend(loc="upper left")

    ax.set_title("Women are less likely to die in every franchise we could measure")
    style.caption(fig,
        "Adjusted for prominence, role, alignment and species. Bars are 95% intervals. "
        f"{significant.sum()} of {len(per)} franchises reach p<0.05 and none point the other way; "
        "the weak cases are null effects, not reversals.")
    fig.savefig(FIG_DIR / "q1_gender_forest.png")
    plt.close(fig)


def fig_importance() -> None:
    """Which character attributes carry the signal."""
    imp = pd.read_csv(CLEAN_DIR / "q1_feature_importance.csv").head(12)
    wiki_meta = {"infobox_field_count_clean", "page_text_length", "appearance_count"}
    imp = imp.iloc[::-1]

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    colors = [SERIES[1] if f in wiki_meta else SERIES[0] for f in imp.feature]
    y = np.arange(len(imp))
    ax.barh(y, imp.importance, height=0.62, color=colors,
            xerr=imp["std"], error_kw={"ecolor": MUTED, "elinewidth": 0.8})

    ax.set_yticks(y, imp.feature)
    ax.set_xlabel("Permutation importance (drop in PR-AUC)")
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    style.strip_spines(ax, keep=("bottom",))

    handles = [
        plt.Line2D([], [], color=SERIES[0], linewidth=6, label="Character attribute"),
        plt.Line2D([], [], color=SERIES[1], linewidth=6, label="Wiki metadata"),
    ]
    ax.legend(handles=handles, loc="lower right")
    ax.set_title("The two strongest features measure the wiki, not the character")
    style.caption(fig,
        "Character-only random forest. How much fans wrote about a character outranks anything about the character. "
        "Gender and archetype are the strongest substantive features.")
    fig.savefig(FIG_DIR / "q1_feature_importance.png")
    plt.close(fig)


def fig_franchise_mortality() -> None:
    """Franchise mortality with both sampling and labelling uncertainty."""
    df = pd.read_csv(CLEAN_DIR / "q3_franchise_mortality.csv")
    # mortality_haiku_filled is the column the writeup's Q3 table quotes, so it
    # is the one plotted. This used to plot mortality_adjusted against a Wilson
    # interval computed on the raw parsed rate, which are different quantities:
    # for Lost and Boardwalk Empire the plotted dot fell outside its own plotted
    # interval.
    df = df[df.n_characters >= 30].sort_values("mortality_haiku_filled")

    fig, ax = plt.subplots(figsize=(7.6, 7.6))
    y = np.arange(len(df))

    # Wide light band: how far the estimate could move if every unlabelled
    # character turned out to be dead.
    ax.barh(y, df.mortality_upper - df.mortality, left=df.mortality,
            height=0.5, color="#cde2fb", label="range if unlabelled characters died")

    # Wilson interval recomputed around the estimate actually being plotted, so
    # the dot always sits inside its own interval.
    for i, (_, row) in enumerate(df.iterrows()):
        successes = round(row.mortality_haiku_filled / 100 * row.n_characters)
        low, high = FranchiseMortality.wilson(successes, int(row.n_characters))
        ax.plot([low, high], [i, i], color=BASELINE, linewidth=1.2)
    ax.scatter(df.mortality_haiku_filled, y, s=28, color=SERIES[0], zorder=3,
               label="estimate")

    ax.set_yticks(y, df.franchise)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of on-screen characters who die (%)")
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    style.strip_spines(ax, keep=("bottom",))
    ax.legend(loc="lower right")

    ax.set_title("How deadly is each franchise?")
    style.caption(fig,
        "Dot = point estimate, using the Haiku-resolved labels where a status could not be parsed. Grey bar = 95% "
        "Wilson interval on that estimate. Blue band = how far it would move if every remaining unlabelled character "
        "turned out to be dead. Franchises with complete wiki coverage have no blue band.")
    fig.savefig(FIG_DIR / "q3_franchise_mortality.png")
    plt.close(fig)


def fig_runspan() -> None:
    """Franchise properties against mortality: the null result for Q3."""
    df = pd.read_csv(CLEAN_DIR / "q3_franchise_mortality.csv")
    df = df[(df.n_characters >= 30) & df.run_span.notna()]

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    sizes = 18 + 120 * (df.n_characters / df.n_characters.max()) ** 0.5
    ax.scatter(df.run_span, df.mortality_adjusted, s=sizes,
               color=SERIES[0], alpha=0.75, edgecolor=SURFACE, linewidth=1.5)

    fit = np.polyfit(df.run_span, df.mortality_adjusted, 1)
    xs = np.linspace(df.run_span.min(), df.run_span.max(), 50)
    ax.plot(xs, np.polyval(fit, xs), color=SERIES[1], linewidth=2)

    # Label only the extremes, never every point.
    for _, row in pd.concat([
        df.nlargest(3, "mortality_adjusted"), df.nsmallest(3, "mortality_adjusted"),
        df.nlargest(2, "run_span"),
    ]).drop_duplicates("franchise").iterrows():
        ax.annotate(row.franchise, (row.run_span, row.mortality_adjusted),
                    textcoords="offset points", xytext=(7, 4),
                    fontsize=7.5, color=INK_SECONDARY)

    ax.set_xlabel("Years between first and last release")
    ax.set_ylabel("Mortality (%)")
    style.strip_spines(ax)
    glm = pd.read_csv(CLEAN_DIR / "q3_glm_summary.csv")
    run_span_p = glm.loc[glm.term == "run_span", "p"].iloc[0]

    ax.set_title("How long a franchise runs says nothing about how deadly it is")
    style.caption(fig,
        f"Marker size is cast size. Run span is the strongest of the five franchise properties we tested and it still "
        f"misses significance (p={run_span_p:.2f}); the whole model reaches R2={glm.r_squared[0]:.2f} on "
        f"{int(glm.n_franchises[0])} franchises, with an adjusted R2 of {glm.r_squared_adj[0]:.2f} -- worse than "
        "predicting every franchise at the overall mean.")
    fig.savefig(FIG_DIR / "q3_run_span.png")
    plt.close(fig)


def fig_text_terms(n: int = 12) -> None:
    """Which description words carry the death signal."""
    coefficients = pd.read_csv(CLEAN_DIR / "q2_text_coefficients.csv")
    toward_death = coefficients.head(n)
    toward_life = coefficients.tail(n)
    shown = pd.concat([toward_life, toward_death]).sort_values("coefficient")

    fig, ax = plt.subplots(figsize=(7.0, 5.6))
    y = np.arange(len(shown))
    # Red for the death direction, blue for survival, matching the demo page.
    colors = [SERIES[7] if c > 0 else SERIES[0] for c in shown.coefficient]
    ax.barh(y, shown.coefficient, height=0.68, color=colors)

    ax.set_yticks(y, shown.term)
    ax.axvline(0, color=INK_SECONDARY, linewidth=1.0)
    ax.set_xlabel("Logistic regression coefficient (TF-IDF, death-stripped text)")
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    style.strip_spines(ax, keep=("bottom",))

    ax.set_title("Narrative role is the strongest text signal for death")
    style.caption(fig,
        f"Top {n} terms in each direction (trainval only). Portable terms like 'antagonist', 'command', 'background' "
        "describe narrative position and transfer across franchises. Less portable terms ('hydra', 'island') are "
        "franchise fingerprints; survival-side terms ('relationships', 'category') are wiki section headings.")
    fig.savefig(FIG_DIR / "q2_text_terms.png")
    plt.close(fig)


def fig_text_transfer() -> None:
    """How much does each model lose when franchises are held out?"""
    report = pd.read_csv(CLEAN_DIR / "q1_model_comparison.csv")

    feats = ["text", "character+text"]
    models = ["logistic", "forest"]
    labels = {"text": "Text only", "character+text": "Character + text"}
    model_labels = {"logistic": "Logistic regression", "forest": "Random forest"}

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    x = np.arange(len(feats))
    width = 0.32

    for i, model in enumerate(models):
        random_scores, grouped_scores = [], []
        for feat in feats:
            r = report[(report.features == feat) & (report.model == model)]
            random_scores.append(r[r.regime == "random"].pr_auc.iloc[0])
            grouped_scores.append(r[r.regime == "grouped"].pr_auc.iloc[0])

        offset = (i - 0.5) * width
        color = SERIES[i]

        # Grouped score as solid bar, random-to-grouped drop as lighter extension
        ax.bar(x + offset, grouped_scores, width * 0.9, color=color,
               label=model_labels[model])
        ax.bar(x + offset, [r - g for r, g in zip(random_scores, grouped_scores)],
               width * 0.9, bottom=grouped_scores, color=color, alpha=0.25)

        for j, (r, g) in enumerate(zip(random_scores, grouped_scores)):
            # Grouped score inside the solid bar
            ax.text(x[j] + offset, g - 0.015, f"{g:.3f}", ha="center", va="top",
                    fontsize=8, color=SURFACE, fontweight="bold")
            # Drop label inside the faded extension
            mid = g + (r - g) / 2
            ax.text(x[j] + offset, mid, f"−{r - g:.3f}", ha="center", va="center",
                    fontsize=7.5, color=INK_SECONDARY)

    ax.set_xticks(x, [labels[f] for f in feats])
    ax.set_ylabel("PR-AUC")
    ax.set_ylim(0, 0.78)
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    style.strip_spines(ax, keep=("left",))
    ax.legend(loc="upper left")

    ax.set_title("Logistic regression transfers better than random forest on text")
    style.caption(fig,
        "Solid bar = grouped CV (franchise held out). Faded extension = drop from random CV. "
        "The forest drops more because it memorises franchise-specific term combinations.")
    fig.savefig(FIG_DIR / "q2_text_transfer.png")
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
    ax.set_title("Three in five scraped pages are not an on-screen character")
    style.caption(fig,
        f"Estimated from {comp['n_annotated'].iloc[0]} annotated pages, reweighted to the "
        f"{comp['n_corpus_represented'].iloc[0]:,} they represent. "
        "Analysis is restricted to the leftmost group.")
    fig.savefig(FIG_DIR / "corpus_composition.png")
    plt.close(fig)


def fig_community_fate() -> None:
    """Do characters in the same Louvain community share fates?"""
    fate = pd.read_csv(CLEAN_DIR / "q1_community_fate.csv")
    fate = fate.sort_values("z")

    fig, axes = plt.subplots(
        1, 2, figsize=(10.5, 6.2), gridspec_kw={"width_ratios": [1.35, 1]}
    )

    ax = axes[0]
    y = np.arange(len(fate))
    significant = fate["p_permutation"] < 0.05
    colors = [SERIES[0] if s else "#c9d6e4" for s in significant]
    ax.barh(y, fate["z"], height=0.7, color=colors)

    # +/-1.96 is the two-sided 5% band of the permutation null, which is
    # standardised to z by construction.
    for bound in (-1.96, 1.96):
        ax.axvline(bound, color=MUTED, linewidth=1)
    ax.axvline(0, color=INK, linewidth=0.9)
    ax.text(1.96, len(fate) - 0.2, "  p = 0.05", color=MUTED, fontsize=7.5,
            va="center")

    ax.set_yticks(y, fate["franchise"], fontsize=7)
    ax.set_xlabel("Fate concentration, standardised against a label-shuffling null")
    ax.set_title("13 of 33 franchises exceed the null", fontsize=9,
                 color=INK_SECONDARY, pad=6)
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    style.strip_spines(ax, keep=("bottom",))

    # Right panel: the partition has to be worth interpreting before its
    # relationship with fate is worth reading, so modularity goes on one axis.
    ax = axes[1]
    ax.scatter(fate["modularity"], fate["d_observed"],
               s=18 + 42 * fate["n_characters"] / fate["n_characters"].max(),
               color=[SERIES[0] if s else "#c9d6e4" for s in significant],
               edgecolor=SURFACE, linewidth=0.6, zorder=3)
    ax.axhline(0, color=INK, linewidth=0.9)
    ax.axvline(0.3, color=MUTED, linewidth=1)
    ax.text(0.303, ax.get_ylim()[1], " Q = 0.3", color=MUTED, fontsize=7.5,
            va="top")

    # Labelling only the extremes keeps the panel readable; the left panel
    # already names every franchise.
    notable = fate[(fate["d_observed"] > 0.05) | (fate["d_observed"] < -0.02)]
    for _, row in notable.iterrows():
        ax.annotate(row["franchise"], (row["modularity"], row["d_observed"]),
                    fontsize=6.5, color=INK_SECONDARY,
                    xytext=(5, 3), textcoords="offset points")

    ax.set_xlabel("Modularity Q of the franchise partition")
    ax.set_ylabel("D, excess same-fate rate within communities")
    ax.set_title("Marker area is cast size", fontsize=9,
                 color=INK_SECONDARY, pad=6)
    style.strip_spines(ax)

    fig.tight_layout()
    fig.suptitle("Characters in the same narrative community often share a fate",
                 x=0.0, y=1.04, ha="left", fontsize=12, fontweight="bold")
    style.caption(fig,
        "Louvain partitions of each franchise's co-mention graph, tested against 2,000 within-franchise label shuffles. "
        "Filled markers reject the null at p < 0.05. Modularity does not predict the effect: Dune and Lord of the Rings "
        "partition equally well (Q = 0.64, 0.62) and sit at opposite ends of D.")
    fig.savefig(FIG_DIR / "q1_community_fate.png")
    plt.close(fig)


def fig_cost_curves() -> None:
    """Where the cost-minimising threshold sits, and what a global one costs."""
    curves = pd.read_csv(CLEAN_DIR / "q1_cost_curves.csv")
    optima = pd.read_csv(CLEAN_DIR / "q1_cost_optima.csv")
    franchises = pd.read_csv(CLEAN_DIR / "q1_cost_by_franchise.csv")

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.3))

    ax = axes[0]
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
    ax.set_title("The best threshold moves with the cost of a missed death",
                 fontsize=9, color=INK_SECONDARY, pad=6)
    ax.legend(loc="upper left", ncol=2, bbox_to_anchor=(0.02, 0.98))
    ax.set_ylim(0, None)
    style.strip_spines(ax)

    ax = axes[1]
    ordered = franchises.sort_values("best_threshold")
    y = np.arange(len(ordered))
    watched = ordered["franchise"].isin(["Spartacus", "Grey's Anatomy"])
    ax.scatter(ordered["best_threshold"], y,
               color=[SERIES[1] if w else "#c9d6e4" for w in watched],
               s=[34 if w else 16 for w in watched], zorder=3)
    ax.axvline(0.5, color=MUTED, linewidth=1)

    for _, (position, row) in enumerate(zip(y, ordered.itertuples())):
        if row.franchise in ("Spartacus", "Grey's Anatomy"):
            ax.annotate(row.franchise, (row.best_threshold, position),
                        fontsize=7.5, color=INK, fontweight="bold",
                        xytext=(6, -1), textcoords="offset points")

    ax.set_yticks([])
    ax.set_xlabel("Threshold each franchise would choose for itself (w1 = 3)")
    ax.set_title("Franchise optima span 0.05 to 0.72", fontsize=9,
                 color=INK_SECONDARY, pad=6)
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    style.strip_spines(ax, keep=("bottom",))

    fig.tight_layout()
    fig.suptitle("The relative cost of the two errors decides the threshold",
                 x=0.0, y=1.06, ha="left", fontsize=12, fontweight="bold")
    style.caption(fig,
        "Left: weighted loss w1*FN + FP swept over thresholds, on out-of-fold predictions under grouped folds. "
        "Markers are the minima, moving from 0.81 at w1 = 0.5 down to 0.10 at w1 = 10. "
        "Right: the threshold each franchise would pick for itself at w1 = 3. These track base mortality closely, "
        "running from 0.05 for Spartacus at 74% mortality up to 0.72 for Grey's Anatomy at 9%. At the global 0.50 "
        "threshold Spartacus carries four times the loss it would under its own.")
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
    fig.suptitle("Fold spread swamps most of the gaps between feature sets",
                 x=0.0, y=1.05, ha="left", fontsize=12, fontweight="bold")
    style.caption(fig,
        "Small markers are the five individual folds, large markers the mean. Both panels share an axis, so the "
        "leftward shift between them is the cost of holding out a whole franchise. Under that rule each fold is a "
        "different set of franchises, and the spread within a feature set is wider than the distance between most of them.")
    fig.savefig(FIG_DIR / "q1_fold_spread.png")
    plt.close(fig)


def fig_cluster_diagnostics() -> None:
    """How k was chosen, and what the resulting partition looks like."""
    from scipy.cluster.hierarchy import dendrogram, linkage

    diag = pd.read_csv(CLEAN_DIR / "q3_kmeans_diagnostics.csv")
    clusters = pd.read_csv(CLEAN_DIR / "q3_franchise_clusters.csv")
    X = np.loadtxt(CLEAN_DIR / "q3_cluster_matrix.csv", delimiter=",")

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6),
                             gridspec_kw={"width_ratios": [1, 1.5]})

    ax = axes[0]
    ax.plot(diag["k"], diag["mean_distance_to_centroid"], color=SERIES[0],
            marker="o", markersize=4, linewidth=1.6, label="mean distance to centroid")
    ax.set_xlabel("k")
    ax.set_ylabel("Mean distance to centroid")
    style.strip_spines(ax)

    twin = ax.twinx()
    twin.plot(diag["k"], diag["silhouette"], color=SERIES[1], marker="s",
              markersize=4, linewidth=1.6, label="silhouette")
    twin.set_ylabel("Silhouette", color=SERIES[1])
    twin.tick_params(axis="y", colors=SERIES[1])
    twin.grid(False)
    for side in ("top", "left", "bottom"):
        twin.spines[side].set_visible(False)

    ax.set_title("The elbow curve bends nowhere; the silhouette picks k = 2",
                 fontsize=9, color=INK_SECONDARY, pad=6)

    ax = axes[1]
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

    fig.tight_layout()
    fig.suptitle("Thirty-eight franchises do not fall into natural groups",
                 x=0.0, y=1.06, ha="left", fontsize=12, fontweight="bold")
    style.caption(fig,
        "Left: neither selection rule finds a clear k. The distance curve falls smoothly with no elbow, and the best "
        "silhouette (0.43 at k = 2) separates six large film franchises from everything else. "
        "Right: the dendrogram shows the same, with one small branch splitting off early and the rest merging at similar heights.")
    fig.savefig(FIG_DIR / "q3_cluster_diagnostics.png")
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
    ax.set_xlabel("Cohen's kappa between two independent coders")
    ax.legend(loc="lower right")
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    style.strip_spines(ax, keep=("bottom",))
    ax.invert_yaxis()

    fig.tight_layout()
    fig.suptitle("Writing the coding scheme down moved the death label two bands",
                 x=0.0, y=1.04, ha="left", fontsize=12, fontweight="bold")
    style.caption(fig,
        "Same 600 characters, same model, four independent coding passes. v1 was reconstructed after the fact; v2 "
        "adjudicates the four cases the v1 disagreements exposed. Vertical rules are the Landis and Koch bands.")
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
        ("q2_text_terms", fig_text_terms),
        ("q2_text_transfer", fig_text_transfer),
        ("q3_franchise_mortality", fig_franchise_mortality),
        ("q3_run_span", fig_runspan),
        ("q3_cluster_diagnostics", fig_cluster_diagnostics),
    ]:
        fn()
        print(f"  wrote {name}.png")

    print(f"\n{len(list(FIG_DIR.glob('*.png')))} figures in {FIG_DIR}")


if __name__ == "__main__":
    from src import setup_run_log
    setup_run_log(__spec__)
    main()
