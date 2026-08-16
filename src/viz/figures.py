"""Render the writeup figures.

  python -m src.viz.figures
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.paths import CLEAN_DIR, FIG_DIR
from src.viz import style
from src.viz.style import BASELINE, INK, INK_SECONDARY, MUTED, SERIES, SURFACE


def fig_model_comparison() -> None:
    """Which predicts death better, the character or the franchise?"""
    report = pd.read_csv(CLEAN_DIR / "q1_model_comparison.csv")
    order = ["franchise", "character_no_wiki_meta", "character",
             "network", "character+network", "both", "rich"]
    labels = {
        "franchise": "Franchise only",
        "character_no_wiki_meta": "Character (no wiki metadata)",
        "character": "Character",
        "network": "Network centrality",
        "character+network": "Character + network",
        "both": "Character + franchise",
        "rich": "All features",
    }

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), sharex=True)
    for ax, regime, title in zip(
        axes, ["random", "grouped"],
        ["Held-out character\n(franchise seen in training)",
         "Held-out franchise\n(never seen in training)"],
    ):
        subset = report[(report.regime == regime) & (report.features.isin(order))]
        best = subset.groupby("features")["pr_auc"].max().reindex(order)
        baseline = report[(report.regime == regime)
                          & (report.features == "baseline:majority")]["pr_auc"].iloc[0]

        y = np.arange(len(best))
        # One series, so one colour. The best bar is emphasised, the rest recede.
        colors = [SERIES[0] if v == best.max() else "#b7d3f6" for v in best]
        ax.barh(y, best.values, height=0.62, color=colors)

        ax.axvline(baseline, color=MUTED, linewidth=1)
        ax.text(baseline, len(best) - 0.35, "  baseline", color=MUTED,
                fontsize=7.5, va="center")

        for i, v in enumerate(best.values):
            ax.text(v + 0.008, i, f"{v:.3f}", va="center", fontsize=7.5,
                    color=INK_SECONDARY)

        ax.set_yticks(y, [labels[f] for f in best.index])
        ax.set_xlim(0.30, 0.92)
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
        "Left: franchise identity is available to the model. Right: whole franchises are held out, so it is not. "
        "Category features add a lot on the left and almost nothing on the right; network centrality adds signal in both.")
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
    df = df[df.n_characters >= 30].sort_values("mortality_adjusted")

    fig, ax = plt.subplots(figsize=(7.6, 7.6))
    y = np.arange(len(df))

    # Wide light band: how far the estimate could move if every unlabelled
    # character turned out to be dead.
    ax.barh(y, df.mortality_upper - df.mortality, left=df.mortality,
            height=0.5, color="#cde2fb", label="range if unlabelled characters died")
    # Wilson interval on the adjusted point estimate.
    for i, (_, row) in enumerate(df.iterrows()):
        ax.plot([row.ci_low, row.ci_high], [i, i], color=BASELINE, linewidth=1.2)
    ax.scatter(df.mortality_adjusted, y, s=28, color=SERIES[0], zorder=3,
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
        "Dot = point estimate, grey bar = 95% Wilson interval, blue band = range if every unlabelled character died. "
        "Franchises with complete wiki coverage have no blue band.")
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


def main() -> None:
    style.apply()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    for name, fn in [
        ("corpus_composition", fig_corpus),
        ("q1_model_comparison", fig_model_comparison),
        ("q1_gender_forest", fig_gender_forest),
        ("q1_feature_importance", fig_importance),
        ("q2_text_terms", fig_text_terms),
        ("q2_text_transfer", fig_text_transfer),
        ("q3_franchise_mortality", fig_franchise_mortality),
        ("q3_run_span", fig_runspan),
    ]:
        fn()
        print(f"  wrote {name}.png")

    print(f"\n{len(list(FIG_DIR.glob('*.png')))} figures in {FIG_DIR}")


if __name__ == "__main__":
    from src import setup_run_log
    setup_run_log(__spec__)
    main()
