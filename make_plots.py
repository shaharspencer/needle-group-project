import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from pathlib import Path

DATA = Path(__file__).parent / "data" / "clean"
OUT = DATA

df = pd.read_csv(DATA / "characters_cleaned.csv", low_memory=False)

# ── Plot 1: mortality rate by franchise (bar chart) ──────────────────────────
mort = (
    df.groupby("franchise")
    .agg(total=("is_dead", "count"), dead=("is_dead", "sum"))
    .reset_index()
)
mort["rate"] = mort["dead"] / mort["total"]
mort = mort[mort["total"] >= 50].sort_values("rate")

colors = ["#c0392b" if r >= 0.5 else "#2980b9" for r in mort["rate"]]

fig, ax = plt.subplots(figsize=(8, 9))
bars = ax.barh(mort["franchise"], mort["rate"], color=colors, height=0.7)
ax.axvline(mort["rate"].mean(), color="#555", linewidth=1, linestyle="--", label=f"Overall mean ({mort['rate'].mean():.0%})")
ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
ax.set_xlabel("Character mortality rate")
ax.set_title("Character Mortality Rate by Franchise\n(franchises with ≥ 50 characters)", fontsize=13, pad=10)
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlim(0, 1.05)

for bar, val in zip(bars, mort["rate"]):
    ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.0%}", va="center", fontsize=8)

plt.tight_layout()
fig.savefig(OUT / "mortality_by_franchise.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved mortality_by_franchise.png")

# ── Plot 2: infobox field count distribution, dead vs alive (box plot) ───────
alive = df.loc[df["is_dead"] == 0, "infobox_field_count"].clip(upper=40)
dead  = df.loc[df["is_dead"] == 1, "infobox_field_count"].clip(upper=40)

fig, ax = plt.subplots(figsize=(6, 5))
bp = ax.boxplot(
    [alive, dead],
    labels=["Alive", "Dead"],
    patch_artist=True,
    widths=0.5,
    medianprops=dict(color="black", linewidth=2),
    flierprops=dict(marker="o", markersize=2, alpha=0.3, linestyle="none"),
)
bp["boxes"][0].set_facecolor("#2980b9")
bp["boxes"][1].set_facecolor("#c0392b")
for patch in bp["boxes"]:
    patch.set_alpha(0.7)

medians = [alive.median(), dead.median()]
for i, med in enumerate(medians, start=1):
    ax.text(i, med + 0.4, f"median = {med:.0f}", ha="center", fontsize=9, color="black")

ax.set_ylabel("Infobox field count (capped at 40)")
ax.set_title("Wiki Documentation Depth\nfor Alive vs Dead Characters", fontsize=13, pad=10)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
fig.savefig(OUT / "infobox_prominence_vs_death.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved infobox_prominence_vs_death.png")
