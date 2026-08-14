"""Generates the two milestone sanity-check figures:
   q2_release_year.png            - characters per franchise release year
   q3_mortality_by_franchise.png  - % dead per franchise
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path

DATA = Path(__file__).parent / "data" / "clean"
df = pd.read_csv(DATA / "characters_raw.csv", low_memory=False)

# ── Q2: characters per release year ──────────────────────────────────────────
counts = df["franchise_release_year"].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(counts.index, counts.values, width=1.4, color="#4878a8")
ax.set_yscale("log")
ax.set_xlabel("Franchise release year")
ax.set_ylabel("Number of characters (log scale)")
ax.set_title("Characters by Franchise Release Year")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
fig.savefig(DATA / "q2_release_year.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved q2_release_year.png")

# ── Q3: mortality rate by franchise ──────────────────────────────────────────
mort = (df.groupby("franchise")
          .agg(n=("is_dead", "count"), dead=("is_dead", "sum")))
mort["rate"] = mort["dead"] / mort["n"]
mort = mort.sort_values("rate")
overall = df["is_dead"].mean()

fig, ax = plt.subplots(figsize=(7, 8))
ax.barh(mort.index, mort["rate"], color="#4878a8", height=0.7)
ax.axvline(overall, color="#888", linewidth=1, linestyle="--",
           label=f"Overall ({overall:.0%})")
ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
ax.set_xlabel("Mortality rate (% of characters dead)")
ax.set_title("Mortality Rate by Franchise")
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlim(0, 1)
plt.tight_layout()
fig.savefig(DATA / "q3_mortality_by_franchise.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved q3_mortality_by_franchise.png")
