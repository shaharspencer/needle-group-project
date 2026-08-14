"""Alternative Q2 sanity check: one dot per franchise on a year axis.
   Directly shows whether any release_year value is out of range."""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA = Path(__file__).parent / "data" / "clean"
df = pd.read_csv(DATA / "characters_raw.csv", low_memory=False)

yr = df.groupby("franchise")["franchise_release_year"].first().sort_values()

fig, ax = plt.subplots(figsize=(7, 9))
ax.scatter(yr.values, range(len(yr)), color="#4878a8", s=40, zorder=3)
ax.set_yticks(range(len(yr)))
ax.set_yticklabels(yr.index, fontsize=8)
ax.set_xlabel("Franchise release year")
ax.set_title("Release Year per Franchise (one point each)")
ax.grid(axis="x", linewidth=0.5, alpha=0.4)
ax.spines[["top", "right"]].set_visible(False)
ax.margins(y=0.01)
plt.tight_layout()
fig.savefig(DATA / "q2_release_year_dotplot.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved q2_release_year_dotplot.png")
