"""Within-franchise gender mortality gap (dumbbell plot).

For each franchise with enough male and female characters, plots female and
male mortality as two connected dots. Males sit to the right almost everywhere,
showing the gap is consistent across universes, not a franchise artifact.
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path

DATA = Path(__file__).parent / "data" / "clean"
df = pd.read_csv(DATA / "characters_raw.csv", low_memory=False)
sub = df[df["gender"].isin(["Male", "Female"])]

rows = []
for fr, g in sub.groupby("franchise"):
    m = g[g["gender"] == "Male"]["is_dead"]
    f = g[g["gender"] == "Female"]["is_dead"]
    if len(m) >= 30 and len(f) >= 30:
        rows.append((fr, m.mean(), f.mean()))
gap = pd.DataFrame(rows, columns=["franchise", "male", "female"])
gap = gap.sort_values("male").reset_index(drop=True)

fig, ax = plt.subplots(figsize=(8, 9))
y = range(len(gap))
ax.hlines(y, gap["female"], gap["male"], color="#cccccc", linewidth=2, zorder=1)
ax.scatter(gap["female"], y, color="#e08214", s=45, zorder=2, label="Female")
ax.scatter(gap["male"], y, color="#4878a8", s=45, zorder=2, label="Male")
ax.set_yticks(y)
ax.set_yticklabels(gap["franchise"], fontsize=8)
ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
ax.set_xlabel("Mortality rate")
ax.set_title("Male vs Female Mortality, Within Each Franchise")
ax.legend(fontsize=9, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
ax.margins(y=0.01)
plt.tight_layout()
fig.savefig(DATA / "gender_gap_by_franchise.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved gender_gap_by_franchise.png")
