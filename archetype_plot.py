"""Mortality by archetype and gender (grouped bar).

Shows two clean findings at once: roles order by deadliness (Military/Royalty/
Criminal high, Worker/Entertainer low), and within every role males die more.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path

DATA = Path(__file__).parent / "data" / "clean"
df = pd.read_csv(DATA / "characters_raw.csv", low_memory=False)
g = df[df.gender.isin(["Male", "Female"])]

piv = g.pivot_table(index="archetype", columns="gender",
                    values="is_dead", aggfunc="mean")
piv = piv[piv.index != "Unknown"].sort_values("Male")

y = np.arange(len(piv))
h = 0.38
fig, ax = plt.subplots(figsize=(8, 7))
ax.barh(y + h/2, piv["Male"],   height=h, color="#4878a8", label="Male")
ax.barh(y - h/2, piv["Female"], height=h, color="#e08214", label="Female")
ax.axvline(df["is_dead"].mean(), color="#888", linewidth=1, linestyle="--",
           label=f"Overall ({df['is_dead'].mean():.0%})")
ax.set_yticks(y)
ax.set_yticklabels(piv.index, fontsize=9)
ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
ax.set_xlabel("Mortality rate")
ax.set_title("Mortality Rate by Archetype and Gender")
ax.legend(fontsize=9, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
fig.savefig(DATA / "archetype_gender_mortality.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved archetype_gender_mortality.png")
