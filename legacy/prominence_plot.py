"""Mortality rate vs franchise-relative prominence.

Ranks each character's infobox_field_count within its own franchise (percentile),
bins into deciles, and plots the death rate per decile. This is the clean,
non-leaky version of the Check 3 finding (corr +0.262).
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path

DATA = Path(__file__).parent / "data" / "clean"
df = pd.read_csv(DATA / "characters_raw.csv", low_memory=False)

df["prom_pct"] = df.groupby("franchise")["infobox_field_count"].rank(pct=True)
df["decile"] = (df["prom_pct"] * 10).clip(upper=9.999).astype(int)  # 0..9

grp = df.groupby("decile")["is_dead"].agg(["mean", "count"])
overall = df["is_dead"].mean()
labels = [f"{i*10}-{i*10+10}%" for i in range(10)]

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(range(10), grp["mean"], color="#4878a8", width=0.8)
ax.axhline(overall, color="#888", linewidth=1, linestyle="--",
           label=f"Overall ({overall:.0%})")
ax.set_xticks(range(10))
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
ax.set_xlabel("Infobox field count, percentile within franchise (low → high prominence)")
ax.set_ylabel("Mortality rate")
ax.set_title("Mortality Rate by Within-Franchise Prominence")
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
fig.savefig(DATA / "prominence_decile_mortality.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved prominence_decile_mortality.png")
print(grp.assign(rate=lambda d: (d["mean"]*100).round(1)).to_string())
