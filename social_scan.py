"""Mortality by gender and other social attributes, raw and franchise-controlled."""
import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path(__file__).parent / "data" / "clean"
df = pd.read_csv(DATA / "characters_raw.csv", low_memory=False)
overall = df["is_dead"].mean()
print(f"overall mortality = {overall:.3f}  n={len(df):,}\n")

def by(col, mincov=0):
    g = df.groupby(col)["is_dead"].agg(["mean", "count"])
    g = g[g["count"] >= mincov].sort_values("mean", ascending=False)
    print(f"=== mortality by {col} (coverage {df[col].notna().mean():.0%}) ===")
    print(g.assign(rate=lambda d: (d["mean"]*100).round(1))[["rate", "count"]].to_string())
    print()

for c in ["gender", "affiliation_alignment", "is_human", "archetype", "has_family"]:
    if c in df.columns:
        by(c)

# ── gender gap controlling for franchise ─────────────────────────────────────
print("=== male vs female mortality WITHIN each franchise ===")
sub = df[df["gender"].isin(["Male", "Female"])]
rows = []
for fr, g in sub.groupby("franchise"):
    m = g[g["gender"] == "Male"]["is_dead"]
    f = g[g["gender"] == "Female"]["is_dead"]
    if len(m) >= 30 and len(f) >= 30:
        rows.append((fr, m.mean(), f.mean(), m.mean()-f.mean(), len(m), len(f)))
gap = pd.DataFrame(rows, columns=["franchise","male","female","gap(m-f)","n_m","n_f"])
gap = gap.sort_values("gap(m-f)", ascending=False)
print(gap.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print(f"\nfranchises where male > female mortality: {(gap['gap(m-f)']>0).sum()} / {len(gap)}")
print(f"mean within-franchise gap (male - female): {gap['gap(m-f)'].mean():+.3f}")
print(f"raw overall: male={sub[sub.gender=='Male']['is_dead'].mean():.3f}  "
      f"female={sub[sub.gender=='Female']['is_dead'].mean():.3f}")
