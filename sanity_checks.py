"""Sanity checks for the milestone:
   Q2 - distribution of characters by franchise release year (look for weird values)
   Q3 - mortality rate ("% dead") per franchise
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path

DATA = Path(__file__).parent / "data" / "clean"
df = pd.read_csv(DATA / "characters_raw.csv", low_memory=False)

print(f"rows={len(df):,}  cols={df.shape[1]}")
print(f"franchises={df['franchise'].nunique()}")
print(f"overall mortality={df['is_dead'].mean():.3f}")

# ── Q2: release-year sanity check ────────────────────────────────────────────
yr = "franchise_release_year"
print("\n=== release_year describe ===")
print(df[yr].describe())
print("\nnulls:", df[yr].isna().sum())
print("\n=== rows per release_year ===")
print(df[yr].value_counts().sort_index())
print("\n=== release_year per franchise (one row each) ===")
print(df.groupby("franchise")[yr].first().sort_values())

# ── Q3: mortality per franchise ──────────────────────────────────────────────
mort = (df.groupby("franchise")
          .agg(n=("is_dead", "count"), dead=("is_dead", "sum")))
mort["pct_dead"] = mort["dead"] / mort["n"]
mort = mort.sort_values("pct_dead", ascending=False)
print("\n=== mortality by franchise ===")
print(mort.to_string(float_format=lambda x: f"{x:.3f}"))
