import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path(__file__).parent / "data" / "clean"
df = pd.read_csv(DATA / "characters_raw.csv", low_memory=False)
overall = df["is_dead"].mean()
print(f"overall = {overall:.3f}\n")

# ── alignment, controlled within franchise (Evil vs Good gap per franchise) ──
print("=== Evil vs Good mortality WITHIN franchise (n>=30 each) ===")
sub = df[df["affiliation_alignment"].isin(["Good", "Evil"])]
rows = []
for fr, g in sub.groupby("franchise"):
    e = g[g["affiliation_alignment"] == "Evil"]["is_dead"]
    go = g[g["affiliation_alignment"] == "Good"]["is_dead"]
    if len(e) >= 30 and len(go) >= 30:
        rows.append((fr, e.mean(), go.mean(), e.mean()-go.mean(), len(e), len(go)))
al = pd.DataFrame(rows, columns=["franchise","evil","good","gap(e-g)","n_e","n_g"])
al = al.sort_values("gap(e-g)", ascending=False)
print(al.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print(f"evil>good in {(al['gap(e-g)']>0).sum()}/{len(al)}; mean gap {al['gap(e-g)'].mean():+.3f}\n")

# ── archetype mortality (sorted) ─────────────────────────────────────────────
arch = df.groupby("archetype")["is_dead"].agg(["mean","count"]).sort_values("mean")
print("=== archetype (100% coverage) ===")
print(arch.assign(rate=lambda d:(d['mean']*100).round(1))[["rate","count"]].to_string(), "\n")

# ── gender x archetype interaction ───────────────────────────────────────────
print("=== mortality by archetype x gender (Male/Female) ===")
g2 = df[df.gender.isin(["Male","Female"])]
piv = g2.pivot_table(index="archetype", columns="gender", values="is_dead", aggfunc="mean")
cnt = g2.pivot_table(index="archetype", columns="gender", values="is_dead", aggfunc="count")
piv = (piv*100).round(1)
piv["gap_m_minus_f"] = (piv["Male"]-piv["Female"]).round(1)
print(piv.sort_values("Male", ascending=False).to_string(), "\n")

# ── marital-ish coverage check ───────────────────────────────────────────────
for c in ["marvel_marital","got_married","lotr_has_spouse","has_alias","has_dob","has_pronouns","prominence_tier"]:
    if c in df.columns:
        cov = df[c].notna().mean()
        print(f"{c:18s} coverage={cov:.0%}")
