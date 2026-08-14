"""Feature signal scan against is_dead.

For every non-leaky feature computes:
  - coverage (non-null fraction)
  - mutual information with is_dead (single batched call, NaN-imputed)
  - signed point-biserial correlation (direction: + => associated with death)

Also adds within-franchise rank-normalized versions of the structural count
features so we can compare raw vs franchise-relative signal.

Text/NLP features are scanned too but reported in a SEPARATE block because
they are still a draft and need manual checking.
"""
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from pathlib import Path

DATA = Path(__file__).parent / "data" / "clean"
df = pd.read_csv(DATA / "characters_full.csv", low_memory=False)
y = df["is_dead"].astype(int).values

# ── columns we must never feed the model (encode the label) ──────────────────
LEAKY = {
    "is_dead", "franchise_mortality_rate", "franchise_weight",
    "hg_winner", "hg_rank",                       # winning == survived
    "has_dod", "dod", "has_causeofdeath", "has_death_section", "info_completeness",
    "actual", "pred", "alive", "plod", "isAlive", "DateoFdeath", "Alive",
    "birth", "death",
}

# ── text/NLP draft features (reported separately) ────────────────────────────
TEXT = {
    "word_count", "sentence_count", "paragraph_count", "section_count",
    "avg_sentence_length", "unique_word_ratio", "dialogue_count",
    "relationship_mentions", "named_characters_mentioned",
    "violence_word_count", "conflict_word_count", "leadership_word_count",
    "family_word_count", "hero_word_count", "villain_word_count",
    "power_word_count", "vulnerability_word_count",
    "violence_density", "conflict_density", "leadership_density",
    "family_density", "hero_density", "villain_density", "power_density",
    "vulnerability_density", "dialogue_density", "relationship_density",
    "social_connectedness", "moral_alignment", "power_vulnerability_ratio",
    "has_biography_section", "has_dialogue", "is_described_young",
    "is_described_old", "age_mentioned", "sentiment_score",
    "name_word_count", "has_title_in_name", "name_centrality", "page_mention_rank",
}

# structural counts to also test in franchise-relative form
NORMALIZE = ["page_text_length", "infobox_field_count",
             "fandom_category_count", "appearance_count"]

candidates = [c for c in df.columns if c not in LEAKY]

# add within-franchise percentile-rank versions
for col in NORMALIZE:
    if col in df.columns:
        newc = col + "__frank_pct"
        df[newc] = df.groupby("franchise")[col].rank(pct=True)
        candidates.append(newc)
        TEXT.discard(newc)

# ── build encoded matrix + discrete mask ─────────────────────────────────────
rows = []          # (name, is_text, coverage, corr)
X_cols, disc_mask = [], []
for c in candidates:
    s = df[c]
    cov = s.notna().mean()
    if not pd.api.types.is_numeric_dtype(s):
        codes = pd.factorize(s, use_na_sentinel=True)[0].astype(float)  # NaN -> -1
        codes[codes < 0] = -1
        X_cols.append(codes)
        disc_mask.append(True)
        corr = np.nan
    else:
        x = s.astype(float)
        nun = x.nunique(dropna=True)
        is_disc = nun <= 15            # binary / small-integer => discrete
        fill = x.median() if not is_disc else -1.0
        xf = x.fillna(fill).values
        # signed correlation for direction (skip if constant)
        corr = np.corrcoef(xf, y)[0, 1] if np.std(xf) > 0 else np.nan
        X_cols.append(xf)
        disc_mask.append(is_disc)
    rows.append([c, c in TEXT, cov, corr])

X = np.column_stack(X_cols)
mi = mutual_info_classif(X, y, discrete_features=np.array(disc_mask),
                         random_state=0)

res = pd.DataFrame(rows, columns=["feature", "is_text", "coverage", "corr"])
res["mi"] = mi
res = res.sort_values("mi", ascending=False)

def show(block, title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")
    print(f"{'feature':32s} {'MI':>7} {'corr':>7} {'cov':>6}")
    for _, r in block.iterrows():
        corr = f"{r['corr']:+.3f}" if pd.notna(r["corr"]) else "   cat"
        print(f"{r['feature']:32s} {r['mi']:7.4f} {corr:>7} {r['coverage']:6.2f}")

struct = res[~res["is_text"]]
text = res[res["is_text"]]
show(struct.head(30), "STRUCTURAL / EXTERNAL FEATURES (top 30 by MI)")
show(text.head(20), "TEXT / NLP DRAFT FEATURES (top 20 by MI) -- VERIFY MANUALLY")

print("\nraw vs franchise-normalized (structural counts):")
for col in NORMALIZE:
    r = res[res["feature"] == col]["mi"]
    n = res[res["feature"] == col + "__frank_pct"]["mi"]
    if len(r) and len(n):
        print(f"  {col:24s} raw MI={r.values[0]:.4f}   frank_pct MI={n.values[0]:.4f}")
