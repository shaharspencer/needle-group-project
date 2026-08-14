"""Careful re-check of the text 'violence' features for label leakage.

Recomputes violence density straight from raw page_text under three settings:
  A. original lexicon, full text            (reproduces characters_full.csv)
  B. lexicon with explicit death words removed
  C. original lexicon, but death-describing sentences dropped first
and compares corr / mutual information with is_dead. A big drop from A->B/C
means the signal was mostly the page announcing the character's own death.
"""
import json, re
import numpy as np
from pathlib import Path
from sklearn.feature_selection import mutual_info_classif

RAW = Path(__file__).parent / "data" / "raw"

VIOLENCE = {"kill","killed","murder","murdered","death","dead","die","died",
    "dies","dying","slay","slain","execute","executed","assassinate","assassinated",
    "sacrifice","sacrificed","massacre","slaughter","strangle","strangled","poison",
    "poisoned","stab","stabbed","shot","shoot","behead","beheaded","hang","hanged",
    "drown","drowned","explode","explosion","bomb","destroy","destroyed"}

# words that describe a death event (the leakage source)
DEATH = {"kill","killed","murder","murdered","death","dead","die","died","dies",
    "dying","slay","slain","execute","executed","assassinate","assassinated",
    "behead","beheaded","strangle","strangled","drown","drowned"}

VIOLENCE_CLEAN = VIOLENCE - DEATH
WORD = re.compile(r"\b[a-z]+\b")

def density(words, lex):
    n = max(len(words), 1)
    return sum(1 for w in words if w in lex) / n

ys, A, B, C = [], [], [], []
for path in RAW.glob("*_characters.jsonl"):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("is_dead") is None or r.get("is_redirect"):
                continue
            text = (r.get("page_text") or "")[:5000]
            if not text:
                continue
            words = WORD.findall(text.lower())
            # variant C: drop sentences that mention a death event
            sents = re.split(r"[.!?]+", text)
            kept = " ".join(s for s in sents
                            if not (set(WORD.findall(s.lower())) & DEATH))
            words_c = WORD.findall(kept.lower())

            ys.append(int(r["is_dead"]))
            A.append(density(words, VIOLENCE))
            B.append(density(words, VIOLENCE_CLEAN))
            C.append(density(words_c, VIOLENCE))

y = np.array(ys)
print(f"n={len(y):,}  dead={y.mean():.3f}\n")
print(f"{'variant':45s} {'corr':>8} {'MI':>8}")
for name, arr in [("A. original lexicon, full text", A),
                  ("B. death words removed from lexicon", B),
                  ("C. death sentences dropped first", C)]:
    x = np.array(arr)
    corr = np.corrcoef(x, y)[0, 1]
    mi = mutual_info_classif(x.reshape(-1, 1), y,
                             discrete_features=False, random_state=0)[0]
    print(f"{name:45s} {corr:+8.3f} {mi:8.4f}")
