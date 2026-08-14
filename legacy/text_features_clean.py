"""Trustworthy re-scan of ALL lexicon-based text features.

For every keyword density (violence/conflict/leadership/family/hero/villain/
power/vulnerability) computes the signal vs is_dead in two settings:
  orig  - original lexicon, full page text  (= characters_full.csv)
  clean - death-event words stripped from the lexicon AND death-describing
          sentences dropped before counting

A large orig->clean drop means the feature was riding on the page announcing
the character's own death rather than on genuine narrative content.
"""
import json, re
import numpy as np
from pathlib import Path
from sklearn.feature_selection import mutual_info_classif

RAW = Path(__file__).parent / "data" / "raw"
WORD = re.compile(r"\b[a-z]+\b")

LEX = {
"violence": {"kill","killed","murder","murdered","death","dead","die","died","dies",
    "dying","slay","slain","execute","executed","assassinate","assassinated","sacrifice",
    "sacrificed","massacre","slaughter","strangle","strangled","poison","poisoned","stab",
    "stabbed","shot","shoot","behead","beheaded","hang","hanged","drown","drowned",
    "explode","explosion","bomb","destroy","destroyed"},
"conflict": {"war","battle","fight","fought","attack","attacked","combat","siege",
    "invasion","raid","ambush","assault","conflict","rebellion","revolt","uprising"},
"leadership": {"king","queen","lord","lady","captain","general","commander","president",
    "leader","chief","ruler","emperor","empress","prince","princess","governor","senator",
    "chancellor","director","boss","head"},
"family": {"father","mother","son","daughter","brother","sister","husband","wife","child",
    "children","parent","sibling","family","uncle","aunt","cousin","grandfather",
    "grandmother","nephew","niece","married","spouse","wedding"},
"hero": {"hero","heroic","heroism","save","saved","saves","saving","rescue","rescued",
    "protect","protected","protector","defend","defended","defender","brave","bravery",
    "courage","courageous","noble","honor","honour","honorable","honourable","loyal",
    "loyalty","justice","righteous","ally","allies","compassion","compassionate","mercy",
    "merciful","sacrifice"},
"villain": {"villain","villainous","evil","cruel","cruelty","ruthless","corrupt",
    "corruption","betray","betrayed","betrayal","traitor","treachery","treacherous",
    "tyrant","tyranny","oppressor","manipulate","manipulated","manipulative","deceive",
    "deceived","deception","scheme","schemer","sinister","malicious","sadistic","merciless",
    "enemy","enemies","criminal","crime","crimes"},
"power": {"power","powerful","force","authority","command","control","influence","dominate",
    "dominated","dominance","reign","rule","ruled","conquer","conquered","empire","throne",
    "weapon","weapons","army","armies","military","fleet"},
"vulnerability": {"captured","imprisoned","tortured","wounded","injured","helpless",
    "vulnerable","trapped","prisoner","hostage","kidnapped","abducted","enslaved","slave",
    "suffering","trauma","traumatic","grief","loss","lost","fear","afraid","desperate",
    "despair","exile","exiled","banished"},
}

# death-event words: removed from every lexicon and used to detect death sentences
DEATH = {"kill","killed","murder","murdered","death","dead","die","died","dies","dying",
    "slay","slain","execute","executed","assassinate","assassinated","behead","beheaded",
    "strangle","strangled","drown","drowned","sacrifice","sacrificed"}
# extra markers (not in lexicons) to better catch death sentences
DEATH_SENT = DEATH | {"deceased","perish","perished","funeral","grave","posthumous",
    "fatally","fatal","mourned","mourning","corpse","buried"}

LEX_CLEAN = {k: (v - DEATH) for k, v in LEX.items()}

def densities(words, lex_set):
    n = max(len(words), 1)
    return {k: sum(1 for w in words if w in lex) / n for k, lex in lex_set.items()}

ys = []
orig = {k: [] for k in LEX}
clean = {k: [] for k in LEX}
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
            sents = re.split(r"[.!?]+", text)
            kept = " ".join(s for s in sents
                            if not (set(WORD.findall(s.lower())) & DEATH_SENT))
            words_c = WORD.findall(kept.lower())

            ys.append(int(r["is_dead"]))
            for k, v in densities(words, LEX).items():
                orig[k].append(v)
            for k, v in densities(words_c, LEX_CLEAN).items():
                clean[k].append(v)

y = np.array(ys)
print(f"n={len(y):,}  dead={y.mean():.3f}\n")

def stats(arr):
    x = np.array(arr)
    corr = np.corrcoef(x, y)[0, 1] if np.std(x) > 0 else float("nan")
    mi = mutual_info_classif(x.reshape(-1, 1), y,
                             discrete_features=False, random_state=0)[0]
    return corr, mi

rows = []
for k in LEX:
    oc, om = stats(orig[k])
    cc, cm = stats(clean[k])
    rows.append((k, om, cm, oc, cc))
rows.sort(key=lambda r: r[2], reverse=True)  # rank by CLEAN MI

print(f"{'feature':16s} {'orig_MI':>8} {'clean_MI':>9} {'orig_r':>8} {'clean_r':>8}  drop")
for k, om, cm, oc, cc in rows:
    drop = f"{(1 - cm/om)*100:4.0f}%" if om > 0 else "  n/a"
    print(f"{k:16s} {om:8.4f} {cm:9.4f} {oc:+8.3f} {cc:+8.3f}  {drop}")
