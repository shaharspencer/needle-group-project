"""
add_nlp_features.py
-------------------
Optional second pipeline: adds NLP text-analysis features on top of
characters_raw.csv produced by build_pipeline.py.

Inputs  : data/clean/characters_raw.csv
          data/clean/characters_keys.csv   (name/url index)
          data/raw/*.jsonl                 (for page text)

Output  : data/clean/characters_full.csv

Features added:
  name_word_count, has_title_in_name
  name_centrality, page_mention_rank    (co-mention network within franchise)
  sentiment_score                        (TextBlob polarity of page text)
  violence/conflict/leadership/family/hero/villain/power/vulnerability _word_count + _density
  dialogue_count, relationship_mentions, named_characters_mentioned
  social_connectedness, moral_alignment, power_vulnerability_ratio
  word_count, sentence_count, paragraph_count, section_count
  avg_sentence_length, unique_word_ratio
  has_biography_section, has_dialogue
  is_described_young, is_described_old, age_mentioned

Runtime : ~15 min
Usage   : python add_nlp_features.py
"""

import json
import re
from pathlib import Path

import pandas as pd

try:
    from textblob import TextBlob
    _HAS_TEXTBLOB = True
except ImportError:
    _HAS_TEXTBLOB = False
    print("[warn] textblob not installed. Run: pip install textblob")

HERE      = Path(__file__).parent
RAW_DIR   = HERE / "data" / "raw"
CLEAN_DIR = HERE / "data" / "clean"

# ── Keyword sets (from clean_and_features.py) ─────────────────────────────────

_VIOLENCE_WORDS = frozenset(["kill","killed","murder","murdered","death","dead","die","died",
    "dies","dying","slay","slain","execute","executed","assassinate","assassinated",
    "sacrifice","sacrificed","massacre","slaughter","strangle","strangled","poison",
    "poisoned","stab","stabbed","shot","shoot","behead","beheaded","hang","hanged",
    "drown","drowned","explode","explosion","bomb","destroy","destroyed"])
_CONFLICT_WORDS = frozenset(["war","battle","fight","fought","attack","attacked","combat",
    "siege","invasion","raid","ambush","assault","conflict","rebellion","revolt","uprising"])
_LEADERSHIP_WORDS = frozenset(["king","queen","lord","lady","captain","general","commander",
    "president","leader","chief","ruler","emperor","empress","prince","princess",
    "governor","senator","chancellor","director","boss","head"])
_FAMILY_WORDS = frozenset(["father","mother","son","daughter","brother","sister",
    "husband","wife","child","children","parent","sibling","family","uncle","aunt",
    "cousin","grandfather","grandmother","nephew","niece","married","spouse","wedding"])
_HERO_WORDS = frozenset(["hero","heroic","heroism","save","saved","saves","saving",
    "rescue","rescued","protect","protected","protector","defend","defended","defender",
    "brave","bravery","courage","courageous","noble","honor","honour","honorable",
    "honourable","loyal","loyalty","justice","righteous","ally","allies",
    "compassion","compassionate","mercy","merciful","sacrifice"])
_VILLAIN_WORDS = frozenset(["villain","villainous","evil","cruel","cruelty","ruthless",
    "corrupt","corruption","betray","betrayed","betrayal","traitor","treachery",
    "treacherous","tyrant","tyranny","oppressor","manipulate","manipulated",
    "manipulative","deceive","deceived","deception","scheme","schemer","sinister",
    "malicious","sadistic","merciless","enemy","enemies","criminal","crime","crimes"])
_POWER_WORDS = frozenset(["power","powerful","force","authority","command","control",
    "influence","dominate","dominated","dominance","reign","rule","ruled","conquer",
    "conquered","empire","throne","weapon","weapons","army","armies","military","fleet"])
_VULNERABILITY_WORDS = frozenset(["captured","imprisoned","tortured","wounded","injured",
    "helpless","vulnerable","trapped","prisoner","hostage","kidnapped","abducted",
    "enslaved","slave","suffering","trauma","traumatic","grief","loss","lost","fear",
    "afraid","desperate","despair","exile","exiled","banished"])

_TITLE_WORDS_RE = re.compile(
    r"\b(lord|lady|king|queen|prince|princess|captain|general|doctor|dr|"
    r"commander|sergeant|colonel|admiral|emperor|empress|chancellor|"
    r"master|maester|ser|dame|baron|count|duke|duchess)\b", re.I)

_YOUNG_RE = re.compile(r"\b(child|young|teenager|teen|adolescent|juvenile|youth|boy|girl|kid)\b", re.I)
_OLD_RE   = re.compile(r"\b(elderly|old man|old woman|senior|aged|ancient|veteran|grandfather|grandmother)\b", re.I)
_AGE_RE   = re.compile(r"\b(\d{1,3})\s*(?:years?[\s-]old|year[\s-]old)\b", re.I)
_DIALOGUE_RE  = re.compile(r"(?:^|\n)[A-Z][^:]{0,30}:", re.M)
_RELATION_RE  = re.compile(r"\b(?:son|daughter|brother|sister|father|mother|husband|wife|partner|ally|enemy|friend|rival)\b", re.I)
_NAMED_CHAR_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b")

CENTRALITY_CAP = 2000  # skip franchises larger than this


def count_keywords(words_lower: list, keyword_set: frozenset) -> int:
    return sum(1 for w in words_lower if w in keyword_set)


def extract_nlp(text: str, name: str) -> dict:
    if not text:
        return {}
    words_lower = re.findall(r"\b[a-z]+\b", text.lower())
    n = max(len(words_lower), 1)
    sentences   = re.split(r"[.!?]+", text)
    paragraphs  = [p for p in text.split("\n\n") if p.strip()]
    sections    = len(re.findall(r"^={2,}[^=]+=+$", text, re.M))
    avg_sent_len = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
    unique_ratio = len(set(words_lower)) / n

    viol  = count_keywords(words_lower, _VIOLENCE_WORDS)
    conf  = count_keywords(words_lower, _CONFLICT_WORDS)
    lead  = count_keywords(words_lower, _LEADERSHIP_WORDS)
    fam   = count_keywords(words_lower, _FAMILY_WORDS)
    hero  = count_keywords(words_lower, _HERO_WORDS)
    vil   = count_keywords(words_lower, _VILLAIN_WORDS)
    pow_  = count_keywords(words_lower, _POWER_WORDS)
    vuln  = count_keywords(words_lower, _VULNERABILITY_WORDS)

    dial_lines = len(_DIALOGUE_RE.findall(text))
    rels       = len(_RELATION_RE.findall(text))
    named      = len(set(_NAMED_CHAR_RE.findall(text)) - {name})

    return {
        "word_count":                    n,
        "sentence_count":                len(sentences),
        "paragraph_count":               len(paragraphs),
        "section_count":                 sections,
        "avg_sentence_length":           round(avg_sent_len, 2),
        "unique_word_ratio":             round(unique_ratio, 4),
        "violence_word_count":           viol,
        "conflict_word_count":           conf,
        "leadership_word_count":         lead,
        "family_word_count":             fam,
        "hero_word_count":               hero,
        "villain_word_count":            vil,
        "power_word_count":              pow_,
        "vulnerability_word_count":      vuln,
        "violence_density":              round(viol / n, 5),
        "conflict_density":              round(conf / n, 5),
        "leadership_density":            round(lead / n, 5),
        "family_density":                round(fam / n, 5),
        "hero_density":                  round(hero / n, 5),
        "villain_density":               round(vil / n, 5),
        "power_density":                 round(pow_ / n, 5),
        "vulnerability_density":         round(vuln / n, 5),
        "dialogue_count":                dial_lines,
        "relationship_mentions":         rels,
        "named_characters_mentioned":    named,
        "social_connectedness":          round((rels + named) / n, 5),
        "moral_alignment":               round((hero - vil) / (hero + vil + 1), 4),
        "power_vulnerability_ratio":     round((pow_ + lead) / (vuln + 1), 4),
        "has_biography_section":         int("biography" in text.lower() or "background" in text.lower()),
        "has_dialogue":                  int(dial_lines > 0),
        "is_described_young":            int(bool(_YOUNG_RE.search(text[:1000]))),
        "is_described_old":              int(bool(_OLD_RE.search(text[:1000]))),
        "age_mentioned":                 int(bool(_AGE_RE.search(text[:1000]))),
        "sentiment_score":               (round(TextBlob(text[:2000]).sentiment.polarity, 3)
                                          if _HAS_TEXTBLOB else 0.0),
    }


def main():
    print("=" * 60)
    print("Pipeline 2 — NLP Feature Extraction")
    print("=" * 60)

    raw_csv  = CLEAN_DIR / "characters_raw.csv"
    keys_csv = CLEAN_DIR / "characters_keys.csv"
    if not raw_csv.exists():
        print("ERROR: characters_raw.csv not found. Run build_pipeline.py first.")
        return

    df   = pd.read_csv(raw_csv, low_memory=False)
    keys = pd.read_csv(keys_csv, low_memory=False) if keys_csv.exists() else None
    print(f"Loaded: {df.shape}")

    # Load all page texts
    print("Loading page texts from JSONL...")
    url_to_text = {}
    url_to_name = {}
    for path in sorted(RAW_DIR.glob("*_characters.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        r = json.loads(line)
                        url = r.get("page_url","")
                        if url:
                            url_to_text[url] = (r.get("page_text","") or "")[:5000]
                            url_to_name[url] = r.get("name","")
                    except Exception:
                        pass
    print(f"  {len(url_to_text):,} page texts loaded")

    # Name features (from keys file)
    if keys is not None:
        df["name_word_count"]   = keys["name"].str.split().str.len().fillna(1).astype(int).values
        df["has_title_in_name"] = keys["name"].apply(
            lambda n: int(bool(_TITLE_WORDS_RE.search(str(n))))).values

    # NLP features per row (via page_url from keys)
    print("Extracting NLP features...")
    if keys is not None:
        urls = keys["page_url"].tolist()
        names = keys["name"].tolist()
    else:
        urls  = [""] * len(df)
        names = [""] * len(df)

    nlp_rows = []
    for i, (url, name) in enumerate(zip(urls, names)):
        text = url_to_text.get(url, "")
        nlp_rows.append(extract_nlp(text, name))
        if (i + 1) % 5000 == 0:
            print(f"  {i+1:,}/{len(urls):,} rows processed")

    nlp_df = pd.DataFrame(nlp_rows)
    for col in nlp_df.columns:
        df[col] = nlp_df[col].values

    # Name centrality (co-mention within franchise)
    print("Computing name centrality (skips franchises >2000 chars)...")
    if keys is not None:
        centrality_map = {}
        franchise_groups = keys.groupby("franchise")
        for franchise, grp in franchise_groups:
            urls_f  = grp["page_url"].tolist()
            names_f = grp["name"].str.strip().str.lower().tolist()
            texts_f = [url_to_text.get(u,"") for u in urls_f]
            if len(urls_f) > CENTRALITY_CAP:
                for url in urls_f:
                    centrality_map[url] = 0
                continue
            combined = " ".join(texts_f)
            for url, name, own_text in zip(urls_f, names_f, texts_f):
                if not name or len(name) < 3:
                    centrality_map[url] = 0
                else:
                    total = combined.count(name)
                    self_ = own_text.count(name)
                    centrality_map[url] = max(0, total - self_)

        df["name_centrality"]  = keys["page_url"].map(centrality_map).fillna(0).astype(int).values
        df["page_mention_rank"] = (
            pd.Series(keys["franchise"].values).map(
                lambda f: None  # placeholder, computed below
            )
        )
        # Compute page_mention_rank as percentile within franchise
        tmp = pd.DataFrame({"franchise": keys["franchise"].values,
                             "name_centrality": df["name_centrality"].values})
        df["page_mention_rank"] = tmp.groupby("franchise")["name_centrality"].rank(pct=True).round(3).values

    print(f"Final shape: {df.shape}")

    out = CLEAN_DIR / "characters_full.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
