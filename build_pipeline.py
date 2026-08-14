"""
build_pipeline.py
-----------------
Unified data pipeline: builds characters_raw.csv from scratch.

Inputs  : data/raw/*.jsonl  (38 raw Fandom scrape files)
          Internet access   (auto-downloads Kaggle, CMU, and API data)

Outputs : data/clean/characters_raw.csv   — full feature set, no NLP
          data/clean/characters_keys.csv  — name/url index for external merges

Run     : python build_pipeline.py
          KAGGLE_API_TOKEN=<token> python build_pipeline.py

Runtime : ~35–40 min (Fandom batch queries ≈25 min, TVmaze ≈8 min)
"""

import io
import json
import os
import re
import sys
import tarfile
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote
from urllib.request import urlopen

import pandas as pd
import requests

# ── Paths ──────────────────────────────────────────────────────────────────────
HERE      = Path(__file__).parent
RAW_DIR   = HERE / "data" / "raw"
CLEAN_DIR = HERE / "data" / "clean"
EXT_DIR   = HERE / "data" / "external"
CMU_DIR   = EXT_DIR / "cmu" / "MovieSummaries"
KG_DIR    = EXT_DIR / "kaggle"
CLEAN_DIR.mkdir(parents=True, exist_ok=True)
KG_DIR.mkdir(parents=True, exist_ok=True)
CMU_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "NeedleInHaystack/1.0 (Hebrew University Data Science)"}

# ── Checkpointing ──────────────────────────────────────────────────────────────
CKPT_DIR = CLEAN_DIR / ".ckpt"

def _ckpt_save(name: str, df: pd.DataFrame, keys: pd.DataFrame):
    import pickle
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CKPT_DIR / f"{name}.pkl", "wb") as fh:
        pickle.dump({"df": df, "keys": keys}, fh, protocol=4)
    print(f"  [ckpt] saved: {name}")

def _ckpt_load(name: str):
    import pickle
    p = CKPT_DIR / f"{name}.pkl"
    if not p.exists():
        return None, None
    with open(p, "rb") as fh:
        d = pickle.load(fh)
    print(f"  [ckpt] loaded: {name}")
    return d["df"], d["keys"]

def _ckpt_clear():
    import shutil
    if CKPT_DIR.exists():
        shutil.rmtree(CKPT_DIR)

# Ordered step names — later steps imply earlier ones are complete
_STEPS = ["s2_structural", "s3_apis", "s4_tvmaze", "s5_cmu",
          "s6_kaggle", "s7_fandom"]


def _get(url, **kwargs):
    return requests.get(url, headers=HEADERS, **kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Parse raw JSONL
# ══════════════════════════════════════════════════════════════════════════════

CANONICAL_FIELDS = {
    "real_name": "real_name", "name": "real_name", "real name": "real_name",
    "fullname": "real_name", "full name": "real_name",
    "gender": "gender", "sex": "gender",
    "species": "species", "race": "species",
    "nationality": "nationality", "citizenship": "nationality",
    "culture": "nationality", "origin": "nationality",
    "homeworld": "homeworld",
    "born": "dob", "birth": "dob", "dob": "dob",
    "birth date": "dob", "birthdate": "dob", "birth_date": "dob",
    "died": "dod", "death": "dod", "dod": "dod",
    "deathdate": "dod", "death date": "dod",
    "actor": "actor", "portrayed": "actor", "portrayer": "actor",
    "portrayed by": "actor", "voice": "actor", "voiced by": "actor",
    "affiliation": "affiliation", "loyalty": "affiliation",
    "house": "affiliation", "allegiance": "affiliation",
    "faction": "affiliation", "clan": "affiliation", "side": "affiliation",
    "title": "title", "titles": "title", "occupation": "title",
    "profession": "title", "job": "title", "rank": "title",
    "role": "title", "livelihood": "title", "class": "title",
}

_JUNK_VALUES = {"unknown", "n/a", "none", "null", "--", "--,", "?", "tbd", "amortal"}
_STAR_MARKUP_RE = re.compile(r"\*\d+px\|link=[^\s*]*")


def first_bullet_value(text):
    if not isinstance(text, str):
        return text
    text = _STAR_MARKUP_RE.sub("", text)
    parts = [p.strip() for p in text.split("*") if p.strip()]
    return parts[0] if parts else None


def extract_flat(record: dict, field_map: dict) -> dict:
    infobox = record.get("infobox", {}) or {}
    flat = {
        "wiki": record.get("wiki", ""),
        "franchise": record.get("franchise", ""),
        "content_rating": record.get("content_rating", ""),
        "name": record.get("name", ""),
        "page_url": record.get("page_url", ""),
        "is_dead": record.get("is_dead"),
        "real_name": None, "gender": None, "species": None,
        "nationality": None, "homeworld": None, "dob": None, "dod": None,
        "actor": None, "affiliation": None, "title": None,
        "infobox_field_count": len(infobox),
        "page_text_length": len(record.get("page_text", "") or ""),
        "has_dob": 0, "has_dod": 0, "has_causeofdeath": 0,
        "has_family": 0, "has_image": 0, "has_alias": 0,
        "has_pronouns": 0, "appearance_count": 0,
        "hair_color": None, "eye_color": None,
    }

    for raw_key, raw_val in infobox.items():
        lk = raw_key.lower().strip()
        canonical = field_map.get(lk) or field_map.get(raw_key.strip()) or CANONICAL_FIELDS.get(lk)
        if canonical and canonical in flat and flat[canonical] is None:
            flat[canonical] = str(raw_val)[:300]

    for col in ("real_name", "species", "affiliation", "title", "nationality", "actor"):
        if flat.get(col):
            flat[col] = first_bullet_value(flat[col])

    for col in ("dob", "dod"):
        val = flat.get(col)
        if val and val.strip().lower() in _JUNK_VALUES:
            flat[col] = None

    if flat.get("dob"): flat["has_dob"] = 1
    if flat.get("dod"): flat["has_dod"] = 1

    cod_keys = {"causeofdeath", "cause of death", "cause", "death cause",
                "manner of death", "deathby", "death by", "killedby",
                "killed by", "death reason", "deathreason"}
    family_keys = {"children", "siblings", "families", "parents", "partners",
                   "family", "spouse", "husband", "wife", "brother", "sister",
                   "mother", "father", "son", "daughter", "relatives", "relationships"}
    alias_keys  = {"alias", "aliases", "nickname", "nicknames", "aka",
                   "other names", "also known as"}
    appear_keys = {"tv series", "movie", "game", "book", "comic", "first", "last",
                   "appearances", "seasons", "episodes", "films", "series", "appears in"}
    app_count = 0
    for raw_key, val in infobox.items():
        lk = raw_key.lower().strip()
        if lk in cod_keys and val: flat["has_causeofdeath"] = 1
        if lk in family_keys and val: flat["has_family"] = 1
        if lk in alias_keys and val: flat["has_alias"] = 1
        if lk == "image" and val: flat["has_image"] = 1
        if lk == "pronouns" and val: flat["has_pronouns"] = 1
        if lk in appear_keys and val: app_count += 1
    flat["appearance_count"] = app_count

    hair_raw = eye_raw = None
    for raw_key, val in infobox.items():
        lk = raw_key.lower().strip()
        if lk in ("hair", "hair color", "hair colour", "haircolor"): hair_raw = val
        elif lk in ("eyes", "eye color", "eye colour", "eyecolor", "eye"): eye_raw = val
    for raw, col in [(hair_raw, "hair_color"), (eye_raw, "eye_color")]:
        if raw:
            for c in ['blonde','blond','brown','black','white','red','blue','green',
                      'grey','gray','hazel','silver','gold','yellow','purple','pink']:
                if c in str(raw).lower():
                    flat[col] = 'blonde' if c == 'blond' else ('grey' if c == 'gray' else c)
                    break

    g = (flat.get("gender") or "").lower().strip()
    flat["gender"] = ("Male" if g in ("male", "m", "masculine") else
                      "Female" if g in ("female", "f", "feminine") else
                      "Other/Unknown" if g else None)
    return flat


def parse_jsonl() -> pd.DataFrame:
    print("=" * 60)
    print("STEP 1 — Parsing raw JSONL files")
    print("=" * 60)

    try:
        sys.path.insert(0, str(HERE.parent))
        from group_project.scraper.wiki_configs import WIKI_CONFIGS
    except ImportError:
        try:
            from scraper.wiki_configs import WIKI_CONFIGS
        except ImportError:
            WIKI_CONFIGS = {}

    records = []
    for path in sorted(RAW_DIR.glob("*_characters.jsonl")):
        wiki_key = path.stem.replace("_characters", "")
        config   = WIKI_CONFIGS.get(wiki_key, {})
        field_map = config.get("field_map", {})
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        flat_rows = [extract_flat(r, field_map) for r in rows]
        records.extend(flat_rows)
        print(f"  [{wiki_key:25s}] {len(flat_rows):6,} rows")

    df = pd.DataFrame(records)
    print(f"\nTotal before dedup: {len(df):,}")
    df = df.drop_duplicates(subset=["page_url"])
    df = df.drop_duplicates(subset=["franchise", "name"])
    df = df[df["is_dead"].notna()].copy()
    df["is_dead"] = df["is_dead"].astype(int)
    print(f"After dedup + label filter: {len(df):,} rows  "
          f"({df['is_dead'].mean()*100:.1f}% dead)")
    return df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Structural features
# ══════════════════════════════════════════════════════════════════════════════

FRANCHISE_META = {
    "Star Wars":                       {"release_year": 1977, "medium": "Film", "genre": "Sci-Fi"},
    "Marvel Cinematic Universe":       {"release_year": 2008, "medium": "Film", "genre": "Superhero"},
    "Harry Potter":                    {"release_year": 1997, "medium": "Film", "genre": "Fantasy"},
    "Indiana Jones":                   {"release_year": 1981, "medium": "Film", "genre": "Adventure"},
    "Game of Thrones":                 {"release_year": 2011, "medium": "TV",   "genre": "Fantasy"},
    "James Bond 007":                  {"release_year": 1962, "medium": "Film", "genre": "Action"},
    "Alien vs Predator":               {"release_year": 1979, "medium": "Film", "genre": "Sci-Fi/Horror"},
    "DC Extended Universe":            {"release_year": 2013, "medium": "Film", "genre": "Superhero"},
    "Pirates of the Caribbean":        {"release_year": 2003, "medium": "Film", "genre": "Adventure"},
    "Jurassic Park":                   {"release_year": 1993, "medium": "Film", "genre": "Sci-Fi"},
    "Grey's Anatomy":                  {"release_year": 2005, "medium": "TV",   "genre": "Drama"},
    "Lord of the Rings":               {"release_year": 1954, "medium": "Film", "genre": "Fantasy"},
    "Lost":                            {"release_year": 2004, "medium": "TV",   "genre": "Drama"},
    "Breaking Bad / Better Call Saul": {"release_year": 2008, "medium": "TV",   "genre": "Crime"},
    "Dune":                            {"release_year": 1965, "medium": "Film", "genre": "Sci-Fi"},
    "Avatar":                          {"release_year": 2009, "medium": "Film", "genre": "Sci-Fi"},
    "Transformers":                    {"release_year": 1984, "medium": "Film", "genre": "Sci-Fi"},
    "Sons of Anarchy":                 {"release_year": 2008, "medium": "TV",   "genre": "Crime"},
    "Supernatural":                    {"release_year": 2005, "medium": "TV",   "genre": "Fantasy"},
    "The Hunger Games":                {"release_year": 2008, "medium": "Film", "genre": "Dystopian"},
    "The Sopranos":                    {"release_year": 1999, "medium": "TV",   "genre": "Crime"},
    "The Twilight Saga":               {"release_year": 2005, "medium": "Film", "genre": "Fantasy"},
    "Spartacus":                       {"release_year": 2010, "medium": "TV",   "genre": "Historical"},
    "Dexter":                          {"release_year": 2006, "medium": "TV",   "genre": "Crime"},
    "The 100":                         {"release_year": 2014, "medium": "TV",   "genre": "Dystopian"},
    "Vikings":                         {"release_year": 2013, "medium": "TV",   "genre": "Historical"},
    "Westworld":                       {"release_year": 2016, "medium": "TV",   "genre": "Sci-Fi"},
    "Ozark":                           {"release_year": 2017, "medium": "TV",   "genre": "Crime"},
    "Fast & Furious":                  {"release_year": 2001, "medium": "Film", "genre": "Action"},
    "Mission: Impossible":             {"release_year": 1996, "medium": "Film", "genre": "Action"},
    "Stranger Things":                 {"release_year": 2016, "medium": "TV",   "genre": "Sci-Fi/Horror"},
    "The Boys":                        {"release_year": 2019, "medium": "TV",   "genre": "Superhero"},
    "The Walking Dead":                {"release_year": 2010, "medium": "TV",   "genre": "Horror"},
    "Prison Break":                    {"release_year": 2005, "medium": "TV",   "genre": "Action"},
    "Peaky Blinders":                  {"release_year": 2013, "medium": "TV",   "genre": "Crime"},
    "The Matrix":                      {"release_year": 1999, "medium": "Film", "genre": "Sci-Fi"},
    "Boardwalk Empire":                {"release_year": 2010, "medium": "TV",   "genre": "Crime"},
    "The Wire":                        {"release_year": 2002, "medium": "TV",   "genre": "Crime"},
}

_ARCHETYPE_RULES = [
    ("Military",    {"soldier","private","corporal","sergeant","lieutenant","captain",
                     "major","colonel","general","admiral","commander","officer","pilot",
                     "trooper","marine","guard","warrior","fighter","ranger","clone trooper",
                     "stormtrooper","legionary","gladiator"}),
    ("Royalty",     {"king","queen","prince","princess","emperor","empress","lord","lady",
                     "ser","sir","duke","duchess","baron","count","countess","heir",
                     "monarch","regent"}),
    ("Criminal",    {"criminal","mobster","gangster","thief","smuggler","drug dealer",
                     "hitman","assassin","pirate","outlaw","henchman","mercenary",
                     "bounty hunter","enforcer","crime lord","drug lord","sith lord",
                     "dark lord","crime boss","thug","bandit","raider"}),
    ("Force User",  {"jedi","sith","force-sensitive","padawan","jedi master",
                     "jedi knight","dark side"}),
    ("Law/Order",   {"detective","agent","sheriff","marshal","police","cop","fbi","cia",
                     "spy","intelligence operative","investigator","auror","judge",
                     "lawyer","attorney","law enforcement","constable"}),
    ("Medical",     {"doctor","m.d.","nurse","healer","medic","surgeon","physician","paramedic"}),
    ("Academic",    {"professor","scientist","researcher","scholar","archaeologist",
                     "author","teacher","student","prefect","historian","librarian","inventor"}),
    ("Political",   {"president","senator","governor","chancellor","minister","mayor",
                     "politician","ambassador","diplomat","councilor","grand moff","moff",
                     "viceroy","magistrate"}),
    ("Religious",   {"priest","priestess","monk","nun","reverend","bishop","septon",
                     "septa","maester","oracle","cleric","shaman","witch"}),
    ("Worker",      {"farmer","engineer","mechanic","miner","builder","cook","bartender",
                     "driver","firefighter","technician","craftsman","blacksmith",
                     "trader","merchant","shopkeeper"}),
    ("Entertainer", {"actor","musician","singer","dancer","artist","reporter",
                     "journalist","writer","photographer","entertainer","performer"}),
]
_TEXT_ARCHETYPE_PATTERNS = [
    ("Military",    re.compile(r"\b(soldier|officer|served in the (?:army|navy|military)|trooper|fought (?:in|during) the|enlisted|military career|commanding officer|squadron|battalion|regiment)\b", re.I)),
    ("Royalty",     re.compile(r"\b(king|queen|prince|princess|emperor|empress|lord of|lady of|ruled|reign(?:ed)?|royal family|heir to the throne|noble(?:man|woman)?|aristocrat)\b", re.I)),
    ("Criminal",    re.compile(r"\b(criminal|crime (?:lord|boss|family)|drug (?:lord|dealer|trade)|smuggl|assassin|bounty hunter|pirate|thief|gangster|mobster|hitman|sith lord|dark lord|dark side)\b", re.I)),
    ("Force User",  re.compile(r"\b(jedi|sith|force[- ]sensitive|padawan|lightsaber|the force|dark side of the force|jedi (?:master|knight|council)|sith (?:lord|apprentice))\b", re.I)),
    ("Law/Order",   re.compile(r"\b(detective|police|law enforcement|agent of|special agent|sheriff|marshal|investigat|auror|spy|espionage|intelligence)\b", re.I)),
    ("Medical",     re.compile(r"\b(doctor|physician|surgeon|nurse|medic(?:al)?|healer|hospital|medical (?:school|practice|career))\b", re.I)),
    ("Academic",    re.compile(r"\b(professor|scientist|researcher|scholar|archaeolog|teacher|student at|studied at|university|academy|hogwarts)\b", re.I)),
    ("Political",   re.compile(r"\b(senator|governor|chancellor|president of|politician|ambassador|diplomat|political career|elected|council(?:lor|man|woman))\b", re.I)),
    ("Religious",   re.compile(r"\b(priest|priestess|monk|nun|temple|religious|cleric|shaman|spiritual leader|septon|septa)\b", re.I)),
    ("Worker",      re.compile(r"\b(engineer|mechanic|farmer|miner|builder|technician|craftsman|blacksmith|merchant|trader|bartender|shopkeeper)\b", re.I)),
    ("Entertainer", re.compile(r"\b(reporter|journalist|musician|singer|dancer|artist|performer|actor|actress|writer|author)\b", re.I)),
]
_MALE_PRONOUNS   = frozenset(["he","him","his","himself"])
_FEMALE_PRONOUNS = frozenset(["she","her","hers","herself"])
_HUMAN_SPECIES   = frozenset(["human","humans","homo sapiens","man","woman",
                               "human/mutate","human/inhuman","enhanced human",
                               "augmented human","men","numenorean","dunedain",
                               "gondorian","rohirrim","hobbit","hobbits"])
_HUMAN_ONLY_FRANCHISES = frozenset([
    "Game of Thrones","James Bond 007","Grey's Anatomy","Lost",
    "Breaking Bad / Better Call Saul","Sons of Anarchy","The Sopranos",
    "Vikings","Dexter","Ozark","Peaky Blinders","Prison Break",
    "Boardwalk Empire","The Wire","Spartacus","The Hunger Games",
    "Indiana Jones","Fast & Furious","Mission: Impossible",
])
_GOOD_AFFILIATIONS = re.compile(
    r"(rebel|rebellion|alliance|republic|resistance|jedi|gryffindor|hufflepuff|"
    r"ravenclaw|order of the phoenix|avengers|s\.h\.i\.e\.l\.d|justice league|"
    r"the 100|skaikru|night.?s watch|stark|ministry of magic|rebel alliance|"
    r"order of merlin|dumbledore|free folk|autobots?)", re.I)
_EVIL_AFFILIATIONS = re.compile(
    r"(empire|imperial|sith|dark side|galactic empire|first order|separatist|"
    r"death eater|slytherin|hydra|ten rings|crime|criminal|cartel|"
    r"dark|evil|tyrell|lannister|weyland.?yutani|decepticons?|"
    r"white walkers?|voldemort|hutt|jabba|cersei)", re.I)


def classify_archetype(title):
    if not isinstance(title, str) or not title.strip():
        return "Unknown"
    t = title.lower()
    for archetype, kws in _ARCHETYPE_RULES:
        if any(kw in t for kw in kws):
            return archetype
    return "Other"


def classify_archetype_from_text(text, n=2000):
    if not text:
        return "Unknown"
    snippet = text[:n].lower()
    scores = {}
    for archetype, pat in _TEXT_ARCHETYPE_PATTERNS:
        hits = len(pat.findall(snippet))
        if hits:
            scores[archetype] = hits
    return max(scores, key=scores.get) if scores else "Unknown"


def infer_gender(text, n=500):
    if not text:
        return None
    words = text.lower().split()[:n]
    m = sum(1 for w in words if w in _MALE_PRONOUNS)
    f = sum(1 for w in words if w in _FEMALE_PRONOUNS)
    if m >= 2 and m > f * 1.5: return "Male"
    if f >= 2 and f > m * 1.5: return "Female"
    return None


def derive_structural(df: pd.DataFrame, jsonl_texts: dict) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("STEP 2 — Deriving structural features")
    print("=" * 60)

    # Franchise metadata
    df["franchise_release_year"] = df["franchise"].map(
        lambda f: FRANCHISE_META.get(f, {}).get("release_year"))
    df["franchise_decade"]  = df["franchise_release_year"].apply(
        lambda y: f"{int(y)//10*10}s" if pd.notna(y) else None)
    df["franchise_era"] = df["franchise_release_year"].apply(
        lambda y: ("Pre-1980" if y and y < 1980 else
                   "1980s-90s" if y and y < 2000 else
                   "2000s" if y and y < 2010 else
                   "2010s+" if y else None))
    df["medium"] = df["franchise"].map(lambda f: FRANCHISE_META.get(f, {}).get("medium"))
    df["genre"]  = df["franchise"].map(lambda f: FRANCHISE_META.get(f, {}).get("genre"))

    # Franchise aggregate stats (computed from labelled dataset)
    fstats = df.groupby("franchise").agg(
        franchise_size=("is_dead", "count"),
        franchise_mortality_rate=("is_dead", "mean"),
    ).reset_index()
    fstats["franchise_female_ratio"] = (
        df.groupby("franchise").apply(
            lambda g: (g["gender"] == "Female").mean()
        ).reset_index(drop=True)
    )
    fstats["franchise_weight"] = 1.0 / fstats["franchise_size"]
    df = df.merge(fstats, on="franchise", how="left")

    # Prominence tier
    def tier(n):
        if n < 500:   return "Minor"
        if n < 2000:  return "Supporting"
        if n < 10000: return "Major"
        return "Lead"
    df["prominence_tier"] = df["page_text_length"].apply(tier)

    # is_human
    df["is_human"] = df.apply(
        lambda r: (1 if isinstance(r["species"], str) and
                   r["species"].lower().strip() in _HUMAN_SPECIES else
                   1 if r["franchise"] in _HUMAN_ONLY_FRANCHISES else
                   0 if isinstance(r["species"], str) and r["species"].strip() else None),
        axis=1)

    # Affiliation alignment
    df["affiliation_alignment"] = df["affiliation"].apply(
        lambda a: ("Ambiguous" if isinstance(a, str) and
                   _GOOD_AFFILIATIONS.search(a) and _EVIL_AFFILIATIONS.search(a) else
                   "Good" if isinstance(a, str) and _GOOD_AFFILIATIONS.search(a) else
                   "Evil" if isinstance(a, str) and _EVIL_AFFILIATIONS.search(a) else
                   "Neutral" if isinstance(a, str) and a.strip() else None))

    # Archetype + gender inference from page text
    archetypes, genders_inferred = [], []
    for _, row in df.iterrows():
        text = jsonl_texts.get(row["page_url"], "")
        arch = classify_archetype(row.get("title"))
        if arch in ("Unknown", "Other"):
            arch = classify_archetype_from_text(text)
        archetypes.append(arch)
        if pd.isna(row["gender"]) or row["gender"] is None:
            genders_inferred.append(infer_gender(text))
        else:
            genders_inferred.append(row["gender"])
    df["archetype"] = archetypes
    df["gender"]    = genders_inferred

    # Drop leaky + high-null raw fields; keep key columns in keys file
    LEAKY = ["has_dod", "dod", "has_causeofdeath"]
    HIGH_NULL = ["nationality", "homeworld", "actor", "title",
                 "hair_color", "eye_color", "species", "affiliation", "real_name", "dob"]
    df = df.drop(columns=LEAKY + HIGH_NULL, errors="ignore")

    print(f"  Structural features added. Shape: {df.shape}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Franchise APIs (free, no auth)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_got_api() -> pd.DataFrame:
    rows, page = [], 1
    while True:
        data = _get(f"https://anapioficeandfire.com/api/characters",
                    params={"pageSize": 50, "page": page}, timeout=20).json()
        if not data: break
        for c in data:
            name = c.get("name") or (c.get("aliases") or [""])[0]
            rows.append({
                "char_lower":         name.strip().lower(),
                "got_tv_seasons":     len(c.get("tvSeries", [])),
                "got_book_count":     len(c.get("books", [])),
                "got_pov_count":      len(c.get("povBooks", [])),
                "got_allegiances":    len(c.get("allegiances", [])),
                "got_titles_count":   len(c.get("titles", [])),
            })
        if len(data) < 50: break
        page += 1
        time.sleep(0.3)
    return pd.DataFrame(rows).drop_duplicates("char_lower")


def fetch_hp_api() -> pd.DataFrame:
    data = _get("https://hp-api.onrender.com/api/characters", timeout=30).json()
    rows = [{"char_lower":         (c.get("name") or "").strip().lower(),
              "hp_house":          c.get("house", "") or "",
              "hp_ancestry":       c.get("ancestry", "") or "",
              "hp_hogwarts_student": int(c.get("hogwartsStudent", False)),
              "hp_hogwarts_staff":   int(c.get("hogwartsStaff", False))} for c in data]
    return pd.DataFrame(rows).drop_duplicates("char_lower")


def fetch_swapi() -> pd.DataFrame:
    rows, url = [], "https://swapi.py4e.com/api/people/"
    while url:
        data = _get(url, timeout=15).json()
        for c in data.get("results", []):
            rows.append({"char_lower":    (c.get("name") or "").strip().lower(),
                          "sw_film_count": len(c.get("films", [])),
                          "sw_height":     c.get("height"),
                          "sw_mass":       c.get("mass"),
                          "sw_birth_year": c.get("birth_year")})
        url = data.get("next")
        time.sleep(0.5)
    return pd.DataFrame(rows).drop_duplicates("char_lower")


def fetch_potterdb() -> pd.DataFrame:
    rows, page = [], 1
    while True:
        data = _get("https://api.potterdb.com/v1/characters",
                    params={"page[size]": 100, "page[number]": page}, timeout=20).json()
        if not data.get("data"): break
        for c in data["data"]:
            a = c.get("attributes", {})
            rows.append({
                "char_lower":       (a.get("name") or "").strip().lower(),
                "pdb_house":        a.get("house") or "",
                "pdb_blood_status": a.get("blood_status") or "",
                "pdb_species":      a.get("species") or "",
                "pdb_nationality":  a.get("nationality") or "",
                "pdb_animagus":     int(bool(a.get("animagus"))),
                "pdb_patronus":     int(bool(a.get("patronus"))),
                "pdb_titles_count": len(a.get("titles") or []),
                "pdb_jobs_count":   len(a.get("jobs") or []),
            })
        total_pages = data.get("meta", {}).get("pagination", {}).get("last", page)
        if page >= total_pages: break
        page += 1
        time.sleep(0.5)
    return pd.DataFrame(rows).drop_duplicates("char_lower")


def fetch_sw_akabab() -> pd.DataFrame:
    data = _get("https://akabab.github.io/starwars-api/api/all.json", timeout=15).json()
    rows = [{"char_lower":          (c.get("name") or "").strip().lower(),
              "sw_species_detail":  c.get("species") or "",
              "sw_affiliations_n":  len(c.get("affiliations") or []),
              "sw_has_masters":     int(bool(c.get("masters"))),
              "sw_has_apprentices": int(bool(c.get("apprentices"))),
              "sw_has_cybernetics": int(bool(c.get("cybernetics")))} for c in data]
    return pd.DataFrame(rows).drop_duplicates("char_lower")


def merge_franchise_apis(df: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("STEP 3 — Franchise APIs")
    print("=" * 60)

    def merge_on(franchise, api_df, on="char_lower"):
        idx = keys[keys["franchise"] == franchise].index
        bridge = keys[keys["franchise"] == franchise].merge(api_df, on=on, how="left")
        for col in [c for c in api_df.columns if c != on]:
            df.loc[idx, col] = bridge[col].values
        n = bridge[api_df.columns[1]].notna().sum()
        print(f"  {franchise}: {n}/{len(idx)} matched")

    print("  Fetching An API of Ice and Fire (GoT)...")
    got = fetch_got_api()
    merge_on("Game of Thrones", got)

    print("  Fetching Harry Potter API...")
    hp = fetch_hp_api()
    merge_on("Harry Potter", hp)

    print("  Fetching SWAPI...")
    sw = fetch_swapi()
    merge_on("Star Wars", sw)

    print("  Fetching PotterDB (54 pages)...")
    pdb = fetch_potterdb()
    merge_on("Harry Potter", pdb)

    print("  Fetching SW Akabab...")
    aka = fetch_sw_akabab()
    merge_on("Star Wars", aka)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — TVmaze: episode counts + actor age
# ══════════════════════════════════════════════════════════════════════════════

TV_SHOWS = {
    "Game of Thrones":                 [("Game of Thrones", "Game of Thrones")],
    "Breaking Bad / Better Call Saul": [("Breaking Bad", "Breaking Bad"),
                                         ("Better Call Saul", "Breaking Bad / Better Call Saul")],
    "The Walking Dead":                [("The Walking Dead", "The Walking Dead")],
    "Supernatural":                    [("Supernatural", "Supernatural")],
    "Grey's Anatomy":                  [("Grey's Anatomy", "Grey's Anatomy")],
    "Lost":                            [("Lost", "Lost")],
    "Dexter":                          [("Dexter", "Dexter")],
    "Sons of Anarchy":                 [("Sons of Anarchy", "Sons of Anarchy")],
    "Boardwalk Empire":                [("Boardwalk Empire", "Boardwalk Empire")],
    "The Wire":                        [("The Wire", "The Wire")],
    "Prison Break":                    [("Prison Break", "Prison Break")],
    "Westworld":                       [("Westworld", "Westworld")],
    "The 100":                         [("The 100", "The 100")],
    "Stranger Things":                 [("Stranger Things", "Stranger Things")],
    "Vikings":                         [("Vikings", "Vikings")],
    "Spartacus":                       [("Spartacus Blood and Sand", "Spartacus")],
    "Peaky Blinders":                  [("Peaky Blinders", "Peaky Blinders")],
    "The Boys":                        [("The Boys", "The Boys")],
    "Ozark":                           [("Ozark", "Ozark")],
    "The Sopranos":                    [("The Sopranos", "The Sopranos")],
    # Avatar is the 2009 James Cameron film — not a TV show, excluded from TVmaze
}


def fetch_tvmaze(df: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("STEP 4 — TVmaze (episode counts + actor age)")
    print("=" * 60)
    base = "https://api.tvmaze.com"
    req_count = [0]

    def tvget(url):
        req_count[0] += 1
        if req_count[0] % 18 == 0:
            time.sleep(10)
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"    [warn] {e}")
            return None

    episode_counts  = {}  # (franchise, char_lower) -> count
    actor_ages      = {}  # (franchise, char_lower) -> age
    main_cast_keys  = set()  # (franchise, char_lower) for main cast members

    for franchise, show_list in TV_SHOWS.items():
        for search_term, fr_key in show_list:
            show = tvget(f"{base}/singlesearch/shows?q={requests.utils.quote(search_term)}")
            if not show:
                continue
            show_id  = show["id"]
            premiere = show.get("premiered") or ""
            try:
                premiere_dt = datetime.strptime(premiere[:10], "%Y-%m-%d")
            except Exception:
                premiere_dt = None

            # Main cast → total episode count + actor ages
            cast = tvget(f"{base}/shows/{show_id}/cast") or []
            eps  = tvget(f"{base}/shows/{show_id}/episodes") or []
            total_eps = len(eps)

            for c in cast:
                char_name = ((c.get("character") or {}).get("name") or "").strip().lower()
                birthday  = (c.get("person") or {}).get("birthday")
                if char_name:
                    episode_counts[(fr_key, char_name)] = total_eps
                    main_cast_keys.add((fr_key, char_name))
                    if premiere_dt and birthday:
                        try:
                            bday = datetime.strptime(birthday[:10], "%Y-%m-%d")
                            age  = (premiere_dt - bday).days / 365.25
                            if 0 < age < 100:
                                actor_ages[(fr_key, char_name)] = round(age, 1)
                        except Exception:
                            pass

            # Guest cast per episode — increment per episode, skip main cast
            for ep in eps:
                guests = tvget(f"{base}/episodes/{ep['id']}/guestcast") or []
                for g in guests:
                    char_name = ((g.get("character") or {}).get("name") or "").strip().lower()
                    if char_name and (fr_key, char_name) not in main_cast_keys:
                        key = (fr_key, char_name)
                        episode_counts[key] = episode_counts.get(key, 0) + 1

            print(f"  [{franchise}] done")

    # Map to df
    ep_col  = keys.apply(lambda r: episode_counts.get((r["franchise"], r["char_lower"]), 0), axis=1)
    age_col = keys.apply(lambda r: actor_ages.get((r["franchise"], r["char_lower"])), axis=1)

    df["tvmaze_episode_count"] = ep_col.values
    df["tvmaze_is_main_cast"]  = (ep_col > 5).astype(int).values
    df["tvmaze_found"]         = (ep_col > 0).astype(int).values
    df["actor_age_at_release"] = age_col.values
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — CMU Movie Summary Corpus (actor age for film chars)
# ══════════════════════════════════════════════════════════════════════════════

CMU_URL = "http://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz"
FILM_FRANCHISE_KEYWORDS = {
    "Star Wars":                       "star wars",
    "Marvel Cinematic Universe":       r"aveng|iron man|captain america|thor|black panther",
    "Indiana Jones":                   "indiana jones",
    "Jurassic Park":                   "jurassic",
    "Pirates of the Caribbean":        "pirates of the caribbean",
    "The Matrix":                      "matrix",
    "The Twilight Saga":               "twilight",
    "Transformers":                    "transformers",
    "The Hunger Games":                "hunger games",
    "Lord of the Rings":               "lord of the rings|hobbit",
    "Avatar":                          r"avatar",
    "Dune":                            "dune",
    "James Bond 007":                  r"james bond|007|casino royale|skyfall|spectre",
    "Alien vs Predator":               r"alien|predator",
    "DC Extended Universe":            r"superman|batman|wonder woman|aquaman|suicide squad",
    "Fast & Furious":                  r"fast.*furious|furious",
    "Mission: Impossible":             "mission.*impossible",
    "Harry Potter":                    "harry potter|fantastic beasts",
}


def fetch_cmu(df: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("STEP 5 — CMU Movie Summary Corpus")
    print("=" * 60)

    char_meta = CMU_DIR / "character.metadata.tsv"
    movie_meta = CMU_DIR / "movie.metadata.tsv"

    if not char_meta.exists():
        print(f"  Downloading CMU corpus (~48 MB)...")
        r = requests.get(CMU_URL, headers=HEADERS, stream=True, timeout=60)
        data = b"".join(chunk for chunk in r.iter_content(1 << 20))
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as t:
            t.extractall(CMU_DIR.parent)
        print("  Extracted.")

    cmu_cols = ["wiki_movie_id","freebase_movie_id","release_date","char_name",
                "actor_dob","actor_gender","actor_height","actor_ethnicity",
                "actor_name","actor_age_at_release","fb1","fb2","fb3"]
    cmu = pd.read_csv(char_meta, sep="\t", header=None, names=cmu_cols, low_memory=False)
    movies = pd.read_csv(movie_meta, sep="\t", header=None,
                         names=["wiki_id","fb_id","movie_name","rd","bo","rt","lang","ctry","genre"],
                         low_memory=False)
    cmu = cmu.merge(movies[["wiki_id","movie_name"]], left_on="wiki_movie_id",
                    right_on="wiki_id", how="left")
    cmu["char_lower"]  = cmu["char_name"].str.strip().str.lower()
    cmu["movie_lower"] = cmu["movie_name"].str.strip().str.lower()
    cmu["actor_age_at_release"] = pd.to_numeric(cmu["actor_age_at_release"], errors="coerce")

    cmu_lookup = {}
    for franchise, pattern in FILM_FRANCHISE_KEYWORDS.items():
        mask = cmu["movie_lower"].str.contains(pattern, case=False, na=False, regex=True)
        ages = cmu[mask].groupby("char_lower")["actor_age_at_release"].median()
        for char, age in ages.items():
            cmu_lookup[(franchise, char)] = round(age, 1)
    print(f"  CMU lookup: {len(cmu_lookup):,} franchise+character entries")

    null_age = df["actor_age_at_release"].isna()
    fill_ages = keys.apply(
        lambda r: cmu_lookup.get((r["franchise"], r["char_lower"])), axis=1)
    df.loc[null_age, "actor_age_at_release"] = fill_ages[null_age].values
    print(f"  Actor age non-null: {df['actor_age_at_release'].notna().sum():,}")

    def age_group(a):
        if pd.isna(a): return "unknown"
        if a < 18:     return "child"
        if a < 35:     return "young_adult"
        if a < 60:     return "adult"
        return "senior"
    df["age_group"] = df["actor_age_at_release"].apply(age_group)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Kaggle datasets
# ══════════════════════════════════════════════════════════════════════════════

KAGGLE_DATASETS = [
    ("mylesoneill/game-of-thrones", None),
    ("thedevastator/uncover-the-mystery-behind-got-characters-screen", None),
    ("claudiodavi/superhero-set", None),
    ("paultimothymooney/lord-of-the-rings-data", None),
    ("mexwell/the-lord-of-the-rings", None),
    ("saguit03/the-hunger-games-characters", None),
    ("thedevastator/the-hunger-games-dataset-a-survival-analysis", None),
    ("bac3917/frank-herberts-dune-characters", None),
    ("kyleakepanidtaworn/marvel-characters-dataset", None),
    # Star Wars supplemental
    ("ulrikthygepedersen/star-wars-characters", None),
    ("kacperrabczewski/star-wars-characters", None),
    # MCU supplemental
    ("apriandito/mcu-characters", None),
]


def ensure_kaggle_files():
    token = os.environ.get("KAGGLE_API_TOKEN")
    if not token:
        print("  [warn] KAGGLE_API_TOKEN not set — skipping Kaggle auto-download.")
        print("  [warn] Place Kaggle CSVs in data/external/kaggle/ manually.")
        return
    os.environ["KAGGLE_API_TOKEN"] = token
    try:
        import kaggle as kg
        api = kg.KaggleApi()
        api.authenticate()
    except Exception as e:
        print(f"  [warn] Kaggle auth failed: {e}")
        return
    for ref, _ in KAGGLE_DATASETS:
        try:
            api.dataset_download_files(ref, path=str(KG_DIR), unzip=True, quiet=True)
            print(f"  Downloaded {ref}")
        except Exception as e:
            print(f"  [warn] {ref}: {e}")


def _kcsv(fname) -> pd.DataFrame:
    path = KG_DIR / fname
    if not path.exists():
        # Search recursively
        matches = list(KG_DIR.rglob(fname))
        if not matches:
            return pd.DataFrame()
        path = matches[0]
    return pd.read_csv(path, encoding="latin1")


def merge_kaggle(df: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("STEP 6 — Kaggle datasets")
    print("=" * 60)
    ensure_kaggle_files()

    def idx_bridge(franchise):
        idx = keys[keys["franchise"] == franchise].index
        return idx, keys[keys["franchise"] == franchise].copy()

    # ── GoT deaths (FiveThirtyEight) ─────────────────────────────────────────
    deaths = _kcsv("character-deaths.csv")
    if not deaths.empty:
        deaths["char_lower"] = deaths["Name"].str.strip().str.lower()
        deaths = deaths.rename(columns={
            "GoT":"book1_appears","CoK":"book2_appears","SoS":"book3_appears",
            "FfC":"book4_appears","DwD":"book5_appears","Nobility":"got_nobility"})
        deaths = deaths[["char_lower","got_nobility","book1_appears","book2_appears",
                          "book3_appears","book4_appears","book5_appears"]].drop_duplicates("char_lower")
        idx, br = idx_bridge("Game of Thrones")
        merged = br.merge(deaths, on="char_lower", how="left").reset_index(drop=True)
        for col in ["got_nobility","book1_appears","book2_appears","book3_appears",
                    "book4_appears","book5_appears"]:
            df.loc[idx, col] = merged[col].values
        df["got_books_total"] = (
            df[["book1_appears","book2_appears","book3_appears","book4_appears","book5_appears"]]
            .apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1))
        print(f"  GoT deaths: {merged['got_nobility'].notna().sum()}/{len(idx)} matched")

    # ── GoT predictions ───────────────────────────────────────────────────────
    preds = _kcsv("character-predictions.csv")
    if not preds.empty:
        LEAKY_PRED = ["actual","pred","alive","plod","isAlive","DateoFdeath",
                      "isAliveMother","isAliveFather","isAliveHeir","isAliveSpouse"]
        preds = preds.drop(columns=[c for c in LEAKY_PRED if c in preds.columns])
        preds["char_lower"] = preds["name"].str.strip().str.lower()
        keep = ["char_lower","popularity","age","numDeadRelations","isMarried","isNoble"]
        preds_k = preds[[c for c in keep if c in preds.columns]].drop_duplicates("char_lower")
        idx, br = idx_bridge("Game of Thrones")
        merged = br.merge(preds_k, on="char_lower", how="left").reset_index(drop=True)
        df.loc[idx, "got_popularity"]     = merged.get("popularity", pd.Series()).values
        df.loc[idx, "got_age"]            = merged.get("age", pd.Series()).values
        df.loc[idx, "got_dead_relations"] = merged.get("numDeadRelations", pd.Series()).values
        df.loc[idx, "got_married"]        = merged.get("isMarried", pd.Series()).values
        df.loc[idx, "got_noble"]          = merged.get("isNoble", pd.Series()).values
        print(f"  GoT predictions: {merged.get('popularity', pd.Series()).notna().sum()}/{len(idx)} matched")

    # ── GoT screen time ───────────────────────────────────────────────────────
    st = _kcsv("GOT_screentimes.csv")
    if not st.empty:
        st["char_lower"] = st["name"].str.strip().str.lower()
        st["screentime"]  = pd.to_numeric(st["screentime"], errors="coerce")
        st["episodes"]    = pd.to_numeric(st["episodes"],   errors="coerce")
        st_k = st[["char_lower","screentime","episodes"]].rename(
            columns={"screentime":"got_screentime_min","episodes":"got_episode_count_imdb"}
        ).drop_duplicates("char_lower")
        idx, br = idx_bridge("Game of Thrones")
        merged = br.merge(st_k, on="char_lower", how="left").reset_index(drop=True)
        df.loc[idx, "got_screentime_min"]     = merged["got_screentime_min"].values
        df.loc[idx, "got_episode_count_imdb"] = merged["got_episode_count_imdb"].values
        print(f"  GoT screen time: {merged['got_screentime_min'].notna().sum()}/{len(idx)} matched")

    # ── Superhero info (MCU + DC) ─────────────────────────────────────────────
    heroes = _kcsv("heroes_information.csv")
    if not heroes.empty:
        heroes["char_lower"] = heroes["name"].str.strip().str.lower()
        heroes_k = heroes[["char_lower","Alignment","Race"]].rename(
            columns={"Alignment":"hero_alignment","Race":"hero_race"}
        ).drop_duplicates("char_lower")
        for fr in ["Marvel Cinematic Universe","DC Extended Universe"]:
            idx, br = idx_bridge(fr)
            merged = br.merge(heroes_k, on="char_lower", how="left").reset_index(drop=True)
            df.loc[idx, "hero_alignment"] = merged["hero_alignment"].values
            df.loc[idx, "hero_race"]      = merged["hero_race"].values
        print(f"  Superhero info: {heroes_k['hero_alignment'].notna().sum()} entries")

    # ── Superhero powers ──────────────────────────────────────────────────────
    powers = _kcsv("super_hero_powers.csv")
    if not powers.empty:
        powers["char_lower"] = powers["hero_names"].str.strip().str.lower()
        pw_cols = ["Immortality","Magic","Resurrection","The Force","Telepathy",
                   "Mind Control","Invulnerability","Regeneration","Flight","Super Strength"]
        pw_cols = [c for c in pw_cols if c in powers.columns]
        powers_k = powers[["char_lower"] + pw_cols].drop_duplicates("char_lower")
        for fr in ["Marvel Cinematic Universe","DC Extended Universe"]:
            idx, br = idx_bridge(fr)
            merged = br.merge(powers_k, on="char_lower", how="left").reset_index(drop=True)
            for col in pw_cols:
                df.loc[idx, f"power_{col.lower().replace(' ','_')}"] = merged[col].values

    # ── LOTR ──────────────────────────────────────────────────────────────────
    lotr = _kcsv("lotr_characters.csv")
    if not lotr.empty:
        lotr = lotr.drop(columns=["birth","death"], errors="ignore")
        lotr["char_lower"] = lotr["name"].str.strip().str.lower()
        lotr["lotr_has_spouse"] = lotr["spouse"].notna().astype(int)
        lotr_k = lotr[["char_lower","race","realm","lotr_has_spouse"]].rename(
            columns={"race":"lotr_race","realm":"lotr_realm"}
        ).drop_duplicates("char_lower")
        idx, br = idx_bridge("Lord of the Rings")
        merged = br.merge(lotr_k, on="char_lower", how="left").reset_index(drop=True)
        for col in ["lotr_race","lotr_realm","lotr_has_spouse"]:
            df.loc[idx, col] = merged[col].values
        print(f"  LOTR chars: {merged['lotr_race'].notna().sum()}/{len(idx)} matched")

    # ── LOTR words ────────────────────────────────────────────────────────────
    words = _kcsv("WordsByCharacter.csv")
    if not words.empty:
        words["first_name_lower"] = words["Character"].str.strip().str.lower()
        wsum = words.groupby("first_name_lower")["Words"].sum().reset_index()
        wsum = wsum.rename(columns={"Words":"lotr_total_words"})
        idx, br = idx_bridge("Lord of the Rings")
        br["first_name_lower"] = br["char_lower"].str.split().str[0]
        merged = br.merge(wsum, on="first_name_lower", how="left").reset_index(drop=True)
        df.loc[idx, "lotr_total_words"] = merged["lotr_total_words"].values

    # ── LOTR scripts dialogue ─────────────────────────────────────────────────
    scripts = _kcsv("lotr_scripts.csv")
    if not scripts.empty:
        scripts["char_lower"]  = scripts["char"].str.strip().str.lower()
        scripts["first_name_lower"] = scripts["char_lower"].str.split().str[0]
        scripts["word_count"] = scripts["dialog"].fillna("").str.split().str.len()
        dial = scripts.groupby("first_name_lower")["word_count"].sum().reset_index()
        dial = dial.rename(columns={"word_count":"lotr_dialogue_words"})
        idx, br = idx_bridge("Lord of the Rings")
        br["first_name_lower"] = br["char_lower"].str.split().str[0]
        merged = br.merge(dial, on="first_name_lower", how="left").reset_index(drop=True)
        df.loc[idx, "lotr_dialogue_words"] = merged["lotr_dialogue_words"].values

    # ── Hunger Games ─────────────────────────────────────────────────────────
    hg = _kcsv("HungerGames_Characters_Dataset_ALL.csv")
    if not hg.empty:
        hg["char_lower"] = hg["Name"].str.strip().str.lower()
        hg["hg_is_tribute"] = (hg["Tribute"].astype(str).str.lower().str.contains("yes")).astype(int)
        hg["hg_winner"]     = (hg["Winner"].astype(str).str.lower() == "yes").astype(int)
        hg["hg_is_mentor"]  = hg["Mentor"].notna().astype(int)
        hg_k = hg[["char_lower","District","hg_is_tribute","hg_winner",
                    "hg_is_mentor","Profession"]].rename(
            columns={"District":"hg_district","Profession":"hg_profession"}
        ).drop_duplicates("char_lower")
        idx, br = idx_bridge("The Hunger Games")
        merged = br.merge(hg_k, on="char_lower", how="left").reset_index(drop=True)
        for col in ["hg_district","hg_is_tribute","hg_winner","hg_is_mentor","hg_profession"]:
            df.loc[idx, col] = merged[col].values
        print(f"  HG chars: {merged['hg_district'].notna().sum()}/{len(idx)} matched")

    hgs = _kcsv("Hunger Games survival analysis data set.csv")
    if not hgs.empty:
        hgs["char_lower"] = hgs["name"].str.strip().str.lower()
        hgs_k = hgs[["char_lower","age","career","rating"]].rename(
            columns={"age":"hg_age","career":"hg_career","rating":"hg_rating"}
        ).drop_duplicates("char_lower")
        idx, br = idx_bridge("The Hunger Games")
        merged = br.merge(hgs_k, on="char_lower", how="left").reset_index(drop=True)
        for col in ["hg_age","hg_career","hg_rating"]:
            df.loc[idx, col] = merged[col].values

    # ── Dune ──────────────────────────────────────────────────────────────────
    dune = _kcsv("duneCharacters_kaggle.csv")
    if not dune.empty:
        dune = dune.drop(columns=["Born","Died","to","URL","Book","Detail"], errors="ignore")
        dune["char_lower"] = dune["Character"].str.strip().str.lower()
        dune_k = dune[["char_lower","House_Allegiance","Culture"]].rename(
            columns={"House_Allegiance":"dune_house","Culture":"dune_culture"}
        ).drop_duplicates("char_lower")
        idx, br = idx_bridge("Dune")
        merged = br.merge(dune_k, on="char_lower", how="left").reset_index(drop=True)
        for col in ["dune_house","dune_culture"]:
            df.loc[idx, col] = merged[col].values
        print(f"  Dune chars: {merged['dune_house'].notna().sum()}/{len(idx)} matched")

    # ── Marvel 92k ────────────────────────────────────────────────────────────
    marvel = _kcsv("marvel_characters_dataset.csv")
    if not marvel.empty:
        marvel = marvel.drop(columns=["Alive","URL","PageID","Unnamed: 0"], errors="ignore")
        marvel["char_lower"]    = marvel["Name"].str.strip().str.lower()
        marvel["marvel_has_teams"] = marvel["Teams"].notna().astype(int)
        marvel_k = marvel[["char_lower","Identity","Marital Status","marvel_has_teams","Origin","Universe"]].rename(
            columns={"Identity":"marvel_identity","Marital Status":"marvel_marital","Origin":"marvel_origin","Universe":"marvel_universe"}
        ).drop_duplicates("char_lower")
        idx, br = idx_bridge("Marvel Cinematic Universe")
        merged = br.merge(marvel_k, on="char_lower", how="left").reset_index(drop=True)
        for col in ["marvel_identity","marvel_marital","marvel_has_teams","marvel_origin","marvel_universe"]:
            df.loc[idx, col] = merged[col].values
        print(f"  Marvel: {merged['marvel_identity'].notna().sum()}/{len(idx)} matched")

    # ── MCU apriandito (335 chars, hero/villain category + species + media presence) ─
    # alias field is comma-separated (e.g. "Iron Man, Tony, Shellhead") — Fandom
    # uses a single name per character, so we explode all aliases into a lookup dict.
    mcu_ap = _kcsv("mcu_characters.csv")
    if not mcu_ap.empty:
        LEAKY_MCU = ["date_of_death", "status"]
        mcu_ap = mcu_ap.drop(columns=[c for c in LEAKY_MCU if c in mcu_ap.columns])
        mcu_ap["mcu_in_movie"] = mcu_ap["movie"].notna().astype(int)
        mcu_ap["mcu_in_tv"]    = mcu_ap["tv_series"].notna().astype(int)
        mcu_ap["mcu_in_comic"] = mcu_ap["comic"].notna().astype(int)
        # Build a lookup: any name variant → feature dict
        mcu_lookup = {}
        feat_cols = ["category", "species", "mcu_in_movie", "mcu_in_tv", "mcu_in_comic"]
        for _, row in mcu_ap.iterrows():
            feats = {
                "mcu_hero_villain": row.get("category"),
                "mcu_species":      row.get("species"),
                "mcu_in_movie":     row["mcu_in_movie"],
                "mcu_in_tv":        row["mcu_in_tv"],
                "mcu_in_comic":     row["mcu_in_comic"],
            }
            # Add each comma-split alias
            for raw in [row.get("alias"), row.get("real_name")]:
                if not isinstance(raw, str):
                    continue
                for part in raw.split(","):
                    key = part.strip().lower()
                    if key and key not in mcu_lookup:
                        mcu_lookup[key] = feats
        idx, br = idx_bridge("Marvel Cinematic Universe")
        matched = 0
        for feat in ["mcu_hero_villain","mcu_species","mcu_in_movie","mcu_in_tv","mcu_in_comic"]:
            df.loc[idx, feat] = None
        for pos, row_idx in enumerate(idx):
            cl = br.iloc[pos]["char_lower"]
            if cl in mcu_lookup:
                for feat, val in mcu_lookup[cl].items():
                    df.loc[row_idx, feat] = val
                matched += 1
        print(f"  MCU apriandito: {matched}/{len(idx)} matched")

    # ── Star Wars — ulrik (87 chars, vehicles/starships/skin_color) ───────────
    sw_ulrik = _kcsv("star_wars_character_dataset.csv")
    if not sw_ulrik.empty:
        sw_ulrik["char_lower"] = sw_ulrik["name"].str.strip().str.lower()
        sw_ulrik["sw_vehicles_count"]  = sw_ulrik["vehicles"].apply(
            lambda v: len(str(v).split(",")) if pd.notna(v) and str(v).strip() else 0)
        sw_ulrik["sw_starships_count"] = sw_ulrik["starships"].apply(
            lambda v: len(str(v).split(",")) if pd.notna(v) and str(v).strip() else 0)
        # Normalise skin_color to first value
        sw_ulrik["sw_skin_color"] = sw_ulrik["skin_color"].apply(
            lambda v: str(v).split(",")[0].strip().lower() if pd.notna(v) else None)
        sw_u_k = sw_ulrik[["char_lower","sw_vehicles_count","sw_starships_count","sw_skin_color"]
                           ].drop_duplicates("char_lower")
        idx, br = idx_bridge("Star Wars")
        merged = br.merge(sw_u_k, on="char_lower", how="left").reset_index(drop=True)
        for col in ["sw_vehicles_count","sw_starships_count","sw_skin_color"]:
            df.loc[idx, col] = merged[col].values
        print(f"  SW ulrik: {merged['sw_skin_color'].notna().sum()}/{len(idx)} matched")

    # ── Star Wars — kacper name lists (canon/legend/droid/creature flags) ─────
    sw_canon    = _kcsv("sw_canon_names.csv")
    sw_legend   = _kcsv("sw_legend_names.csv")
    sw_droids   = _kcsv("sw_droid_names.csv")
    sw_creature = _kcsv("sw_creature_names.csv")
    idx, br = idx_bridge("Star Wars")
    if not sw_canon.empty:
        canon_set   = set(sw_canon.iloc[:,0].str.strip().str.lower())
        df.loc[idx, "sw_is_canon"]   = br["char_lower"].isin(canon_set).astype(int).values
    if not sw_legend.empty:
        legend_set  = set(sw_legend.iloc[:,0].str.strip().str.lower())
        df.loc[idx, "sw_is_legend"]  = br["char_lower"].isin(legend_set).astype(int).values
    if not sw_droids.empty:
        droid_set   = set(sw_droids.iloc[:,0].str.strip().str.lower())
        df.loc[idx, "sw_is_droid"]   = br["char_lower"].isin(droid_set).astype(int).values
    if not sw_creature.empty:
        creature_set = set(sw_creature.iloc[:,0].str.strip().str.lower())
        df.loc[idx, "sw_is_creature"] = br["char_lower"].isin(creature_set).astype(int).values
    canon_n = (br["char_lower"].isin(canon_set).sum() if not sw_canon.empty else 0)
    print(f"  SW kacper: {canon_n} canon, "
          f"{br['char_lower'].isin(droid_set).sum() if not sw_droids.empty else 0} droids matched")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Fandom wiki metadata (batch MediaWiki API)
# ══════════════════════════════════════════════════════════════════════════════

_LEAKY_CAT = re.compile(r"deceased|dead|death|killed|died|murder", re.I)


def _parse_wiki_url(url: str):
    m = re.match(r"https?://([^/]+)/wiki/(.+)", str(url))
    if m:
        return m.group(1), unquote(m.group(2))
    return None, None


def fetch_fandom_meta(df: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    import pickle
    print("\n" + "=" * 60)
    print("STEP 7 — Fandom wiki metadata (batch API, ~25 min)")
    print("=" * 60)
    keys["wiki_domain"] = keys["page_url"].apply(lambda u: _parse_wiki_url(u)[0])
    keys["page_title"]  = keys["page_url"].apply(lambda u: _parse_wiki_url(u)[1])
    valid = keys.dropna(subset=["wiki_domain","page_title"])

    # Load partial Fandom results if a mid-run save exists
    fandom_partial = CKPT_DIR / "fandom_partial.pkl"
    if fandom_partial.exists():
        with open(fandom_partial, "rb") as fh:
            results = pickle.load(fh)
        print(f"  Resuming Fandom fetch — {len(results):,} pages already cached")
    else:
        results = {}

    wikis = valid.groupby("wiki_domain")
    total_batches = sum(-(-(len(g)) // 50) for _, g in wikis)
    done = 0

    for wiki, group in wikis:
        titles = group["page_title"].tolist()
        urls   = group["page_url"].tolist()
        t2url  = dict(zip(titles, urls))
        for i in range(0, len(titles), 50):
            batch = titles[i:i+50]
            # Skip URLs already fetched
            batch_urls = [t2url.get(t) for t in batch]
            if all(u in results for u in batch_urls if u):
                done += 1
                continue
            try:
                r = _get(f"https://{wiki}/api.php", params={
                    "action":"query","titles":"|".join(batch),
                    "prop":"info|categories","cllimit":"100","format":"json"
                }, timeout=20)
                pages = r.json().get("query", {}).get("pages", {})
                for p in pages.values():
                    if "missing" in p:
                        continue
                    title = p.get("title","")
                    cats  = [c["title"] for c in p.get("categories",[])
                             if not _LEAKY_CAT.search(c["title"])]
                    results[t2url.get(title)] = {
                        "fandom_category_count": len(cats),
                        "fandom_is_stub":        int(any("stub" in c.lower() for c in cats)),
                        "fandom_is_featured":    int(any("featured" in c.lower() or "good article" in c.lower() for c in cats)),
                        "fandom_is_recurring":   int(any("recurring" in c.lower() for c in cats)),
                    }
            except Exception as e:
                print(f"  [warn] {wiki}: {e}")
            done += 1
            if done % 100 == 0:
                print(f"  Batch {done}/{total_batches} — {len(results):,} pages fetched")
                # Save partial progress so a crash is resumable
                CKPT_DIR.mkdir(parents=True, exist_ok=True)
                with open(fandom_partial, "wb") as fh:
                    pickle.dump(results, fh, protocol=4)
            time.sleep(0.3)

    for col in ["fandom_category_count","fandom_is_stub","fandom_is_featured","fandom_is_recurring"]:
        df[col] = keys["page_url"].map(
            lambda u: results.get(u, {}).get(col, 0)).values
    print(f"  Done. {len(results):,} pages with metadata")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — Species category (computed)
# ══════════════════════════════════════════════════════════════════════════════

_ROBOT_RE   = re.compile(r"droid|robot|android|synthetic|ai|machine|terminator|cyborg", re.I)
_UNDEAD_RE  = re.compile(r"zombie|vampire|undead|ghost|lich|wraith|wight", re.I)
_MAGIC_RE   = re.compile(r"wizard|witch|elf|dwarf|hobbit|halfling|orc|goblin|ent|giant|troll|dragon|fairy|demon|angel|god|deity", re.I)
_ALIEN_RE   = re.compile(r"alien|wookiee|twi.?lek|hutt|zabrak|kree|skrull|asgardian|togruta|rodian|nautolan", re.I)
_ANIMAL_RE  = re.compile(r"\bcat\b|\bdog\b|wolf|bear|lion|eagle|hawk|serpent|snake|horse|direwolf", re.I)


def derive_species_category(row) -> str:
    text = " ".join(filter(None, [
        str(row.get("sw_species_detail") or ""),
        str(row.get("pdb_species") or ""),
        str(row.get("lotr_race") or ""),
    ])).lower()
    if _ROBOT_RE.search(text):  return "robot_ai"
    if _UNDEAD_RE.search(text): return "undead"
    if _MAGIC_RE.search(text):  return "magical_creature"
    if _ALIEN_RE.search(text):  return "alien"
    if _ANIMAL_RE.search(text): return "animal"
    if "human" in text:         return "human"
    ih = row.get("is_human")
    if ih == 1: return "human"
    if ih == 0: return "nonhuman"
    return "unknown"


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()

    # ── Auto-resume: find the latest saved checkpoint ─────────────────────────
    resume_from = 0
    df = keys = None
    for i, step in enumerate(reversed(_STEPS)):
        df, keys = _ckpt_load(step)
        if df is not None:
            resume_from = len(_STEPS) - i  # index of next step to run
            print(f"Resuming from checkpoint after step: {step}  "
                  f"(df shape: {df.shape})")
            break

    if resume_from == 0:
        # Step 1: parse JSONL
        df = parse_jsonl()

        # Load page texts for archetype/gender inference (Step 2)
        print("  Loading page texts for inference...")
        jsonl_texts = {}
        for path in sorted(RAW_DIR.glob("*_characters.jsonl")):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            r = json.loads(line)
                            url  = r.get("page_url","")
                            text = r.get("page_text","") or ""
                            if url:
                                jsonl_texts[url] = text
                        except Exception:
                            pass

        # Step 2: structural features
        df = derive_structural(df, jsonl_texts)
        del jsonl_texts  # free memory

        # Save key file (name + url bridge for external merges)
        keys = pd.DataFrame({
            "row_index":  df.index,
            "page_url":   df.get("page_url", pd.Series(dtype=str)),
            "name":       df.get("name", pd.Series(dtype=str)),
            "franchise":  df["franchise"],
            "char_lower": df.get("name", pd.Series(dtype=str)).str.strip().str.lower(),
        })
        keys_path = CLEAN_DIR / "characters_keys.csv"
        keys.to_csv(keys_path, index=False)
        print(f"\n  Key file saved: {keys_path}")

        # Drop identifiers from main df
        df = df.drop(columns=["name","page_url","wiki"], errors="ignore")
        _ckpt_save("s2_structural", df, keys)

    # Helper: should we run this step?
    def _should(step_name):
        return _STEPS.index(step_name) >= resume_from

    # Step 3: franchise APIs
    if _should("s3_apis"):
        df = merge_franchise_apis(df, keys)
        _ckpt_save("s3_apis", df, keys)

    # Step 4: TVmaze
    if _should("s4_tvmaze"):
        df = fetch_tvmaze(df, keys)
        _ckpt_save("s4_tvmaze", df, keys)

    # Step 5: CMU
    if _should("s5_cmu"):
        df = fetch_cmu(df, keys)
        _ckpt_save("s5_cmu", df, keys)

    # Step 6: Kaggle
    if _should("s6_kaggle"):
        df = merge_kaggle(df, keys)
        _ckpt_save("s6_kaggle", df, keys)

    # Step 7: Fandom metadata
    if _should("s7_fandom"):
        df = fetch_fandom_meta(df, keys)
        _ckpt_save("s7_fandom", df, keys)

    # Step 8: species category
    print("\n" + "=" * 60)
    print("STEP 8 — Species category")
    print("=" * 60)
    df["species_category"] = df.apply(derive_species_category, axis=1)
    print(f"  {df['species_category'].value_counts().to_dict()}")

    # Final cleanup
    df = df.copy()  # defragment

    # Null report
    print("\n" + "=" * 60)
    print("NULL RATE REPORT")
    print("=" * 60)
    nulls = df.isnull().sum()
    pct   = (nulls / len(df) * 100).round(1)
    report = pd.DataFrame({"null_count": nulls, "null_pct": pct})
    report = report[report["null_count"] > 0].sort_values("null_pct", ascending=False)
    print(report.to_string())

    # Save
    out = CLEAN_DIR / "characters_raw.csv"
    df.to_csv(out, index=False)
    elapsed = (time.time() - t0) / 60
    print(f"\n{'='*60}")
    print(f"DONE — characters_raw.csv saved")
    print(f"Shape: {df.shape}  |  Elapsed: {elapsed:.1f} min")
    print(f"{'='*60}")

    # Clean up checkpoints now that we have a good output
    _ckpt_clear()
    print("Checkpoints cleared.")


if __name__ == "__main__":
    main()
