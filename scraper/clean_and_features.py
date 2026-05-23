"""
clean_and_features.py
---------------------
Comprehensive cleaning + feature extraction pipeline.
Reads raw JSONL files, fixes known issues, extracts universal features
(including NLP features from page text), and outputs a clean CSV.

Societal features:
  - Gender inferred from pronouns when missing from infobox
  - Character archetype classified from title/occupation
  - Moral alignment signals from text (hero vs villain language)
  - Franchise era (release decade)
  - Medium (TV vs Film)
  - Is-human flag (species normalization)
  - Prominence tier from page length

Usage:
    python -m scraper.clean_and_features
"""

import json
import re
import sys
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np

HERE = Path(__file__).resolve().parent
try:
    RAW_DIR = HERE.parent / "data" / "raw"
    CLEAN_DIR = HERE.parent / "data" / "clean"
    sys.path.insert(0, str(HERE.parent.parent))
    from group_project.scraper.wiki_configs import WIKI_CONFIGS
except ImportError:
    RAW_DIR = HERE.parent / "data" / "raw"
    CLEAN_DIR = HERE.parent / "data" / "clean"
    from scraper.wiki_configs import WIKI_CONFIGS

CLEAN_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = CLEAN_DIR / "characters_v2.csv"

# ─── Franchise metadata ─────────────────────────────────────────────────────
# release_year = first major release; medium = primary medium

FRANCHISE_META = {
    "Star Wars":                     {"release_year": 1977, "medium": "Film", "genre": "Sci-Fi"},
    "Marvel Cinematic Universe":     {"release_year": 2008, "medium": "Film", "genre": "Superhero"},
    "Harry Potter":                  {"release_year": 1997, "medium": "Film", "genre": "Fantasy"},
    "Indiana Jones":                 {"release_year": 1981, "medium": "Film", "genre": "Adventure"},
    "Game of Thrones":               {"release_year": 2011, "medium": "TV",   "genre": "Fantasy"},
    "James Bond 007":                {"release_year": 1962, "medium": "Film", "genre": "Action"},
    "Alien vs Predator":             {"release_year": 1979, "medium": "Film", "genre": "Sci-Fi/Horror"},
    "DC Extended Universe":          {"release_year": 2013, "medium": "Film", "genre": "Superhero"},
    "Pirates of the Caribbean":      {"release_year": 2003, "medium": "Film", "genre": "Adventure"},
    "Jurassic Park":                 {"release_year": 1993, "medium": "Film", "genre": "Sci-Fi"},
    "Grey's Anatomy":                {"release_year": 2005, "medium": "TV",   "genre": "Drama"},
    "Lord of the Rings":             {"release_year": 1954, "medium": "Film", "genre": "Fantasy"},
    "Lost":                          {"release_year": 2004, "medium": "TV",   "genre": "Drama"},
    "Breaking Bad / Better Call Saul": {"release_year": 2008, "medium": "TV", "genre": "Crime"},
    "Dune":                          {"release_year": 1965, "medium": "Film", "genre": "Sci-Fi"},
    "Avatar":                        {"release_year": 2009, "medium": "Film", "genre": "Sci-Fi"},
    "Transformers":                  {"release_year": 1984, "medium": "Film", "genre": "Sci-Fi"},
    "Sons of Anarchy":               {"release_year": 2008, "medium": "TV",   "genre": "Crime"},
    "Supernatural":                  {"release_year": 2005, "medium": "TV",   "genre": "Fantasy"},
    "The Hunger Games":              {"release_year": 2008, "medium": "Film", "genre": "Dystopian"},
    "The Sopranos":                  {"release_year": 1999, "medium": "TV",   "genre": "Crime"},
    "The Twilight Saga":             {"release_year": 2005, "medium": "Film", "genre": "Fantasy"},
    "Spartacus":                     {"release_year": 2010, "medium": "TV",   "genre": "Historical"},
    "Dexter":                        {"release_year": 2006, "medium": "TV",   "genre": "Crime"},
    "The 100":                       {"release_year": 2014, "medium": "TV",   "genre": "Dystopian"},
    "Vikings":                       {"release_year": 2013, "medium": "TV",   "genre": "Historical"},
    "Westworld":                     {"release_year": 2016, "medium": "TV",   "genre": "Sci-Fi"},
    "Ozark":                         {"release_year": 2017, "medium": "TV",   "genre": "Crime"},
    "Fast & Furious":                {"release_year": 2001, "medium": "Film", "genre": "Action"},
    "Mission: Impossible":           {"release_year": 1996, "medium": "Film", "genre": "Action"},
    "Stranger Things":               {"release_year": 2016, "medium": "TV",   "genre": "Sci-Fi/Horror"},
    "The Boys":                      {"release_year": 2019, "medium": "TV",   "genre": "Superhero"},
    "The Walking Dead":              {"release_year": 2010, "medium": "TV",   "genre": "Horror"},
    "Prison Break":                  {"release_year": 2005, "medium": "TV",   "genre": "Action"},
    "Peaky Blinders":                {"release_year": 2013, "medium": "TV",   "genre": "Crime"},
    "The Matrix":                    {"release_year": 1999, "medium": "Film", "genre": "Sci-Fi"},
    "Boardwalk Empire":              {"release_year": 2010, "medium": "TV",   "genre": "Crime"},
    "The Wire":                      {"release_year": 2002, "medium": "TV",   "genre": "Crime"},
}


# ─── Canonical field mappings ────────────────────────────────────────────────

CANONICAL_FIELDS = {
    "real_name": "real_name", "name": "real_name", "real name": "real_name",
    "fullname": "real_name", "full name": "real_name",
    "gender": "gender", "sex": "gender",
    "species": "species", "race": "species",
    "nationality": "nationality", "citizenship": "nationality",
    "culture": "nationality", "origin": "nationality",
    "homeworld": "homeworld",
    "born": "dob", "birth": "dob", "dob": "dob", "birth date": "dob",
    "birthdate": "dob", "birth_date": "dob",
    "died": "dod", "death": "dod", "dod": "dod", "deathdate": "dod",
    "death date": "dod",
    "actor": "actor", "portrayed": "actor", "portrayer": "actor",
    "portrayed by": "actor", "voice": "actor", "voiced by": "actor",
    "affiliation": "affiliation", "loyalty": "affiliation",
    "house": "affiliation", "allegiance": "affiliation",
    "faction": "affiliation", "clan": "affiliation", "side": "affiliation",
    "title": "title", "titles": "title", "occupation": "title",
    "profession": "title", "job": "title", "rank": "title",
    "role": "title", "livelihood": "title", "class": "title",
}

_JUNK_VALUES = {"unknown", "n/a", "none", "null", "--", "--,", "?", "tbd",
                "amortal", "varies", "various", "see below", "many"}

_STAR_MARKUP_RE = re.compile(r"\*\d+px\|link=[^\s*]*")

# ─── Keyword sets for text features ─────────────────────────────────────────

_VIOLENCE_WORDS = frozenset([
    "kill", "killed", "murder", "murdered", "death", "dead", "die", "died",
    "dies", "dying", "slay", "slain", "execute", "executed", "assassinate",
    "assassinated", "sacrifice", "sacrificed", "massacre", "slaughter",
    "strangle", "strangled", "poison", "poisoned", "stab", "stabbed",
    "shot", "shoot", "behead", "beheaded", "hang", "hanged", "drown",
    "drowned", "explode", "explosion", "bomb", "destroy", "destroyed",
])

_CONFLICT_WORDS = frozenset([
    "war", "battle", "fight", "fought", "attack", "attacked", "combat",
    "siege", "invasion", "raid", "ambush", "assault", "conflict",
    "rebellion", "revolt", "uprising",
])

_LEADERSHIP_WORDS = frozenset([
    "king", "queen", "lord", "lady", "captain", "general", "commander",
    "president", "leader", "chief", "ruler", "emperor", "empress",
    "prince", "princess", "governor", "senator", "chancellor",
    "director", "boss", "head",
])

_FAMILY_WORDS = frozenset([
    "father", "mother", "son", "daughter", "brother", "sister",
    "husband", "wife", "child", "children", "parent", "sibling",
    "family", "uncle", "aunt", "cousin", "grandfather", "grandmother",
    "nephew", "niece", "married", "spouse", "wedding",
])

_HERO_WORDS = frozenset([
    "hero", "heroic", "heroism", "save", "saved", "saves", "saving",
    "rescue", "rescued", "protect", "protected", "protector",
    "defend", "defended", "defender", "brave", "bravery", "courage",
    "courageous", "noble", "honor", "honour", "honorable", "honourable",
    "loyal", "loyalty", "justice", "righteous", "ally", "allies",
    "compassion", "compassionate", "mercy", "merciful", "sacrifice",
])

_VILLAIN_WORDS = frozenset([
    "villain", "villainous", "evil", "cruel", "cruelty", "ruthless",
    "corrupt", "corruption", "betray", "betrayed", "betrayal", "traitor",
    "treachery", "treacherous", "tyrant", "tyranny", "oppressor",
    "manipulate", "manipulated", "manipulative", "deceive", "deceived",
    "deception", "scheme", "schemer", "sinister", "malicious",
    "sadistic", "merciless", "ruthless", "enemy", "enemies",
    "criminal", "crime", "crimes", "terrorize", "terrorized",
])

_POWER_WORDS = frozenset([
    "power", "powerful", "force", "authority", "command", "control",
    "influence", "dominate", "dominated", "dominance", "reign",
    "rule", "ruled", "conquer", "conquered", "empire", "throne",
    "weapon", "weapons", "army", "armies", "military", "fleet",
])

_VULNERABILITY_WORDS = frozenset([
    "captured", "imprisoned", "tortured", "wounded", "injured",
    "helpless", "vulnerable", "trapped", "prisoner", "hostage",
    "kidnapped", "abducted", "enslaved", "slave", "suffering",
    "trauma", "traumatic", "grief", "loss", "lost", "fear",
    "afraid", "desperate", "despair", "exile", "exiled", "banished",
])


# ─── Character archetype from title ─────────────────────────────────────────

_ARCHETYPE_RULES = [
    ("Military",     {"soldier", "private", "corporal", "sergeant", "lieutenant",
                      "captain", "major", "colonel", "general", "admiral",
                      "commander", "officer", "pilot", "trooper", "marine",
                      "guard", "warrior", "fighter", "ranger", "clone trooper",
                      "stormtrooper", "legionary", "gladiator"}),
    ("Royalty",      {"king", "queen", "prince", "princess", "emperor", "empress",
                      "lord", "lady", "ser", "sir", "duke", "duchess", "baron",
                      "count", "countess", "heir", "monarch", "regent"}),
    ("Criminal",     {"criminal", "mobster", "gangster", "thief", "smuggler",
                      "drug dealer", "hitman", "assassin", "pirate", "outlaw",
                      "henchman", "mercenary", "bounty hunter", "enforcer",
                      "crime lord", "drug lord", "sith lord", "dark lord",
                      "crime boss", "thug", "bandit", "raider"}),
    ("Force User",   {"jedi", "sith", "force-sensitive", "padawan",
                      "jedi master", "jedi knight", "dark side"}),
    ("Law/Order",    {"detective", "agent", "sheriff", "marshal", "police",
                      "cop", "fbi", "cia", "spy", "intelligence operative",
                      "investigator", "auror", "judge", "lawyer", "attorney",
                      "law enforcement", "constable"}),
    ("Medical",      {"doctor", "m.d.", "nurse", "healer", "medic", "surgeon",
                      "physician", "paramedic"}),
    ("Academic",     {"professor", "scientist", "researcher", "scholar",
                      "archaeologist", "author", "teacher", "student", "prefect",
                      "historian", "librarian", "inventor"}),
    ("Political",    {"president", "senator", "governor", "chancellor",
                      "minister", "mayor", "politician", "ambassador",
                      "diplomat", "councilor", "grand moff", "moff",
                      "viceroy", "magistrate"}),
    ("Religious",    {"priest", "priestess", "monk", "nun", "reverend",
                      "bishop", "septon", "septa", "maester", "oracle",
                      "cleric", "shaman", "witch"}),
    ("Worker",       {"farmer", "engineer", "mechanic", "miner", "builder",
                      "cook", "bartender", "driver", "firefighter",
                      "technician", "craftsman", "blacksmith", "trader",
                      "merchant", "shopkeeper"}),
    ("Entertainer",  {"actor", "musician", "singer", "dancer", "artist",
                      "reporter", "journalist", "writer", "photographer",
                      "entertainer", "performer"}),
]

# Text-based archetype inference patterns (applied to first 500 words of page text)
_TEXT_ARCHETYPE_PATTERNS = [
    ("Military",   re.compile(r"\b(soldier|officer|served in the (?:army|navy|military)|trooper|fought (?:in|during) the|enlisted|military career|commanding officer|squadron|battalion|regiment)\b", re.I)),
    ("Royalty",    re.compile(r"\b(king|queen|prince|princess|emperor|empress|lord of|lady of|ruled|reign(?:ed)?|royal family|heir to the throne|noble(?:man|woman)?|aristocrat)\b", re.I)),
    ("Criminal",   re.compile(r"\b(criminal|crime (?:lord|boss|family)|drug (?:lord|dealer|trade)|smuggl|assassin|bounty hunter|pirate|thief|gangster|mobster|hitman|sith lord|dark lord|dark side)\b", re.I)),
    ("Force User", re.compile(r"\b(jedi|sith|force[- ]sensitive|padawan|lightsaber|the force|dark side of the force|jedi (?:master|knight|council)|sith (?:lord|apprentice))\b", re.I)),
    ("Law/Order",  re.compile(r"\b(detective|police|law enforcement|agent of|special agent|sheriff|marshal|investigat|auror|spy|espionage|intelligence)\b", re.I)),
    ("Medical",    re.compile(r"\b(doctor|physician|surgeon|nurse|medic(?:al)?|healer|hospital|medical (?:school|practice|career))\b", re.I)),
    ("Academic",   re.compile(r"\b(professor|scientist|researcher|scholar|archaeolog|teacher|student at|studied at|university|academy|hogwarts)\b", re.I)),
    ("Political",  re.compile(r"\b(senator|governor|chancellor|president of|politician|ambassador|diplomat|political career|elected|council(?:lor|man|woman))\b", re.I)),
    ("Religious",  re.compile(r"\b(priest|priestess|monk|nun|temple|religious|cleric|shaman|spiritual leader|septon|septa)\b", re.I)),
    ("Worker",     re.compile(r"\b(engineer|mechanic|farmer|miner|builder|technician|craftsman|blacksmith|merchant|trader|bartender|shopkeeper)\b", re.I)),
    ("Entertainer", re.compile(r"\b(reporter|journalist|musician|singer|dancer|artist|performer|actor|actress|writer|author)\b", re.I)),
]


def classify_archetype(title):
    if not isinstance(title, str) or not title.strip():
        return "Unknown"
    title_lower = title.lower()
    for archetype, keywords in _ARCHETYPE_RULES:
        for kw in keywords:
            if kw in title_lower:
                return archetype
    return "Other"


def classify_archetype_from_text(page_text, first_n_chars=2000):
    """Fallback archetype classification from page text when title is missing."""
    if not page_text:
        return "Unknown"
    snippet = page_text[:first_n_chars].lower()
    scores = {}
    for archetype, pattern in _TEXT_ARCHETYPE_PATTERNS:
        hits = len(pattern.findall(snippet))
        if hits > 0:
            scores[archetype] = hits
    if not scores:
        return "Unknown"
    return max(scores, key=scores.get)


# ─── Gender inference from pronouns ──────────────────────────────────────────

_MALE_PRONOUNS = frozenset(["he", "him", "his", "himself"])
_FEMALE_PRONOUNS = frozenset(["she", "her", "hers", "herself"])


def infer_gender_from_text(page_text, first_n_words=500):
    """Infer gender from pronoun usage in the first N words of the page."""
    if not page_text:
        return None
    words = page_text.lower().split()[:first_n_words]
    male_count = sum(1 for w in words if w in _MALE_PRONOUNS)
    female_count = sum(1 for w in words if w in _FEMALE_PRONOUNS)
    # Relaxed threshold: 2+ mentions and 1.5x ratio
    if male_count >= 2 and male_count > female_count * 1.5:
        return "Male"
    if female_count >= 2 and female_count > male_count * 1.5:
        return "Female"
    return None


# ─── Species normalization ───────────────────────────────────────────────────

_HUMAN_SPECIES = frozenset([
    "human", "humans", "homo sapiens", "man", "woman", "human/mutate",
    "human/inhuman", "enhanced human", "augmented human",
    "men", "numenorean", "dunedain", "gondorian", "rohirrim",
    "hobbit", "hobbits",
])

# Franchises set entirely in the real world — every character is human
_HUMAN_ONLY_FRANCHISES = frozenset([
    "Game of Thrones", "James Bond 007", "Grey's Anatomy", "Lost",
    "Breaking Bad / Better Call Saul", "Sons of Anarchy", "The Sopranos",
    "Vikings", "Dexter", "Ozark", "Peaky Blinders", "Prison Break",
    "Boardwalk Empire", "The Wire", "Spartacus", "The Hunger Games",
    "Indiana Jones", "Fast & Furious", "Mission: Impossible",
])


def is_human_species(species, franchise=None):
    if isinstance(species, str):
        s = species.lower().strip()
        if s in _HUMAN_SPECIES:
            return 1
        return 0
    # No species data — infer from franchise
    if franchise in _HUMAN_ONLY_FRANCHISES:
        return 1
    return None


# ─── Affiliation moral alignment ─────────────────────────────────────────────

_GOOD_AFFILIATIONS = re.compile(
    r"(rebel|rebellion|alliance|republic|resistance|jedi|gryffindor|hufflepuff|"
    r"ravenclaw|order of the phoenix|avengers|s\.h\.i\.e\.l\.d|justice league|"
    r"the 100|skaikru|night.?s watch|stark|ministry of magic|rebel alliance|"
    r"order of merlin|dumbledore|free folk|autobots?)",
    re.I
)
_EVIL_AFFILIATIONS = re.compile(
    r"(empire|imperial|sith|dark side|galactic empire|first order|separatist|"
    r"death eater|slytherin|hydra|ten rings|nihil|crime|criminal|cartel|"
    r"confederation|confederacy|dark|evil|tyrell|lannister|"
    r"weyland.?yutani|decepticons?|white walkers?|voldemort|"
    r"eternal empire|lost tribe|black sun|hutt|jabba|cersei)",
    re.I
)


def classify_affiliation_alignment(affiliation):
    if not isinstance(affiliation, str) or not affiliation.strip():
        return "Unknown"
    if _GOOD_AFFILIATIONS.search(affiliation):
        if _EVIL_AFFILIATIONS.search(affiliation):
            return "Ambiguous"
        return "Good"
    if _EVIL_AFFILIATIONS.search(affiliation):
        return "Evil"
    return "Neutral"


# ─── Title cleanup ───────────────────────────────────────────────────────────

_TITLE_JUNK_RE = re.compile(
    r"(\s*\(formerly\)|\'{2,}|Pre-Apocalypse|Post-Apocalypse)",
    re.I
)


def clean_title(title):
    if not isinstance(title, str):
        return title
    cleaned = _TITLE_JUNK_RE.sub("", title).strip()
    if cleaned.lower() in ("unknown", "n/a", "none", "", "?"):
        return None
    return cleaned


# ─── Helpers ─────────────────────────────────────────────────────────────────

def first_bullet_value(text):
    if not isinstance(text, str):
        return text
    text = _STAR_MARKUP_RE.sub("", text)
    parts = [p.strip() for p in text.split("*") if p.strip()]
    return parts[0] if parts else None


def extract_primary_color(text):
    if not isinstance(text, str):
        return None
    text = text.lower()
    colors = ['blonde', 'blond', 'brown', 'black', 'white', 'red', 'blue',
              'green', 'grey', 'gray', 'hazel', 'silver', 'gold', 'yellow',
              'purple', 'pink', 'pale', 'fair']
    for color in colors:
        if color in text:
            if color == 'blond': return 'blonde'
            if color == 'gray': return 'grey'
            return color
    return 'other'


def fix_concatenated_title(title):
    if not isinstance(title, str):
        return title
    return re.sub(r'([a-z])([A-Z])', r'\1, \2', title)


def count_keyword_hits(text_lower, wordset):
    words = text_lower.split()
    return sum(1 for w in words if w in wordset)


# ─── Text feature extraction ────────────────────────────────────────────────

_DIALOGUE_RE = re.compile(r'"[^"]{5,}"')
_QUOTE_RE = re.compile(r'[""“”][^""“”]{5,}[""“”]')
_AGE_RE = re.compile(r"\b(\d{1,3})[\s-]?year[\s-]?old\b", re.I)
_RELATIONSHIP_RE = re.compile(
    r"\b(father|mother|son|daughter|brother|sister|husband|wife|spouse|"
    r"uncle|aunt|cousin|nephew|niece|grandfather|grandmother|lover|"
    r"friend|mentor|apprentice|partner|ally|rival|enemy|companion)\s+of\b",
    re.I
)
_PROPER_NAME_RE = re.compile(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b")


def extract_text_features(page_text):
    """Extract NLP + societal features from page text."""
    empty = {
        "word_count": 0, "sentence_count": 0, "paragraph_count": 0,
        "section_count": 0,
        "violence_word_count": 0, "conflict_word_count": 0,
        "leadership_word_count": 0, "family_word_count": 0,
        "hero_word_count": 0, "villain_word_count": 0,
        "power_word_count": 0, "vulnerability_word_count": 0,
        "has_biography_section": 0, "has_death_section": 0,
        "unique_word_ratio": 0.0, "avg_sentence_length": 0.0,
        "dialogue_count": 0, "has_dialogue": 0,
        "relationship_mentions": 0,
        "named_characters_mentioned": 0,
        "age_mentioned": 0, "age_value": None,
        "is_described_young": 0, "is_described_old": 0,
    }
    if not page_text:
        return empty

    text_lower = page_text.lower()
    words = text_lower.split()
    word_count = len(words)

    sentences = re.split(r'[.!?]+', page_text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    sentence_count = len(sentences)

    paragraphs = [p.strip() for p in page_text.split('\n\n') if p.strip()]
    paragraph_count = len(paragraphs)

    lines = page_text.split('\n')
    section_headers = [l.strip() for l in lines
                       if len(l.strip()) < 60 and l.strip()
                       and l.strip()[0].isupper()
                       and not l.strip().endswith('.')
                       and len(l.strip().split()) <= 5]
    section_count = len(section_headers)

    has_biography = int(any('biography' in h.lower() for h in section_headers))
    has_death_section = int(any(
        w in h.lower() for h in section_headers
        for w in ['death', 'demise', 'fate', 'murder', 'assassination', 'killed']
    ))

    unique_words = set(words)
    unique_word_ratio = len(unique_words) / word_count if word_count > 0 else 0.0
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0.0

    # Dialogue detection
    dialogue_matches = _DIALOGUE_RE.findall(page_text) + _QUOTE_RE.findall(page_text)
    dialogue_count = len(dialogue_matches)

    # Relationship mentions ("father of", "rival of", etc.)
    relationship_mentions = len(_RELATIONSHIP_RE.findall(page_text))

    # Named characters mentioned (proxy for social connectedness)
    named_chars = set(_PROPER_NAME_RE.findall(page_text))
    named_characters_mentioned = len(named_chars)

    # Age extraction
    age_match = _AGE_RE.search(page_text)
    age_value = int(age_match.group(1)) if age_match else None
    age_mentioned = 1 if age_value is not None else 0

    # Young/old descriptors
    young_words = {"young", "youth", "teenage", "teenager", "adolescent", "child", "boy", "girl", "infant", "baby"}
    old_words = {"old", "elderly", "aged", "elder", "ancient", "veteran", "retired", "senior"}
    first_500 = " ".join(words[:500])
    is_described_young = int(any(w in first_500 for w in young_words))
    is_described_old = int(any(f" {w} " in f" {first_500} " for w in old_words))

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "paragraph_count": paragraph_count,
        "section_count": section_count,
        "violence_word_count": count_keyword_hits(text_lower, _VIOLENCE_WORDS),
        "conflict_word_count": count_keyword_hits(text_lower, _CONFLICT_WORDS),
        "leadership_word_count": count_keyword_hits(text_lower, _LEADERSHIP_WORDS),
        "family_word_count": count_keyword_hits(text_lower, _FAMILY_WORDS),
        "hero_word_count": count_keyword_hits(text_lower, _HERO_WORDS),
        "villain_word_count": count_keyword_hits(text_lower, _VILLAIN_WORDS),
        "power_word_count": count_keyword_hits(text_lower, _POWER_WORDS),
        "vulnerability_word_count": count_keyword_hits(text_lower, _VULNERABILITY_WORDS),
        "has_biography_section": has_biography,
        "has_death_section": has_death_section,
        "unique_word_ratio": round(unique_word_ratio, 4),
        "avg_sentence_length": round(avg_sentence_length, 2),
        "dialogue_count": dialogue_count,
        "has_dialogue": int(dialogue_count > 0),
        "relationship_mentions": relationship_mentions,
        "named_characters_mentioned": named_characters_mentioned,
        "age_mentioned": age_mentioned,
        "age_value": age_value,
        "is_described_young": is_described_young,
        "is_described_old": is_described_old,
    }


# ─── Record flattening ──────────────────────────────────────────────────────

def extract_flat(record, config):
    infobox = record.get("infobox", {}) or {}
    field_map = config.get("field_map", {})
    page_text = record.get("page_text", "") or ""

    flat = {
        "wiki": record.get("wiki", ""),
        "franchise": record.get("franchise", ""),
        "content_rating": record.get("content_rating", ""),
        "name": record.get("name", ""),
        "page_url": record.get("page_url", ""),
        "is_dead": record.get("is_dead"),
        "real_name": None, "gender": None, "species": None,
        "nationality": None, "homeworld": None,
        "dob": None, "dod": None,
        "actor": None, "affiliation": None, "title": None,
        "infobox_field_count": len(infobox),
        "page_text_length": len(page_text),
        "has_dob": 0, "has_dod": 0, "has_causeofdeath": 0,
        "has_family": 0, "has_image": 0, "has_alias": 0,
        "has_pronouns": 0, "appearance_count": 0,
        "hair_color": None, "eye_color": None,
    }

    # Map infobox fields
    for raw_key, raw_val in infobox.items():
        lk = raw_key.lower().strip()
        canonical = field_map.get(lk) or field_map.get(raw_key.strip())
        if not canonical:
            canonical = CANONICAL_FIELDS.get(lk)
        if canonical and canonical in flat and flat[canonical] is None:
            flat[canonical] = raw_val[:300]

    for col in ("real_name", "species", "affiliation", "title", "nationality", "actor"):
        if flat.get(col):
            flat[col] = first_bullet_value(flat[col])

    if flat.get("title"):
        flat["title"] = fix_concatenated_title(flat["title"])
        flat["title"] = clean_title(flat["title"])

    # Clean affiliation junk
    if flat.get("affiliation"):
        aff = flat["affiliation"]
        if _TITLE_JUNK_RE.search(aff):
            flat["affiliation"] = _TITLE_JUNK_RE.sub("", aff).strip() or None

    for col in ("dob", "dod"):
        val = flat.get(col)
        if val and val.strip().lower() in _JUNK_VALUES:
            flat[col] = None

    if flat.get("dob"): flat["has_dob"] = 1
    if flat.get("dod"): flat["has_dod"] = 1

    cod_keys = {"causeofdeath", "cause of death", "cause", "death cause",
                "manner of death", "deathby", "death by", "killedby",
                "killed by", "death reason", "deathreason"}
    for raw_key in infobox:
        if raw_key.lower().strip() in cod_keys and infobox[raw_key]:
            flat["has_causeofdeath"] = 1
            break

    # Normalise gender from infobox
    g = (flat.get("gender") or "").lower().strip()
    if g in ("male", "m", "masculine"):
        flat["gender"] = "Male"
    elif g in ("female", "f", "feminine"):
        flat["gender"] = "Female"
    elif g:
        flat["gender"] = "Other/Unknown"
    else:
        flat["gender"] = None

    # Gender inference from pronouns when infobox is missing
    if flat["gender"] is None:
        inferred = infer_gender_from_text(page_text)
        if inferred:
            flat["gender"] = inferred
            flat["gender_source"] = "inferred"
        else:
            flat["gender_source"] = "unknown"
    else:
        flat["gender_source"] = "infobox"

    # Physical traits
    hair_raw = eye_raw = None
    for raw_key, val in infobox.items():
        lk = raw_key.lower().strip()
        if lk in ("hair", "hair color", "hair colour", "haircolor"):
            hair_raw = val
        elif lk in ("eyes", "eye color", "eye colour", "eyecolor", "eye"):
            eye_raw = val
    if hair_raw: flat["hair_color"] = extract_primary_color(hair_raw)
    if eye_raw: flat["eye_color"] = extract_primary_color(eye_raw)

    # Binary flags from infobox keys
    family_keys = {"children", "siblings", "families", "parents", "partners",
                   "family", "spouse", "husband", "wife", "brother", "sister",
                   "mother", "father", "son", "daughter", "relatives", "relationships"}
    alias_keys = {"alias", "aliases", "nickname", "nicknames", "aka",
                  "other names", "also known as"}
    appearance_keys = {"tv series", "movie", "game", "book", "comic", "first",
                       "last", "appearances", "seasons", "episodes", "films",
                       "series", "appears in"}

    app_count = 0
    for raw_key in infobox:
        lk = raw_key.lower().strip()
        val = infobox[raw_key]
        if lk in family_keys and val: flat["has_family"] = 1
        if lk in alias_keys and val: flat["has_alias"] = 1
        if lk == "image" and val: flat["has_image"] = 1
        if lk == "pronouns" and val: flat["has_pronouns"] = 1
        if lk in appearance_keys and val: app_count += 1
    flat["appearance_count"] = app_count

    # ── Societal features ────────────────────────────────────────────────

    # Species normalization (with franchise-based inference)
    flat["is_human"] = is_human_species(flat.get("species"), flat.get("franchise"))

    # Affiliation moral alignment
    flat["affiliation_alignment"] = classify_affiliation_alignment(flat.get("affiliation"))

    # Character archetype: title-based, then text-based fallback
    title_archetype = classify_archetype(flat.get("title"))
    if title_archetype in ("Unknown", "Other"):
        text_archetype = classify_archetype_from_text(page_text)
        if text_archetype != "Unknown":
            flat["archetype"] = text_archetype
            flat["archetype_source"] = "text"
        else:
            flat["archetype"] = title_archetype
            flat["archetype_source"] = "none"
    else:
        flat["archetype"] = title_archetype
        flat["archetype_source"] = "title"

    # Franchise metadata
    fmeta = FRANCHISE_META.get(flat["franchise"], {})
    flat["franchise_release_year"] = fmeta.get("release_year")
    flat["franchise_decade"] = (
        f"{(fmeta['release_year'] // 10) * 10}s" if fmeta.get("release_year") else None
    )
    flat["medium"] = fmeta.get("medium")
    flat["genre"] = fmeta.get("genre")

    # Text-derived features
    text_feats = extract_text_features(page_text)
    flat.update(text_feats)

    # Keyword densities (per 1000 words)
    wc = text_feats["word_count"]
    if wc > 0:
        flat["violence_density"] = round(text_feats["violence_word_count"] / wc * 1000, 2)
        flat["conflict_density"] = round(text_feats["conflict_word_count"] / wc * 1000, 2)
        flat["leadership_density"] = round(text_feats["leadership_word_count"] / wc * 1000, 2)
        flat["family_density"] = round(text_feats["family_word_count"] / wc * 1000, 2)
        flat["hero_density"] = round(text_feats["hero_word_count"] / wc * 1000, 2)
        flat["villain_density"] = round(text_feats["villain_word_count"] / wc * 1000, 2)
        flat["power_density"] = round(text_feats["power_word_count"] / wc * 1000, 2)
        flat["vulnerability_density"] = round(text_feats["vulnerability_word_count"] / wc * 1000, 2)
        flat["dialogue_density"] = round(text_feats["dialogue_count"] / wc * 1000, 2)
        flat["relationship_density"] = round(text_feats["relationship_mentions"] / wc * 1000, 2)
        flat["social_connectedness"] = round(text_feats["named_characters_mentioned"] / wc * 1000, 2)
    else:
        for d in ("violence_density", "conflict_density", "leadership_density",
                   "family_density", "hero_density", "villain_density",
                   "power_density", "vulnerability_density",
                   "dialogue_density", "relationship_density", "social_connectedness"):
            flat[d] = 0.0

    # Moral alignment score: hero_density - villain_density
    flat["moral_alignment"] = round(flat["hero_density"] - flat["villain_density"], 2)

    # Power vs vulnerability score
    flat["power_vulnerability_ratio"] = round(
        flat["power_density"] - flat["vulnerability_density"], 2
    )

    return flat


# ─── Filtering ───────────────────────────────────────────────────────────────

BROKEN_FRANCHISES = {"Boardwalk Empire", "The Wire"}

MIN_NAME_LENGTH = 3
MIN_PAGE_TEXT_LENGTH = 100

_UNNAMED_RE = re.compile(r"^(unnamed|unknown|unidentified)\b", re.IGNORECASE)
_JUNK_NAME_RE = re.compile(
    r"^(list of|category:|template:|file:|talk:|minor characters)",
    re.IGNORECASE
)


def is_valid_character(row):
    name = row.get("name", "")
    if len(name) <= MIN_NAME_LENGTH:
        return False
    if _UNNAMED_RE.match(name):
        return False
    if _JUNK_NAME_RE.match(name):
        return False
    if row.get("page_text_length", 0) < MIN_PAGE_TEXT_LENGTH:
        return False
    if row.get("franchise") in BROKEN_FRANCHISES:
        return False
    return True


# ─── Post-processing (DataFrame-level features) ─────────────────────────────

def add_derived_features(df):
    """Features that require the full DataFrame for context."""
    # Prominence tier (quartile within franchise)
    df["prominence_tier"] = (
        df.groupby("franchise")["word_count"]
        .transform(lambda x: pd.qcut(x, q=4, labels=["Minor", "Supporting", "Major", "Lead"],
                                       duplicates="drop"))
    )

    # Franchise-level mortality rate (contextual feature)
    franchise_mort = df.groupby("franchise")["is_dead"].transform("mean")
    df["franchise_mortality_rate"] = franchise_mort.round(4)

    # Gender ratio per franchise
    gender_counts = df.groupby(["franchise", "gender"]).size().unstack(fill_value=0)
    if "Male" in gender_counts.columns and "Female" in gender_counts.columns:
        gender_ratio = (gender_counts["Female"] /
                        (gender_counts["Male"] + gender_counts["Female"]))
        df["franchise_female_ratio"] = df["franchise"].map(gender_ratio).round(4)
    else:
        df["franchise_female_ratio"] = np.nan

    # Franchise size and balancing weight (inverse frequency)
    franchise_sizes = df["franchise"].value_counts()
    median_size = franchise_sizes.median()
    df["franchise_weight"] = df["franchise"].map(
        lambda f: round(min(median_size / franchise_sizes[f], 5.0), 4)
    )

    # Characters per franchise (useful for filtering small franchises)
    df["franchise_size"] = df["franchise"].map(franchise_sizes)

    # Era grouping for societal analysis
    era_map = {
        "1950s": "Classic (pre-1980)",
        "1960s": "Classic (pre-1980)",
        "1970s": "Classic (pre-1980)",
        "1980s": "Blockbuster (1980-1999)",
        "1990s": "Blockbuster (1980-1999)",
        "2000s": "Peak TV (2000-2009)",
        "2010s": "Streaming (2010+)",
    }
    df["franchise_era"] = df["franchise_decade"].map(era_map)

    # Missing data score — how many key infobox fields are NaN
    key_fields = ["gender", "species", "affiliation", "title", "dob", "dod", "actor"]
    df["missing_field_count"] = df[key_fields].isna().sum(axis=1)
    df["info_completeness"] = ((len(key_fields) - df["missing_field_count"]) / len(key_fields)).round(4)

    # Non-human mortality differential per franchise (societal "othering" signal)
    def _nonhuman_mort_diff(group):
        human = group[group["is_human"] == 1]["is_dead"].mean()
        nonhuman = group[group["is_human"] == 0]["is_dead"].mean()
        if pd.notna(human) and pd.notna(nonhuman):
            return round(nonhuman - human, 4)
        return np.nan
    nh_diff = df.groupby("franchise").apply(_nonhuman_mort_diff)
    df["franchise_othering_index"] = df["franchise"].map(nh_diff)

    return df


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    all_records = []

    for jsonl_path in sorted(RAW_DIR.glob("*_characters.jsonl")):
        wiki_key = jsonl_path.stem.replace("_characters", "")
        config = WIKI_CONFIGS.get(wiki_key, {})

        rows = []
        try:
            with open(jsonl_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rows.append(json.loads(line))
        except Exception as e:
            print(f"  [{wiki_key}] Read error: {e}")
            continue

        flat_rows = [extract_flat(r, config) for r in rows]
        all_records.extend(flat_rows)
        print(f"  [{wiki_key:20s}] {len(flat_rows):6d} records")

    print(f"\nTotal raw records: {len(all_records):,}")

    df = pd.DataFrame(all_records)

    # Dedup
    before = len(df)
    df = df.drop_duplicates(subset=["page_url"])
    print(f"After URL dedup: {len(df):,} (removed {before - len(df):,})")
    before = len(df)
    df = df.drop_duplicates(subset=["franchise", "name"])
    print(f"After franchise+name dedup: {len(df):,} (removed {before - len(df):,})")

    # Filter: labelled only
    df_labelled = df[df["is_dead"].notna()].copy()
    df_labelled["is_dead"] = df_labelled["is_dead"].astype(int)

    # Quality filters
    before = len(df_labelled)
    mask = df_labelled.apply(is_valid_character, axis=1)
    df_clean = df_labelled[mask].copy()
    print(f"\nAfter quality filters: {len(df_clean):,} (removed {before - len(df_clean):,})")

    # Derived features
    df_clean = add_derived_features(df_clean)

    # Summary
    print(f"\n{'='*60}")
    print(f"Final dataset: {len(df_clean):,} characters")
    print(f"  Dead:  {(df_clean.is_dead==1).sum():,} ({(df_clean.is_dead==1).mean()*100:.1f}%)")
    print(f"  Alive: {(df_clean.is_dead==0).sum():,} ({(df_clean.is_dead==0).mean()*100:.1f}%)")
    print(f"  Franchises: {df_clean.franchise.nunique()}")
    print(f"{'='*60}")

    # Gender coverage after inference
    gender_known = df_clean["gender"].notna().sum()
    print(f"\nGender coverage: {gender_known:,}/{len(df_clean):,} ({gender_known/len(df_clean)*100:.1f}%)")
    print(f"  From infobox:  {(df_clean.gender_source=='infobox').sum():,}")
    print(f"  From pronouns: {(df_clean.gender_source=='inferred').sum():,}")
    print(f"  Still unknown: {(df_clean.gender_source=='unknown').sum():,}")

    # Archetype distribution
    print(f"\nArchetype distribution:")
    print(df_clean["archetype"].value_counts().to_string())

    # Genre distribution
    print(f"\nGenre distribution:")
    print(df_clean["genre"].value_counts().to_string())

    # Medium split
    print(f"\nMedium: {df_clean['medium'].value_counts().to_dict()}")

    # Per-franchise summary
    print("\nPer-franchise:")
    summary = (
        df_clean.groupby("franchise")["is_dead"]
        .agg(total="count", dead="sum")
        .assign(
            alive=lambda x: x["total"] - x["dead"],
            pct_dead=lambda x: (x["dead"] / x["total"] * 100).round(1)
        )
        .sort_values("total", ascending=False)
    )
    print(summary.to_string())

    # Column order
    col_order = [
        # Identity
        "wiki", "franchise", "content_rating", "genre", "medium",
        "franchise_release_year", "franchise_decade", "franchise_era",
        "name", "real_name", "gender", "gender_source", "species", "is_human",
        "nationality", "homeworld", "dob", "dod",
        "actor", "affiliation", "affiliation_alignment",
        "title", "archetype", "archetype_source",
        # Target
        "is_dead",
        # Binary features
        "has_dob", "has_dod", "has_causeofdeath", "has_family",
        "has_image", "has_alias", "has_pronouns",
        "has_biography_section", "has_death_section",
        "has_dialogue",
        "is_described_young", "is_described_old",
        "age_mentioned", "age_value",
        # Numeric features
        "appearance_count", "infobox_field_count",
        "page_text_length", "word_count", "sentence_count",
        "paragraph_count", "section_count",
        "avg_sentence_length", "unique_word_ratio",
        "dialogue_count", "relationship_mentions",
        "named_characters_mentioned",
        # Keyword counts
        "violence_word_count", "conflict_word_count",
        "leadership_word_count", "family_word_count",
        "hero_word_count", "villain_word_count",
        "power_word_count", "vulnerability_word_count",
        # Keyword densities (per 1000 words)
        "violence_density", "conflict_density",
        "leadership_density", "family_density",
        "hero_density", "villain_density",
        "power_density", "vulnerability_density",
        "dialogue_density", "relationship_density",
        "social_connectedness",
        # Composite scores
        "moral_alignment", "power_vulnerability_ratio",
        # Prominence
        "prominence_tier",
        # Franchise-level contextual
        "franchise_mortality_rate", "franchise_female_ratio",
        "franchise_size", "franchise_weight", "franchise_othering_index",
        # Data completeness
        "missing_field_count", "info_completeness",
        # Physical
        "hair_color", "eye_color",
        # Meta
        "page_url",
    ]
    col_order = [c for c in col_order if c in df_clean.columns]
    df_clean[col_order].to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"\nSaved {len(df_clean):,} rows x {len(col_order)} columns to {OUT_PATH}")


if __name__ == "__main__":
    main()
