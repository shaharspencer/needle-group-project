"""Shared constants for the character-mortality analysis.

## Rejection-rule policy

Every predicate here that drops or reclassifies a row (`is_non_character`,
`is_unnamed_role`, `clean_infobox_field_count`) must be a pure function of 
that row's own raw scraped fields. To prevent data leakage, they must never 
touch `is_dead`, statistics computed across other rows, or dev/test outcomes.
"""

import os
import re

# Some scraped pages are articles about an in-story object rather than a
# character who could live or die in the narrative. The Harry Potter wiki tags
# its talking paintings explicitly (infobox `theme: portrait`, e.g. "Witch in
# portrait"); the Star Wars wiki doesn't tag its background paintings but names
# them distinctively ("Unidentified Neimoidian (drapery painting)"). Left in,
# these are a free win for any death classifier: whoever a portrait depicts
# died before the portrait existed, so they are all trivially is_dead=1.
#
# The test is purely structural — an infobox field and a name pattern, never
# the death label — so removing these rows cannot leak the target.
# The wiki's own `theme: portrait` tag catches most of these but not all: a
# portrait of a headmaster is themed `hogwarts-staff`, a portrait of a mermaid
# `merperson`. So the name is checked too -- but only where the name says the
# page *is* the artwork, never where it merely mentions one. "Gorgon Portrait
# artist" is the person who painted it, and "Gryffindor Seeker whose portrait
# lost his Golden Snitch" is a person described through his portrait; both are
# real characters, and a bare /portrait/ match would delete them.
_ARTWORK_NAME_RE = re.compile(
    r"^(portraits?|paintings?|statues?|busts?|tapestr(y|ies)|photographs?|"
    r"murals?|drawings?|sketch(es)?)\s+of\b",
    re.IGNORECASE,
)
_ARTWORK_SUFFIX_RE = re.compile(
    r"\((portrait|painting|photo|statue|bust)\)$", re.IGNORECASE
)


def is_non_character(record: dict) -> bool:
    """True for scraped pages that are objects, not characters.

    Applied by every stage that reads data/raw/*.jsonl, so the row sets in
    characters_raw.csv, onscreen_flags.csv and character_features.csv agree.
    """
    infobox = record.get("infobox") or {}
    if str(infobox.get("theme", "")).strip().lower() == "portrait":
        return True
    name = str(record.get("name", ""))
    if _ARTWORK_NAME_RE.match(name) or _ARTWORK_SUFFIX_RE.search(name):
        return True
    # Star Wars doesn't theme these; it names them distinctively.
    return bool(re.match(r"^Unidentified .*\bpainting\)$", name, re.IGNORECASE))


# Wikis give background extras their own pages, titled by role rather than by
# name: "Unidentified Ugnaught worker", "Unnamed pilot". These do appear on
# screen -- TMDB credits background roles too, so they match real credits --
# but they are not characters in the sense this project measures, and the
# annotation rubric behind the corpus breakdown counts "unnamed roles" among
# the pages that are not characters at all.
#
# They are 7.2% of the on-screen set and heavily concentrated in the two
# largest wikis (34.5% of Star Wars, 19.7% of Harry Potter), and they die at
# 22.7% against 35.8% for named characters, so leaving them in drags the
# headline mortality down. Analysis restricts to named characters; the numbers
# with these rows kept are reported alongside as a sensitivity check.
_UNNAMED_ROLE_NAME_RE = re.compile(r"^(unidentified|unnamed|unknown)\b", re.IGNORECASE)


def is_unnamed_role(name: object) -> bool:
    """True for pages titled by role rather than by a character's name."""
    return bool(_UNNAMED_ROLE_NAME_RE.match(str(name or "").strip()))


# infobox_field_count is one of the two strongest features in the project, and
# part of why is circular: a character who dies picks up "Cause of death", "Date
# of death" and similar fields, so counting fields partly counts the label. These
# field names are dropped before counting. Fields that both living and dead
# characters carry -- including `status`, which is filled either way -- are kept,
# since their presence is not informative about the outcome.
_DEATH_INFOBOX_FIELD_RE = re.compile(
    r"death|died|dod|deceas|killed|kill(ed)?[ _]?by|cause|fate|posthum", re.IGNORECASE
)


def clean_infobox_field_count(infobox: dict) -> int:
    """Number of infobox fields, ignoring ones only dead characters acquire."""
    return sum(
        1 for key in (infobox or {})
        if not _DEATH_INFOBOX_FIELD_RE.search(str(key))
    )


# Set INCLUDE_UNNAMED_ROLES=1 to reproduce the sensitivity numbers quoted in the
# README, which keep the unnamed background roles in.
INCLUDE_UNNAMED_ROLES = os.environ.get("INCLUDE_UNNAMED_ROLES", "") == "1"


def analysis_population(df):
    """The rows the write-up analyses: on screen, and named unless overridden.

    Every question restricts the same way, so the counts in the README, the
    figures and the demo all describe one population.
    """
    out = df[df["is_onscreen"] == 1]
    if not INCLUDE_UNNAMED_ROLES and "is_named" in out.columns:
        out = out[out["is_named"] == 1]
    return out.copy()


# Label regimes used by the Fandom wikis.
REGIME_DOD = "dod_only"
REGIME_STATUS = "status"

# Annotation labels collected for each sampled character.
LABEL_IS_CHARACTER = "is_character"
LABEL_APPEARS_ONSCREEN = "appears_onscreen"
LABEL_DIES_IN_NARRATIVE = "dies_in_narrative"
ANNOTATION_LABELS = (
    LABEL_IS_CHARACTER,
    LABEL_APPEARS_ONSCREEN,
    LABEL_DIES_IN_NARRATIVE,
)

# Value used by annotators when a label cannot be determined from the evidence.
UNDECIDABLE = -1

ANNOTATOR_MODEL = "claude-haiku-4-5-20251001"

# Gold sample: characters drawn per (regime, label state) stratum. Weighted
# toward unlabelled rows and dod_only deaths, the two cells that determine the
# corrected label.
STRATA_TARGETS = {
    (REGIME_STATUS, "dropped"): 200,
    (REGIME_DOD, "dead"): 150,
    (REGIME_STATUS, "dead"): 100,
    (REGIME_DOD, "alive"): 75,
    (REGIME_STATUS, "alive"): 75,
}

GOLD_SAMPLE_SEED = 20260814
GOLD_EXCERPT_CHARS = 1000
GOLD_MIN_TEXT_CHARS = 80

# Infobox keys shown to the annotator verbatim.
INTERESTING_INFOBOX_KEYS = frozenset({
    "status", "death", "died", "dod", "date of death", "cause of death",
    "causeofdeath", "actor", "portrayed", "portrayed by", "played by",
    "species", "gender", "born", "birth", "first", "last", "appearances",
    "seasons", "episode", "deathep",
})

# Infobox keys indicating a filmed appearance.
ACTOR_INFOBOX_KEYS = frozenset({
    "actor", "portrayed", "portrayed by", "played by",
})

# TMDB.
TMDB_API_ROOT = "https://api.themoviedb.org/3/"
TMDB_REQUEST_SLEEP = 0.05
TMDB_MAX_DISCOVER_PAGES = 10
TMDB_RETRIES = 4
