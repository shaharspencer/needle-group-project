"""Shared constants for the character-mortality analysis."""

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
