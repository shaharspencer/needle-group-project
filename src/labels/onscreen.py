"""Decide which scraped pages are characters who actually appear on screen.

A single infobox field is not enough. "has an actor field" recalls only 0.61 of
on-screen characters overall and 0.31 on the dod_only wikis, and matching names
against TMDB cast lists recalls 0.58. Both signals are precise but partial, so
this trains a classifier on the annotated sample using them together with
cheap page features.
"""

import re
import unicodedata
from collections import defaultdict

import pandas as pd

from src.constants import ACTOR_INFOBOX_KEYS
from src.paths import CLEAN_DIR

# Words that only appear on pages sourced from unfilmed material, and words
# that mark a filmed appearance.
UNFILMED_TERMS = [
    "novel", "comic", "video game", "book", "legends", "audiobook",
    "short story", "manga", "roleplaying", "sourcebook",
]
FILMED_TERMS = [
    "episode", "season", "film", "movie", "portrayed", "series",
]
APPEARANCE_KEYS = frozenset({
    "seasons", "season", "episode", "episodes", "appearances", "first",
    "last", "films", "film", "movie", "series", "appears in", "deathep",
})

NAME_STOPWORDS = frozenset({
    "the", "a", "of", "sir", "lord", "lady", "king", "queen", "dr", "mr",
    "mrs", "ms", "captain", "general", "young", "old", "voice", "uncredited",
})


class OnscreenFeatures:
    """Feature extraction shared by training and corpus-wide scoring."""

    _cast_index: dict[str, list[set[str]]] | None = None

    @staticmethod
    def normalise(text: str) -> str:
        text = unicodedata.normalize("NFKD", str(text))
        text = "".join(c for c in text if not unicodedata.combining(c)).lower()
        text = re.sub(r"[’'\"]", "", text)
        text = re.sub(r"\([^)]*\)", " ", text)
        text = re.sub(r"[^a-z0-9 ]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def name_tokens(name: str) -> set[str]:
        return {
            t for t in OnscreenFeatures.normalise(name).split()
            if t not in NAME_STOPWORDS and len(t) > 2
        }

    @staticmethod
    def cast_index() -> dict[str, list[set[str]]]:
        """Franchise -> token sets of every credited character name."""
        if OnscreenFeatures._cast_index is not None:
            return OnscreenFeatures._cast_index

        path = CLEAN_DIR / "tmdb_cast.csv"
        index: dict[str, list[set[str]]] = defaultdict(list)
        if path.exists():
            cast = pd.read_csv(path)
            for franchise, character in zip(cast["franchise"], cast["character"]):
                tokens = OnscreenFeatures.name_tokens(character)
                if tokens:
                    index[franchise].append(tokens)

        OnscreenFeatures._cast_index = index
        return index

    @staticmethod
    def tmdb_match(name: str, franchise: str) -> int:
        """1 if the name matches a credited TMDB character in that franchise."""
        tokens = OnscreenFeatures.name_tokens(name)
        if not tokens:
            return 0
        for credited in OnscreenFeatures.cast_index().get(franchise, ()):
            if tokens <= credited or credited <= tokens:
                return 1
        return 0

    @staticmethod
    def build(name: str, franchise: str, infobox: dict, page_text: str) -> dict:
        keys = {k.lower() for k in infobox}
        text = (page_text or "").lower()
        length = max(len(text), 1)

        features = {
            "has_actor": int(bool(keys & ACTOR_INFOBOX_KEYS)),
            "tmdb_match": OnscreenFeatures.tmdb_match(name, franchise),
            "has_appearance_field": int(bool(keys & APPEARANCE_KEYS)),
            "infobox_field_count": len(infobox),
            "log_text_length": len(text) ** 0.5,
        }
        for term in FILMED_TERMS:
            features[f"filmed_{term.replace(' ', '_')}"] = 1000 * text.count(term) / length
        for term in UNFILMED_TERMS:
            features[f"unfilmed_{term.replace(' ', '_')}"] = 1000 * text.count(term) / length
        return features

    @staticmethod
    def columns() -> list[str]:
        base = ["has_actor", "tmdb_match", "has_appearance_field",
                "infobox_field_count", "log_text_length"]
        base += [f"filmed_{t.replace(' ', '_')}" for t in FILMED_TERMS]
        base += [f"unfilmed_{t.replace(' ', '_')}" for t in UNFILMED_TERMS]
        return base
