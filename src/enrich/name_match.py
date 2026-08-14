"""Match wiki character names against TMDB cast credits.

Exact token-set matching on the page title alone finds a credit for only 34% of
on-screen characters. Two things cause the misses: wikis title a page with a
character's full formal name while TMDB credits a short form ("Eddard Stark"
versus "Ned"), and TMDB credits carry decoration ("Sir Jaime 'Kingslayer'
Lannister", "Bran Stark (voice)").

This generates candidate names from the infobox alias fields as well as the
page title, normalises both sides, and matches on token overlap rather than
equality.
"""

import re
import unicodedata
from collections import defaultdict

import pandas as pd

from src.paths import CLEAN_DIR

ALIAS_KEYS = frozenset({
    "aka", "alias", "aliases", "nickname", "nicknames", "other names",
    "also known as", "real name", "full name", "birth name", "name",
    "titles", "title",
})

# Dropped before matching: honorifics, credit decoration, and role words that
# appear in TMDB credits but never in a wiki page title.
NAME_STOPWORDS = frozenset({
    "the", "a", "of", "and", "sir", "lord", "lady", "king", "queen", "prince",
    "princess", "dr", "doctor", "mr", "mrs", "ms", "miss", "captain", "colonel",
    "general", "sergeant", "officer", "agent", "young", "older", "old", "adult",
    "child", "teen", "baby", "voice", "uncredited", "archive", "footage",
    "flashback", "credit", "credits", "scenes", "deleted", "jr", "sr",
})

# Split multi-value infobox fields: bullets, semicolons, slashes, newlines.
VALUE_SPLIT_RE = re.compile(r"[;\n•|/]+|,(?=\s*[A-Z])")
PAREN_RE = re.compile(r"\([^)]*\)")
NON_ALPHA_RE = re.compile(r"[^a-z0-9 ]")
WHITESPACE_RE = re.compile(r"\s+")

# A single shared token identifies a character only if few credits carry it.
SINGLE_TOKEN_MAX_CREDITS = 4


class NameMatcher:
    """Franchise-scoped fuzzy matcher from character names to TMDB credits."""

    def __init__(self, cast: pd.DataFrame):
        # franchise -> token -> row indices, so a lookup only scans credits
        # that share at least one token.
        self._token_index: dict[str, dict[str, set[int]]] = defaultdict(
            lambda: defaultdict(set)
        )
        self._entries: dict[str, list[tuple[frozenset, float, float]]] = defaultdict(list)

        for franchise, group in cast.groupby("franchise"):
            for _, row in group.iterrows():
                tokens = NameMatcher.tokens(row["character"])
                if not tokens:
                    continue
                position = len(self._entries[franchise])
                self._entries[franchise].append(
                    (frozenset(tokens), row["billing_order"], row["episode_count"])
                )
                for token in tokens:
                    self._token_index[franchise][token].add(position)

    @staticmethod
    def normalise(text: str) -> str:
        text = unicodedata.normalize("NFKD", str(text))
        text = "".join(c for c in text if not unicodedata.combining(c)).lower()
        text = re.sub(r"[’'\"`]", "", text)
        text = PAREN_RE.sub(" ", text)
        text = NON_ALPHA_RE.sub(" ", text)
        return WHITESPACE_RE.sub(" ", text).strip()

    @staticmethod
    def tokens(text: str) -> set[str]:
        return {
            t for t in NameMatcher.normalise(text).split()
            if t not in NAME_STOPWORDS and len(t) > 2
        }

    @staticmethod
    def candidates(name: str, infobox: dict) -> list[set[str]]:
        """Token sets for the page title and every alias field."""
        raw = [name]
        for key, value in (infobox or {}).items():
            if key.lower() in ALIAS_KEYS and value:
                raw.extend(VALUE_SPLIT_RE.split(str(value)))

        out, seen = [], set()
        for item in raw:
            tokens = NameMatcher.tokens(item)
            # Single very short tokens match far too much to be safe.
            if not tokens or len(" ".join(tokens)) < 4:
                continue
            key = frozenset(tokens)
            if key not in seen:
                seen.add(key)
                out.append(tokens)
        return out

    def lookup(self, name: str, franchise: str, infobox: dict) -> dict:
        """Best credit for a character, or empty values if nothing matches."""
        entries = self._entries.get(franchise)
        if not entries:
            return {"tmdb_match": 0, "billing_order": None, "episode_count": None}

        index = self._token_index[franchise]
        best = None

        for candidate in NameMatcher.candidates(name, infobox):
            # Only credits sharing a token can possibly match.
            positions: set[int] = set()
            for token in candidate:
                positions |= index.get(token, set())

            for position in positions:
                credited, billing, episodes = entries[position]
                shared = len(candidate & credited)
                if not shared:
                    continue

                # Subset either way, or majority overlap.
                union = len(candidate | credited)
                subset = candidate <= credited or credited <= candidate
                overlap = shared / union
                if not (subset or overlap >= 0.5):
                    continue

                # One shared token is only convincing when that token is
                # distinctive within the franchise. "Brienne" identifies one
                # person; "Stark" or "Trooper" does not.
                if shared == 1 and not subset:
                    continue
                if shared == 1:
                    token = next(iter(candidate & credited))
                    if len(index.get(token, ())) > SINGLE_TOKEN_MAX_CREDITS:
                        continue

                score = (overlap, -(billing if pd.notna(billing) else 999))
                if best is None or score > best[0]:
                    best = (score, billing, episodes)

        if best is None:
            return {"tmdb_match": 0, "billing_order": None, "episode_count": None}
        return {"tmdb_match": 1, "billing_order": best[1], "episode_count": best[2]}

    @staticmethod
    def from_disk() -> "NameMatcher":
        path = CLEAN_DIR / "tmdb_cast.csv"
        if not path.exists():
            raise SystemExit(f"{path} missing. Run python -m src.enrich.tmdb_enrich first.")
        return NameMatcher(pd.read_csv(path))
