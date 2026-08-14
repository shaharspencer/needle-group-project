import pandas as pd
import pytest

from src.constants import STRATA_TARGETS
from src.labels.build_gold_set import GoldSetBuilder


def test_char_id_is_stable_and_short():
    url = "https://gameofthrones.fandom.com/wiki/Eddard_Stark"
    assert GoldSetBuilder.char_id(url) == GoldSetBuilder.char_id(url)
    assert len(GoldSetBuilder.char_id(url)) == 12


def test_char_id_differs_between_pages():
    a = GoldSetBuilder.char_id("https://x.fandom.com/wiki/A")
    b = GoldSetBuilder.char_id("https://x.fandom.com/wiki/B")
    assert a != b


@pytest.mark.parametrize("is_dead,expected", [
    (1, "dead"),
    (0, "alive"),
    (None, "dropped"),
])
def test_label_state(is_dead, expected):
    assert GoldSetBuilder.label_state(is_dead) == expected


def test_death_sentences_picks_only_death_mentions():
    text = "He was a knight. He was killed at the Trident. He liked horses."
    found = GoldSetBuilder.death_sentences(text)

    assert "killed at the Trident" in found
    assert "liked horses" not in found


def test_death_sentences_empty_when_nothing_matches():
    assert GoldSetBuilder.death_sentences("He was a knight who liked horses.") == ""


def test_summarize_infobox_keeps_only_interesting_keys():
    infobox = {"Status": "Deceased", "Hair": "Brown", "Actor": "Sean Bean"}
    summary = GoldSetBuilder.summarize_infobox(infobox)

    assert "Deceased" in summary
    assert "Sean Bean" in summary
    assert "Brown" not in summary


def _pool(n_per_franchise: dict[str, int], regime: str, state: str) -> pd.DataFrame:
    rows = []
    for franchise, count in n_per_franchise.items():
        for i in range(count):
            rows.append({
                "char_id": f"{franchise}{i}",
                "franchise": franchise,
                "regime": regime,
                "state": state,
            })
    return pd.DataFrame(rows)


def test_sample_spreads_across_franchises():
    # One franchise dominates by 20x; round-robin should still cover both.
    regime, state = next(iter(STRATA_TARGETS))
    pool = _pool({"Star Wars": 2000, "Ozark": 100}, regime, state)

    sample = GoldSetBuilder.sample(pool, seed=1)

    assert sample["franchise"].nunique() == 2
    assert (sample["franchise"] == "Ozark").sum() > 10


def test_sample_is_deterministic_for_a_seed():
    regime, state = next(iter(STRATA_TARGETS))
    pool = _pool({"A": 500, "B": 500}, regime, state)

    first = GoldSetBuilder.sample(pool, seed=7)["char_id"].tolist()
    second = GoldSetBuilder.sample(pool, seed=7)["char_id"].tolist()

    assert first == second
