import pandas as pd
import pytest

from src.labels.eval_labels import LabelEvaluator


@pytest.mark.parametrize("a,b,expected", [
    ([1, 1, 0, 0], [1, 1, 0, 0], 1.0),      # perfect agreement
    ([1, 0, 1, 0], [0, 1, 0, 1], -1.0),     # perfect disagreement
])
def test_cohen_kappa_extremes(a, b, expected):
    kappa = LabelEvaluator.cohen_kappa(pd.Series(a), pd.Series(b))
    assert kappa == pytest.approx(expected)


def test_cohen_kappa_is_nan_when_one_label_is_unanimous():
    # Both raters say 1 every time: agreement is 100% but chance is also 100%.
    kappa = LabelEvaluator.cohen_kappa(pd.Series([1, 1, 1]), pd.Series([1, 1, 1]))
    assert pd.isna(kappa)


def test_cohen_kappa_below_raw_agreement():
    # 80% raw agreement on a skewed distribution should correct downward.
    a = pd.Series([1, 1, 1, 1, 0])
    b = pd.Series([1, 1, 1, 1, 1])
    kappa = LabelEvaluator.cohen_kappa(a, b)
    assert kappa < 0.8


def _scored_frame(rows):
    return pd.DataFrame([
        {
            "stratum": "s", "is_character": 1, "appears_onscreen": 1,
            "auto_is_dead": auto, "dies_in_narrative": truth,
        }
        for auto, truth in rows
    ])


def test_score_counts_confusion_matrix():
    df = _scored_frame([(1, 1), (1, 0), (0, 1), (0, 0)])
    row = LabelEvaluator.score(df).iloc[0]

    assert (row.TP, row.FP, row.FN, row.TN) == (1, 1, 1, 1)
    assert row.precision == pytest.approx(0.5)
    assert row.recall == pytest.approx(0.5)


def test_score_treats_unlabelled_rows_as_alive():
    # auto_is_dead is NaN for rows with no parsed status; they count as alive.
    df = _scored_frame([(None, 0), (None, 1)])
    row = LabelEvaluator.score(df).iloc[0]

    assert (row.TP, row.FP) == (0, 0)
    assert (row.FN, row.TN) == (1, 1)


def test_score_excludes_undecidable_and_offscreen():
    df = pd.DataFrame([
        {"stratum": "s", "is_character": 1, "appears_onscreen": 1,
         "auto_is_dead": 1, "dies_in_narrative": -1},   # undecidable
        {"stratum": "s", "is_character": 1, "appears_onscreen": 0,
         "auto_is_dead": 1, "dies_in_narrative": 1},    # never filmed
        {"stratum": "s", "is_character": 0, "appears_onscreen": 1,
         "auto_is_dead": 1, "dies_in_narrative": 1},    # not a character
        {"stratum": "s", "is_character": 1, "appears_onscreen": 1,
         "auto_is_dead": 1, "dies_in_narrative": 1},    # the only scorable row
    ])
    report = LabelEvaluator.score(df)

    assert len(report) == 1
    assert report.iloc[0].n == 1
