"""Tests for the weighted-loss threshold selection.

The cost sweep decides which threshold the writeup reports, so the arithmetic
behind it is checked here against hand-worked cases.
"""

import numpy as np
import pytest

from src.q1_character.costs import CostAnalysis


def test_metrics_match_a_hand_counted_confusion_matrix():
    y = np.array([1, 1, 1, 0, 0, 0])
    proba = np.array([0.9, 0.8, 0.2, 0.7, 0.1, 0.1])

    result = CostAnalysis.metrics(y, proba, threshold=0.5)

    # Above 0.5: two true positives and one false positive.
    assert (result["TP"], result["FP"], result["FN"], result["TN"]) == (2, 1, 1, 2)
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == pytest.approx(2 / 3)
    assert result["f1"] == pytest.approx(2 / 3)
    assert result["fp_rate"] == pytest.approx(1 / 3)
    assert result["accuracy"] == pytest.approx(4 / 6)


def test_raising_the_threshold_trades_recall_for_precision():
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.4).astype(int)
    proba = np.clip(0.25 * y + rng.random(500) * 0.7, 0, 1)

    low = CostAnalysis.metrics(y, proba, 0.3)
    high = CostAnalysis.metrics(y, proba, 0.7)

    assert low["recall"] > high["recall"]
    assert low["fp_rate"] > high["fp_rate"]


def test_loss_is_the_weighted_error_count_per_character():
    y = np.array([1, 1, 0, 0])
    proba = np.array([0.9, 0.2, 0.8, 0.1])

    curve = CostAnalysis.cost_curve(y, proba, ratio=3.0)
    row = curve[curve["threshold"] == 0.5].iloc[0]

    # One false negative (0.2) and one false positive (0.8).
    assert (row["FN"], row["FP"]) == (1, 1)
    assert row["loss"] == pytest.approx((3.0 * 1 + 1) / 4)


def test_expensive_misses_push_the_threshold_down():
    """A higher cost on false negatives should never raise the optimum.

    This is the property the writeup leans on when it says Spartacus wants a low
    threshold and Grey's Anatomy a high one.
    """
    rng = np.random.default_rng(1)
    y = (rng.random(600) < 0.35).astype(int)
    proba = np.clip(0.3 * y + rng.random(600) * 0.65, 0, 1)

    _, best = CostAnalysis.sweep(y, proba)
    thresholds = best.sort_values("cost_ratio")["threshold"].to_numpy()

    assert np.all(np.diff(thresholds) <= 0)


def test_a_perfect_ranker_has_zero_loss_at_some_threshold():
    y = np.array([0, 0, 0, 1, 1, 1])
    proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

    curve = CostAnalysis.cost_curve(y, proba, ratio=5.0)
    assert curve["loss"].min() == pytest.approx(0.0)
