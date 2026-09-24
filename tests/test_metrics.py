"""Hand-computed metric fixtures (DESIGN.md §6 requires verified scorers)."""

import math

import pytest

from jev_observatory.metrics import (
    accuracy,
    brier_binary,
    brier_multiclass,
    ece_equal_width,
    log_loss_clipped,
    log_loss_exact,
    rank_probability_score,
    reliability,
    risk_coverage,
    ordinal_mae,
    wilson_interval,
)


def test_accuracy_wilson_hand_computed():
    result = accuracy([True] * 70 + [False] * 30)
    assert result["n"] == 100 and result["n_correct"] == 70
    interval = wilson_interval(70, 100)
    assert interval.low == pytest.approx(0.60415, abs=1e-4)
    assert interval.high == pytest.approx(0.78105, abs=1e-4)
    # degenerate cases clamp to [0,1] (within floating-point noise)
    assert wilson_interval(0, 100).low == pytest.approx(0.0, abs=1e-9)
    assert wilson_interval(100, 100).high == pytest.approx(1.0, abs=1e-9)


def test_accuracy_empty_sample():
    result = accuracy([])
    assert result["accuracy"] is None and result["note"] == "empty sample"


def test_brier_binary_hand_computed():
    assert brier_binary(0.8, True) == pytest.approx(0.04)
    assert brier_binary(0.3, False) == pytest.approx(0.09)
    assert brier_binary(1.0, False) == pytest.approx(1.0)


def test_brier_multiclass_scale_documented():
    # probabilities {a: 0.7, b: 0.2, c: 0.1}, gold b:
    # (0.7-0)^2 + (0.2-1)^2 + (0.1-0)^2 = 0.49 + 0.64 + 0.01 = 1.14
    result = brier_multiclass({"a": 0.7, "b": 0.2, "c": 0.1}, "b")
    assert result["brier_sum"] == pytest.approx(1.14)
    assert "0..K-1" in result["scale"]


def test_log_loss_exact_and_clipped():
    assert log_loss_exact(0.25) == pytest.approx(-math.log(0.25))
    assert math.isinf(log_loss_exact(0.0))  # exact is infinite, never silently clipped
    assert log_loss_clipped(0.0) == -math.log(1e-12)
    assert log_loss_clipped(1e-14) == -math.log(1e-12)  # floored at epsilon
    # clipping never *raises* a probability that is already above epsilon
    assert log_loss_clipped(0.5, epsilon=1e-6) == pytest.approx(-math.log(0.5))


def test_reliability_bins_counts():
    # bins are half-open [lo, hi): p=0.1 belongs to bin 1, not bin 0
    pairs = [(0.1, False), (0.15, True), (0.75, True), (0.85, False), (0.95, True)]
    bins = reliability(pairs, n_bins=10)
    bin1 = [b for b in bins if b["bin"] == 1][0]
    assert bin1["n"] == 2
    assert bin1["low"] == 0.1 and bin1["high"] == 0.2
    assert bin1["mean_probability"] == pytest.approx(0.125)
    assert bin1["empirical_accuracy"] == pytest.approx(0.5)
    # bin 9 is the last bin and includes its right edge
    bin9 = [b for b in bins if b["bin"] == 9][0]
    assert bin9["n"] == 1 and bin9["mean_probability"] == pytest.approx(0.95)
    # empty bins are omitted, not zero-filled
    assert all(b["n"] > 0 for b in bins)
    assert len(bins) == 4  # only bins 1, 7, 8, 9 are populated



def test_ece_equal_width_hand_computed():
    # bin1 [(0.1,F),(0.15,T)] mean p=0.125 emp=0.5 -> 2*0.375
    # bin7 [(0.75,T)] -> 1*0.25; bin8 [(0.85,F)] -> 1*0.85; bin9 [(0.95,T)] -> 1*0.05
    pairs = [(0.1, False), (0.15, True), (0.75, True), (0.85, False), (0.95, True)]
    ece = ece_equal_width(pairs, n_bins=10)
    expected = (2 * 0.375 + 0.25 + 0.85 + 0.05) / 5
    assert ece == pytest.approx(expected)


def test_risk_coverage_monotone_accuracy_bounds():
    pairs = [(0.9, True), (0.8, True), (0.7, False), (0.1, False)]
    curve = risk_coverage(pairs)
    assert curve[0] == {"coverage": 0.25, "accuracy": 1.0, "n": 1}
    assert curve[-1]["accuracy"] == pytest.approx(0.5)
    assert curve[-1]["coverage"] == pytest.approx(1.0)
    # accuracy along the curve is cumulative-correct / n
    assert curve[2]["accuracy"] == pytest.approx(2 / 3)


def test_risk_coverage_ties_are_reproducible():
    pairs = [(0.5, True), (0.5, False), (0.5, True)]
    curve = risk_coverage(pairs)
    # insertion order breaks ties deterministically (rounded to 6dp on output)
    assert [c["accuracy"] for c in curve] == [1.0, 0.5, round(2 / 3, 6)]


def test_rank_probability_score_hand_computed():
    # probs [0.2, 0.5, 0.3], gold level 1:
    # CDF1=0.2 vs Y1=0 -> 0.04; CDF2=0.7 vs Y2=1 -> 0.09; /2 = 0.065
    assert rank_probability_score([0.2, 0.5, 0.3], 1) == pytest.approx(0.065)
    # perfect prediction
    assert rank_probability_score([0.0, 1.0], 1) == pytest.approx(0.0)
    assert rank_probability_score([1.0, 0.0], 0) == pytest.approx(0.0)


def test_ordinal_mae():
    assert ordinal_mae([1.2, 0.8], [1.0, 1.0]) == pytest.approx(0.2)
    assert ordinal_mae([], []) is None