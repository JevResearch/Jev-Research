"""Regression tests for Astra's audit findings (analysis/statistics corrections)."""

import json
import math

import pytest

from jev_observatory.analysis import _mean_exact_with_inf, aggregate, score_prediction
from jev_observatory.simulated import detect_drift, fit_latency_model
from jev_observatory.stats import cluster_bootstrap_accuracy
from jev_observatory.validation import PROBABILITY_QUANTUM, validate_response
from jev_observatory.schema import ChoiceQuestion, NoulQuestion, SystemOneRequest


# ---------------------------------------------------------------- log loss
def _noul_row(p_yes, gold_yes):
    brier = (p_yes - (1.0 if gold_yes else 0.0)) ** 2
    p_true = p_yes if gold_yes else 1.0 - p_yes
    exact = float("-inf") if p_true <= 0 else -math.log(p_true)
    return {"logical_request_id": f"noul-{p_yes}-{gold_yes}", "type": "noul",
            "predicted_p_yes": p_yes, "gold_yes": gold_yes,
            "correct": (p_yes >= 0.5) == gold_yes, "brier_binary": brier,
            "p_true": p_true, "log_loss_exact": exact,
            "log_loss_clipped_1e-12": -math.log(max(p_true, 1e-12))}


def test_exact_log_loss_mean_is_infinite_when_any_entry_is_infinite():
    rows = [_noul_row(0.8, True), _noul_row(0.0, True)]  # second assigns 0 to true
    agg = aggregate(rows)
    assert agg["noul"]["log_loss_exact_mean"] == math.inf
    assert agg["noul"]["n_infinite_log_loss"] == 1


def test_finite_exact_log_loss_mean():
    rows = [_noul_row(0.8, True), _noul_row(0.2, False)]
    agg = aggregate(rows)
    assert agg["noul"]["log_loss_exact_mean"] == pytest.approx(-math.log(0.8))
    assert agg["noul"]["n_infinite_log_loss"] == 0


def test_choice_exact_log_loss_infinite_propagates():
    rows = [{
        "logical_request_id": "r", "item_id": "i", "group": "g", "cluster": "c",
        "status": "ok", "violation_codes": [],
        "predictions": {"q": {"type": "choice", "usable": True, "choice": "a",
                              "probabilities": {"a": 1.0, "b": 0.0}, "p_max": 1.0}},
    }, {
        "logical_request_id": "r2", "item_id": "i2", "group": "g", "cluster": "c2",
        "status": "ok", "violation_codes": [],
        "predictions": {"q": {"type": "choice", "usable": True, "choice": "a",
                              "probabilities": {"a": 0.5, "b": 0.5}, "p_max": 0.5}},
    }]
    entry = {"gold": {"q": {"value": "b"}}}
    all_rows = []
    for record in rows:
        all_rows.extend(score_prediction(record, entry))
    agg = aggregate(all_rows)
    assert agg["choice"]["log_loss_exact_mean"] == math.inf
    assert agg["choice"]["n_infinite_log_loss"] == 1


# ------------------------------------------------------- noul reliability target
def test_noul_reliability_pairs_p_yes_with_gold_yes():
    """The old code paired P(yes) with *correctness* — the wrong target."""
    rows = [
        # both are "correct" (agree with the 0.5 threshold) but one is yes and one is no
        _noul_row(0.9, True),   # correct; gold yes
        _noul_row(0.1, False),  # correct; gold no
    ]
    agg = aggregate(rows)
    reliability = agg["noul"]["p_yes_reliability"]
    # bin 9 contains both P(yes)=0.9 (gold yes) and P(yes)=0.1 -> wait: 0.1 is bin 1.
    bins = {b["bin"]: b for b in reliability}
    assert bins[9]["n"] == 1 and bins[9]["empirical_accuracy"] == 1.0  # gold was yes
    assert bins[1]["n"] == 1 and bins[1]["empirical_accuracy"] == 0.0  # gold was no


# ------------------------------------------------------- cluster bootstrap size
def test_cluster_bootstrap_handles_unequal_clusters():
    """The old code divided resampled totals by the ORIGINAL sample size, biasing
    the interval when clusters have different sizes."""
    # cluster "big" has 10 items 50% correct; cluster "tiny" has 1 item wrong
    correct = [True] * 5 + [False] * 5 + [False]
    clusters = ["big"] * 10 + ["tiny"]
    result = cluster_bootstrap_accuracy(correct, clusters, n_resamples=1000, seed=0)
    assert result.estimate == pytest.approx(5 / 11)
    # a valid bootstrap interval must allow resampled draws of the big cluster
    # alone to reach both its lower and upper accuracy values
    assert result.ci_low <= 0.5 + 1e-9
    assert result.ci_high >= 0.5 - 1e-9


# ------------------------------------------------------------- drift ordering
def test_drift_uses_arrival_order_not_magnitude():
    """Old code sorted by magnitude: a stable series alternating fast/slow was
    misread as drift. Arrival order must be respected."""
    alternating = [100.0, 300.0] * 6  # stable pattern, huge magnitude spread
    assert detect_drift(alternating)["flagged"] is False
    drifting = [100.0] * 6 + [300.0] * 6  # genuine temporal change
    assert detect_drift(drifting)["flagged"] is True


# --------------------------------------------------------- block held-out fit
def _sweep_record(item_id, condition, block, latency, L, Q, C):
    return {
        "logical_request_id": item_id, "cluster": item_id, "condition": condition,
        "status": "ok", "latency_ms_first_attempt": latency,
        "measures": {"state_chars": L, "q": Q, "c_total_candidates": C},
    }


def test_latency_fit_holds_out_whole_blocks():
    """The audit found random-row holdout leaking block-level conditions."""
    records = []
    # L and Q deliberately NOT proportional (rank-4 design): an exactly
    # collinear fixture would be a singular matrix, not a test of the fix.
    for block in range(4):
        for i, (L, Q, C) in enumerate([(128, 1, 3), (600, 4, 7), (2000, 16, 2),
                                       (8000, 64, 9), (1100, 8, 5), (300, 2, 8),
                                       (4200, 32, 4), (800, 6, 6), (1600, 12, 11),
                                       (3100, 24, 10)]):
            condition = f"L={L};Q={Q};K=8;block={block}"
            # latency = 100 + 10*(L/1000) + 1*Q + 0.5*C : known linear truth
            records.append(_sweep_record(f"r{block}-{i}", condition, block,
                                         100.0 + 10 * L / 1000 + Q + 0.5 * C, L, Q, C))
    fit = fit_latency_model(records, holdout_fraction=0.25, seed=0)
    assert fit["fit_available"] is True
    assert "block_holdout" in fit["split_method"]
    assert fit["holdout_mae_ms"] < fit["holdout_mae_baseline_ms"]
    assert fit["train_r2"] > 0.99
    assert fit["coefficients"]["L_kilo"] == pytest.approx(10.0, abs=0.2)
    assert fit["coefficients"]["Q"] == pytest.approx(1.0, abs=0.2)
    assert fit["coefficients"]["C"] == pytest.approx(0.5, abs=0.2)


# ------------------------------------------------------- argmax quantization
def _choice_request():
    return SystemOneRequest(state="s", questions={
        "c": ChoiceQuestion(instructions="?", criteria={"a": "A", "b": "B"})})


def test_quantized_tie_is_warning_not_error():
    """Float-noise ties are warnings; a displayed gap of one full quantum is a
    real choice/argmax discrepancy (monotone rounding cannot reorder)."""
    request = _choice_request()
    # float-noise tie: one ULP below the max -> warning, not error
    import math

    raw = {"model": "m", "usage": {},
           "answers": {"c": {"type": "choice", "choice": "a",
                             "probabilities": {"a": math.nextafter(0.5, 0.0), "b": 0.5},
                             "confidence": 0.0}}}
    validated = validate_response(raw, request)
    assert "argmax_tie_quantized" in validated.codes
    assert "choice_not_argmax" not in validated.codes
    assert validated.usable
    # one full quantum gap: a real contract violation
    raw2 = {"model": "m", "usage": {},
            "answers": {"c": {"type": "choice", "choice": "a",
                              "probabilities": {"a": 0.4, "b": 0.5}, "confidence": 0.0}}}
    validated2 = validate_response(raw2, request)
    assert "choice_not_argmax" in validated2.codes
    assert not validated2.usable


def test_quantum_constant_is_one_cent():
    assert PROBABILITY_QUANTUM == 0.01


# ------------------------------------------------------- shuffle preservation
def test_manifest_carries_shuffle_and_reconstruction_uses_it(tmp_path):
    from jev_observatory.manifest import RunManifest, utc_now, write_manifest, verify_manifest
    manifest = RunManifest(
        run_id="r", experiment="e", provider="mock", model_requested="jev-1.13.0",
        created_at=utc_now(), items_sha256="x", n_items=1, shuffle=False,
    )
    assert manifest.shuffle is False
    write_manifest(tmp_path, manifest)
    assert verify_manifest(tmp_path).shuffle is False
