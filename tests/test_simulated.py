"""Simulated endpoint tests: known-cost recovery, failures, drift."""

import json

from jev_observatory.schema import SystemOneRequest
from jev_observatory.simulated import SimConfig, SimulatedTransport, detect_drift, ols_fit
from jev_observatory.validation import validate_response


def _payload(n_questions=2, k=4, state_chars=1000):
    state = "word " * (state_chars // 5)
    questions = {}
    for q in range(n_questions):
        questions[f"q{q}"] = {
            "type": "choice",
            "instructions": f"pick {q}",
            "criteria": {f"opt_{i:03d}": f"option {i}" for i in range(k)},
        }
    return {"state": state, "model": "jev-1.13.0", "questions": questions}


def test_simulated_transport_returns_contract_valid_responses():
    transport = SimulatedTransport(SimConfig(seed=1))
    payload = _payload()
    response = transport.post("/v1/systemone", payload)
    assert response.status_code == 200 and response.error is None
    request = SystemOneRequest.model_validate(payload)
    validated = validate_response(json.loads(response.body), request)
    assert validated.contract_valid, validated.codes


def test_latency_increases_with_L_Q_and_C():
    """Means over noise must rank correctly: the cost model is monotone."""
    transport = SimulatedTransport(SimConfig(seed=2, noise_ms=1.0))
    small = transport.post("/x", _payload(n_questions=2, k=4, state_chars=500)).total_ms
    big = transport.post("/x", _payload(n_questions=32, k=64, state_chars=8000)).total_ms
    assert big > small + 50  # 7.5k chars + 30 questions + ~380 candidates


def test_cold_first_call_penalized_then_warm():
    transport = SimulatedTransport(SimConfig(seed=3, noise_ms=0.5, cold_penalty_ms=40.0))
    cold = transport.post("/x", _payload(state_chars=100)).total_ms
    warm = transport.post("/x", _payload(state_chars=100)).total_ms
    assert cold > warm + 20
    assert transport.post("/x", _payload()).cold_connection is False


def test_failure_knob_produces_retryable_status():
    transport = SimulatedTransport(SimConfig(seed=4, failure_rate=1.0))
    response = transport.post("/x", _payload())
    assert response.status_code == 529 and response.body == b""


def test_drift_inflates_later_calls():
    transport = SimulatedTransport(SimConfig(seed=5, drift_after_calls=5, drift_multiplier=3.0,
                                             noise_ms=0.5))
    early = [transport.post("/x", _payload(state_chars=100)).total_ms for _ in range(5)]
    late = [transport.post("/x", _payload(state_chars=100)).total_ms for _ in range(5)]
    assert sum(late) / 5 > 2.5 * sum(early) / 5


def test_ols_recovers_exact_coefficients_without_noise():
    """Hand-built data with known linear model: OLS must recover it."""
    # y = 3 + 2*x1 + 5*x2
    X = [[1.0, x1, x2] for x1, x2 in [(1, 1), (1, 2), (2, 1), (2, 2), (3, 3), (4, 1)]]
    y = [3 + 2 * row[1] + 5 * row[2] for row in X]
    coeffs = ols_fit(X, y)
    assert coeffs == pytest.approx([3.0, 2.0, 5.0])


import pytest  # noqa: E402


def test_drift_detector_flags_and_clears():
    flagged = detect_drift([100.0] * 6 + [250.0] * 6)
    assert flagged["flagged"] is True and flagged["ratio"] >= 1.5
    quiet = detect_drift([100.0] * 6 + [105.0] * 6)
    assert quiet["flagged"] is False
    sparse = detect_drift([100.0, 200.0])
    assert sparse["flagged"] is False and "need" in sparse["reason"]
