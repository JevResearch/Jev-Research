"""Analysis tests: prediction/gold join and per-type scoring correctness."""

import json
import math

import pytest

from jev_observatory.analysis import analyze_run, gold_for_question, score_prediction
from jev_observatory.experiment import logical_requests
from jev_observatory.providers import MockProvider
from jev_observatory.runner import Runner, load_run_store, plan_run


def _spec():
    return {
        "experiment": "analysis-smoke",
        "model": "jev-1.13.0",
        "seeds": {"order": 1},
        "items": [
            {
                "id": "ch-1", "group": "g", "cluster": "c1",
                "state": "Ticket one.",
                "gold": {"q": {"value": "tech"}},
                "questions": {"q": {"type": "choice", "instructions": "which?",
                                    "criteria": {"billing": "money", "tech": "bugs"}}},
            },
            {
                "id": "no-1", "group": "g", "cluster": "c2",
                "state": "Ticket two.",
                "gold": {"q": {"value": True}},
                "questions": {"q": {"type": "noul", "instructions": "urgent?"}},
            },
            {
                "id": "sc-1", "group": "g", "cluster": "c3",
                "state": "Ticket three.",
                "gold": {"q": {"value": 1}},
                "questions": {"q": {"type": "score", "instructions": "anger?", "criteria": ["calm", "angry"]}},
            },
        ],
    }


def test_gold_for_question_shapes():
    entry = {"gold": {"a": {"value": "x"}, "b": True, "c": 2}}
    assert gold_for_question(entry, "a") == "x"
    assert gold_for_question(entry, "b") is True
    assert gold_for_question(entry, "c") == 2
    assert gold_for_question(entry, "zzz") is None


def test_score_prediction_choice_noul_score(tmp_path):
    spec = _spec()
    record = {
        "logical_request_id": "native:ch-1", "item_id": "ch-1", "group": "g", "cluster": "c1",
        "condition": "native", "status": "ok", "violation_codes": [],
        "usage_input_tokens": 100, "latency_ms_total": 5.0,
        "predictions": {
            "q": {"type": "choice", "usable": True, "choice": "tech",
                  "probabilities": {"billing": 0.2, "tech": 0.8}, "p_max": 0.8},
        },
    }
    entry = spec["items"][0]
    rows = score_prediction(record, entry)
    assert len(rows) == 1
    row = rows[0]
    assert row["correct"] is True
    assert row["p_true"] == 0.8
    assert row["log_loss_exact"] == pytest.approx(-math.log(0.8))
    assert row["brier_sum"] == pytest.approx((0.2 - 0.0) ** 2 + (0.8 - 1.0) ** 2)  # 0.08


def test_noul_threshold_uses_preregistered_0_5():
    entry = {"gold": {"q": {"value": False}}}
    record = {
        "logical_request_id": "r", "item_id": "i", "status": "ok", "violation_codes": [],
        "predictions": {"q": {"type": "noul", "usable": True, "noul": 0.49}},
    }
    rows = score_prediction(record, entry)
    assert rows[0]["correct"] is True  # 0.49 < 0.5 -> no
    assert rows[0]["brier_binary"] == pytest.approx(0.49**2)


def test_zero_probability_true_label_is_infinite_log_loss():
    entry = {"gold": {"q": {"value": "tech"}}}
    record = {
        "logical_request_id": "r", "item_id": "i", "status": "ok", "violation_codes": [],
        "predictions": {"q": {"type": "choice", "usable": True, "choice": "billing",
                              "probabilities": {"billing": 1.0, "tech": 0.0}, "p_max": 1.0}},
    }
    rows = score_prediction(record, entry)
    assert rows[0]["correct"] is False
    assert math.isinf(rows[0]["log_loss_exact"])
    assert rows[0]["log_loss_clipped_1e-12"] == pytest.approx(-math.log(1e-12))


def test_missing_gold_marked_not_invented():
    entry = {"gold": {}}
    record = {
        "logical_request_id": "r", "item_id": "i", "status": "ok", "violation_codes": [],
        "predictions": {"q": {"type": "choice", "usable": True, "choice": "a",
                              "probabilities": {"a": 1.0}, "p_max": 1.0}},
    }
    rows = score_prediction(record, entry)
    assert rows[0]["correct"] is None


def _mock_run(tmp_path, spec):
    run_id, _ = plan_run(spec, provider="mock", root=str(tmp_path))
    store = load_run_store(run_id, str(tmp_path))
    Runner(MockProvider(seed=0), store).run(logical_requests(spec))
    return run_id


def test_analyze_run_produces_tables_and_synthetic_flag(tmp_path):
    spec = _spec()
    run_id = _mock_run(tmp_path, spec)
    metrics = analyze_run(run_id, str(tmp_path))
    assert metrics["synthetic"] is True
    assert metrics["warning"] == "MOCK/SIMULATED RESULTS"
    assert metrics["n_logical_results"] == 3
    assert "choice" in metrics["aggregates"]
    assert metrics["aggregates"]["choice"]["accuracy"]["n"] > 0
    derived = tmp_path / run_id / "derived"
    for name in ("metrics.json", "attempts.parquet", "results.parquet", "prediction_rows.parquet"):
        assert (derived / name).exists(), name
    # report is buildable and labelled synthetic
    from jev_observatory.report import build_report

    text = build_report(run_id, str(tmp_path))
    assert "SYNTHETIC" in text


def _mock_run(tmp_path, spec):
    return _mock_run_inner(tmp_path, spec)


def _mock_run_inner(tmp_path, spec):
    run_id, _ = plan_run(spec, provider="mock", root=str(tmp_path))
    from jev_observatory.runner import load_run_store

    store = load_run_store(run_id, str(tmp_path))
    Runner(MockProvider(seed=0), store).run(logical_requests(spec))
    return run_id


def test_analyze_unknown_item_refused(tmp_path):
    run_id = _mock_run(tmp_path, _spec())
    # Remove an item from items.jsonl to simulate a label/reference mismatch.
    items_path = tmp_path / run_id / "items.jsonl"
    lines = [l for l in items_path.read_text().splitlines() if l][:-1]
    items_path.write_text("\n".join(lines) + "\n")
    with pytest.raises(KeyError, match="refusing to analyze"):
        analyze_run(run_id, str(tmp_path))