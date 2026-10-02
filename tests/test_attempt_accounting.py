"""Focused tests: attempt-cost accounting, paid-route pin, reservation bounds.

Covers the reported CLI bug (429-no-usage then 200-with-cost crashed on
``cost_sum += float``), unknown+known mix policy, the DeepInfra paid routing
pin (routing only), and the audited per-dataset input reservation bounds.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "rbr", ROOT / "scripts" / "benchmark" / "run_baseline_revision.py")
rbr = importlib.util.module_from_spec(_spec)
sys.modules["rbr"] = rbr
_spec.loader.exec_module(rbr)

from jev_observatory.revision_run import PersistentBudget  # noqa: E402


def test_attempt_cost_retry_429_then_200_sums_known_and_never_crashes():
    ac = rbr.AttemptCost()
    ac.add(None)             # 429 attempt: no usage -> unknown charge flagged
    ac.add(0.0021)           # 200 attempt: billed cost must still be summed
    assert ac.known_sum == pytest.approx(0.0021)   # no TypeError, no lost bill
    assert ac.unknown is True
    assert ac.reported is None                     # fail closed: total unknown


def test_attempt_cost_all_known_reports_total():
    ac = rbr.AttemptCost()
    ac.add(0.001)
    ac.add(0.002)
    assert ac.reported == pytest.approx(0.003)
    assert ac.unknown is False


def test_attempt_cost_unknown_plus_known_records_valid_and_holds_budget():
    ac = rbr.AttemptCost()
    ac.add(None)
    ac.add(0.5)
    budget = PersistentBudget(None, 10.0)
    rid = budget.reserve("k", 2.0)                 # worst-case reservation
    budget.settle(rid, ac.reported)                # None -> held, never free
    snap = budget.snapshot()
    assert snap["held_usd"] == pytest.approx(2.0)
    assert snap["settled_usd"] == 0.0
    # fully-known cost settles at the summed attempts
    ac2 = rbr.AttemptCost()
    ac2.add(0.1)
    ac2.add(0.2)
    rid2 = budget.reserve("k", 1.0)
    budget.settle(rid2, ac2.reported)
    snap2 = budget.snapshot()
    assert snap2["settled_usd"] == pytest.approx(0.3)
    assert snap2["held_usd"] == pytest.approx(2.0)


def test_route_pin_is_routing_only_for_gemma():
    task = {"model": "google/gemma-3-4b-it",
            "wire": {"extra_body": {"temperature": 0, "top_p": 0.9}}}
    eb = rbr._extra_body_for(task)
    assert eb["provider"] == {"order": ["deepinfra/bf16"],
                              "allow_fallbacks": False}
    # generation params preserved untouched
    assert eb["temperature"] == 0 and eb["top_p"] == 0.9
    assert set(eb) == {"temperature", "top_p", "provider"}
    assert rbr._route_label(task) == "deepinfra/bf16-paid-pinned"


def test_route_pin_absent_for_other_models():
    for model in ("xiaomi/mimo-v2.6-pro", "qwen/qwen3.7-flash"):
        task = {"model": model, "wire": {"extra_body": {"temperature": 0}}}
        eb = rbr._extra_body_for(task)
        assert "provider" not in eb
        assert rbr._route_label(task) == "provider-default"
    assert rbr._extra_body_for({"model": "m", "wire": {}}) is None


def test_reservation_uses_audited_per_dataset_input_bounds():
    prices = {"google/gemma-3-4b-it": {"input_per_token": 5e-8,
                                       "output_per_token": 1e-7}}
    hle = {"model": "google/gemma-3-4b-it", "dataset": "hle_text_mc",
           "wire": {"max_output_tokens": 4096}}
    mmlu = {**hle, "dataset": "mmlu"}
    # HLE input bound 16384 (the flat 4096 underbound long prompts)
    assert rbr.RESERVATION_INPUT_TOKENS["hle_text_mc"] == 16384
    assert rbr._max_cost_of(prices, hle) > rbr._max_cost_of(prices, mmlu)
    expected_hle = 3 * (16384 * 5e-8 + 4096 * 1e-7)
    assert rbr._max_cost_of(prices, hle) == pytest.approx(expected_hle)
