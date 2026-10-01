"""ACTIVE-cell rerun freeze tests: selector fidelity, unique task identity,
current-price reservations (offline only; no network, no model calls)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "benchmark"))
sys.path.insert(0, str(ROOT / "src"))

from freeze_rerun_plan import (  # noqa: E402
    active_cell_selector, fetch_prices, _read_jsonl, _raw2_index,
    _rerun_class, V2_RESULTS, V3_RESULTS, SUMMARY_V3, V3_FREEZE,
    PLAN_OUT, TASKS_OUT, TRANSPORT_ATTEMPTS, RESERVATION_SAFETY_INPUT_TOKENS,
)

PLAN = json.loads(PLAN_OUT.read_text(encoding="utf-8"))
TASKS = _read_jsonl(TASKS_OUT)


def test_selector_matches_published_summary_v3_labels():
    """The frozen selector must reproduce summary_v3's cell-level merge
    exactly: complete v3 cells supersede v2; v2 survives only as carryover."""
    v2 = [r for r in _read_jsonl(V2_RESULTS) if r.get("terminal")]
    v3 = [r for r in _read_jsonl(V3_RESULTS) if r.get("terminal")]
    v3_models = {m["id"] for m in json.loads(V3_FREEZE.read_text())["models"]}
    n_req = {"mmlu": 1000, "gpqa": 196, "math500_choice": 261,
             "hle_text_mc": 494, "arc": 1172}
    active = active_cell_selector(v2, v3, n_req, v3_models)
    summary = json.loads(SUMMARY_V3.read_text())["models"]
    for (model, dataset), (version, _rows) in active.items():
        label = ((summary.get(model) or {}).get(dataset) or {}).get("protocol", "")
        assert label.startswith("provider") == (version == "v3"), (model, dataset)
    # supervisor anchors: Mistral Small active MATH is v3's 235, never v2+v3 459
    assert PLAN["active_selector"]["n_v2_carryover_cells"] == 5
    assert PLAN["active_selector"]["n_active_cells"] == 59


def test_active_math_anchor_counts():
    math = {c["model"]: c["n_rows_to_rerun"] for c in PLAN["rerun_cells"]
            if c["dataset"] == "math500_choice"}
    assert math["mistralai/mistral-small-3.2-24b-instruct"] == 235
    assert math["google/gemma-3-4b-it"] == 167
    assert math["xiaomi/mimo-v2.6-pro"] == 139


def test_task_ids_unique_and_carry_original_wire():
    ids = [t["logical_request_id"] for t in TASKS]
    assert len(ids) == len(set(ids))
    assert len(ids) == PLAN["rerun_total_rows"]
    for t in TASKS[:50]:
        ver, model, dataset, item = t["logical_request_id"].split("|")
        assert ver == t["source_version"] and model == t["model"]
        assert dataset == t["dataset"] and item == t["item_id"]
        w = t["wire"]
        assert w["model"] == model and isinstance(w, dict)
        # v2-sourced replacements keep the v2 wire (temperature 0 etc.)
        if ver == "v2":
            assert (w.get("extra_body") or {}).get("temperature") == 0


def test_superseded_v2_history_is_labeled_not_rerun():
    sup = PLAN["superseded_v2_history"]
    assert sup, "v3-active cells must record their superseded v2 history"
    for s in sup:
        assert "uncorrected-superseded" in s["label"]
    # no task may come from a superseded row
    for t in TASKS:
        key = (t["model"], t["dataset"])
        if t["source_version"] == "v2":
            assert key in {("ibm-granite/granite-4.0-h-micro", "gpqa"),
                           ("ibm-granite/granite-4.0-h-micro", "hle_text_mc"),
                           ("ibm-granite/granite-4.0-h-micro", "math500_choice"),
                           ("ibm-granite/granite-4.0-h-micro", "mmlu"),
                           ("qwen/qwen3.8-max-0902", "mmlu")}


def test_reservations_use_current_prices_and_cover_retries():
    prices = json.loads((ROOT / "data_report" / "provider_price_metadata.json")
                        .read_text())["prices"]
    for c in PLAN["rerun_cells"]:
        if not c["n_rows_to_rerun"]:
            continue
        rate = prices[c["model"]]
        expect = TRANSPORT_ATTEMPTS * (
            RESERVATION_SAFETY_INPUT_TOKENS * rate["input_per_token"]
            + c["wire_max_output_tokens"] * rate["output_per_token"])
        assert abs(c["reservation_per_task_usd"] - round(expect, 4)) < 1e-6
        assert c["reservation_per_task_usd"] > 0


def test_row_classification_rule():
    raws = _raw2_index()
    stable = {"logical_request_id": "x", "terminal": True, "status": "ok",
              "recovery_stage": "exact"}
    assert _rerun_class(stable, "v3", raws) == "keep_original"
    non_ok = {"logical_request_id": "x", "terminal": True, "status": "http_error",
              "recovery_stage": "unrecovered"}
    assert _rerun_class(non_ok, "v3", raws) == "settled_non_ok"
    v3_nonstable = {"logical_request_id": "never-in-raw2", "terminal": True,
                    "status": "ok", "recovery_stage": "isolated_upper_key"}
    assert _rerun_class(v3_nonstable, "v3", raws) == "rerun"
    # v2 rows never borrow v3 state and vice versa: raws consulted for v2 only
    assert _rerun_class(v3_nonstable, "v2", raws) == "rerun"
