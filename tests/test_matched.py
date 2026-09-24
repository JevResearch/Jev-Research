"""Matched modern-baseline comparison tests (offline only, scripted transports).

Covers the parent gate's required test list: malformed outputs, numeric source
keys, missing-gold isolation, bool label conversion, weighted sampling totals,
paired-score hand fixtures, all-item denominator including failures, no
synthetic probability arrays for text baselines, error/resume budget guard,
and wire-order preservation across the freeze round-trip.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from jev_observatory.matched import (
    DATASETS,
    HARD_COST_CAP_USD,
    MATCHED_SEED,
    NO,
    NOUL_CONVERSION_LABEL,
    NINE_MODELS,
    NOUL_CONVERSION_NOTE,
    YES,
    MatchedCostGuard,
    MatchedError,
    MatchedExecutor,
    _deterministic_sha256,
    baseline_questions,
    build_baseline_request,
    call_cost_usd,
    estimate_input_tokens,
    load_catalog_prices,
    logical_id,
    run_id_for,
    sample_arc_matched,
    sample_mmlu_matched,
    split_logical_id,
)
from jev_observatory.matched_report import (
    EQUIVALENCE_TOLERANCE,
    MatchedReportError,
    build_chart_tables,
    gold_records,
    jev_correct_flags,
)
from jev_observatory.schema import ChoiceQuestion, SystemOneRequest


def _sha(payload: dict) -> str:
    from hashlib import sha256

    return sha256(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


# ------------------------------------------------------------- tiny fixtures
def _choice_item(item_id: str, keys: list[str], gold: str, state: str = "Pick one.") -> dict:
    return {
        "id": item_id,
        "group": "g1",
        "cluster": item_id,
        "condition": "native",
        "state": state,
        "leakage_check": False,
        "gold": {"q": {"value": gold}},
        "questions": {"q": {"type": "choice",
                            "instructions": "Answer the multiple-choice question. Choose the single best option.",
                            "criteria": {k: f"option {k}" for k in keys}}},
    }


def _noul_item(item_id: str, gold_bool: bool) -> dict:
    return {
        "id": item_id,
        "group": "fresh_v3_dates",
        "cluster": "fresh_v3_dates:4242",
        "condition": "role=fresh_v3;family=synthetic_dates;seed=4242",
        "state": "Planning calendar rules: every month has 30 days. The review is on day 5; the audit is 10 days later.",
        "leakage_check": False,
        "gold": {"answer": {"value": gold_bool}},
        "questions": {"answer": {"type": "noul",
                                 "instructions": "Based only on the rules stated, answer yes or no: do they share a month?"}},
    }


def _mini_freeze(tmp_path: Path, *, n_mmlu: int = 2, n_arc: int = 1, n_fresh: int = 2,
                 models: list[str] | None = None, hard_cap_usd: float = HARD_COST_CAP_USD) -> dict:
    models = models or ["vendor/model-x"]
    mmlu_items = [_choice_item(f"mmlu-{i}", ["A", "B", "C"], "B") for i in range(n_mmlu)]
    arc_items = [_choice_item(f"arc-{i}", ["1", "2", "3"], "2") for i in range(n_arc)]
    fresh_items = [_choice_item("fresh-mi-0", ["opt_0", "opt_1"], "opt_1",
                                state="Registry: Levan's desk is on floor 2."),
                   _noul_item("fresh-dates-0", False)]
    from jev_observatory.openrouter import WireConfig, build_chat_payload

    wire_sha: dict[str, str] = {}
    est: dict[str, int] = {}
    conversions: dict[str, str] = {}
    order: dict[str, list[str]] = {}
    for model in models:
        wire = WireConfig(model=model, reasoning_effort="low", max_output_tokens=1024)
        lids: list[str] = []
        for dataset, items in (("mmlu", mmlu_items), ("arc", arc_items), ("fresh", fresh_items)):
            for item in items:
                request, conversion = build_baseline_request(item, model=model)
                payload = build_chat_payload(request, wire)
                lid = logical_id(model, dataset, str(item["id"]))
                wire_sha[lid] = _sha(payload)
                est[lid] = estimate_input_tokens(len(json.dumps(payload, ensure_ascii=False)))
                if conversion:
                    conversions[lid] = conversion
                lids.append(lid)
        order[model] = lids
    pilot_ids = {m: order[m][:2] for m in models}
    doc = {
        "freeze_version": "matched-freeze-1.0.0",
        "executor_version": "matched-executor-1.0.0",
        "created_at": "2026-09-20T00:00:00+00:00",
        "seed": MATCHED_SEED,
        "condition": {"name": "low-effort bounded direct answer",
                      "reasoning_effort": "low", "max_output_tokens": 1024,
                      "protocol": "standard chat API, one attempt, strict exact-key parsing, no retries",
                      "note": "test fixture"},
        "models": models,
        "wire": {"reasoning_effort": "low", "max_output_tokens": 1024},
        "datasets": {
            "mmlu": {"n": n_mmlu, "sampling": {"population_sizes": {"g1": n_mmlu}},
                     "source": {"file": "fixture", "sha256": "0" * 64, "url": "fixture"},
                     "items": mmlu_items},
            "arc": {"n": n_arc, "sampling": {}, "source": {"file": "fixture", "sha256": "0" * 64, "url": "fixture"},
                    "items": arc_items},
            "fresh": {"n": n_fresh, "sampling": {}, "items": fresh_items},
        },
        "noul_conversion": {"label": NOUL_CONVERSION_LABEL, "note": NOUL_CONVERSION_NOTE,
                            "yes_key": YES, "no_key": NO},
        "jev_join": {"mmlu": {}, "arc": {}, "fresh": {}},
        "catalog": {"catalog_file": "docs/modern-comparison/openrouter-catalog-nine-20260919.json",
                    "reasoning_effort_supported": {m: True for m in models}, "effort_flag_note": "fixture"},
        "cost_guard": {"hard_cap_usd": hard_cap_usd, "estimates": {}},
        "pilot_n": 2,
        "pilot_request_ids": pilot_ids,
        "request_order": order,
        "wire_sha256": wire_sha,
        "est_input_tokens": est,
        "wire_conversions": conversions,
        "execution_safety": {},
    }
    doc["deterministic_sha256"] = _deterministic_sha256(doc)
    return doc


def _write_result(run_dir: Path, model: str, dataset: str, item_id: str, *,
                  status: str = "ok", choice: str | None = "B",
                  probabilities: dict | None = None, latency: float = 100.0,
                  usable: bool = True) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "logical_request_id": logical_id(model, dataset, item_id),
        "terminal": True,
        "status": status,
        "item_id": item_id,
        "model_returned": model,
        "predictions": {"answer": {"type": "choice", "usable": usable, "choice": choice}},
        "latency_ms_total": latency,
    }
    if probabilities is not None:
        record["predictions"]["answer"]["probabilities"] = probabilities
    with (run_dir / "results.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def _write_attempt(run_dir: Path, model: str, dataset: str, item_id: str, *,
                   usage_in: int | None = 100, usage_out: int | None = 10,
                   cost: float | None = 0.001, finish: str = "stop",
                   outcome: str = "ok") -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "attempt_id": f"{logical_id(model, dataset, item_id)}.0",
        "logical_request_id": logical_id(model, dataset, item_id),
        "model_requested": model,
        "estimated_input_tokens": 500,
        "usage_input_tokens": usage_in,
        "usage_output_tokens": usage_out,
        "extra": {"wire_config": {"model": model, "reasoning_effort": "low",
                                  "max_output_tokens": 1024},
                  "usage_raw": {"prompt_tokens": usage_in, "completion_tokens": usage_out,
                                "cost": cost},
                  "finish_reason": finish},
        "outcome": outcome,
    }
    with (run_dir / "attempts.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


# ------------------------------------------------------------------- tests
def test_smoke_models_and_matched_models_are_identical():
    spec = importlib.util.spec_from_file_location(
        "smoke_openrouter", Path("scripts/benchmark/smoke_openrouter.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert list(NINE_MODELS) == list(module.NINE_MODELS)


def test_numeric_source_keys_preserved_never_relabeled():
    item = _choice_item("arc-x", ["1", "2", "3"], "2")
    request, conversion = build_baseline_request(item, model="vendor/m")
    assert conversion is None
    question = request.questions["q"]
    # actual source option keys, source order, numeric strings preserved
    assert list(question.criteria) == ["1", "2", "3"]
    assert request.request_sha256() == _sha_request(item, model="vendor/m")


def test_wire_order_preserved_and_never_sorted():
    # deliberate non-sorted insertion order must survive serialization
    item = _choice_item("w-1", ["C", "A", "B"], "A")
    request, _ = build_baseline_request(item, model="vendor/m")
    content = request.questions["q"].criteria
    assert list(content) == ["C", "A", "B"]
    from jev_observatory.openrouter import build_chat_payload, WireConfig

    payload = build_chat_payload(request, WireConfig(model="vendor/m", reasoning_effort="low",
                                                     max_output_tokens=1024))
    rendered = payload["messages"][0]["content"]
    assert rendered.index("C. option C") < rendered.index("A. option A") < rendered.index("B. option B")
    # freeze round-trip: JSON dump/reload keeps dict order, hash stable
    round_trip = json.loads(json.dumps(payload))
    assert json.dumps(round_trip, ensure_ascii=False, separators=(",", ":")) == \
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def test_bool_label_conversion_is_explicit():
    item = _noul_item("fresh-dates-0", False)
    converted, conversion = baseline_questions(item)
    assert conversion == NOUL_CONVERSION_LABEL
    question = converted["answer"]
    assert question["type"] == "choice"
    assert list(question["criteria"]) == [YES, NO]
    assert question["instructions"] == item["questions"]["answer"]["instructions"]
    gold = gold_records([item], "fresh")["fresh-dates-0"]
    assert gold["value"] == NO and gold["label_conversion"] == NOUL_CONVERSION_LABEL
    true_item = _noul_item("fresh-dates-1", True)
    gold_true = gold_records([true_item], "fresh")["fresh-dates-1"]
    assert gold_true["value"] == YES


def test_choice_questions_pass_through_verbatim():
    item = _choice_item("c-1", ["A", "B"], "A")
    converted, conversion = baseline_questions(item)
    assert conversion is None
    assert converted == item["questions"]


def test_weighted_sampling_totals_and_reproducibility():
    from jev_observatory.datasets.mmlu_pro import stratum_sizes

    population = ([_choice_item(f"mmlu-a-{i}", ["A", "B"], "B", state=f"q {i}")
                   for i in range(150)] +
                  [_choice_item(f"mmlu-b-{i}", ["A", "B"], "A", state=f"q {i}")
                   for i in range(50)])
    for i, item in enumerate(population):
        item["group"] = "bio" if i < 150 else "chem"
    sample, meta = sample_mmlu_matched(population, n=40, seed=MATCHED_SEED)
    assert len(sample) == 40
    assert sum(v["sampled"] for v in meta["per_subject"].values()) == 40
    for name, block in meta["per_subject"].items():
        assert block["sampled"] == round(40 * block["population"] / 200)
        assert block["inclusion_probability"] == block["sampled"] / block["population"]
    # same seed -> same sample; success/failure never observed
    again, _ = sample_mmlu_matched(population, n=40, seed=MATCHED_SEED)
    assert [it["id"] for it in again] == [it["id"] for it in sample]
    # not equal allocation: proportional strata differ
    assert meta["per_subject"]["bio"]["sampled"] != meta["per_subject"]["chem"]["sampled"]


def test_arc_srs_same_seed_family_reproducible(tmp_path):
    items = [_choice_item(f"arc-{i}", ["1", "2", "3"], "2") for i in range(50)]
    sample, meta = sample_arc_matched(items, n=10, seed=MATCHED_SEED)
    assert len(sample) == 10
    assert meta["mode"] == "simple_random_sample"
    again, _ = sample_arc_matched(items, n=10, seed=MATCHED_SEED)
    assert [it["id"] for it in again] == [it["id"] for it in sample]


def test_missing_gold_is_hard_error_and_predictions_stay_gold_free(tmp_path):
    bad = _choice_item("g-1", ["A", "B"], "B")
    bad["gold"] = {}
    with pytest.raises(MatchedReportError, match="no gold value"):
        gold_records([bad], "mmlu")
    # structural gold isolation: outbound payload keys can never contain gold
    request, _ = build_baseline_request(_choice_item("g-2", ["A", "B"], "A"), model="m")
    payload = request.to_payload()
    assert set(payload) == {"state", "model", "questions"}
    assert set(next(iter(payload["questions"].values()))) <= {"type", "instructions", "criteria"}
    # a result whose item is not in the frozen gold set is refused
    freeze = _mini_freeze(tmp_path)
    run_dir = tmp_path / run_id_for(freeze, "vendor/model-x")
    _write_result(run_dir, "vendor/model-x", "mmlu", "ghost-item", choice="A")
    from jev_observatory.matched_report import score_model_dataset

    prices, _ = load_catalog_prices()
    with pytest.raises(MatchedReportError, match="not in the frozen gold set"):
        score_model_dataset(run_dir, freeze, "vendor/model-x", "mmlu", prices=prices)


def test_all_item_denominator_includes_failures(tmp_path):
    freeze = _mini_freeze(tmp_path)
    model = "vendor/model-x"
    run_dir = tmp_path / run_id_for(freeze, model)
    # 1 valid correct + 1 strict-format failure -> all-request denominator
    _write_result(run_dir, model, "mmlu", "mmlu-0", choice="B")
    _write_result(run_dir, model, "mmlu", "mmlu-1", status="contract_invalid", choice=None)
    _write_attempt(run_dir, model, "mmlu", "mmlu-0")
    _write_attempt(run_dir, model, "mmlu", "mmlu-1")
    from jev_observatory.matched_report import score_model_dataset

    prices, _ = load_catalog_prices()
    block = score_model_dataset(run_dir, freeze, model, "mmlu", prices=prices)
    head = block["headline"]
    assert head["n_requested"] == 2
    assert head["correct_all"] == 1
    assert head["accuracy_all"] == 0.5
    assert head["accuracy_conditional"] == 1.0
    assert head["n_format_failed"] == 1 and head["n_error"] == 0 and head["n_missing"] == 0


def test_no_synthetic_probabilities_for_text_baselines(tmp_path):
    freeze = _mini_freeze(tmp_path)
    model = "vendor/model-x"
    run_dir = tmp_path / run_id_for(freeze, model)
    _write_result(run_dir, model, "mmlu", "mmlu-0", choice="B",
                  probabilities={"A": 0.2, "B": 0.8})
    from jev_observatory.matched_report import modern_outcomes

    gold = gold_records(freeze["datasets"]["mmlu"]["items"], "mmlu")
    results = [json.loads(line) for line in
               (run_dir / "results.jsonl").read_text().split("\n") if line.strip()]
    with pytest.raises(MatchedReportError, match="probability array"):
        modern_outcomes(results, gold, "mmlu")


def test_malformed_model_output_is_counted_not_repaired(tmp_path):
    from jev_observatory.openrouter import OpenRouterProvider, WireConfig
    from jev_observatory.transport import ScriptedTransport

    item = _choice_item("m-1", ["A", "B", "C"], "B")
    request, _ = build_baseline_request(item, model="vendor/model-x")
    provider = OpenRouterProvider(
        ScriptedTransport([{"status_code": 200, "body": {
            "id": "x", "model": "vendor/model-x",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant",
                                     "content": "The answer is B because..."}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8}}}]),
        wire=WireConfig(model="vendor/model-x", reasoning_effort="low", max_output_tokens=1024))
    outcome = provider.ask(request, logical_id("vendor/model-x", "mmlu", "m-1"))
    assert outcome.status == "contract_invalid"
    assert len(outcome.attempts) == 1  # never retried
    assert outcome.validated.codes == ["strict_format_not_exact_key"]
    assert outcome.attempts[-1].extra["finish_reason"] == "stop"


def test_paired_score_hand_fixtures():
    from jev_observatory.matched_report import paired_analysis

    gold = {f"it{i}": {"cluster": f"c{i}", "value": "A"} for i in range(10)}
    modern = {f"it{i}": i % 2 == 0 for i in range(10)}      # 5 correct
    jev = {f"it{i}": i % 2 == 1 for i in range(10)}         # 5 correct, fully discordant
    doc = paired_analysis(modern, jev, gold)
    assert doc["paired_n"] == 10
    assert doc["b01_jev_right_model_wrong"] == 5
    assert doc["b10_model_right_jev_wrong"] == 5
    assert doc["point_diff_model_minus_jev"] == 0.0
    from jev_observatory.paired import mcnemar_exact

    assert doc["mcnemar_exact_p"] == mcnemar_exact(5, 5)
    assert doc["ties_concordant"] == 0
    # identical series: all ties, exact p=1, paired interval collapses to 0 -> equivalence
    same = {f"it{i}": i % 2 == 0 for i in range(400)}
    doc_tie = paired_analysis(same, same, {f"it{i}": {"cluster": f"c{i}", "value": "A"}
                                           for i in range(400)})
    assert doc_tie["ties_concordant"] == 400
    assert doc_tie["mcnemar_exact_p"] == 1.0
    assert doc_tie["equivalent_within_3pp"] is True
    assert doc_tie["bootstrap"]["ci95_low"] == doc_tie["bootstrap"]["ci95_high"] == 0.0
    # dominant Jev: model far behind -> NOT equivalent within 3pp
    doc_gap = paired_analysis({f"it{i}": False for i in range(400)},
                              {f"it{i}": True for i in range(400)},
                              {f"it{i}": {"cluster": f"c{i}", "value": "A"} for i in range(400)})
    assert doc_gap["equivalent_within_3pp"] is False
    assert doc_gap["bootstrap"]["ci95_high"] < -EQUIVALENCE_TOLERANCE


def test_holm_family_across_nine_comparisons():
    from jev_observatory.matched_report import paired_analysis

    gold = {f"it{i}": {"cluster": f"c{i}", "value": "A"} for i in range(120)}
    p_values = {}
    for m in range(9):
        modern = {f"it{i}": i % 2 == (m % 2) for i in range(120)}
        jev = {f"it{i}": True for i in range(120)}
        p_values[f"model-{m}"] = paired_analysis(modern, jev, gold)["mcnemar_exact_p"]
    from jev_observatory.mech_dryrun import holm_stepdown

    adjusted = holm_stepdown(p_values)
    assert set(adjusted) == set(p_values)
    assert all(adjusted[k] >= p_values[k] for k in p_values)
    assert max(adjusted.values()) <= 1.0


def test_cost_guard_shared_cap_and_unknown_usage():
    prices = {"vendor/model-x": {"input_per_token": 1e-3, "output_per_token": 1e-3}}
    guard = MatchedCostGuard(prices, hard_cap_usd=0.03, max_output_tokens=10,
                             max_requests=100)
    # reservation = est_input*in + max_output*out = 10*1e-3 = 0.01 per call
    r1 = guard.reserve("vendor/model-x", est_input_tokens=0)
    assert abs(r1.cost_usd - 0.01) < 1e-12
    r2 = guard.reserve("vendor/model-x", est_input_tokens=0)  # ceiling 0.02 <= 0.03
    # a big reservation would push the shared aggregate ceiling past the cap
    with pytest.raises(MatchedError, match="hard cost cap"):
        guard.reserve("vendor/model-x", est_input_tokens=10)  # 0.02 more -> 0.04
    # commit with provider-reported cost: reservation released, cost billed
    billed = guard.commit(r1, input_tokens=10, output_tokens=5,
                          usage_raw={"cost": 0.004})
    assert billed["reported"] == 0.004
    snap = guard.snapshot()
    assert snap["billed_reported_usd"] == pytest.approx(0.004)
    assert snap["reserved_usd"] == pytest.approx(0.01)  # r2 still in flight
    assert snap["ceiling_usd"] == pytest.approx(0.014)
    # unknown usage is never counted as free: it keeps a reservation
    guard.release(r2, keep_reserved=True)
    assert guard.snapshot()["unknown_reserved_usd"] == pytest.approx(0.01)
    # catalog-estimated fallback when the provider does not report cost
    r3 = guard.reserve("vendor/model-x", est_input_tokens=0)  # ceiling 0.024 <= 0.03
    billed2 = guard.commit(r3, input_tokens=10, output_tokens=5, usage_raw={})
    assert billed2["reported"] is None
    assert billed2["estimated"] == pytest.approx(15 * 1e-3)
    assert guard.snapshot()["billed_estimated_usd"] == pytest.approx(0.015)
    # unknown usage keeps a reservation: the cap binds again once room is gone
    with pytest.raises(MatchedError, match="hard cost cap"):
        guard.reserve("vendor/model-x", est_input_tokens=0)  # 0.004+0.015+0.01+0.01 > 0.03


def test_cost_guard_max_requests_and_restore():
    prices = {"vendor/model-x": {"input_per_token": 1e-3, "output_per_token": 1e-3}}
    guard = MatchedCostGuard(prices, hard_cap_usd=1e9, max_output_tokens=1,
                             max_requests=1)
    guard.reserve("vendor/model-x", est_input_tokens=0)
    with pytest.raises(MatchedError, match="max_requests"):
        guard.reserve("vendor/model-x", est_input_tokens=0)
    # resume restores committed costs from the stored attempt ledger
    guard2 = MatchedCostGuard(prices, hard_cap_usd=1e9, max_output_tokens=10)
    guard2.restore([{"model_requested": "vendor/model-x",
                     "usage_input_tokens": 100, "usage_output_tokens": 5,
                     "estimated_input_tokens": 200,
                     "extra": {"usage_raw": {"cost": 0.02}}}])
    assert guard2.snapshot()["billed_reported_usd"] == pytest.approx(0.02)


def test_resume_counts_pilot_once_and_restores_budget(tmp_path, monkeypatch):
    freeze = _mini_freeze(tmp_path, models=["vendor/model-x"])
    model = "vendor/model-x"
    run_dir = tmp_path / run_id_for(freeze, model)
    pilot_ids = freeze["pilot_request_ids"][model]
    for lid in pilot_ids:
        _m, dataset, item_id = split_logical_id(lid)
        _write_result(run_dir, model, dataset, item_id, choice="B")
        _write_attempt(run_dir, model, dataset, item_id, cost=0.01)
    monkeypatch.setenv("JEVO_ALLOW_LIVE", "1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-not-a-real-key")
    monkeypatch.setattr("jev_observatory.matched.load_catalog_prices",
                        lambda *a, **k: ({"vendor/model-x": {
                            "input_per_token": 1e-3, "output_per_token": 1e-3}}, {}))
    monkeypatch.setattr(MatchedExecutor, "_new_provider",
                        lambda self, freeze, model: _ScriptedFakeProvider(freeze, model))
    executor = MatchedExecutor(tmp_path, freeze=freeze, models=[model],
                               max_wall_seconds=60.0)
    state = executor.run(live=True)
    stage = state["stages"][model]
    # pilot items are NEVER re-dispatched on continuation
    assert stage["skipped_resume"] == len(pilot_ids)
    assert stage["dispatched"] == len(freeze["request_order"][model]) - len(pilot_ids)
    assert state["global_stop_reason"] is None
    # restored pilot spend stays in the ledger, plus the new dispatch costs
    expected = 0.01 * len(pilot_ids) + 0.01 * (len(freeze["request_order"][model]) - len(pilot_ids))
    assert state["guard_final"]["billed_reported_usd"] == pytest.approx(expected)


def test_uncertain_timeout_fails_resume_closed(tmp_path, monkeypatch):
    freeze = _mini_freeze(tmp_path, models=["vendor/model-x"])
    model = "vendor/model-x"
    run_dir = tmp_path / run_id_for(freeze, model)
    run_dir.mkdir(parents=True, exist_ok=True)
    lid = freeze["request_order"][model][0]
    _m, dataset, item_id = split_logical_id(lid)
    _write_attempt(run_dir, model, dataset, item_id, outcome="timeout")
    # mark it uncertain like the ledger does for timeouts
    lines = (run_dir / "attempts.jsonl").read_text().split("\n")
    records = [json.loads(l) for l in lines if l.strip()]
    records[0]["uncertain"] = True
    (run_dir / "attempts.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n")
    monkeypatch.setenv("JEVO_ALLOW_LIVE", "1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-not-a-real-key")
    executor = MatchedExecutor(tmp_path, freeze=freeze, models=[model])
    with pytest.raises(MatchedError, match="fail-closed resume refusal"):
        executor.run(live=True)


def test_restored_budget_over_cap_refuses_resume(tmp_path, monkeypatch):
    freeze = _mini_freeze(tmp_path, models=["vendor/model-x"], hard_cap_usd=0.02)
    model = "vendor/model-x"
    run_dir = tmp_path / run_id_for(freeze, model)
    lid = freeze["request_order"][model][0]
    _m, dataset, item_id = split_logical_id(lid)
    _write_attempt(run_dir, model, dataset, item_id, cost=0.9)
    monkeypatch.setenv("JEVO_ALLOW_LIVE", "1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-not-a-real-key")
    monkeypatch.setattr("jev_observatory.matched.load_catalog_prices",
                        lambda *a, **k: ({"vendor/model-x": {
                            "input_per_token": 1e-3, "output_per_token": 1e-3}}, {}))
    executor = MatchedExecutor(tmp_path, freeze=freeze, models=[model],
                               hard_cap_usd=0.02, max_wall_seconds=60.0)
    with pytest.raises(MatchedError, match="exceeds the"):
        executor.run(live=True)


def test_dry_run_verifies_all_maps_offline(tmp_path):
    freeze = _mini_freeze(tmp_path, models=["vendor/model-x"])
    executor = MatchedExecutor(tmp_path, freeze=freeze, max_wall_seconds=60.0)
    state = executor.run(live=False)
    assert state["live"] is False
    assert "dry run" in state["note"]
    assert state["global_stop_reason"] is None
    # every frozen wire hash was rebuilt and compared above (would raise otherwise)
    assert len(freeze["wire_sha256"]) == 5  # 2 mmlu + 1 arc + 2 fresh


def test_live_requires_opt_in_and_env_key(tmp_path):
    freeze = _mini_freeze(tmp_path)
    executor = MatchedExecutor(tmp_path, freeze=freeze, max_wall_seconds=60.0)
    with pytest.raises(MatchedError, match="JEVO_ALLOW_LIVE"):
        executor.run(live=True)


def test_jev_noul_threshold_recorded_separately():
    gold = {"d1": {"value": "Yes", "native_value": True, "is_noul": True,
                   "label_conversion": NOUL_CONVERSION_LABEL},
            "d2": {"value": "No", "native_value": False, "is_noul": True,
                   "label_conversion": NOUL_CONVERSION_LABEL},
            "d3": {"value": "No", "native_value": False, "is_noul": True,
                   "label_conversion": NOUL_CONVERSION_LABEL}}
    records = {
        "d1": {"terminal": True, "status": "ok",
                "predictions": {"answer": {"type": "noul", "usable": True, "noul": 0.67}}},
        "d2": {"terminal": True, "status": "ok",
                "predictions": {"answer": {"type": "noul", "usable": True, "noul": 0.49}}},
    }
    flags, reasons = jev_correct_flags(records, gold)
    assert flags == {"d1": True, "d2": True}   # 0.49 >= 0.5 is False == gold False
    # an unusable Jev prediction is excluded with a recorded reason, never guessed
    records["d1"]["predictions"]["answer"]["usable"] = False
    flags, reasons = jev_correct_flags(records, gold)
    assert "d1" not in flags and reasons["d1"] == "jev_prediction_unusable"


class _ScriptedFakeProvider:
    """Deterministic offline provider used to exercise executor resume/stop paths."""

    name = "scripted-fake"

    def __init__(self, freeze, model) -> None:
        self.wire = freeze["wire"]
        self.model = model

    def ask(self, request, logical_request_id, *, hook=None, cancellation=None,
            new_attempt_id=lambda lid, i: f"{lid}.{i}", store=None):
        from jev_observatory.providers import AttemptRecord, CallOutcome

        attempt = AttemptRecord(
            attempt_id=new_attempt_id(logical_request_id, 0),
            logical_request_id=logical_request_id, attempt_index=0,
            provider=self.name, model_requested=self.model,
            started_at="t", finished_at="t", latency_ms=1.0, first_byte_ms=None,
            cold_connection=None, http_status=200, outcome="ok", uncertain=False,
            error=None, usage_input_tokens=100, usage_output_tokens=10,
            estimated_input_tokens=500, response_bytes=10,
            request_sha256=request.request_sha256(),
        )
        from jev_observatory.validation import ValidatedAnswer, ValidatedResponse

        qid = next(iter(request.questions))
        attempt.extra = {"finish_reason": "stop", "usage_raw": {
            "prompt_tokens": 100, "completion_tokens": 10, "cost": 0.01}}
        validated = ValidatedResponse(
            contract_valid=True, usable=True, requested_model=self.model,
            returned_model=self.model,
            answers={qid: ValidatedAnswer(question_id=qid, question_type="choice",
                                          usable=True, raw="B", values={"choice": "B"},
                                          violations=[])},
            violations=[], usage={}, raw={})
        return CallOutcome(
            logical_request_id=logical_request_id, status="ok", terminal=True,
            attempts=[attempt], validated=validated,
            usage_input_tokens=100, usage_output_tokens=10, total_latency_ms=1.0,
            n_retries=0, request_sha256=request.request_sha256())

    def close(self) -> None:
        return None


def test_chart_tables_structure():
    report = {
        "comparison_table": {"m": {d: {"accuracy_all": 0.5, "accuracy_conditional": 0.6,
                                       "accuracy_weighted": None, "n_requested": 2,
                                       "correct": 1} for d in DATASETS}},
        "per_model": {"m": {"blocks": {d: {
            "headline": {"n_error": 0, "n_format_failed": 1, "n_requested": 2},
            "cost": {"billed_reported_usd": 0.1, "billed_estimated_usd": 0.0,
                     "n_unknown_usage": 0},
            "latency": {"mean_ms": 10.0, "p95_ms": 20.0},
            "usage_totals": {"reasoning_tokens": None},
            "by_group": {"g1": {"accuracy_conditional": 0.6}},
        } for d in DATASETS}}},
        "order_robustness": {"matched_base_items": 0},
        "reviewed_references": {},
        "generation_addendum": {},
        "jev_native_full_scores": {"mmlu": {"by_group": {"g1": {"accuracy": 0.9}}}},
    }
    tables = build_chart_tables(report)
    assert tables["matched_heatmap"]["rows"] == ["m"]
    assert tables["matched_heatmap"]["columns"] == list(DATASETS)
    assert tables["error_rate_cost_latency"]["m"]["mmlu"]["error_rate"] == 0.5
    assert tables["mmlu_subject_breakdown"]["jev_full"] == {"g1": 0.9}
    assert tables["mmlu_subject_breakdown"]["m"]["g1"] == 0.6


def test_pricing_flags_unsupported_effort():
    prices, flags = load_catalog_prices()
    assert set(prices) == set(NINE_MODELS)
    # qwen3.8-flash lists 'reasoning' but NOT 'reasoning_effort' -> must be flagged
    assert flags["reasoning_effort_supported"]["qwen/qwen3.8-flash"] is False
    assert flags["reasoning_effort_supported"]["openai/gpt-6-astra"] is True


def test_call_cost_reported_wins_and_unknown_is_none():
    prices = {"m": {"input_per_token": 2e-6, "output_per_token": 1e-5}}
    reported, estimated = call_cost_usd(prices, "m", input_tokens=100,
                                        output_tokens=10, usage_raw={"cost": 0.0007})
    assert reported == 0.0007
    assert estimated == pytest.approx(100 * 2e-6 + 10 * 1e-5)
    reported, estimated = call_cost_usd(prices, "m", input_tokens=100,
                                        output_tokens=10, usage_raw={})
    assert reported is None and estimated == pytest.approx(100 * 2e-6 + 10 * 1e-5)
    reported, estimated = call_cost_usd(prices, "m", input_tokens=None,
                                        output_tokens=None, usage_raw=None)
    assert reported is None and estimated is None


def test_logical_id_round_trip():
    lid = logical_id("vendor/model-x", "arc", "ARCCH_1")
    assert split_logical_id(lid) == ("vendor/model-x", "arc", "ARCCH_1")
    with pytest.raises(MatchedError):
        split_logical_id("bad-id")


def _sha_request(item: dict, model: str) -> str:
    from jev_observatory.schema import SystemOneRequest

    converted, _ = baseline_questions(item)
    return SystemOneRequest(state=item["state"], model=model,
                            questions=converted).request_sha256()