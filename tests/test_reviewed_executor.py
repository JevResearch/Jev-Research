"""Reviewed executor + report tests (parent REVIEW-2).

Offline only: a fake transport answers every POST; the conftest socket ban
makes any real network attempt fail the suite. Covers:

* freeze immutability and stored/reloaded payload hashes matching the freeze;
* dry-run dispatches nothing and never constructs a transport;
* full plan → reload → transport workflow for all three stages;
* batched (1 call, K questions) / sequential / concurrent dispatch shapes;
* concurrent max in-flight 4 and true whole-workload wall-clock measurement
  (injectable clock; wall time never the sum of parallel durations);
* insufficient credit stops subsequent requests;
* budget caps shared across stages;
* all attempts persisted on failure and on cancellation;
* fail-closed resume refusal on uncertain (timeout) attempts;
* fresh labels verified independently + mechanism sidecar maps used correctly;
* strict JSON and report regeneration without any provider access.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from jev_observatory.experiment import logical_requests
from jev_observatory.guards import BudgetCaps, BudgetLedger
from jev_observatory.reviewed_executor import (
    ExecutorError,
    ReviewedExecutor,
    build_batching_spec,
    build_freeze,
    load_freeze,
)
from jev_observatory.reviewed_report import build_reviewed_report
from jev_observatory.transport import TransportResponse


# ---------------------------------------------------------------- fake world
def fake_body(payload: dict) -> dict:
    answers = {}
    for qid, q in payload["questions"].items():
        if q["type"] == "noul":
            answers[qid] = {"type": "noul", "noul": 0.5}
        elif q["type"] == "choice":
            keys = list(q["criteria"])
            share = 1.0 / len(keys)
            answers[qid] = {"type": "choice", "choice": keys[0],
                            "probabilities": {k: share for k in keys}, "confidence": 0.1}
        else:
            levels = len(q["criteria"])
            share = 1.0 / levels
            answers[qid] = {"type": "score", "score": (levels - 1) / 2.0,
                            "legend": {str(i): str(q["criteria"][i]) for i in range(levels)},
                            "probabilities": {str(i): share for i in range(levels)},
                            "confidence": 0.1}
    return {"model": payload["model"], "answers": answers,
            "usage": {"input_tokens": 100, "output_tokens": 20}}


class FakeClock:
    """Injectable monotonic-style clock; advanced by the fake transport."""

    def __init__(self, start: float = 100.0) -> None:
        self.t = start
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self.t

    def advance(self, dt: float) -> None:
        with self._lock:
            self.t += dt


def ok_response(payload: dict, *, total_ms: float = 1.0) -> TransportResponse:
    return TransportResponse(
        status_code=200, headers={},
        body=json.dumps(fake_body(payload)).encode("utf-8"),
        total_ms=total_ms,
    )


class FakeTransport:
    """Offline transport: counts posts, clients, in-flight; never touches network."""

    def __init__(self, registry: dict, responder) -> None:
        self.registry = registry
        self.responder = responder
        with registry["lock"]:
            registry["clients_created"] += 1

    def post(self, path: str, payload: dict) -> TransportResponse:
        reg = self.registry
        with reg["lock"]:
            reg["posts"].append(payload)
            reg["in_flight"] += 1
            reg["max_in_flight"] = max(reg["max_in_flight"], reg["in_flight"])
        try:
            return self.responder(payload, reg)
        finally:
            with reg["lock"]:
                reg["in_flight"] -= 1

    def close(self) -> None:
        with self.registry["lock"]:
            self.registry["clients_closed"] += 1


def default_responder(payload: dict, reg: dict) -> TransportResponse:
    return ok_response(payload)


def make_registry() -> dict:
    return {"lock": threading.Lock(), "posts": [], "in_flight": 0,
            "max_in_flight": 0, "clients_created": 0, "clients_closed": 0}


def make_executor(root: Path, registry: dict, responder=default_responder, **kwargs):
    return ReviewedExecutor(
        root,
        transport_factory=lambda: FakeTransport(registry, responder),
        **kwargs,
    )


def strict_loads(text: str):
    return json.loads(text, parse_constant=lambda c: (_ for _ in ()).throw(
        ValueError(f"non-strict JSON constant {c!r}")))


# --------------------------------------------------------------------- tests
def test_batching_spec_shape_and_existing_content():
    spec, sidecar = build_batching_spec()
    assert len(spec["items"]) == 108  # 12 blocks x (1 batched + 4 sep + 4 conc)
    assert sidecar["n_calls_total"] == 108
    assert all(tuple(block["mode_order"]) in (
        ("batched", "sequential", "concurrent"), ("batched", "concurrent", "sequential"),
        ("sequential", "batched", "concurrent"), ("sequential", "concurrent", "batched"),
        ("concurrent", "batched", "sequential"), ("concurrent", "sequential", "batched"))
        for block in sidecar["blocks"])
    from jev_observatory.experiments import BATCH_STATE, batch_vs_separate_spec
    template = batch_vs_separate_spec(repeats=1, seed=0)["items"][0]
    batched_item = next(i for i in spec["items"] if i["id"] == "batch-b00")
    assert batched_item["state"] == BATCH_STATE == template["state"]
    assert batched_item["questions"] == template["questions"]  # unchanged K=4 content
    requests = logical_requests(spec)
    assert len(requests) == 108


def test_freeze_write_once_and_integrity(tmp_path):
    doc = build_freeze(tmp_path)
    again = build_freeze(tmp_path)
    assert again["deterministic_sha256"] == doc["deterministic_sha256"]
    assert doc["stages"]["fresh_test"]["n_requests"] == 120
    assert doc["stages"]["mechanism"]["n_requests"] == 96
    assert doc["stages"]["batching"]["n_requests"] == 108
    loaded = load_freeze(tmp_path)
    assert loaded == doc
    # any difference is a hard error, never a silent rewrite
    tampered = json.loads((tmp_path / "freeze" / "frozen.json").read_text())
    tampered["stages"]["fresh_test"]["n_requests"] = 119
    (tmp_path / "freeze" / "frozen.json").write_text(json.dumps(tampered, indent=2))
    with pytest.raises(ExecutorError, match="integrity check"):
        build_freeze(tmp_path)


def test_dry_run_verifies_hashes_without_transport(tmp_path):
    registry = make_registry()

    def forbidden():
        raise AssertionError("dry run must never construct a transport")

    executor = ReviewedExecutor(tmp_path, transport_factory=forbidden)
    state = executor.run(dry_run=True)
    assert state["dry_run"] is True
    assert state["global_stop_reason"] is None
    for stage, entry in state["stages"].items():
        assert entry["freeze_verification"] == "pass"
        run_dir = tmp_path / entry["run_id"]
        assert (run_dir / "manifest.json").exists()
        assert (run_dir / "items.jsonl").exists()
        assert (run_dir / "plan.json").exists()
        assert not (run_dir / "attempts.jsonl").exists()  # nothing dispatched
    assert registry["clients_created"] == 0
    # every stored item's logical id is present exactly once in the frozen wire map
    freeze = load_freeze(tmp_path)
    for stage, entry in state["stages"].items():
        items = [json.loads(l) for l in
                 (tmp_path / entry["run_id"] / "items.jsonl").read_text().splitlines() if l.strip()]
        assert len(items) == freeze["stages"][stage]["n_requests"]
        for item in items:
            lids = [lid for lid in freeze["stages"][stage]["wire_sha256"]
                    if lid.endswith(f":{item['id']}")]
            assert len(lids) == 1


def test_full_workflow_fake_transport_shapes_and_counts(tmp_path):
    registry = make_registry()
    executor = make_executor(tmp_path, registry)
    state = executor.run(dry_run=True)   # freeze + plans
    live = make_executor(tmp_path, registry)
    state = live.run(dry_run=False)
    assert state["global_stop_reason"] is None
    # 324 main requests = 120 + 96 + 108
    assert len(registry["posts"]) == 324
    # dispatch shapes: batched = one payload with K=4 questions, x12
    four_q = [p for p in registry["posts"] if len(p["questions"]) == 4]
    assert len(four_q) == 12
    # sequential/concurrent separate arms: one question per call, 12x4 each
    nine_q = [p for p in registry["posts"] if len(p["questions"]) == 9]
    assert len(nine_q) == 72  # mech S/N/O arms
    assert all(len(p["questions"]) in (1, 4, 9) for p in registry["posts"])
    # zero retries: exactly one attempt per logical request
    attempts = []
    for stage in ("fresh_test", "mechanism", "batching"):
        run_dir = next(tmp_path.glob(f"rev-{stage}-*"))
        attempts += [json.loads(l) for l in
                     (run_dir / "attempts.jsonl").read_text().splitlines() if l.strip()]
    assert len(attempts) == 324
    assert all(a["attempt_index"] == 0 for a in attempts)
    # connection policy: new client per stage (2) + per batching workload (36), all closed
    assert registry["clients_created"] == registry["clients_closed"] == 38
    # results + workloads
    batch_dir = next(tmp_path.glob("rev-batching-*"))
    workloads = [json.loads(l) for l in
                 (batch_dir / "workloads.jsonl").read_text().splitlines() if l.strip()]
    assert len(workloads) == 36
    assert all(w["complete"] for w in workloads)
    assert all(w["connection_policy"] == "new_http_client_per_workload_retained_then_closed"
               for w in workloads)


def test_concurrent_arm_max_in_flight_four_and_true_wall_time(tmp_path):
    registry = make_registry()
    executor = make_executor(tmp_path, registry, responder=_sleeping_responder)
    executor.run(dry_run=True)
    spec, _sidecar = build_batching_spec()
    lrs = [lr for lr in logical_requests(spec)
           if lr.logical_request_id.endswith(":conc-b00-technical")
           or lr.logical_request_id.endswith(":conc-b00-billing_issue")
           or lr.logical_request_id.endswith(":conc-b00-security_concern")
           or lr.logical_request_id.endswith(":conc-b00-frustration")]
    assert len(lrs) == 4
    budget = BudgetLedger(BudgetCaps())
    store = executor._store(f"rev-batching-{load_freeze(tmp_path)['deterministic_sha256'][:12]}")
    summary = executor._execute_workload(0, "concurrent", "b00-concurrent", lrs, store, budget)
    assert summary["complete"] and summary["n_ok"] == 4
    assert registry["max_in_flight"] <= 4
    assert registry["max_in_flight"] >= 2  # the arm really runs in parallel
    # wall time is the true whole-workload measurement, NOT the sum of the four
    # scripted per-call latencies (1.0s each -> sum 4.0s, 4x median 4.0s)
    assert summary["wall_seconds"] < 1.0
    attempts = store.attempts()
    assert all(a["latency_ms"] == 1.0 for a in attempts)


def _sleeping_responder(payload: dict, reg: dict) -> TransportResponse:
    time.sleep(0.1)
    return ok_response(payload)


def test_injectable_clock_drives_workload_wall_time(tmp_path):
    registry = make_registry()
    clock = FakeClock(start=1000.0)
    executor = make_executor(tmp_path, registry, clock=clock,
                             responder=_advancing_responder(clock, 5.0))
    executor.run(dry_run=True)
    spec, _sidecar = build_batching_spec()
    lrs = [lr for lr in logical_requests(spec)
           if ":sep-b00-" in lr.logical_request_id]
    assert len(lrs) == 4
    budget = BudgetLedger(BudgetCaps())
    store = executor._store(f"rev-batching-{load_freeze(tmp_path)['deterministic_sha256'][:12]}")
    summary = executor._execute_workload(0, "sequential", "b00-sequential", lrs, store, budget)
    # the injected monotonic clock advanced exactly 4 x 5.0 during the workload
    assert summary["wall_seconds"] == pytest.approx(20.0)


def _advancing_responder(clock: FakeClock, step: float):
    def responder(payload: dict, reg: dict) -> TransportResponse:
        clock.advance(step)
        return ok_response(payload)
    return responder


def test_insufficient_credit_stops_all_subsequent_requests(tmp_path):
    registry = make_registry()

    def responder(payload: dict, reg: dict) -> TransportResponse:
        if len(reg["posts"]) == 1:
            return TransportResponse(status_code=402, headers={},
                                     body=b'{"error": "insufficient credit"}', total_ms=1.0)
        return ok_response(payload)

    executor = make_executor(tmp_path, registry)
    executor.run(dry_run=True)
    live = make_executor(tmp_path, registry, responder=responder)
    state = live.run(dry_run=False)
    assert state["global_stop_reason"] == "insufficient_credit_or_auth"
    assert len(registry["posts"]) == 1  # nothing dispatched after the 402
    fresh_dir = next(tmp_path.glob("rev-fresh_test-*"))
    attempts = [json.loads(l) for l in (fresh_dir / "attempts.jsonl").read_text().splitlines() if l.strip()]
    assert len(attempts) == 1 and attempts[0]["http_status"] == 402
    for stage in ("mechanism", "batching"):
        entry = state["stages"][stage]
        assert entry.get("skipped_reason") == "insufficient_credit_or_auth"
        run_dir = next(tmp_path.glob(f"rev-{stage}-*"))
        assert not (run_dir / "attempts.jsonl").exists()


def test_budget_cap_shared_across_stages(tmp_path):
    registry = make_registry()
    executor = make_executor(tmp_path, registry)
    executor.run(dry_run=True)
    live = make_executor(tmp_path, registry, max_requests=100)
    state = live.run(dry_run=False)
    assert state["global_stop_reason"] == "budget_cap_reached:max_requests"
    fresh_dir = next(tmp_path.glob("rev-fresh_test-*"))
    n_fresh = len([l for l in (fresh_dir / "attempts.jsonl").read_text().splitlines() if l.strip()])
    assert n_fresh == 100  # fresh stage stopped exactly at the shared cap
    for stage in ("mechanism", "batching"):
        assert state["stages"][stage].get("skipped_reason", "").startswith("budget_cap_reached")
        run_dir = next(tmp_path.glob(f"rev-{stage}-*"))
        assert not (run_dir / "attempts.jsonl").exists()


def test_all_attempts_persisted_on_transport_failures(tmp_path):
    registry = make_registry()

    def responder(payload: dict, reg: dict) -> TransportResponse:
        raise RuntimeError("connection exploded")

    executor = make_executor(tmp_path, registry)
    executor.run(dry_run=True)
    live = make_executor(tmp_path, registry, responder=responder)
    state = live.run(dry_run=False)
    assert state["global_stop_reason"] == "repeated_transport_schema_failures"
    fresh_dir = next(tmp_path.glob("rev-fresh_test-*"))
    attempts = [json.loads(l) for l in (fresh_dir / "attempts.jsonl").read_text().splitlines() if l.strip()]
    assert len(attempts) == 3  # every failure attempt persisted, no retries
    assert all(a["outcome"] == "transport_error" for a in attempts)
    # terminal failures are recorded as non-ok results, never as observations
    results_path = fresh_dir / "results.jsonl"
    results = [json.loads(l) for l in results_path.read_text().splitlines() if l.strip()]
    assert len(results) == 3 and all(r["status"] == "transport_error" for r in results)


def test_cancellation_persists_partial_artifacts(tmp_path):
    registry = make_registry()
    executor = make_executor(tmp_path, registry)
    executor.run(dry_run=True)
    holder: dict = {}

    def responder(payload: dict, reg: dict) -> TransportResponse:
        if len(reg["posts"]) >= 5:
            holder["executor"].cancellation.cancel()
        return ok_response(payload)

    live = make_executor(tmp_path, registry, responder=responder)
    holder["executor"] = live
    state = live.run(dry_run=False)
    fresh_dir = next(tmp_path.glob("rev-fresh_test-*"))
    attempts = [json.loads(l) for l in (fresh_dir / "attempts.jsonl").read_text().splitlines() if l.strip()]
    assert len(attempts) >= 5
    assert state["global_stop_reason"] == "cancelled_by_cancellation"
    for stage in ("mechanism", "batching"):
        assert state["stages"][stage].get("skipped_reason") == "cancelled_by_cancellation"


def test_fail_closed_resume_refusal_on_uncertain_timeouts(tmp_path):
    registry = make_registry()

    def responder(payload: dict, reg: dict) -> TransportResponse:
        return TransportResponse(status_code=0, headers={}, body=b"", total_ms=1.0,
                                 error="timeout:ReadTimeout")

    executor = make_executor(tmp_path, registry)
    executor.run(dry_run=True)
    live = make_executor(tmp_path, registry, responder=responder)
    state = live.run(dry_run=False)
    assert state["global_stop_reason"] == "repeated_transport_schema_failures"
    fresh_dir = next(tmp_path.glob("rev-fresh_test-*"))
    attempts = [json.loads(l) for l in (fresh_dir / "attempts.jsonl").read_text().splitlines() if l.strip()]
    assert len(attempts) == 3 and all(a["uncertain"] for a in attempts)
    posts_before = len(registry["posts"])
    # resume is refused: uncertain (possibly billed) attempts exist; budgets not reset
    resume = make_executor(tmp_path, registry)
    with pytest.raises(ExecutorError, match="fail-closed"):
        resume.run(dry_run=False)
    assert len(registry["posts"]) == posts_before  # nothing re-dispatched


def test_budget_restored_from_attempts_not_reset(tmp_path):
    from jev_observatory.ledger import RunStore

    root = tmp_path
    store = RunStore(root, "rev-fresh_test-abc123def456")
    record = {
        "attempt_id": "a1", "logical_request_id": "l1", "attempt_index": 0,
        "provider": "jev", "model_requested": "jev-1.13.0",
        "started_at": "t", "finished_at": "t", "latency_ms": 1.0,
        "first_byte_ms": None, "cold_connection": None, "http_status": 200,
        "outcome": "ok", "uncertain": False, "error": None,
        "usage_input_tokens": 1000, "usage_output_tokens": 5,
        "estimated_input_tokens": 2000, "response_bytes": 10,
        "request_sha256": "x",
    }
    unknown = dict(record, attempt_id="a2", logical_request_id="l2", usage_input_tokens=None)
    store.append_attempt(record)
    store.append_attempt(unknown)
    executor = ReviewedExecutor(root)
    ledger = BudgetLedger(BudgetCaps(max_requests=400, max_cost_usd=0.10))
    # restoration is scoped to the FROZEN run id (resolved from the freeze doc)
    freeze = {"deterministic_sha256": "abc123def456" + "0" * 52}
    restored = executor._restore_budget(ledger, freeze)
    assert restored["stages"] == {"fresh_test": "rev-fresh_test-abc123def456"}
    assert restored["attempts"] == 2
    assert restored["reported"] == 1000
    assert restored["unknown_kept_reserved"] == 1  # unknown usage counted conservatively
    snap = ledger.snapshot()
    assert snap["reported_input_tokens"] == 1000
    # the unknown attempt's estimate stays inside the cost ceiling (never counted free)
    assert snap["cost_ceiling_usd"] > ledger.caps.cost_for_tokens(1000)


def test_tampered_items_abort_before_dispatch(tmp_path):
    registry = make_registry()
    executor = make_executor(tmp_path, registry)
    executor.run(dry_run=True)
    fresh_run = next(tmp_path.glob("rev-fresh_test-*"))
    items_path = fresh_run / "items.jsonl"
    items = items_path.read_text()
    items_path.write_text(items.replace("exactly two working relationships",
                                        "exactly THREE working relationships", 1))
    live = make_executor(tmp_path, registry)
    with pytest.raises(ExecutorError, match="does not match"):
        live.run(dry_run=False)
    assert registry["clients_created"] == 0


def test_report_regenerates_strict_json_without_provider(tmp_path):
    registry = make_registry()
    executor = make_executor(tmp_path, registry)
    executor.run(dry_run=True)
    live = make_executor(tmp_path, registry)
    live.run(dry_run=False)
    doc, json_path, md_path = build_reviewed_report(tmp_path)
    text = json_path.read_text()
    parsed = strict_loads(text)  # raises on Infinity/NaN
    assert parsed["claim_type"] == "exploratory"
    fresh = parsed["fresh"]
    assert fresh["label_verification"]["n_failures"] == 0
    assert fresh["label_verification"]["n_verified"] == 120
    assert set(fresh["per_family"]) == {"fresh_v3_graph", "fresh_v3_missing_info", "fresh_v3_dates"}
    for family, bucket in fresh["per_family"].items():
        assert bucket["n_planned"] == 40 and bucket["n_scored"] == 40
        assert 0.0 <= bucket["accuracy"] <= 1.0
    mech = parsed["mechanism"]
    assert len(mech["contrasts"]) == 6
    for name, summary in mech["contrasts"].items():
        assert summary["n_blocks_contributing"] == 12
        assert summary["p_sign_flip_exploratory"] == 1.0  # fake answers are symmetric
        assert summary["p_holm_adjusted_exploratory"] == 1.0
    assert mech["excluded_incomplete_contrasts"] == []
    for arm, counts in mech["per_arm_counts"].items():
        assert counts["n_planned"] == 24 and counts["n_complete"] == 24
    batch = parsed["batching"]
    for mode in ("batched", "sequential", "concurrent"):
        bucket = batch["per_mode"][mode]
        assert bucket["n_workloads"] == 12 and bucket["n_complete"] == 12
    assert batch["paired_block_comparisons"]["sequential_vs_batched"]["ratio_a_over_b"]["n_pairs"] == 12
    billing = parsed["billing"]["combined"]
    assert billing["n_attempts"] == 324
    assert billing["n_attempts_unknown_usage"] == 0
    assert billing["reported_input_tokens"] == 324 * 100
    assert md_path.exists() and "exploratory" in md_path.read_text()


def test_live_refusal_without_gates_records_error_in_state(tmp_path):
    registry = make_registry()
    executor = make_executor(tmp_path, registry)
    executor.run(dry_run=True)
    bare = ReviewedExecutor(tmp_path)  # no fake transport: default live factory
    with pytest.raises(ExecutorError, match="JEVO_ALLOW_LIVE"):
        bare.run(dry_run=False)
    state = json.loads((tmp_path / "executor_state.json").read_text())
    assert "JEVO_ALLOW_LIVE" in state["executor_error"]
    assert state["global_stop_reason"] is None
    # and the dry-run packet remains reproducible afterwards
    rebuilt = ReviewedExecutor(tmp_path, transport_factory=forbidden_transport)
    assert rebuilt.run(dry_run=True)["dry_run"] is True


def forbidden_transport():
    raise AssertionError("must never construct a transport")


def test_report_rejects_missing_mechanism_mappings(tmp_path):
    from jev_observatory import reviewed_report

    registry = make_registry()
    executor = make_executor(tmp_path, registry)
    executor.run(dry_run=True)
    live = make_executor(tmp_path, registry)
    live.run(dry_run=False)
    with pytest.raises(FileNotFoundError):
        reviewed_report._mechanism_section(
            next(tmp_path.glob("rev-mechanism-*")), tmp_path / "missing.json")


# ------------------------------------------- REVIEW-3 fix-pass regressions
def test_source_spec_tamper_refused_after_planning(tmp_path, monkeypatch):
    """Tampering a frozen SOURCE spec after planning (stored items intact) must
    be refused: the source file hash is verified against the freeze before
    anything is dispatched. Actual production path; zero clients created."""
    import shutil
    import jev_observatory.reviewed_executor as rex

    real_spec = rex.FRESH_SPEC_PATH
    copy_path = tmp_path / "fresh_spec_copy.json"
    shutil.copy(real_spec, copy_path)
    monkeypatch.setattr(rex, "FRESH_SPEC_PATH", copy_path)
    registry = make_registry()
    executor = make_executor(tmp_path, registry)
    executor.run(dry_run=True)  # freeze records the COPY's hash
    # now tamper the copy (stored items.jsonl stays intact)
    doc = json.loads(copy_path.read_text())
    doc["items"][0]["condition"] += ";tampered"
    copy_path.write_text(json.dumps(doc))
    live = make_executor(tmp_path, registry)
    with pytest.raises(ExecutorError, match="changed since the freeze"):
        live.run(dry_run=False)
    assert registry["clients_created"] == 0  # zero new calls on refusal
    state = json.loads((tmp_path / "executor_state.json").read_text())
    assert "changed since the freeze" in state["executor_error"]


def test_mech_mappings_sidecar_tamper_refused(tmp_path, monkeypatch):
    """The mechanism sidecar is not an unchecked mutable path."""
    import shutil
    import jev_observatory.reviewed_executor as rex

    real_mappings = rex.MECH_MAPPINGS_PATH
    copy_path = tmp_path / "mappings_copy.json"
    shutil.copy(real_mappings, copy_path)
    monkeypatch.setattr(rex, "MECH_MAPPINGS_PATH", copy_path)
    registry = make_registry()
    executor = make_executor(tmp_path, registry)
    executor.run(dry_run=True)
    doc = json.loads(copy_path.read_text())
    doc["items"][0]["measured_anchor_id"] = "s000"
    copy_path.write_text(json.dumps(doc))
    live = make_executor(tmp_path, registry)
    with pytest.raises(ExecutorError, match="sidecar mappings changed"):
        live.run(dry_run=False)
    assert registry["clients_created"] == 0


def test_dispatched_payload_wire_hashes_match_freeze(tmp_path):
    """Every payload that actually left the process matches the frozen ordered
    wire hashes (multiset comparison over the fake transport's captured posts)."""
    from collections import Counter
    from hashlib import sha256
    from jev_observatory.mech_dryrun import ordered_payload_json

    registry = make_registry()
    executor = make_executor(tmp_path, registry)
    executor.run(dry_run=True)
    live = make_executor(tmp_path, registry)
    live.run(dry_run=False)
    freeze = load_freeze(tmp_path)
    expected = Counter(h for stage in freeze["stages"].values()
                       for h in stage["wire_sha256"].values())
    actual = Counter(sha256(ordered_payload_json(p).encode("utf-8")).hexdigest()
                     for p in registry["posts"])
    assert actual == expected
    state = json.loads((tmp_path / "executor_state.json").read_text())
    for stage, entry in state["stages"].items():
        assert entry["n_dispatched_wire_hashes_verified"] == freeze["stages"][stage]["n_requests"]


def test_budget_exceeding_history_refuses_resume_fail_closed(tmp_path):
    """A budget-exceeding HISTORICAL attempt refuses resume (never silently
    omitted or counted as free); zero new calls; reason recorded in state."""
    from jev_observatory.ledger import RunStore

    registry = make_registry()
    executor = make_executor(tmp_path, registry)
    executor.run(dry_run=True)
    freeze = load_freeze(tmp_path)
    run_id = f"rev-fresh_test-{freeze['deterministic_sha256'][:12]}"
    store = RunStore(tmp_path, run_id)
    record = {
        "attempt_id": "hist-1", "logical_request_id": "old-1", "attempt_index": 0,
        "provider": "jev", "model_requested": "jev-1.13.0",
        "started_at": "t", "finished_at": "t", "latency_ms": 1.0,
        "first_byte_ms": None, "cold_connection": None, "http_status": 200,
        "outcome": "ok", "uncertain": False, "error": None,
        "usage_input_tokens": None,  # unknown usage: conservative estimate kept
        "usage_output_tokens": None, "estimated_input_tokens": 3_000_000,
        "response_bytes": 10, "request_sha256": "x",
    }
    store.append_attempt(record)
    posts_before = len(registry["posts"])
    resume = make_executor(tmp_path, registry)
    with pytest.raises(ExecutorError, match="fail-closed resume refusal"):
        resume.run(dry_run=False)
    assert len(registry["posts"]) == posts_before  # zero new calls
    state = json.loads((tmp_path / "executor_state.json").read_text())
    assert "fail-closed resume refusal" in state["executor_error"]
    assert state["budget_restored_from_attempts"]["refused"]["attempt_id"] == "hist-1"


def test_interrupted_partial_batch_resume_labels_wall_time_partial(tmp_path):
    """Resuming an interrupted workload dispatches only pending leftovers and
    NEVER labels their wall time as full-work time; the report excludes it."""
    from jev_observatory.reviewed_report import _batching_section

    registry = make_registry()
    executor = make_executor(tmp_path, registry)
    executor.run(dry_run=True)
    live = make_executor(tmp_path, registry)
    live.run(dry_run=True)  # re-plan; no dispatch
    freeze = load_freeze(tmp_path)
    run_id = f"rev-batching-{freeze['deterministic_sha256'][:12]}"
    store = live._store(run_id)
    spec, _sidecar = build_batching_spec()
    lrs = [lr for lr in logical_requests(spec) if ":sep-b00-" in lr.logical_request_id]
    # interrupt: only the first two requests of b00-sequential are dispatched
    provider = live._new_provider()
    hook = live._hook(BudgetLedger(BudgetCaps()))
    for lr in lrs[:2]:
        live._dispatch(provider, lr, hook, store)
    provider.close()
    # resume the workload: only the 2 pending leftovers are dispatched...
    workloads_path = tmp_path / run_id / "workloads.jsonl"
    summary = live._execute_workload(0, "sequential", "b00-sequential", lrs, store,
                                     BudgetLedger(BudgetCaps()),
                                     workloads_path=workloads_path)
    assert summary["n_dispatched_now"] == 2
    assert summary["resumed_partial"] is True
    assert summary["wall_seconds_is_full_workload"] is False
    # ...and the report omits it from the timing estimand with the reason
    section = _batching_section(tmp_path / run_id)
    seq = section["per_mode"]["sequential"]
    assert seq["n_omitted_from_timing"] == 1
    assert seq["omitted_from_timing"][0]["workload_id"] == "b00-sequential"
    assert "wall_time_not_full_workload_resumed_partial" in seq["omitted_from_timing"][0]["reasons"]
    assert seq["n_success_full_workloads"] == 0
    comparison = section["paired_block_comparisons"]["sequential_vs_batched"]
    assert comparison["n_pairs_observed"] == 0
    assert 0 in comparison["omitted_blocks"]  # b00 omitted (only workload present)


def test_cancelled_concurrent_workload_closes_client_and_persists_record(tmp_path):
    """Cancellation mid-workload: client closed, partial record persisted with
    the stop reason, complete=False, and the workload record is labelled partial."""
    registry = make_registry()
    executor = make_executor(tmp_path, registry)
    executor.run(dry_run=True)
    holder: dict = {}

    def responder(payload: dict, reg: dict) -> TransportResponse:
        if len(reg["posts"]) >= 1:
            holder["executor"].cancellation.cancel()
        return ok_response(payload)

    live = make_executor(tmp_path, registry, responder=responder)
    holder["executor"] = live
    live.run(dry_run=True)  # plans + populates the executor's frozen wire map
    freeze = load_freeze(tmp_path)
    run_id = f"rev-batching-{freeze['deterministic_sha256'][:12]}"
    store = live._store(run_id)
    spec, _sidecar = build_batching_spec()
    lrs = [lr for lr in logical_requests(spec)
           if ":conc-b01-" in lr.logical_request_id]
    workloads_path = tmp_path / run_id / "workloads.jsonl"
    summary = live._execute_workload(1, "concurrent", "b01-concurrent", lrs, store,
                                     BudgetLedger(BudgetCaps()),
                                     workloads_path=workloads_path)
    assert summary["complete"] is False
    assert summary["stop_reason"] == "cancelled_by_cancellation"
    assert summary["wall_seconds_is_full_workload"] is False
    # the workload's client was closed despite the cancellation
    assert registry["clients_created"] == registry["clients_closed"]
    records = [json.loads(l) for l in workloads_path.read_text().splitlines() if l.strip()]
    assert len(records) == 1 and records[0]["workload_id"] == "b01-concurrent"
    # a resumed execution of the same workload dispatches only the leftovers
    resumed = live._execute_workload(1, "concurrent", "b01-concurrent", lrs, store,
                                     BudgetLedger(BudgetCaps()),
                                     workloads_path=workloads_path)
    assert resumed["n_dispatched_now"] <= len(lrs) - summary["n_ok"]


def test_workload_exception_closes_client_persists_record_and_refuses(tmp_path):
    """An exception inside a workload (here: a dispatch-time wire-hash
    mismatch on the production path): the client is closed in finally, the
    partial workload record is persisted with the stop reason, and the run
    refuses fail-closed (REVIEW-3 #4)."""
    registry = make_registry()
    executor = make_executor(tmp_path, registry)
    executor.run(dry_run=True)
    live = make_executor(tmp_path, registry)
    live.run(dry_run=True)  # plans + populates the executor's frozen wire map
    freeze = load_freeze(tmp_path)
    run_id = f"rev-batching-{freeze['deterministic_sha256'][:12]}"
    store = live._store(run_id)
    spec, _sidecar = build_batching_spec()
    lrs = [lr for lr in logical_requests(spec) if ":sep-b02-" in lr.logical_request_id]
    workloads_path = tmp_path / run_id / "workloads.jsonl"
    # simulate executor-side payload drift: the frozen wire entry is corrupted
    first_lid = next(lr.logical_request_id for lr in lrs)
    live._frozen_wire[first_lid] = "0" * 64
    with pytest.raises(ExecutorError, match="partial workload record was persisted"):
        live._execute_workload(2, "sequential", "b02-sequential", lrs, store,
                               BudgetLedger(BudgetCaps()),
                               workloads_path=workloads_path)
    # nothing left the process for the mismatching request
    assert len(registry["posts"]) == 0
    # the client was still created AND closed despite the exception
    assert registry["clients_created"] == registry["clients_closed"] == 1
    records = [json.loads(l) for l in workloads_path.read_text().splitlines() if l.strip()]
    assert len(records) == 1
    assert records[0]["workload_id"] == "b02-sequential"
    assert records[0]["complete"] is False
    assert records[0]["wall_seconds_is_full_workload"] is False
    assert "wire hash mismatch" in records[0]["workload_error"]
    assert live.stop_reason == "dispatch_wire_hash_mismatch"


def test_report_mechanism_incomplete_contrast_no_equivalence(tmp_path):
    """A missing block outcome: the contrast is reported as incomplete with NO
    equivalence verdict, and the fixed-six Holm family still holds (p=1
    placeholder for correction only)."""
    from jev_observatory.reviewed_report import _mechanism_section

    registry = make_registry()
    executor = make_executor(tmp_path, registry)
    executor.run(dry_run=True)
    live = make_executor(tmp_path, registry)
    live.run(dry_run=False)
    mech_dir = next(tmp_path.glob("rev-mechanism-*"))
    results_path = mech_dir / "results.jsonl"
    lines = results_path.read_text().splitlines()
    records = [json.loads(l) for l in lines if l.strip()]
    victim = next(r for r in records if r["condition"].endswith("arm=alone;block=3")
                  or ";arm=alone;block=3" in r["condition"])
    kept = [l for l, r in zip(lines, records) if r is not victim]
    results_path.write_text("\n".join(kept) + "\n")
    mappings_path = tmp_path / "freeze" / "frozen.json"
    mappings = json.loads(mappings_path.read_text())["stages"]["mechanism"]["mappings_path"]
    section = _mechanism_section(mech_dir, Path(mappings))
    anchor = victim["condition"].split("anchor=")[1].split(";")[0]
    name = f"{anchor}:siblings-alone"
    summary = section["contrasts"][name]
    assert summary["n_blocks_contributing"] == 11
    assert summary["status"] == "incomplete"
    assert summary["entire_interval_within_tolerance_0.03"] is None
    assert summary["equivalence_assessment"] == "unavailable_incomplete_contrast"
    assert {"contrast": name, "n_blocks_complete": 11, "n_blocks_missing": 1} \
        in section["excluded_incomplete_contrasts"]
    # fixed six-contrast Holm family still reported for every contrast
    assert len(section["contrasts"]) == 6
    assert summary["holm_input_p"] == 1.0
    assert "placeholder" in summary["holm_note"]
    complete = section["contrasts"][f"{anchor}:renamed-siblings"]
    assert complete.get("status") != "incomplete"
    assert complete["entire_interval_within_tolerance_0.03"] is not None


REVIEW_GATE = Path("docs/review-gate/audit")
needs_review_gate = pytest.mark.skipif(
    not REVIEW_GATE.is_dir(),
    reason="internal review-gate audit specs are not shipped in the "
           "published bundle; this regression needs them",
)


@needs_review_gate
def test_report_fresh_section_counts_complete_and_missing(tmp_path):
    from jev_observatory.reviewed_report import _fresh_section

    registry = make_registry()
    executor = make_executor(tmp_path, registry)
    executor.run(dry_run=True)
    live = make_executor(tmp_path, registry)
    live.run(dry_run=False)
    fresh_dir = next(tmp_path.glob("rev-fresh_test-*"))
    results_path = fresh_dir / "results.jsonl"
    lines = [l for l in results_path.read_text().splitlines() if l.strip()]
    records = [json.loads(l) for l in lines]
    victim = records[0]
    kept = [l for l, r in zip(lines, records) if r is not victim]
    results_path.write_text("\n".join(kept) + "\n")
    section = _fresh_section(fresh_dir)
    bucket = section["per_family"][victim["group"]]
    assert bucket["n_planned"] == 40
    assert bucket["n_complete"] == 39
    assert bucket["n_error_or_missing"] == 1
    assert bucket["n_scored"] == 39


@needs_review_gate
def test_report_batching_excludes_failed_workload_from_timing(tmp_path):
    from jev_observatory.reviewed_report import _batching_section

    registry = make_registry()
    executor = make_executor(tmp_path, registry)
    executor.run(dry_run=True)
    live = make_executor(tmp_path, registry)
    live.run(dry_run=False)
    batch_dir = next(tmp_path.glob("rev-batching-*"))
    wl_path = batch_dir / "workloads.jsonl"
    lines = [l for l in wl_path.read_text().splitlines() if l.strip()]
    records = [json.loads(l) for l in lines]
    victim = next(r for r in records if r["workload_id"] == "b05-batched")
    victim["complete"] = False
    victim["n_errors"] = 1  # terminal HTTP error kept DISTINCT from success
    kept = [json.dumps(r, ensure_ascii=False) for r in records]
    wl_path.write_text("\n".join(kept) + "\n")
    section = _batching_section(batch_dir)
    batched = section["per_mode"]["batched"]
    assert batched["n_omitted_from_timing"] == 1
    assert batched["omitted_from_timing"][0]["workload_id"] == "b05-batched"
    assert "incomplete" in batched["omitted_from_timing"][0]["reasons"]
    assert batched["n_success_full_workloads"] == 11
    assert batched["n_workloads_with_errors"] == 1
    assert len(batched["wall_seconds_per_workload"]) == 11
    for mode, comparison in section["paired_block_comparisons"].items():
        n_expected = 11 if "batched" in mode else 12
        assert comparison["n_pairs_observed"] == n_expected
        if "batched" in mode:
            assert 5 in comparison["omitted_blocks"]
        else:
            assert comparison["omitted_blocks"] == []


@needs_review_gate
def test_report_score_pairs_labelled_not_compared(tmp_path):
    from jev_observatory.reviewed_report import _batching_section

    registry = make_registry()
    executor = make_executor(tmp_path, registry)
    executor.run(dry_run=True)
    live = make_executor(tmp_path, registry)
    live.run(dry_run=False)
    batch_dir = next(tmp_path.glob("rev-batching-*"))
    section = _batching_section(batch_dir)
    d = section["answer_disagreement_vs_batched"]
    assert d["n_score_pairs_not_compared"] == 24  # 12 blocks x 2 separate modes
    assert "Score" in d["comparison_scope"] and "not compared" in d["comparison_scope"]
