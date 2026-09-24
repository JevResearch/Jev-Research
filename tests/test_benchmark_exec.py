"""Full-benchmark executor tests (offline only, no network, no credentials).

Covers: freeze immutability, dry-run dispatches nothing, full plan → dispatch
with concurrency and wire-hash verification, honest resume accounting (finished
items never re-dispatched, budgets restored from stored attempts), fail-closed
refusals (uncertain timeouts, budget excess, mutated items), stop conditions
(401/402/403, repeated failures, request-cap), and partial-artifact
preservation under every stop.
"""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from jev_observatory.benchmark_exec import (
    ExecutorError,
    BenchmarkExecutor,
    build_freeze,
    load_freeze,
    stage_run_id,
)
from jev_observatory.benchmark_spec import build_benchmark_suite
from jev_observatory.ledger import RunStore
from jev_observatory.providers import JevProvider, MockProvider, RetryPolicy
from jev_observatory.transport import ScriptedTransport

EXPECTED = {"mmlu_full": 20, "arc_test": 6}


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


def _mmlu_records(n: int = 20) -> list[dict]:
    letters = "ABCDEFGHIJ"
    return [{
        "question_id": f"q{i:04d}",
        "question": f"Fixture question {i}?",
        "options": [f"option {i}.{j}" for j in range(4 + (i % 7))],
        "answer_index": i % (4 + (i % 7)),
        "answer": letters[i % (4 + (i % 7))],
        "category": ("math", "law")[i % 2],
        "src": "fixture",
    } for i in range(n)]


def _arc_records(n: int = 6) -> list[dict]:
    labels = ["A", "B", "C", "D"]
    return [{
        "id": f"arc{i:04d}", "question": f"ARC fixture {i}?",
        "choices": {"text": [f"c{i}.{j}" for j in range(4)], "label": labels},
        "answerKey": labels[i % 4],
    } for i in range(n)]


@pytest.fixture
def prepared(tmp_path) -> tuple[Path, dict, dict]:
    """Offline preparation: fixtures → suite → freeze → spec.json + plan."""
    data = tmp_path / "data"
    data.mkdir()
    mmlu_path = data / "mmlu_test.jsonl"
    arc_path = data / "arc_test.jsonl"
    _write_jsonl(mmlu_path, _mmlu_records())
    _write_jsonl(arc_path, _arc_records())
    for path in (mmlu_path, arc_path):
        Path(str(path) + ".sha256.json").write_text(json.dumps({
            "file": str(path),
            "sha256": sha256(path.read_bytes()).hexdigest(),
            "url": "fixture://source",
        }) + "\n", encoding="utf-8")
    root = tmp_path / "runs_benchmark"
    suite = build_benchmark_suite(mmlu_path=mmlu_path, arc_path=arc_path,
                                  expected=EXPECTED)
    freeze = build_freeze(suite, root=root)
    from jev_observatory.runner import plan_run

    for stage in ("mmlu_full", "option_rotations", "arc_test"):
        run_id = stage_run_id(freeze, stage)
        directory = root / run_id
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "spec.json").write_text(
            json.dumps(suite[stage]["spec"], ensure_ascii=False) + "\n", encoding="utf-8")
        if stage == "option_rotations":
            (directory / "rotations.json").write_text(
                json.dumps(suite[stage]["spec"]["rotations"], ensure_ascii=False) + "\n",
                encoding="utf-8")
        plan_run(suite[stage]["spec"], provider="jev", root=str(root), run_id=run_id,
                 retry_policy={"max_attempts": 1})
    return root, freeze, suite


def test_freeze_is_write_once_and_immutable(prepared):
    root, freeze, suite = prepared
    again = build_freeze(suite, root=root)  # deterministic recomputation reuses it
    assert again["deterministic_sha256"] == freeze["deterministic_sha256"]
    assert load_freeze(root)["deterministic_sha256"] == freeze["deterministic_sha256"]
    assert freeze["total_requests"] == 86


def _provider_factory(provider: MockProvider):
    return lambda: provider


def test_full_run_concurrent_persistence_and_accounting(prepared):
    root, freeze, _suite = prepared
    provider = MockProvider(seed=3)
    executor = BenchmarkExecutor(root, provider_factory=_provider_factory(provider),
                                 concurrency=4, rpm=600)
    state = executor.run(dry_run=False)
    assert state["global_stop_reason"] is None
    total = 0
    for stage in ("mmlu_full", "option_rotations", "arc_test"):
        summary = state["stages"][stage]
        assert summary["ok"] == summary["n_requests"]
        assert summary["skipped_resume"] == 0
        total += summary["n_requests"]
        store = RunStore(root, summary["run_id"])
        attempts = store.attempts()
        results = store.results()
        assert len(attempts) == summary["n_requests"]
        assert len(results) == summary["n_requests"]
        assert all(r["status"] == "ok" for r in results)
        assert all(a["usage_input_tokens"] is not None for a in attempts)
        assert summary["n_dispatched_wire_hashes_verified"] == summary["n_requests"]
    assert total == 86
    # every dispatched request must also be wire-verified (mock answers everything)
    assert len(provider.calls) == 86


def test_dry_run_dispatches_nothing(prepared):
    root, freeze, _suite = prepared
    executor = BenchmarkExecutor(root, provider_factory=lambda: pytest.fail(
        "dry run must never construct a provider"))
    state = executor.run(dry_run=True)
    assert state["dry_run"] is True
    assert state["stages"]  # planned + hash-verified
    assert state["note"].startswith("dry run")


def test_resume_never_redispatches_finished_items(prepared):
    root, freeze, _suite = prepared
    provider = MockProvider(seed=5)
    executor = BenchmarkExecutor(root, provider_factory=_provider_factory(provider))
    executor.run(dry_run=False)
    n_first = len(provider.calls)
    assert n_first == 86
    resumed = BenchmarkExecutor(
        root, provider_factory=_provider_factory(provider)).run(dry_run=False)
    assert len(provider.calls) == n_first  # nothing re-dispatched
    for stage in ("mmlu_full", "option_rotations", "arc_test"):
        summary = resumed["stages"][stage]
        assert summary["dispatched"] == 0
        assert summary["skipped_resume"] == summary["n_requests"]
    assert resumed["budget_restored_from_attempts"]["attempts"] == 86


def test_uncertain_timeout_fails_resume_closed(prepared):
    root, freeze, _suite = prepared
    # timeout every dispatch: attempts recorded, never terminal
    provider = MockProvider(seed=7, default_fault={"outcome": "timeout",
                                                   "error": "timeout:ReadTimeout"})
    executor = BenchmarkExecutor(root, provider_factory=_provider_factory(provider),
                                 consecutive_failure_limit=3)
    state = executor.run(dry_run=False)
    assert state["global_stop_reason"] == "repeated_transport_schema_failures"
    store = RunStore(root, stage_run_id(freeze, "mmlu_full"))
    assert store.attempts()
    assert store.uncertain_ids()
    # resume is REFUSED outright: the provider may have billed unobserved work
    with pytest.raises(ExecutorError, match="fail-closed resume refusal"):
        BenchmarkExecutor(root, provider_factory=_provider_factory(provider)).run(
            dry_run=False)


def test_budget_restore_refusal_when_caps_exceeded(prepared):
    root, freeze, _suite = prepared
    BenchmarkExecutor(root, provider_factory=_provider_factory(MockProvider(seed=9))).run(
        dry_run=False)
    # an absurdly low token cap cannot honestly hold the stored attempts
    with pytest.raises(ExecutorError, match="exceeds the budget"):
        BenchmarkExecutor(root, provider_factory=_provider_factory(MockProvider(seed=9)),
                          max_estimated_input_tokens=10).run(dry_run=False)


def test_request_cap_stops_mid_run_then_resumes_honestly(prepared):
    root, freeze, _suite = prepared
    provider = MockProvider(seed=11)
    executor = BenchmarkExecutor(root, provider_factory=_provider_factory(provider),
                                 max_requests=10)
    state = executor.run(dry_run=False)
    assert state["global_stop_reason"] == "budget_cap_reached:max_requests"
    first_results = sum(len(RunStore(root, state["stages"][s]["run_id"]).results())
                        for s in state["stages"])
    assert first_results == 10
    # resume with the full cap restores the 10 stored attempts, then finishes
    resumed = BenchmarkExecutor(root, provider_factory=_provider_factory(provider)).run(
        dry_run=False)
    restored = resumed["budget_restored_from_attempts"]
    assert restored["attempts"] == 10
    total = 0
    for stage in ("mmlu_full", "option_rotations", "arc_test"):
        summary = resumed["stages"][stage]
        store = RunStore(root, summary["run_id"])
        total += len(store.results())
    assert total == 86
    assert len(provider.calls) == 86  # exactly the 10 + 76 needed, no more


def test_fatal_http_status_stops_and_preserves_partial(prepared):
    root, freeze, _suite = prepared
    responses = [{"status_code": 401,
                  "body": {"error": {"message": "insufficient credits"}}}]
    provider = JevProvider(ScriptedTransport(responses), retry_policy=RetryPolicy.none())
    executor = BenchmarkExecutor(root, provider_factory=lambda: provider,
                                 concurrency=1)
    state = executor.run(dry_run=False)
    assert state["global_stop_reason"] == "insufficient_credit_or_auth"
    for stage in ("option_rotations", "arc_test"):
        assert state["stages"][stage]["skipped_reason"] == "insufficient_credit_or_auth"
    store = RunStore(root, stage_run_id(freeze, "mmlu_full"))
    # the failed attempt AND its terminal http_error record are persisted;
    # nothing else was dispatched
    assert len(store.attempts()) == 1
    assert store.attempts()[0]["http_status"] == 401
    results = store.results()
    assert len(results) == 1
    assert results[0]["status"] == "http_error"


def test_live_default_refuses_without_opt_in(prepared, monkeypatch):
    root, _freeze, _suite = prepared
    monkeypatch.delenv("JEVO_ALLOW_LIVE", raising=False)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    executor = BenchmarkExecutor(root)
    with pytest.raises(ExecutorError, match="JEVO_ALLOW_LIVE"):
        executor.run(dry_run=False)


def test_mutated_stored_items_are_refused(prepared):
    root, freeze, _suite = prepared
    run_id = stage_run_id(freeze, "mmlu_full")
    items_path = root / run_id / "items.jsonl"
    items = [json.loads(line) for line in items_path.read_text().splitlines() if line]
    items[0]["state"] = "MUTATED STATE TEXT"
    _write_jsonl(items_path, items)
    executor = BenchmarkExecutor(root, provider_factory=_provider_factory(MockProvider()))
    with pytest.raises(ExecutorError, match="does not match its manifest hash"):
        executor.run(dry_run=False)
