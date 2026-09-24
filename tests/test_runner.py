"""Runner end-to-end tests against the mock provider (fully offline)."""

import json

import pytest

from jev_observatory.experiment import SpecError, logical_requests
from jev_observatory.guards import BudgetCaps, BudgetExceeded, BudgetLedger, Cancellation
from jev_observatory.ledger import DuplicateRecord
from jev_observatory.providers import MockProvider
from jev_observatory.redact import Redactor
from jev_observatory.runner import Runner, UncertainPolicyError, plan_run


def _spec(n_items=6, **item_overrides):
    items = []
    for i in range(n_items):
        items.append({
            "id": f"item-{i}",
            "group": "smoke",
            "cluster": f"c{i % 2}",
            "state": f"Ticket number {i} about a refund.",
            "gold": {"q": {"value": "billing"}},
            "questions": {"dep": {"type": "choice", "instructions": "Which team?",
                                  "criteria": {"billing": "money", "tech": "bugs"}}},
            **item_overrides,
        })
    return {"experiment": "m0-smoke", "model": "jev-1.13.0", "items": items, "seeds": {"order": 7}}


def _runner(store, provider=None, **kw):
    return Runner(provider or MockProvider(seed=0), store, **kw)


def test_full_offline_mock_run(tmp_path):
    spec = _spec(4)
    run_id, manifest = plan_run(spec, provider="mock", root=str(tmp_path))
    from jev_observatory.runner import load_run_store

    store = load_run_store(run_id, str(tmp_path))
    requests = logical_requests(spec)
    runner = _runner(store)
    result = runner.run(requests)

    assert result.n_dispatched == 4
    assert result.n_ok == 4
    assert result.n_contract_invalid == 0
    assert len(store.results()) == 4
    assert len(store.attempts()) == 4
    assert store.completed_ids() == {r.logical_request_id for r in requests}
    # option map round trip present per result
    for record in store.results():
        assert record["option_map"]["dep"] == {"type": "choice", "options": ["billing", "tech"]}


def test_resume_skips_completed_and_never_overwrites(tmp_path):
    spec = _spec(3)
    run_id, _ = plan_run(spec, provider="mock", root=str(tmp_path))
    from jev_observatory.runner import load_run_store

    store = load_run_store(run_id, str(tmp_path))
    requests = logical_requests(spec)
    _runner(store).run(requests)
    before = {r["logical_request_id"]: r for r in store.results()}

    result = _runner(store).run(requests)  # second pass = resume
    assert result.n_dispatched == 0
    assert result.n_skipped_resume == 3
    assert len(store.results()) == 3
    with pytest.raises(DuplicateRecord):
        store.append_result({**store.results()[0], "status": "error"})


def test_uncertain_requires_explicit_policy(tmp_path):
    spec = _spec(2)
    run_id, _ = plan_run(spec, provider="mock", root=str(tmp_path))
    from jev_observatory.runner import load_run_store

    store = load_run_store(run_id, str(tmp_path))
    # Simulate an earlier timeout attempt with no terminal result.
    store.append_attempt({"attempt_id": "x1", "logical_request_id": "native:item-0", "uncertain": True})
    requests = logical_requests(spec)
    with pytest.raises(UncertainPolicyError):
        _runner(store).run(requests, on_uncertain="error")

    result = _runner(store).run(requests, on_uncertain="skip")
    assert result.n_dispatched == 1  # item-1 only
    assert store.uncertain_ids() == {"native:item-0"}

    # retry policy dispatches the uncertain item again as a NEW attempt
    result2 = _runner(store).run(requests, on_uncertain="retry")
    assert result2.n_dispatched == 1
    attempts = store.attempts_for("native:item-0")
    assert len(attempts) >= 2


def test_budget_cap_stops_dispatch_and_reports_denial(tmp_path):
    spec = _spec(10)
    run_id, _ = plan_run(spec, provider="mock", root=str(tmp_path))
    from jev_observatory.runner import load_run_store

    store = load_run_store(run_id, str(tmp_path))
    budget = BudgetLedger(BudgetCaps(max_requests=3))
    result = _runner(store, budget=budget).run(logical_requests(spec))
    assert result.n_dispatched == 3
    assert result.halted is True
    assert len(store.results()) == 3
    assert budget.snapshot()["denied_by_limit"] == ["max_requests"]
    assert len(store.attempts()) == 3  # no extra requests went out


def test_pre_dispatch_reservation_never_oversubscribes(tmp_path):
    """Concurrent workers cannot exceed a request cap via a race."""
    spec = _spec(50)
    run_id, _ = plan_run(spec, provider="mock", root=str(tmp_path))
    from jev_observatory.runner import load_run_store

    store = load_run_store(run_id, str(tmp_path))
    budget = BudgetLedger(BudgetCaps(max_requests=7))
    result = _runner(store, budget=budget, concurrency=8).run(logical_requests(spec))
    assert result.n_dispatched == 7
    assert len(store.attempts()) == 7


def test_cancellation_dispatches_nothing_more(tmp_path):
    spec = _spec(20)
    run_id, _ = plan_run(spec, provider="mock", root=str(tmp_path))
    from jev_observatory.runner import load_run_store

    store = load_run_store(run_id, str(tmp_path))
    cancellation = Cancellation()
    cancellation.cancel()
    runner = _runner(store)
    runner.cancellation = cancellation
    result = runner.run(logical_requests(spec))
    assert result.n_dispatched == 0
    assert len(store.attempts()) == 0


def test_gold_labels_never_in_results_or_attempts(tmp_path):
    spec = _spec(2)
    run_id, _ = plan_run(spec, provider="mock", root=str(tmp_path))
    from jev_observatory.runner import load_run_store

    store = load_run_store(run_id, str(tmp_path))
    _runner(store).run(logical_requests(spec))
    results_text = (store.directory / "results.jsonl").read_text()
    attempts_text = (store.directory / "attempts.jsonl").read_text()
    assert '"gold"' not in results_text
    assert '"gold"' not in attempts_text
    # the raw request blobs also carry no labels
    for path in (store.directory / "raw").glob("*.request.json"):
        assert '"gold"' not in path.read_text()


def test_leakage_guard_rejects_gold_in_state(tmp_path):
    spec = _spec(1, **{"state": "The correct answer is billing for this ticket."})
    with pytest.raises(SpecError):
        logical_requests(spec)


def test_manifest_captures_plan_inputs(tmp_path):
    spec = _spec(2)
    run_id, manifest = plan_run(spec, provider="mock", root=str(tmp_path),
                                budgets={"max_requests": 5}, retry_policy={"mode": "timing"})
    assert manifest.n_items == 2
    assert manifest.budgets == {"max_requests": 5}
    assert manifest.retry_policy == {"mode": "timing"}
    assert manifest.option_maps  # per logical request
    assert len(manifest.option_maps) == 2
    assert manifest.pricing["input_usd_per_mtok"] == 0.042


def test_run_summary_written(tmp_path):
    spec = _spec(2)
    run_id, _ = plan_run(spec, provider="mock", root=str(tmp_path))
    from jev_observatory.runner import load_run_store

    store = load_run_store(run_id, str(tmp_path))
    runner = _runner(store, budget=BudgetLedger(BudgetCaps()))
    runner.run(logical_requests(spec))
    summary = runner.write_summary()
    assert summary["results"] == 2
    assert (store.directory / "run_summary.json").exists()