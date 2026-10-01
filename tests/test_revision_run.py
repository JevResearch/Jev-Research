"""Versioned resumable run-support tests: no overwrite, full private raws,
active-result pointers, usage accounting, bounded concurrency (offline only)."""

from __future__ import annotations

import json
import threading
import time

import pytest

from jev_observatory.revision_run import (
    BudgetExceeded,
    PersistentBudget,
    RevisionRunError,
    VersionedRunStore,
    bounded_dispatch,
    is_retryable_status,
    pending_tasks,
    plan_fingerprint,
    retry_after_seconds,
)


def _plan(models=("m1",), datasets=("d1",)):
    return {"models": list(models), "datasets": list(datasets), "wire": {"t": 0}}


def _result(lid, model="m1", dataset="d1", *, terminal=True, correct=False,
            stage="exact", gold="A"):
    return {"logical_request_id": lid, "model": model, "dataset": dataset,
            "terminal": terminal, "status": "ok", "recovery_stage": stage,
            "correct_recovered": correct, "gold": gold}


def test_initialize_is_idempotent_and_binds_one_plan(tmp_path):
    store = VersionedRunStore(tmp_path, "v4")
    meta = store.initialize(_plan())
    assert meta["plan_fingerprint"] == plan_fingerprint(_plan())
    again = VersionedRunStore(tmp_path, "v4").initialize(_plan())
    assert again["created_at"] == meta["created_at"]     # resumed, not rewritten
    with pytest.raises(RevisionRunError, match="fingerprint mismatch"):
        store.initialize(_plan(models=("m2",)))
    # original evidence untouched by a different version's store
    other = VersionedRunStore(tmp_path, "v5")
    other.initialize(_plan(models=("m2",)))
    assert (tmp_path / "v4" / "run_meta.json").exists()
    assert json.loads((tmp_path / "v4" / "run_meta.json").read_text())["plan"] == _plan()


def test_append_result_refuses_duplicate_terminal_ids(tmp_path):
    store = VersionedRunStore(tmp_path, "v4")
    store.initialize(_plan())
    store.append_result(_result("m1:d1:i1"))
    with pytest.raises(RevisionRunError, match="duplicate terminal"):
        store.append_result(_result("m1:d1:i1"))
    assert store.terminal_ids() == {"m1:d1:i1"}


def test_resume_skips_terminal_requests(tmp_path):
    store = VersionedRunStore(tmp_path, "v4")
    store.initialize(_plan())
    store.append_result(_result("m1:d1:i1"))
    tasks = [("i1",), ("i2",)]
    pending = pending_tasks(tasks, lambda t: f"m1:d1:{t[0]}", store.terminal_ids())
    assert pending == [("i2",)]


def test_raw_responses_are_complete_and_private(tmp_path):
    store = VersionedRunStore(tmp_path, "v4")
    store.initialize(_plan())
    store.append_raw("m1:d1:i1", {
        "model": "m1", "content": "x" * 5000,      # FULL content, no 400-char cut
        "reasoning": "y" * 5000,
        "finish_reason": "length",
        "http_status": 200,
    })
    raws = store.read_raws()
    assert raws[0]["finish_reason"] == "length"
    assert len(raws[0]["content"]) == 5000 and len(raws[0]["reasoning"]) == 5000
    assert (store.private_dir / "m1.jsonl").exists()
    # a raw record without a finish_reason is refused: the evidence contract
    with pytest.raises(RevisionRunError, match="finish_reason"):
        store.append_raw("m1:d1:i2", {"model": "m1", "content": "", "reasoning": "",
                                      "http_status": 200})


def test_public_summary_carries_no_gold_and_no_licensed_text(tmp_path):
    store = VersionedRunStore(tmp_path, "v4")
    store.initialize(_plan())
    store.append_result(_result("m1:d1:i1", gold="C", correct=True))
    store.record_usage("m1:d1:i1", {"input_tokens": 10, "output_tokens": 5,
                                    "cost_usd": 0.001})
    public = json.dumps(store.public_summary())
    assert '"gold"' not in public and '"C"' not in public
    assert "content" not in public
    assert store.public_summary()["cells"]["m1:d1"]["n_correct_recovered"] == 1
    manifest = store.publication_manifest()
    assert any("private_raw" in p for p in manifest["private"])


def test_usage_accounting_counts_unknown_as_unknown(tmp_path):
    store = VersionedRunStore(tmp_path, "v4")
    store.initialize(_plan())
    store.record_usage("m1:d1:i1", {"input_tokens": 10, "output_tokens": 5, "cost_usd": 0.5})
    store.record_usage("m1:d1:i2", {"input_tokens": None, "output_tokens": None,
                                    "cost_usd": None})
    summary = store.usage_summary()
    assert summary["n_usage_rows"] == 2
    assert summary["usage_input_tokens_sum"] == 10
    assert summary["cost_usd_sum"] == 0.5
    assert summary["unknown_usage_rows"] == 1      # never counted as free
    assert summary["unknown_cost_rows"] == 1


def test_active_result_pointers_are_append_only(tmp_path):
    store = VersionedRunStore(tmp_path, "v4")
    store.initialize(_plan())
    store.set_active("m1:d1", note="first")
    store.set_active("m1:d1", note="supersede v2 cell")
    pointers = store.active_pointers()
    assert pointers["active"]["m1:d1"]["note"] == "supersede v2 cell"
    assert len(pointers["history"]) == 2
    assert pointers["history"][0]["note"] == "first"


def test_bounded_dispatch_respects_limits_and_spend_cap(tmp_path):
    lock = threading.Lock()
    inflight = {"total": 0, "max_total": 0, "per": {}, "max_per": {}}
    seen = []

    def worker(task):
        key, idx, cost = task
        with lock:
            inflight["total"] += 1
            inflight["max_total"] = max(inflight["max_total"], inflight["total"])
            inflight["per"][key] = inflight["per"].get(key, 0) + 1
            inflight["max_per"][key] = max(inflight["max_per"].get(key, 0),
                                           inflight["per"][key])
        time.sleep(0.01)
        with lock:
            inflight["total"] -= 1
            inflight["per"][key] -= 1
            seen.append(task)
        return {"cost": cost}

    tasks = [(f"k{i % 3}", i, 0.4) for i in range(12)]
    budget = PersistentBudget(tmp_path / "budget.json", 1.0)
    stats = bounded_dispatch(tasks, worker, total_workers=4, per_key_limit=2,
                             key_of=lambda t: t[0], budget=budget,
                             max_cost_of=lambda t: t[2],
                             price_of=lambda r: r.get("cost"))
    assert inflight["max_total"] <= 4
    assert all(v <= 2 for v in inflight["max_per"].values())
    assert stats["budget_stop"] is True
    snap = budget.snapshot()
    # PRE-CALL reservations bound worst-case exposure to the cap at all times
    assert snap["exposure_usd"] <= 1.0 + 1e-9
    assert snap["settled_usd"] <= 1.0 + 1e-9
    assert len(seen) < 12        # cap stopped further dispatches


def test_bounded_dispatch_counts_unknown_cost_rows(tmp_path):
    budget = PersistentBudget(tmp_path / "budget.json", 1.0)
    stats = bounded_dispatch(
        [(1,), (2,)], lambda t: {"cost": None}, total_workers=2,
        per_key_limit=2, key_of=lambda t: "k", budget=budget,
        max_cost_of=lambda t: 0.3, price_of=lambda r: r.get("cost"))
    assert stats["unknown_cost_rows"] == 2
    assert stats["spend_usd"] == 0.0
    # unknown costs FAIL CLOSED: both reservations stay held, never free
    snap = budget.snapshot()
    assert snap["held_usd"] == pytest.approx(0.6)
    assert snap["exposure_usd"] == pytest.approx(0.6)


def _racing_dispatch(budget, n_tasks, max_cost, worker, *, workers=8):
    """Run a racing dispatch while sampling exposure; return max exposure."""
    stop = threading.Event()
    peak = [0.0]
    lock = threading.Lock()

    def sample():
        while not stop.is_set():
            e = budget.snapshot()["exposure_usd"]
            with lock:
                peak[0] = max(peak[0], e)
            time.sleep(0.001)

    sampler = threading.Thread(target=sample)
    sampler.start()
    stats = bounded_dispatch(
        [(i,) for i in range(n_tasks)], worker, total_workers=workers,
        per_key_limit=4, key_of=lambda t: "k", budget=budget,
        max_cost_of=lambda t: max_cost, price_of=lambda r: r.get("cost"))
    stop.set()
    sampler.join()
    with lock:
        peak[0] = max(peak[0], budget.snapshot()["exposure_usd"])
    return stats, peak[0]


def test_worst_case_concurrency_cannot_exceed_cap(tmp_path):
    """Many calls racing under the cap can never push worst-case exposure
    (settled + held + in-flight reservations) past the cap."""
    budget = PersistentBudget(tmp_path / "budget.json", 1.0)

    def worker(task):
        time.sleep(0.005)
        return {"cost": 0.05}      # settles BELOW the 0.15 reservation

    stats, peak = _racing_dispatch(budget, 60, 0.15, worker)
    assert peak <= 1.0 + 1e-9, f"worst-case exposure {peak} broke the cap"
    assert stats["budget_stop"] is True
    assert budget.snapshot()["exposure_usd"] <= 1.0 + 1e-9


def test_resume_cannot_reset_the_budget(tmp_path):
    """A second run over the same ledger (resume) is bound by past spend,
    held reservations and the same cap — never a fresh allowance."""
    ledger = tmp_path / "budget.json"

    def worker(task):
        return {"cost": 0.3}

    b1 = PersistentBudget(ledger, 1.0)
    bounded_dispatch([(i,) for i in range(10)], worker, total_workers=2,
                     per_key_limit=2, key_of=lambda t: "k", budget=b1,
                     max_cost_of=lambda t: 0.3, price_of=lambda r: r.get("cost"))
    spent_first = b1.snapshot()["exposure_usd"]
    assert spent_first > 0
    b2 = PersistentBudget(ledger, 1.0)          # resumed process
    assert b2.snapshot()["exposure_usd"] == pytest.approx(spent_first)
    stats2, peak2 = _racing_dispatch(b2, 10, 0.3, worker, workers=2)
    total = b2.snapshot()
    assert peak2 <= 1.0 + 1e-9
    assert total["exposure_usd"] <= 1.0 + 1e-9   # BOTH runs share one cap
    # cap mismatch is refused outright: no silently lowered/raised ceiling
    with pytest.raises(RevisionRunError, match="cap mismatch"):
        PersistentBudget(ledger, 2.0)


def test_retries_are_reserved_before_any_attempt(tmp_path):
    """max_cost_of must cover the worst case INCLUDING bounded retries, so
    a retried call cannot spend more than was reserved before attempt one,
    and a call that dies mid-retries fails closed."""
    budget = PersistentBudget(tmp_path / "budget.json", 1.0)
    attempts = {"n": 0}

    def worker(task):
        # bounded transport retries happen INSIDE the worker; every attempt
        # may bill the provider, so the pre-call reservation covers all 3
        for _ in range(3):
            attempts["n"] += 1
            raise RuntimeError("transport failed on attempt")
        return {"cost": 0.2}

    # one task, up to 3 attempts at 0.2 each -> reservation must be 0.6
    rid1 = budget.reserve("k", 3 * 0.2)
    with pytest.raises(BudgetExceeded):
        budget.reserve("k", 3 * 0.2)   # 2nd worst case (0.6+0.6) does not fit
    budget.settle(rid1, None)          # died mid-retries -> held, not free
    assert budget.snapshot()["held_usd"] == pytest.approx(0.6)

    stats = bounded_dispatch(
        [(2,)], worker, total_workers=1, per_key_limit=1,
        key_of=lambda t: "k", budget=budget,
        max_cost_of=lambda t: 0.35, price_of=lambda r: r.get("cost"))
    # worker raised inside dispatch -> full worst-case reservation HELD
    assert stats["failed"] == 1
    assert attempts["n"] >= 1
    snap = budget.snapshot()
    assert snap["held_usd"] == pytest.approx(0.6 + 0.35)
    assert snap["exposure_usd"] == pytest.approx(0.95)  # never reusable


def test_unknown_billed_cost_never_becomes_free_budget(tmp_path):
    """Rows whose billed cost is unknown keep their worst-case reservation
    held forever: running many of them cannot bypass the cap."""
    budget = PersistentBudget(tmp_path / "budget.json", 1.0)
    stats = bounded_dispatch(
        [(i,) for i in range(40)], lambda t: {"cost": None},
        total_workers=8, per_key_limit=4, key_of=lambda t: "k",
        budget=budget, max_cost_of=lambda t: 0.2,
        price_of=lambda r: r.get("cost"))
    snap = budget.snapshot()
    assert stats["budget_stop"] is True
    assert snap["exposure_usd"] <= 1.0 + 1e-9
    assert snap["held_usd"] == pytest.approx(snap["exposure_usd"])  # all held
    assert snap["n_reservations"] <= 5            # 5 x 0.2 = 1.0 max
    with pytest.raises(BudgetExceeded):
        budget.reserve("k", 0.05)


def test_budgeted_dispatch_requires_worst_case_cost(tmp_path):
    """A budget with no declared worst-case cost is refused: zero-cost
    reservations would silently bypass the cap."""
    budget = PersistentBudget(tmp_path / "budget.json", 1.0)
    with pytest.raises(RevisionRunError, match="max_cost_of"):
        bounded_dispatch([(1,)], lambda t: {"cost": 0.1}, total_workers=1,
                         per_key_limit=1, key_of=lambda t: "k",
                         budget=budget, price_of=lambda r: r.get("cost"))


def test_release_unsent_is_the_only_free_release(tmp_path):
    budget = PersistentBudget(tmp_path / "budget.json", 1.0)
    rid = budget.reserve("k", 0.9, label="validated locally, never sent")
    budget.release_unsent(rid)
    assert budget.snapshot()["exposure_usd"] == 0.0
    rid2 = budget.reserve("k", 0.9)
    budget.settle(rid2, None)                     # sent, cost unknown -> held
    assert budget.snapshot()["held_usd"] == pytest.approx(0.9)
    with pytest.raises(RevisionRunError, match="unknown reservation"):
        budget.settle(rid2, 0.1)                  # never double-settled

# ------------------------------------------- transport-recovery / rate limits
def _provisional_row(lid, model="m1", dataset="d1", http_status=429):
    return {"logical_request_id": lid, "model": model, "dataset": dataset,
            "item_id": lid.split(":")[-1], "gold": "A", "terminal": True,
            "status": "http_error", "http_status": http_status,
            "transport_retryable": True, "provisional": True,
            "pred_strict": None, "pred_recovered": None,
            "recovery_stage": "unrecovered", "correct_recovered": False}


def test_retryable_status_and_backoff_policy():
    assert is_retryable_status(429) and is_retryable_status(0)
    assert is_retryable_status(503) and not is_retryable_status(200)
    assert not is_retryable_status(400)   # 4xx contract errors are not transport
    # provider Retry-After honored, but capped
    assert retry_after_seconds({"retry-after": "7"}, 1) == 7.0
    assert retry_after_seconds({"Retry-After": "600"}, 1) == 60.0
    # exponential fallback, capped
    assert retry_after_seconds({}, 1) == 2.0
    assert retry_after_seconds({}, 9) == 18.0
    assert retry_after_seconds({}, 30) == 60.0
    assert retry_after_seconds({"retry-after": "later"}, 3) == 6.0


def test_provisional_429_rows_are_not_capability_failures(tmp_path):
    store = VersionedRunStore(tmp_path, "v4")
    store.initialize(_plan())
    store.append_result(_provisional_row("m1:d1:i1"))
    store.append_result(_result("m1:d1:i2", correct=True))
    rows, n_prov = store.resolved_rows()
    assert n_prov == 1
    assert rows[0]["provisional"] is True
    public = store.public_summary()
    cell = public["cells"]["m1:d1"]
    assert cell["n_provisional_transport"] == 1
    assert cell["complete"] is False            # never chartable as complete
    assert "capability" in cell["provisional_note"]
    # provisional rows are excluded from the settled denominator
    assert cell["accuracy_recovered_on_settled"] == 1.0
    assert public["n_provisional_transport_rows"] == 1


def test_transport_recovery_is_immutable_and_pointer_resolved(tmp_path):
    store = VersionedRunStore(tmp_path, "v4")
    store.initialize(_plan())
    original = _provisional_row("m1:d1:i1")
    store.append_result(original)
    # recovery attempt: explicit, immutable, separately stored
    rec = {"logical_request_id": "m1:d1:i1", "model": "m1", "dataset": "d1",
           "item_id": "i1", "gold": "A", "terminal": True, "status": "ok",
           "http_status": 200, "recovery_attempt": 1,
           "attempt_kind": "transport-recovery", "pred_recovered": "A",
           "recovery_stage": "exact", "correct_recovered": True}
    store.append_transport_recovery(rec)
    store.set_recovery_pointer("m1:d1:i1", 1)
    rows, n_prov = store.resolved_rows()
    assert n_prov == 0
    assert rows[0]["resolved_by_transport_recovery"] is True
    assert rows[0]["correct_recovered"] is True
    # the original terminal row is NOT deleted or overwritten
    base = store.read_results()[0]
    assert base["http_status"] == 429 and base["status"] == "http_error"
    # duplicates of the same attempt are refused (append-only immutability)
    with pytest.raises(RevisionRunError, match="duplicate transport-recovery"):
        store.append_transport_recovery(rec)
    public = store.public_summary()
    assert public["cells"]["m1:d1"]["complete"] is True
    assert public["cells"]["m1:d1"]["n_transport_recovered"] == 1


def test_unresolved_recovery_attempt_stays_provisional(tmp_path):
    store = VersionedRunStore(tmp_path, "v4")
    store.initialize(_plan())
    store.append_result(_provisional_row("m1:d1:i1"))
    # recovery attempt ALSO hit a rate limit -> still provisional, not wrong
    store.append_transport_recovery({
        "logical_request_id": "m1:d1:i1", "model": "m1", "dataset": "d1",
        "status": "http_error", "http_status": 429, "recovery_attempt": 1,
        "terminal": True, "correct_recovered": False})
    store.set_recovery_pointer("m1:d1:i1", 1)
    rows, n_prov = store.resolved_rows()
    assert n_prov == 1 and rows[0]["provisional"] is True
    assert store.public_summary()["cells"]["m1:d1"]["complete"] is False


def test_per_key_limit_of_lowers_rate_limited_models(tmp_path):
    lock = threading.Lock()
    inflight = {"g": 0, "max_g": 0}

    def worker(task):
        with lock:
            inflight["g"] += 1
            inflight["max_g"] = max(inflight["max_g"], inflight["g"])
        time.sleep(0.02)
        with lock:
            inflight["g"] -= 1
        return {"cost": 0.001}

    bounded_dispatch([("g", i) for i in range(10)], worker,
                     total_workers=8, per_key_limit=4,
                     key_of=lambda t: t[0],
                     per_key_limit_of=lambda k: 2 if k == "g" else 4,
                     spend_cap_usd=5.0, max_cost_of=lambda t: 0.01,
                     price_of=lambda r: r.get("cost"))
    assert inflight["max_g"] <= 2      # rate-limited lane is the tighter one


# ----------------------------------------------- audited cap grant ($40 user)
def test_cap_grant_preserves_spend_and_is_audited(tmp_path):
    ledger = tmp_path / "budget.json"
    b = PersistentBudget(ledger, 10.0)
    r1 = b.reserve("k", 2.0)
    b.settle(r1, 1.5)                    # settled 1.5
    r2 = b.reserve("k", 3.0)
    b.settle(r2, None)                   # held 3.0 (unknown, fail closed)
    r3 = b.reserve("k", 2.0)             # active 2.0  -> exposure 6.5
    before = b.snapshot()
    snap = b.grant_cap(40.0, from_cap_usd=10.0,
                       granted_by="user: $40 combined ceiling (parent-approved)",
                       note="superseding grant")
    after = b.snapshot()
    # settled + held + active PRESERVED exactly — never reset
    assert after["settled_usd"] == before["settled_usd"]
    assert after["held_usd"] == before["held_usd"]
    assert after["active_reserved_usd"] == before["active_reserved_usd"]
    assert after["exposure_usd"] == before["exposure_usd"] == 6.5
    assert after["cap_usd"] == 40.0
    # audit event records oldcap/newcap + grantee
    events = (tmp_path / "budget_events.jsonl").read_text().splitlines()
    grant = [json.loads(e) for e in events
             if json.loads(e)["event"] == "cap_grant"]
    assert grant and grant[-1]["old_cap_usd"] == 10.0
    assert grant[-1]["new_cap_usd"] == 40.0
    assert "user" in grant[-1]["granted_by"]
    history = json.loads(ledger.read_text())["cap_history"]
    assert history[-1]["old_cap_usd"] == 10.0 and history[-1]["new_cap_usd"] == 40.0
    assert history[-1]["exposure_preserved_usd"] == 6.5
    # resumed instance follows the granted cap; OLD cap is now refused
    b2 = PersistentBudget(ledger, 40.0)
    assert b2.snapshot()["exposure_usd"] == 6.5
    with pytest.raises(RevisionRunError, match="cap mismatch"):
        PersistentBudget(ledger, 10.0)


def test_cap_grant_refused_without_exact_previous_cap(tmp_path):
    ledger = tmp_path / "budget.json"
    b = PersistentBudget(ledger, 10.0)
    with pytest.raises(RevisionRunError, match="must name the exact previous cap"):
        b.grant_cap(40.0, from_cap_usd=7.5, granted_by="user:x")
    assert b.snapshot()["cap_usd"] == 10.0      # nothing changed
    with pytest.raises(RevisionRunError, match="granted_by"):
        b.grant_cap(40.0, from_cap_usd=10.0, granted_by="")
    assert b.snapshot()["cap_usd"] == 10.0


def test_cap_grant_never_reduces_below_commitments(tmp_path):
    ledger = tmp_path / "budget.json"
    b = PersistentBudget(ledger, 10.0)
    r1 = b.reserve("k", 6.0)
    b.settle(r1, 6.0)                     # settled 6.0
    with pytest.raises(RevisionRunError, match="below current commitments"):
        b.grant_cap(3.0, from_cap_usd=10.0, granted_by="user:x")
    snap = b.snapshot()
    assert snap["cap_usd"] == 10.0 and snap["exposure_usd"] == 6.0
    # and a rename-style ledger/version bypass is impossible: same file, same
    # spend — only an audited grant can move the cap
    b.grant_cap(40.0, from_cap_usd=10.0, granted_by="user:x")
    assert b.snapshot()["exposure_usd"] == 6.0
    with pytest.raises(BudgetExceeded):
        b.reserve("k", 35.0)             # 6.0 + 35.0 > 40.0


def test_wall_stop_checkpoints_cleanly_without_abandoned_reservations(tmp_path):
    """--max-wall-seconds style checkpoint: at the deadline no NEW call starts,
    in-flight calls settle normally, so no reservation is left active."""
    budget = PersistentBudget(tmp_path / "budget.json", 10.0)

    def worker(task):
        time.sleep(0.05)
        return {"cost": 0.01}

    stats = bounded_dispatch(
        [(i,) for i in range(200)], worker, total_workers=4, per_key_limit=4,
        key_of=lambda t: "k", budget=budget, max_cost_of=lambda t: 0.05,
        price_of=lambda r: r.get("cost"), stop_after_seconds=0.3)
    snap = budget.snapshot()
    assert stats["wall_stop"] is True
    assert stats["dispatched"] < 200
    assert snap["active_reserved_usd"] == 0.0      # nothing abandoned
    assert snap["exposure_usd"] == pytest.approx(snap["settled_usd"])
