"""Experiment execution: randomized order, bounded concurrency, budgets, resume.

Guarantees implemented here (DESIGN.md §11, IMPLEMENTATION.md M0):

* no dispatch after cancellation or a budget denial (fail-fast halt);
* every attempt and every terminal observation is appended to the ledger;
* resume never re-dispatches a terminal logical request and never overwrites one;
* timeouts are recorded as `uncertain` and a resume needs an explicit policy;
* the runner never sees gold labels: analysis joins them later from the spec.
"""

from __future__ import annotations

import json
import signal
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .experiment import LogicalRequest, logical_requests, item_hashes, spec_sha256
from .experiments import condition_param
from .guards import BudgetCaps, BudgetExceeded, BudgetLedger, Cancellation, RateLimiter
from .manifest import RunManifest, canonical_sha256, utc_now, verify_manifest, write_manifest
from .providers import AttemptHook, BaseProvider, CallOutcome
from .schema import DEFAULT_MODEL_PIN, estimate_input_tokens

_UNCERTAIN_POLICIES = ("error", "retry", "skip")


class UncertainPolicyError(RuntimeError):
    """Resuming into uncertain attempts without an explicit policy is refused."""


@dataclass
class HookAdapter(AttemptHook):
    """Per-attempt accounting: cancel-check -> budget reserve -> rate-limit."""

    budget: BudgetLedger
    rate: RateLimiter
    cancellation: Cancellation

    def before(self, estimated_input_tokens: int):
        self.cancellation.raise_if_cancelled()
        reservation = self.budget.reserve(estimated_input_tokens)  # raises BudgetExceeded
        try:
            self.rate.acquire(estimated_input_tokens, self.cancellation)
        except BaseException:
            self.budget.release(reservation)
            raise
        return reservation

    def after(self, handle, reported_input_tokens: int | None) -> None:
        self.budget.reconcile(handle, reported_input_tokens)


@dataclass
class RunResult:
    run_id: str
    n_dispatched: int = 0
    n_skipped_resume: int = 0
    n_ok: int = 0
    n_contract_invalid: int = 0
    n_error: int = 0
    n_uncertain: int = 0
    n_cancelled: int = 0
    halted: bool = False


class Runner:
    def __init__(
        self,
        provider: BaseProvider,
        store: Any,
        *,
        concurrency: int = 1,
        budget: BudgetLedger | None = None,
        rate: RateLimiter | None = None,
        transport_reset: Any = None,
    ) -> None:
        self.provider = provider
        self.store = store
        self.concurrency = max(1, int(concurrency))
        self.budget = budget
        self.rate = rate
        self.transport_reset = transport_reset  # called before `cold` conditions
        self.cancellation = Cancellation()

    def install_signal_handlers(self) -> None:
        def handler(signum, frame):  # pragma: no cover - signal path
            self.cancellation.cancel()

        try:
            signal.signal(signal.SIGINT, handler)
            signal.signal(signal.SIGTERM, handler)
        except ValueError:  # pragma: no cover - not main thread
            pass

    def run(self, requests: list[LogicalRequest], *, on_uncertain: str = "error") -> RunResult:
        if on_uncertain not in {"error", "retry", "skip"}:
            raise ValueError("on_uncertain must be error|retry|skip")

        completed = self.store.completed_ids()
        uncertain = self.store.uncertain_ids()
        if uncertain and on_uncertain == "error":
            raise UncertainPolicyError(
                f"{len(uncertain)} uncertain request(s) exist ({sorted(uncertain)[:3]}...); "
                "resume requires an explicit --on-uncertain retry|skip|error policy"
            )
        # A pre-cancelled runner dispatches nothing; that is a valid outcome,
        # not an error (workers check cancellation before every dispatch).
        tally = {"dispatched": 0, "ok": 0, "contract_invalid": 0, "error": 0, "uncertain": 0, "cancelled": 0}
        pending: list[LogicalRequest] = []
        n_skipped = 0
        for lr in requests:
            lid = lr.logical_request_id
            if lid in completed:
                n_skipped += 1
                continue
            if lid in uncertain and on_uncertain == "skip":
                n_skipped += 1
                continue
            pending.append(lr)

        hook = HookAdapter(
            budget=self.budget or BudgetLedger(BudgetCaps()),
            rate=self.rate or RateLimiter(),
            cancellation=self.cancellation,
        )
        halt = threading.Event()
        lock = threading.Lock()

        def worker(lr: LogicalRequest) -> None:
            if halt.is_set() or self.cancellation.cancelled:
                return
            if (
                self.transport_reset is not None
                and condition_param(lr.condition, "cold") == "1"
            ):
                self.transport_reset()  # next attempt is genuinely cold
            try:
                outcome = self.provider.ask(
                    lr.request,
                    lr.logical_request_id,
                    hook=hook,
                    cancellation=self.cancellation,
                    store=self.store,
                )
            except BudgetExceeded:
                halt.set()
                return
            with lock:
                tally["dispatched"] += 1
                if outcome.status == "ok":
                    tally["ok"] += 1
                elif outcome.status == "contract_invalid":
                    tally["contract_invalid"] += 1
                elif outcome.status == "timeout":
                    tally["uncertain"] += 1
                elif outcome.status == "cancelled":
                    tally["cancelled"] += 1
                else:
                    tally["error"] += 1
            for attempt in outcome.attempts:
                self.store.append_attempt(attempt.to_dict())
            if outcome.terminal and outcome.status not in {"cancelled", "timeout"}:
                self.store.append_result(self._result_record(outcome, lr))

        def drain(pool: ThreadPoolExecutor, futures) -> None:
            for future in as_completed(futures):
                try:
                    future.result()
                except BudgetExceeded:  # noqa: PERF203 - one denial stops scheduling
                    halt.set()

        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = [pool.submit(worker, lr) for lr in pending]
            try:
                drain(pool, futures)
            except (BudgetExceeded, KeyboardInterrupt):
                self.cancellation.cancel()
                drain(pool, futures)

        return RunResult(
            run_id=self.store.run_id,
            n_dispatched=tally["dispatched"],
            n_skipped_resume=n_skipped,
            n_ok=tally["ok"],
            n_contract_invalid=tally["contract_invalid"],
            n_error=tally["error"],
            n_uncertain=tally["uncertain"],
            n_cancelled=tally["cancelled"],
            halted=halt.is_set(),
        )

    # ---------------------------------------------------------------- result
    def _result_record(self, outcome: CallOutcome, lr: LogicalRequest) -> dict[str, Any]:
        validated = outcome.validated
        predictions: dict[str, Any] = {}
        if validated is not None:
            for qid, answer in validated.answers.items():
                predictions[qid] = {"type": answer.question_type, "usable": answer.usable, **_prediction_fields(answer)}
        first = outcome.attempts[0] if outcome.attempts else None
        return {
            "logical_request_id": lr.logical_request_id,
            "terminal": True,
            "status": outcome.status,
            "item_id": lr.item.item_id,
            "group": lr.item.group,
            "cluster": lr.item.cluster,
            "condition": lr.condition,
            "provider": outcome.attempts[-1].provider if outcome.attempts else None,
            "model_requested": lr.request.model,
            "model_returned": validated.returned_model if validated is not None else None,
            "n_attempts": len(outcome.attempts),
            "attempt_ids": [a.attempt_id for a in outcome.attempts],
            "n_retries": outcome.n_retries,
            "request_sha256": outcome.request_sha256,
            "measures": lr.request.measures(),
            "option_map": lr.request.option_maps(),
            "predictions": predictions,
            "violation_codes": outcome.violation_codes(),
            "usage_input_tokens": outcome.usage_input_tokens,
            "usage_output_tokens": outcome.usage_output_tokens,
            "latency_ms_total": round(outcome.total_latency_ms, 3),
            "latency_ms_first_attempt": (
                round(outcome.attempts[0].latency_ms, 3) if outcome.attempts else None
            ),
            "first_byte_ms_first_attempt": (
                round(outcome.attempts[0].first_byte_ms, 3)
                if outcome.attempts and outcome.attempts[0].first_byte_ms is not None
                else None
            ),
            "cold_connection_first_attempt": outcome.attempts[0].cold_connection if outcome.attempts else None,
            "recorded_at": utc_now(),
            "note": outcome.note,
        }

    def run_summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "run_id": self.store.run_id,
            "recorded_at": utc_now(),
            "attempts": len(self.store.attempts()),
            "results": len(self.store.results()),
            "uncertain_open": sorted(self.store.uncertain_ids()),
        }
        if self.budget is not None:
            summary["budget"] = self.budget.snapshot()
        return summary

    def write_summary(self) -> dict[str, Any]:
        """Write the current summary; prior summaries are kept in a history file."""
        summary = self.run_summary()
        path = self.store.directory / "run_summary.json"
        text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
        if path.exists():
            history = self.store.directory / "run_summary.history.jsonl"
            with history.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(summary, sort_keys=True, default=str) + "\n")
        with path.open("w", encoding="utf-8") as handle:
            handle.write(text)
        return summary


def _prediction_fields(answer) -> dict[str, Any]:
    """Flatten validated values into per-question prediction fields."""
    values = answer.values
    if answer.question_type == "noul":
        return {"noul": values.get("p_yes")}
    if answer.question_type == "choice":
        return {
            "choice": values.get("choice"),
            "probabilities": values.get("probabilities"),
            "p_max": values.get("p_max"),
        }
    return {
        "score": values.get("score"),
        "probabilities": values.get("probabilities"),
        "expectation_from_probabilities": values.get("expectation_from_probabilities"),
    }


def store_root(path: str | None = None) -> Path:
    root = Path(path or "runs")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _hashable_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """The exact keys a run reconstructs from its artifacts.

    The manifest hash must cover precisely what `_spec_from_run` rebuilds
    (experiment, model, items, seeds), not incidental spec keys, or every run
    would fail its own integrity check.
    """
    return {
        "experiment": spec.get("experiment", "unspecified"),
        "model": spec.get("model", DEFAULT_MODEL_PIN),
        "items": spec["items"],
        "seeds": spec.get("seeds", {}),
        "shuffle": bool(spec.get("shuffle", True)),
    }


def plan_run(
    spec: dict[str, Any],
    *,
    provider: str,
    root: str | None = None,
    budgets: dict[str, Any] | None = None,
    retry_policy: dict[str, Any] | None = None,
    run_id: str | None = None,
) -> tuple[str, RunManifest]:
    """Build the run directory, manifest and plan estimate. Dispatches nothing.

    ``run_id`` may be pinned (reviewed-executor runs use deterministic ids so a
    resume addresses the same directory); the default remains date+hash based.
    """
    requests = logical_requests(spec)  # also validates the schema + leakage guard
    items_digest = spec_sha256(_hashable_spec(spec))
    run_id = run_id or f"{utc_now()[:10]}-{canonical_sha256({'spec': items_digest})[:12]}"
    manifest = RunManifest(
        run_id=run_id,
        experiment=spec.get("experiment", "unspecified"),
        provider=provider,
        model_requested=spec.get("model", "jev-1.13.0"),
        created_at=utc_now(),
        items_sha256=spec_sha256(_hashable_spec(spec)),
        n_items=len(requests),
        item_hashes=item_hashes(spec),
        option_maps={r.logical_request_id: r.request.option_maps() for r in requests},
        prompt_templates=spec.get("prompt_templates", {}),
        dataset=spec.get("dataset", {}),
        seeds=spec.get("seeds", {}),
        budgets=budgets or {},
        retry_policy=retry_policy or {},
        inclusion_rules=spec.get("inclusion_rules", {}),
        shuffle=bool(spec.get("shuffle", True)),
        pricing={"input_usd_per_mtok": 0.042, "output_tokens": "free (vendor quote S8; not a guarantee)"},
        test_family=spec.get("test_family", "unspecified"),
        preregistration=spec.get("preregistration"),
        execution_region=spec.get("execution_region"),
    )
    directory = store_root(root) / run_id
    directory.mkdir(parents=True, exist_ok=True)
    write_manifest(directory, manifest)
    # Gold labels are copied here for later offline join; they are structurally
    # excluded from every outbound payload (see experiment.check_leakage).
    # NOTE: written WITHOUT sort_keys. Nested insertion order (question order,
    # option order) is design data for randomized/rename/reorder arms; sorting
    # keys here silently destroyed it on the plan→load path (LEAD-GATE blocker).
    # Canonical manifest hashes are computed separately over sorted keys
    # (item_hashes/spec_sha256) and are unaffected.
    items_path = directory / "items.jsonl"
    if not items_path.exists():
        with items_path.open("w", encoding="utf-8") as handle:
            for raw in spec["items"]:
                handle.write(json.dumps(raw, ensure_ascii=False) + "\n")
    estimate = sum(
        estimate_input_tokens(len(json.dumps(r.request.to_payload(), ensure_ascii=False)))
        for r in requests
    )
    plan_doc = {
        "run_id": run_id,
        "n_logical_requests": len(requests),
        "estimated_input_tokens_conservative": estimate,
        "estimated_cost_usd_conservative": round(estimate * 0.042 / 1_000_000.0, 6),
        "budgets": budgets or {},
        "retry_policy": retry_policy or {},
        "manifest_sha256": manifest.sha256(),
        "note": "estimates assume 1 token/char; the server is the only tokenisation authority",
    }
    with (directory / "plan.json").open("w", encoding="utf-8") as handle:
        json.dump(plan_doc, handle, indent=2)
        handle.write("\n")
    return run_id, manifest


def load_manifest(run_id: str, root: str | None = None) -> RunManifest:
    return verify_manifest(store_root(root) / run_id)


def load_run_store(run_id: str, root: str | None = None, redactor=None):
    from .ledger import RunStore

    return RunStore(store_root(root), run_id, redactor=redactor)