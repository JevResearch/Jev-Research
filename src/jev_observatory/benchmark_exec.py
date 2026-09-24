"""Full-benchmark executor (parent BENCHMARK-EXECUTION-GATE).

Executes EXACTLY the frozen benchmark stages — full MMLU-Pro TEST
(12,032), the 140×3 option-rotation audit (420), and the ARC-Challenge TEST
(1,172) — against a single Jev HTTP endpoint. It is a small module on top of
the inspected primitives (RunStore, BudgetLedger, RateLimiter, JevProvider,
plan_run), not a framework.

Safety contract (gate "Execution safety / acceptance"):

* hard request ceiling 15,000 across the whole suite, checked at freeze time
  and re-enforced by the BudgetLedger ``max_requests`` cap;
* zero retries; strict-format failures are recorded truth, never re-asked;
* low concurrency (default 4) with a conservative RPM ceiling (default 600);
* wall-time budget; cancellation is cooperative (checked before every dispatch);
* atomic per-attempt persistence (RunStore fsync + file locks);
* fail-closed resume: uncertain (timeout) attempts refuse the run outright —
  the provider may have billed unobserved work; a NEW run is required;
* budget restore from stored attempts; if restoring would exceed the caps the
  run is refused — historical attempts are never silently omitted and
  unknown usage is never counted as free;
* finished (terminal) logical requests are never re-dispatched or relabelled;
* stop on 401/402/403 (credit/auth), on repeated transport/schema failures,
  on budget denial, on wall-time expiry, and on cancellation; partial
  artifacts are always preserved;
* dispatch-time wire-hash verification: the exact payload about to leave the
  process must match the frozen ordered payload hash;
* dry-run plans + hash-verifies everything and constructs NO transport.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable

from .benchmark_spec import (
    EXTENDED_STAGES,
    EXTENDED_STAGE_TITLES,
    REQUEST_CEILING,
    STAGES,
    BenchmarkSpecError,
)
from .experiment import logical_requests, spec_sha256
from .guards import BudgetCaps, BudgetExceeded, BudgetLedger, Cancelled, Cancellation, RateLimiter
from .manifest import utc_now, verify_manifest
from .mech_dryrun import ordered_payload_json
from .providers import JevProvider, RetryPolicy
from .redact import Redactor
from .runner import HookAdapter, Runner, plan_run
from .reviewed_executor import _NonRaisingTransport

EXECUTOR_VERSION = "benchmark-executor-1.0.0"
FREEZE_VERSION = "benchmark-freeze-1.0.0"

DEFAULT_CONCURRENCY = 4
DEFAULT_RPM = 600
CONSECUTIVE_FAILURE_LIMIT = 3
FATAL_HTTP_STATUSES = frozenset({401, 402, 403})
FAILURE_STATUSES = frozenset({"http_error", "transport_error", "malformed_json",
                              "contract_invalid", "timeout"})
# Budget defaults for the FULL Jev benchmark (parent: cost is not the
# constraint; accuracy and recoverability are). The 15,000-request ceiling and
# the conservative token cap are the real gates.
DEFAULT_MAX_REQUESTS = REQUEST_CEILING
DEFAULT_MAX_ESTIMATED_INPUT_TOKENS = 60_000_000   # ~60M conservative chars≈tokens
DEFAULT_MAX_WALL_SECONDS = 6 * 3600.0

FREEZE_ROOT_DEFAULT = "runs_benchmark"


class ExecutorError(RuntimeError):
    """Fail-closed condition: the executor refuses to continue."""


# --------------------------------------------------------------------- freeze
def _file_sha256(path: Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def _freeze_deterministic(doc: dict[str, Any]) -> str:
    trimmed = {k: v for k, v in doc.items() if k not in ("created_at", "deterministic_sha256")}
    return sha256(json.dumps(trimmed, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":")).encode("utf-8")).hexdigest()


def _wire_hashes(spec: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    """Ordered (wire) + canonical payload hash per logical request id."""
    wire: dict[str, str] = {}
    canonical: dict[str, str] = {}
    for lr in logical_requests(spec):
        payload = lr.request.to_payload()
        wire[lr.logical_request_id] = sha256(
            ordered_payload_json(payload).encode("utf-8")).hexdigest()
        canonical[lr.logical_request_id] = lr.request.request_sha256()
    return wire, canonical


def build_freeze(suite: dict[str, Any], root: str | Path = FREEZE_ROOT_DEFAULT) -> dict[str, Any]:
    """Freeze specs + ordered outbound payload hashes BEFORE any dispatch.

    Write-once: an existing freeze is reused only if it byte-matches the
    deterministic recomputation; any difference is a hard error.
    """
    root_path = Path(root)
    total = suite["total_requests"]
    if total > REQUEST_CEILING:
        raise BenchmarkSpecError(
            f"freeze refused: {total} requests exceed the {REQUEST_CEILING} ceiling"
        )
    stages: dict[str, Any] = {}
    for stage in suite:
        if stage in ("counts", "total_requests", "request_ceiling"):
            continue
        if stage not in EXTENDED_STAGES:
            raise ExecutorError(f"unknown suite stage {stage!r}; known: {EXTENDED_STAGES}")
        spec = suite[stage]["spec"]
        wire, canonical = _wire_hashes(spec)
        stages[stage] = {
            "title": EXTENDED_STAGE_TITLES[stage],
            "experiment": spec["experiment"],
            "model": spec["model"],
            "n_requests": len(spec["items"]),
            "spec_sha256": spec_sha256({
                "experiment": spec.get("experiment", "unspecified"),
                "model": spec.get("model", "jev-1.13.0"),
                "items": spec["items"],
                "seeds": spec.get("seeds", {}),
                "shuffle": bool(spec.get("shuffle", True)),
            }),
            "source_file": suite[stage]["source"]["file"],
            "source_sha256": suite[stage]["source"]["sha256"],
            "source_url": suite[stage]["source"]["url"],
            "wire_sha256": wire,
            "canonical_sha256": canonical,
        }
    doc = {
        "freeze_version": FREEZE_VERSION,
        "created_at": utc_now(),
        "executor_version": EXECUTOR_VERSION,
        "total_requests": total,
        "request_ceiling": REQUEST_CEILING,
        "budget_caps": {
            "max_requests": DEFAULT_MAX_REQUESTS,
            "max_estimated_input_tokens": DEFAULT_MAX_ESTIMATED_INPUT_TOKENS,
            "retries": "none",
            "concurrency": DEFAULT_CONCURRENCY,
            "rpm_ceiling": DEFAULT_RPM,
        },
        "protocol": {
            "fixed_instructions": True,
            "no_chain_of_thought": True,
            "correct_answer_retries": 0,
            "gold_in_outbound": "forbidden (structural)",
        },
        "stages": stages,
    }
    doc["deterministic_sha256"] = _freeze_deterministic(doc)
    freeze_path = root_path / "freeze" / "frozen.json"
    if freeze_path.exists():
        existing = load_freeze(root_path)
        if existing.get("deterministic_sha256") == doc["deterministic_sha256"]:
            return existing
        raise ExecutorError(
            "an existing freeze document differs from the current specs; refusing to "
            "overwrite or relabel frozen items (build a new freeze root instead)"
        )
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    freeze_path.write_text(
        json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return doc


def load_freeze(root: str | Path = FREEZE_ROOT_DEFAULT) -> dict[str, Any]:
    path = Path(root) / "freeze" / "frozen.json"
    if not path.exists():
        raise ExecutorError(f"no freeze document at {path}; run the offline preparation first")
    doc = json.loads(path.read_text(encoding="utf-8"))
    if _freeze_deterministic(doc) != doc.get("deterministic_sha256"):
        raise ExecutorError(f"freeze document {path} failed its integrity check")
    return doc


def stage_run_id(freeze: dict[str, Any], stage: str) -> str:
    """Deterministic run id resolved FROM the freeze, never lexicographic glob."""
    return f"bench-{stage}-{freeze['deterministic_sha256'][:12]}"


# ------------------------------------------------------------------- executor
class BenchmarkExecutor:
    """Freeze-verify → dispatch the frozen benchmark stages (or dry-run)."""

    def __init__(
        self,
        root: str | Path = FREEZE_ROOT_DEFAULT,
        *,
        provider_factory: Callable[[], Any] | None = None,
        clock: Callable[[], float] | None = None,
        sleeper: Callable[[float], None] | None = None,
        max_requests: int = DEFAULT_MAX_REQUESTS,
        max_estimated_input_tokens: int = DEFAULT_MAX_ESTIMATED_INPUT_TOKENS,
        max_wall_seconds: float = DEFAULT_MAX_WALL_SECONDS,
        concurrency: int = DEFAULT_CONCURRENCY,
        rpm: int = DEFAULT_RPM,
        consecutive_failure_limit: int = CONSECUTIVE_FAILURE_LIMIT,
        stages: tuple[str, ...] | None = None,
    ) -> None:
        self.root = Path(root)
        self.provider_factory = provider_factory
        self.clock = clock or time.monotonic
        self.sleeper = sleeper or time.sleep
        self.max_requests = max_requests
        self.max_estimated_input_tokens = max_estimated_input_tokens
        self.max_wall_seconds = max_wall_seconds
        self.concurrency = max(1, int(concurrency))
        self.rpm = int(rpm)
        self.consecutive_failure_limit = consecutive_failure_limit
        self.stages = tuple(stages) if stages else STAGES
        unknown = [s for s in self.stages if s not in EXTENDED_STAGES]
        if unknown:
            raise ExecutorError(f"unknown stage(s) {unknown}; known: {EXTENDED_STAGES}")
        self.cancellation = Cancellation()
        self.stop_reason: str | None = None
        self.budget: BudgetLedger | None = None
        self.deadline: float | None = None
        self._consecutive_failures = 0
        self._state_lock = threading.Lock()
        self._frozen_wire: dict[str, str] = {}
        self._n_wire_verified = 0
        self._last_provider_exception: str | None = None

    # ------------------------------------------------------------- top level
    def run(self, *, dry_run: bool = True) -> dict[str, Any]:
        freeze = load_freeze(self.root)
        state = self._state(dry_run=dry_run, freeze=freeze)
        stage_run_ids: dict[str, str] = {}
        verified_specs: dict[str, dict[str, Any]] = {}
        for stage in self.stages:
            if stage not in freeze["stages"]:
                raise ExecutorError(
                    f"requested stage {stage!r} is not in the freeze "
                    f"({sorted(freeze['stages'])}); re-run offline preparation"
                )
            run_id, verification, spec = self._plan_and_verify(stage, freeze)
            stage_run_ids[stage] = run_id
            verified_specs[stage] = spec
            state["stages"][stage] = {
                "run_id": run_id,
                "freeze_verification": "pass",
                "manifest_sha256": verification["manifest_sha256"],
                "plan": verification["plan"],
            }
        if dry_run:
            state["note"] = "dry run: planned + hash-verified; nothing dispatched, no provider constructed"
            self._write_state(state)
            return state
        frozen_wire: dict[str, str] = {}
        for stage in stage_run_ids:
            frozen_wire.update(freeze["stages"][stage]["wire_sha256"])
        self._frozen_wire = frozen_wire
        self._n_wire_verified = 0
        try:
            # Fail-closed resume gate BEFORE any budget restoration: uncertain
            # (possibly billed) attempts refuse the run outright.
            for stage, run_id in stage_run_ids.items():
                uncertain = self._store(run_id).uncertain_ids()
                if uncertain:
                    raise ExecutorError(
                        f"fail-closed resume refusal: {len(uncertain)} uncertain (timeout) "
                        f"attempt(s) in stage {stage!r} ({sorted(uncertain)[:3]}...); the provider "
                        "may have billed unobserved work. This executor does not resume past "
                        "ambiguous timeouts: start a NEW run. Budgets were not reset."
                    )
            budget = BudgetLedger(BudgetCaps(
                max_requests=self.max_requests,
                max_estimated_input_tokens=self.max_estimated_input_tokens,
            ))
            restored: dict[str, Any] = {
                "stages": {}, "attempts": 0, "reported": 0, "unknown_kept_reserved": 0,
                "refused": None,
            }
            state["budget_restored_from_attempts"] = restored
            self._restore_budget(budget, freeze, stage_run_ids, restored)
            self.budget = budget
            self.deadline = self.clock() + self.max_wall_seconds
            for stage in self.stages:
                if stage not in stage_run_ids:
                    continue
                if self.stop_reason is not None or self.cancellation.cancelled:
                    if self.stop_reason is None:
                        self._stop("cancelled_by_cancellation")
                    state["stages"][stage]["skipped_reason"] = self.stop_reason
                    state["stages"][stage]["budget_snapshot"] = budget.snapshot()
                    continue
                wire_before = self._n_wire_verified
                summary = self._run_stage(stage, stage_run_ids[stage], budget,
                                          verified_specs[stage])
                summary["n_dispatched_wire_hashes_verified"] = self._n_wire_verified - wire_before
                state["stages"][stage].update(summary)
                state["stages"][stage]["budget_snapshot"] = budget.snapshot()
                self._write_state(state)
        except ExecutorError as exc:
            state["executor_error"] = str(exc)
            state["global_stop_reason"] = self.stop_reason
            self._write_state(state)
            raise
        state["budget_final"] = budget.snapshot()
        state["global_stop_reason"] = self.stop_reason
        state["consecutive_failures_at_stop"] = self._consecutive_failures
        self._write_state(state)
        return state

    def _state(self, *, dry_run: bool, freeze: dict[str, Any]) -> dict[str, Any]:
        return {
            "executor_version": EXECUTOR_VERSION,
            "dry_run": bool(dry_run),
            "freeze_sha256": freeze["deterministic_sha256"],
            "budget_caps": {
                "max_requests": self.max_requests,
                "max_estimated_input_tokens": self.max_estimated_input_tokens,
                "max_wall_seconds": self.max_wall_seconds,
                "concurrency": self.concurrency,
                "rpm": self.rpm,
                "retries": "none",
            },
            "stages": {},
            "global_stop_reason": None,
            "recorded_at": utc_now(),
        }

    # ----------------------------------------------------------- plan/verify
    def _plan_and_verify(self, stage: str, freeze: dict[str, Any]
                         ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        """Plan (or reuse the immutable plan), reload stored items, verify the
        payload hashes against the freeze. Only these requests may dispatch."""
        freeze_stage = freeze["stages"][stage]
        run_id = stage_run_id(freeze, stage)
        directory = self.root / run_id
        if (directory / "manifest.json").exists():
            if not (directory / "items.jsonl").exists():
                raise ExecutorError(
                    f"stage {stage}: incomplete plan at {run_id} (manifest without items)"
                )
        else:
            # plan_run needs the real items to expand; they come from the spec
            # written next to the run dir by offline preparation and verified
            # against the freeze hash BEFORE any dispatch.
            spec_path = directory / "spec.json"
            if not spec_path.exists():
                raise ExecutorError(
                    f"stage {stage}: no planned spec at {spec_path}; re-run offline "
                    "preparation (it must write spec.json + manifest + items)"
                )
            source_spec = json.loads(spec_path.read_text(encoding="utf-8"))
            plan_run(source_spec, provider="jev", root=str(self.root), run_id=run_id,
                     budgets={"max_requests": self.max_requests,
                              "max_estimated_input_tokens": self.max_estimated_input_tokens},
                     retry_policy={"max_attempts": 1})
        manifest = verify_manifest(self.root / run_id)
        items = [json.loads(line) for line in
                 (self.root / run_id / "items.jsonl").read_text(encoding="utf-8").split("\n")
                 if line.strip()]
        reloaded_spec = {
            "experiment": manifest.experiment,
            "model": manifest.model_requested,
            "items": items,
            "seeds": manifest.seeds,
            "shuffle": manifest.shuffle,
        }
        if spec_sha256(reloaded_spec) != manifest.items_sha256:
            raise ExecutorError(
                f"stage {stage}: stored items.jsonl does not match its manifest hash; "
                "refusing to run against mutated items"
            )
        if spec_sha256(reloaded_spec) != freeze_stage["spec_sha256"]:
            raise ExecutorError(
                f"stage {stage}: stored items do not match the frozen spec hash; "
                "items changed since the freeze — aborting"
            )
        wire, canonical = _wire_hashes(reloaded_spec)
        if wire != freeze_stage["wire_sha256"] or canonical != freeze_stage["canonical_sha256"]:
            raise ExecutorError(
                f"stage {stage}: reloaded payload hashes do not match the frozen ordered "
                "payload hashes; items changed since the freeze — aborting"
            )
        if len(reloaded_spec["items"]) != freeze_stage["n_requests"]:
            raise ExecutorError(f"stage {stage}: item count drifted from the freeze")
        plan_doc = json.loads((self.root / run_id / "plan.json").read_text(encoding="utf-8"))
        return run_id, {"manifest_sha256": manifest.sha256(), "plan": plan_doc}, reloaded_spec

    def _store(self, run_id: str):
        from .ledger import RunStore

        return RunStore(self.root, run_id, redactor=Redactor.from_environment())

    # ---------------------------------------------------------------- budget
    def _restore_budget(self, budget: BudgetLedger, freeze: dict[str, Any],
                        stage_run_ids: dict[str, str], restored: dict[str, Any]) -> None:
        """Restore aggregate reservations from recorded attempts (resume).

        Server-reported usage is committed; attempts without usage keep their
        conservative estimate reserved (unknown usage is never counted as
        free). If restoring would exceed the caps the run is REFUSED
        fail-closed — historical attempts are never silently omitted.
        """
        for stage, run_id in stage_run_ids.items():
            directory = self.root / run_id
            if not (directory / "attempts.jsonl").exists():
                continue
            store = self._store(run_id)
            restored["stages"][stage] = run_id
            for attempt in store.attempts():
                try:
                    reservation = budget.reserve(
                        max(1, int(attempt.get("estimated_input_tokens") or 1)))
                except BudgetExceeded as exc:
                    restored["refused"] = {
                        "run_id": run_id, "attempt_id": attempt.get("attempt_id"),
                        "limit": exc.limit,
                    }
                    raise ExecutorError(
                        f"fail-closed resume refusal: historical attempt "
                        f"{attempt.get('attempt_id')!r} in {run_id} exceeds the budget "
                        f"caps while restoring reservations ({exc}). Historical attempts "
                        "are never silently omitted or counted as free; budgets were "
                        "not reset. Start a NEW run instead."
                    ) from exc
                budget.reconcile(reservation, attempt.get("usage_input_tokens"))
                restored["attempts"] += 1
                if attempt.get("usage_input_tokens") is None:
                    restored["unknown_kept_reserved"] += 1
                else:
                    restored["reported"] += int(attempt["usage_input_tokens"])

    # ------------------------------------------------------------ live stage
    def _run_stage(self, stage: str, run_id: str, budget: BudgetLedger,
                   verified_spec: dict[str, Any]) -> dict[str, Any]:
        store = self._store(run_id)
        wall_start = self.clock()
        requests = logical_requests(verified_spec)
        provider = self._new_provider()
        hook = HookAdapter(budget=budget,
                           rate=RateLimiter(requests_per_minute=self.rpm,
                                            clock=self.clock, sleeper=self.sleeper),
                           cancellation=self.cancellation)
        lock = threading.Lock()
        tally = {"dispatched": 0, "ok": 0, "error": 0, "skipped_resume": 0}
        completed_before = store.completed_ids()

        def worker(lr) -> None:
            if self.cancellation.cancelled or self.stop_reason is not None:
                return
            if lr.logical_request_id in completed_before:
                return  # counted once, below, from the stored set
            counts = self._dispatch(provider, lr, hook, store)
            if counts is None:
                return
            with lock:
                tally["dispatched"] += 1
                if counts["status"]:
                    tally["ok"] += 1
                else:
                    tally["error"] += 1

        try:
            with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                list(pool.map(worker, requests))
        finally:
            provider.close()
        tally["skipped_resume"] = len([lr for lr in requests
                                       if lr.logical_request_id in completed_before])
        summary = {
            "run_id": run_id,
            "connection_policy": "one_client_per_stage_retained_then_closed",
            "concurrency": self.concurrency,
            "rpm_ceiling": self.rpm,
            **tally,
            "n_requests": len(requests),
            "stop_reason": self.stop_reason,
            "wall_start_monotonic": wall_start,
            "wall_end_monotonic": self.clock(),
            "stage_wall_seconds": round(self.clock() - wall_start, 6),
            "budget_snapshot": budget.snapshot(),
        }
        return summary

    def _new_provider(self) -> Any:
        if self.provider_factory is not None:
            return self.provider_factory()
        transport = _NonRaisingTransport(self._default_jev_transport_factory()())
        # The environment-derived credential redactor is propagated so that
        # provider-side exceptions can never leak the env key literal.
        return JevProvider(transport, retry_policy=RetryPolicy.none(),
                           sleeper=self.sleeper, redactor=Redactor.from_environment())

    def _default_jev_transport_factory(self) -> Callable[[], Any]:
        from .redact import API_KEY_ENV, BASE_URL_ENV, LIVE_OPT_IN_ENV, base_url, get_api_key
        import os

        if not os.environ.get(LIVE_OPT_IN_ENV) == "1":
            raise ExecutorError(f"live calls require environment {LIVE_OPT_IN_ENV}=1")
        api_key = get_api_key()
        if not api_key:
            raise ExecutorError(f"no API key in process environment ({API_KEY_ENV})")
        url = base_url()

        def factory():
            from .transport import HttpxTransport

            return HttpxTransport(url, api_key, timeout_seconds=300.0)

        return factory

    def _stop(self, reason: str) -> None:
        with self._state_lock:
            if self.stop_reason is None:
                self.stop_reason = reason
        self.cancellation.cancel()

    def _observe_outcome(self, outcome) -> None:
        stop: str | None = None
        with self._state_lock:
            if outcome.attempts and outcome.attempts[-1].http_status in FATAL_HTTP_STATUSES:
                stop = "insufficient_credit_or_auth"
            elif outcome.status == "ok":
                self._consecutive_failures = 0
            elif outcome.status in FAILURE_STATUSES:
                self._consecutive_failures += 1
                if self._consecutive_failures >= self.consecutive_failure_limit:
                    stop = "repeated_transport_schema_failures"
        if stop is not None:
            self._stop(stop)

    def _verify_dispatch_wire(self, lr) -> None:
        """Verify the EXACT payload about to leave the process against the freeze."""
        expected = self._frozen_wire.get(lr.logical_request_id)
        if expected is None:
            raise ExecutorError(
                f"refusing to dispatch {lr.logical_request_id!r}: not present in the "
                "frozen wire hash map"
            )
        payload = lr.request.to_payload()
        actual = sha256(ordered_payload_json(payload).encode("utf-8")).hexdigest()
        if actual != expected:
            self._stop("dispatch_wire_hash_mismatch")
            raise ExecutorError(
                f"dispatch wire hash mismatch for {lr.logical_request_id!r}: the payload "
                "about to leave the process does not match the frozen ordered payload "
                "hash — nothing was dispatched"
            )
        with self._state_lock:
            self._n_wire_verified += 1

    def _dispatch(self, provider, lr, hook, store) -> dict[str, int] | None:
        """Dispatch one logical request; persist attempts + terminal result."""
        if self.cancellation.cancelled or self.stop_reason is not None:
            return None  # never start a new call after stop
        if self.deadline is not None and self.clock() > self.deadline:
            self._stop("wall_time_budget_exceeded")
            return None
        self._verify_dispatch_wire(lr)
        try:
            outcome = provider.ask(lr.request, lr.logical_request_id, hook=hook,
                                   cancellation=self.cancellation, store=store)
        except BudgetExceeded as exc:
            self._stop(f"budget_cap_reached:{exc.limit}")
            return None
        except Cancelled:
            # cooperative cancellation raised from the hook-before path:
            # recorded cleanly, no further calls started
            self._stop("cancelled_by_cancellation")
            return None
        except Exception as exc:  # provider explosions are recorded, never raised past here
            self._observe_failure_exception(exc)
            return None
        for attempt in outcome.attempts:
            store.append_attempt(attempt.to_dict())
        if outcome.terminal and outcome.status not in {"cancelled", "timeout"}:
            store.append_result(Runner._result_record(None, outcome, lr))
        self._observe_outcome(outcome)
        return {"status": 1 if outcome.status == "ok" else 0, "dispatched": 1}

    def _observe_failure_exception(self, exc: BaseException) -> None:
        # Observability: provider exceptions are otherwise swallowed after three
        # consecutive failures; record the LAST one (redacted repr) so a
        # fail-closed stop is always diagnosable offline.
        try:
            redacted = self._store("observability").redactor.text(repr(exc))
        except Exception:  # noqa: BLE001 - observability must never raise
            redacted = repr(exc)
        self._last_provider_exception = redacted[:500]
        with self._state_lock:
            self._consecutive_failures += 1
            stop = (self._consecutive_failures >= self.consecutive_failure_limit)
        if stop:
            self._stop("provider_exception_failures")

    def _write_state(self, state: dict[str, Any]) -> None:
        state["last_provider_exception"] = self._last_provider_exception
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / "executor_state.json"
        path.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
