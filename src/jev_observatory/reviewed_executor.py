"""Bounded reviewed executor (parent REVIEW-2, 2026-09-18).

Executes EXACTLY the three approved workloads against the frozen, audited
specs — nothing else:

1. ``fresh_test``  — the 120 accepted fresh-v3 TEST items (DEV stays unused);
2. ``mechanism``   — the frozen controlled 12×2×4 mechanism spec (96 requests);
3. ``batching``    — 12 randomized blocks × 3 modes over the EXISTING
   ``BATCH_STATE`` / batch-question content: batched (1 call), sequential-
   separate (4 calls), concurrent-separate (4 calls, max in-flight 4) = 108
   calls, 36 (block, mode) workloads.

Hard limits (parent-approved, enforced here):
* request ceiling 400 across ALL stages, one shared :class:`BudgetLedger`;
* aggregate quoted dollar reservation ceiling $0.10 (same ledger);
* wall-time budget 20 minutes (injectable clock);
* zero retries (``RetryPolicy.none()``) — every attempt is billed truth;
* stop on insufficient credit/authentication (401/402/403) or on repeated
  transport/schema failures; partial artifacts are always preserved;
* fail-closed resume: uncertain (timeout) attempts refuse resume entirely and
  require a new reviewed run; budgets are restored from stored attempts and
  unknown usage is counted conservatively — never silently reset;
* credentials come ONLY from the process environment;
* connection policy: a NEW HTTP client per (batching) workload/mode, retained
  for that workload, closed afterwards. Wall time is the true whole-workload
  monotonic start/end — never the sum of parallel call durations or
  4×median. This is a cold-client workload comparison, not model-compute
  timing and not a warm-cache benchmark.

Dry-run mode plans all three stages, reloads the stored items, and verifies
every ordered outbound payload hash against the freeze document WITHOUT
constructing any transport, credential or network call.
"""

from __future__ import annotations

import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable

from .experiment import load_experiment, logical_requests, spec_sha256
from .experiments import BATCH_STATE, batch_vs_separate_spec
from .guards import BudgetCaps, BudgetExceeded, BudgetLedger, Cancelled, RateLimiter
from .manifest import utc_now, verify_manifest
from .mech_dryrun import ordered_payload_json
from .providers import JevProvider, RetryPolicy
from .redact import (
    API_KEY_ENV,
    LIVE_OPT_IN_ENV,
    Redactor,
    base_url,
    get_api_key,
    live_calls_allowed,
)
from .runner import Runner, plan_run
from .transport import Transport, TransportResponse

EXECUTOR_VERSION = "reviewed-executor-1.0.0"
FREEZE_VERSION = "reviewed-freeze-1.0.0"

FRESH_SPEC_PATH = Path("docs/review-gate/audit/fresh_v3_test_spec.json")
MECH_SPEC_PATH = Path("docs/review-gate/audit/mech_dryrun_spec.json")
MECH_MAPPINGS_PATH = Path("docs/review-gate/audit/mech_dryrun_mappings.json")

STAGES = ("fresh_test", "mechanism", "batching")
STAGE_TITLES = {
    "fresh_test": "fresh-v3 TEST 120 (accepted, frozen)",
    "mechanism": "controlled mechanism 12x2x4 (96, frozen)",
    "batching": "batching comparison 12 blocks x 3 modes (108 calls)",
}

DEFAULT_MAX_REQUESTS = 400          # hard ceiling across ALL stages
DEFAULT_MAX_COST_USD = 0.10         # aggregate quoted dollar reservation ceiling
DEFAULT_MAX_WALL_SECONDS = 1200.0   # 20 minutes
CONSECUTIVE_FAILURE_LIMIT = 3       # repeated transport/schema failures stop the run
FATAL_HTTP_STATUSES = frozenset({401, 402, 403})  # auth / payment required / forbidden

BATCH_BLOCKS = 12
BATCH_SEED = 7700
BATCH_MODES = ("batched", "sequential", "concurrent")
MODE_CONCURRENCY = {"batched": 1, "sequential": 1, "concurrent": 4}
CONNECTION_POLICY = "new_http_client_per_workload_retained_then_closed"

FAILURE_STATUSES = frozenset({"http_error", "transport_error", "malformed_json", "contract_invalid", "timeout"})


class ExecutorError(RuntimeError):
    """A fail-closed condition: the executor refuses to continue."""


# ------------------------------------------------------------------ batching
def build_batching_spec(
    *,
    blocks: int = BATCH_BLOCKS,
    seed: int = BATCH_SEED,
    model: str = "jev-1.13.0",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the 12-block × 3-mode batching spec from the EXISTING content.

    Uses the fixed ``BATCH_STATE`` state text and the fixed K=4 question set of
    ``batch_vs_separate_spec`` verbatim — no new benchmark or content panel is
    introduced. Within each block the three (block, mode) workloads are
    executed in a randomized order recorded in the sidecar; the K separate
    questions keep their canonical order so answers are matched across modes.
    """
    template = batch_vs_separate_spec(repeats=1, seed=0)
    batch_questions = template["items"][0]["questions"]  # fixed K=4 content
    qids = list(batch_questions)
    rng = random.Random(f"reviewed-batching:{seed}")
    items: list[dict[str, Any]] = []
    sidecar: dict[str, Any] = {
        "design": "batching_12blocks_3modes",
        "version": "batching-reviewed-1.0.0",
        "blocks": blocks,
        "k_questions": len(qids),
        "question_ids": list(qids),
        "modes": list(BATCH_MODES),
        "mode_order_seed": seed,
        "connection_policy": CONNECTION_POLICY,
        "state_source": "experiments.BATCH_STATE (byte-identical, unchanged)",
        "question_source": "experiments.batch_vs_separate_spec batch questions (byte-identical, unchanged)",
        "n_calls_total": blocks * (1 + len(qids) + len(qids)),
        "blocks": [],
    }
    for block in range(blocks):
        mode_order = list(BATCH_MODES)
        rng.shuffle(mode_order)  # recorded randomized mode order within the block
        per_mode: dict[str, Any] = {}
        for mode in mode_order:
            if mode == "batched":
                item_id = f"batch-b{block:02d}"
                items.append({
                    "id": item_id,
                    "group": "batching_reviewed",
                    "cluster": f"batch-b{block:02d}",
                    "condition": f"role=batch;mode=batched;block={block}",
                    "state": BATCH_STATE,
                    "gold": {},
                    "questions": json.loads(json.dumps(batch_questions)),
                })
                per_mode[mode] = {"item_ids": [item_id], "question_order": list(qids)}
            else:
                prefix = "sep" if mode == "sequential" else "conc"
                item_ids = []
                for qid in qids:
                    item_id = f"{prefix}-b{block:02d}-{qid}"
                    items.append({
                        "id": item_id,
                        "group": "batching_reviewed",
                        "cluster": f"batch-b{block:02d}",
                        "condition": f"role=batch;mode={mode};block={block};question={qid}",
                        "state": BATCH_STATE,
                        "gold": {},
                        "questions": {qid: json.loads(json.dumps(batch_questions[qid]))},
                    })
                    item_ids.append(item_id)
                per_mode[mode] = {"item_ids": item_ids, "question_order": list(qids)}
        sidecar["blocks"].append({"block": block, "mode_order": mode_order, "modes": per_mode})
    spec = {
        "experiment": "batching-reviewed-12x3",
        "model": "jev-1.13.0" if model is None else model,
        "test_family": "mechanism-batching",
        "seeds": {"mode_order": seed},
        "shuffle": False,  # dispatch follows the recorded block/mode layout
        "claim_type": "exploratory",
        "generator_version": "batching-reviewed-1.0.0",
        "dataset": {
            "name": "batching-reviewed-12x3",
            "source_url": "generated locally",
            "license_note": "generated",
            "n_population": len(items),
            "sampling": {
                "mode": "matched_batching_workloads",
                "blocks": blocks,
                "k_questions": len(qids),
                "modes": list(BATCH_MODES),
                "n_workloads_per_mode": blocks,
                "note": ("12 (block, mode) workloads per arm; NOT 108 independent "
                         "workload measurements"),
            },
        },
        "items": items,
    }
    return spec, sidecar


# --------------------------------------------------------------------- freeze
def _file_sha256(path: Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def _wire_hashes(spec: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    """Ordered outbound payload hash + canonical request hash per logical id.

    The wire hash covers the ORDERED representation (key order preserved,
    never sorted); the canonical hash is the existing sorted-keys hash.
    """
    wire: dict[str, str] = {}
    canonical: dict[str, str] = {}
    for lr in logical_requests(spec):
        payload = lr.request.to_payload()
        wire[lr.logical_request_id] = sha256(
            ordered_payload_json(payload).encode("utf-8")).hexdigest()
        canonical[lr.logical_request_id] = lr.request.request_sha256()
    return wire, canonical


def _freeze_deterministic(doc: dict[str, Any]) -> str:
    trimmed = {k: v for k, v in doc.items() if k not in ("created_at", "deterministic_sha256")}
    return sha256(json.dumps(trimmed, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":")).encode("utf-8")).hexdigest()


def build_freeze(root: str | Path = "runs_reviewed") -> dict[str, Any]:
    """Freeze spec + ordered outbound payload hashes BEFORE any dispatch.

    Write-once: an existing freeze is reused only if it byte-matches the
    deterministic recomputation; any difference is a hard error (items may
    never be silently replaced).
    """
    root_path = Path(root)
    freeze_path = root_path / "freeze" / "frozen.json"
    stages: dict[str, Any] = {}
    for stage, path in (("fresh_test", FRESH_SPEC_PATH), ("mechanism", MECH_SPEC_PATH)):
        spec = load_experiment(path)
        wire, canonical = _wire_hashes(spec)
        stages[stage] = {
            "spec_path": str(path),
            "spec_file_sha256": _file_sha256(path),
            "n_requests": len(spec["items"]),
            "wire_sha256": wire,
            "canonical_sha256": canonical,
        }
    stages["mechanism"]["mappings_path"] = str(MECH_MAPPINGS_PATH)
    stages["mechanism"]["mappings_sha256"] = _file_sha256(MECH_MAPPINGS_PATH)
    bspec, sidecar = build_batching_spec()
    wire, canonical = _wire_hashes(bspec)
    stages["batching"] = {
        "spec_sha256": spec_sha256(bspec),
        "n_requests": len(bspec["items"]),
        "wire_sha256": wire,
        "canonical_sha256": canonical,
        "sidecar": sidecar,
        "mode_concurrency": {m: MODE_CONCURRENCY[m] for m in BATCH_MODES},
    }
    doc = {
        "freeze_version": FREEZE_VERSION,
        "created_at": utc_now(),
        "connection_policy": CONNECTION_POLICY,
        "budget_caps": {
            "max_requests": DEFAULT_MAX_REQUESTS,
            "max_cost_usd": DEFAULT_MAX_COST_USD,
            "max_wall_seconds": DEFAULT_MAX_WALL_SECONDS,
            "retries": "none",
        },
        "stages": stages,
    }
    doc["deterministic_sha256"] = _freeze_deterministic(doc)
    if freeze_path.exists():
        existing = load_freeze(root_path)  # verifies the document's own integrity first
        if existing.get("deterministic_sha256") == doc["deterministic_sha256"]:
            return existing
        raise ExecutorError(
            "an existing freeze document differs from the current specs; refusing to "
            "overwrite or relabel frozen items (a new reviewed run would be required)"
        )
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    freeze_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return doc


def load_freeze(root: str | Path = "runs_reviewed") -> dict[str, Any]:
    path = Path(root) / "freeze" / "frozen.json"
    if not path.exists():
        raise ExecutorError(f"no freeze document at {path}; run the dry-run first")
    doc = json.loads(path.read_text(encoding="utf-8"))
    if _freeze_deterministic(doc) != doc.get("deterministic_sha256"):
        raise ExecutorError(f"freeze document {path} failed its integrity check")
    return doc


def stage_run_id(freeze: dict[str, Any], stage: str) -> str:
    """Deterministic run id resolved FROM the freeze, never lexicographic glob."""
    return f"rev-{stage}-{freeze['deterministic_sha256'][:12]}"


def verify_frozen_sources(freeze: dict[str, Any]) -> None:
    """Verify every frozen SOURCE spec + sidecar mapping hash against the freeze.

    Used before execution AND before offline reporting: a mutable source file
    whose bytes changed since the freeze is refused outright (REVIEW-3 #1).
    """
    for stage in ("fresh_test", "mechanism"):
        entry = freeze["stages"][stage]
        path = Path(entry["spec_path"])
        if not path.exists():
            raise ExecutorError(
                f"frozen source spec for stage {stage!r} is missing: {path}; "
                "the frozen input can no longer be verified — refusing"
            )
        if _file_sha256(path) != entry["spec_file_sha256"]:
            raise ExecutorError(
                f"frozen source spec for stage {stage!r} changed since the freeze "
                f"({path}); the frozen input no longer matches its recorded hash — refusing"
            )
    mech = freeze["stages"]["mechanism"]
    mappings_path = Path(mech["mappings_path"])
    if not mappings_path.exists():
        raise ExecutorError(
            f"mechanism sidecar mappings are missing: {mappings_path}; refusing"
        )
    if _file_sha256(mappings_path) != mech["mappings_sha256"]:
        raise ExecutorError(
            f"mechanism sidecar mappings changed since the freeze ({mappings_path}); "
            "the sidecar is not an unchecked mutable path — refusing"
        )


# ------------------------------------------------------------------ executor
class _NonRaisingTransport:
    """Wrap any transport so a raising post becomes a recorded transport error.

    Guarantees the provider produces a real AttemptRecord for every dispatch,
    so attempts persist even when a fake/real transport explodes.
    """

    name = "non_raising"

    def __init__(self, inner: Transport) -> None:
        self.inner = inner

    def post(self, path: str, payload: dict[str, Any]) -> TransportResponse:
        try:
            return self.inner.post(path, payload)
        except Exception as exc:  # noqa: BLE001 - recorded, never raised past here
            return TransportResponse(
                status_code=0, headers={}, body=b"", total_ms=0.0,
                error=f"transport_error:{type(exc).__name__}",
            )

    def close(self) -> None:
        self.inner.close()


class ReviewedExecutor:
    """Plan → freeze-verify → dispatch the three approved stages (or dry-run)."""

    def __init__(
        self,
        root: str | Path = "runs_reviewed",
        *,
        transport_factory: Callable[[], Transport] | None = None,
        clock: Callable[[], float] = None,
        sleeper: Callable[[float], None] = None,
        max_requests: int = DEFAULT_MAX_REQUESTS,
        max_cost_usd: float = DEFAULT_MAX_COST_USD,
        max_wall_seconds: float = DEFAULT_MAX_WALL_SECONDS,
        consecutive_failure_limit: int = CONSECUTIVE_FAILURE_LIMIT,
    ) -> None:
        import time as _time

        self.root = Path(root)
        self.transport_factory = transport_factory
        self.clock = clock or _time.monotonic
        self.sleeper = sleeper or _time.sleep
        self.max_requests = max_requests
        self.max_cost_usd = max_cost_usd
        self.max_wall_seconds = max_wall_seconds
        self.consecutive_failure_limit = consecutive_failure_limit
        self.cancellation = _make_cancellation()
        self.stop_reason: str | None = None
        self.budget: BudgetLedger | None = None
        self.deadline: float | None = None
        self._consecutive_failures = 0
        self._state_lock = threading.Lock()  # guards stop_reason / consecutive-failure state
        self._frozen_wire: dict[str, str] = {}
        self._n_wire_verified = 0

    # ------------------------------------------------------------- top level
    def run(self, *, dry_run: bool = True) -> dict[str, Any]:
        if dry_run:
            freeze = build_freeze(self.root)
        else:
            freeze = load_freeze(self.root)
        state: dict[str, Any] = {
            "executor_version": EXECUTOR_VERSION,
            "dry_run": bool(dry_run),
            "freeze_sha256": freeze["deterministic_sha256"],
            "connection_policy": CONNECTION_POLICY,
            "budget_caps": {
                "max_requests": self.max_requests,
                "max_cost_usd": self.max_cost_usd,
                "max_wall_seconds": self.max_wall_seconds,
                "retries": "none",
            },
            "stages": {},
            "global_stop_reason": None,
            "recorded_at": utc_now(),
        }
        stage_run_ids: dict[str, str] = {}
        verified_specs: dict[str, dict[str, Any]] = {}
        for stage in STAGES:
            run_id, verification, spec = self._plan_and_verify(stage, freeze)
            stage_run_ids[stage] = run_id
            verified_specs[stage] = spec
            state["stages"][stage] = {
                "run_id": run_id,
                "title": STAGE_TITLES[stage],
                "freeze_verification": "pass",
                "manifest_sha256": verification["manifest_sha256"],
                "plan": verification["plan"],
            }
        # Frozen wire hashes are cached for dispatch-time verification in BOTH
        # modes (direct workload execution included).
        frozen_wire: dict[str, str] = {}
        for stage in STAGES:
            frozen_wire.update(freeze["stages"][stage]["wire_sha256"])
        self._frozen_wire = frozen_wire
        self._n_wire_verified = 0
        try:
            # The frozen SOURCE spec files + mechanism sidecar mappings are
            # verified against the freeze BEFORE anything else happens
            # (execution or report) — a mutable source path is refused.
            verify_frozen_sources(freeze)
            if dry_run:
                state["note"] = "dry run: planned + hash-verified; nothing dispatched, no provider constructed"
                self._write_state(state)
                return state
            # Fail-closed resume gate BEFORE any budget restoration: uncertain
            # (possibly billed) attempts refuse the run outright.
            for stage in STAGES:
                store = self._store(stage_run_ids[stage])
                uncertain = store.uncertain_ids()
                if uncertain:
                    raise ExecutorError(
                        f"fail-closed resume refusal: {len(uncertain)} uncertain (timeout) "
                        f"attempt(s) in stage {stage!r} ({sorted(uncertain)[:3]}...); the provider "
                        "may have billed unobserved work. Per parent REVIEW-2 this executor does "
                        "not resume past ambiguous timeouts: start a NEW reviewed run instead. "
                        "Budgets were not reset."
                    )
            budget = BudgetLedger(BudgetCaps(
                max_requests=self.max_requests,
                max_cost_usd=self.max_cost_usd,
            ))
            restored: dict[str, Any] = {
                "stages": {}, "attempts": 0, "reported": 0, "unknown_kept_reserved": 0,
                "refused": None,
            }
            # attached to the state BEFORE restoring so a refusal still records
            # what was being restored and why it was refused
            state["budget_restored_from_attempts"] = restored
            self._restore_budget(budget, freeze, restored)
            self.budget = budget
            self.deadline = self.clock() + self.max_wall_seconds
            for stage in STAGES:
                if self.stop_reason is not None or self.cancellation.cancelled:
                    if self.stop_reason is None:
                        self._stop("cancelled_by_cancellation")
                    state["stages"][stage]["skipped_reason"] = self.stop_reason
                    state["stages"][stage]["budget_snapshot"] = budget.snapshot()
                    continue
                wire_before = self._n_wire_verified
                summary = self._run_stage(
                    stage, stage_run_ids[stage], freeze, budget, verified_specs[stage])
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

    # ----------------------------------------------------------- plan/verify
    def _plan_and_verify(self, stage: str, freeze: dict[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any]]:
        """Plan (or reuse the immutable plan), reload stored items, and verify
        their payload hashes against the freeze. Returns the verified
        stored/reloaded spec: ONLY these requests may ever be dispatched."""
        spec = self._stage_spec(stage)
        freeze_stage = freeze["stages"][stage]
        run_id = f"rev-{stage}-{freeze['deterministic_sha256'][:12]}"
        directory = self.root / run_id
        if (directory / "manifest.json").exists():
            # already planned (dry run / earlier attempt); manifests are immutable
            if not (directory / "items.jsonl").exists():
                raise ExecutorError(
                    f"stage {stage}: incomplete plan at {run_id} (manifest without items); "
                    "start a new reviewed run"
                )
        else:
            plan_run(spec, provider="jev", root=str(self.root), run_id=run_id)
        manifest = verify_manifest(self.root / run_id)
        # Real plan → reload path: rebuild requests from the STORED items.
        items = [json.loads(line) for line in
                 (self.root / run_id / "items.jsonl").read_text(encoding="utf-8").splitlines()
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
                f"stage {stage}: stored items.jsonl does not match its manifest hash; refusing "
                "to run against mutated items"
            )
        wire, canonical = _wire_hashes(reloaded_spec)
        if wire != freeze_stage["wire_sha256"] or canonical != freeze_stage["canonical_sha256"]:
            raise ExecutorError(
                f"stage {stage}: stored/reloaded payload hashes do not match the frozen "
                "ordered payload hashes; items changed since the freeze — aborting"
            )
        plan_doc = json.loads((self.root / run_id / "plan.json").read_text(encoding="utf-8"))
        # Only these verified, stored/reloaded requests may ever be dispatched.
        return run_id, {"manifest_sha256": manifest.sha256(), "plan": plan_doc}, reloaded_spec

    def _stage_spec(self, stage: str) -> dict[str, Any]:
        if stage == "fresh_test":
            return load_experiment(FRESH_SPEC_PATH)
        if stage == "mechanism":
            return load_experiment(MECH_SPEC_PATH)
        if stage == "batching":
            spec, _sidecar = build_batching_spec()
            return spec
        raise ExecutorError(f"unknown stage {stage!r}")

    def _store(self, run_id: str):
        from .ledger import RunStore

        return RunStore(self.root, run_id, redactor=Redactor.from_environment())

    # ---------------------------------------------------------------- budget
    def _restore_budget(self, budget: BudgetLedger, freeze: dict[str, Any],
                        restored: dict[str, Any] | None = None) -> dict[str, Any]:  # noqa: C901
        """Restore aggregate reservations from recorded attempts (resume).

        Scoped STRICTLY to the frozen run ids (resolved from the freeze hash,
        never lexicographic glob). Server-reported usage is committed; attempts
        without usage keep their conservative estimate reserved (unknown usage
        is never counted as free). Historical attempts are never silently
        omitted: if restoring an attempt would exceed the caps, the run is
        REFUSED fail-closed with the reason recorded — budgets are not reset
        and excluded attempts are never treated as free (REVIEW-3 #2).
        """
        restored = restored if restored is not None else {
            "stages": {}, "attempts": 0, "reported": 0, "unknown_kept_reserved": 0,
            "refused": None,
        }
        for stage in STAGES:
            run_id = stage_run_id(freeze, stage)
            directory = self.root / run_id
            if not (directory / "attempts.jsonl").exists():
                continue
            store = self._store(run_id)
            restored["stages"][stage] = run_id
            for attempt in store.attempts():
                try:
                    reservation = budget.reserve(max(1, int(attempt.get("estimated_input_tokens") or 1)))
                except BudgetExceeded as exc:
                    restored["refused"] = {
                        "run_id": run_id, "attempt_id": attempt.get("attempt_id"),
                        "limit": exc.limit,
                    }
                    raise ExecutorError(
                        f"fail-closed resume refusal: historical attempt "
                        f"{attempt.get('attempt_id')!r} in {run_id} exceeds the budget caps "
                        f"while restoring reservations ({exc}). Historical attempts are never "
                        "silently omitted or counted as free; budgets were not reset. "
                        "Start a NEW reviewed run instead."
                    ) from exc
                budget.reconcile(reservation, attempt.get("usage_input_tokens"))
                restored["attempts"] += 1
                if attempt.get("usage_input_tokens") is None:
                    restored["unknown_kept_reserved"] += 1
                else:
                    restored["reported"] += int(attempt["usage_input_tokens"])
        return restored

    # ------------------------------------------------------------ live stage
    def _run_stage(self, stage: str, run_id: str, freeze: dict[str, Any], budget: BudgetLedger,
                   verified_spec: dict[str, Any]) -> dict[str, Any]:
        store = self._store(run_id)
        wall_start = self.clock()
        if stage == "batching":
            summary = self._run_batching_stage(run_id, freeze, budget, verified_spec)
        else:
            summary = self._run_sequential_stage(stage, run_id, budget, verified_spec)
        if self.stop_reason is not None:
            summary.setdefault("stop_reason", self.stop_reason)
        summary["wall_start_monotonic"] = wall_start
        summary["wall_end_monotonic"] = self.clock()
        summary["stage_wall_seconds"] = round(summary["wall_end_monotonic"] - wall_start, 6)
        summary["budget_snapshot"] = budget.snapshot()
        return summary

    def _new_provider(self) -> JevProvider:
        if self.transport_factory is None:
            self.transport_factory = self._default_transport_factory()
        transport = _NonRaisingTransport(self.transport_factory())
        # The environment-derived credential redactor is propagated so that
        # provider-side exceptions can never leak the env key literal.
        return JevProvider(transport, retry_policy=RetryPolicy.none(),
                           sleeper=self.sleeper, redactor=Redactor.from_environment())

    def _default_transport_factory(self) -> Callable[[], Transport]:
        if not live_calls_allowed():
            raise ExecutorError(f"live calls require environment {LIVE_OPT_IN_ENV}=1")
        api_key = get_api_key()
        if not api_key:
            raise ExecutorError(f"no API key in process environment ({API_KEY_ENV})")
        url = base_url()

        def factory() -> Transport:
            from .transport import HttpxTransport

            return HttpxTransport(url, api_key)  # a NEW client per call; caller closes it

        return factory

    def _hook(self, budget: BudgetLedger):
        from .runner import HookAdapter

        return HookAdapter(budget=budget, rate=RateLimiter(), cancellation=self.cancellation)

    def _stop(self, reason: str) -> None:
        with self._state_lock:  # shared stop state is concurrency-safe
            if self.stop_reason is None:
                self.stop_reason = reason
        self.cancellation.cancel()

    def _observe_outcome(self, outcome) -> None:
        stop: str | None = None
        with self._state_lock:  # shared consecutive-failure state is concurrency-safe
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
        """Verify the EXACT payload about to leave the process against the freeze.

        The verification performed at planning time is not enough: the wire
        hash of the actual dispatched payload is re-checked immediately before
        every dispatch (REVIEW-3 #1: the approved model/state/questions/order
        must be what leaves the process).
        """
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
        for attempt in outcome.attempts:
            store.append_attempt(attempt.to_dict())
        if outcome.terminal and outcome.status not in {"cancelled", "timeout"}:
            store.append_result(Runner._result_record(None, outcome, lr))
        self._observe_outcome(outcome)
        return {"status": 1 if outcome.status == "ok" else 0, "dispatched": 1}

    def _run_sequential_stage(self, stage: str, run_id: str, budget: BudgetLedger,
                              verified_spec: dict[str, Any]) -> dict[str, Any]:
        # ONLY the verified stored/reloaded spec from _plan_and_verify is used;
        # the mutable source spec is never re-read for dispatch (REVIEW-3 #1).
        requests = logical_requests(verified_spec)
        store = self._store(run_id)
        completed = store.completed_ids()
        provider = self._new_provider()
        hook = self._hook(budget)
        tally = {"dispatched": 0, "ok": 0, "error": 0, "skipped_resume": 0}
        try:
            for lr in requests:
                if self.cancellation.cancelled or self.stop_reason is not None:
                    break
                if lr.logical_request_id in completed:
                    tally["skipped_resume"] += 1
                    continue
                counts = self._dispatch(provider, lr, hook, store)
                if counts is None:
                    continue
                tally["dispatched"] += 1
                if counts["status"]:
                    tally["ok"] += 1
        finally:
            provider.close()
        summary = {
            "run_id": run_id,
            "connection_policy": "one_new_client_for_stage_retained_then_closed",
            **tally,
            "stop_reason": self.stop_reason,
        }
        return summary

    def _run_batching_stage(self, run_id: str, freeze: dict[str, Any], budget: BudgetLedger,
                            verified_spec: dict[str, Any]) -> dict[str, Any]:
        sidecar = freeze["stages"]["batching"]["sidecar"]  # immutable: inside the verified freeze
        by_id = {lr.logical_request_id: lr for lr in logical_requests(verified_spec)}
        store = self._store(run_id)
        workloads_path = self.root / run_id / "workloads.jsonl"
        recorded: dict[str, dict[str, Any]] = {}
        if workloads_path.exists():
            for line in workloads_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    record = json.loads(line)
                    recorded[record["workload_id"]] = record
        n_executed = n_resumed_partial = 0
        for block_entry in sidecar["blocks"]:
            if self.stop_reason is not None:
                break
            block = block_entry["block"]
            for mode in block_entry["mode_order"]:
                if self.stop_reason is not None:
                    break
                workload_id = f"b{block:02d}-{mode}"
                prior = recorded.get(workload_id)
                if prior is not None and prior.get("complete"):
                    continue  # fully recorded: never re-executed
                # A recorded-but-INCOMPLETE workload (interrupted partial batch)
                # is resumed over its pending requests only; its wall time is
                # labelled partial and never presented as full-work time.
                item_ids = block_entry["modes"][mode]["item_ids"]
                lrs = [_logical(by_id, item_id) for item_id in item_ids]
                summary = self._execute_workload(
                    block, mode, workload_id, lrs, store, budget,
                    workloads_path=workloads_path,
                    resumed_from_record=prior is not None,
                )
                if prior is not None or not summary["wall_seconds_is_full_workload"]:
                    n_resumed_partial += 1
                n_executed += 1
        return {
            "run_id": run_id,
            "connection_policy": CONNECTION_POLICY,
            "n_workloads_recorded": len(recorded) + n_executed,
            "n_workloads_executed_now": n_executed,
            "n_workloads_resumed_partial": n_resumed_partial,
            "stop_reason": self.stop_reason,
        }

    # ---------------------------------------------------------------- workload
    def _execute_workload(self, block, mode, workload_id, lrs, store, budget,
                          *, workloads_path: Path | None = None,
                          resumed_from_record: bool = False) -> dict[str, Any]:
        """One (block, mode) workload: new client, retained, closed; true wall time.

        The client is closed in ``finally`` even on exceptions, and a partial
        workload record (with the stop reason) is persisted when the workload
        raises, so interrupted batches are never silently lost (REVIEW-3 #4).
        Wall time covers FULL workloads only when nothing was skipped as
        already-complete; resumed leftovers are labelled partial.
        """
        provider = self._new_provider()  # NEW HTTP client per workload
        concurrency = MODE_CONCURRENCY[mode]
        hook = self._hook(budget)
        wall_start = self.clock()
        started_at = utc_now()
        tally = {"dispatched": 0, "ok": 0, "errors": 0}
        lock = threading.Lock()
        completed_before = store.completed_ids()
        pending = [lr for lr in lrs if lr.logical_request_id not in completed_before]
        n_skipped_as_complete = len(lrs) - len(pending)

        def worker(lr) -> None:
            if self.cancellation.cancelled or self.stop_reason is not None:
                return
            counts = self._dispatch(provider, lr, hook, store)
            if counts is None:
                return
            with lock:
                tally["dispatched"] += counts["dispatched"]
                if counts["status"]:
                    tally["ok"] += 1
                else:
                    tally["errors"] += 1

        error: str | None = None
        exc_holder: list[BaseException] = []
        try:
            if concurrency == 1:
                for lr in pending:
                    if self.stop_reason is not None:
                        break
                    worker(lr)
            else:
                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    list(pool.map(worker, pending))
        except Exception as exc:  # recorded fail-closed; client still closed below
            error = self._redact_text(f"{type(exc).__name__}: {exc}")
            exc_holder.append(exc)
        finally:
            wall_end = self.clock()
            provider.close()  # closed even on exception; next workload gets a new client
        lr_ids = {lr.logical_request_id for lr in lrs}
        usage_in = usage_out = unknown = 0
        for attempt in store.attempts():
            if attempt.get("logical_request_id") in lr_ids:
                if attempt.get("usage_input_tokens") is not None:
                    usage_in += int(attempt["usage_input_tokens"])
                else:
                    unknown += 1
                if attempt.get("usage_output_tokens") is not None:
                    usage_out += int(attempt["usage_output_tokens"])
        complete = all(lr.logical_request_id in store.completed_ids() for lr in lrs)
        # Wall time counts as FULL WORKLOAD time only when every request was
        # actually dispatched inside this window and the workload completed.
        full_wall = (n_skipped_as_complete == 0) and complete and error is None
        summary = {
            "workload_id": workload_id,
            "block": block,
            "mode": mode,
            "item_ids": [lr.logical_request_id for lr in lrs],
            "n_requests": len(lrs),
            "n_dispatched_now": tally["dispatched"],
            "n_ok": tally["ok"],
            "n_errors": tally["errors"],
            "complete": complete,
            "resumed_partial": bool(n_skipped_as_complete or resumed_from_record),
            "wall_seconds_is_full_workload": full_wall,
            "wall_start_monotonic": wall_start,
            "wall_end_monotonic": wall_end,
            "wall_seconds": round(wall_end - wall_start, 6),
            "started_at": started_at,
            "finished_at": utc_now(),
            "connection_policy": CONNECTION_POLICY,
            "client_created": True,
            "client_closed_after_workload": True,
            "cold_client_workload": True,
            "usage_input_tokens_sum": usage_in,
            "usage_output_tokens_sum": usage_out,
            "usage_unknown_attempts": unknown,
            "stop_reason": self.stop_reason or (
                "cancelled_by_cancellation" if self.cancellation.cancelled else None),
            "workload_error": error,
            "note": ("cold-client workload comparison: true whole-workload monotonic wall time; "
                     "never the sum of parallel durations or 4x median; wall_seconds counts as "
                     "full-workload time only when nothing was skipped and the workload completed"),
        }
        if workloads_path is not None:
            with workloads_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
        if error is not None:
            # fail-closed: the partial workload record is persisted above, then
            # the run refuses; the client was already closed in the finally.
            self._stop(self.stop_reason or "workload_execution_error")
            raise ExecutorError(
                f"workload {workload_id!r} failed ({error}); the partial workload record "
                "was persisted with the stop reason and the client was closed — refusing "
                "to continue"
            ) from exc_holder[0]
        return summary

    def _redact_text(self, text: str) -> str:
        return Redactor.from_environment().text(text)

    # ------------------------------------------------------------------ misc
    def _write_state(self, state: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / "executor_state.json"
        path.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# import-time helpers kept tiny
def _make_cancellation():
    from .guards import Cancellation

    return Cancellation()


def _logical(by_id: dict[str, Any], item_id: str):
    """Resolve a sidecar item id to its LogicalRequest (ids are item ids)."""
    matches = [lid for lid in by_id if lid.endswith(f":{item_id}")]
    if len(matches) != 1:
        raise ExecutorError(f"sidecar item {item_id!r} does not match exactly one stored request")
    return by_id[matches[0]]
