"""Versioned, resumable, evidence-preserving run storage for baseline re-runs.

Motivation: the matched cheap-baseline sweeps v2/v3 lost evidence (v2 stored
TRUNCATED content/reasoning excerpts and no finish reasons; v3 stored no raw
output at all), so the fixed recovery parser cannot be applied to them offline
and their recovered scores can never be honestly "repaired" from artifacts
alone.  Future runs must therefore:

* live in a FRESH versioned run directory — original run evidence is never
  overwritten, never deleted, never re-scored in place (``initialize``
  refuses a version directory whose recorded plan fingerprint differs);
* store COMPLETE private raw responses (full content, full reasoning, the
  provider's finish_reason, http status) under ``private_raw/`` — private,
  outside any public bundle;
* keep append-only ``results.jsonl`` whose terminal logical ids are resumed,
  never re-dispatched and never double-counted (``append_result`` refuses a
  duplicate terminal id);
* maintain ``active_results.json`` — non-destructive active-result POINTERS
  per (model, dataset) cell, so consumers always know which version's results
  are active without any file being replaced;
* account usage (tokens and reported cost) in a ledger whose summary counts
  unknown usage as UNKNOWN, never as free;
* dispatch with bounded concurrency (total + per-key semaphores), a hard
  spend cap and cooperative stop.

Publication discipline (``publication_manifest`` / ``public_summary``):
public aggregates carry counts, accuracies, recovery-stage counts and usage
totals ONLY — never gold keys and never licensed dataset text (questions,
prompts, raw model output).
"""

from __future__ import annotations

import fcntl
import json
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Iterable

from .manifest import utc_now

REVISION_RUN_VERSION = "revision-run-1.0.0"
BUDGET_STATE_VERSION = "budget-state-1.0.0"
BUDGET_STATE_NAME = "budget_ledger.json"
BUDGET_EVENTS_NAME = "budget_events.jsonl"
RECOVERY_NAME = "transport_recovery.jsonl"
RECOVERY_POINTERS_NAME = "transport_recovery_pointers.json"
PRIVATE_DIR_NAME = "private_raw"
RESULTS_NAME = "results.jsonl"
RAW_SUFFIX = ".jsonl"
USAGE_LEDGER_NAME = "usage_ledger.jsonl"
ACTIVE_POINTERS_NAME = "active_results.json"
RUN_META_NAME = "run_meta.json"
USAGE_SUMMARY_NAME = "usage_summary.json"
PUBLICATION_MANIFEST_NAME = "publication_manifest.json"

# fields that must never appear in public aggregates: gold keys and licensed
# dataset/raw text
_PRIVATE_RESULT_FIELDS = ("gold", "state", "question", "problem", "prompt",
                          "content", "reasoning", "text", "solution",
                          "pred_raw", "raw")


class RevisionRunError(RuntimeError):
    """Fail-closed condition for revision-run storage."""


# transport-level outcomes that are RETRYABLE and must never become terminal
# capability failures in a repaired cell: rate limits, gateway errors,
# timeouts.  Rows landing here are PROVISIONAL pending transport recovery.
RETRYABLE_STATUSES = frozenset({0, 429, 500, 502, 503, 504})
RETRY_AFTER_CAP_SECONDS = 60.0


def is_retryable_status(status_code: int | None) -> bool:
    """True when an http outcome is a transport condition (rate limit /
    gateway / timeout), not a model capability answer."""
    return status_code in RETRYABLE_STATUSES


def retry_after_seconds(headers: dict[str, Any] | None, attempt: int) -> float:
    """Backoff for the next transport attempt: the provider's Retry-After
    when present, else exponential in the attempt number; capped so a
    rate-limited provider (e.g. Gemma under 429) can never stall the run."""
    raw = (headers or {}).get("retry-after") or (headers or {}).get("Retry-After")
    if raw is not None:
        try:
            return min(max(float(raw), 1.0), RETRY_AFTER_CAP_SECONDS)
        except (TypeError, ValueError):
            pass
    return min(2.0 * attempt, RETRY_AFTER_CAP_SECONDS)


class BudgetExceeded(RevisionRunError):
    """A pre-call reservation would exceed the hard spend cap; nothing ran."""


class PersistentBudget:
    """Atomic PRE-CALL spend reservations against a persistent USD cap.

    The budget is a hard bound on WORST-CASE exposure, never an
    after-the-fact tally: a call may only start after ``reserve`` grants a
    reservation for its worst-case cost (including every bounded retry), and
    the cap check is ``settled + held + active_reservations + new <= cap``.

    Fail-closed rules:

    * a call that settles with a KNOWN cost converts its reservation to that
      cost (the difference is released only then);
    * a call whose billed cost is UNKNOWN (missing/unusable usage, worker
      exception where the provider may have billed) keeps its FULL
      reservation held FOREVER — held reservations are never released as
      free;
    * only a reservation whose request demonstrably never left the process
      (``release_unsent``) is given back without charge.

    State lives in one JSON file updated under an exclusive file lock, so a
    resumed process (or a second process on the same ledger) inherits past
    spend, held reservations and still-active reservations — resume can
    never reset the budget.
    """

    def __init__(self, path: Path | str | None, cap_usd: float) -> None:
        if cap_usd <= 0:
            raise RevisionRunError(f"budget cap must be positive, got {cap_usd!r}")
        self.path = Path(path) if path is not None else None
        self.cap_usd = float(cap_usd)
        self._lock = threading.Lock()
        self._mem_state: dict[str, Any] | None = (
            self._default_state() if path is None else None)
        if self.path is not None:
            if self.path.exists():
                recorded = json.loads(
                    self.path.read_text(encoding="utf-8")).get("cap_usd")
                if (recorded is not None
                        and abs(float(recorded) - self.cap_usd) > 1e-9):
                    raise RevisionRunError(
                        f"{self.path}: budget cap mismatch; ledger records "
                        f"${float(recorded):.2f} but this run asked for "
                        f"${self.cap_usd:.2f} — the cap is bound to the ledger")
            else:
                # create the ledger NOW so the cap is bound to it from
                # construction, not from the first reservation
                self._mutate(lambda state: state)

    # -------------------------------------------- state IO (atomic txns)
    def _default_state(self) -> dict[str, Any]:
        return {"budget_state_version": BUDGET_STATE_VERSION,
                "cap_usd": self.cap_usd, "settled_usd": 0.0, "held_usd": 0.0,
                "active": {}, "n_reservations": 0, "n_denials": 0}

    def _mutate(self, fn: Callable[[dict[str, Any]], Any]) -> Any:
        """One atomic read-mutate-write transaction on the ledger.

        Thread lock + exclusive flock serialise every mutation (including
        from other processes), so lost updates cannot silently free budget.
        """
        with self._lock:
            if self.path is None:
                # in-memory ledger: state persists on the instance
                return fn(self._mem_state)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            lock_path = self.path.with_name(self.path.name + ".lock")
            with lock_path.open("a+") as lock_fh:
                fcntl.flock(lock_fh, fcntl.LOCK_EX)
                state = (json.loads(self.path.read_text(encoding="utf-8"))
                         if self.path.exists() else self._default_state())
                result = fn(state)
                tmp = self.path.with_name(self.path.name + ".tmp")
                tmp.write_text(json.dumps(state, indent=1) + "\n",
                               encoding="utf-8")
                os.replace(tmp, self.path)
                fcntl.flock(lock_fh, fcntl.LOCK_UN)
            return result

    def _event(self, kind: str, **fields: Any) -> None:
        if self.path is None:
            return
        entry = {"at": utc_now(), "event": kind, **fields}
        events = self.path.with_name(BUDGET_EVENTS_NAME)
        with events.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            fh.flush()

    @staticmethod
    def _exposure(state: dict[str, Any]) -> float:
        active = sum(float(a.get("amount", 0.0))
                     for a in state.get("active", {}).values())
        return (float(state.get("settled_usd", 0.0))
                + float(state.get("held_usd", 0.0)) + active)

    def reserve(self, key: str, max_cost_usd: float, *, label: str = "") -> str:
        """Atomically reserve worst-case cost for ONE upcoming call.

        Raises ``BudgetExceeded`` (and records a denial) when the cap cannot
        cover ``settled + held + active + max_cost``.  No call may start
        without a granted reservation — the cap is enforced BEFORE the call,
        never from after-the-fact reported spend.
        """
        if max_cost_usd < 0:
            raise RevisionRunError(f"negative reservation {max_cost_usd!r}")

        def _apply(state: dict[str, Any]) -> str | None:
            cap = float(state.get("cap_usd", self.cap_usd))
            projected = self._exposure(state) + float(max_cost_usd)
            if projected > cap + 1e-12:
                state["n_denials"] = int(state.get("n_denials", 0)) + 1
                self._event("denied", key=key, max_cost_usd=max_cost_usd,
                            projected_usd=round(projected, 6))
                return None
            rid = f"r-{uuid.uuid4().hex[:12]}"
            state.setdefault("active", {})[rid] = {
                "amount": float(max_cost_usd), "key": key, "label": label,
                "reserved_at": utc_now()}
            state["n_reservations"] = int(state.get("n_reservations", 0)) + 1
            self._event("reserved", rid=rid, key=key, max_cost_usd=max_cost_usd,
                        exposure_usd=round(self._exposure(state), 6))
            return rid

        rid = self._mutate(_apply)
        if rid is None:
            raise BudgetExceeded(
                f"budget cap: reserving ${max_cost_usd:.4f} for {key!r} would "
                f"exceed worst-case exposure; nothing is dispatched")
        return rid

    def grant_cap(self, new_cap_usd: float, *, from_cap_usd: float,
                  granted_by: str, note: str = "") -> dict[str, Any]:
        """Explicit AUDITED user grant migrating the ledger cap.

        The migration PRESERVES settled + held + active reservations verbatim
        (spend is never reset and no new ledger/version is opened to bypass
        previous spend).  It is refused unless ``from_cap_usd`` matches the
        ledger exactly (so unauthorized changes still hit the cap-mismatch
        safety), unless ``granted_by`` names the grant, and never below
        current commitments.  Both caps are recorded in the ledger's
        ``cap_history`` and in an audit event.
        """
        if not granted_by:
            raise RevisionRunError(
                "cap migration requires an explicit granted_by audit label")
        if new_cap_usd <= 0:
            raise RevisionRunError(f"budget cap must be positive, got {new_cap_usd!r}")

        def _apply(state: dict[str, Any]) -> dict[str, Any]:
            current = float(state.get("cap_usd", self.cap_usd))
            if abs(current - float(from_cap_usd)) > 1e-9:
                # unauthorized / mismatched migration: nothing changes
                self._event("cap_grant_refused", recorded_cap_usd=current,
                            claimed_from_cap_usd=float(from_cap_usd),
                            attempted_new_cap_usd=float(new_cap_usd),
                            granted_by=granted_by)
                raise RevisionRunError(
                    f"cap grant refused: ledger records ${current:.2f} but the "
                    f"grant claims ${float(from_cap_usd):.2f}; the migration "
                    f"must name the exact previous cap")
            exposure = self._exposure(state)
            if float(new_cap_usd) + 1e-9 < exposure:
                self._event("cap_grant_refused", recorded_cap_usd=current,
                            attempted_new_cap_usd=float(new_cap_usd),
                            exposure_usd=round(exposure, 6), granted_by=granted_by)
                raise RevisionRunError(
                    f"cap grant refused: ${float(new_cap_usd):.2f} is below "
                    f"current commitments ${exposure:.4f} (settled + held + "
                    f"active); the cap can never be reduced below spend")
            state["cap_usd"] = float(new_cap_usd)
            state.setdefault("cap_history", []).append({
                "at": utc_now(), "old_cap_usd": current,
                "new_cap_usd": float(new_cap_usd), "granted_by": granted_by,
                "note": note, "exposure_preserved_usd": round(exposure, 6)})
            self._event("cap_grant", old_cap_usd=current,
                        new_cap_usd=float(new_cap_usd), granted_by=granted_by,
                        exposure_usd=round(exposure, 6), note=note)
            return self._state_summary(state)

        snapshot = self._mutate(_apply)
        self.cap_usd = float(new_cap_usd)
        return snapshot

    def settle(self, reservation_id: str, actual_cost_usd: float | None) -> dict[str, Any]:
        """Settle a reservation after the call finished.

        Known cost: the reservation converts to the actual charge (excess is
        released ONLY here).  ``None``/non-numeric: the full reservation is
        moved to ``held_usd`` permanently — unknown billed cost is never
        released as free.
        """
        def _apply(state: dict[str, Any]) -> dict[str, Any]:
            entry = state.get("active", {}).pop(reservation_id, None)
            if entry is None:
                raise RevisionRunError(f"unknown reservation {reservation_id!r}")
            amount = float(entry.get("amount", 0.0))
            known = (isinstance(actual_cost_usd, (int, float))
                     and not isinstance(actual_cost_usd, bool))
            if known:
                state["settled_usd"] = (float(state.get("settled_usd", 0.0))
                                        + float(actual_cost_usd))
            else:
                # FAIL CLOSED: keep the whole worst-case reservation held
                state["held_usd"] = float(state.get("held_usd", 0.0)) + amount
            self._event("settled", rid=reservation_id, key=entry.get("key"),
                        actual_cost_usd=(float(actual_cost_usd) if known else None),
                        held_usd=(0.0 if known else amount),
                        exposure_usd=round(self._exposure(state), 6))
            return self._state_summary(state)

        return self._mutate(_apply)

    def release_unsent(self, reservation_id: str) -> None:
        """Give a reservation back ONLY when the request demonstrably never
        left the process.  Any outcome that may have billed must use
        ``settle``/``settle(None)`` instead (fail closed)."""

        def _apply(state: dict[str, Any]) -> None:
            entry = state.get("active", {}).pop(reservation_id, None)
            if entry is None:
                raise RevisionRunError(f"unknown reservation {reservation_id!r}")
            self._event("released_unsent", rid=reservation_id,
                        key=entry.get("key"))

        self._mutate(_apply)

    @classmethod
    def _state_summary(cls, state: dict[str, Any]) -> dict[str, Any]:
        return {
            "budget_state_version": BUDGET_STATE_VERSION,
            "cap_usd": float(state.get("cap_usd", 0.0)),
            "settled_usd": round(float(state.get("settled_usd", 0.0)), 6),
            "held_usd": round(float(state.get("held_usd", 0.0)), 6),
            "active_reserved_usd": round(sum(
                float(a.get("amount", 0.0))
                for a in state.get("active", {}).values()), 6),
            "exposure_usd": round(cls._exposure(state), 6),
            "n_reservations": int(state.get("n_reservations", 0)),
            "n_denials": int(state.get("n_denials", 0)),
        }

    def snapshot(self) -> dict[str, Any]:
        return self._mutate(self._state_summary)


def plan_fingerprint(plan: dict[str, Any]) -> str:
    """Stable fingerprint of a dispatch plan (deterministic JSON)."""
    return sha256(json.dumps(plan, sort_keys=True, ensure_ascii=False)
                  .encode("utf-8")).hexdigest()


def pending_tasks(tasks: Iterable[Any], lid_of: Callable[[Any], str],
                  terminal_ids: set[str]) -> list[Any]:
    """Resume view: tasks whose logical request has no terminal result yet."""
    return [t for t in tasks if lid_of(t) not in terminal_ids]


def bounded_dispatch(tasks: list[Any], worker: Callable[[Any], dict[str, Any]],
                     *, total_workers: int, per_key_limit: int,
                     key_of: Callable[[Any], str],
                     spend_cap_usd: float | None = None,
                     price_of: Callable[[dict[str, Any]], float | None] | None = None,
                     on_result: Callable[[Any, dict[str, Any]], None] | None = None,
                     budget: PersistentBudget | None = None,
                     max_cost_of: Callable[[Any], float] | None = None,
                     per_key_limit_of: Callable[[str], int] | None = None,
                     stop_after_seconds: float | None = None,
                     ) -> dict[str, Any]:
    """Dispatch with bounded concurrency and a HARD pre-call spend bound.

    ``worker(task) -> record``; ``price_of(record)`` returns the reported cost
    of one call (or None when unknown).  At most ``total_workers`` calls run
    at once and at most ``per_key_limit`` per ``key_of(task)``.

    Spend safety is PRE-CALL, never after the fact: before a call starts, a
    worst-case reservation ``max_cost_of(task)`` (which must already include
    every bounded retry of that call) is taken from ``budget`` (or from an
    in-memory budget capped at ``spend_cap_usd``).  When a reservation is
    refused no further tasks START (in-flight calls settle normally).  A
    record with an unknown cost — and any worker exception — FAILS CLOSED:
    the full reservation is held, never released as free.
    """
    if budget is None and spend_cap_usd is not None:
        budget = PersistentBudget(None, spend_cap_usd)
    if budget is not None and max_cost_of is None:
        raise RevisionRunError(
            "budgeted dispatch requires max_cost_of(task): the cap binds "
            "worst-case cost per call, so every call must declare it")
    lock = threading.Lock()
    sem_total = threading.Semaphore(total_workers)
    sems: dict[str, threading.Semaphore] = {}
    spend = [0.0]
    stopped = threading.Event()
    wall_timer = None
    if stop_after_seconds is not None:
        # clean bounded checkpoint: stop STARTING calls at the deadline so
        # in-flight calls settle normally — no reservation is abandoned
        wall_timer = threading.Timer(stop_after_seconds, stopped.set)
        wall_timer.daemon = True
        wall_timer.start()
    stats = {"submitted": 0, "dispatched": 0, "completed": 0, "failed": 0,
             "budget_stop": False, "spend_usd": 0.0, "unknown_cost_rows": 0,
             "denied": 0, "held_usd": 0.0, "wall_stop": False}

    def sem_for(key: str) -> threading.Semaphore:
        with lock:
            if key not in sems:
                limit = per_key_limit_of(key) if per_key_limit_of else per_key_limit
                sems[key] = threading.Semaphore(limit)
            return sems[key]

    def work(task: Any) -> None:
        if stopped.is_set():
            return
        key = key_of(task)
        with sem_total, sem_for(key):
            if stopped.is_set():
                return
            with lock:
                stats["dispatched"] += 1   # a call actually STARTS here
            reservation = None
            if budget is not None:
                try:
                    reservation = budget.reserve(
                        key, float(max_cost_of(task)), label=repr(task)[:120])
                except BudgetExceeded:
                    with lock:
                        stats["denied"] += 1
                        stats["budget_stop"] = True
                    stopped.set()
                    return
            try:
                record = worker(task)
            except Exception:
                # the provider may have billed an attempt we never parsed:
                # fail closed, the reservation stays held
                if budget is not None and reservation is not None:
                    budget.settle(reservation, None)
                with lock:
                    stats["failed"] += 1
                    stats["held_usd"] = round(
                        stats["held_usd"] + (float(max_cost_of(task))
                                            if reservation is not None else 0.0), 6)
                return
            cost = price_of(record) if price_of else None
            known = isinstance(cost, (int, float)) and not isinstance(cost, bool)
            if budget is not None and reservation is not None:
                budget.settle(reservation, float(cost) if known else None)
            with lock:
                stats["completed"] += 1
                if known:
                    spend[0] += float(cost)
                else:
                    stats["unknown_cost_rows"] += 1
                    stats["held_usd"] = round(
                        stats["held_usd"] + (float(max_cost_of(task))
                                            if reservation is not None else 0.0), 6)
                stats["spend_usd"] = round(spend[0], 6)
                if on_result is not None:
                    on_result(task, record)
            if budget is not None:
                snap = budget.snapshot()
                if snap["exposure_usd"] >= budget.cap_usd - 1e-9:
                    stopped.set()
                    stats["budget_stop"] = True
            elif spend_cap_usd is not None and spend[0] >= spend_cap_usd:
                stopped.set()
                stats["budget_stop"] = True

    with ThreadPoolExecutor(max_workers=total_workers) as pool:
        futures = []
        for task in tasks:
            if stopped.is_set():
                break
            with lock:
                stats["submitted"] += 1
            futures.append(pool.submit(work, task))
        for _ in as_completed(futures):
            pass
    if wall_timer is not None:
        wall_timer.cancel()
        if not stats["budget_stop"]:
            stats["wall_stop"] = True
    return stats


class VersionedRunStore:
    """Storage for ONE run version under ``base_dir/<version>/``.

    The store is append-only and idempotent: re-opening an existing version
    resumes it; a DIFFERENT plan under the same version is refused; other
    versions' directories are never touched.
    """

    def __init__(self, base_dir: Path | str, version: str) -> None:
        if not version or "/" in version or version.startswith("."):
            raise RevisionRunError(f"invalid version label {version!r}")
        self.base_dir = Path(base_dir)
        self.version = version
        self.root = self.base_dir / version
        self.private_dir = self.root / PRIVATE_DIR_NAME
        self.results_path = self.root / RESULTS_NAME
        self.usage_path = self.root / USAGE_LEDGER_NAME
        self.active_path = self.root / ACTIVE_POINTERS_NAME
        self.meta_path = self.root / RUN_META_NAME

    # ------------------------------------------------------------- lifecycle
    def initialize(self, plan: dict[str, Any], *,
                   run_fingerprint: str | None = None) -> dict[str, Any]:
        """Create (or idempotently re-open) this version's run metadata."""
        fingerprint = run_fingerprint or plan_fingerprint(plan)
        if self.meta_path.exists():
            meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
            if meta.get("plan_fingerprint") != fingerprint:
                raise RevisionRunError(
                    f"{self.meta_path}: plan fingerprint mismatch; a version "
                    f"directory is bound to ONE plan and originals are never "
                    f"overwritten (open a new version)")
            return meta
        self.root.mkdir(parents=True, exist_ok=True)
        self.private_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "revision_run_version": REVISION_RUN_VERSION,
            "version": self.version,
            "created_at": utc_now(),
            "plan_fingerprint": fingerprint,
            "plan": plan,
        }
        self.meta_path.write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n",
                                  encoding="utf-8")
        return meta

    def _require_initialized(self) -> None:
        if not self.meta_path.exists():
            raise RevisionRunError(f"{self.root}: run not initialized")

    # -------------------------------------------------------------- results
    def terminal_ids(self) -> set[str]:
        ids: set[str] = set()
        if self.results_path.exists():
            for line in self.results_path.read_text(encoding="utf-8").split("\n"):
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("terminal"):
                    ids.add(str(record["logical_request_id"]))
        return ids

    def append_result(self, record: dict[str, Any]) -> None:
        self._require_initialized()
        if record.get("terminal"):
            lid = str(record.get("logical_request_id"))
            if not lid or lid in self.terminal_ids():
                raise RevisionRunError(
                    f"duplicate terminal logical_request_id {lid!r}; results are "
                    f"append-only and a request is counted exactly once")
        self.root.mkdir(parents=True, exist_ok=True)
        with self.results_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()

    def read_results(self) -> list[dict[str, Any]]:
        records = []
        if self.results_path.exists():
            for line in self.results_path.read_text(encoding="utf-8").split("\n"):
                if line.strip():
                    records.append(json.loads(line))
        return records

    # ------------------------------------------------------------ raw (private)
    def append_raw(self, logical_request_id: str, raw: dict[str, Any]) -> Path:
        """Store the COMPLETE raw response privately (never published).

        ``raw`` must carry the full ``content``, full ``reasoning``, the
        provider ``finish_reason`` and the http status — the evidence needed
        to apply any future parser fix offline.  Nothing here is truncated.
        """
        self._require_initialized()
        for field in ("content", "reasoning", "finish_reason", "http_status"):
            if field not in raw:
                raise RevisionRunError(f"raw record missing {field!r}")
        model = str(raw.get("model") or "unknown").replace("/", "_")
        self.private_dir.mkdir(parents=True, exist_ok=True)
        path = self.private_dir / f"{model}{RAW_SUFFIX}"
        entry = {"lid": logical_request_id, "recorded_at": utc_now(), **raw}
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            fh.flush()
        return path

    def read_raws(self) -> list[dict[str, Any]]:
        records = []
        if self.private_dir.exists():
            for path in sorted(self.private_dir.glob(f"*{RAW_SUFFIX}")):
                for line in path.read_text(encoding="utf-8").split("\n"):
                    if line.strip():
                        records.append(json.loads(line))
        return records

    # --------------------------------------------------------------- usage
    def record_usage(self, logical_request_id: str, usage: dict[str, Any]) -> None:
        self._require_initialized()
        entry = {"lid": logical_request_id, "recorded_at": utc_now(), **usage}
        with self.usage_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            fh.flush()

    def usage_summary(self) -> dict[str, Any]:
        rows = []
        if self.usage_path.exists():
            for line in self.usage_path.read_text(encoding="utf-8").split("\n"):
                if line.strip():
                    rows.append(json.loads(line))

        def _sum(field: str) -> int | None:
            values = [r.get(field) for r in rows if isinstance(r.get(field), int)]
            return sum(values) if values else None
        cost_values = [r["cost_usd"] for r in rows
                       if isinstance(r.get("cost_usd"), (int, float))]
        summary = {
            "revision_run_version": REVISION_RUN_VERSION,
            "n_usage_rows": len(rows),
            "usage_input_tokens_sum": _sum("input_tokens"),
            "usage_output_tokens_sum": _sum("output_tokens"),
            "cost_usd_sum": round(sum(cost_values), 6) if cost_values else None,
            "unknown_usage_rows": len([r for r in rows
                                       if not isinstance(r.get("input_tokens"), int)]),
            "unknown_cost_rows": len(rows) - len(cost_values),
            "accounting_note": ("unknown usage/cost is counted as UNKNOWN rows, "
                                "never as zero or free"),
        }
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / USAGE_SUMMARY_NAME).write_text(
            json.dumps(summary, indent=1) + "\n", encoding="utf-8")
        return summary

    # ------------------------------------------------- active-result pointers
    def set_active(self, cell: str, *, note: str = "") -> dict[str, Any]:
        """Point a (model, dataset) cell at THIS version's results — append-only.

        The pointer file records every activation with its timestamp; earlier
        pointers are kept in ``history`` and no results file is ever replaced.
        """
        self._require_initialized()
        pointers = {"schema": "active-results-1.0.0", "active": {}, "history": []}
        if self.active_path.exists():
            pointers = json.loads(self.active_path.read_text(encoding="utf-8"))
        entry = {"cell": cell, "version": self.version, "results": RESULTS_NAME,
                 "set_at": utc_now(), "note": note}
        pointers["history"].append(entry)
        pointers["active"][cell] = entry
        self.active_path.write_text(
            json.dumps(pointers, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        return entry

    def active_pointers(self) -> dict[str, Any]:
        if self.active_path.exists():
            return json.loads(self.active_path.read_text(encoding="utf-8"))
        return {"schema": "active-results-1.0.0", "active": {}, "history": []}

    # ------------------------------------------ transport-recovery attempts
    @property
    def recovery_path(self) -> Path:
        return self.root / RECOVERY_NAME

    @property
    def recovery_pointers_path(self) -> Path:
        return self.root / RECOVERY_POINTERS_NAME

    def append_transport_recovery(self, record: dict[str, Any]) -> None:
        """Append ONE immutable transport-recovery attempt for a provisional
        row.  The original terminal row is never deleted or overwritten — a
        recovery attempt is a NEW evidence row resolved through pointers."""
        self._require_initialized()
        lid = str(record.get("logical_request_id"))
        attempt = record.get("recovery_attempt")
        if not lid or not isinstance(attempt, int):
            raise RevisionRunError(
                "transport-recovery rows need logical_request_id and integer "
                "recovery_attempt")
        for existing in self.read_transport_recoveries():
            if (str(existing.get("logical_request_id")) == lid
                    and existing.get("recovery_attempt") == attempt):
                raise RevisionRunError(
                    f"duplicate transport-recovery attempt {lid!r}#{attempt}; "
                    f"attempts are immutable and append-only")
        self.root.mkdir(parents=True, exist_ok=True)
        with self.recovery_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()

    def read_transport_recoveries(self) -> list[dict[str, Any]]:
        if not self.recovery_path.exists():
            return []
        return [json.loads(line) for line in
                self.recovery_path.read_text(encoding="utf-8").split("\n")
                if line.strip()]

    def set_recovery_pointer(self, lid: str, recovery_attempt: int,
                             *, note: str = "") -> dict[str, Any]:
        """Point a provisional lid at its latest transport-recovery attempt —
        append-only history, earlier pointers retained."""
        self._require_initialized()
        pointers = {"schema": "transport-recovery-pointers-1.0.0",
                    "active": {}, "history": []}
        if self.recovery_pointers_path.exists():
            pointers = json.loads(
                self.recovery_pointers_path.read_text(encoding="utf-8"))
        entry = {"lid": lid, "recovery_attempt": recovery_attempt,
                 "set_at": utc_now(), "note": note}
        pointers["history"].append(entry)
        pointers["active"][lid] = entry
        self.recovery_pointers_path.write_text(
            json.dumps(pointers, indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8")
        return entry

    def recovery_pointers(self) -> dict[str, Any]:
        if self.recovery_pointers_path.exists():
            return json.loads(
                self.recovery_pointers_path.read_text(encoding="utf-8"))
        return {"schema": "transport-recovery-pointers-1.0.0",
                "active": {}, "history": []}

    def resolved_rows(self) -> tuple[list[dict[str, Any]], int]:
        """Terminal rows with transport-recovery pointers applied.

        Returns (rows, n_provisional_unresolved).  A provisional row (final
        outcome still retryable: 429/5xx/timeout) is NOT a capability
        failure: it stays provisional until a recovery attempt settles it.
        Pointer application is non-destructive — the original row is kept
        verbatim and annotated as superseded.
        """
        attempts = {(str(r.get("logical_request_id")), r.get("recovery_attempt")): r
                    for r in self.read_transport_recoveries()}
        pointers = self.recovery_pointers().get("active", {})
        rows: list[dict[str, Any]] = []
        n_provisional = 0
        for record in self.read_results():
            if not record.get("terminal"):
                continue
            row = dict(record)
            lid = str(record.get("logical_request_id"))
            retryable = (bool(record.get("transport_retryable"))
                         or (record.get("status") == "http_error"
                             and is_retryable_status(record.get("http_status"))))
            if retryable:
                row["provisional"] = True
            pointer = pointers.get(lid)
            if pointer is not None:
                recovery = attempts.get((lid, pointer.get("recovery_attempt")))
                if recovery is not None and (
                        recovery.get("status") == "ok"
                        or not is_retryable_status(recovery.get("http_status"))):
                    # resolved: the recovery attempt is the active outcome
                    row = {**recovery,
                           "superseded_recovery_attempt":
                               pointer.get("recovery_attempt"),
                           "provisional": False,
                           "resolved_by_transport_recovery": True}
                    retryable = False
            if retryable:
                n_provisional += 1
            rows.append(row)
        return rows, n_provisional

    # ------------------------------------------------------------ publication
    def public_summary(self) -> dict[str, Any]:
        """Public aggregate over this version's results: counts only.

        Never includes gold keys, dataset text or raw model output — see
        ``publication_manifest``.  Transport-recovery pointers are applied:
        provisional retryable rows (429/5xx/timeout) are counted as
        PROVISIONAL pending recovery — never as capability wrong answers —
        and a cell carrying unresolved provisional rows is flagged
        ``complete: false`` so it can never be charted as complete.
        """
        results, n_provisional_total = self.resolved_rows()
        per_cell: dict[str, dict[str, Any]] = {}
        for record in results:
            cell = f"{record.get('model')}:{record.get('dataset')}"
            bucket = per_cell.setdefault(cell, {
                "n_terminal": 0, "n_correct_recovered": 0,
                "n_provisional_transport": 0, "n_transport_recovered": 0,
                "recovery_stages": {}, "status_counts": {}})
            if record.get("terminal") or record.get("resolved_by_transport_recovery"):
                bucket["n_terminal"] += 1
            if record.get("resolved_by_transport_recovery"):
                bucket["n_transport_recovered"] += 1
            if record.get("provisional"):
                bucket["n_provisional_transport"] += 1
            if record.get("correct_recovered"):
                bucket["n_correct_recovered"] += 1
            stage = str(record.get("recovery_stage"))
            bucket["recovery_stages"][stage] = bucket["recovery_stages"].get(stage, 0) + 1
            status = str(record.get("status"))
            bucket["status_counts"][status] = bucket["status_counts"].get(status, 0) + 1
        for bucket in per_cell.values():
            n = bucket["n_terminal"]
            settled = n - bucket["n_provisional_transport"]
            bucket["accuracy_recovered_on_settled"] = (
                round(bucket["n_correct_recovered"] / settled, 6)
                if settled else None)
            bucket["complete"] = bucket["n_provisional_transport"] == 0
            if not bucket["complete"]:
                bucket["provisional_note"] = (
                    "unresolved transport-retryable rows (429/5xx/timeout) are "
                    "provisional pending recovery, not capability failures; do "
                    "not chart this cell as complete")
        return {
            "revision_run_version": REVISION_RUN_VERSION,
            "version": self.version,
            "generated_at": utc_now(),
            "cells": per_cell,
            "n_provisional_transport_rows": n_provisional_total,
            "usage": self.usage_summary(),
            "publication": "public aggregate: counts/accuracy/usage only; no gold keys, no licensed text",
        }

    def publication_manifest(self) -> dict[str, Any]:
        manifest = {
            "schema": "publication-manifest-1.0.0",
            "public": [RESULTS_NAME, RUN_META_NAME, ACTIVE_POINTERS_NAME,
                       USAGE_SUMMARY_NAME, PUBLICATION_MANIFEST_NAME],
            "private": [f"{PRIVATE_DIR_NAME}/ (complete raw responses, finish "
                        f"reasons — never published)"],
            "rules": [
                "no gold answer keys in public output",
                "no licensed dataset text (questions/prompts/solutions) in public output",
                "no raw model output in public output",
                "results.jsonl rows carry gold only for offline scoring; the "
                "PUBLIC aggregate is public_summary()",
            ],
        }
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / PUBLICATION_MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
        return manifest