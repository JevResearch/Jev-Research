#!/usr/bin/env python3
"""Versioned baseline re-run CLI (revision-run tooling; replaces v1-v3 runners).

Uses jev_observatory.revision_run.VersionedRunStore so baseline repair runs:
live in a FRESH versioned directory (originals never overwritten), store
COMPLETE private raw responses incl. finish_reason, resume by terminal
logical id, keep active-result pointers, account usage (unknown = unknown),
and dispatch with bounded concurrency under a HARD pre-call spend budget
(PersistentBudget: atomic reservations against one persistent USD ledger —
past spend, held unknown-cost reservations and in-flight worst-case
reservations all count; resume can never reset the budget).

Tasks come from the frozen minimum-rerun manifest
(data_report/baseline_rerun_tasks.jsonl, produced offline by
scripts/benchmark/freeze_rerun_plan.py); every task carries its ORIGINAL
row's recorded wire so a replacement matches the protocol it replaces.

Offline commands (no key, no network):
  python scripts/benchmark/run_baseline_revision.py --plan --version v4r1
  python scripts/benchmark/run_baseline_revision.py --score --version v4r1

Live dispatch (gated; keys come from the private credentials file and are
injected into the process environment only — never printed, never logged):
  JEVO_ALLOW_LIVE=1 \
    python scripts/benchmark/run_baseline_revision.py --live --version v4r1 \
      --spend-cap 10.0 \
      --credentials-file /tmp/jev-private-credentials-47rujcux/credentials.json

Public aggregates (no gold keys, no licensed text) are written to
data_report/baselines/<version>/ — a stable path for the integration stage.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import argparse
import importlib.util
import json
import os
import time

from jev_observatory.answer_recovery import recover_choice, recovery_spec
from jev_observatory.matched import build_baseline_request
from jev_observatory.revision_run import (BUDGET_STATE_NAME, PersistentBudget,
                                          VersionedRunStore, bounded_dispatch,
                                          is_retryable_status, pending_tasks,
                                          retry_after_seconds)

ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = ROOT / "runs_matched_cheap"
PUBLIC_AGG_DIR = ROOT / "data_report" / "baselines"
TASKS_PATH = ROOT / "data_report" / "baseline_rerun_tasks.jsonl"
DEFAULT_SPEND_CAP_USD = 10.0      # stage authorization ceiling (both providers)
TOTAL_WORKERS = 32                # bounded threaded concurrency (matches the
                                  # v3 runner's total lane)
PER_MODEL_WORKERS = 8             # standard lane (the earlier reduction applied
                                  # to the rate-limited provider; see below)
# models/providers observed under 429 rate limiting keep the tighter lane
RATE_LIMITED_PER_MODEL = {"google/gemma-3-4b-it": 2}
TRANSPORT_ATTEMPTS = 3            # bounded retries per logical request
RETRY_STATUS = {0, 429, 500, 502, 503, 504}   # kept for reference; policy lives
                                             # in revision_run.is_retryable_status
RESERVATION_SAFETY_INPUT_TOKENS = 4096
KEY_ENV_NAME = "OPENROUTER_API_KEY"

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("rcm2", _HERE / "run_cheap_matched2.py")
rcm2 = importlib.util.module_from_spec(_spec)
sys.modules["rcm2"] = rcm2
_spec.loader.exec_module(rcm2)


def _read_tasks(path: Path = TASKS_PATH) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"[plan] frozen task manifest missing: {path} "
                         f"(run scripts/benchmark/freeze_rerun_plan.py first)")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n")
            if line.strip()]


def _lid(task: dict) -> str:
    return str(task["logical_request_id"])


def _max_cost_of(prices: dict[str, dict[str, float]], task: dict) -> float:
    """Worst-case pre-call reservation: bounded retries x (generous input at
    the provider input price + the FULL configured output cap at output price)."""
    wire = task.get("wire") or {}
    rate = prices[str(task["model"])]
    max_out = int(wire.get("max_output_tokens") or 4096)
    per_call = (RESERVATION_SAFETY_INPUT_TOKENS * rate["input_per_token"]
                + max_out * rate["output_per_token"])
    return TRANSPORT_ATTEMPTS * per_call


def _fetch_prices(transport) -> tuple[dict[str, dict[str, float]], dict]:
    """CURRENT provider-reported price metadata (non-billable catalog GET)."""
    from jev_observatory.openrouter import MODELS_PATH
    resp = transport.get(MODELS_PATH)
    body = json.loads(resp.body.decode("utf-8", "replace")) if resp.body else {}
    prices: dict[str, dict[str, float]] = {}
    for entry in body.get("data", []):
        pricing = entry.get("pricing") or {}
        try:
            prices[str(entry["id"])] = {
                "input_per_token": float(pricing["prompt"]),
                "output_per_token": float(pricing["completion"])}
        except (KeyError, TypeError, ValueError):
            continue
    meta = {"source": "openrouter /models (provider-reported, fetched at run start)",
            "http_status": resp.status_code, "n_models": len(prices)}
    return prices, meta


def _interleave(tasks: list[dict]) -> list[dict]:
    """Deterministic round-robin ACROSS (model, dataset) cells so different
    models run in parallel (per-model lanes stay bounded) instead of one
    slow thinking-model cell blocking every other cell."""
    cells: dict[tuple[str, str], list[dict]] = {}
    for t in tasks:
        cells.setdefault((t["model"], t["dataset"]), []).append(t)
    queues = [cells[k] for k in sorted(cells)]
    out: list[dict] = []
    idx = 0
    while any(idx < len(q) for q in queues):
        for q in queues:
            if idx < len(q):
                out.append(q[idx])
        idx += 1
    return out


def cmd_plan(version: str, spend_cap: float) -> int:
    tasks = _read_tasks()
    plan = {
        "wire_policy": ("each replacement reuses its ORIGINAL row's recorded "
                        "wire (selective mixed-time replacement protocol)"),
        "recovery": recovery_spec(),
        "task_manifest": str(TASKS_PATH.relative_to(ROOT)),
        "n_calls": len(tasks),
        "n_cells": len({(t["model"], t["dataset"]) for t in tasks}),
        "budget": {
            "mechanism": "revision_run.PersistentBudget pre-call reservations",
            "cap_usd": spend_cap,
            "transport_attempts_per_request": TRANSPORT_ATTEMPTS,
            "unknown_cost": "fail closed: reservation stays held",
        },
    }
    store = VersionedRunStore(BASE_DIR, version)
    meta = store.initialize(plan)
    PersistentBudget(store.root / BUDGET_STATE_NAME, spend_cap)
    store.publication_manifest()
    print(f"[plan] {version}: {plan['n_calls']:,} calls in {plan['n_cells']} cells, "
          f"fingerprint {meta['plan_fingerprint'][:12]}, budget cap ${spend_cap:.2f}")
    return 0


def cmd_live(version: str, spend_cap: float, credentials_file: Path,
             datasets: set[str] | None = None,
             recover_transport: bool = False,
             max_wall_seconds: float | None = None) -> int:
    if os.environ.get("JEVO_ALLOW_LIVE") != "1":
        raise SystemExit("[gate] JEVO_ALLOW_LIVE=1 required")
    if not credentials_file.exists():
        raise SystemExit(f"[gate] credentials file not found: {credentials_file}")
    creds = json.loads(credentials_file.read_text(encoding="utf-8"))
    key = creds.get(KEY_ENV_NAME)
    if not key:
        raise SystemExit(f"[gate] {KEY_ENV_NAME} missing in credentials file")
    os.environ[KEY_ENV_NAME] = str(key)   # inject; never printed/logged
    del creds, key

    from jev_observatory.openrouter import (CHAT_PATH, OpenRouterHttpTransport,
                                            WireConfig, build_chat_payload)
    store = VersionedRunStore(BASE_DIR, version)
    if not store.meta_path.exists():
        raise SystemExit("[plan] run not initialized; run --plan --version first")
    budget = PersistentBudget(store.root / BUDGET_STATE_NAME, spend_cap)
    recovery_tasks: list[dict] = []
    if recover_transport:
        # TRANSPORT RECOVERY: re-dispatch provisional rows (final outcome was
        # 429/5xx/timeout) as explicit, budget-reserved recovery attempts.
        # Originals are immutable: attempts land in transport_recovery.jsonl
        # and are resolved through versioned pointers, never by overwrite.
        rows, _n_prov = store.resolved_rows()
        attempts_by_lid: dict[str, int] = {}
        for rec in store.read_transport_recoveries():
            lid = str(rec.get("logical_request_id"))
            attempts_by_lid[lid] = max(attempts_by_lid.get(lid, 0),
                                       int(rec.get("recovery_attempt") or 0))
        for row in rows:
            if not row.get("provisional"):
                continue
            lid = str(row["logical_request_id"])
            recovery_tasks.append({
                "logical_request_id": lid,
                "model": row["model"], "dataset": row["dataset"],
                "item_id": str(row["item_id"]),
                "source_version": row.get("source_version"),
                "source_recovery_stage": row.get("source_recovery_stage"),
                "wire": row.get("wire") or {},
                "recovery_attempt": attempts_by_lid.get(lid, 0) + 1,
            })
        tasks = recovery_tasks
        print(f"[live] {version}: TRANSPORT RECOVERY of {len(tasks):,} "
              f"provisional rows (attempts are immutable + pointer-resolved)")
    else:
        all_tasks = _read_tasks()
        if datasets:
            all_tasks = [t for t in all_tasks if t["dataset"] in datasets]
        tasks = pending_tasks(all_tasks, _lid, store.terminal_ids())
        tasks = _interleave(tasks)
    print(f"[live] {version}: {len(tasks):,} pending "
          f"({len(store.terminal_ids()):,} terminal); budget "
          f"${budget.snapshot()['exposure_usd']:.4f}/${spend_cap:.2f} already "
          f"exposed (resume never resets the ledger)")

    transports = {}
    prices: dict[str, dict[str, float]] = {}
    price_meta = {}

    def transport_for(model: str):
        t = transports.get(model)
        if t is None:
            t = transports[model] = OpenRouterHttpTransport(
                os.environ[KEY_ENV_NAME], timeout_seconds=900.0)
        return t

    def worker(task: dict) -> dict:
        model, dataset, item_id = task["model"], task["dataset"], task["item_id"]
        items = _items_by_key.get((dataset, item_id))
        if items is None:
            # manifest row whose item is not in the frozen set: fail loudly,
            # never silently substitute
            raise RuntimeError(f"item {dataset}:{item_id} missing from frozen items")
        req, _conv = build_baseline_request(items, model=model)
        keys = list(req.questions[next(iter(req.questions))].criteria)
        wire_d = task.get("wire") or {}
        wire = WireConfig(model=model,
                          reasoning_effort=wire_d.get("reasoning_effort"),
                          max_output_tokens=int(wire_d.get("max_output_tokens")
                                                or 4096),
                          extra_body=wire_d.get("extra_body"))
        payload = build_chat_payload(req, wire)
        transport = transport_for(model)
        cost_sum: float | None = 0.0
        resp = None
        body = None
        for attempt in range(1, TRANSPORT_ATTEMPTS + 1):
            resp = transport.post(CHAT_PATH, payload)
            body = (json.loads(resp.body.decode("utf-8", "replace"))
                    if resp.body else None)
            usage = (body or {}).get("usage") if isinstance(body, dict) else None
            c = (usage or {}).get("cost")
            if isinstance(c, (int, float)) and not isinstance(c, bool):
                cost_sum += float(c)          # every attempt is billed & counted
            else:
                cost_sum = None               # fail closed downstream
            if resp.status_code == 200 or not is_retryable_status(resp.status_code):
                break
            if attempt < TRANSPORT_ATTEMPTS:
                # transport-only backoff: honors the provider's Retry-After
                # (rate-limited models such as Gemma under 429), capped
                time.sleep(retry_after_seconds(resp.headers, attempt))
        content, reasoning = rcm2._extract_content_reasoning(body)
        content = content or ""          # empty-body transport outcomes
        reasoning = reasoning or ""
        finish = rcm2._extract_finish_reason(body)
        usage = (body or {}).get("usage") if isinstance(body, dict) else None
        pred_strict = content.strip() if content.strip() in keys else None
        pred, stage = recover_choice(content, keys, reasoning)
        retryable_final = is_retryable_status(resp.status_code)
        record = {
            "logical_request_id": _lid(task), "model": model, "dataset": dataset,
            "item_id": str(item_id), "gold": rcm2.gold_of(items),
            "status": "ok" if resp.status_code == 200 else "http_error",
            "http_status": resp.status_code,
            "transport_retryable": retryable_final,
            "provisional": retryable_final,
            "pred_strict": pred_strict, "pred_recovered": pred,
            "recovery_stage": stage,
            "correct_strict": pred_strict == rcm2.gold_of(items) if pred_strict else False,
            "correct_recovered": pred == rcm2.gold_of(items) if pred else False,
            "cost_reported": cost_sum,
            "source_version": task.get("source_version"),
            "source_recovery_stage": task.get("source_recovery_stage"),
            "wire": wire_d,
            "terminal": True,
            "protocol": "selective-replacement (original recorded wire preserved)",
        }
        # COMPLETE private raw evidence (never truncated, never published)
        store.append_raw(record["logical_request_id"], {
            "model": model, "content": content, "reasoning": reasoning,
            "finish_reason": finish, "http_status": resp.status_code,
        })
        store.record_usage(record["logical_request_id"], {
            "input_tokens": (usage or {}).get("prompt_tokens"),
            "output_tokens": (usage or {}).get("completion_tokens"),
            "cost_usd": cost_sum,
        })
        return record

    def on_result(task, record):
        if recover_transport:
            # immutable recovery attempt + versioned pointer; the original
            # terminal row is never deleted or overwritten
            record = {**record, "recovery_attempt": task["recovery_attempt"],
                      "attempt_kind": "transport-recovery"}
            store.append_transport_recovery(record)
            store.set_recovery_pointer(
                str(record["logical_request_id"]), task["recovery_attempt"],
                note="transport recovery over provisional row")
        else:
            store.append_result(record)

    def boot_prices():
        nonlocal prices, price_meta
        prices, price_meta = _fetch_prices(transports.setdefault(
            "_catalog", OpenRouterHttpTransport(
                os.environ[KEY_ENV_NAME], timeout_seconds=120.0)))
        missing = sorted({t["model"] for t in tasks} - set(prices))
        if missing:
            raise SystemExit(f"[price] provider metadata lacks {missing}; "
                             f"refusing to guess prices")
        (store.root / "provider_price_metadata.json").write_text(
            json.dumps({"meta": price_meta,
                        "prices": prices}, indent=1) + "\n", encoding="utf-8")

    boot_prices()
    _items_by_key = {}
    for dataset, items in rcm2.load_items().items():
        for item in items:
            _items_by_key[(dataset, str(item["id"]))] = item

    stats = bounded_dispatch(
        tasks, worker, total_workers=TOTAL_WORKERS,
        per_key_limit=PER_MODEL_WORKERS, key_of=lambda t: t["model"],
        per_key_limit_of=lambda k: RATE_LIMITED_PER_MODEL.get(k,
                                                              PER_MODEL_WORKERS),
        budget=budget, max_cost_of=lambda t: _max_cost_of(prices, t),
        price_of=lambda r: r.get("cost_reported"),
        on_result=on_result, stop_after_seconds=max_wall_seconds)
    for model, transport in list(transports.items()):
        if model != "_catalog":
            transport.close()
    transports["_catalog"].close()
    cells = sorted({f"{t['model']}:{t['dataset']}" for t in tasks})
    for cell in cells:
        store.set_active(cell)
    store.usage_summary()
    store.publication_manifest()
    stats["budget_final"] = budget.snapshot()
    print(f"[live] {json.dumps(stats)}")
    return 0


def cmd_grant(version: str, from_cap: float, to_cap: float,
              granted_by: str, note: str) -> int:
    """Migrate the version's persistent budget cap under an explicit,
    audited USER grant (oldcap/newcap event; settled + held + active
    preserved; nothing is reset and no new ledger is opened)."""
    store = VersionedRunStore(BASE_DIR, version)
    if not store.meta_path.exists():
        raise SystemExit(f"[grant] run {version} not initialized")
    budget = PersistentBudget(store.root / BUDGET_STATE_NAME, from_cap)
    snap = budget.grant_cap(to_cap, from_cap_usd=from_cap,
                            granted_by=granted_by, note=note)
    print(f"[grant] {version}: cap ${from_cap:.2f} -> ${to_cap:.2f} "
          f"(granted_by={granted_by!r}); exposure preserved at "
          f"${snap['exposure_usd']:.4f} "
          f"(settled ${snap['settled_usd']}, held ${snap['held_usd']}, "
          f"active ${snap['active_reserved_usd']})")
    return 0


def cmd_score(version: str) -> int:
    store = VersionedRunStore(BASE_DIR, version)
    public = store.public_summary()
    store.publication_manifest()
    out = PUBLIC_AGG_DIR / version / "public_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(public, indent=1, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(f"[score] wrote {out.relative_to(ROOT)} "
          f"({len(public['cells'])} cells)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--version", required=True, help="run version label (e.g. v4r1)")
    ap.add_argument("--spend-cap", type=float, default=DEFAULT_SPEND_CAP_USD,
                    help="HARD worst-case exposure cap in USD (persistent ledger)")
    ap.add_argument("--credentials-file", type=Path,
                    default=Path("/tmp/jev-private-credentials-47rujcux/credentials.json"),
                    help="private credentials JSON (values are never printed)")
    ap.add_argument("--datasets", default="",
                    help="comma list of datasets to dispatch (priority staging; "
                         "empty = all pending tasks)")
    ap.add_argument("--recover-transport", action="store_true",
                    help="re-dispatch provisional transport rows (429/5xx/"
                         "timeout) as immutable, pointer-resolved recovery "
                         "attempts under budget reservation")
    ap.add_argument("--max-wall-seconds", type=float, default=None,
                    help="clean bounded checkpoint: stop starting calls at the "
                         "deadline, let in-flight calls settle (no abandoned "
                         "reservations), exit resumably")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--plan", action="store_true")
    g.add_argument("--live", action="store_true")
    g.add_argument("--score", action="store_true")
    g.add_argument("--grant-cap", action="store_true",
                   help="audited cap migration: --grant-cap-from OLD to "
                        "--spend-cap NEW under --granted-by")
    ap.add_argument("--grant-cap-from", type=float, default=None,
                    help="the exact previous cap the grant supersedes")
    ap.add_argument("--granted-by", default="",
                    help="explicit grant label for the audit event")
    ap.add_argument("--grant-note", default="")
    a = ap.parse_args()
    if a.grant_cap:
        if a.grant_cap_from is None or not a.granted_by:
            raise SystemExit("[grant] --grant-cap-from and --granted-by required")
        return cmd_grant(a.version, a.grant_cap_from, a.spend_cap,
                         a.granted_by, a.grant_note)
    if a.plan:
        return cmd_plan(a.version, a.spend_cap)
    if a.live:
        datasets = {d.strip() for d in a.datasets.split(",") if d.strip()} or None
        return cmd_live(a.version, a.spend_cap, a.credentials_file, datasets,
                        recover_transport=a.recover_transport,
                        max_wall_seconds=a.max_wall_seconds)
    return cmd_score(a.version)


if __name__ == "__main__":
    sys.exit(main())
