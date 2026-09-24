#!/usr/bin/env python3
"""Matched modern-baseline dispatch (parent MATCHED-RUN-GATE, worker-built).

Parent-invokable. Default is an OFFLINE dry run (freeze verified, all maps
checked, nothing dispatched, no transport constructed). Live dispatch requires
``--live`` AND the environment opt-in (JEVO_ALLOW_LIVE=1) AND
OPENROUTER_API_KEY — read ONLY from the process environment, never echoed.

Exact parent commands (see docs/modern-comparison/MATCHED-EXECUTION-PACKET.md):

  # 1. dry run (offline; also proves the freeze reloads byte-exact)
  python scripts/benchmark/run_matched.py --root runs_matched --dry-run

  # 2. pilot: exactly 10 items/model = 90 calls, raw budget/finish check
  JEVO_ALLOW_LIVE=1 python scripts/benchmark/run_matched.py --root runs_matched --live --pilot

  # 3. after parent review of the pilot summary: continuation (pilot counted once)
  JEVO_ALLOW_LIVE=1 python scripts/benchmark/run_matched.py --root runs_matched --live

Safety: hard aggregate $75 cost cap including outstanding reservations (from
the frozen cost guard); <=4 concurrent requests TOTAL, <=2 per model; ONE
attempt per item, no retries of any kind; stop on 401/402/403, repeated
transport/format failures, budget denial, wall-time expiry or cancellation;
resume skips finished items and refuses to resume past uncertain (timeout)
attempts; accounting restored from the append-only attempt ledger.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import argparse
import json

from jev_observatory.matched import MatchedError, run_id_for
from jev_observatory.matched_report import score_model_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="runs_matched")
    parser.add_argument("--live", action="store_true",
                        help="dispatch real requests (requires JEVO_ALLOW_LIVE=1 and "
                             "OPENROUTER_API_KEY in the process environment)")
    parser.add_argument("--dry-run", action="store_true",
                        help="offline verification only (default; accepted for explicitness)")
    parser.add_argument("--pilot", action="store_true",
                        help="dispatch ONLY the frozen 10-item pilot per model "
                             "(90 calls total); finished items are never re-run")
    parser.add_argument("--models", default=None,
                        help="comma-separated subset of the nine frozen model ids")
    parser.add_argument("--max-wall-seconds", type=float, default=6 * 3600.0)
    parser.add_argument("--rpm", type=int, default=600)
    args = parser.parse_args()

    if args.pilot and not args.live:
        print("[refused] --pilot is a dispatch mode; pass --live (use --dry-run for "
              "offline verification)", file=sys.stderr)
        return 2
    if args.dry_run and args.live:
        print("[refused] --dry-run and --live are mutually exclusive", file=sys.stderr)
        return 2
    models = [m.strip() for m in args.models.split(",") if m.strip()] \
        if args.models else None

    from jev_observatory.matched import MatchedExecutor, load_matched_freeze

    try:
        freeze = load_matched_freeze(args.root)
        executor = MatchedExecutor(
            args.root, freeze=freeze, models=models, pilot_only=bool(args.pilot),
            max_wall_seconds=args.max_wall_seconds, rpm=args.rpm,
        )
        state = executor.run(live=args.live)
    except MatchedError as exc:
        print(f"[fail-closed] {exc}", file=sys.stderr)
        return 3

    summary = {
        "live": state["live"],
        "pilot_only": state["pilot_only"],
        "models_run": state["models_run"],
        "n_pending": state.get("n_pending"),
        "global_stop_reason": state["global_stop_reason"],
        "guard_final": state["guard_final"],
        "stages": {m: {
            "run_id": s["run_id"], "dispatched": s["dispatched"], "ok": s["ok"],
            "error": s["error"], "skipped_resume": s["skipped_resume"],
        } for m, s in state["stages"].items()},
        "recorded_at": state["recorded_at"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.pilot and args.live:
        print(json.dumps(_pilot_budget_finish_check(args.root, state), indent=2))
    return 0


def _pilot_budget_finish_check(root: str, state: dict) -> dict:
    """Raw per-call pilot accounting: usage, provider cost, finish_reason."""
    from jev_observatory.ledger import RunStore
    from jev_observatory.redactor import Redactor

    freeze = load_matched_freeze(root)
    prices, _ = _prices()
    per_model = {}
    for model in state["models_run"]:
        store = RunStore(root, run_id_for(freeze, model), redactor=Redactor())
        attempts = store.attempts()
        calls = []
        for attempt in attempts:
            extra = attempt.get("extra") or {}
            usage_raw = extra.get("usage_raw") or {}
            calls.append({
                "logical_request_id": attempt.get("logical_request_id"),
                "status": attempt.get("outcome"),
                "http_status": attempt.get("http_status"),
                "finish_reason": extra.get("finish_reason"),
                "input_tokens": attempt.get("usage_input_tokens"),
                "output_tokens": attempt.get("usage_output_tokens"),
                "provider_cost": usage_raw.get("cost"),
            })
        block = score_model_dataset(store.directory, freeze, model, "mmlu", prices=prices)
        per_model[model] = {
            "n_calls": len(calls),
            "calls": calls,
            "mmlu_headline": block["headline"],
        }
    return {"pilot_calls_total": sum(v["n_calls"] for v in per_model.values()),
            "per_model": per_model}


def _prices():
    from jev_observatory.matched import load_catalog_prices

    return load_catalog_prices()


if __name__ == "__main__":
    sys.exit(main())