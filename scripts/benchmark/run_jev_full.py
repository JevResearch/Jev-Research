#!/usr/bin/env python3
"""Full Jev benchmark executor entry point (parent-invokable, inside screen).

Environment-inheriting: reads ONLY the process environment for credentials
(TYPESAFE_API_KEY / TYPESAFE_BASE_URL) and the live opt-in (JEVO_ALLOW_LIVE=1).
It never prints or stores a key, and never makes a model call unless
``--live`` is passed AND the opt-in AND key are present.

Exact parent commands:

  # 1. dry run (offline): plan + hash-verify every stage, dispatch nothing
  python scripts/benchmark/run_jev_full.py --root runs_benchmark

  # 2. live full run (inside screen 80506 session, after parent approval):
  JEVO_ALLOW_LIVE=1 python scripts/benchmark/run_jev_full.py --root runs_benchmark --live

Resume is automatic and honest: finished logical requests are never
re-dispatched; uncertain (timeout) attempts fail the run closed; budgets are
restored from the stored attempt ledger and never silently reset.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import argparse
import json
import sys
from pathlib import Path

from jev_observatory.benchmark_exec import (
    DEFAULT_CONCURRENCY,
    DEFAULT_MAX_ESTIMATED_INPUT_TOKENS,
    DEFAULT_MAX_WALL_SECONDS,
    DEFAULT_RPM,
    REQUEST_CEILING,
    BenchmarkExecutor,
    ExecutorError,
    load_freeze,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="runs_benchmark")
    parser.add_argument("--live", action="store_true",
                        help="dispatch real requests (requires JEVO_ALLOW_LIVE=1 "
                             "and TYPESAFE_API_KEY in the environment)")
    parser.add_argument("--stages", default="mmlu_full,option_rotations,arc_test",
                        help="comma-separated stage subset to run (default: all)")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--rpm", type=int, default=DEFAULT_RPM)
    parser.add_argument("--max-requests", type=int, default=REQUEST_CEILING)
    parser.add_argument("--max-estimated-input-tokens", type=int,
                        default=DEFAULT_MAX_ESTIMATED_INPUT_TOKENS)
    parser.add_argument("--max-wall-seconds", type=float, default=DEFAULT_MAX_WALL_SECONDS)
    args = parser.parse_args()

    # Pre-flight: distinguish "not prepared" from "credentials missing".
    freeze_path = Path(args.root) / "freeze" / "frozen.json"
    if not freeze_path.exists():
        print(f"[refused] no freeze at {freeze_path}; run "
              "scripts/benchmark/prepare_freeze.py first", file=sys.stderr)
        return 2
    freeze = load_freeze(args.root)
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    missing = [s for s in stages if s not in freeze["stages"]]
    if missing:
        print(f"[refused] stages not in the freeze: {missing}", file=sys.stderr)
        return 2

    import os

    if args.live:
        if os.environ.get("JEVO_ALLOW_LIVE") != "1":
            print("[refused] live dispatch requires JEVO_ALLOW_LIVE=1", file=sys.stderr)
            return 2
        if not os.environ.get("TYPESAFE_API_KEY"):
            print("[refused] live dispatch requires TYPESAFE_API_KEY in the "
                  "environment (never echoed)", file=sys.stderr)
            return 2

    executor = BenchmarkExecutor(
        args.root,
        stages=tuple(stages),
        max_requests=args.max_requests,
        max_estimated_input_tokens=args.max_estimated_input_tokens,
        max_wall_seconds=args.max_wall_seconds,
        concurrency=args.concurrency,
        rpm=args.rpm,
    )
    try:
        state = executor.run(dry_run=not args.live)
    except ExecutorError as exc:
        print(f"[fail-closed] {exc}", file=sys.stderr)
        return 3
    print(json.dumps({
        "dry_run": state["dry_run"],
        "global_stop_reason": state.get("global_stop_reason"),
        "stages": {stage: {
            key: value for key, value in summary.items()
            if key in {"run_id", "dispatched", "ok", "error", "skipped_resume",
                       "n_requests", "stop_reason", "stage_wall_seconds"}
        } for stage, summary in state["stages"].items()},
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
