#!/usr/bin/env python3
"""Live architecture-probe dispatcher (sequential, randomized, gate-guarded).

Runs the jev_observatory.arch_probe battery through the real endpoint and
records raw evidence rows + analysis. Dispatch is STRICTLY sequential (one
request in flight) so latency is never queue-contaminated; the call order is
randomized with a recorded seed; cold/warm connection state is controlled for
the coldwarm family.

Live execution requires BOTH the opt-in and the key (same gate as the
benchmark runner); otherwise this only prints the plan.

  JEVO_ALLOW_LIVE=1 TYPESAFE_API_KEY=... \
    python scripts/benchmark/run_arch_probe.py --out runs_archprobe

A --limit N runs a cheap randomized subset (smoke). The key is only ever read
from the environment and never written to any artifact.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from jev_observatory import arch_probe as ap
from jev_observatory.redact import Redactor, base_url, get_api_key, live_calls_allowed
from jev_observatory.transport import HttpxTransport


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs_archprobe")
    parser.add_argument("--families",
                        default=",".join(ap.FAMILIES))
    parser.add_argument("--limit", type=int, default=0,
                        help="run only N randomized calls (cheap smoke)")
    parser.add_argument("--seed", type=int, default=97)
    parser.add_argument("--reps", type=int, default=0,
                        help="override per-family reps (0 = builders' default)")
    parser.add_argument("--fresh", action="store_true",
                        help="truncate rows.jsonl instead of appending")
    parser.add_argument("--reset-every", type=int, default=40,
                        help="force a cold connection every N warm calls")
    args = parser.parse_args()

    families = [f.strip() for f in args.families.split(",") if f.strip()]
    kw = {} if not args.reps else {}
    calls: list[ap.ProbeCall] = []
    for fam in families:
        builder = ap.BUILDERS[fam]
        try:
            calls.extend(builder(**kw) if args.reps == 0 else _with_reps(builder, args.reps))
        except TypeError:
            calls.extend(builder())

    live = live_calls_allowed() and bool(get_api_key())
    if not live:
        print("[plan-only] live gate not satisfied (need JEVO_ALLOW_LIVE=1 and "
              "TYPESAFE_API_KEY). Planned calls:")
        for fam in families:
            print(f"  {fam:12s} {sum(1 for c in calls if c.family == fam):4d}")
        return 1

    rng = random.Random(args.seed)
    rng.shuffle(calls)
    if args.limit:
        calls = calls[: args.limit]
    # keep concurrency groups intact (shuffle group order, not members)
    seq_calls = [c for c in calls if c.family != "concurrency"]
    groups: dict[str, list] = {}
    for c in calls:
        if c.family == "concurrency":
            groups.setdefault(c.meta["group"], []).append(c)
    ordered_groups = list(groups.values())
    rng.shuffle(ordered_groups)
    dispatch = seq_calls + [m for g in ordered_groups for m in g]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "rows.jsonl"
    # append by default so re-running one family never destroys prior evidence;
    # --fresh opts into a clean grid
    if args.fresh or not rows_path.exists():
        rows_path.write_text("", encoding="utf-8")

    redactor = Redactor.from_environment()
    http = HttpxTransport(base_url=base_url(), api_key=get_api_key() or "",
                          timeout_seconds=120.0, redactor=redactor)
    client = http._client

    rows: list[dict] = []

    def _fire(call, cold):
        return ap.record_from_response(call, client, cold=cold)

    seq = 0
    with rows_path.open("a", encoding="utf-8") as fh:
        # ---- sequential families (concurrency-1: no queue contamination) ----
        since_reset = 0
        for call in seq_calls:
            cold = call.meta.get("connection") == "cold"
            if cold or since_reset >= args.reset_every:
                http.reset_pool(); client = http._client; since_reset = 0
                cold = cold or True
            row = _fire(call, cold)
            row["seq"] = seq; seq += 1
            since_reset += 1
            rows.append(row)
            fh.write(redactor.text(json.dumps(row, ensure_ascii=False)) + "\n")
            fh.flush()
            if row.get("error"):
                print(f"[err] {call.config_id}: {row['error']}", file=sys.stderr)
        # ---- concurrency groups (fire each group with a thread pool) ----
        for grp in ordered_groups:
            cvals = {m.meta["c"] for m in grp}
            workers = max(1, int(next(iter(cvals))))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futs = {pool.submit(_fire, m, False): m for m in grp}
                for fut in futs:
                    pass
                results = [fut.result() for fut in list(futs)]
            for row in results:
                row["seq"] = seq; seq += 1
                rows.append(row)
                fh.write(redactor.text(json.dumps(row, ensure_ascii=False)) + "\n")
                fh.flush()
    http.close()

    analysis = ap.analyze_rows(rows)
    (out / "analysis.json").write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    total_in = sum(r.get("usage_input_tokens") or 0 for r in rows)
    total_out = sum(r.get("usage_output_tokens") or 0 for r in rows)
    (out / "BILLING.json").write_text(json.dumps({
        "n_calls": len(rows),
        "n_errors": sum(1 for r in rows if r.get("error")),
        "usage_input_tokens": total_in,
        "usage_output_tokens": total_out,
        "estimated_cost_usd_at_0p042_per_M_in": round(total_in / 1e6 * 0.042, 4),
        "seed": args.seed, "families": families, "limit": args.limit,
    }, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"n_calls": len(rows), "input_tokens": total_in,
                      "rows": str(rows_path), "analysis": str(out / "analysis.json")},
                     indent=2))
    for fam in ("prefill", "headcount", "optioncount", "coldwarm",
                "concurrency", "ancestry", "mergerate"):
        a = analysis.get(fam) or {}
        summary = {k: v for k, v in a.items()
                   if k not in ("note", "self_batch_note", "interpretation")}
        print(f"--- {fam}: {json.dumps(summary, ensure_ascii=False)}")
    return 0


def _with_reps(builder, reps):
    import inspect
    params = inspect.signature(builder).parameters
    if "reps" in params:
        return builder(reps=reps)
    return builder()


if __name__ == "__main__":
    sys.exit(main())
