#!/usr/bin/env python3
"""Offline matched-freeze preparation (parent MATCHED-RUN-GATE, worker-built).

Builds the write-once matched freeze under --root (default runs_matched/):
1000 MMLU-Pro TEST (proportional subject stratification, seed 20260920),
300 ARC-Challenge TEST (simple random sample, same seed family), and all 120
reviewed fresh TEST items. Then runs the FULL offline dry-run check: every
item/label/wire map is rebuilt from the reloaded freeze and compared to the
frozen wire hashes — all before any model call exists.

  python scripts/benchmark/prepare_matched_freeze.py --root runs_matched

Offline only: no credentials, no network, no model calls, no image access.
Existing artifacts under runs_benchmark/ and runs_reviewed/ are read-only.
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

from jev_observatory.matched import (
    MatchedError,
    build_matched_freeze,
    load_catalog_prices,
    load_matched_freeze,
    smoke_calibration,
)
from jev_observatory.openrouter import WireConfig, build_chat_payload


def _load_smoke_module():
    """Load the parent's smoke script for its EXACT smoke question + model ids."""
    path = Path(__file__).resolve().parent / "smoke_openrouter.py"
    spec = importlib.util.spec_from_file_location("smoke_openrouter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="runs_matched")
    parser.add_argument("--mmlu", default="data/test-00000-of-00001.parquet")
    parser.add_argument("--arc", default="data/arc_challenge_test.parquet")
    parser.add_argument("--benchmark-root", default="runs_benchmark",
                        help="Jev run root used for the paired join verification")
    args = parser.parse_args()

    # Calibration input: chars of the exact smoke payload the parent sent
    # (replayed as text here; the saved live usage carries the token counts).
    # The live smoke ran with reasoning_effort=low and max_output_tokens=1024
    # (recorded in its wire_config); the mean over the nine requested model ids
    # is used as the calibration character count.
    smoke_module = _load_smoke_module()
    smoke_chars_values = [
        len(json.dumps(build_chat_payload(
            smoke_module.SMOKE_QUESTION,
            WireConfig(model=m, reasoning_effort="low", max_output_tokens=1024)),
            ensure_ascii=False))
        for m in smoke_module.NINE_MODELS
    ]
    smoke_chars = round(sum(smoke_chars_values) / len(smoke_chars_values))

    try:
        freeze = build_matched_freeze(
            root=args.root, mmlu_path=args.mmlu, arc_path=args.arc,
            benchmark_root=args.benchmark_root, smoke_payload_chars=smoke_chars)
    except MatchedError as exc:
        print(f"[fail-closed] {exc}", file=sys.stderr)
        return 3

    prices, flags = load_catalog_prices()
    calibration = smoke_calibration()
    summary = {
        "freeze": f"{args.root}/freeze/matched_frozen.json",
        "freeze_sha256": freeze["deterministic_sha256"][:12],
        "models": len(freeze["models"]),
        "datasets": {d: freeze["datasets"][d]["n"] for d in ("mmlu", "arc", "fresh")},
        "requests_per_model": len(freeze["request_order"][freeze["models"][0]]),
        "total_requests": len(freeze["request_order"][freeze["models"][0]]) * len(freeze["models"]),
        "pilot_calls": freeze["pilot_n"] * len(freeze["models"]),
        "mmlu_sampling": freeze["datasets"]["mmlu"]["sampling"]["per_subject"],
        "cost_guard_cap_usd": freeze["cost_guard"]["hard_cap_usd"],
        "cost_estimates": {
            "total_expected_usd": freeze["cost_guard"]["estimates"]["total_expected_usd"],
            "total_worst_case_usd": freeze["cost_guard"]["estimates"]["total_worst_case_usd"],
        },
        "reasoning_effort_supported": flags["reasoning_effort_supported"],
        "smoke_usage": calibration["usage"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    # Full offline dry-run: rebuild every request/label/wire map from the
    # RELOADED freeze and compare with the stored hashes.
    from jev_observatory.matched import MatchedExecutor

    reloaded = load_matched_freeze(args.root)
    executor = MatchedExecutor(args.root, freeze=reloaded)
    state = executor.run(live=False)
    n_wire = len(reloaded["wire_sha256"])
    print(f"[ok] freeze written and verified offline: {n_wire} wire hashes match "
          f"across {len(reloaded['models'])} models "
          f"(executor dry-run stop_reason={state['global_stop_reason']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())