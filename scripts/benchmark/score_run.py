#!/usr/bin/env python3
"""Offline benchmark scorer CLI: gold join + accuracy + honest accounting.

Parent-invokable, offline, no credentials, no network. Scores one benchmark
stage run directory (planned by prepare_freeze.py, dispatched by run_jev_full.py)
and writes machine-readable JSON for the report graphics.

  python scripts/benchmark/score_run.py --root runs_benchmark --stage mmlu_full
  python scripts/benchmark/score_run.py --root runs_benchmark --stage option_rotations
  python scripts/benchmark/score_run.py --root runs_benchmark --stage arc_test
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

from jev_observatory.benchmark_exec import load_freeze, stage_run_id
from jev_observatory.benchmark_score import ScoringError, score_run, write_score


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="runs_benchmark")
    parser.add_argument("--stage", required=True,
                        choices=["mmlu_full", "option_rotations", "arc_test",
                                 "arc_agi2_choice", "arc_agi2_score", "arc_agi2_task",
                                 "math500_choice", "math500_score",
                                 "gpqa_diamond", "hle_text_mc"])
    parser.add_argument("--out", default=None,
                        help="output path (default <run_dir>/derived/score.json)")
    args = parser.parse_args()

    freeze = load_freeze(args.root)
    if args.stage not in freeze["stages"]:
        print(f"[refused] stage {args.stage!r} not in the freeze", file=sys.stderr)
        return 2
    run_dir = Path(args.root) / stage_run_id(freeze, args.stage)
    try:
        if args.stage.startswith("arc_agi2"):
            from jev_observatory.arc_agi2_score import aggregate_grid_stage

            doc = aggregate_grid_stage(run_dir, stage=args.stage)
        elif args.stage == "math500_score":
            from jev_observatory.benchmark_score import score_digit_readout_run

            doc = score_digit_readout_run(run_dir, stage=args.stage)
        else:
            doc = score_run(run_dir, stage=args.stage)
    except ScoringError as exc:
        print(f"[scoring-error] {exc}", file=sys.stderr)
        return 3
    out = Path(args.out) if args.out else run_dir / "derived" / "score.json"
    write_score(doc, out)
    summary = {
        "stage": doc["stage"],
        "score_path": str(out),
    }
    if args.stage.startswith("arc_agi2"):
        summary.update({
            "n_chunk_items_expected": doc["n_chunk_items_expected"],
            "n_chunk_items_terminal": doc["n_chunk_items_terminal"],
            "n_missing_terminal": doc["n_missing_terminal"],
            "status_counts": doc["status_counts"],
            "tasks_solved": doc["tasks_solved"],
            "task_accuracy": doc["task_accuracy"],
            "grids_correct": doc["grids_correct"],
            "grid_accuracy": doc["grid_accuracy"],
            "cell_accuracy_diagnostic": doc["cell_accuracy_diagnostic"],
            "cells_correct": doc["cells_correct"],
            "cells_total": doc["cells_total"],
        })
    else:
        summary.update({
            "n_expected": doc["n_expected"],
            "n_scored": doc["n_scored"],
            "n_missing_terminal": doc["n_missing_terminal"],
            "accuracy": doc["accuracy"],
            "wilson_95": doc["wilson_95"],
            "strict_format_failures": doc["strict_format_failures"],
            "status_counts": doc["status_counts"],
            "by_group_n": {g: v["n"] for g, v in doc["by_group"].items()},
            "by_rotation_variant": {
                k: v["accuracy"] for k, v in doc["by_rotation_variant"].items()},
        })
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
