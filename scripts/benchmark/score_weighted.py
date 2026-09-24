#!/usr/bin/env python3
"""Probability-weighted scoring CLI (SEPARATE diagnostic variant).

Offline, no model calls: computes the probability-weighted score from stored
run artifacts (p(gold) per item; see jev_observatory.weighted_score) and
writes derived/weighted_score.json next to the greedy score.json.

  python scripts/benchmark/score_weighted.py --root runs_benchmark_ext --stage math500_choice
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

from jev_observatory.benchmark_exec import load_freeze, stage_run_id
from jev_observatory.weighted_score import ScoringError, weighted_score_run, write_weighted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="runs_benchmark_ext")
    parser.add_argument("--stage", required=True,
                        choices=["mmlu_full", "option_rotations", "arc_test",
                                 "arc_agi2_choice", "arc_agi2_score", "arc_agi2_task",
                                 "math500_choice", "math500_score",
                                 "gpqa_diamond", "hle_text_mc"])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    freeze = load_freeze(args.root)
    if args.stage not in freeze["stages"]:
        print(f"[refused] stage {args.stage!r} not in the freeze", file=sys.stderr)
        return 2
    run_dir = Path(args.root) / stage_run_id(freeze, args.stage)
    try:
        doc = weighted_score_run(run_dir, stage=args.stage)
    except ScoringError as exc:
        print(f"[weighted-scoring-error] {exc}", file=sys.stderr)
        return 3
    out = Path(args.out) if args.out else run_dir / "derived" / "weighted_score.json"
    write_weighted(doc, out)
    summary = {
        "stage": doc["stage"],
        "variant": doc["variant"],
        "n_expected": doc["n_expected"],
        "n_weighted_scored": doc["n_weighted_scored"],
        "n_excluded": doc["n_excluded"],
        "weighted_accuracy_mean": doc["weighted_accuracy_mean"],
        "weighted_accuracy_p50": doc.get("weighted_accuracy_p50"),
    }
    if doc.get("per_cell_weighted_mean") is not None:
        summary["per_cell_weighted_mean"] = doc["per_cell_weighted_mean"]
    summary["score_path"] = str(out)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
