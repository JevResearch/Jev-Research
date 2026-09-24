#!/usr/bin/env python3
"""Offline matched-comparison report (parent MATCHED-RUN-GATE, worker-built).

Entirely offline: joins stored modern-baseline predictions to the frozen gold
labels, computes ALL-REQUESTED headline accuracies, conditional accuracies,
cost/latency summaries, paired model-minus-Jev diagnostics (exact McNemar +
cluster bootstrap, Holm across the 9-comparison family per benchmark, ±0.03
equivalence tolerance), the order-robustness table (canonical-label restored
rotations vs native, matched items, never pooled), and numeric chart tables.

  python scripts/benchmark/matched_report.py --root runs_matched

Writes runs_matched/derived/report.json and runs_matched/derived/report_data.json
plus a printed headline summary. No model calls, no credentials, no images.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import argparse
import json

from jev_observatory.matched import DATASETS, MatchedError
from jev_observatory.matched_report import (
    MatchedReportError,
    build_chart_tables,
    build_report,
    write_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="runs_matched")
    parser.add_argument("--benchmark-root", default="runs_benchmark")
    parser.add_argument("--runs-reviewed-root", default="runs_reviewed")
    parser.add_argument("--models", default=None,
                        help="comma-separated subset of models to report (default: all frozen)")
    args = parser.parse_args()

    try:
        report = build_report(args.root, benchmark_root=args.benchmark_root,
                              runs_reviewed_root=args.runs_reviewed_root)
    except (MatchedError, MatchedReportError) as exc:
        print(f"[report-error] {exc}", file=sys.stderr)
        return 3

    derived = Path(args.root) / "derived"
    write_report(report, derived / "report.json")
    tables = build_chart_tables(report)
    (derived / "report_data.json").write_text(
        json.dumps(tables, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    models = [m for m in report["comparison_table"]]
    if args.models:
        wanted = {m.strip() for m in args.models.split(",") if m.strip()}
        models = [m for m in models if m in wanted]
    headline = {}
    for model in models:
        row = report["comparison_table"][model]
        headline[model] = {
            d: {"acc_all": row[d]["accuracy_all"],
                "acc_cond": row[d]["accuracy_conditional"],
                "acc_weighted": row[d]["accuracy_weighted"],
                "n": row[d]["n_requested"], "correct": row[d]["correct"]}
            for d in DATASETS
        }
    print(json.dumps({
        "freeze_sha256": report["freeze_sha256"],
        "headline": headline,
        "paired": {d: {m: {
            "paired_n": v["paired_n"],
            "point_diff": v["point_diff_model_minus_jev"],
            "mcnemar_p": v["mcnemar_exact_p"],
            "holm_p": v["p_holm_adjusted"],
            "ci": [v["bootstrap"]["ci_low"], v["bootstrap"]["ci_high"]] if v["bootstrap"] else None,
            "equiv_3pp": v["equivalent_within_3pp"],
        } for m, v in report["paired"][d].items()} for d in DATASETS},
        "report_json": str(derived / "report.json"),
        "report_data_json": str(derived / "report_data.json"),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())