#!/usr/bin/env python3
"""Inventory raw baseline recoverability + minimum-rerun plan (offline).

The matched cheap-baseline sweeps lost raw evidence:

* v2 stored TRUNCATED excerpts (content[:400], reasoning[:300]) and no
  finish_reason;
* v3 stored NO raw output at all (only parsed predictions).

Consequences (recorded in the output, enforced in the report):
* v2 raws can never support a claim about REPAIRED v3 scores — v2 and v3 ran
  different protocols and v3 has no raws to reparse;
* the fixed recovery parser (jev_observatory.answer_recovery 2.0) can only
  be applied offline to rows whose raw excerpt is COMPLETE;
* rows whose stored recovery stage is parser-stable (exact/stripped) keep
  their stored answer under any parser version and need no rerun;
* every other row lacking complete raws must be RE-RUN under the versioned
  revision-run tooling (scripts/benchmark/run_baseline_revision.py) — that
  is the minimum rerun set computed here.

Outputs:
  data_report/raw_recoverability_inventory.json
  data_report/baseline_rerun_plan.json

  python scripts/benchmark/raw_recoverability.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import json

from jev_observatory.answer_recovery import RECOVERY_SPEC_VERSION, recover_choice

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data_report"
RAW2_DIR = ROOT / "runs_matched_cheap" / "raw2"
V2_RESULTS = ROOT / "runs_matched_cheap" / "v2" / "results.jsonl"
V3_RESULTS = ROOT / "runs_matched_cheap" / "v3" / "results.jsonl"

# the v2 writer stored content[:400] / reasoning[:300]; a stored row AT the
# limit may or may not be a complete response -> "possibly truncated"
V2_CONTENT_LIMIT = 400
V2_REASONING_LIMIT = 300
# stages whose recovered answer is identical under any parser version given
# the same content (whole content is the key)
PARSER_STABLE_STAGES = ("exact", "stripped", "stripped_char")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n")
            if line.strip()]


def _raw2_index() -> dict[str, dict]:
    index = {}
    for path in sorted(RAW2_DIR.glob("*.jsonl")):
        for record in _read_jsonl(path):
            index[str(record["lid"])] = record
    return index


def _raw_complete(record: dict | None) -> bool:
    if record is None:
        return False
    if len(str(record.get("content") or "")) >= V2_CONTENT_LIMIT:
        return False
    if len(str(record.get("reasoning") or "")) >= V2_REASONING_LIMIT:
        return False
    return True


def _item_keys_index() -> dict[tuple[str, str], list[str]]:
    """(dataset, item_id) -> allowed option keys, from the frozen item sets."""
    sys.path.insert(0, str(ROOT / "scripts" / "benchmark"))
    import run_cheap_matched2 as rcm2  # noqa: E402  (offline item loader)
    index = {}
    for dataset, items in rcm2.load_items().items():
        for item in items:
            keys = None
            for question in (item.get("questions") or {}).values():
                criteria = question.get("criteria")
                if isinstance(criteria, dict):
                    keys = list(criteria.keys())
            index[(dataset, str(item["id"]))] = keys
    return index


def main() -> int:
    keys_index = _item_keys_index()
    inventory = {
        "generated_by": "scripts/benchmark/raw_recoverability.py (offline, deterministic)",
        "recovery_parser_version": RECOVERY_SPEC_VERSION,
        "sources": {
            "v2": {"results": str(V2_RESULTS.relative_to(ROOT)),
                   "raw": "runs_matched_cheap/raw2/ (content[:400], reasoning[:300] — TRUNCATED; no finish_reason)"},
            "v3": {"results": str(V3_RESULTS.relative_to(ROOT)),
                   "raw": None,
                   "note": "the v3 original runner stored NO raw output; repaired v3 scores can never be claimed from artifacts"},
        },
        "do_not_use": ("v2 raw excerpts must NOT be used to claim repaired v3 "
                       "scores: protocols differ (v2 temperature-0/effort-low vs "
                       "v3 provider defaults) and v3 has no raws at all"),
        "stages": {},
    }
    rerun_cells: dict[tuple[str, str], int] = {}
    for version, results_path, raws in (("v2", V2_RESULTS, _raw2_index()),
                                        ("v3", V3_RESULTS, {})):
        rows = [r for r in _read_jsonl(results_path) if r.get("terminal")]
        ok_rows = [r for r in rows if r.get("status") == "ok"]
        complete = [r for r in ok_rows if _raw_complete(raws.get(str(r["logical_request_id"])))]
        stable = [r for r in ok_rows if r.get("recovery_stage") in PARSER_STABLE_STAGES]
        changed = 0
        reparsed_wrong_vs_stored = 0
        rerun = 0
        per_cell: dict[str, dict[str, int]] = {}
        for row in ok_rows:
            cell = f"{row['model']}:{row['dataset']}"
            bucket = per_cell.setdefault(cell, {"n_ok": 0, "n_stable": 0,
                                                "n_complete_raws": 0,
                                                "n_rerun_required": 0})
            bucket["n_ok"] += 1
            stage = row.get("recovery_stage")
            is_stable = stage in PARSER_STABLE_STAGES
            raw = raws.get(str(row["logical_request_id"]))
            has_complete = _raw_complete(raw)
            if is_stable:
                bucket["n_stable"] += 1
            if has_complete:
                bucket["n_complete_raws"] += 1
            if not is_stable and not has_complete:
                bucket["n_rerun_required"] += 1
                rerun += 1
                rerun_cells[(row["model"], row["dataset"])] = \
                    rerun_cells.get((row["model"], row["dataset"]), 0) + 1
            if has_complete and not is_stable and version == "v2":
                # offline reparse under the fixed parser — V2-ONLY diagnostic
                keys = keys_index.get((row["dataset"], str(row["item_id"])))
                if keys:
                    pred, _stage = recover_choice(str(raw.get("content") or ""),
                                                  keys,
                                                  str(raw.get("reasoning") or "") or None)
                    if pred != row.get("pred_recovered"):
                        changed += 1
                    if row.get("pred_recovered") is not None and pred is None:
                        reparsed_wrong_vs_stored += 1
        inventory["stages"][f"{version}_raw_recoverability"] = {
            "n_terminal_rows": len(rows),
            "n_ok_rows": len(ok_rows),
            "n_parser_stable_rows": len(stable),
            "n_complete_raw_rows": len(complete),
            "n_reparse_candidates": len([r for r in ok_rows
                                         if r.get("recovery_stage") not in PARSER_STABLE_STAGES
                                         and _raw_complete(raws.get(str(r["logical_request_id"])))]),
            "n_rerun_required_rows": rerun,
            "truncation_rule": ("a raw row is complete only when content < "
                                f"{V2_CONTENT_LIMIT} and reasoning < {V2_REASONING_LIMIT} "
                                "characters (the v2 writer's slice limits); rows at "
                                "the limit are counted as possibly truncated"),
        }
        if version == "v2":
            inventory["stages"]["v2_offline_reparse_diagnostic"] = {
                "scope": ("V2 ONLY: fixed-parser reparse of complete raw rows; "
                          "never evidence for v3 and never a chart number"),
                "recovery_parser_version": RECOVERY_SPEC_VERSION,
                "n_reparsed": len([r for r in ok_rows
                                   if r.get("recovery_stage") not in PARSER_STABLE_STAGES
                                   and _raw_complete(raws.get(str(r["logical_request_id"])))]),
                "answers_changed_vs_stored_parser": changed,
                "answers_recovered_now_unrecovered": reparsed_wrong_vs_stored,
            }
        inventory["stages"][f"{version}_per_cell"] = per_cell

    plan = {
        "generated_by": "scripts/benchmark/raw_recoverability.py (offline, deterministic)",
        "principle": ("minimum rerun = every ok row whose stored recovery stage is "
                      "NOT parser-stable AND whose complete raw response is "
                      "missing/truncated; parser-stable rows (exact/stripped) keep "
                      "their stored answer under any parser and are NOT rerun"),
        "tooling": ("scripts/benchmark/run_baseline_revision.py --plan/--live/--score "
                    "with jev_observatory.revision_run.VersionedRunStore: fresh "
                    "versioned run dirs (originals never overwritten), complete "
                    "private raws incl. finish_reason, resume by terminal logical "
                    "id, active-result pointers, usage accounting (unknown = "
                    "unknown), bounded concurrency + hard spend cap"),
        "rerun_cells": [
            {"model": model, "dataset": dataset, "n_rows_to_rerun": count}
            for (model, dataset), count in sorted(rerun_cells.items())
        ],
        "rerun_total_rows": sum(rerun_cells.values()),
        "excluded_from_rerun": ("rows with stored recovery stage exact/stripped "
                                "(parser-stable) and rows already settled by "
                                "pred_strict"),
        "public_aggregates_path": "data_report/baselines/<version>/public_summary.json",
        "notes": [
            "v2 raws are truncated excerpts; reparsing them yields V2-only "
            "diagnostics, never repaired v3 scores",
            "v3 rows have no raw output; ALL non-parser-stable v3 rows need rerun",
            "future runs must store complete private raws (see revision_run)",
        ],
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "raw_recoverability_inventory.json").write_text(
        json.dumps(inventory, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT_DIR / "baseline_rerun_plan.json").write_text(
        json.dumps(plan, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("[ok] wrote data_report/raw_recoverability_inventory.json")
    print("[ok] wrote data_report/baseline_rerun_plan.json "
          f"({plan['rerun_total_rows']:,} rows across {len(rerun_cells)} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())