#!/usr/bin/env python3
"""Derived prediction overrides: offline reparse of all stored COMPLETE raw
responses with the current recovery parser (answer-recovery-2.1.0).

The append-only originals (results.jsonl, transport_recovery.jsonl, usage
ledger, wires) are NEVER modified and costs are never re-counted.  Where the
new parser's prediction differs from the prediction currently in force, a
DERIVED override row is written with explicit parser-revision and source-hash
provenance; build_active_summary consumes the file by explicit version.

Outputs (versioned, under the v4r1 run's derived/):
  derived/prediction_overrides-<parser-version>.jsonl
  derived/override_provenance.json   (source hashes + exact before/after counts)
"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import collections
import hashlib
import json

from jev_observatory.answer_recovery import (RECOVERY_SPEC_VERSION,
                                             recover_choice)

import build_active_summary as bas

ROOT = Path(__file__).resolve().parents[2]
V4R1 = ROOT / "runs_matched_cheap" / "v4r1"
DERIVED = V4R1 / "derived"
ACTIVE_ITEMS = ROOT / "data_report" / "baselines" / "v4r1" / "active_items.jsonl"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _raw_for_lid(lid: str, http_status: object, raws: list[dict]) -> dict | None:
    """The COMPLETE raw of the active attempt: latest record for this lid with
    the active attempt's http status, else latest overall."""
    matches = [r for r in raws if str(r.get("lid")) == lid]
    if not matches:
        return None
    same = [r for r in matches if r.get("http_status") == http_status]
    return (same or matches)[-1]


def main() -> int:
    resolved, _n_att, _finish = bas._v4r1_state()
    keys = bas._item_keys()
    raws: list[dict] = []
    raw_paths = sorted((V4R1 / "private_raw").glob("*.jsonl"))
    for path in raw_paths:
        for line in path.read_text(encoding="utf-8").split("\n"):
            if line.strip():
                raws.append(json.loads(line))
    raw2 = bas._raw2_index() if hasattr(bas, "_raw2_index") else None
    if raw2 is None:
        from freeze_rerun_plan import _raw2_index as r2
        raw2 = r2()

    # gold by lid from the immutable originals
    gold_by_lid: dict[str, object] = {}
    for row in bas._read_jsonl(bas.V2_RESULTS) + bas._read_jsonl(bas.V3_RESULTS):
        gold_by_lid[str(row.get("logical_request_id"))] = row.get("gold")
    for row in resolved.values():
        gold_by_lid[str(row.get("logical_request_id"))] = row.get("gold")

    before_items = [json.loads(l) for l in ACTIVE_ITEMS.read_text(
        encoding="utf-8").split("\n") if l.strip()]

    overrides: list[dict] = []
    per_cell = collections.defaultdict(lambda: {
        "n_reparsed": 0, "n_prediction_changed": 0,
        "before_correct": 0, "after_correct": 0})
    raw_source_hashes = {str(p.relative_to(ROOT)): _sha256(p)
                         for p in raw_paths}

    for item in before_items:
        klass = item.get("row_class")
        if klass not in ("replacement", "replacement_resolved", "salvage"):
            continue
        cell = str(item["cell"])
        model, dataset = cell.split(":", 1)
        version = str(item["source_version"])
        item_id = str(item["item_id"])
        lid = f"{version}|{model}|{dataset}|{item_id}"
        bucket = per_cell[cell]
        bucket["n_reparsed"] += 1
        if item.get("correct_recovered"):
            bucket["before_correct"] += 1

        raw = None
        if klass == "salvage":
            entry = raw2.get(f"{model}:{dataset}:{item_id}")
            if entry is not None:
                raw = {"content": entry.get("content"),
                       "reasoning": entry.get("reasoning")}
        else:
            row = resolved.get(lid)
            if row is not None:
                raw = _raw_for_lid(lid, row.get("http_status"), raws)
        if raw is None:
            continue

        ks = keys.get((dataset, item_id), [])
        new_pred, new_stage = recover_choice(
            str(raw.get("content") or ""), ks, str(raw.get("reasoning") or ""))
        gold_lid = (f"{model}:{dataset}:{item_id}"
                    if klass == "salvage" else lid)
        gold = gold_by_lid.get(gold_lid)
        new_correct = bool(new_pred) and new_pred == gold
        if new_correct:
            bucket["after_correct"] += 1

        if klass == "salvage":
            # salvage predictions are recomputed by the builder from the same
            # raw + same parser: no override row needed, but a correctness
            # change vs the previous build is still counted above
            continue
        old = resolved[lid]
        old_pred = old.get("pred_recovered")
        old_stage = str(old.get("recovery_stage"))
        if new_pred != old_pred or new_stage != old_stage:
            bucket["n_prediction_changed"] += 1
            overrides.append({
                "lid": lid, "cell": cell, "item_id": item_id,
                "row_class": klass,
                "parser_version": RECOVERY_SPEC_VERSION,
                "prev_pred_recovered": old_pred,
                "prev_recovery_stage": old_stage,
                "pred_recovered": new_pred,
                "recovery_stage": new_stage,
                "correct_recovered": new_correct,
                "source_raw": (f"runs_matched_cheap/v4r1/private_raw/"
                               f"{str(raw.get('model') or model).replace('/', '_')}.jsonl"),
            })

    DERIVED.mkdir(parents=True, exist_ok=True)
    overrides_path = DERIVED / f"prediction_overrides-{RECOVERY_SPEC_VERSION}.jsonl"
    overrides_path.write_text(
        "".join(json.dumps(o, ensure_ascii=False) + "\n" for o in overrides),
        encoding="utf-8")
    provenance = {
        "schema": "prediction-override-provenance-1.0.0",
        "parser_version": RECOVERY_SPEC_VERSION,
        "overrides_file": str(overrides_path.relative_to(ROOT)),
        "overrides_sha256": _sha256(overrides_path),
        "source_raw_sha256": raw_source_hashes,
        "policy": ("append-only originals untouched; no cost re-counting; "
                   "overrides change PREDICTIONS only, derived and versioned"),
        "totals": {
            "n_reparsed": sum(b["n_reparsed"] for b in per_cell.values()),
            "n_overrides": len(overrides),
            "before_correct": sum(b["before_correct"] for b in per_cell.values()),
            "after_correct": sum(b["after_correct"] for b in per_cell.values()),
        },
        "per_cell": {cell: dict(b) for cell, b in sorted(per_cell.items())},
    }
    (DERIVED / "override_provenance.json").write_text(
        json.dumps(provenance, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8")
    t = provenance["totals"]
    print(f"[overrides] reparsed {t['n_reparsed']} raw-backed rows; "
          f"{t['n_overrides']} prediction overrides; "
          f"correct {t['before_correct']} -> {t['after_correct']}")
    changed_cells = {c: b for c, b in provenance["per_cell"].items()
                     if b["before_correct"] != b["after_correct"]
                     or b["n_prediction_changed"]}
    for cell, b in sorted(changed_cells.items()):
        print(f"  {cell}: changed={b['n_prediction_changed']} "
              f"correct {b['before_correct']}->{b['after_correct']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
