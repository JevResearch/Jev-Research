#!/usr/bin/env python3
"""ACTIVE per-cell summary builder: original stable rows + offline-salvaged
rows + versioned replacements, against FULL original sampled denominators.

This is the complete-cell aggregation seam for integration.  It NEVER charts
accuracy from the selected rerun subset alone: every cell's denominator is the
full active sample (n_requested), and any cell with pending replacements or
unresolved provisional transport rows is marked PARTIAL with explicit
exclusions (never a chartable accuracy).

Row classes per ACTIVE cell (merged selector = run_matched3.cmd_score mirror):
  original_stable  stored answer kept (parser-stable stage; invariant)
  settled_non_ok   original terminal non-ok (wrong under all-requested)
  salvage          fixed parser applied offline over COMPLETE v2 raws
  replacement      versioned v4r1 rerun row (original wire preserved),
                   incl. transport-recovery pointer resolution
  replacement_pending / replacement_provisional  -> cell is PARTIAL

Outputs (license-safe: counts/accuracy/cost/flags only — never gold keys,
never licensed text, never raw model output):
  data_report/baselines/v4r1/public_summary.json
  data_report/baselines/v4r1/active_source_map.json
  data_report/baselines/v4r1/active_items.jsonl
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
import glob
import hashlib
import json

from jev_observatory.answer_recovery import (RECOVERY_SPEC_VERSION,
                                             recover_choice)

from freeze_rerun_plan import (PARSER_STABLE_STAGES, V2_CONTENT_LIMIT,
                               V2_REASONING_LIMIT, _raw2_index, _raw_complete,
                               _read_jsonl, active_cell_selector)

ROOT = Path(__file__).resolve().parents[2]
V2_RESULTS = ROOT / "runs_matched_cheap" / "v2" / "results.jsonl"
V3_RESULTS = ROOT / "runs_matched_cheap" / "v3" / "results.jsonl"
V4R1_DIR = ROOT / "runs_matched_cheap" / "v4r1"
OUT_DIR = ROOT / "data_report" / "baselines" / "v4r1"

RETRYABLE = {0, 429, 500, 502, 503, 504}


def _item_keys() -> dict[tuple[str, str], list[str]]:
    sys.path.insert(0, str(ROOT / "scripts" / "benchmark"))
    import run_cheap_matched2 as rcm2  # offline item loader
    keys = {}
    for dataset, items in rcm2.load_items().items():
        for item in items:
            ks = None
            for question in (item.get("questions") or {}).values():
                criteria = question.get("criteria")
                if isinstance(criteria, dict):
                    ks = list(criteria.keys())
            keys[(dataset, str(item["id"]))] = ks or []
    return keys


def _jev_map() -> dict[str, dict[str, bool]]:
    src = {
        "mmlu": "runs_benchmark/bench-mmlu_full-*/derived/score.json",
        "gpqa": "runs_benchmark_ext2/bench-gpqa_diamond-*/derived/score.json",
        "math500_choice": "runs_benchmark_ext/bench-math500_choice-*/derived/score.json",
        "hle_text_mc": "runs_benchmark_ext2/bench-hle_text_mc-*/derived/score.json",
        "arc": "runs_benchmark/bench-arc_test-*/derived/score.json",
    }
    out = {}
    for ds, pat in src.items():
        found = glob.glob(str(ROOT / pat))
        if not found:
            continue
        doc = json.loads(Path(found[0]).read_text(encoding="utf-8"))
        out[ds] = {str(it["item_id"]).split(":")[-1]: bool(it.get("correct"))
                   for it in doc.get("per_item") or []}
    return out


def _mn(b: int, c: int) -> float:
    import math
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def _v4r1_state() -> tuple[dict[str, dict], dict[str, int], dict[str, str]]:
    """lid -> resolved replacement row, lid -> n recovery attempts, lid ->
    finish_reason (from complete private raws)."""
    results = {}
    for row in _read_jsonl(V4R1_DIR / "results.jsonl"):
        results[str(row["logical_request_id"])] = row
    attempts = collections.defaultdict(list)
    for row in _read_jsonl(V4R1_DIR / "transport_recovery.jsonl"):
        attempts[str(row["logical_request_id"])].append(row)
    pointers = {}
    pp = V4R1_DIR / "transport_recovery_pointers.json"
    if pp.exists():
        pointers = json.loads(pp.read_text(encoding="utf-8")).get("active", {})
    finish: dict[str, str] = {}
    for path in sorted((V4R1_DIR / "private_raw").glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").split("\n"):
            if line.strip():
                rec = json.loads(line)
                if rec.get("finish_reason") is not None:
                    finish[str(rec["lid"])] = str(rec["finish_reason"])
    resolved: dict[str, dict] = {}
    for lid, row in results.items():
        retryable = (bool(row.get("transport_retryable"))
                     or (row.get("status") == "http_error"
                         and row.get("http_status") in RETRYABLE))
        active = dict(row)
        pointer = pointers.get(lid)
        if pointer is not None:
            cand = [a for a in attempts[lid]
                    if a.get("recovery_attempt") == pointer.get("recovery_attempt")]
            rec = cand[0] if cand else None
            if rec is not None and (rec.get("status") == "ok"
                                    or rec.get("http_status") not in RETRYABLE):
                active = rec
                active["resolved_by_transport_recovery"] = True
                retryable = False
        active["_provisional"] = bool(retryable)
        active["_n_recovery_attempts"] = len(attempts[lid])
        resolved[lid] = active
    n_attempts = {lid: len(v) for lid, v in attempts.items()}
    return resolved, n_attempts, finish


def _load_overrides() -> tuple[dict[str, dict], dict]:
    """Derived, versioned prediction overrides (explicit parser revision +
    source-hash provenance; verified before use)."""
    prov_path = V4R1_DIR / "derived" / "override_provenance.json"
    if not prov_path.exists():
        return {}, {}
    prov = json.loads(prov_path.read_text(encoding="utf-8"))
    path = ROOT / prov["overrides_file"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != prov.get("overrides_sha256"):
        raise RuntimeError(
            f"{prov_path}: override file hash mismatch — refusing to apply")
    rows = {}
    for line in path.read_text(encoding="utf-8").split("\n"):
        if line.strip():
            rec = json.loads(line)
            rows[str(rec["lid"])] = rec
    return rows, prov


def build() -> dict:
    raws = _raw2_index()
    keys = _item_keys()
    jev = _jev_map()
    v2_rows = [r for r in _read_jsonl(V2_RESULTS) if r.get("terminal")]
    v3_rows = [r for r in _read_jsonl(V3_RESULTS) if r.get("terminal")]
    v3_plan = json.loads((ROOT / "runs_matched_cheap" / "v3" / "freeze_v3.json")
                         .read_text(encoding="utf-8"))["models"]
    v3_models = {m["id"] for m in v3_plan}
    n_req = {}
    sys.path.insert(0, str(ROOT / "scripts" / "benchmark"))
    import run_cheap_matched2 as rcm2
    n_req = {ds: len(v) for ds, v in rcm2.load_items().items()}

    active = active_cell_selector(v2_rows, v3_rows, n_req, v3_models)
    repl, n_attempts, finish = _v4r1_state()
    overrides, override_prov = _load_overrides()
    n_overrides_applied = 0

    per_cell: dict[str, dict] = {}
    items_out: list[dict] = []
    totals = collections.Counter()

    for (model, dataset), (version, rows) in sorted(active.items()):
        cell = f"{model}:{dataset}"
        block: dict[str, object] = {
            "model": model, "dataset": dataset,
            "active_source_version": version,
            "selector_rule": ("complete v3 supersedes v2; incomplete v3 falls "
                              "back to complete v2; v2-only carryovers kept"),
            "original_wire_protocol": ("v2-temperature0-effort-low"
                                       if version == "v2"
                                       else "v3-provider-defaults"),
            "n_requested": len(rows),
            "counts": collections.Counter(),
            "n_correct": 0,
            "cost": {"original_reported_usd": 0.0, "original_n_billed": 0,
                     "original_n_unknown": 0,
                     "replacement_reported_usd": 0.0,
                     "replacement_n_billed": 0, "replacement_n_unknown": 0,
                     "replacement_n_attempts_billed": 0},
            "effects": {"strict_format_failures_recovered": 0,
                        "unrecovered_settled": 0,
                        "cap_finish_length_replacements": 0,
                        "finish_reason_unknown_original_rows": 0},
            "n_gold_join_mismatch": 0,
        }
        counts: collections.Counter = block["counts"]  # type: ignore[assignment]
        n_pending = n_prov = 0
        correct = 0
        for row in rows:
            stage = str(row.get("recovery_stage"))
            status = str(row.get("status"))
            gold = row.get("gold")
            lid = f"{version}|{model}|{dataset}|{row['item_id']}"
            item_rec: dict[str, object] = {
                "item_id": str(row["item_id"]), "cell": cell,
                "gold_join": "item_id", "source_version": version,
            }
            if status != "ok":
                klass, ok, pred = "settled_non_ok", False, None
                counts["settled_non_ok"] += 1
                block["effects"]["finish_reason_unknown_original_rows"] += 1  # type: ignore[index]
                cost = row.get("cost_reported")
                if isinstance(cost, (int, float)) and not isinstance(cost, bool):
                    block["cost"]["original_reported_usd"] += float(cost)  # type: ignore[index]
                    block["cost"]["original_n_billed"] += 1  # type: ignore[index]
                else:
                    block["cost"]["original_n_unknown"] += 1  # type: ignore[index]
            elif stage in PARSER_STABLE_STAGES:
                klass = "original_stable"
                counts["original_stable"] += 1
                ok = bool(row.get("correct_recovered"))
                pred = row.get("pred_recovered")
                block["effects"]["finish_reason_unknown_original_rows"] += 1  # type: ignore[index]
                cost = row.get("cost_reported")
                if isinstance(cost, (int, float)) and not isinstance(cost, bool):
                    block["cost"]["original_reported_usd"] += float(cost)  # type: ignore[index]
                    block["cost"]["original_n_billed"] += 1  # type: ignore[index]
                else:
                    block["cost"]["original_n_unknown"] += 1  # type: ignore[index]
            elif (version == "v2"
                  and _raw_complete(raws.get(str(row.get("logical_request_id"))))):
                # OFFLINE SALVAGE: fixed parser over the complete raw
                klass = "salvage"
                counts["salvage"] += 1
                raw = raws[str(row.get("logical_request_id"))]
                ks = keys.get((dataset, str(row["item_id"])), [])
                pred, _stage = recover_choice(str(raw.get("content") or ""),
                                              ks,
                                              str(raw.get("reasoning") or ""))
                ok = bool(pred) and pred == gold
                cost = row.get("cost_reported")
                if isinstance(cost, (int, float)) and not isinstance(cost, bool):
                    block["cost"]["original_reported_usd"] += float(cost)  # type: ignore[index]
                    block["cost"]["original_n_billed"] += 1  # type: ignore[index]
                else:
                    block["cost"]["original_n_unknown"] += 1  # type: ignore[index]
            else:
                rep = repl.get(lid)
                if rep is None:
                    klass = "replacement_pending"
                    counts["replacement_pending"] += 1
                    n_pending += 1
                elif rep.get("_provisional"):
                    klass = "replacement_provisional"
                    counts["replacement_provisional"] += 1
                    n_prov += 1
                    if rep.get("gold") is not None and rep.get("gold") != gold:
                        block["n_gold_join_mismatch"] += 1  # type: ignore[index]
                else:
                    klass = ("replacement_resolved"
                             if rep.get("resolved_by_transport_recovery")
                             else "replacement")
                    counts[klass] += 1
                    ok = bool(rep.get("correct_recovered"))
                    pred = rep.get("pred_recovered")
                    stage = str(rep.get("recovery_stage"))
                    if rep.get("gold") is not None and rep.get("gold") != gold:
                        block["n_gold_join_mismatch"] += 1  # type: ignore[index]
                    cost = rep.get("cost_reported")
                    if isinstance(cost, (int, float)) and not isinstance(cost, bool):
                        block["cost"]["replacement_reported_usd"] += float(cost)  # type: ignore[index]
                        block["cost"]["replacement_n_billed"] += 1  # type: ignore[index]
                    else:
                        block["cost"]["replacement_n_unknown"] += 1  # type: ignore[index]
                    fr = finish.get(lid)
                    if fr == "length":
                        block["effects"]["cap_finish_length_replacements"] += 1  # type: ignore[index]
                    if rep.get("pred_strict") is None and rep.get("pred_recovered"):
                        block["effects"]["strict_format_failures_recovered"] += 1  # type: ignore[index]
                    elif not rep.get("pred_recovered"):
                        block["effects"]["unrecovered_settled"] += 1  # type: ignore[index]
            ov = overrides.get(lid)
            if ov is not None and klass in ("salvage", "replacement",
                                           "replacement_resolved"):
                # DERIVED prediction override (versioned parser revision);
                # originals untouched, costs not re-counted
                pred = ov.get("pred_recovered")
                stage = str(ov.get("recovery_stage"))
                ok = bool(pred) and pred == gold
                n_overrides_applied += 1
            if klass in ("original_stable", "settled_non_ok", "salvage",
                         "replacement", "replacement_resolved"):
                if ok:
                    correct += 1
            item_rec.update({
                "row_class": klass,
                "prediction_override": ov is not None,
                "parser_revision": (RECOVERY_SPEC_VERSION
                                    if ov is not None else None),
                "provider_route": ("deepinfra/bf16-paid-pinned"
                                   if (klass.startswith("replacement")
                                       and model == "google/gemma-3-4b-it")
                                   else ("provider-unknown-original"
                                         if not klass.startswith("replacement")
                                         else "provider-default")),
                "protocol": (block["original_wire_protocol"]
                             if klass in ("original_stable", "settled_non_ok",
                                          "salvage")
                             else "replacement-original-wire-preserved"),
                "correct_recovered": ok if klass in (
                    "original_stable", "settled_non_ok", "salvage",
                    "replacement", "replacement_resolved") else None,
                "recovery_stage": (stage if klass != "salvage"
                                   else (stage if ov is not None
                                         else "salvage-reparse")),
                "cost_reported_usd": (row.get("cost_reported")
                                      if not klass.startswith("replacement")
                                      else (rep.get("cost_reported")
                                            if klass not in ("replacement_pending",
                                                             "replacement_provisional")
                                            else None)),
                "n_recovery_attempts": n_attempts.get(lid, 0),
                "finish_reason": finish.get(lid),
            })
            items_out.append(item_rec)

        n_settled = len(rows) - n_pending - n_prov
        block["n_correct"] = correct
        block["n_settled"] = n_settled
        block["n_pending_replacements"] = n_pending
        block["n_provisional_transport"] = n_prov
        block["complete"] = (n_pending == 0 and n_prov == 0)
        block["status"] = "complete" if block["complete"] else "partial"
        if block["complete"]:
            block["accuracy_all_requested"] = round(correct / len(rows), 6)
        else:
            block["accuracy_all_requested"] = None
            block["partial_settled_accuracy"] = (
                round(correct / n_settled, 6) if n_settled else None)
            block["exclusion_reason"] = (
                f"{n_pending} pending replacement(s), {n_prov} unresolved "
                f"provisional transport row(s) — NOT chartable; "
                f"partial_settled_accuracy is diagnostic only")
        jm = jev.get(dataset) or {}
        pairs = [(jm.get(str(r["item_id"])), None) for r in rows
                 if str(r["item_id"]) in jm]
        if pairs and block["complete"]:
            b = sum(1 for j, _ in pairs if j)
            block["jev_paired"] = {
                "n_paired": len(pairs),
                "jev_correct_on_subset": b,
                "note": "paired on item_id; baseline correctness joined per row",
            }
        block["counts"] = dict(counts)
        block["cost"]["original_reported_usd"] = round(
            block["cost"]["original_reported_usd"], 6)  # type: ignore[index]
        block["cost"]["replacement_reported_usd"] = round(
            block["cost"]["replacement_reported_usd"], 6)  # type: ignore[index]
        block["cost"]["bill_note"] = ("provider-reported per-item costs; "
                                      "unknown = counted unknown, never free")
        per_cell[cell] = block
        totals["n_requested"] += len(rows)
        totals["n_correct"] += correct
        totals["n_pending"] += n_pending
        totals["n_provisional"] += n_prov
        totals["n_complete_cells" if block["complete"] else "n_partial_cells"] += 1

    summary = {
        "schema": "active-public-summary-1.0.0",
        "version": "v4r1",
        "generated_by": "scripts/benchmark/build_active_summary.py",
        "denominator_policy": ("every cell's denominator is its FULL original "
                               "sampled set (n_requested) joined to actual gold "
                               "by item_id — never the rerun subset; accuracy is "
                               "emitted ONLY for complete cells"),
        "parser_revision": {
            "version": RECOVERY_SPEC_VERSION,
            "overrides_file": override_prov.get("overrides_file"),
            "overrides_sha256": override_prov.get("overrides_sha256"),
            "source_raw_sha256": override_prov.get("source_raw_sha256"),
            "n_overrides_applied": n_overrides_applied,
            "note": ("derived, versioned prediction overrides over stored "
                     "complete raws; append-only originals untouched; costs "
                     "not re-counted"),
        },
        "mixed_time_note": ("original stable rows (original run time/wire) + "
                            "offline salvages + versioned replacements "
                            "(original wire preserved): selective mixed-time "
                            "replacement; provider drift uncontrolled"),
        "provider_routing_provenance": {
            "gemma_replacements": ("model google/gemma-3-4b-it pinned to the "
                                   "DeepInfra PAID route (provider.order "
                                   "['deepinfra/bf16'], allow_fallbacks=false; "
                                   "ROUTING only — generation parameters "
                                   "unchanged); actual provider + generation "
                                   "id + error metadata recorded in private_raw"),
            "gemma_originals": ("provider UNKNOWN — original raws carry no "
                                "provider info; free-route use is NOT claimed"),
            "routing_confound": ("gemma rows mix unpinned original runs with "
                                 "DeepInfra-pinned replacements: backend "
                                 "provider differs across the mixed-time row "
                                 "classes — a recorded confound, never "
                                 "silently averaged away"),
            "residual_429_cause": ("OpenRouter body metadata: 'google/"
                                   "gemma-3-4b-it is temporarily rate-limited "
                                   "upstream' (upstream rate limit — not "
                                   "account quota or billing); remaining "
                                   "provisional rows excluded, not retried "
                                   "endlessly"),
        },
        "totals": {
            "n_cells": len(per_cell),
            "n_complete_cells": totals["n_complete_cells"],
            "n_partial_cells": totals["n_partial_cells"],
            "n_requested": totals["n_requested"],
            "n_correct_settled": totals["n_correct"],
            "n_pending_replacements": totals["n_pending"],
            "n_provisional_transport": totals["n_provisional"],
        },
        "cells": per_cell,
        "cost_totals": {
            "original_reported_usd": round(sum(
                c["cost"]["original_reported_usd"] for c in per_cell.values()), 6),
            "replacement_reported_usd": round(sum(
                c["cost"]["replacement_reported_usd"] for c in per_cell.values()), 6),
            "typesafe_arc_tariff_derived_usd": 1.160391372,
            "typesafe_arc_plan_worstcase_usd": 2.09,
            "typesafe_note": ("usage x tariff (choice 13,910,533 tok = "
                              "$0.584242386 + score 13,717,833 tok = "
                              "$0.576148986 at $0.042/M); NOT invoice-verified; "
                              "budget keeps the 2.09 conservative allocation"),
        },
        "publication": ("public aggregate: counts/accuracy/cost/flags only; no "
                        "gold keys, no licensed text, no raw model output"),
    }
    source_map = {
        "schema": "active-source-map-1.0.0",
        "version": "v4r1",
        "selector": "run_matched3.cmd_score cell-level merge (v3-preferred)",
        "sources": {
            "original_v2": "runs_matched_cheap/v2/results.jsonl",
            "original_v3": "runs_matched_cheap/v3/results.jsonl",
            "salvage_raws": "runs_matched_cheap/raw2/ (complete excerpts only)",
            "replacements": "runs_matched_cheap/v4r1/results.jsonl",
            "transport_recovery": "runs_matched_cheap/v4r1/transport_recovery.jsonl (+ pointers)",
            "replacement_raws": "runs_matched_cheap/v4r1/private_raw/ (never published)",
            "jev_scores": {ds: p for ds, p in {
                "mmlu": "runs_benchmark/bench-mmlu_full-*/derived/score.json",
                "gpqa": "runs_benchmark_ext2/bench-gpqa_diamond-*/derived/score.json",
                "math500_choice": "runs_benchmark_ext/bench-math500_choice-*/derived/score.json",
                "hle_text_mc": "runs_benchmark_ext2/bench-hle_text_mc-*/derived/score.json",
                "arc": "runs_benchmark/bench-arc_test-*/derived/score.json"}.items()},
            "arc_corrected": ("runs_benchmark_ext_rerun1/"
                              "bench-arc_agi2_{choice,score}-d98f20eb693e"),
        },
        "cells": {cell: {
            "active_source_version": b["active_source_version"],
            "original_wire_protocol": b["original_wire_protocol"],
            "status": b["status"],
            "n_requested": b["n_requested"],
            "counts": b["counts"],
        } for cell, b in per_cell.items()},
        "per_item_file": "data_report/baselines/v4r1/active_items.jsonl",
    }
    return {"summary": summary, "source_map": source_map, "items": items_out}


def main() -> int:
    out = build()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "public_summary.json").write_text(
        json.dumps(out["summary"], indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8")
    (OUT_DIR / "active_source_map.json").write_text(
        json.dumps(out["source_map"], indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8")
    with (OUT_DIR / "active_items.jsonl").open("w", encoding="utf-8") as fh:
        for rec in out["items"]:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    t = out["summary"]["totals"]
    print(f"[summary] cells {t['n_cells']} (complete {t['n_complete_cells']}, "
          f"partial {t['n_partial_cells']}); rows requested {t['n_requested']}; "
          f"pending {t['n_pending_replacements']}; provisional "
          f"{t['n_provisional_transport']}")
    print(f"[summary] wrote {OUT_DIR.relative_to(ROOT)}/public_summary.json, "
          f"active_source_map.json, active_items.jsonl ({len(out['items'])} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
