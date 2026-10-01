#!/usr/bin/env python3
"""Freeze the ACTIVE-cell minimum rerun plan with cost estimates (offline).

Scope rule (supervisor-corrected): the published summary_v3 selection is a
MERGED cell-level selector — a COMPLETE v3 cell supersedes its v2 history; an
incomplete v3 cell falls back to the complete v2 cell (labeled); cells never
re-run in v3 keep v2 (granite/*, qwen3.8-max mmlu).  Repair reruns cover ONLY
rows that feed ACTIVE publication cells.  Superseded historical v2 rows are
immutable evidence: labeled uncorrected, never rerun.

The selector below mirrors scripts/benchmark/run_matched3.py cmd_score
exactly and is validated against summary_v3.json's per-cell protocol labels.

Outputs:
  data_report/baseline_rerun_tasks.jsonl       unique
      (source_version|model|dataset|item_id) tasks, each with its ORIGINAL
      recorded wire (replacements keep the original protocol)
  data_report/baseline_rerun_frozen_plan.json  ACTIVE-only counts, current
      provider price metadata, per-cell observed + expected costs,
      conservative pre-call reservations, minimality audit, priority scope
      composition and the mixed-time selective-replacement protocol
  data_report/provider_price_metadata.json     current public /models prices
      (metadata GET only — never a model call, no credentials)

No model calls, no secrets.  Original run evidence is read-only.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import json
import statistics
import collections
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data_report"
RAW2_DIR = ROOT / "runs_matched_cheap" / "raw2"
V2_RESULTS = ROOT / "runs_matched_cheap" / "v2" / "results.jsonl"
V3_RESULTS = ROOT / "runs_matched_cheap" / "v3" / "results.jsonl"
V3_FREEZE = ROOT / "runs_matched_cheap" / "v3" / "freeze_v3.json"
SUMMARY_V3 = ROOT / "runs_matched_cheap" / "v3" / "summary_v3.json"
PRICES_OUT = OUT_DIR / "provider_price_metadata.json"
TASKS_OUT = OUT_DIR / "baseline_rerun_tasks.jsonl"
PLAN_OUT = OUT_DIR / "baseline_rerun_frozen_plan.json"
MODELS_URL = "https://openrouter.ai/api/v1/models"

V2_CONTENT_LIMIT = 400
V2_REASONING_LIMIT = 300
PARSER_STABLE_STAGES = ("exact", "stripped", "stripped_char")
TRANSPORT_ATTEMPTS = 3
RESERVATION_SAFETY_INPUT_TOKENS = 4096
BUDGET_CAP_USD = 10.0
ARC_CHOICE_SCORE_CONSERVATIVE_USD = 1.015868 + 1.074609   # from ext plan.json
ARC_RESERVED_USD = 2.50        # hard allocation for the Typesafe ARC rerun


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


def _cells(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    out: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for row in rows:
        out[(str(row["model"]), str(row["dataset"]))].append(row)
    return dict(out)


def active_cell_selector(v2_rows: list[dict], v3_rows: list[dict],
                         n_req: dict[str, int],
                         v3_models: set[str]) -> dict[tuple[str, str], tuple[str, list[dict]]]:
    """Exact mirror of run_matched3.cmd_score's cell-level merge:
    complete v3 cell supersedes v2; incomplete v3 falls back to complete v2
    (labeled); v2-only cells survive for models never re-run in v3 (and the
    qwen3.8-max mmlu carryover)."""
    c2, c3 = _cells(v2_rows), _cells(v3_rows)
    active: dict[tuple[str, str], tuple[str, list[dict]]] = {}
    for key, rs in c3.items():
        need = n_req.get(key[1], len(rs))
        ok3 = sum(1 for r in rs if r.get("status") == "ok")
        ok2 = sum(1 for r in c2.get(key, []) if r.get("status") == "ok")
        if ok3 >= need or key not in c2 or ok2 < need:
            active[key] = ("v3", rs)
        else:
            active[key] = ("v2", c2[key])
    for key, rs in c2.items():
        if key not in c3 and (key[0] not in v3_models
                              or key == ("qwen/qwen3.8-max-0902", "mmlu")):
            active[key] = ("v2", rs)
    return active


def _rerun_class(row: dict, version: str, raws: dict[str, dict]) -> str:
    """rerun / salvage / keep_original / settled_non_ok for ONE active row."""
    if not row.get("terminal"):
        return "non_terminal"
    if row.get("status") != "ok":
        return "settled_non_ok"     # wrong under all-requested, parser-invariant
    stage = str(row.get("recovery_stage"))
    if stage in PARSER_STABLE_STAGES:
        return "keep_original"      # selection preserved; stored answer stands
    if version == "v2" and _raw_complete(raws.get(str(row.get("logical_request_id")))):
        return "salvage"            # fixed parser can be applied offline
    return "rerun"


def fetch_prices() -> tuple[dict[str, dict[str, float]], dict]:
    """CURRENT provider-reported price metadata (public catalog GET — never a
    model call, no credentials used)."""
    req = urllib.request.Request(
        MODELS_URL, headers={"User-Agent": "jev-benchmark-planner/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    prices: dict[str, dict[str, float]] = {}
    for entry in body.get("data", []):
        pricing = entry.get("pricing") or {}
        try:
            prices[str(entry["id"])] = {
                "input_per_token": float(pricing["prompt"]),
                "output_per_token": float(pricing["completion"])}
        except (KeyError, TypeError, ValueError):
            continue
    meta = {"source": "openrouter /models public catalog (metadata GET)",
            "url": MODELS_URL, "n_models": len(prices)}
    return prices, meta


def main() -> int:
    raws = _raw2_index()
    v2_rows = [r for r in _read_jsonl(V2_RESULTS) if r.get("terminal")]
    v3_rows = [r for r in _read_jsonl(V3_RESULTS) if r.get("terminal")]
    v3_plan = json.loads(V3_FREEZE.read_text(encoding="utf-8"))["models"]
    v3_models = {m["id"] for m in v3_plan}

    sys.path.insert(0, str(ROOT / "scripts" / "benchmark"))
    import run_cheap_matched2 as rcm2  # noqa: E402  (offline item loader)
    n_req = {ds: len(v) for ds, v in rcm2.load_items().items()}

    active = active_cell_selector(v2_rows, v3_rows, n_req, v3_models)

    # validate selector against the published summary_v3 protocol labels
    summary = json.loads(SUMMARY_V3.read_text(encoding="utf-8"))["models"]
    mismatches = []
    for (model, dataset), (version, _rows) in active.items():
        label = ((summary.get(model) or {}).get(dataset) or {}).get("protocol", "")
        expect_v3 = label.startswith("provider")
        if expect_v3 != (version == "v3"):
            mismatches.append((model, dataset, version, label))
    if mismatches:
        raise SystemExit(f"selector mismatch vs summary_v3: {mismatches[:5]}")

    prices, price_meta = fetch_prices()
    PRICES_OUT.write_text(
        json.dumps({"meta": price_meta, "prices": prices}, indent=1) + "\n",
        encoding="utf-8")

    # ---------------------------------------------- per-cell classification
    tasks_out = []
    frozen_cells = []
    superseded_v2 = []
    counts = collections.Counter()
    for key in sorted(active):
        model, dataset = key
        version, rows = active[key]
        rows_by_class: dict[str, list[dict]] = collections.defaultdict(list)
        for row in rows:
            rows_by_class[_rerun_class(row, version, raws)].append(row)
        for klass, rs in rows_by_class.items():
            counts[klass] += len(rs)
        rerun_rows = rows_by_class.get("rerun", [])
        s_costs = [float(r["cost_reported"]) for r in rows
                   if isinstance(r.get("cost_reported"), (int, float))
                   and not isinstance(r.get("cost_reported"), bool)]
        s_ins = [r["usage_in"] for r in rows if isinstance(r.get("usage_in"), int)]
        s_outs = [r["usage_out"] for r in rows if isinstance(r.get("usage_out"), int)]
        wire_max = max((int((r.get("wire") or {}).get("max_output_tokens") or 0)
                        for r in rows), default=0) or 4096
        rate = prices.get(model)
        if rate is None:
            raise SystemExit(f"[price] current provider metadata lacks {model!r}; "
                             f"refusing to plan with non-actual prices")
        per_call_max = (RESERVATION_SAFETY_INPUT_TOKENS * rate["input_per_token"]
                        + wire_max * rate["output_per_token"])
        mean_cost = statistics.mean(s_costs) if s_costs else None
        frozen_cells.append({
            "model": model, "dataset": dataset,
            "active_source_version": version,
            "n_active_rows": len(rows),
            "n_keep_original": len(rows_by_class.get("keep_original", [])),
            "n_settled_non_ok": len(rows_by_class.get("settled_non_ok", [])),
            "n_offline_salvage": len(rows_by_class.get("salvage", [])),
            "n_rows_to_rerun": len(rerun_rows),
            "observed_mean_reported_cost_usd": (round(mean_cost, 6)
                                                if mean_cost is not None else None),
            "expected_cost_usd": (round(mean_cost * len(rerun_rows), 4)
                                  if mean_cost is not None and rerun_rows else 0.0),
            "wire_max_output_tokens": wire_max,
            "reservation_per_task_usd": round(TRANSPORT_ATTEMPTS * per_call_max, 4),
        })
        for row in rerun_rows:
            wire = row.get("wire") or {}
            item_id = str(row["item_id"])
            tasks_out.append({
                "logical_request_id": f"{version}|{model}|{dataset}|{item_id}",
                "model": model, "dataset": dataset, "item_id": item_id,
                "source_version": version,
                "source_recovery_stage": str(row.get("recovery_stage")),
                "wire": {"model": model,
                         "reasoning_effort": wire.get("reasoning_effort"),
                         "max_output_tokens": wire.get("max_output_tokens"),
                         "extra_body": wire.get("extra_body")},
            })
        # superseded v2 history for v3-active cells: immutable, labeled only
        if version == "v3":
            v2_cell_rows = [r for r in v2_rows
                            if (str(r["model"]), str(r["dataset"])) == key]
            if v2_cell_rows:
                superseded_v2.append({
                    "model": model, "dataset": dataset,
                    "n_rows": len(v2_cell_rows),
                    "label": "uncorrected-superseded (v2 history kept immutable; "
                             "active cell is v3 — NOT rerun)"})

    # unique task ids (source_version|model|dataset|item) — assert
    ids = [t["logical_request_id"] for t in tasks_out]
    assert len(ids) == len(set(ids)), "task ids must be unique"

    total_rows = sum(c["n_rows_to_rerun"] for c in frozen_cells)
    total_expected = sum(c["expected_cost_usd"] or 0.0 for c in frozen_cells)
    by_model: dict[str, dict] = collections.defaultdict(
        lambda: {"n_rows_to_rerun": 0, "expected_cost_usd": 0.0, "cells": []})
    for c in frozen_cells:
        if not c["n_rows_to_rerun"]:
            continue
        m = by_model[c["model"]]
        m["n_rows_to_rerun"] += c["n_rows_to_rerun"]
        m["expected_cost_usd"] = round(m["expected_cost_usd"]
                                       + (c["expected_cost_usd"] or 0.0), 4)
        m["cells"].append(f"{c['dataset']}:{c['active_source_version']}:"
                          f"{c['n_rows_to_rerun']}")

    # ---------------------------------------- priority scope composition
    math_cells = [c for c in frozen_cells
                  if c["dataset"] == "math500_choice" and c["n_rows_to_rerun"]]
    math_expected = round(sum(c["expected_cost_usd"] or 0 for c in math_cells), 4)
    math_rows = sum(c["n_rows_to_rerun"] for c in math_cells)
    rest_cells = sorted((c for c in frozen_cells
                         if c["dataset"] != "math500_choice" and c["n_rows_to_rerun"]),
                        key=lambda c: (c["expected_cost_usd"] or 0.0))
    remainder_budget = round(BUDGET_CAP_USD - ARC_RESERVED_USD - math_expected, 4)
    affordable, deferred = [], []
    running = 0.0
    for c in rest_cells:
        if running + (c["expected_cost_usd"] or 0.0) <= remainder_budget:
            affordable.append(c)
            running += (c["expected_cost_usd"] or 0.0)
        else:
            deferred.append(c)

    plan = {
        "frozen_plan_version": "baseline-rerun-frozen-active-1.0.0",
        "generated_by": "scripts/benchmark/freeze_rerun_plan.py (offline; ACTIVE selector)",
        "corrections_vs_prior_draft": [
            "scope is ACTIVE publication cells only (merged selector: complete "
            "v3 supersedes v2; incomplete v3 falls back to complete v2; v2-only "
            "cells = granite/* + qwen3.8-max mmlu carryovers)",
            "superseded v2 history is labeled uncorrected-superseded, never rerun",
            "task ids are unique (source_version|model|dataset|item_id)",
            "prices are CURRENT provider-reported catalog metadata "
            "(metadata GET; the earlier fallback-20x-observed heuristic is "
            "NOT actual worst-case pricing and is withdrawn)",
            "prior message arithmetic corrections: 15.62-4.92-4.44+2.09 = 8.35 "
            "(not 9.90); 8825-1307-1369 = 6149 (not 6115) — both moot: the "
            "ACTIVE-only totals below supersede them",
        ],
        "active_selector": {
            "definition": "mirror of run_matched3.cmd_score cell-level merge",
            "validated_against": "runs_matched_cheap/v3/summary_v3.json per-cell protocol labels",
            "n_active_cells": len(active),
            "n_v3_cells": sum(1 for v, _ in active.values() if v == "v3"),
            "n_v2_carryover_cells": sum(1 for v, _ in active.values() if v == "v2"),
            "v2_carryover_cells": sorted(f"{m}:{d}" for (m, d), (v, _) in active.items()
                                         if v == "v2"),
            "selector_mismatches_vs_summary_v3": 0,
        },
        "minimality_audit": {
            "method": ("per ACTIVE row: rerun iff status-ok AND stored recovery "
                       "stage not parser-stable AND no complete raw evidence in "
                       "the active source; stored exact rows keep their answer "
                       "(even wrong ones — selection changes only where the "
                       "parser could change the answer)"),
            "n_keep_original_rows": counts["keep_original"],
            "n_settled_non_ok_rows": counts["settled_non_ok"],
            "n_offline_salvage_rows": counts["salvage"],
            "n_rerun_rows_active": total_rows,
            "note": ("historical v2 rows inside v3-active cells are superseded "
                     "evidence, labeled and never rerun"),
        },
        "superseded_v2_history": superseded_v2,
        "rerun_cells": frozen_cells,
        "rerun_total_rows": total_rows,
        "expected_total_cost_usd": round(total_expected, 2),
        "mixed_time_selective_replacement_protocol": {
            "statement": ("complete-cell summaries combine ORIGINAL parser-stable "
                          "rows kept from the active source run, offline-salvaged "
                          "rows (fixed parser over complete v2 raws), and "
                          "versioned live REPLACEMENTS — selective, mixed-time: "
                          "each replacement reuses its original row's recorded "
                          "wire; provider drift between times is uncontrolled"),
            "required_summary_fields": [
                "original stable exact rows plus justified versioned replacements",
                "actual gold join on item_id (never re-labeled)",
                "paired Jev metrics on the shared item set",
                "complete denominator (all active rows), never accuracy on the rerun subset alone",
            ],
        },
        "reservation_policy": {
            "mechanism": ("revision_run.PersistentBudget: atomic PRE-CALL "
                          "reservations against one persistent USD ledger "
                          "(settled + held + in-flight worst-case <= cap); "
                          "bounded retries inside the per-task reservation; "
                          "unknown billed cost FAILS CLOSED (reservation held)"),
            "transport_attempts_per_request": TRANSPORT_ATTEMPTS,
            "reservation_per_task_usd_formula": (
                "3 * (4096 * input_per_token + wire_max_output_tokens * "
                "output_per_token) with CURRENT provider prices"),
            "price_metadata": str(PRICES_OUT.relative_to(ROOT)),
        },
        "priority_scope": {
            "budget_cap_usd": BUDGET_CAP_USD,
            "arc_reserved_usd": ARC_RESERVED_USD,
            "arc_choice_score": {
                "conservative_cost_usd": round(ARC_CHOICE_SCORE_CONSERVATIVE_USD, 2),
                "n_requests": 517 + 517,
                "provider": "typesafe (jev-1.13.0 REMOTE system-one API)",
                "first": True},
            "priority_1_active_math_cells": {
                "cells": [f"{c['model']}:{c['active_source_version']}:"
                          f"{c['n_rows_to_rerun']}" for c in math_cells],
                "n_rows": math_rows, "expected_usd": math_expected},
            "priority_2_cheapest_first_complete_cells": {
                "cells": [f"{c['model']}:{c['dataset']}:{c['n_rows_to_rerun']}"
                          for c in affordable],
                "n_rows": sum(c["n_rows_to_rerun"] for c in affordable),
                "expected_usd": round(running, 4)},
            "deferred_needs_more_budget": {
                "cells": [f"{c['model']}:{c['dataset']}:{c['n_rows_to_rerun']}"
                          for c in deferred],
                "n_rows": sum(c["n_rows_to_rerun"] for c in deferred),
                "expected_usd": round(sum(c["expected_cost_usd"] or 0
                                          for c in deferred), 4)},
        },
        "by_model": dict(sorted(by_model.items())),
        "sources": {
            "v2": "runs_matched_cheap/v2/results.jsonl (raw excerpts truncated)",
            "v3": "runs_matched_cheap/v3/results.jsonl (no raws)",
            "selector": "scripts/benchmark/run_matched3.py cmd_score",
            "summary": "runs_matched_cheap/v3/summary_v3.json",
            "tasks": "data_report/baseline_rerun_tasks.jsonl",
        },
    }

    TASKS_OUT.write_text(
        "".join(json.dumps(t, ensure_ascii=False) + "\n" for t in tasks_out),
        encoding="utf-8")
    PLAN_OUT.write_text(json.dumps(plan, indent=1, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(f"[freeze-active] {total_rows} ACTIVE rerun rows / "
          f"{sum(1 for c in frozen_cells if c['n_rows_to_rerun'])} cells; "
          f"expected ${total_expected:.2f}; salvage {counts['salvage']}; "
          f"keep {counts['keep_original']}; settled-non-ok "
          f"{counts['settled_non_ok']}")
    print(f"[freeze-active] math priority {math_rows} rows ${math_expected}; "
          f"cheapest-first {sum(c['n_rows_to_rerun'] for c in affordable)} rows "
          f"${running:.2f}; deferred {sum(c['n_rows_to_rerun'] for c in deferred)} "
          f"rows ${sum(c['expected_cost_usd'] or 0 for c in deferred):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
