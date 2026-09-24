"""Offline matched-comparison scorer and report-data builder.

Runs entirely on stored artifacts — never a model call, never a credential.
Headline accounting uses the ALL-REQUESTED denominator: a model's accuracy
over a dataset counts EVERY requested item, with strict-format failures,
errors and missing terminal results counted as not-correct and reported
separately. Conditional accuracy (over valid outputs only) is a DISTINCT
column. Native Jev full-suite scores and matched-subset scores are DISTINCT
records and never averaged together.

Paired analysis: model-minus-Jev per matched item (only items where BOTH
sides produced a usable output enter the paired test; exclusions are
counted), exact McNemar on discordant pairs, paired cluster bootstrap of
the accuracy difference, Holm step-down across the fixed 9-comparison
family per benchmark, and a predeclared practical-parity tolerance of
3 percentage points (equivalence requires the ENTIRE paired interval
within ±0.03; small fresh-n comparisons may remain imprecise).

Text baselines never receive synthesized probability arrays: a probability
array on the modern side is a hard error, and proper-score metrics are
structurally empty for text baselines (explicitly excluded, never invented).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from .matched import (
    DATASETS,
    JEV_NOUL_THRESHOLD,
    MATCHED_ROOT_DEFAULT,
    NOUL_CONVERSION_LABEL,
    load_matched_freeze,
    run_id_for,
    split_logical_id,
)
from .mech_dryrun import holm_stepdown
from .metrics import wilson_interval
from .stats import paired_difference_cluster_bootstrap, stratified_accuracy

EQUIVALENCE_TOLERANCE = 0.03   # predeclared practical-parity tolerance
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260920
FINITE_SET_CAVEAT = (
    "Wilson intervals are item-sampling diagnostics within this fixed, finite "
    "matched item set; they are not superpopulation claims."
)
ERROR_STATUSES = frozenset({"http_error", "transport_error", "malformed_json", "timeout"})


class MatchedReportError(RuntimeError):
    pass


# ------------------------------------------------------------------- loaders
def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise MatchedReportError(f"missing artifact {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n")
            if line.strip()]


def gold_records(items: list[dict[str, Any]], dataset: str) -> dict[str, dict[str, Any]]:
    """item_id -> gold record. Missing gold is a HARD error (isolation: gold is
    joined ONLY here, offline, and never appears in any prediction record)."""
    out: dict[str, dict[str, Any]] = {}
    for item in items:
        item_id = str(item["id"])
        gold_block = item.get("gold") or {}
        value = None
        if isinstance(gold_block, dict):
            for payload in gold_block.values():
                if isinstance(payload, dict) and payload.get("value") is not None:
                    value = payload["value"]
                    break
        if value is None:
            raise MatchedReportError(
                f"{dataset} item {item_id}: no gold value; refusing to score")
        qtype = next(iter(item["questions"].values()))["type"]
        is_noul = qtype == "noul" or isinstance(value, bool)
        out[item_id] = {
            "item_id": item_id,
            "group": str(item.get("group", "unknown")),
            "cluster": str(item.get("cluster", item_id)),
            "value": ("Yes" if value is True else "No") if is_noul else str(value),
            "native_value": value,
            "is_noul": is_noul,
            "label_conversion": NOUL_CONVERSION_LABEL if is_noul else None,
        }
    return out


def jev_correct_flags(
    records: dict[str, dict[str, Any]], gold: dict[str, dict[str, Any]],
) -> tuple[dict[str, bool], dict[str, str]]:
    """Jev-side correctness per item, with the exclusion reason per invalid item.

    Choice items: usable prediction whose choice equals the gold source key.
    Noul items: usable native Noul probability thresholded at the recorded
    JEV_NOUL_THRESHOLD = 0.5 — a DISTINCT encoding from the baseline Yes/No
    choice; recorded separately, never silently merged.
    """
    flags: dict[str, bool] = {}
    reasons: dict[str, str] = {}
    for item_id, g in gold.items():
        record = records.get(item_id)
        if record is None:
            reasons[item_id] = "missing_jev_result"
            continue
        if record.get("status") != "ok" or record.get("terminal") is not True:
            reasons[item_id] = f"jev_status_{record.get('status')}"
            continue
        predictions = record.get("predictions") or {}
        prediction = next(iter(predictions.values()), None) if predictions else {}
        if not prediction or not prediction.get("usable"):
            reasons[item_id] = "jev_prediction_unusable"
            continue
        if prediction.get("type") == "noul" or g["is_noul"]:
            p = prediction.get("noul")
            if not isinstance(p, (int, float)) or isinstance(p, bool):
                reasons[item_id] = "jev_noul_probability_missing"
                continue
            flags[item_id] = (float(p) >= JEV_NOUL_THRESHOLD) == bool(g["native_value"])
        else:
            choice = prediction.get("choice")
            if not isinstance(choice, str):
                reasons[item_id] = "jev_choice_missing"
                continue
            flags[item_id] = choice == g["value"]
    return flags, reasons


def modern_outcomes(
    results: list[dict[str, Any]], gold: dict[str, dict[str, Any]], dataset: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Modern-side per-item outcome records for one dataset, keyed by item_id."""
    outcomes: dict[str, dict[str, Any]] = {}
    reasons: dict[str, str] = {}
    for record in results:
        if record.get("terminal") is not True:
            continue
        model, dataset_key, item_id = split_logical_id(str(record["logical_request_id"]))
        if dataset_key != dataset:
            continue
        if item_id not in gold:
            raise MatchedReportError(
                f"result {record['logical_request_id']}: item not in the frozen gold "
                "set; refusing to score")
        entry: dict[str, Any] = {
            "logical_request_id": record["logical_request_id"],
            "item_id": item_id,
            "group": gold[item_id]["group"],
            "status": str(record.get("status")),
            "model_returned": record.get("model_returned"),
            "choice": None,
            "correct": None,
            "excluded_reason": None,
        }
        if entry["status"] == "ok":
            predictions = record.get("predictions") or {}
            prediction = next(iter(predictions.values()), None) if predictions else {}
            if not prediction:
                entry["excluded_reason"] = "prediction_missing"
                reasons[item_id] = entry["excluded_reason"]
            elif not prediction.get("usable"):
                entry["excluded_reason"] = "prediction_unusable"
                reasons[item_id] = entry["excluded_reason"]
            elif prediction.get("probabilities"):
                # a text baseline carries NO probabilities; a synthesized array
                # here is a contract violation, never a calibration finding
                raise MatchedReportError(
                    f"{record['logical_request_id']}: text baseline returned a "
                    "probability array; refusing to score synthesized distributions")
            else:
                choice = prediction.get("choice")
                if not isinstance(choice, str):
                    entry["excluded_reason"] = "choice_missing"
                    reasons[item_id] = entry["excluded_reason"]
                else:
                    entry["choice"] = choice
                    entry["correct"] = choice == gold[item_id]["value"]
        else:
            entry["excluded_reason"] = f"status_{entry['status']}"
            reasons[item_id] = entry["excluded_reason"]
        # EVERY terminal outcome is recorded — failures and errors included —
        # so the all-request denominator is structurally complete
        outcomes[item_id] = entry
    return outcomes, reasons


# ------------------------------------------------------------------- scoring
def score_model_dataset(
    run_dir: Path, freeze: dict[str, Any], model: str, dataset: str, *,
    prices: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """One model x one dataset: counts, ALL-REQUESTED headline, conditional
    accuracy, per-group breakdown, cost and latency summaries."""
    gold = gold_records(freeze["datasets"][dataset]["items"], dataset)
    n_requested = len(gold)
    results = [r for r in _read_jsonl(Path(run_dir) / "results.jsonl")
               if r.get("terminal") is True
               and split_logical_id(str(r["logical_request_id"]))[1] == dataset]
    attempts_path = Path(run_dir) / "attempts.jsonl"
    attempts = ([a for a in _read_jsonl(attempts_path)
                 if split_logical_id(str(a["logical_request_id"]))[1] == dataset]
                if attempts_path.exists() else [])
    outcomes, reasons = modern_outcomes(results, gold, dataset)

    valid = [e for e in outcomes.values() if e["correct"] is not None]
    format_failed = [e for e in outcomes.values()
                     if e["status"] == "contract_invalid"
                     or e["excluded_reason"] in ("prediction_unusable", "choice_missing")]
    errors = [e for e in outcomes.values() if e["status"] in ERROR_STATUSES]
    n_valid = len(valid)
    n_correct_valid = sum(1 for e in valid if e["correct"])
    n_missing = n_requested - len(outcomes)

    headline: dict[str, Any] = {
        "n_requested": n_requested,
        "n_terminal": len(outcomes),
        "n_valid": n_valid,
        "n_format_failed": len(format_failed),
        "n_error": len(errors),
        "n_missing": n_missing,
        "correct_all_denominator": n_requested,
        "correct_all": _correct_all(outcomes, n_requested),
        "accuracy_all": _accuracy_all(outcomes, n_requested),
        "wilson_all": None,
        "accuracy_conditional": (n_correct_valid / n_valid) if n_valid else None,
        "wilson_conditional": wilson_interval(n_correct_valid, n_valid).to_dict()
        if n_valid else None,
        "finite_set_caveat": FINITE_SET_CAVEAT,
        "denominator_note": ("headline accuracy divides by ALL requested items; "
                             "format failures, errors and missing results count as "
                             "not-correct and are reported separately"),
    }
    if n_requested:
        headline["wilson_all"] = wilson_interval(headline["correct_all"],
                                                 n_requested).to_dict()
    if dataset == "mmlu" and outcomes:
        population_sizes = freeze["datasets"]["mmlu"]["sampling"]["population_sizes"]
        strata = [gold[e["item_id"]]["group"] for e in outcomes.values()]
        correct_series = [1.0 if e["correct"] else 0.0 for e in outcomes.values()]
        weighted = stratified_accuracy(correct_series, strata, population_sizes)
        headline["accuracy_weighted_to_population"] = weighted["estimate"]
        headline["weighted_note"] = weighted.get("note", (
            "stratum weights N_h/n_h target the full TEST population; recorded "
            "inclusion probabilities are in the freeze"))
        headline["weights"] = weighted["weights"]

    status_counts: dict[str, int] = {}
    for e in outcomes.values():
        status_counts[e["status"]] = status_counts.get(e["status"], 0) + 1
    headline["status_counts"] = status_counts

    by_group: dict[str, dict[str, Any]] = {}
    for e in outcomes.values():
        bucket = by_group.setdefault(e["group"], {"n_valid": 0, "correct_valid": 0})
        if e["correct"] is not None:
            bucket["n_valid"] += 1
            bucket["correct_valid"] += 1 if e["correct"] else 0
    for bucket in by_group.values():
        bucket["accuracy_conditional"] = (bucket["correct_valid"] / bucket["n_valid"]
                                          if bucket["n_valid"] else None)
        bucket["wilson_conditional"] = (
            wilson_interval(bucket["correct_valid"], bucket["n_valid"]).to_dict()
            if bucket["n_valid"] else None)

    latencies = [float(r["latency_ms_total"]) for r in results
                 if isinstance(r.get("latency_ms_total"), (int, float))]
    return {
        "model": model,
        "dataset": dataset,
        "run_dir": str(run_dir),
        "headline": headline,
        "by_group": dict(sorted(by_group.items())),
        "cost": _cost_summary(attempts, prices, model),
        "latency": {
            "n": len(latencies),
            "mean_ms": round(sum(latencies) / len(latencies), 1) if latencies else None,
            "p95_ms": _percentile(latencies, 95) if latencies else None,
            "max_ms": max(latencies) if latencies else None,
        },
        "usage_totals": {
            "input_tokens": _sum_int(attempts, "usage_input_tokens"),
            "output_tokens": _sum_int(attempts, "usage_output_tokens"),
            "reasoning_tokens": _reasoning_tokens(attempts),
            "attempts_without_usage": len([a for a in attempts
                                           if a.get("usage_input_tokens") is None]),
            "finish_reasons": _finish_reason_counts(attempts),
            "model_returned_values": sorted({str(r.get("model_returned")) for r in results
                                             if r.get("model_returned")}),
        },
        "per_item": sorted(outcomes.values(), key=lambda e: e["item_id"]),
    }


def _correct_all(outcomes: dict[str, dict[str, Any]], n_requested: int) -> int:
    return sum(1 for e in outcomes.values() if e["correct"] is True)


def _accuracy_all(outcomes: dict[str, dict[str, Any]], n_requested: int) -> float | None:
    return (sum(1 for e in outcomes.values() if e["correct"]) / n_requested) if n_requested else None


def _cost_summary(attempts: list[dict[str, Any]], prices: dict[str, dict[str, float]],
                  model: str) -> dict[str, Any]:
    from .matched import call_cost_usd

    reported = 0.0
    estimated = 0.0
    n_reported = n_estimated = n_unknown = 0
    for attempt in attempts:
        usage_raw = (attempt.get("extra") or {}).get("usage_raw") \
            if isinstance(attempt.get("extra"), dict) else None
        rep, est = call_cost_usd(prices, model,
                                 input_tokens=attempt.get("usage_input_tokens"),
                                 output_tokens=attempt.get("usage_output_tokens"),
                                 usage_raw=usage_raw)
        if rep is not None:
            reported += rep
            n_reported += 1
        elif est is not None:
            estimated += est
            n_estimated += 1
        else:
            n_unknown += 1
    return {
        "billed_reported_usd": round(reported, 6),
        "billed_estimated_usd": round(estimated, 6),
        "n_attempts_with_reported_cost": n_reported,
        "n_attempts_catalog_estimated": n_estimated,
        "n_unknown_usage": n_unknown,
        "note": ("provider-reported cost (usage_raw.cost) is the billing authority; "
                 "attempts without reported cost are catalog-priced from actual "
                 "tokens; attempts with no usage at all stay unknown and are "
                 "counted here, never treated as free"),
    }


def _sum_int(records: list[dict[str, Any]], field: str) -> int | None:
    values = [r.get(field) for r in records if isinstance(r.get(field), int)]
    return sum(values) if values else None


def _reasoning_tokens(attempts: list[dict[str, Any]]) -> int | None:
    """Sum provider-reported reasoning tokens from raw usage details (None if absent)."""
    total = 0
    seen = False
    for attempt in attempts:
        usage_raw = (attempt.get("extra") or {}).get("usage_raw") \
            if isinstance(attempt.get("extra"), dict) else None
        details = usage_raw.get("details") if isinstance(usage_raw, dict) else None
        if not isinstance(details, dict):
            continue
        for key, value in details.items():
            if "reasoning" in str(key) and isinstance(value, (int, float)) \
                    and not isinstance(value, bool):
                total += int(value)
                seen = True
    return total if seen else None


def _finish_reason_counts(attempts: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for attempt in attempts:
        extra = attempt.get("extra") if isinstance(attempt.get("extra"), dict) else None
        reason = (extra or {}).get("finish_reason")
        key = str(reason)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q / 100.0 * (len(ordered) - 1)))))
    return ordered[index]


# -------------------------------------------------------------------- paired
def paired_analysis(
    modern_flags: dict[str, bool], jev_flags: dict[str, bool],
    gold: dict[str, dict[str, Any]], *,
    n_resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Model-minus-Jev paired diagnostics over items BOTH sides answered validly."""
    from .paired import mcnemar_exact

    paired_items = sorted(set(modern_flags) & set(jev_flags))
    correct_m = [modern_flags[i] for i in paired_items]
    correct_j = [jev_flags[i] for i in paired_items]
    clusters = [gold[i]["cluster"] for i in paired_items]
    b01 = sum(1 for m, j in zip(correct_m, correct_j) if not m and j)   # jev right, model wrong
    b10 = sum(1 for m, j in zip(correct_m, correct_j) if m and not j)   # model right, jev wrong
    ties = sum(1 for m, j in zip(correct_m, correct_j) if m == j)
    doc: dict[str, Any] = {
        "paired_n": len(paired_items),
        "modern_only_valid": len(set(modern_flags) - set(jev_flags)),
        "jev_only_valid": len(set(jev_flags) - set(modern_flags)),
        "excluded_modern": len(set(gold) - set(modern_flags)),
        "excluded_jev": len(set(gold) - set(jev_flags)),
        "b01_jev_right_model_wrong": b01,
        "b10_model_right_jev_wrong": b10,
        "ties_concordant": ties,
        "point_diff_model_minus_jev": (
            (sum(correct_m) - sum(correct_j)) / len(paired_items)) if paired_items else None,
        "mcnemar_exact_p": mcnemar_exact(b01, b10),
        "bootstrap": None,
        "equivalent_within_3pp": None,
        "equiv_tolerance": EQUIVALENCE_TOLERANCE,
        "note": ("paired set = items with a usable output on BOTH sides; headline "
                 "denominators remain ALL requested items; a failure to reject a "
                 "difference is never a universal parity claim — equivalence needs "
                 "the ENTIRE paired interval inside ±0.03; small fresh-n cells may "
                 "remain imprecise"),
    }
    if len(paired_items) >= 2:
        try:
            if len(set(clusters)) < 2:
                raise ValueError("fewer than two clusters")
            boot = paired_difference_cluster_bootstrap(
                correct_m, correct_j, clusters, n_resamples=n_resamples, seed=seed)
            boot_doc = boot.to_dict()
        except ValueError:
            # small sets can collapse to a single cluster; fall back to
            # item-level resampling, explicitly recorded
            diffs = [(1.0 if m else 0.0) - (1.0 if j else 0.0)
                     for m, j in zip(correct_m, correct_j)]
            point = sum(diffs) / len(diffs)
            boot_doc = _percentile_bootstrap(point, _sorted_sample_means(
                diffs, random.Random(seed), n_resamples))
            boot_doc["method"] = "item_percentile_bootstrap_single_cluster"
        doc["bootstrap"] = boot_doc
        lo, hi = boot_doc["ci95_low"], boot_doc["ci95_high"]
        doc["difference_uncertain"] = bool(lo < 0.0 < hi)
        doc["equivalent_within_3pp"] = bool(
            lo >= -EQUIVALENCE_TOLERANCE and hi <= EQUIVALENCE_TOLERANCE)
        doc["equiv_tolerance"] = EQUIVALENCE_TOLERANCE
    return doc


def _sorted_sample_means(diffs: list[float], rng: random.Random, n_resamples: int) -> list[float]:
    means: list[float] = []
    n_items = len(diffs)
    for _ in range(n_resamples):
        sample = [diffs[rng.randrange(n_items)] for _ in range(n_items)]
        means.append(sum(sample) / n_items)
    means.sort()
    return means


def _percentile_bootstrap(point: float, sorted_means: list[float]) -> dict[str, Any]:
    return {
        "estimate": point,
        "ci95_low": sorted_means[int(0.025 * len(sorted_means))],
        "ci95_high": sorted_means[min(len(sorted_means) - 1, int(0.975 * len(sorted_means)))],
        "n_resamples": len(sorted_means),
        "method": "paired_percentile_bootstrap",
    }


# -------------------------------------------------------------------- report
def build_report(
    root: str | Path = MATCHED_ROOT_DEFAULT,
    *,
    benchmark_root: str | Path = "runs_benchmark",
    runs_reviewed_root: str | Path = "runs_reviewed",
    out_path: str | Path | None = None,
) -> dict[str, Any]:
    """Full offline report doc. Numeric chart tables come from build_chart_tables."""
    freeze = load_matched_freeze(root)
    prices, _ = _prices_from_freeze(freeze)
    models = list(freeze["models"])

    per_model: dict[str, dict[str, Any]] = {}
    modern_flags_by_model: dict[str, dict[str, dict[str, bool]]] = {}
    for model in models:
        run_dir = Path(root) / run_id_for(freeze, model)
        blocks: dict[str, Any] = {}
        flags_by_dataset: dict[str, dict[str, bool]] = {}
        for dataset in DATASETS:
            block = score_model_dataset(run_dir, freeze, model, dataset, prices=prices)
            blocks[dataset] = block
            flags_by_dataset[dataset] = {
                e["item_id"]: e["correct"] for e in block["per_item"]
                if e["correct"] is not None}
        per_model[model] = {"blocks": blocks}
        modern_flags_by_model[model] = flags_by_dataset

    jev_side: dict[str, Any] = {}
    jev_flags_by_dataset: dict[str, dict[str, bool]] = {}
    for dataset in DATASETS:
        gold = gold_records(freeze["datasets"][dataset]["items"], dataset)
        records = _jev_records(freeze, dataset, benchmark_root, runs_reviewed_root)
        flags, reasons = jev_correct_flags(records, gold)
        jev_flags_by_dataset[dataset] = flags
        jev_side[dataset] = {
            "join": freeze["jev_join"][dataset],
            "n_usable": len(flags),
            "excluded": len(gold) - len(flags),
            "exclusion_reasons": sorted(set(reasons.values())),
        }

    paired_doc: dict[str, dict[str, Any]] = {d: {} for d in DATASETS}
    for dataset in DATASETS:
        gold = gold_records(freeze["datasets"][dataset]["items"], dataset)
        contrasts: dict[str, dict[str, Any]] = {}
        p_values: dict[str, float] = {}
        for model in models:
            contrast = paired_analysis(modern_flags_by_model[model].get(dataset, {}),
                                       jev_flags_by_dataset[dataset], gold)
            contrasts[model] = contrast
            p_values[model] = contrast["mcnemar_exact_p"]
        adjusted = holm_stepdown(p_values)
        for model in models:
            contrasts[model]["p_holm_adjusted"] = adjusted.get(model)
        paired_doc[dataset] = contrasts

    comparison_table: dict[str, Any] = {}
    for model in models:
        row: dict[str, Any] = {}
        for dataset in DATASETS:
            head = per_model[model]["blocks"][dataset]["headline"]
            row[dataset] = {
                "n_requested": head["n_requested"],
                "completed": head["n_terminal"],
                "valid": head["n_valid"],
                "format_failed": head["n_format_failed"],
                "error": head["n_error"],
                "missing": head["n_missing"],
                "correct": head["correct_all"],
                "accuracy_all": head["accuracy_all"],
                "accuracy_conditional": head["accuracy_conditional"],
                "accuracy_weighted": head.get("accuracy_weighted_to_population"),
            }
        comparison_table[model] = row

    report: dict[str, Any] = {
        "generated_at_utc": _utc_now(),
        "freeze_sha256": freeze["deterministic_sha256"],
        "condition": freeze["condition"],
        "noul_conversion": freeze["noul_conversion"],
        "cost_guard": freeze["cost_guard"],
        "per_model": per_model,
        "comparison_table": comparison_table,
        "jev_side": jev_side,
        "jev_native_full_scores": _jev_native_scores(benchmark_root, freeze),
        "jev_matched_subset_scores": _jev_matched_scores(
            freeze, benchmark_root, runs_reviewed_root),
        "paired": paired_doc,
        "order_robustness": _order_robustness(benchmark_root, freeze, modern_flags_by_model),
        "effort_flags": _effort_flags(freeze),
        "reviewed_references": _reviewed_references(runs_reviewed_root),
        "generation_addendum": _generation_addendum(),
        "caveats": [
            "MMLU benchmark label-quality caveat retained; no silent adjudication to boost Jev.",
            "Published 5-shot CoT / reasoning-enabled scores are historical_contextual "
            "references only — NOT protocol-compatible with this direct native protocol.",
            "No universal parity claim follows from a failure to reject a difference; "
            "equivalence uses the predeclared ±0.03 paired-interval tolerance.",
            "GPQA remains a gated access dependency and is not bypassed here.",
        ],
    }
    return report


def build_chart_tables(report: dict[str, Any]) -> dict[str, Any]:
    """Numeric chart tables/JSON (no rendering, no screenshots, no image requests)."""
    comparison = report["comparison_table"]
    models = list(comparison)
    datasets = list(DATASETS)
    heatmap = {
        "rows": models,
        "columns": datasets,
        "accuracy_all": [[comparison[m][d]["accuracy_all"] for d in datasets] for m in models],
        "accuracy_conditional": [[comparison[m][d]["accuracy_conditional"] for d in datasets]
                                 for m in models],
    }
    error_cost_latency = {}
    for model in models:
        blocks = report["per_model"][model]["blocks"]
        error_cost_latency[model] = {
            d: {
                "error_rate": (
                    (blocks[d]["headline"]["n_error"] + blocks[d]["headline"]["n_format_failed"])
                    / blocks[d]["headline"]["n_requested"])
                if blocks[d]["headline"]["n_requested"] else None,
                "cost_billed_reported_usd": blocks[d]["cost"]["billed_reported_usd"],
                "cost_billed_estimated_usd": blocks[d]["cost"]["billed_estimated_usd"],
                "attempts_without_usage": blocks[d]["cost"]["n_unknown_usage"],
                "latency_mean_ms": blocks[d]["latency"]["mean_ms"],
                "latency_p95_ms": blocks[d]["latency"]["p95_ms"],
                "reasoning_tokens_total": blocks[d]["usage_totals"]["reasoning_tokens"],
            } for d in datasets
        }
    mmlu_subjects: dict[str, Any] = {
        "jev_full": {g: b["accuracy"] for g, b in
                     report["jev_native_full_scores"]["mmlu"]["by_group"].items()},
    }
    for model in models:
        by_group = report["per_model"][model]["blocks"]["mmlu"]["by_group"]
        mmlu_subjects[model] = {g: b["accuracy_conditional"] for g, b in by_group.items()}
    return {
        "mmlu_subject_breakdown": mmlu_subjects,
        "matched_heatmap": heatmap,
        "error_rate_cost_latency": error_cost_latency,
        "order_robustness": report["order_robustness"],
        "reviewed_mechanism_batching_references": report["reviewed_references"],
        "generation_addendum_references": report["generation_addendum"],
    }


def write_report(report: dict[str, Any], out_path: str | Path) -> None:
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")


# ------------------------------------------------------------------- helpers
def _prices_from_freeze(freeze: dict[str, Any]) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    from .matched import load_catalog_prices

    return load_catalog_prices(freeze["catalog"]["catalog_file"])


def _jev_records(freeze: dict[str, Any], dataset: str, benchmark_root: str | Path,
                 runs_reviewed_root: str | Path) -> dict[str, dict[str, Any]]:
    join = freeze["jev_join"][dataset]
    if "run_id" in join:
        results_path = Path(benchmark_root) / join["run_id"] / "results.jsonl"
    else:
        results_path = Path(runs_reviewed_root) / "rev-fresh_test-576d9a002eaa" / "results.jsonl"
    return {str(r["item_id"]): r for r in _read_jsonl(results_path)
            if r.get("terminal") is True}


def _jev_native_scores(benchmark_root: str | Path, freeze: dict[str, Any]) -> dict[str, Any]:
    """DISTINCT record: native full-suite Jev scores (never mixed with matched)."""
    out: dict[str, Any] = {}
    for dataset, stage in (("mmlu", "mmlu_full"), ("arc", "arc_test")):
        run_id = freeze["jev_join"][dataset]["run_id"]
        score_path = Path(benchmark_root) / run_id / "derived" / "score.json"
        if not score_path.exists():
            out[dataset] = {"available": False, "score_path": str(score_path)}
            continue
        score = json.loads(score_path.read_text(encoding="utf-8"))
        out[dataset] = {
            "run_id": run_id,
            "n_expected": score["n_expected"],
            "n_scored": score["n_scored"],
            "accuracy": score["accuracy"],
            "wilson_95": score["wilson_95"],
            "by_group": score["by_group"],
            "note": "full-suite native direct score; DISTINCT from matched-subset records",
        }
    out["mmlu_label_quality_caveat"] = (
        "MMLU-Pro contains items with contested labels; the caveat is retained and "
        "no silent adjudication is performed to boost any side")
    return out


def _jev_matched_scores(freeze: dict[str, Any], benchmark_root: str | Path,
                        runs_reviewed_root: str | Path) -> dict[str, Any]:
    """Jev accuracy restricted to the SAME matched items (DISTINCT column)."""
    out: dict[str, Any] = {}
    for dataset in DATASETS:
        gold = gold_records(freeze["datasets"][dataset]["items"], dataset)
        records = _jev_records(freeze, dataset, benchmark_root, runs_reviewed_root)
        flags, reasons = jev_correct_flags(records, gold)
        n = len(flags)
        k = sum(1 for v in flags.values() if v)
        out[dataset] = {
            "n_matched": len(gold),
            "n_usable": n,
            "correct": k,
            "accuracy_conditional": (k / n) if n else None,
            "wilson_conditional": wilson_interval(k, n).to_dict() if n else None,
            "excluded": len(gold) - n,
            "exclusion_reasons": sorted(set(reasons.values())),
            "note": "Jev restricted to the same matched items; DISTINCT from native full-suite records",
        }
    return out


def _order_robustness(benchmark_root: str | Path, freeze: dict[str, Any],
                      modern_flags_by_model: dict[str, dict[str, dict[str, bool]]]) -> dict[str, Any]:
    """Native vs option-rotation robustness, canonical labels restored, matched items.

    Uses the Jev rotation stage (recorded remappings; canonical labels restored
    through the recorded bijection) restricted to matched base items, modern
    native scores on the same items; variants reported separately, NEVER pooled
    into a best-of.
    """
    from .benchmark_exec import load_freeze, stage_run_id

    jev_freeze = load_freeze(benchmark_root)
    rotation_run_id = stage_run_id(jev_freeze, "option_rotations")
    mmlu_run_id = stage_run_id(jev_freeze, "mmlu_full")
    rot_score = json.loads((Path(benchmark_root) / rotation_run_id / "derived" / "score.json")
                           .read_text(encoding="utf-8"))
    rot_per_item = {r["item_id"]: r for r in rot_score["per_item"]}
    base_items = sorted({item_id.rsplit(":", 1)[0] for item_id in rot_per_item
                         if item_id.rsplit(":", 1)[-1].startswith("perm")})
    matched_ids = {str(it["id"]) for it in freeze["datasets"]["mmlu"]["items"]}
    matched_base = sorted(set(base_items) & matched_ids)
    variants = sorted({item_id.rsplit(":", 1)[-1] for item_id in rot_per_item})

    def _flags_for(ids: set[str], source: list[dict[str, Any]]) -> dict[str, bool]:
        return {r["item_id"]: r["correct"] for r in source
                if r.get("correct") is not None and r["item_id"] in ids}

    mmlu_score = json.loads((Path(benchmark_root) / mmlu_run_id / "derived" / "score.json")
                            .read_text(encoding="utf-8"))
    native_flags = _flags_for(set(matched_base), mmlu_score["per_item"])
    variants_doc: dict[str, Any] = {}
    for variant in variants:
        flags = _flags_for({f"{b}:{variant}" for b in matched_base}, rot_score["per_item"])
        n = len(flags)
        k = sum(1 for v in flags.values() if v)
        variants_doc[variant] = {
            "n": n, "correct": k, "accuracy": (k / n) if n else None,
            "note": ("canonical labels restored via the recorded remapping; reported "
                     "as a separate variant, never pooled into a best-of"),
        }
    modern_doc: dict[str, Any] = {}
    for model in freeze["models"]:
        flags = modern_flags_by_model[model]["mmlu"]
        sub = {b: flags[b] for b in matched_base if b in flags}
        n = len(sub)
        k = sum(1 for v in sub.values() if v)
        modern_doc[model] = {"n": n, "correct": k, "accuracy": (k / n) if n else None}
    n = len(native_flags)
    k = sum(1 for v in native_flags.values() if v)
    return {
        "matched_base_items": len(matched_base),
        "rotation_stage_run_id": rotation_run_id,
        "jev_native": {"n": n, "correct": k, "accuracy": (k / n) if n else None},
        "jev_rotation_variants": variants_doc,
        "modern_native": modern_doc,
    }


def _effort_flags(freeze: dict[str, Any]) -> dict[str, Any]:
    return {
        "reasoning_effort_supported": freeze["catalog"]["reasoning_effort_supported"],
        "note": freeze["catalog"]["effort_flag_note"],
        "wire": freeze["wire"],
    }


def _reviewed_references(runs_reviewed_root: str | Path) -> dict[str, Any]:
    root = Path(runs_reviewed_root)
    refs: dict[str, Any] = {}
    for name, rel in (
        ("reviewed_report", "derived/reviewed_report.json"),
        ("mechanism_run", "rev-mechanism-576d9a002eaa"),
        ("batching_run", "rev-batching-576d9a002eaa"),
        ("fresh_items", "rev-fresh_test-576d9a002eaa/items.jsonl"),
    ):
        path = root / rel
        refs[name] = {
            "path": str(path),
            "exists": path.exists(),
            "sha256": _file_sha256(path) if path.is_file() else None,
        }
    return refs


def _generation_addendum() -> dict[str, Any]:
    refs: dict[str, Any] = {}
    for name, rel in (
        ("reference_matrix", "docs/modern-comparison/REFERENCE-MATRIX.md"),
        ("reference_records", "docs/modern-comparison/reference-records.json"),
        ("talk_audit", "docs/modern-comparison/TALK-AUDIT.md"),
    ):
        path = Path(rel)
        refs[name] = {
            "path": rel,
            "exists": path.exists(),
            "sha256": _file_sha256(path) if path.is_file() else None,
        }
    refs["note"] = (
        "older side-generation runs are reported with the method caveats from "
        "TALK-AUDIT.md; corrected causal reruns are NOT silently performed and "
        "would require separate parent approval; 'scores not found' is never "
        "reported as 'none published'")
    return refs


def _file_sha256(path: Path) -> str:
    from hashlib import sha256

    return sha256(Path(path).read_bytes()).hexdigest()


def _utc_now() -> str:
    from .manifest import utc_now

    return utc_now()