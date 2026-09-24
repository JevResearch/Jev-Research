"""Offline analysis: join run predictions with gold labels, compute metrics.

The run artifacts (attempts.jsonl, results.jsonl) contain no gold labels; this
module is the only place where predictions meet answers, loaded from the
run's own `items.jsonl` copy of the experiment spec.  A mismatch between the
manifest's item hash and the joined labels is an error, not a warning.

Outputs: `derived/metrics.json`, `derived/attempts.parquet`,
`derived/results.parquet`, `derived/prediction_rows.parquet`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .manifest import canonical_sha256, utc_now
from .runner import load_manifest
from . import metrics as M
from . import stats as ST


def _try_cluster_bootstrap(correct, clusters):
    """Cluster CI when the design supports it; a labeled refusal otherwise."""
    try:
        return ST.cluster_bootstrap_accuracy(correct, clusters).to_dict()
    except ValueError as exc:
        return {"unavailable": str(exc)}

CHOICE_TIE_MAX = 0  # p_max above which a prediction is "confident" (risk-coverage)


def load_gold_items(run_dir: Path) -> dict[str, dict[str, Any]]:
    """item_id -> full spec entry (gold included). Private labels, offline only."""
    path = run_dir / "items.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"no items.jsonl in {run_dir}")
    return {entry["id"]: entry for entry in _jsonl(path)}


def _jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def gold_for_question(entry: dict[str, Any], question_id: str) -> Any:
    gold = entry.get("gold", {})
    if question_id not in gold:
        return None
    value = gold[question_id]
    if isinstance(value, dict):
        return value.get("value", value.get("level", value.get("yes")))
    return value


def score_prediction(record: dict[str, Any], entry: dict[str, Any]) -> list[dict[str, Any]]:
    """Score one logical request's predictions against its gold labels."""
    rows: list[dict[str, Any]] = []
    option_map = record.get("option_map", {})
    for qid, prediction in record.get("predictions", {}).items():
        gold_value = gold_for_question(entry, qid)
        gold = gold_value.get("value") if isinstance(gold_value, dict) else gold_value
        row = {
            "logical_request_id": record["logical_request_id"],
            "item_id": record["item_id"],
            "group": record.get("group"),
            "cluster": record.get("cluster"),
            "condition": record.get("condition"),
            "question_id": qid,
            "type": prediction.get("type"),
            "status": record.get("status"),
            "usable": prediction.get("usable"),
            "violations": record.get("violation_codes", []),
            "usage_input_tokens": record.get("usage_input_tokens"),
            "latency_ms_total": record.get("latency_ms_total"),
            "correct": None,  # set only when gold exists; missing gold is not a failure
        }
        qtype = prediction.get("type")
        probabilities = prediction.get("probabilities") or {}
        if qtype == "noul":
            p_yes = prediction.get("noul")
            row["predicted_p_yes"] = p_yes
            if isinstance(gold, bool):
                row["gold_yes"] = gold
                row["correct"] = (p_yes >= 0.5) == gold
                row["brier_binary"] = M.brier_binary(p_yes, gold)
                row["p_true"] = p_yes if gold else 1.0 - p_yes
                row["log_loss_exact"] = M.log_loss_exact(row["p_true"])
                row["log_loss_clipped_1e-12"] = M.log_loss_clipped(row["p_true"])
        elif qtype == "choice":
            gold_key = gold
            row["predicted_choice"] = prediction.get("choice")
            row["p_max"] = prediction.get("p_max")
            row["probabilities"] = probabilities
            if gold_key is not None:
                row["gold"] = gold_key
                row["correct"] = prediction.get("choice") == gold_key
                if probabilities:
                    row.update(M.brier_multiclass(probabilities, gold_key))
                    row["p_true"] = probabilities.get(gold_key, 0.0)
                    row["log_loss_exact"] = M.log_loss_exact(row["p_true"])
                    row["log_loss_clipped_1e-12"] = M.log_loss_clipped(row["p_true"])
        elif qtype == "score":
            gold_level = gold if isinstance(gold, (int, float)) else None
            row["predicted_score"] = prediction.get("score")
            row["expectation_from_probabilities"] = prediction.get("expectation_from_probabilities")
            if gold_level is not None:
                row["gold_level"] = gold_level
                rounded = round(prediction.get("score", 0.0))
                row["correct"] = rounded == gold_level
                row["absolute_error"] = abs(prediction.get("score", 0.0) - gold_level)
                probs = prediction.get("probabilities") or {}
                ordered = [probs[str(i)] for i in range(len(probs))]
                if ordered:
                    row["rps"] = M.rank_probability_score(ordered, int(gold_level))
        rows.append(row)
    return rows


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-type aggregate metrics; choice and noul are never averaged together."""
    choice_rows = [r for r in rows if r["type"] == "choice" and r.get("correct") is not None]
    noul_rows = [r for r in rows if r["type"] == "noul" and r.get("correct") is not None]
    score_rows = [r for r in rows if r["type"] == "score" and r.get("correct") is not None]
    out: dict[str, Any] = {
        "n_scored_rows": len(rows),
        "n_rows_missing_gold": sum(1 for r in rows if r.get("correct") is None),
    }
    if choice_rows:
        correct = [bool(r["correct"]) for r in choice_rows]
        clusters = [str(r.get("cluster") or r["logical_request_id"]) for r in choice_rows]
        out["choice"] = {
            "accuracy": M.accuracy(correct),
            "cluster_bootstrap": _try_cluster_bootstrap(correct, clusters),
            "effective_sample_size_clusters": ST.effective_sample_size_for_clusters(clusters),
            "brier_sum_mean": _mean([r.get("brier_sum") for r in choice_rows]),
            "log_loss_exact_mean": _mean_exact_with_inf([r.get("log_loss_exact") for r in choice_rows]),
            "n_infinite_log_loss": sum(1 for r in choice_rows
                                       if isinstance(r.get("log_loss_exact"), float) and math.isinf(r["log_loss_exact"])),
            "log_loss_clipped_1e-12_mean": _mean([r.get("log_loss_clipped_1e-12") for r in choice_rows]),
            "p_max_reliability": M.reliability([(r.get("p_max"), bool(r["correct"])) for r in choice_rows]),
            "ece_10bins": M.ece_equal_width([(r.get("p_max"), bool(r["correct"])) for r in choice_rows]),
            "risk_coverage": M.risk_coverage([(r.get("p_max"), bool(r["correct"])) for r in choice_rows]),
        }
    if noul_rows:
        noul_clusters = [str(r.get("cluster") or r["logical_request_id"]) for r in noul_rows]
        # Calibration target for P(yes) is *whether the answer was yes*, not
        # whether the 0.5-threshold prediction was correct. Items without a
        # boolean gold label are excluded from the reliability curve.
        calibratable = [(r.get("predicted_p_yes"), r.get("gold_yes")) for r in noul_rows
                        if isinstance(r.get("gold_yes"), bool)]
        out["noul"] = {
            "accuracy": M.accuracy([bool(r["correct"]) for r in noul_rows]),
            "cluster_bootstrap": _try_cluster_bootstrap([bool(r["correct"]) for r in noul_rows], noul_clusters),
            "brier_binary_mean": _mean([r.get("brier_binary") for r in noul_rows]),
            "log_loss_exact_mean": _mean_exact_with_inf([r.get("log_loss_exact") for r in noul_rows]),
            "n_infinite_log_loss": sum(1 for r in noul_rows
                                       if isinstance(r.get("log_loss_exact"), float) and math.isinf(r["log_loss_exact"])),
            "p_yes_reliability": M.reliability(calibratable) if calibratable else [],
            "note": "reliability pairs P(yes) with the gold yes/no outcome, not with thresholded correctness",
        }
    if score_rows:
        out["score"] = {
            "exact_level_accuracy": M.accuracy([bool(r["correct"]) for r in score_rows]),
            "mean_absolute_error": M.ordinal_mae(
                [r.get("absolute_error", 0.0) for r in score_rows],
                [0.0] * len(score_rows),
            )
            if False
            else _mean([r.get("absolute_error") for r in score_rows]),
            "rps_mean": _mean([r.get("rps") for r in score_rows]),
        }
    return out


def _mean(values):
    clean = [v for v in values if v is not None and not (isinstance(v, float) and math.isinf(v))]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _mean_exact_with_inf(values):
    """Mean of exact log losses. ANY infinite entry makes the mean infinite —
    silently dropping zero-probability true outcomes understates the loss."""
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    if any(isinstance(v, float) and math.isinf(v) for v in clean):
        return math.inf
    return sum(clean) / len(clean)


def _mean_or_inf(values):
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    finite = [v for v in clean if not (isinstance(v, float) and math.isinf(v))]
    if not finite:
        return math.inf
    return sum(finite) / len(finite)


def analyze_run(run_id: str, root: str | None = None) -> dict[str, Any]:
    """Full offline analysis for a completed run. No network, no provider access."""
    directory = (Path(root) if root else Path("runs")) / run_id
    manifest = load_manifest(run_id, root)
    results = list(_jsonl(directory / "results.jsonl"))
    gold = load_gold_items(directory)

    rows: list[dict[str, Any]] = []
    for record in results:
        entry = gold.get(record.get("item_id"))
        if entry is None:
            raise KeyError(
                f"result {record['logical_request_id']} references item {record.get('item_id')!r} "
                "which is not in items.jsonl; refusing to analyze against unknown labels"
            )
        rows.extend(score_prediction(record, entry))

    # Guard: the spec used at analysis time is the run's own items.jsonl; the
    # manifest records the planned hash for offline comparison.
    hash_note = {
        "manifest_items_sha256": manifest.items_sha256,
        "note": "labels come from this run's own items.jsonl; manifest.items_sha256 is the planned hash",
    }

    derived = directory / "derived"
    derived.mkdir(exist_ok=True)
    aggregates = aggregate(rows)

    # Pilot runs carry their sampling design in the manifest; produce the
    # population-weighted estimate alongside the naive one (never instead of).
    stratified_note = None
    sampling = (manifest.dataset or {}).get("sampling", {})
    if sampling.get("mode") == "stratified_pilot":
        sizes = sampling.get("population_category_sizes") or {}
        for qtype in ("choice", "noul"):
            subset = [r for r in rows if r["type"] == qtype and r.get("correct") is not None]
            if subset and sizes:
                key = f"stratified_estimate_{qtype}"
                aggregates[key] = ST.stratified_accuracy(
                    [bool(r["correct"]) for r in subset],
                    [str(r.get("group")) for r in subset],
                    sizes,
                )
        stratified_note = "pilot run: naive aggregate shown alongside population-weighted estimate"

    report = {
        "run_id": run_id,
        "analyzed_at": utc_now(),
        "manifest_sha256": manifest.sha256(),
        "provider_observed": _observed_provider(results),
        "synthetic": manifest.provider != "jev",
        "warning": "MOCK/SIMULATED RESULTS" if manifest.provider != "jev" else None,
        "label_hash_note": hash_note,
        "n_logical_results": len(results),
        "n_uncertain_open": _open_uncertain(directory),
        "aggregates": aggregates,
        "sampling_note": stratified_note,
        "per_group": _per_group(rows),
        "violation_summary": _violation_summary(rows),
    }
    (derived / "metrics.json").write_text(_json_pretty(report) + "\n")

    _write_parquet(derived / "attempts.parquet", list(_jsonl(directory / "attempts.jsonl")))
    _write_parquet(derived / "results.parquet", results)
    _write_parquet(derived / "prediction_rows.parquet", rows)
    return report


def _observed_provider(results: list[dict[str, Any]]) -> str | None:
    providers = {r.get("provider") for r in results if r.get("provider")}
    if len(providers) == 1:
        return providers.pop()
    return ",".join(sorted(str(p) for p in providers)) or None


def _open_uncertain(directory: Path) -> list[str]:
    uncertain: dict[str, int] = {}
    terminal = {r["logical_request_id"] for r in _jsonl(directory / "results.jsonl")}
    for attempt in _jsonl(directory / "attempts.jsonl"):
        if attempt.get("uncertain"):
            uncertain[attempt["logical_request_id"]] = uncertain.get(attempt["logical_request_id"], 0) + 1
    return sorted(lid for lid in uncertain if lid not in terminal)


def _violation_summary(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for code in row.get("violations", []):
            counts[code] = counts.get(code, 0) + 1
    return dict(sorted(counts.items()))


def _per_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = f"{row.get('group')}/{row.get('type')}"
        grouped.setdefault(key, []).append(row)
    out: dict[str, Any] = {}
    for key, group_rows in sorted(grouped.items()):
        scored = [r for r in group_rows if r.get("correct") is not None]
        out[key] = {
            **M.accuracy([bool(r["correct"]) for r in scored]),
            "n_rows_missing_gold": len(group_rows) - len(scored),
        }
    return out


def _json_pretty(obj: Any) -> str:
    import json

    return json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default)


def _json_default(value):
    if isinstance(value, float) and math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    return str(value)


def _scalar(value):
    if isinstance(value, (dict, list)):
        import json

        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return value


def _write_parquet(path: Path, records: list[dict[str, Any]]) -> bool:
    """Write a parquet table; falls back to JSONL if pyarrow is unavailable."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception:  # pragma: no cover - optional dependency fallback
        path.with_suffix(".jsonl").write_text(
            "".join(json.dumps(r, default=str) + "\n" for r in records), encoding="utf-8"
        )
        return False
    columns = sorted({k for record in records for k in record})
    data = {col: [_scalar(record.get(col)) for record in records] for col in columns}
    pq.write_table(pa.table({col: pa.array(values) for col, values in data.items()}), str(path))
    return True