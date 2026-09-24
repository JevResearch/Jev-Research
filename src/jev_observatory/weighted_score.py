"""Probability-weighted scoring — a SEPARATE reported variant, never merged.

For every stage the headline remains GREEDY accuracy (the server's argmax
choice, failures counted as incorrect — benchmark_score.py).  This module
computes the complementary PROBABILITY-WEIGHTED diagnostic on the SAME stored
artifacts (offline, never a model call, never a re-dispatch):

* choice questions: weighted score per item = p(gold) — the probability mass
  the model itself assigned to the correct option (A=40% & gold A -> 0.4);
  the stage number is the mean over items ("expected accuracy under the
  model's own distribution");
* score questions (per-digit / per-cell readouts): weighted score per
  question = p(gold level); for the MATH-500 digit readout the item-level
  weighted score is the PRODUCT of the per-digit gold probabilities (the
  model-implied probability of the whole correct readout — labeled: assumes
  independent digit judgments); ARC cells report per-cell weighted only,
  because a product over ~400 cells is not a usable statistic.

Validity is the same contract as benchmark_score._score_probabilities: the
probability vector must be present, finite, in [0,1] and sum to ~1 (within
5e-2); invalid vectors are EXCLUDED and counted, never renormalized.  Items
with missing terminal results are excluded from the weighted mean and counted.
Output: derived/weighted_score.json — a distinct artifact, clearly labeled,
never pooled into the greedy headline.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .benchmark_score import ScoringError, score_run

PROB_SUM_TOLERANCE = 5e-2


def weighted_score_run(run_dir: Path | str, *, stage: str) -> dict[str, Any]:
    run_dir = Path(run_dir)
    gold_single, gold_maps = _load_gold(run_dir)
    if stage.startswith("arc_agi2"):
        gold_single = None  # multi-question items: per-question weighted only
    results = {str(r.get("item_id")): r for r in _read_results(run_dir)
               if r.get("terminal") is True}
    weighted_items: dict[str, float | None] = {}
    per_question: list[dict[str, Any]] = []
    n_excluded = 0
    exclusion_reasons: dict[str, int] = {}

    def exclude(item_id: str, reason: str) -> None:
        nonlocal n_excluded
        n_excluded += 1
        exclusion_reasons[reason] = exclusion_reasons.get(reason, 0) + 1
        weighted_items[item_id] = None

    if gold_single is not None:
        # single-question choice items: p(gold) per item
        for item_id, gold in gold_single.items():
            result = results.get(item_id)
            if result is None or result.get("status") != "ok":
                exclude(item_id, f"missing_or_nonok_{result.get('status') if result else 'missing'}")
                continue
            predictions = result.get("predictions") or {}
            answer = next(iter(predictions.values()), None) if predictions else None
            p_gold, reason = _p_gold_choice(answer, gold["value"])
            if p_gold is None:
                exclude(item_id, reason or "invalid_probability_vector")
                continue
            weighted_items[item_id] = p_gold
            per_question.append({"item_id": item_id, "p_gold": p_gold})
    else:
        # multi-question items: per-question p(gold level)
        for item_id, gmap in gold_maps.items():
            result = results.get(item_id)
            if result is None or result.get("status") != "ok":
                exclude(item_id, f"missing_or_nonok_{result.get('status') if result else 'missing'}")
                continue
            predictions = result.get("predictions") or {}
            item_product = 1.0
            valid = True
            for qid, gold_value in gmap.items():
                answer = predictions.get(qid) or {}
                p_gold, reason = _p_gold_level(answer, gold_value)
                if p_gold is None:
                    valid = False
                    exclude(item_id, reason or "invalid_probability_vector")
                    break
                item_product *= p_gold
                per_question.append({"item_id": item_id, "qid": qid,
                                     "p_gold": p_gold})
            if valid:
                weighted_items[item_id] = item_product

    values = [v for v in weighted_items.values() if v is not None]
    values.sort()
    mean = (sum(values) / len(values)) if values else None
    if stage.startswith("arc_agi2"):
        # a product over hundreds of cells is not a usable statistic; the ARC
        # stages report the PER-CELL weighted mean instead (see below)
        mean = None
    doc: dict[str, Any] = {
        "stage": stage,
        "run_dir": str(run_dir),
        "variant": "probability_weighted",
        "definition": (
            "weighted score per item = the probability mass the model assigned "
            "to the gold option (choice) / gold level (score questions); the "
            "stage number is the mean over scored items — the model-implied "
            "expected accuracy under its own distribution"),
        "separateness_note": (
            "SEPARATE diagnostic artifact: never merged into, substituted for, "
            "or averaged with the greedy headline accuracy"),
        "n_expected": len(weighted_items),
        "n_weighted_scored": len(values),
        "n_excluded": n_excluded,
        "exclusion_reasons": exclusion_reasons,
        "weighted_accuracy_mean": mean,
        "weighted_accuracy_p05": (values[max(0, int(0.05 * len(values)) - 1)]
                                  if values else None),
        "weighted_accuracy_p50": (values[len(values) // 2] if values else None),
        "weighted_accuracy_p95": (values[min(len(values) - 1,
                                             int(0.95 * len(values)))]
                                  if values else None),
        "validity_note": ("probability vectors must be finite, in [0,1] and sum "
                          "to ~1 (within 5e-2); invalid vectors are excluded and "
                          "counted, never renormalized"),
        "per_item": [{"item_id": k, "weighted": v}
                     for k, v in sorted(weighted_items.items())],
    }
    if stage == "math500_score":
        doc["item_product_note"] = (
            "item weighted score = PRODUCT of per-digit gold probabilities: the "
            "model-implied probability of the whole correct readout under an "
            "independent-digits assumption (approximation, labeled)")
    if stage.startswith("arc_agi2"):
        question_p = [q["p_gold"] for q in per_question]
        doc["per_cell_weighted_mean"] = (sum(question_p) / len(question_p)
                                         if question_p else None)
        doc["item_product_note"] = (
            "per-cell weighted mean only; a product over hundreds of cells is "
            "not a usable statistic and is deliberately not reported")
        doc.pop("weighted_accuracy_p05", None)
        doc.pop("weighted_accuracy_p95", None)
    return doc


def _p_gold_choice(answer: Any, gold_value: str) -> tuple[float | None, str | None]:
    # Choice and level vectors share the same validity contract (finite, in
    # [0,1], sum ~1); only the key space differs.
    return _p_gold_level(answer, gold_value)


def _p_gold_level(answer: Any, gold_value: str) -> tuple[float | None, str | None]:
    if not isinstance(answer, dict) or not answer.get("usable"):
        return None, "prediction_unusable"
    probabilities = answer.get("probabilities")
    if not isinstance(probabilities, dict) or not probabilities:
        return None, "probability_vector_missing"
    total = 0.0
    for key, value in probabilities.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return None, "probability_not_numeric"
        if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            return None, "probability_out_of_range"
        total += float(value)
    if not math.isfinite(total) or abs(total - 1.0) > PROB_SUM_TOLERANCE:
        return None, "probability_sum_out_of_tolerance"
    if gold_value not in probabilities:
        return None, "gold_level_missing_from_vector"
    return float(probabilities[gold_value]), None


def _load_gold(run_dir: Path | str) -> tuple[dict[str, dict[str, Any]] | None,
                                             dict[str, dict[str, str]]]:
    """(single-value gold per item, per-question gold maps) from items.jsonl."""
    path = Path(run_dir) / "items.jsonl"
    single: dict[str, dict[str, Any]] = {}
    maps: dict[str, dict[str, str]] = {}
    for line in path.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        record = json.loads(line)
        item_id = str(record["id"])
        values: dict[str, str] = {}
        for qid, payload in (record.get("gold") or {}).items():
            value = None
            if isinstance(payload, dict):
                if payload.get("value") is not None:      # {"value": v}
                    value = payload["value"]
                else:                                      # {group: {"value": v}}
                    for inner in payload.values():
                        if isinstance(inner, dict) and inner.get("value") is not None:
                            value = inner["value"]
                            break
            if value is not None:
                values[qid] = str(value)
        maps[item_id] = values
        first = next(iter(values.values()), None)
        if first is not None and len(values) == 1:
            single[item_id] = {
                "value": first,
                "group": str(record.get("group", "unknown")),
            }
    return (single if len(single) == len(maps) else None), maps


def _read_results(run_dir: Path | str) -> list[dict[str, Any]]:
    path = Path(run_dir) / "results.jsonl"
    if not path.exists():
        raise ScoringError(f"missing run artifact {path}")
    records = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        if line.strip():
            records.append(json.loads(line))
    return records


def write_weighted(doc: dict[str, Any], path: Path | str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
