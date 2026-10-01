"""Offline benchmark scorer: gold join, accuracy, robustness, honest accounting.

Everything here runs offline against stored run artifacts — never a model call.
The gold labels come ONLY from the frozen ``items.jsonl`` of the planned run
(structurally excluded from every outbound payload); predictions come from
``results.jsonl``. The scorer never repairs a bad answer, never drops a
failure, and never averages robustness data into the headline.

Scoring policy (``benchmark-score-2.0.0``; see ``SCORING_POLICY``):

* HEADLINE accuracy is ALL-REQUESTED: every requested logical request is in
  the denominator; missing terminal results, strict-format failures and
  unusable answers count as WRONG.  The usable-response accuracy and count
  are reported SEPARATELY and are never substituted for the headline;
* by-group and by-rotation-variant blocks follow the same all-requested
  denominator policy, with Wilson 95% intervals computed on it;
* an explicit finite-population caveat on every interval (a benchmark is a
  finite item set, so the interval is a diagnostic for item sampling within
  that fixed set, NOT a superpopulation claim);
* duplicate terminal ``logical_request_id`` (or ``item_id``) values are
  REFUSED with a hard error — never silently double-counted;
* rotation robustness scored against the canonical labels restored through
  the recorded remapping, reported per variant separately, with a hard error
  if the restored comparison disagrees with the direct comparison.  The
  probability KEYS of rotated items are canonicalized through the same
  remapping before any proper-score or zero-gold computation (looking up
  canonical gold against still-rotated keys produced spurious zero-gold
  counts);
* zero-probability gold count (gold key carries probability exactly 0 where
  the server reported probabilities; text baselines report ``null``);
* proper scores (multiclass Brier, clipped log loss) ONLY where probabilities
  are present, contract-usable and finite; otherwise an explicit null with a
  count of exclusions — never false calibration certainty.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .metrics import brier_multiclass, wilson_interval

WILSON_Z = 1.959963984540054
SCORE_SCHEMA_VERSION = "benchmark-score-2.0.0"
# Aggregates generated before the all-requested policy were written with this
# (implicit) denominator: usable answers only.  Preserved verbatim, never
# regenerated in place (see preserve_original_aggregate).
PRE_POLICY_AGGREGATE_VERSION = "benchmark-score-pre-policy-v1 (usable-answered denominator; rotation probability keys un-remapped)"
FINITE_SET_CAVEAT = (
    "Wilson intervals are item-sampling diagnostics within this fixed, finite "
    "benchmark item set; they are not superpopulation claims and the benchmark "
    "does not exhaust the item space of the subject."
)
SCORING_POLICY = {
    "headline": (
        "all-requested: every requested logical request is in the headline "
        "denominator; missing terminal results, strict-format failures and "
        "unusable answers count as wrong"),
    "usable_response_accuracy": (
        "separately reported over ok+usable responses only, with its own "
        "count and Wilson interval; never substituted for the headline"),
    "by_group_and_rotation": (
        "denominators and Wilson intervals follow the all-requested policy"),
    "proper_scores": (
        "displayed probability vectors only; for rotated items the "
        "probability KEYS are canonicalized through the recorded remapping "
        "before comparison with canonical gold"),
    "duplicates": (
        "duplicate terminal logical_request_id (or item_id) values are "
        "refused with a hard error, never double-counted"),
}


class ScoringError(RuntimeError):
    pass


def load_items_gold(run_dir: Path | str, *, require_single_value: bool = True) -> dict[str, dict[str, Any]]:
    """gold + group per ITEM id, from the run's frozen items.jsonl.

    Multi-question items (e.g. the ARC-AGI-2 whole-task/grid stages, where an
    item carries one question per output cell) have per-QUESTION gold values
    rather than a single item-level gold; with ``require_single_value=False``
    the loader accepts such items and exposes the first question's value as a
    placeholder ``None``-free record (``per_question`` carries the full map for
    the grid aggregator, which scores cells directly).
    """
    items_path = Path(run_dir) / "items.jsonl"
    gold: dict[str, dict[str, Any]] = {}
    for line in items_path.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        record = json.loads(line)
        item_id = str(record["id"])
        entry = (record.get("gold") or {})
        value = None
        if isinstance(entry, dict):
            for payload in entry.values():
                if isinstance(payload, dict) and payload.get("value") is not None:
                    value = str(payload["value"])
                    break
        if value is None:
            if require_single_value:
                raise ScoringError(f"item {item_id}: no gold value in items.jsonl; refusing to score")
            gold[item_id] = {"value": None, "group": str(record.get("group", "unknown"))}
            continue
        gold[item_id] = {"value": value, "group": str(record.get("group", "unknown"))}
    return gold


def score_digit_readout_run(run_dir: Path | str, *, stage: str) -> dict[str, Any]:
    """Score a multi-question digit-readout stage (e.g. MATH-500 score mode).

    Each item carries one question per digit; the item's value is the exact
    concatenation of its per-question golds (question ids sort as d0, d1, ...).
    An item is correct only when EVERY digit question has a usable prediction
    whose choice equals its gold — partial digits are wrong, never imputed.
    Missing terminal results count as incorrect (all-requested denominator).
    """
    run_dir = Path(run_dir)
    gold_maps: dict[str, dict[str, str]] = {}
    for line in (run_dir / "items.jsonl").read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        record = json.loads(line)
        values: dict[str, str] = {}
        for qid, payload in (record.get("gold") or {}).items():
            for inner in payload.values():
                if isinstance(inner, dict) and inner.get("value") is not None:
                    values[qid] = str(inner["value"])
                    break
        if not values:
            raise ScoringError(f"item {record['id']}: no gold digits; refusing to score")
        gold_maps[str(record["id"])] = values
    results = _read_jsonl(run_dir / "results.jsonl")
    terminal = [r for r in results if r.get("terminal") is True]
    _reject_duplicate_terminals(terminal)
    by_item = {str(r.get("item_id")): r for r in terminal}
    attempts = _read_jsonl(run_dir / "attempts.jsonl")

    correct_flags: list[bool] = []
    usable_flags: list[bool] = []
    group_flags: dict[str, list[bool]] = {}
    status_counts: dict[str, int] = {}
    per_item: list[dict[str, Any]] = []
    for item_id in sorted(gold_maps):
        gmap = gold_maps[item_id]
        group = _item_group(run_dir, item_id)
        result = by_item.get(item_id)
        record: dict[str, Any] = {
            "item_id": item_id, "group": group,
            "status": result.get("status") if result else "missing",
            "correct": None, "predicted": None, "usable": False,
            "gold": "".join(
                gmap[q] for q in sorted(gmap)),
        }
        status = record["status"]
        status_counts[status] = status_counts.get(status, 0) + 1
        correct = False
        if result is not None and result.get("status") == "ok":
            predictions = result.get("predictions") or {}
            usable_all = all(bool((predictions.get(q) or {}).get("usable"))
                             for q in gmap)
            predicted = "".join(str(argmax_level(
                (predictions.get(q) or {}).get("probabilities")))
                for q in sorted(gmap))
            record["predicted"] = predicted
            record["usable"] = bool(usable_all)
            correct = bool(usable_all and predicted == record["gold"])
        record["correct"] = correct
        correct_flags.append(correct)
        usable_flags.append(record["usable"])
        group_flags.setdefault(group, []).append(correct)
        per_item.append(record)
    n = len(correct_flags)
    k = sum(correct_flags)
    n_usable = sum(usable_flags)
    doc: dict[str, Any] = {
        "score_schema_version": SCORE_SCHEMA_VERSION,
        "scoring_policy": SCORING_POLICY,
        "stage": stage,
        "run_dir": str(run_dir),
        "n_requested": n,
        "n_expected": n,
        "n_scored": n,
        "n_missing_terminal": n - len(by_item),
        "missing_logical_ids": sorted(set(gold_maps) - set(by_item)),
        "n_correct": k,
        "accuracy": (k / n) if n else None,
        "wilson_95": wilson_interval(k, n).to_dict() if n else None,
        "finite_set_caveat": FINITE_SET_CAVEAT,
        "exact_concat_note": ("item correct only when every digit question is "
                              "usable and the concatenated digits equal the gold; "
                              "missing/error items count as incorrect"),
        "status_counts": status_counts,
        "strict_format_failures": sum(
            1 for r in per_item if r["status"] == "ok" and not r["usable"]),
        "n_usable_responses": n_usable,
        "usable_response_accuracy": (sum(1 for r in per_item if r["usable"] and r["correct"])
                                     / n_usable) if n_usable else None,
        "by_group": {},
        "by_rotation_variant": {},
        "attempts_summary": {
            "n_attempts": len(attempts),
            "n_uncertain_attempts": len([a for a in attempts if a.get("uncertain")]),
            "usage_input_tokens_sum": _sum_field(attempts, "usage_input_tokens"),
            "usage_output_tokens_sum": _sum_field(attempts, "usage_output_tokens"),
        },
        "per_item": per_item,
    }
    for group, flags in sorted(group_flags.items()):
        doc["by_group"][group] = _accuracy_block(flags)
    return doc


def _item_group(run_dir: Path | str, item_id: str) -> str:
    for line in (run_dir / "items.jsonl").read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        record = json.loads(line)
        if str(record["id"]) == item_id:
            return str(record.get("group", "unknown"))
    raise ScoringError(f"item {item_id} not found in items.jsonl")



def argmax_level(probabilities: Any) -> str | None:
    """Predicted level for a score answer: the highest-probability level index.

    Ties resolve to the LOWEST level index (deterministic, preregistered).
    Returns None when the probability vector is missing or malformed.
    """
    if not isinstance(probabilities, dict) or not probabilities:
        return None
    best_key: str | None = None
    best_value = -1.0
    for key, value in probabilities.items():
        try:
            level = int(key)
            p = float(value)
        except (TypeError, ValueError):
            return None
        if p > best_value or (p == best_value and level < int(best_key or 10**9)):
            best_key, best_value = key, p
    return best_key



def load_rotations(run_dir: Path | str) -> dict[str, dict[str, Any]]:
    """Rotation remappings for a rotation stage (missing for non-rotation stages)."""
    path = Path(run_dir) / "rotations.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def score_run(
    run_dir: Path | str,
    *,
    stage: str,
    gold_override: dict[str, dict[str, Any]] | None = None,
    rotations: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Score one benchmark stage run. Deterministic and offline.

    ``logical_request_id`` is ``{condition}:{item_id}``; predictions and
    attempts join on it.  Every requested logical request is in the headline
    denominator: a request with no terminal result — and a request answered
    with a strict-format failure or an unusable answer — counts as WRONG
    (all-requested policy), while the usable-response accuracy and count are
    reported separately.  Duplicate terminal logical ids are refused.
    """
    run_dir = Path(run_dir)
    gold = gold_override or load_items_gold(
        run_dir, require_single_value=not stage.startswith("arc_agi2"))
    rotations = rotations if rotations is not None else load_rotations(run_dir)
    results = _read_jsonl(run_dir / "results.jsonl")
    attempts = _read_jsonl(run_dir / "attempts.jsonl")
    terminal = [r for r in results if r.get("terminal") is True]
    _reject_duplicate_terminals(terminal)

    n_requested = len(gold)
    requested_correct: dict[str, bool] = dict.fromkeys(gold, False)
    requested_usable: dict[str, bool] = dict.fromkeys(gold, False)
    condition_of: dict[str, str] = {}
    zero_prob_gold = 0
    zero_prob_candidates = 0
    proper_scores: list[dict[str, Any]] = []
    prob_exclusions = 0
    strict_format_failures = 0
    status_counts: dict[str, int] = {}
    per_item: list[dict[str, Any]] = []

    for result in terminal:
        lid = result["logical_request_id"]
        item_id, condition = _split_logical_id(lid)
        if item_id not in gold:
            raise ScoringError(
                f"result {lid}: item not in the frozen gold set; refusing to score"
            )
        condition_of[item_id] = condition
        status = str(result.get("status"))
        status_counts[status] = status_counts.get(status, 0) + 1
        record: dict[str, Any] = {
            "logical_request_id": lid,
            "item_id": item_id,
            "condition": condition,
            "group": gold[item_id]["group"],
            "status": status,
            "model_returned": result.get("model_returned"),
            "correct": None,
            "usable": False,
            "choice": None,
            "canonical_choice": None,
        }
        if status == "ok":
            predictions = result.get("predictions") or {}
            answer = next(iter(predictions.values()), None) if predictions else None
            choice = (answer or {}).get("choice") if isinstance(answer, dict) else None
            usable = bool((answer or {}).get("usable"))
            if choice is None or not usable:
                strict_format_failures += 1
                status_counts["strict_format_failure"] = status_counts.get("strict_format_failure", 0) + 1
            else:
                record["usable"] = True
                requested_usable[item_id] = True
                gold_value = gold[item_id]["value"]
                canonical_choice, gold_value = _canonical_pair(
                    item_id, condition, choice, gold_value, rotations)
                record["choice"] = choice
                record["canonical_choice"] = canonical_choice
                record["gold_canonical"] = gold_value
                if gold[item_id]["value"] is not None:
                    # multi-question items (gold value None) are scored at the
                    # cell level by the grid aggregator, not here
                    correct = canonical_choice == gold_value
                    record["correct"] = correct
                    requested_correct[item_id] = bool(correct)
                probabilities = (answer or {}).get("probabilities")
                if gold_value is not None and isinstance(probabilities, dict) and probabilities:
                    scored_probs = probabilities
                    if item_id in rotations:
                        # canonicalize the probability KEYS through the same
                        # remapping used for the choice; canonical gold looked
                        # up against still-rotated keys is a scoring bug
                        scored_probs = _canonicalize_probabilities(
                            probabilities, rotations[item_id]["remapping"])
                    if scored_probs is None:
                        prob_exclusions += 1
                    else:
                        scored_prob = _score_probabilities(scored_probs, gold_value)
                        if scored_prob is None:
                            prob_exclusions += 1
                        else:
                            p_gold, brier, log_loss = scored_prob
                            if p_gold == 0.0:
                                zero_prob_gold += 1
                            zero_prob_candidates += 1
                            proper_scores.append({
                                "logical_request_id": lid,
                                "p_gold": p_gold, "brier": brier, "log_loss": log_loss,
                            })
        elif status in {"contract_invalid", "malformed_json"}:
            # a strict-format failure is COUNTED, never repaired and never retried
            strict_format_failures += 1
        per_item.append(record)

    n_correct = sum(requested_correct.values())
    usable_rows = [rec for rec in per_item
                   if rec["usable"] and rec["correct"] is not None]
    usable_correct = sum(1 for rec in usable_rows if rec["correct"])
    n_usable = sum(requested_usable.values())
    accuracy = (n_correct / n_requested) if n_requested else None
    doc: dict[str, Any] = {
        "score_schema_version": SCORE_SCHEMA_VERSION,
        "scoring_policy": SCORING_POLICY,
        "stage": stage,
        "run_dir": str(run_dir),
        "n_requested": n_requested,
        "n_expected": n_requested,
        "n_terminal_results": len(terminal),
        "n_correct": n_correct,
        "accuracy": accuracy,
        "wilson_95": (wilson_interval(n_correct, n_requested).to_dict()
                      if n_requested else None),
        "n_missing_terminal": n_requested - len(terminal),
        "missing_logical_ids": sorted(
            set(gold) - {rec["item_id"] for rec in per_item}),
        "n_usable_responses": n_usable,
        "usable_response_correct": usable_correct,
        "usable_response_accuracy": ((usable_correct / len(usable_rows))
                                     if usable_rows else None),
        "usable_response_wilson_95": (
            wilson_interval(usable_correct, len(usable_rows)).to_dict()
            if usable_rows else None),
        "n_usable_multi_question_excluded": n_usable - len(usable_rows),
        "usable_response_note": (
            "SEPARATE diagnostic over ok+usable responses only "
            "(n_usable_responses; multi-question chunk items are excluded "
            "from usable_response_accuracy because item-level correctness is "
            "not defined for them); never the headline denominator"),
        "finite_set_caveat": FINITE_SET_CAVEAT,
        "status_counts": status_counts,
        "strict_format_failures": strict_format_failures,
        "zero_probability_gold": zero_prob_gold if zero_prob_candidates else None,
        "zero_probability_gold_candidates": zero_prob_candidates or None,
        "proper_scores": {
            "n": len(proper_scores),
            "excluded_no_valid_probabilities": prob_exclusions,
            "brier_mean": _mean([p["brier"] for p in proper_scores]),
            "log_loss_mean": _mean([p["log_loss"] for p in proper_scores]),
            "note": ("proper scores only over attempts with usable, finite "
                     "probabilities (rotated items scored on probability keys "
                     "canonicalized through the recorded remapping); "
                     "text-only baselines contribute none"),
        },
        "by_group": {},
        "by_rotation_variant": {},
        "model_returned_values": sorted({str(r.get("model_returned")) for r in results
                                         if r.get("model_returned")}),
        "per_item": per_item,
    }
    group_flags: dict[str, list[bool]] = {}
    for item_id, is_correct in requested_correct.items():
        group_flags.setdefault(gold[item_id]["group"], []).append(bool(is_correct))
    for group, flags in sorted(group_flags.items()):
        doc["by_group"][group] = _accuracy_block(flags)
    variant_flags: dict[str, list[bool]] = {}
    for item_id in gold:
        if item_id not in rotations:
            continue
        variant = condition_of.get(item_id) or f"perm{rotations[item_id]['variant']}"
        variant_flags.setdefault(variant, []).append(bool(requested_correct[item_id]))
    for variant, flags in sorted(variant_flags.items()):
        doc["by_rotation_variant"][variant] = _accuracy_block(flags)
    doc["attempts_summary"] = {
        "n_attempts": len(attempts),
        "n_uncertain_attempts": len([a for a in attempts if a.get("uncertain")]),
        "usage_input_tokens_sum": _sum_field(attempts, "usage_input_tokens"),
        "usage_output_tokens_sum": _sum_field(attempts, "usage_output_tokens"),
        "attempts_without_usage": len([a for a in attempts
                                       if a.get("usage_input_tokens") is None]),
    }
    return doc


def _reject_duplicate_terminals(terminal: list[dict[str, Any]]) -> None:
    """Duplicate terminal logical ids (or item ids) are refused, never averaged."""
    seen_lids: set[str] = set()
    seen_items: set[str] = set()
    for result in terminal:
        lid = str(result.get("logical_request_id"))
        if lid in seen_lids:
            raise ScoringError(
                f"duplicate terminal logical_request_id {lid!r}; refusing to "
                f"double-count a request")
        seen_lids.add(lid)
        item_id = _split_logical_id(lid)[0]
        if item_id in seen_items:
            raise ScoringError(
                f"duplicate terminal item_id {item_id!r} (via {lid!r}); refusing "
                f"to double-count a request")
        seen_items.add(item_id)


def _canonical_pair(item_id: str, condition: str, choice: str,
                    gold_value: str | None, rotations: dict[str, dict[str, Any]]) -> tuple[str, str | None]:
    """Compare through the canonical label mapping for rotated items.

    ``remapping[k]`` is the new key holding the description originally under
    key ``k``. The canonical key of a predicted option is therefore
    ``inverse[choice]``, and the base item's gold letter is
    ``inverse[rotated_gold]``. The scorer verifies BOTH directions: the
    remapping must be a bijection, and the restored canonical comparison must
    agree with the direct comparison against the rotated item's own gold (which
    follows the description). Any disagreement is a hard error.
    """
    if item_id in rotations and gold_value is not None:
        remapping = rotations[item_id]["remapping"]
        inverse = {v: k for k, v in remapping.items()}
        if len(inverse) != len(remapping):
            raise ScoringError(
                f"rotation {item_id}: recorded remapping is not a bijection; refusing"
            )
        canonical_choice = inverse.get(choice)
        base_gold = inverse.get(gold_value)
        if canonical_choice is None or base_gold is None:
            raise ScoringError(
                f"rotation {item_id}: predicted key {choice!r} or rotated gold "
                f"{gold_value!r} not covered by the recorded remapping; refusing"
            )
        if (choice == gold_value) != (canonical_choice == base_gold):
            raise ScoringError(
                f"rotation {item_id}: canonical-label comparison ({canonical_choice!r} vs "
                f"{base_gold!r}) disagrees with the direct rotated-gold comparison; refusing"
            )
        return canonical_choice, base_gold
    return choice, gold_value


def _canonicalize_probabilities(probabilities: dict[str, float],
                                remapping: dict[str, str]) -> dict[str, float] | None:
    """Rotate probability keys back to canonical labels; None excludes.

    ``remapping[k]`` is the new key holding the description originally under
    key ``k``, so a probability reported under key ``k_new`` belongs to the
    canonical key ``inverse[k_new]``.  A key outside the recorded remapping
    makes the vector unscorable under the recorded rotation: an exclusion,
    never a silent re-key.
    """
    inverse = {v: k for k, v in remapping.items()}
    if len(inverse) != len(remapping):
        return None
    canonical: dict[str, float] = {}
    for key, value in probabilities.items():
        target = inverse.get(key)
        if target is None:
            return None
        canonical[target] = value
    return canonical


def _score_probabilities(probabilities: dict[str, float],
                         gold: str) -> tuple[float, float, float] | None:
    """Proper scores where probabilities are valid; None excludes the attempt.

    Validity: every option key present, finite, in [0, 1], sum ≈ 1. Never
    normalizes or clips silently — an invalid vector is an exclusion, counted.
    """
    values = []
    for key, value in probabilities.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return None
        if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            return None
        values.append(float(value))
    total = sum(values)
    if not math.isfinite(total) or abs(total - 1.0) > 5e-2:
        return None
    p_gold = float(probabilities.get(gold, 0.0))
    if not math.isfinite(p_gold):
        return None
    brier_doc = brier_multiclass(probabilities, gold)
    brier = brier_doc["brier_sum"]
    log_loss = _log_loss_clipped(p_gold)
    return p_gold, brier, log_loss


def _log_loss_clipped(p_true: float, epsilon: float = 1e-15) -> float:
    return -math.log(max(epsilon, min(1.0 - epsilon, p_true)))


def _accuracy_block(flags: list[bool]) -> dict[str, Any]:
    n = len(flags)
    interval = wilson_interval(sum(flags), n) if n else None
    return {
        "n": n,
        "correct": sum(flags),
        "accuracy": (sum(flags) / n) if n else None,
        "wilson_95": interval.to_dict() if interval else None,
        "finite_set_caveat": FINITE_SET_CAVEAT,
        "denominator_note": ("all requested items in this group/variant; "
                             "missing, invalid and unusable count as wrong"),
    }


def _split_logical_id(lid: str) -> tuple[str, str]:
    """``{condition}:{item_id}`` — the condition is the FIRST component.

    Rotation item ids themselves contain colons (``{qid}:perm{n}``), so the
    item id is everything after the first colon.
    """
    condition, sep, item_id = lid.partition(":")
    if not sep or not condition or not item_id:
        raise ScoringError(f"malformed logical_request_id {lid!r}")
    return item_id, condition


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise ScoringError(f"missing run artifact {path}")
    records = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        if line.strip():
            records.append(json.loads(line))
    return records


def _mean(values: list[float]) -> float | None:
    finite = [v for v in values if math.isfinite(v)]
    return (sum(finite) / len(finite)) if finite else None


def _sum_field(records: list[dict[str, Any]], field: str) -> int | None:
    values = [r.get(field) for r in records if isinstance(r.get(field), int)]
    return sum(values) if values else None


def preserve_original_aggregate(path: Path | str) -> dict[str, Any]:
    """Preserve an existing aggregate with versioned provenance, never in place.

    Copies ``<path>`` to ``<dir>/preserved-pre-policy-v1/<name>`` (refusing to
    overwrite an earlier preservation) and records the provenance entry in
    ``<dir>/aggregate_provenance.json``.  Returns the provenance entry.
    """
    source = Path(path)
    if not source.exists():
        raise ScoringError(f"no aggregate to preserve at {source}")
    preserved_dir = source.parent / "preserved-pre-policy-v1"
    preserved_dir.mkdir(parents=True, exist_ok=True)
    target = preserved_dir / source.name
    if not target.exists():
        target.write_bytes(source.read_bytes())
    entry = {
        "artifact": source.name,
        "preserved_as": str(target),
        "sha256": _sha256_bytes(target.read_bytes()),
        "scoring_version": PRE_POLICY_AGGREGATE_VERSION,
        "regenerated_as": {
            "path": source.name,
            "scoring_version": SCORE_SCHEMA_VERSION,
        },
    }
    provenance_path = source.parent / "aggregate_provenance.json"
    provenance: dict[str, Any] = {"schema": "aggregate-provenance-1.0.0", "entries": []}
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    entries = [e for e in provenance.get("entries", [])
               if e.get("artifact") != source.name]
    entries.append(entry)
    provenance["entries"] = sorted(entries, key=lambda e: e["artifact"])
    provenance_path.write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    return entry


def _sha256_bytes(data: bytes) -> str:
    from hashlib import sha256
    return sha256(data).hexdigest()


def write_score(doc: dict[str, Any], path: Path | str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")