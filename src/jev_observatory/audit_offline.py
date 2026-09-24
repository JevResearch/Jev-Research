"""Reproducible offline audit reconstruction (LEAD-GATE 2026-09-18, requirement A).

Loads immutable run artifacts (manifest.json, items.jsonl, results.jsonl) and
independently reconstructs:

* the paired BoolQ two-encoding table (accuracy, exact McNemar, binary
  P(yes)-vs-gold Brier with a paired percentile bootstrap that resamples base
  records and recomputes the full statistic);
* the mechanism-v3 paired signed P(yes) shifts using the *actual six repeat
  pairs* per anchor/arm as sample units (NOT the 15 within-baseline or 36
  cross-arm pairwise distances, which are dependent and were never independent
  observations), plus an exact sign-flip diagnostic over those six pairs.

Rules enforced here:

* fresh derived artifacts are written to an explicit output directory; the run
  directory itself is never modified;
* input file SHA-256 hashes, the analyzer version, the estimator definitions,
  the number of independent calls/blocks, and missing/invalid observation
  counts are recorded in every output;
* JSON is strict: a non-finite float is represented explicitly as
  ``{"nonfinite": "inf" | "-inf" | "nan"}`` and is never serialised as a bare
  ``Infinity``/``NaN`` literal;
* invalid or failed observations are never silently dropped: they are counted
  per condition/arm and, when present, the tables are labelled incomplete;
* all claims are labelled exploratory: these are already-seen observations and
  no new confirmatory labels are invented.

Historical manifest hashes and fingerprints describe the *raw stored artifacts*
of their run; they are not endorsements of any current interpretation and do
not transfer to newly generated specs.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any

ANALYZER_VERSION = "audit_offline-1.0.0"

# Saved-regression anchors from the prior offline audit of the paired BoolQ run
# (docs/review-gate/ source review; constants predate this module and are NOT
# recomputed by the code under test):
REGRESSION_PAIRED_BOOLQ = {
    "choice_brier": 0.0898544,
    "noul_brier": 0.0783124,
    "choice_correct": 449,
    "noul_correct": 448,
    "n_pairs": 500,
}


# ------------------------------------------------------------------ strict IO
def finite_or_marker(value: Any) -> Any:
    """Recursively map non-finite floats to explicit markers; JSON stays strict."""
    if isinstance(value, float):
        if math.isnan(value):
            return {"nonfinite": "nan"}
        if math.isinf(value):
            return {"nonfinite": "inf" if value > 0 else "-inf"}
        return value
    if isinstance(value, dict):
        return {str(k): finite_or_marker(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_or_marker(v) for v in value]
    return value


def assert_strict_json(value: Any, path: str = "$") -> None:
    """Raise unless every float in `value` is finite (no Infinity/NaN literals)."""
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite float at {path}: {value!r}")
    elif isinstance(value, dict):
        for key, item in value.items():
            assert_strict_json(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_strict_json(item, f"{path}[{index}]")


def dump_strict_json(value: Any, path: Path | str) -> None:
    assert_strict_json(finite_or_marker(value))
    Path(path).write_text(
        json.dumps(finite_or_marker(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_run_artifacts(run_dir: Path | str) -> dict[str, Any]:
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    results = _read_jsonl(run_dir / "results.jsonl")
    items = _read_jsonl(run_dir / "items.jsonl")
    return {
        "run_dir": str(run_dir),
        "manifest": manifest,
        "results": results,
        "items": {str(entry["id"]): entry for entry in items},
        "input_hashes": {
            name: sha256_file(run_dir / name)
            for name in ("manifest.json", "items.jsonl", "results.jsonl")
            if (run_dir / name).exists()
        },
    }


# ------------------------------------------------------------- paired BoolQ
def _finite_probability(value: Any) -> float | None:
    """A probability only if it is a finite number inside [0, 1]; never a
    default, never a silent coercion of missing/invalid values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        return None
    return number


def _extract_p_yes(record: dict[str, Any], item: dict[str, Any]) -> tuple[tuple[bool, float, bool] | None, str | None]:
    """((correct, p_yes, gold_yes), reason) for one binary yes/no observation.

    FAILS CLOSED (REVIEW-1 #6): a missing, non-finite or out-of-range
    probability is an invalid observation with a reason — it is NEVER replaced
    by a default such as 0.0 or 0.5.
    """
    if record.get("status") != "ok":
        return None, f"status={record.get('status')!r}"
    prediction = (record.get("predictions") or {}).get("answer")
    gold_value = (item.get("gold", {}).get("answer") or {})
    gold = gold_value.get("value") if isinstance(gold_value, dict) else gold_value
    if prediction is None or not prediction.get("usable", False):
        return None, "prediction missing or unusable"
    if prediction.get("type") == "choice":
        if not isinstance(gold, str):
            return None, "choice prediction without a string gold"
        probabilities = prediction.get("probabilities") or {}
        if "yes" not in probabilities:
            return None, "choice prediction missing the 'yes' probability"
        p_yes = _finite_probability(probabilities["yes"])
        if p_yes is None:
            return None, f"'yes' probability invalid: {probabilities['yes']!r}"
        return (prediction.get("choice") == gold, p_yes, gold == "yes"), None
    if prediction.get("type") == "noul":
        if not isinstance(gold, bool):
            return None, "noul prediction without a boolean gold"
        if prediction.get("noul") is None:
            return None, "noul prediction missing P(yes)"
        p_yes = _finite_probability(prediction["noul"])
        if p_yes is None:
            return None, f"noul P(yes) invalid: {prediction['noul']!r}"
        return ((p_yes >= 0.5) == gold, p_yes, gold), None
    return None, f"unknown prediction type {prediction.get('type')!r}"


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q / 100.0 * (len(ordered) - 1)))))
    return ordered[index]


def reconstruct_paired_boolq(
    run_dir: Path | str,
    *,
    n_bootstrap: int = 2000,
    seed: int = 7,
    question_id: str = "answer",
) -> dict[str, Any]:
    """Independently rebuild the paired two-encoding BoolQ table from raw rows.

    Estimator notes (recorded in the output):
    * accuracy: fraction correct among valid observations per condition; paired
      accuracy difference over shared base records with exact McNemar on the
      discordant counts;
    * Brier: mean squared error of P(yes) against the binary gold yes/no;
      paired difference with a percentile bootstrap that resamples *base
      records* (the underlying calls) and recomputes the full paired statistic
      per replicate — the pairwise per-record differences are never resampled
      as if they were the independent units.
    """
    art = load_run_artifacts(run_dir)
    # planned per-condition counts come from the run's own items.jsonl (REVIEW-1 #6:
    # planned-but-missing results must be counted, never silently ignored)
    planned: dict[str, int] = {}
    for item in art["items"].values():
        condition = str(item.get("condition", "")).split(";")[0]
        planned[condition] = planned.get(condition, 0) + 1
    by_condition: dict[str, dict[str, dict[str, Any]]] = {}
    invalid: dict[str, list[dict[str, str]]] = {}
    for record in art["results"]:
        condition = str(record.get("condition", "")).split(";")[0]
        base = record.get("cluster") or record.get("logical_request_id")
        extracted, reason = _extract_p_yes(record, art["items"].get(record["item_id"], {}))
        if extracted is None:
            invalid.setdefault(condition, []).append(
                {"item_id": str(record.get("item_id")), "reason": reason or "unknown"})
            continue
        # fail closed on duplicate per-condition observations: overwrite would
        # silently replace one observation with another
        condition_rows = by_condition.setdefault(condition, {})
        if base in condition_rows:
            raise ValueError(
                f"duplicate observation for condition {condition!r}, base {base!r}; "
                "refusing to overwrite an existing per-condition record"
            )
        condition_rows[base] = {
            "correct": extracted[0], "p_yes": extracted[1], "gold_yes": extracted[2],
        }
    conditions = sorted(set(by_condition) | set(planned))
    if len(conditions) != 2:
        raise ValueError(
            f"paired reconstruction needs exactly 2 conditions, found {conditions!r} "
            f"(invalid rows: { {k: len(v) for k, v in invalid.items()} })"
        )
    shared = sorted(set(by_condition.get(conditions[0], {})) & set(by_condition.get(conditions[1], {})))
    n_invalid_total = {k: len(v) for k, v in invalid.items()}
    n_unmatched = {
        c: len(by_condition.get(c, {})) - len(shared) for c in conditions
    }
    n_missing_results = {
        c: max(0, planned.get(c, 0)
               - len(by_condition.get(c, {})) - n_invalid_total.get(c, 0))
        for c in conditions
    }
    unavailable = not shared or any(
        not by_condition.get(c) for c in conditions
    )
    if unavailable:
        return {
            "analysis": "paired_boolq_two_encoding_reconstruction",
            "analyzer_version": ANALYZER_VERSION,
            "claim_type": "exploratory",
            "result_available": False,
            "unavailable_reason": (
                "empty paired sample: no shared base records with valid observations; "
                "no statistic is computed rather than dividing by zero"
            ),
            "run_dir": art["run_dir"],
            "input_hashes": art["input_hashes"],
            "n_planned_per_condition": planned,
            "n_invalid_or_missing_rows_total": n_invalid_total,
            "n_unmatched_pairs_per_condition": n_unmatched,
            "n_missing_results_per_condition": n_missing_results,
            "incomplete": True,
            "conditions": conditions,
            "per_condition": None,
            "paired": None,
        }

    per_condition: dict[str, Any] = {}
    for condition in conditions:
        rows = [by_condition[condition][base] for base in shared]
        per_condition[condition] = {
            "n_planned": planned.get(condition, 0),
            "n_valid": len(rows),
            "n_correct": sum(1 for r in rows if r["correct"]),
            "accuracy": (sum(1 for r in rows if r["correct"]) / len(rows)) if rows else None,
            "binary_brier_p_yes": (
                statistics.mean((r["p_yes"] - (1.0 if r["gold_yes"] else 0.0)) ** 2 for r in rows)
                if rows else None
            ),
            "n_invalid_or_missing_rows": n_invalid_total.get(condition, 0),
            "n_missing_results": n_missing_results.get(condition, 0),
            "n_unmatched_pairs": n_unmatched.get(condition, 0),
            "invalid_row_examples": invalid.get(condition, [])[:10],
        }

    incomplete = any(n_invalid_total.values()) or any(n_missing_results.values()) \
        or any(n_unmatched.values())
    first, second = conditions
    brier_diff: list[float] = []
    acc_pairs: list[tuple[bool, bool]] = []
    for base in shared:
        a, b = by_condition[first][base], by_condition[second][base]
        brier_diff.append(
            (a["p_yes"] - (1.0 if a["gold_yes"] else 0.0)) ** 2
            - (b["p_yes"] - (1.0 if b["gold_yes"] else 0.0)) ** 2
        )
        acc_pairs.append((a["correct"], b["correct"]))
    n = len(shared)
    acc_diff = (sum(1 for a, _ in acc_pairs if a) - sum(1 for _, b in acc_pairs if b)) / n
    b01 = sum(1 for a, b in acc_pairs if a and not b)
    b10 = sum(1 for a, b in acc_pairs if not a and b)

    rng = random.Random(seed)
    brier_boot: list[float] = []
    for _ in range(n_bootstrap):
        sample = rng.choices(range(n), k=n)
        brier_boot.append(statistics.mean(brier_diff[i] for i in sample))
    brier_boot.sort()

    from math import comb

    mcnemar_n = b01 + b10
    mcnemar_p = (
        1.0
        if mcnemar_n == 0
        else min(1.0, 2.0 * sum(comb(mcnemar_n, k) for k in range(0, min(b01, b10) + 1)) / 2 ** mcnemar_n)
    )

    return {
        "analysis": "paired_boolq_two_encoding_reconstruction",
        "analyzer_version": ANALYZER_VERSION,
        "claim_type": "exploratory",
        "exploratory_note": (
            "already-seen observations re-analysed offline; no new confirmatory label is created. "
            "Bootstrap intervals are uncertainty descriptions, not significance verdicts; a "
            "non-significant difference is not evidence of equivalence."
        ),
        "run_dir": art["run_dir"],
        "input_hashes": art["input_hashes"],
        "manifest_fingerprint": art["manifest"].get("fingerprint", {}),
        "provenance_note": (
            "manifest hashes and fingerprints describe the raw stored artifacts of this run only"
        ),
        "estimator": {
            "accuracy": "fraction correct per condition; paired difference with exact McNemar",
            "brier": "mean (P(yes) - gold_yes)^2 per condition; paired percentile bootstrap, "
                     "resampling base records and recomputing the full paired statistic",
            "n_bootstrap": n_bootstrap,
            "bootstrap_seed": seed,
        },
        "n_independent_calls_per_condition": {c: len(by_condition.get(c, {})) for c in conditions},
        "n_paired_base_records": n,
        "n_planned_per_condition": planned,
        "n_invalid_or_missing_rows_total": n_invalid_total,
        "n_missing_results_per_condition": n_missing_results,
        "n_unmatched_pairs_per_condition": n_unmatched,
        "incomplete": incomplete,
        "conditions": conditions,
        "per_condition": per_condition,
        "paired": {
            "accuracy_difference_first_minus_second": acc_diff,
            "discordant_counts": {"first_right_second_wrong": b01, "first_wrong_second_right": b10},
            "mcnemar_exact_p_two_sided": mcnemar_p,
            "brier_difference_first_minus_second": statistics.mean(brier_diff),
            "brier_difference_ci95": {
                "low": _percentile(brier_boot, 2.5),
                "high": _percentile(brier_boot, 97.5),
            },
        },
    }


# ------------------------------------------------------------- mechanism v3
def _anchor_question_id(item: dict[str, Any]) -> str | None:
    """Anchor qid from the stored item (a choice question with yes/no options).

    This is a lookup over recorded artifacts (manifest option_maps / items), not
    free-text parsing of condition strings.
    """
    for qid, raw in item.get("questions", {}).items():
        if raw.get("type") == "choice" and set(raw.get("criteria", {})) == {"yes", "no"}:
            return qid
    return None


def _p_yes(record: dict[str, Any], qid: str) -> float | None:
    prediction = (record.get("predictions") or {}).get(qid)
    if not prediction or prediction.get("type") != "choice" or not prediction.get("usable", False):
        return None
    probs = prediction.get("probabilities") or {}
    if "yes" not in probs:
        return None
    return _finite_probability(probs["yes"])


def reconstruct_mechanism_v3(
    run_dir: Path | str,
    *,
    reference_arm: str = "alone",
    treatment_arms: tuple[str, ...] = ("sib8", "renamed8", "reorder8"),
) -> dict[str, Any]:
    """Paired signed P(yes) shifts for mechanism v3, using the six repeat pairs.

    Sample units are the six per-anchor repeat *pairs* (treatment repeat r vs
    alone repeat r) — not the 15 within-baseline or 36 cross-arm pairwise
    distances, which are dependent transformations of the same 24 calls and
    were previously bootstrapped as if independent. The diagnostic is an exact
    sign-flip randomisation over the six paired signed shifts.

    Existing mechanism v3 fixed the arm dispatch order and renamed/reordered
    conditions also changed sibling content or anchor position; the design is
    therefore confounded and NO causal interpretation is supported. Labels:
    exploratory only.
    """
    art = load_run_artifacts(run_dir)
    # planned (anchor, arm, repeat) cells come from the run's own items.jsonl
    planned_cells: set[tuple[str, str, int]] = set()
    for item in art["items"].values():
        params = dict(
            part.split("=", 1)
            for part in str(item.get("condition", "")).split(";")
            if "=" in part
        )
        if params.get("anchor") and params.get("variant") and params.get("repeat"):
            planned_cells.add((params["anchor"], params["variant"], int(params["repeat"])))
    # group: (anchor, arm, repeat) -> p_yes ; anchor qid per item
    observations: dict[tuple[str, str, int], dict[str, Any]] = {}
    invalid: list[dict[str, Any]] = []
    for record in art["results"]:
        params = dict(
            part.split("=", 1)
            for part in str(record.get("condition", "")).split(";")
            if "=" in part
        )
        anchor, variant = params.get("anchor"), params.get("variant")
        repeat_raw = params.get("repeat")
        if anchor is None or variant is None or repeat_raw is None:
            continue
        repeat = int(repeat_raw)
        item = art["items"].get(record["item_id"], {})
        qid = _anchor_question_id(item)
        if record.get("status") != "ok" or qid is None:
            invalid.append({"item_id": record.get("item_id"), "status": record.get("status")})
            continue
        p_yes = _p_yes(record, qid)
        if p_yes is None:
            invalid.append({"item_id": record.get("item_id"), "reason": "no usable P(yes)"})
            continue
        key = (anchor, variant, repeat)
        # fail closed on duplicate per-block/per-arm observations: an overwrite
        # would silently replace one measured block with another
        if key in observations:
            raise ValueError(
                f"duplicate observation for anchor={anchor!r} arm={variant!r} repeat={repeat!r}; "
                "refusing to overwrite an existing per-block record"
            )
        observations[key] = {
            "p_yes": p_yes, "anchor_qid": qid, "item_id": record["item_id"],
        }

    repeats = sorted({r for (_, _, r) in observations})
    anchors = sorted({a for (a, _, _) in observations})
    per_anchor: dict[str, Any] = {}
    for anchor in anchors:
        arms: dict[str, Any] = {}
        for arm in (reference_arm, *treatment_arms):
            p_vals = [
                observations.get((anchor, arm, r), {}).get("p_yes")
                for r in repeats
            ]
            n_missing = sum(1 for p in p_vals if p is None)
            present = [p for p in p_vals if p is not None]
            arms[arm] = {
                "n_expected_repeats": len(repeats),
                "n_valid": len(present),
                "n_missing_or_invalid": n_missing,
                "mean_p_yes": statistics.mean(present) if present else None,
                "p_yes_by_repeat": p_vals,
            }
        shifts: dict[str, Any] = {}
        for arm in treatment_arms:
            paired = []
            for r in repeats:
                t = observations.get((anchor, arm, r), {}).get("p_yes")
                a = observations.get((anchor, reference_arm, r), {}).get("p_yes")
                if t is None or a is None:
                    continue
                paired.append({"repeat": r, "shift_signed": t - a})
            n_pairs = len(paired)
            diffs = [entry["shift_signed"] for entry in paired]
            mean_shift = statistics.mean(diffs) if diffs else None
            # exact sign-flip randomisation over the paired shifts (exploratory)
            p_flip = None
            if diffs and n_pairs <= 20:
                observed = abs(mean_shift)
                count = 0
                total = 2 ** n_pairs
                for mask in range(total):
                    sample = statistics.mean(
                        d if (mask >> i) & 1 == 0 else -d for i, d in enumerate(diffs)
                    )
                    if abs(sample) >= observed - 1e-12:
                        count += 1
                p_flip = count / total
            shifts[arm] = {
                "n_pairs": n_pairs,
                "n_missing_pairs": len(repeats) - n_pairs,
                "mean_signed_shift": mean_shift,
                "signed_shifts_by_repeat": paired,
                "sign_flip_exact_p_two_sided": p_flip,
            }
        per_anchor[anchor] = {
            "arms": arms,
            "paired_shifts_vs_reference": shifts,
            "n_independent_calls_per_arm": {
                arm: arms[arm]["n_valid"] for arm in arms
            },
        }

    any_invalid = bool(invalid) or any(
        arm["n_missing_or_invalid"] for anchor in per_anchor.values() for arm in anchor["arms"].values()
    )
    return {
        "analysis": "mechanism_v3_paired_signed_shift_reconstruction",
        "analyzer_version": ANALYZER_VERSION,
        "claim_type": "exploratory",
        "exploratory_note": (
            "six repeat pairs per anchor/arm; exact sign-flip randomisation; small n means wide "
            "uncertainty and NO equivalence claim is possible from a non-significant result"
        ),
        "design_caveat": (
            "mechanism v3 dispatched arms in a fixed order and the renamed/reordered arms also "
            "changed insertion positions or the anchor's key position; renaming, ordering and "
            "sibling presence are confounded, so these shifts have no causal interpretation"
        ),
        "run_dir": art["run_dir"],
        "input_hashes": art["input_hashes"],
        "manifest_fingerprint": art["manifest"].get("fingerprint", {}),
        "provenance_note": (
            "manifest hashes and fingerprints describe the raw stored artifacts of this run only"
        ),
        "estimator": {
            "unit": "paired repeat (treatment repeat r minus alone repeat r), one signed shift per block",
            "n_units": "6 per anchor/arm-pair (NOT the 15/36 dependent pairwise distances)",
            "diagnostic": "exact sign-flip randomisation over paired signed shifts",
        },
        "n_repeats": len(repeats),
        "n_planned_cells": len(planned_cells),
        "n_missing_planned_cells": len(planned_cells - set(observations)),
        "n_invalid_or_unusable_records": len(invalid),
        "invalid_record_examples": invalid[:10],
        "incomplete": any_invalid or bool(planned_cells - set(observations)),
        "anchors": per_anchor,
    }


# ------------------------------------------------------------------ CLI bundle
def build_audit_bundle(
    *,
    paired_run_dir: Path | str,
    mech_run_dir: Path | str,
    out_dir: Path | str,
) -> list[Path]:
    """Run both reconstructions and write fresh derived artifacts under `out_dir`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    paired = reconstruct_paired_boolq(paired_run_dir)
    mech = reconstruct_mechanism_v3(mech_run_dir)
    bundle = {
        "bundle": "lead-gate-offline-audit",
        "created_with": ANALYZER_VERSION,
        "strict_json": True,
        "inputs": {
            "paired_boolq_run": paired["run_dir"],
            "mechanism_v3_run": mech["run_dir"],
        },
    }
    for name, value in (
        ("boolq_paired_reconstruction.json", paired),
        ("mechanism_v3_reconstruction.json", mech),
        ("audit_bundle_index.json", bundle),
    ):
        path = out_dir / name
        dump_strict_json(value, path)
        outputs.append(path)
    return outputs
