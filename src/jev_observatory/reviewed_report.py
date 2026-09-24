"""Offline reviewed report (parent REVIEW-2).

Regenerates every number from the stored results/items/attempts under
``runs_reviewed/`` — never from cached metrics — and writes a strict-JSON
document plus a restrained markdown report. No provider access, no network.

Sections:
* ``fresh``      — per-family complete/error counts and direct gold comparison
  against INDEPENDENTLY verified labels (fresh_v3_verify re-derivation);
  no pooled causal comparisons of any kind.
* ``mechanism``  — per-block signed P(yes) differences S−A / N−S / O−S using
  the explicit sidecar ``measured_anchor_id`` mappings (never string parsing),
  exact paired sign-flip diagnostics, Holm across the fixed six contrasts,
  paired block bootstrap CIs, per-arm missing/invalid counts. Incomplete
  contrasts are excluded EXPLICITLY; equivalence is never declared when
  outcomes are missing or the interval is not entirely inside ±0.03.
* ``batching``   — per-block whole-workload wall times (n=12 workloads per arm,
  NOT 108 independent measurements), sum-of-usage per workload, paired block
  ratio/shift bootstrap uncertainty, failures and answer disagreement.
* ``billing``    — all attempted stages combined; unknown-usage attempts are
  counted conservatively and marked as unknown charges.

Strict JSON: ``json.dumps(..., allow_nan=False)`` — non-finite values would
raise rather than silently emit ``Infinity``.
"""

from __future__ import annotations

import json
import math
import random
import statistics
from pathlib import Path
from typing import Any

from .datasets.fresh_v3_verify import verify_all
from .mech_dryrun import contrast_summary, holm_stepdown
from .manifest import utc_now


def _load_freeze_verified(root: Path) -> dict[str, Any]:
    """Load the freeze doc (integrity-checked) and verify every frozen source
    spec + sidecar mapping hash against it (REVIEW-3 #1: before execution AND
    report). No provider access; pure file reads."""
    from .reviewed_executor import load_freeze, verify_frozen_sources

    freeze = load_freeze(root)
    verify_frozen_sources(freeze)
    return freeze


def _find_run(root: Path, stage: str, freeze: dict[str, Any]) -> Path:
    """Resolve the run directory FROM the freeze id, never lexicographic glob
    (REVIEW-3 #1: an arbitrary last-matching directory must never be analysed)."""
    run_id = f"rev-{stage}-{freeze['deterministic_sha256'][:12]}"
    path = root / run_id
    if not path.exists():
        raise FileNotFoundError(f"no reviewed run directory for stage {stage!r} at {path} "
                                "(resolved from the freeze id)")
    return path

REPORT_VERSION = "reviewed-report-1.0.0"
PRICE_PER_MTOKEN_INPUT = 0.042  # vendor quote, S8; not a guarantee
MECH_BLOCK_COUNT = 12
PRACTICAL_TOLERANCE = 0.03
BATCH_BOOTSTRAP = 4000
BATCH_BOOTSTRAP_SEED = 0


def _strict_dump(doc: dict[str, Any]) -> str:
    return json.dumps(doc, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------------- fresh
def _fresh_section(run_dir: Path) -> dict[str, Any]:
    items = _read_jsonl(run_dir / "items.jsonl")
    results = {r["item_id"]: r for r in _read_jsonl(run_dir / "results.jsonl")}
    verification = verify_all(items)
    verified_ids = {entry["id"] for entry in verification.get("results", [])}
    per_family: dict[str, Any] = {}
    for item in items:
        family = item["group"]
        bucket = per_family.setdefault(family, {
            "n_planned": 0, "n_complete": 0, "n_contract_invalid": 0,
            "n_error_or_missing": 0, "n_answer_unusable": 0,
            "n_label_verification_failures": 0, "n_scored": 0, "n_correct": 0,
        })
        bucket["n_planned"] += 1
        result = results.get(item["id"])
        if result is None:
            bucket["n_error_or_missing"] += 1
            continue
        if result.get("status") != "ok":
            if result.get("status") == "contract_invalid":
                bucket["n_contract_invalid"] += 1
            else:
                bucket["n_error_or_missing"] += 1
            continue
        bucket["n_complete"] += 1  # status ok; counted even if scoring fails below
        if item["id"] not in verified_ids:
            bucket["n_label_verification_failures"] += 1
            continue
        prediction = (result.get("predictions") or {}).get("answer")
        if not prediction or not prediction.get("usable"):
            bucket["n_answer_unusable"] += 1
            continue
        expected = item["gold"]["answer"]["value"]
        if item["group"] == "fresh_v3_dates":
            # independently re-derived boolean label vs the reported P(yes)
            detail = next(e for e in verification["results"] if e["id"] == item["id"])
            actual = bool(prediction["noul"] >= 0.5)
            correct = actual == bool(detail["same_month"])
        else:
            correct = prediction.get("choice") == expected
        bucket["n_scored"] += 1
        bucket["n_correct"] += int(bool(correct))
    for family, bucket in per_family.items():
        bucket["accuracy"] = (
            round(bucket["n_correct"] / bucket["n_scored"], 6) if bucket["n_scored"] else None
        )
    return {
        "run_id": run_dir.name,
        "n_items": len(items),
        "label_verification": {
            "n_verified": verification["n_verified"],
            "n_failures": verification["n_failures"],
            "verifier": verification["verifier"],
            "all_labels_verified": verification["all_labels_verified"],
        },
        "per_family": per_family,
        "note": ("direct comparison to independently verified labels per family; "
                 "no pooled Choice-vs-Noul causal comparisons; these items do not "
                 "establish general capability claims"),
    }


# --------------------------------------------------------------- mechanism
def _p_yes(result: dict[str, Any], measured_anchor_id: str) -> float | None:
    prediction = (result.get("predictions") or {}).get(measured_anchor_id)
    if not prediction or not prediction.get("usable"):
        return None
    probs = prediction.get("probabilities")
    if not isinstance(probs, dict) or "yes" not in probs:
        return None
    value = probs["yes"]
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) \
            or not 0.0 <= value <= 1.0:
        return None
    return float(value)


def _mechanism_section(run_dir: Path, mappings_path: Path) -> dict[str, Any]:
    items = _read_jsonl(run_dir / "items.jsonl")
    results = {r["item_id"]: r for r in _read_jsonl(run_dir / "results.jsonl")}
    mappings = _read_json(mappings_path)
    entry_by_item = {e["item_id"]: e for e in mappings["items"]}
    if len(entry_by_item) != len(items) or set(entry_by_item) != {item["id"] for item in items}:
        raise ValueError("mechanism sidecar mappings do not cover the stored items exactly")

    per_arm: dict[str, dict[str, int]] = {}
    p_yes: dict[tuple[int, str, str], float] = {}  # (block, anchor, arm) -> P(yes)
    for item in items:
        entry = entry_by_item[item["id"]]
        arm_bucket = per_arm.setdefault(entry["arm"], {
            "n_planned": 0, "n_complete": 0, "n_contract_invalid": 0,
            "n_error_or_missing": 0, "n_invalid_p_yes": 0,
        })
        arm_bucket["n_planned"] += 1
        result = results.get(item["id"])
        if result is None or result.get("status") != "ok":
            if result is not None and result.get("status") == "contract_invalid":
                arm_bucket["n_contract_invalid"] += 1
            else:
                arm_bucket["n_error_or_missing"] += 1
            continue
        value = _p_yes(result, entry["measured_anchor_id"])
        if value is None:
            arm_bucket["n_invalid_p_yes"] += 1
            continue
        arm_bucket["n_complete"] += 1
        key = (entry["block"], entry["anchor"], entry["arm"])
        if key in p_yes:
            # duplicate arm-block observations are never silently overwritten
            raise ValueError(
                f"duplicate mechanism observation for block/anchor/arm {key}; "
                "refusing to overwrite an existing paired observation"
            )
        p_yes[key] = value

    contrasts: dict[str, Any] = {}
    p_values: dict[str, float] = {}
    excluded: list[str] = []
    for anchor in mappings["anchors"]:
        for treatment, reference in (("siblings", "alone"), ("renamed", "siblings"),
                                     ("reordered", "siblings")):
            name = f"{anchor}:{treatment}-{reference}"
            shifts = []
            for block in range(MECH_BLOCK_COUNT):
                t = p_yes.get((block, anchor, treatment))
                r = p_yes.get((block, anchor, reference))
                if t is None or r is None:
                    continue
                shifts.append({"block": block, "shift_signed": t - r})
            n_complete = len(shifts)
            complete = n_complete == MECH_BLOCK_COUNT
            if not complete:
                excluded.append({"contrast": name, "n_blocks_complete": n_complete,
                                 "n_blocks_missing": MECH_BLOCK_COUNT - n_complete})
            summary = contrast_summary(shifts, n_blocks=MECH_BLOCK_COUNT)
            if not complete:
                # Incomplete contrast: exploratory estimates ONLY, reported as
                # incomplete, with NO equivalence verdict of any kind.
                summary["entire_interval_within_tolerance_0.03"] = None
                summary["equivalence_assessment"] = "unavailable_incomplete_contrast"
                summary["status"] = "incomplete"
                summary["note"] = (
                    "incomplete contrast: exploratory estimates only; NO equivalence "
                    "verdict; excluded from measured Holm p-values (a conservative p=1 "
                    "placeholder is contributed to the FIXED six-contrast family for "
                    "correction only, it is NOT a measured p)"
                )
            if summary.get("p_sign_flip_exploratory") is not None and complete:
                p_values[name] = summary["p_sign_flip_exploratory"]
            contrasts[name] = summary
    # Holm multiplicity family stays FIXED at six: an unavailable/incomplete
    # contrast contributes a conservative p=1 placeholder to the CORRECTION only
    # (never presented as a measured p-value).
    holm_input = {name: p_values.get(name, 1.0) for name in contrasts}
    adjusted = holm_stepdown(holm_input)
    for name, summary in contrasts.items():
        summary["p_holm_adjusted_exploratory"] = adjusted.get(name)
        if name not in p_values:
            summary["holm_input_p"] = 1.0
            summary["holm_note"] = ("p=1 conservative placeholder used in the fixed-six "
                                    "Holm correction only; not a measured p-value")
    return {
        "run_id": run_dir.name,
        "mappings_source": str(mappings_path),
        "n_blocks_design": MECH_BLOCK_COUNT,
        "per_arm_counts": per_arm,
        "contrasts": contrasts,
        "excluded_incomplete_contrasts": excluded,
        "holm_family": "fixed six contrasts; unavailable contrasts contribute a conservative "
                       "p=1 placeholder to the correction only (never a measured p)",
        "note": ("exploratory paired block analysis; a non-significant p is NOT equivalence; "
                 "equivalence would require the ENTIRE interval inside ±0.03 AND a complete "
                 "contrast; incomplete contrasts are excluded, never imputed"),
    }


# ---------------------------------------------------------------- batching
def _bootstrap_paired(pairs: list[tuple[float, float]], statistic, n: int = BATCH_BOOTSTRAP,
                      seed: int = BATCH_BOOTSTRAP_SEED) -> dict[str, Any] | None:
    if not pairs:
        return None
    values = [statistic(a, b) for a, b in pairs]
    # never emit None/non-finite statistics: the inputs are pre-filtered to
    # positive finite values, so a None here is a caller bug, not a number
    if any(v is None or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        raise ValueError("bootstrap statistic produced a None or non-finite value")
    rng = random.Random(seed)
    boots = []
    for _ in range(n):
        sample = rng.choices(values, k=len(values))
        boots.append(statistics.mean(sample))
    boots.sort()
    return {
        "n_pairs": len(values),
        "point": round(statistics.mean(values), 6),
        "ci95": {"low": round(boots[int(0.025 * n)], 6),
                 "high": round(boots[min(n - 1, int(0.975 * n))], 6)},
    }


def _wall_time_usable(w: dict[str, Any]) -> tuple[bool, list[str]]:
    """Timing estimand = ALL-SUCCESS FULL workloads. A workload contributes its
    wall time only when it is complete, error-free, covers the full workload
    (never resumed leftovers), and has a positive finite wall time."""
    reasons: list[str] = []
    if not w.get("complete"):
        reasons.append("incomplete")
    if int(w.get("n_errors") or 0) > 0:
        reasons.append("has_dispatch_errors")
    if w.get("wall_seconds_is_full_workload") is False:
        reasons.append("wall_time_not_full_workload_resumed_partial")
    wall = w.get("wall_seconds")
    if not isinstance(wall, (int, float)) or isinstance(wall, bool) \
            or not math.isfinite(wall) or wall <= 0:
        reasons.append("wall_time_not_positive_finite")
    return (not reasons), reasons


def _batching_section(run_dir: Path) -> dict[str, Any]:
    workloads = _read_jsonl(run_dir / "workloads.jsonl")
    results = {r["logical_request_id"]: r for r in _read_jsonl(run_dir / "results.jsonl")}
    by_mode: dict[str, dict[str, Any]] = {}
    for w in workloads:
        mode = w["mode"]
        bucket = by_mode.setdefault(mode, {
            "n_workloads": 0, "n_complete": 0, "n_incomplete": 0,
            "n_workloads_with_errors": 0, "n_success_full_workloads": 0,
            "n_omitted_from_timing": 0, "omitted_from_timing": [],
            "timing_workload_wall_seconds": [],
            "usage_input_tokens_sum": 0, "usage_output_tokens_sum": 0,
            "usage_unknown_attempts": 0, "n_dispatch_errors": 0,
            "terminal_http_error_results": 0, "terminal_error_results": 0,
        })
        bucket["n_workloads"] += 1
        bucket["n_complete"] += int(bool(w.get("complete")))
        bucket["n_incomplete"] += int(not w.get("complete"))
        bucket["n_workloads_with_errors"] += int(int(w.get("n_errors") or 0) > 0)
        bucket["n_dispatch_errors"] += w.get("n_errors") or 0
        bucket["usage_input_tokens_sum"] += w.get("usage_input_tokens_sum") or 0
        bucket["usage_output_tokens_sum"] += w.get("usage_output_tokens_sum") or 0
        bucket["usage_unknown_attempts"] += w.get("usage_unknown_attempts") or 0
        # terminal HTTP errors stay DISTINCT from successful work
        for lid in w.get("item_ids", []):
            result = results.get(lid)
            if result is not None and result.get("status") == "http_error":
                bucket["terminal_http_error_results"] += 1
            elif result is not None and result.get("status") not in (None, "ok"):
                bucket["terminal_error_results"] += 1
        usable, reasons = _wall_time_usable(w)
        if usable:
            bucket["timing_workload_wall_seconds"].append(w["wall_seconds"])
            bucket["n_success_full_workloads"] += 1
        else:
            bucket["n_omitted_from_timing"] += 1
            bucket["omitted_from_timing"].append(
                {"workload_id": w.get("workload_id"), "reasons": reasons})
    for mode, bucket in by_mode.items():
        walls = bucket.pop("timing_workload_wall_seconds")
        bucket["wall_seconds_per_workload"] = walls
        bucket["wall_seconds_mean"] = round(statistics.mean(walls), 6) if walls else None
        bucket["timing_estimand"] = ("all-success FULL workloads (complete, error-free, "
                                    "full-workload wall time, positive finite)")

    # paired block comparisons (same block = matched question content); ONLY
    # timing-eligible workloads are paired, and observed n_pairs is reported
    wall: dict[tuple[str, int], float] = {}
    for w in workloads:
        usable, _ = _wall_time_usable(w)
        if usable:
            wall[(w["mode"], w["block"])] = w["wall_seconds"]
    paired: dict[str, Any] = {}
    for a, b in (("sequential", "batched"), ("concurrent", "batched"), ("concurrent", "sequential")):
        usable_blocks = [blk for blk in range(12)
                         if (a, blk) in wall and (b, blk) in wall]
        omitted_blocks = [blk for blk in range(12) if blk not in usable_blocks]
        pairs = [(wall[(a, blk)], wall[(b, blk)]) for blk in usable_blocks]
        paired[f"{a}_vs_{b}"] = {
            "n_pairs_observed": len(pairs),
            "omitted_blocks": omitted_blocks,
            "ratio_a_over_b": _bootstrap_paired(pairs, lambda x, y: x / y),
            "difference_a_minus_b": _bootstrap_paired(pairs, lambda x, y: x - y),
            "note": ("paired over blocks; n=12 workloads per arm, not 108 independent "
                     "measurements; only all-success full workloads are paired"),
        }

    # answer disagreement across modes, per (block, question). Comparison
    # scope is HONEST: choice keys and noul |Δp| only; Score answers are
    # counted but NOT compared (no Score agreement metric is implemented).
    # Only successful (status ok) results are compared; terminal errors stay
    # out of the agreement tally.
    disagreement = {"n_compared": 0, "n_choice_key_disagreement": 0,
                    "noul_abs_diff_sum": 0.0, "noul_n": 0, "n_missing_side": 0,
                    "n_score_pairs_not_compared": 0,
                    "comparison_scope": ("choice key equality and noul |Δp| only; Score "
                                         "answers are counted but not compared because no "
                                         "Score agreement metric is implemented"),
                    "successful_results_only": True}
    item_to_meta = {}
    for w in workloads:
        for lid in w["item_ids"]:
            item_to_meta[lid] = (w["mode"], w["block"])
    sep_by_key: dict[tuple[str, int, str], dict[str, Any]] = {}
    for lid, meta in item_to_meta.items():
        if meta[0] == "batched":
            continue
        result = results.get(lid)
        if result is None or result.get("status") != "ok":
            continue
        sep_by_key[(meta[0], meta[1], _qid_of(lid))] = result
    batched_results = {}
    for lid, meta in item_to_meta.items():
        if meta[0] == "batched" and lid in results and results[lid].get("status") == "ok":
            batched_results[meta[1]] = results[lid]
    qids = ["technical", "billing_issue", "security_concern", "frustration"]
    for block in range(12):
        batched = batched_results.get(block)
        if batched is None:
            continue
        for qid in qids:
            bp = (batched.get("predictions") or {}).get(qid)
            if bp is None:
                continue
            for mode in ("sequential", "concurrent"):
                sep = sep_by_key.get((mode, block, qid))
                if sep is None:
                    disagreement["n_missing_side"] += 1
                    continue
                sp = (sep.get("predictions") or {}).get(qid)
                if sp is None:
                    disagreement["n_missing_side"] += 1
                    continue
                if bp.get("type") == "score" or sp.get("type") == "score":
                    disagreement["n_score_pairs_not_compared"] += 1
                    continue
                disagreement["n_compared"] += 1
                if bp.get("type") == "choice":
                    if bp.get("choice") != sp.get("choice"):
                        disagreement["n_choice_key_disagreement"] += 1
                elif bp.get("type") == "noul":
                    if isinstance(bp.get("noul"), (int, float)) and isinstance(sp.get("noul"), (int, float)):
                        disagreement["noul_abs_diff_sum"] += abs(bp["noul"] - sp["noul"])
                        disagreement["noul_n"] += 1
    if disagreement["noul_n"]:
        disagreement["noul_abs_diff_mean"] = round(disagreement["noul_abs_diff_sum"] / disagreement["noul_n"], 6)
    return {
        "run_id": run_dir.name,
        "per_mode": by_mode,
        "paired_block_comparisons": paired,
        "answer_disagreement_vs_batched": disagreement,
        "note": ("cold-client workload comparison with true whole-workload monotonic wall "
                 "time; NOT model-compute timing and NOT a warm-cache benchmark; per-call "
                 "latencies are secondary and stored per attempt"),
    }


def _qid_of(logical_request_id: str) -> str:
    # logical id = "<condition>:<item_id>"; item id = "<prefix>-bXX-<qid>"
    item_id = logical_request_id.rsplit(":", 1)[-1]
    return item_id.split("-", 2)[2]


# ----------------------------------------------------------------- billing
def _billing_section(root: Path, freeze: dict[str, Any]) -> dict[str, Any]:
    stages: dict[str, Any] = {}
    total_reported = 0
    total_unknown = 0
    total_estimated_unknown = 0
    total_attempts = 0
    for stage in ("fresh_test", "mechanism", "batching"):
        run_dir = _find_run(root, stage, freeze)  # resolved FROM the freeze id
        attempts = _read_jsonl(run_dir / "attempts.jsonl")
        reported = sum(int(a["usage_input_tokens"]) for a in attempts
                       if a.get("usage_input_tokens") is not None)
        unknown = [a for a in attempts if a.get("usage_input_tokens") is None]
        estimated_unknown = sum(int(a.get("estimated_input_tokens") or 0) for a in unknown)
        stages[stage] = {
            "run_id": run_dir.name,
            "n_attempts": len(attempts),
            "reported_input_tokens": reported,
            "n_attempts_unknown_usage": len(unknown),
            "estimated_input_tokens_unknown": estimated_unknown,
        }
        total_attempts += len(attempts)
        total_reported += reported
        total_unknown += len(unknown)
        total_estimated_unknown += estimated_unknown
    return {
        "per_stage": stages,
        "combined": {
            "n_attempts": total_attempts,
            "reported_input_tokens": total_reported,
            "n_attempts_unknown_usage": total_unknown,
            "estimated_input_tokens_unknown": total_estimated_unknown,
            "reported_cost_usd": round(total_reported * PRICE_PER_MTOKEN_INPUT / 1_000_000.0, 6),
            "unknown_charge_ceiling_usd": round(
                total_estimated_unknown * PRICE_PER_MTOKEN_INPUT / 1_000_000.0, 6),
            "note": ("input-token quote only (S8); server is the tokenisation authority; "
                     "unknown-usage attempts are billed-blind and counted conservatively"),
        },
    }


# ------------------------------------------------------------------- build
def build_reviewed_report(root: str | Path = "runs_reviewed",
                          out_dir: str | Path | None = None) -> tuple[dict[str, Any], Path, Path]:
    """Regenerate the numeric artifacts + markdown report FROM stored artifacts."""
    root = Path(root)
    state_path = root / "executor_state.json"
    if state_path.exists() and _read_json(state_path).get("dry_run") is True:
        raise ValueError(
            "executor_state.json records a dry run (nothing dispatched); the reviewed "
            "report is only meaningful after the gated live run"
        )
    freeze = _load_freeze_verified(root)  # integrity + frozen source/mapping hashes
    mech_run = _find_run(root, "mechanism", freeze)
    mech_mappings = Path(freeze["stages"]["mechanism"]["mappings_path"])
    doc: dict[str, Any] = {
        "report_version": REPORT_VERSION,
        "generated_at": utc_now(),
        "claim_type": "exploratory",
        "freeze_sha256": freeze["deterministic_sha256"],
        "fresh": _fresh_section(_find_run(root, "fresh_test", freeze)),
        "mechanism": _mechanism_section(mech_run, mech_mappings),
        "batching": _batching_section(_find_run(root, "batching", freeze)),
        "billing": _billing_section(root, freeze),
    }
    out_path = Path(out_dir) if out_dir else root / "derived"
    out_path.mkdir(parents=True, exist_ok=True)
    json_path = out_path / "reviewed_report.json"
    json_path.write_text(_strict_dump(doc), encoding="utf-8")
    md_path = out_path / "reviewed_report.md"
    md_path.write_text(_markdown(doc), encoding="utf-8")
    return doc, json_path, md_path


def _markdown(doc: dict[str, Any]) -> str:
    lines: list[str] = [
        "# Reviewed exploratory run — offline report\n",
        f"Generated: {doc['generated_at']} — exploratory; test results describe THESE items only.\n",
    ]
    fresh = doc["fresh"]
    lines.append("## Fresh TEST items (frozen, never tuned on)\n")
    lines.append(f"- label verification: {fresh['label_verification']['n_verified']}/{fresh['n_items']} "
                 f"verified, {fresh['label_verification']['n_failures']} failures")
    for family, bucket in fresh["per_family"].items():
        lines.append(f"- {family}: planned {bucket['n_planned']}, complete {bucket['n_complete']}, "
                     f"invalid {bucket['n_contract_invalid']}, error/missing {bucket['n_error_or_missing']}, "
                     f"scored {bucket['n_scored']}, accuracy {bucket['accuracy']}")
    mech = doc["mechanism"]
    lines.append("\n## Controlled mechanism (paired blocks, exploratory)\n")
    for name, summary in mech["contrasts"].items():
        lines.append(f"- {name}: n_blocks={summary['n_blocks_contributing']}/12, "
                     f"mean shift {summary['mean_signed_shift']}, "
                     f"sign-flip p {summary['p_sign_flip_exploratory']}, "
                     f"Holm p {summary.get('p_holm_adjusted_exploratory')}")
    if mech["excluded_incomplete_contrasts"]:
        lines.append(f"- excluded incomplete contrasts: {mech['excluded_incomplete_contrasts']}")
    batch = doc["batching"]
    lines.append("\n## Batching comparison (cold-client workloads)\n")
    for mode, bucket in batch["per_mode"].items():
        lines.append(f"- {mode}: n={bucket['n_workloads']} workloads, complete {bucket['n_complete']}, "
                     f"mean whole-workload wall {bucket['wall_seconds_mean']}s")
    bill = doc["billing"]["combined"]
    lines.append("\n## Billing (all attempted stages)\n")
    lines.append(f"- attempts {bill['n_attempts']}, reported input tokens {bill['reported_input_tokens']}, "
                 f"unknown-usage attempts {bill['n_attempts_unknown_usage']} "
                 f"(ceiling ${bill['unknown_charge_ceiling_usd']})")
    lines.append("\nLimits: two anchors and three narrowly generated families do not establish "
                 "general model claims; batching numbers are cold-client workload comparisons, "
                 "not model-compute timing or warm-cache benchmarks.\n")
    return "\n".join(lines)
