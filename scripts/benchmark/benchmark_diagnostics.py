#!/usr/bin/env python3
"""Reproducible calibration + paired diagnostics from saved root artifacts.

Offline and deterministic (fixed bootstrap seed; no network; no model calls).
Reads the ORIGINAL run artifacts under runs_benchmark*/ (results.jsonl,
items.jsonl, rotations.json, runs_matched/freeze, runs_matched_cheap/) and
writes machine-readable diagnostics to data_report/benchmark_diagnostics/:

* calibration.json — per stage: all-requested headline accuracy, usable
  response accuracy/count, selected-answer displayed probability vs
  correctness, mean top displayed probability (explicitly NOT "predicted
  greedy accuracy" — a mean displayed p is not an accuracy estimate),
  10 equal-width reliability bins with counts and ECE (with the pre-rounded
  display-probability caveat), Brier/log loss and zero-gold counts.  For the
  rotation stage probability KEYS are canonicalized through the recorded
  remapping first.
* paired_diagnostics.json — MATH-500 intersection pairing (224 original
  shared items; MCQ vs digit-readout), the paired MMLU matched subset
  (Jev 84.0% on 1,000 paired items), and the rotation clustering account
  (419 scored observations on 140 unique base items) with a cluster-aware
  comparison and an explicit qualification of the naive independent McNemar
  (p=0.83 is NOT an equivalence proof).
* arc_payload_audit.json — duplicate outbound payload pairs in the recorded
  ARC-AGI-2 cell runs (7 pairs with different gold in the choice stage) and
  the adapter fix + regression test reference.

  python scripts/benchmark/benchmark_diagnostics.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import glob as _glob
import json
import random
from collections import defaultdict

from jev_observatory.metrics import ece_equal_width, reliability

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data_report" / "benchmark_diagnostics"

STAGE_RUNS = {
    "mmlu_full": "runs_benchmark/bench-mmlu_full-*",
    "option_rotations": "runs_benchmark/bench-option_rotations-*",
    "arc_test": "runs_benchmark/bench-arc_test-*",
    "math500_choice": "runs_benchmark_ext/bench-math500_choice-*",
    "gpqa_diamond": "runs_benchmark_ext2/bench-gpqa_diamond-*",
    "hle_text_mc": "runs_benchmark_ext2/bench-hle_text_mc-*",
}
MATH500_SCORE_RUN = "runs_benchmark_ext/bench-math500_score-*"
BOOTSTRAP_SEED = 20260930
BOOTSTRAP_N = 10_000

ROUNDING_CAVEAT = (
    "displayed probabilities arrive pre-rounded (2 decimals) from the server; "
    "bin membership, mean probabilities and ECE carry up to +/-0.005 per-class "
    "rounding uncertainty; all aggregates here are reported unrounded except "
    "where explicitly noted")
MEAN_PMAX_NOTE = (
    "mean top displayed probability is the model's displayed confidence of its "
    "greedy choice; the mean SELECTED-answer displayed probability is a "
    "predicted confidence and its mean IS the model-implied expected accuracy "
    "of taking that choice — compare it with the observed accuracy to test "
    "calibration.  The distinct caveat is that mean p(gold) is NOT 'predicted "
    "greedy accuracy'; the reliability bins below test calibration directly")


def _run_dir(pattern: str) -> Path:
    hits = sorted(_glob.glob(str(ROOT / pattern)))
    if not hits:
        raise SystemExit(f"no run directory matches {pattern}")
    return Path(hits[0])


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n")
            if line.strip()]


def _gold_map(run_dir: Path) -> dict[str, str]:
    gold = {}
    for record in _read_jsonl(run_dir / "items.jsonl"):
        value = None
        for payload in (record.get("gold") or {}).values():
            if isinstance(payload, dict) and payload.get("value") is not None:
                value = str(payload["value"])
                break
        gold[str(record["id"])] = value
    return gold


def _rotations(run_dir: Path) -> dict:
    path = run_dir / "rotations.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _canonical_probs(probs: dict, remapping: dict) -> dict | None:
    inverse = {v: k for k, v in remapping.items()}
    out = {}
    for key, value in probs.items():
        target = inverse.get(key)
        if target is None:
            return None
        out[target] = value
    return out


def _canonical_label(item_id: str, key: str, rotations: dict) -> str:
    """Canonical label of a rotated key (identity for non-rotated items)."""
    if item_id in rotations:
        inverse = {v: k for k, v in rotations[item_id]["remapping"].items()}
        return inverse.get(key, key)
    return key


def _choice_rows(run_dir: Path) -> list[dict]:
    """Per-request rows: correctness + selected/top displayed probabilities.

    For rotated items the choice and the probability KEYS are canonicalized
    through the recorded remapping before anything is compared or looked up.
    """
    gold = _gold_map(run_dir)
    rotations = _rotations(run_dir)
    rows = []
    for result in _read_jsonl(run_dir / "results.jsonl"):
        if result.get("terminal") is not True:
            continue
        item_id = str(result["item_id"])
        answer = next(iter((result.get("predictions") or {}).values()), None) or {}
        usable = bool(answer.get("usable")) and answer.get("choice") is not None
        probs = answer.get("probabilities")
        if item_id in rotations and isinstance(probs, dict):
            probs = _canonical_probs(probs, rotations[item_id]["remapping"])
        valid_probs = (isinstance(probs, dict) and probs
                       and all(isinstance(v, (int, float)) for v in probs.values()))
        gold_canonical = _canonical_label(item_id, gold.get(item_id), rotations) \
            if gold.get(item_id) is not None else None
        row = {
            "item_id": item_id,
            "base_item": item_id.split(":")[0],
            "usable": usable,
            "correct": bool(usable and gold_canonical is not None
                            and _canonical_label(item_id, answer.get("choice"),
                                                 rotations) == gold_canonical),
            "p_selected": None,
            "p_max": None,
            "p_gold": None,
        }
        if valid_probs:
            choice = _canonical_label(item_id, answer.get("choice"), rotations)
            row["p_selected"] = float(probs.get(choice, 0.0))
            row["p_max"] = max(float(v) for v in probs.values())
            if gold_canonical is not None:
                row["p_gold"] = float(probs.get(gold_canonical, 0.0))
        rows.append(row)
    return rows


def _calibration_block(stage: str, run_dir: Path, rows: list[dict]) -> dict:
    n_requested = len(_gold_map(run_dir))
    usable = [r for r in rows if r["usable"]]
    n_correct = sum(1 for r in rows if r["correct"])
    probs_rows = [r for r in usable if r["p_selected"] is not None]
    correct_rows = [r for r in probs_rows if r["correct"]]
    wrong_rows = [r for r in probs_rows if not r["correct"]]
    pairs = [(r["p_selected"], r["correct"]) for r in probs_rows]
    zero_gold = sum(1 for r in probs_rows if r["p_gold"] == 0.0)
    block = {
        "stage": stage,
        "run_dir": str(run_dir.relative_to(ROOT)),
        "n_requested": n_requested,
        "n_terminal": len(rows),
        "n_correct": n_correct,
        "headline_accuracy_all_requested": round(n_correct / n_requested, 6),
        "n_usable": len(usable),
        "usable_response_accuracy": (round(n_correct / len(usable), 6)
                                     if usable else None),
        "selected_answer_displayed_probability": {
            "mean_all": _mean([r["p_selected"] for r in probs_rows]),
            "mean_when_correct": _mean([r["p_selected"] for r in correct_rows]),
            "mean_when_wrong": _mean([r["p_selected"] for r in wrong_rows]),
            "n": len(probs_rows),
            "note": ("displayed probability of the SELECTED answer, split by "
                     "observed correctness; its overall mean is the model-implied "
                     "expected accuracy of taking that choice (a predicted "
                     "confidence), which the reliability bins test for calibration"),
        },
        "mean_top_displayed_probability": {
            "mean_pmax": _mean([r["p_max"] for r in probs_rows]),
            "note": MEAN_PMAX_NOTE,
        },
        "reliability_bins_equal_width": {
            "binning": ("10 equal-width bins on the selected-answer displayed "
                        "probability; bin i covers [i/10,(i+1)/10), the last "
                        "bin includes the right edge"),
            "bins": reliability(pairs, n_bins=10),
            "ece_10bins": ece_equal_width(pairs, n_bins=10),
            "rounding_caveat": ROUNDING_CAVEAT,
        },
        "proper_scores": {
            "n": len(probs_rows),
            "brier_mean": None,   # multiclass, filled below
            "log_loss_mean_clipped_1e-15": _mean([
                -__import__("math").log(max(1e-15, min(1 - 1e-15, r["p_gold"])))
                for r in probs_rows if r["p_gold"] is not None]),
            "zero_probability_gold": zero_gold,
            "zero_probability_gold_candidates": len(probs_rows),
            "note": ("rotated items scored on probability keys canonicalized "
                     "through the recorded remapping; display rounding at 2dp"),
        },
    }
    # multiclass Brier over the displayed vector, mean per request
    gold_map = _gold_map(run_dir)
    rotations = _rotations(run_dir)
    briers = []
    for result in _read_jsonl(run_dir / "results.jsonl"):
        if result.get("terminal") is not True:
            continue
        item_id = str(result["item_id"])
        answer = next(iter((result.get("predictions") or {}).values()), None) or {}
        probs = answer.get("probabilities")
        if item_id in rotations and isinstance(probs, dict):
            probs = _canonical_probs(probs, rotations[item_id]["remapping"])
        gold_value = gold_map.get(item_id)
        gold_value = (_canonical_label(item_id, gold_value, rotations)
                      if gold_value is not None else None)
        if not isinstance(probs, dict) or not probs or gold_value is None:
            continue
        if not all(isinstance(v, (int, float)) for v in probs.values()):
            continue
        if abs(sum(float(v) for v in probs.values()) - 1.0) > 5e-2:
            continue
        briers.append(sum((float(v) - (1.0 if k == gold_value else 0.0)) ** 2
                          for k, v in probs.items()))
    block["proper_scores"]["brier_mean"] = _mean(briers)
    return block


def _mean(values) -> float | None:
    values = [v for v in values if isinstance(v, (int, float))]
    return round(sum(values) / len(values), 6) if values else None


def _mcnemar_exact(b: int, c: int) -> float:
    import math
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


# ------------------------------------------------------------------ paired
def _math500_intersection() -> dict:
    mcq_dir = _run_dir("runs_benchmark_ext/bench-math500_choice-*")
    dig_dir = _run_dir(MATH500_SCORE_RUN)
    mcq = {r["item_id"]: r for r in _choice_rows(mcq_dir)}
    from jev_observatory.benchmark_score import score_digit_readout_run
    dig = {r["item_id"]: r for r in
           score_digit_readout_run(dig_dir, stage="math500_score")["per_item"]}
    shared = sorted(set(mcq) & set(dig))
    both = sum(1 for i in shared if mcq[i]["correct"] and dig[i]["correct"])
    mcq_only = sum(1 for i in shared if mcq[i]["correct"] and not dig[i]["correct"])
    dig_only = sum(1 for i in shared if not mcq[i]["correct"] and dig[i]["correct"])
    neither = len(shared) - both - mcq_only - dig_only
    return {
        "method": ("pair by MATH-500 problem id across the two recorded stage "
                   "runs; intersection = original shared items present in both"),
        "n_intersection_items": len(shared),
        "mcq_accuracy_on_intersection": round((both + mcq_only) / len(shared), 6),
        "digits_accuracy_on_intersection": round((both + dig_only) / len(shared), 6),
        "paired_table": {"both_correct": both, "mcq_only": mcq_only,
                         "digits_only": dig_only, "neither": neither},
        "mcnemar_exact_p": round(_mcnemar_exact(mcq_only, dig_only), 6),
        "provenance": ["runs_benchmark_ext/bench-math500_choice-*/results.jsonl",
                       "runs_benchmark_ext/bench-math500_score-*/results.jsonl"],
    }


def _mmlu_matched_subset() -> dict:
    freeze = json.loads((ROOT / "runs_matched/freeze/matched_frozen.json").read_text())
    matched_ids = {str(it["id"]) for it in freeze["datasets"]["mmlu"]["items"]}
    jev = {r["item_id"]: r["correct"] for r in _choice_rows(_run_dir("runs_benchmark/bench-mmlu_full-*"))}
    subset = sorted(matched_ids & set(jev))
    jev_acc = sum(1 for i in subset if jev[i]) / len(subset)
    out = {
        "method": ("paired subset = the 1,000 frozen matched MMLU item ids; "
                   "Jev correctness from the recorded mmlu_full run; baseline "
                   "rows joined by item id from the recorded v2/v3 result files"),
        "n_paired_items": len(subset),
        "jev_accuracy_on_paired_subset": round(jev_acc, 4),
        "per_baseline": {},
    }
    for version in ("v2", "v3"):
        path = ROOT / f"runs_matched_cheap/{version}/results.jsonl"
        if not path.exists():
            continue
        by_model: dict[str, list[dict]] = defaultdict(list)
        for record in _read_jsonl(path):
            if record.get("dataset") == "mmlu":
                by_model[record["model"]].append(record)
        cells = {}
        for model, records in sorted(by_model.items()):
            pairs = [(jev.get(str(r["item_id"])), bool(r.get("correct_recovered")))
                     for r in records if str(r["item_id"]) in jev]
            if not pairs:
                continue
            b = sum(1 for j, x in pairs if j and not x)
            c = sum(1 for j, x in pairs if not j and x)
            cells[model] = {
                "n_paired": len(pairs),
                "jev_only": b, "model_only": c,
                "mcnemar_exact_p": round(_mcnemar_exact(b, c), 6),
                "protocol": ("provider-defaults" if version == "v3"
                             else "temperature 0, effort low where accepted"),
            }
        out["per_baseline"][version] = cells
    return out


def _rotation_clusters() -> dict:
    native_rows = _choice_rows(_run_dir("runs_benchmark/bench-mmlu_full-*"))
    native = {r["item_id"]: r["correct"] for r in native_rows if r["usable"]}
    rot_rows = _choice_rows(_run_dir("runs_benchmark/bench-option_rotations-*"))
    paired = [(r["base_item"], native[r["base_item"]], r["correct"])
              for r in rot_rows if r["usable"] and r["base_item"] in native]
    n = len(paired)
    both = sum(1 for _b, a, r in paired if a and r)
    native_only = sum(1 for _b, a, r in paired if a and not r)
    rotated_only = sum(1 for _b, a, r in paired if r and not a)
    naive_p = _mcnemar_exact(native_only, rotated_only)
    # cluster-aware: base item is the resampling unit (variants are repeated
    # measures on the SAME item, not independent items)
    by_base: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    for base, a, r in paired:
        by_base[base].append((bool(a), bool(r)))
    base_diffs = []
    for base, observations in sorted(by_base.items()):
        a = observations[0][0]
        rot_mean = sum(1 for _a, r in observations if r) / len(observations)
        base_diffs.append((base, (1.0 if a else 0.0) - rot_mean, len(observations)))
    diffs = [d for _b, d, _k in base_diffs]
    rng = random.Random(BOOTSTRAP_SEED)
    boot = []
    for _ in range(BOOTSTRAP_N):
        sample = [diffs[rng.randrange(len(diffs))] for _ in diffs]
        boot.append(sum(sample) / len(sample))
    boot.sort()
    lo = boot[int(0.025 * len(boot))]
    hi = boot[min(len(boot) - 1, int(0.975 * len(boot)))]
    positive = sum(1 for d in diffs if d > 0)
    negative = sum(1 for d in diffs if d < 0)
    return {
        "observations": {
            "n_requested_rotation_observations": len(rot_rows),
            "n_paired_observations": n,
            "n_unique_base_items": len(by_base),
            "observation_note": ("the scored rotation observations (419 of 420 "
                                 "requested) are REPEATED variants clustered on "
                                 "140 unique base items; 419 observations are not "
                                 "419 independent items"),
        },
        "paired_table_observation_level": {
            "both_correct": both, "native_only": native_only,
            "rotated_only": rotated_only,
        },
        "naive_independent_mcnemar": {
            "p": round(naive_p, 4),
            "validity": ("INVALID independence assumption: discordant pairs are "
                         "clustered on shared base items, so this p-value is "
                         "anti-conservative and must be qualified, not quoted "
                         "as standalone evidence; a non-significant p is NOT an "
                         "equivalence proof (p=0.83 cannot show the conditions "
                         "are equivalent)"),
        },
        "cluster_aware": {
            "unit": "unique base item (140 clusters; per-base difference = native correct - mean rotated correct)",
            "mean_difference_native_minus_rotated": round(sum(diffs) / len(diffs), 6),
            "cluster_bootstrap_95_ci": [round(lo, 6), round(hi, 6)],
            "bootstrap": {"resamples": BOOTSTRAP_N, "seed": BOOTSTRAP_SEED,
                          "resampling_unit": "base item"},
            "per_base_sign_counts": {"native_higher": positive,
                                     "rotated_higher": negative,
                                     "tied": len(diffs) - positive - negative},
            "sign_test_two_sided_p": round(_mcnemar_exact(positive, negative), 6),
        },
        "reporting_rule": ("quote the cluster-aware interval and the observation "
                           "clustering; if the naive McNemar p is mentioned at "
                           "all it must carry the invalid-independence "
                           "qualification and the non-equivalence note"),
        "provenance": ["runs_benchmark/bench-mmlu_full-*/results.jsonl",
                       "runs_benchmark/bench-option_rotations-*/results.jsonl"],
    }


# ------------------------------------------------------------- arc payloads
def _arc_payload_audit() -> dict:
    out = {"detector": ("duplicate outbound payloads with different gold are "
                        "ambiguous requests; PRIMARY level = whole request "
                        "(state + questions), SECONDARY = per question "
                        "(state, type, instructions, criteria)"),
           "stages": {}}
    for stage in ("bench-arc_agi2_choice", "bench-arc_agi2_score",
                  "bench-arc_agi2_task"):
        run_dir = _run_dir(f"runs_benchmark_ext/{stage}-*")
        items = _read_jsonl(run_dir / "items.jsonl")
        request_payloads: dict[str, list] = defaultdict(list)
        for item in items:
            key = json.dumps({"state": item["state"], "questions": item["questions"]},
                             sort_keys=True, ensure_ascii=False)
            request_payloads[key].append((item["id"],
                                          json.dumps(item["gold"], sort_keys=True)))
        req_pairs = 0
        req_diff_gold = 0
        for group in request_payloads.values():
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    req_pairs += 1
                    if group[i][1] != group[j][1]:
                        req_diff_gold += 1
        payloads: dict[str, list] = defaultdict(list)
        for item in items:
            for qid, question in item["questions"].items():
                key = json.dumps([item["state"], question["type"],
                                  question["instructions"], question["criteria"]],
                                 sort_keys=True, ensure_ascii=False)
                payloads[key].append((item["id"], qid,
                                      json.dumps(item["gold"][qid], sort_keys=True)))
        pairs = 0
        diff_gold = 0
        for group in payloads.values():
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    pairs += 1
                    if group[i][2] != group[j][2]:
                        diff_gold += 1
        out["stages"][stage] = {
            "n_items": len(items),
            "duplicate_request_payload_pairs": req_pairs,
            "duplicate_request_payload_pairs_with_different_gold": req_diff_gold,
            "duplicate_question_payload_pairs": pairs,
            "duplicate_question_payload_pairs_with_different_gold": diff_gold,
        }
    out["cause"] = ("cell-mode question instructions carried only (row, column) "
                    "while the shared state lists ALL test inputs of a task, so "
                    "same-shaped grids with the same palette serialized to "
                    "identical requests with different gold")
    out["fix"] = ("arc_agi2_grid.build_items_cell_mode now states the test-grid "
                  "identity (grid i of n) in every cell question for choice and "
                  "score modes; regression: tests/test_arc_agi2_grid.py "
                  "(distinct outbound payloads per test grid)")
    out["existing_results"] = ("the recorded choice run contains 7 ambiguous "
                              "duplicate REQUEST-payload pairs with different "
                              "gold (936 pairs at question level); those requests "
                              "are uninterpretable individually and are flagged, "
                              "never silently re-scored")
    out["oracle_protocol"] = ("cell_choice palette and cell counts are derived "
                              "from the GOLD output grids — an explicitly "
                              "recorded diagnostic protocol, not official "
                              "ARC-AGI-2 evaluation")
    return out


def main() -> int:
    calibration = {
        "generated_by": "scripts/benchmark/benchmark_diagnostics.py (offline, deterministic)",
        "scoring_policy": ("headline accuracy is ALL-REQUESTED (missing/invalid "
                           "wrong); usable-response accuracy reported separately"),
        "stages": {},
    }
    for stage, pattern in STAGE_RUNS.items():
        run_dir = _run_dir(pattern)
        calibration["stages"][stage] = _calibration_block(stage, run_dir,
                                                          _choice_rows(run_dir))
    paired = {
        "generated_by": "scripts/benchmark/benchmark_diagnostics.py (offline, deterministic)",
        "math500_intersection": _math500_intersection(),
        "mmlu_matched_subset": _mmlu_matched_subset(),
        "rotation_clusters": _rotation_clusters(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "calibration.json").write_text(
        json.dumps(calibration, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "paired_diagnostics.json").write_text(
        json.dumps(paired, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "arc_payload_audit.json").write_text(
        json.dumps(_arc_payload_audit(), indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8")
    for name in ("calibration.json", "paired_diagnostics.json", "arc_payload_audit.json"):
        print(f"[ok] wrote data_report/benchmark_diagnostics/{name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())