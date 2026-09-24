"""Mechanism analyses: isolation stability, candidate odds, latency summaries.

Every output here carries an explicit claim-type label (DESIGN.md §7): a null
isolation result is *not* proof of independence, an odds-ratio invariance is a
*behavioral* property shared by several architectures, and latency statistics
never translate into architecture claims.
"""

from __future__ import annotations

import math
from statistics import median
from typing import Any

from .experiments import condition_param

DEFAULT_EQUIVALENCE_MARGIN_TVD = 0.05


def _probability_vector(prediction: dict[str, Any], option_order: list[str] | None = None) -> dict[str, float] | None:
    if prediction.get("type") != "choice":
        return None
    probs = prediction.get("probabilities")
    if not isinstance(probs, dict):
        return None
    return {str(k): float(v) for k, v in probs.items()}


def total_variation_distance(p: dict[str, float], q: dict[str, float]) -> float:
    """TVD over the union of support; 0 = identical, 1 = disjoint."""
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def max_abs_probability_deviation(p: dict[str, float], q: dict[str, float]) -> float:
    keys = set(p) | set(q)
    return max((abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys), default=0.0)


def isolation_summary(
    results: list[dict[str, Any]],
    *,
    margin: float = DEFAULT_EQUIVALENCE_MARGIN_TVD,
    claim_type: str = "exploratory",
) -> dict[str, Any]:
    """Anchor-distribution stability across sibling-set conditions.

    Groups results by cluster (each repeat), extracts the `anchor` question's
    distribution per sibling_set, and reports pairwise TVD against the smallest
    sibling set.  Equivalence is claimed ONLY if every deviation is below the
    preregistered margin; otherwise instability is flagged.  A null result is
    recorded as such, never upgraded to proof.
    """
    clusters: dict[str, dict[str, dict[str, float]]] = {}
    for record in results:
        if record.get("status") != "ok" or condition_param(record.get("condition", ""), "role") != "anchor":
            continue
        prediction = record.get("predictions", {}).get("anchor")
        probs = _probability_vector(prediction)
        if probs is None:
            continue
        level = condition_param(record.get("condition", ""), "sibling_set")
        clusters.setdefault(record["cluster"], {})[f"siblings={level}"] = probs

    per_cluster: dict[str, Any] = {}
    all_ok = True
    for cluster, by_level in sorted(clusters.items()):
        levels = sorted(by_level, key=lambda name: int(name.split("=")[1]))
        if len(levels) < 2:
            per_cluster[cluster] = {"skipped": "fewer than two sibling levels observed"}
            all_ok = False
            continue
        reference = by_level[levels[0]]
        comparisons = {}
        worst = 0.0
        for level in levels[1:]:
            tvd = total_variation_distance(reference, by_level[level])
            mad = max_abs_probability_deviation(reference, by_level[level])
            comparisons[level] = {"tvd": round(tvd, 6), "max_abs_dev": round(mad, 6)}
            worst = max(worst, tvd)
        within_margin = worst <= margin
        all_ok = all_ok and within_margin
        per_cluster[cluster] = {
            "reference": levels[0],
            "comparisons": comparisons,
            "worst_tvd": round(worst, 6),
            "within_margin": within_margin,
            "margin": margin,
        }
    n_compared = sum(1 for v in per_cluster.values() if "worst_tvd" in v)
    return {
        "claim_type": claim_type,
        "margin_tvd": margin,
        "n_clusters_compared": n_compared,
        "all_within_margin": all_ok if n_compared > 0 else None,
        "verdict": (
            None if n_compared == 0
            else (f"equivalence: all anchor distributions within TVD {margin} of the reference sibling set"
                  if all_ok else
                  f"instability detected: at least one anchor moved beyond TVD {margin} when siblings changed")
        ),
        "caveat": ("equivalence fails to reject change at this margin; it does not prove "
                   "questions are evaluated independently"),
        "clusters": per_cluster,
    }


def odds_summary(results: list[dict[str, Any]], *, claim_type: str = "exploratory") -> dict[str, Any]:
    """Odds of the tracked base pair across distractor variants.

    Candidate-local scoring predicts the A:B odds ratio is preserved when only
    unrelated options are appended.  A duplicate of A may legitimately split A's
    mass — recorded as the documented expectation, not scored as failure.
    """
    per_variant: dict[str, list[dict[str, float]]] = {}
    tracked_pair: tuple[str, str] | None = None
    for record in results:
        if record.get("status") != "ok" or condition_param(record.get("condition", ""), "role") != "odds":
            continue
        pair = condition_param(record.get("condition", ""), "tracked_pair")
        if pair and tracked_pair is None:
            tracked_pair = (pair.split(",")[0], pair.split(",")[1])
        variant = condition_param(record.get("condition", ""), "odds_variant")
        prediction = record.get("predictions", {}).get("verdict")
        probs = _probability_vector(prediction)
        if probs is not None and variant:
            per_variant.setdefault(variant, []).append(probs)

    if not per_variant or tracked_pair is None:
        return {"available": False, "claim_type": claim_type,
                "reason": "no usable odds records"}

    def _odds(probs: dict[str, float], a: str, b: str) -> float | None:
        pa, pb = probs.get(a), probs.get(b)
        if pa is None or pb is None or pb <= 0.0 or pa <= 0.0:
            return None
        return pa / pb

    a, b = tracked_pair
    base_odds = [_odds(p, a, b) for p in per_variant.get("base", [])]
    base_odds = [o for o in base_odds if o is not None]
    if not base_odds:
        return {"available": False, "claim_type": claim_type,
                "reason": f"base variant lacks usable odds for {tracked_pair}"}
    reference = median(base_odds)

    out: dict[str, Any] = {
        "available": True,
        "claim_type": claim_type,
        "tracked_pair": [a, b],
        "base_odds_median": round(reference, 6),
        "n_base": len(base_odds),
        "variants": {},
        "note": ("odds-ratio invariance under unrelated distractors is a behavioral property "
                 "consistent with several architectures; it does not identify one"),
    }
    for variant, probs_list in sorted(per_variant.items()):
        if variant == "base":
            continue
        ratios = []
        for probs in probs_list:
            odds = _odds(probs, a, b)
            if odds is not None:
                ratios.append(odds / reference)
        out["variants"][variant] = {
            "n": len(ratios),
            "odds_ratio_vs_base_median": [round(r, 6) for r in ratios],
            "ratio_median": round(median(ratios), 6) if ratios else None,
            "expected": ("≈1 if candidate-local scoring" if variant == "irrelevant"
                         else "A's mass may split with its duplicate; no invariance expected"),
        }
    return out


def latency_by_condition(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Median/p90 first-attempt latency per condition; failures reported separately.

    p90 requires >= 10 samples (S7: p99 omitted from small cells; same logic).
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in results:
        condition = record.get("condition")
        if not condition:
            continue
        grouped.setdefault(condition, []).append(record)
    out: dict[str, Any] = {}
    for condition, records in sorted(grouped.items()):
        ok_latencies = [r["latency_ms_first_attempt"] for r in records
                        if r.get("status") == "ok" and r.get("latency_ms_first_attempt") is not None]
        failures = [r for r in records if r.get("status") != "ok"]
        entry: dict[str, Any] = {
            "n_total": len(records),
            "n_ok": len(ok_latencies),
            "n_failures": len(failures),
            "failure_rate": round(len(failures) / len(records), 4) if records else None,
        }
        if ok_latencies:
            entry["median_ms"] = round(median(ok_latencies), 3)
            entry["p90_ms"] = (round(_percentile_lazy(ok_latencies, 90), 3)
                               if len(ok_latencies) >= 10 else "insufficient_n (<10)")
            entry["min_ms"] = round(min(ok_latencies), 3)
            entry["max_ms"] = round(max(ok_latencies), 3)
        out[condition] = entry
    return {"claim_type": "confirmatory when preregistered", "conditions": out}


def _percentile_lazy(values: list[float], p: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(p / 100.0 * (len(ordered) - 1)))))
    return ordered[index]


def rename_sensitivity(
    base_results: dict[str, dict[str, float]],
    renamed_results: dict[str, dict[str, float]],
    *,
    jitter: float = 1e-6,
) -> dict[str, Any]:
    """Did renaming question ids move the anchor distribution beyond jitter?"""
    if not base_results or not renamed_results:
        return {"available": False, "reason": "need both renamed and base results"}
    worst = 0.0
    for key, base_probs in base_results.items():
        renamed_probs = renamed_results.get(key)
        if renamed_probs is None:
            continue
        worst = max(worst, total_variation_distance(base_probs, renamed_probs))
    return {
        "claim_type": "exploratory",
        "worst_tvd": round(worst, 6),
        "jitter_tolerance": jitter,
        "ids_visible_beyond_jitter": bool(worst > jitter),
        "note": "falsification probe for the documented claim that question ids never reach the model (S3)",
    }