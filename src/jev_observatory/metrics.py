"""Offline metrics. Every function here is pure and unit-tested against
hand-computed fixtures (DESIGN.md §6)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

WILSON_Z = 1.959963984540054  # two-sided 95%
LOG_LOSS_EPSILON = 1e-12  # preregistered clip; used *only* for the clipped variant


@dataclass(frozen=True)
class Interval:
    low: float
    high: float

    def to_dict(self) -> dict[str, float]:
        return {"low": round(self.low, 6), "high": round(self.high, 6)}


def wilson_interval(successes: int, n: int, z: float = WILSON_Z) -> Interval:
    if n <= 0:
        raise ValueError("n must be positive")
    p = successes / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return Interval(max(0.0, centre - half), min(1.0, centre + half))


def accuracy(correct: Sequence[bool]) -> dict[str, Any]:
    n = len(correct)
    k = sum(1 for c in correct if c)
    result: dict[str, Any] = {"n": n, "n_correct": k}
    if n == 0:
        result.update({"accuracy": None, "wilson_95": None, "note": "empty sample"})
        return result
    result["accuracy"] = k / n
    result["wilson_95"] = wilson_interval(k, n).to_dict()
    return result


def brier_multiclass(probabilities: dict[str, float], gold: str) -> dict[str, Any]:
    """Sum over classes of (p_k - y_k)^2; range 0..K-1, never silently rescaled."""
    total = sum(
        (float(probabilities.get(k, 0.0)) - (1.0 if k == gold else 0.0)) ** 2 for k in probabilities
    )
    return {"brier_sum": total, "scale": "0..K-1 (sum over classes)"}


def brier_binary(p_yes: float, gold_yes: bool) -> float:
    """Binary Brier score on the conventional 0..1 scale."""
    return (float(p_yes) - (1.0 if gold_yes else 0.0)) ** 2


def log_loss_exact(p_true: float) -> float:
    """Natural-log log loss; a zero-probability true outcome is infinite."""
    p = float(p_true)
    if p <= 0.0:
        return math.inf
    return -math.log(p)


def log_loss_clipped(p_true: float, epsilon: float = LOG_LOSS_EPSILON) -> float:
    """Clipped variant for plotting; must always be labelled with its epsilon."""
    return -math.log(max(float(p_true), epsilon))


def reliability(pairs: Sequence[tuple[float, bool]], n_bins: int = 10) -> list[dict[str, Any]]:
    """Equal-width reliability bins with counts; empty bins are omitted."""
    edges = [i / n_bins for i in range(n_bins + 1)]
    out: list[dict[str, Any]] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        members = [(p, y) for p, y in pairs if (lo <= p < hi) or (i == n_bins - 1 and p == hi)]
        if not members:
            continue
        count = len(members)
        mean_p = sum(p for p, _ in members) / count
        empirical = sum(1 for _, y in members if y) / count
        out.append({
            "bin": i,
            "low": lo,
            "high": hi,
            "n": count,
            "mean_probability": round(mean_p, 6),
            "empirical_accuracy": round(empirical, 6),
        })
    return out


def ece_equal_width(pairs: Sequence[tuple[float, bool]], n_bins: int = 10) -> float:
    """Expected calibration error over equal-width bins (empty bins contribute 0)."""
    bins = reliability(pairs, n_bins)
    total = sum(b["n"] for b in bins)
    if total == 0:
        return float("nan")
    return sum(b["n"] * abs(b["mean_probability"] - b["empirical_accuracy"]) for b in bins) / total


def risk_coverage(pairs: Sequence[tuple[float, bool]]) -> list[dict[str, Any]]:
    """Risk vs coverage when selective prediction ranks by p_max descending.

    `pairs` are (p_max, correct).  Ties share one coverage step (ranked by
    insertion order), so the curve is reproducible.
    """
    if not pairs:
        return []
    ordered = sorted(range(len(pairs)), key=lambda i: (-pairs[i][0], i))
    curve: list[dict[str, Any]] = []
    correct = 0
    for rank, index in enumerate(ordered, start=1):
        correct += 1 if pairs[index][1] else 0
        coverage = rank / len(pairs)
        curve.append({
            "coverage": round(coverage, 6),
            "accuracy": round(correct / rank, 6),
            "n": rank,
        })
    return curve


def ordinal_mae(predictions: Sequence[float], golds: Sequence[float]) -> float | None:
    if not predictions:
        return None
    return sum(abs(p - g) for p, g in zip(predictions, golds)) / len(predictions)


def rank_probability_score(probs: Sequence[float], gold_level: int) -> float:
    """Ranked probability score for ordinal score questions (lower is better).

    RPS = sum over levels 1..K-1 of (CDF_i - Y_i)^2, divided by (K - 1).
    """
    if not probs:
        raise ValueError("no probabilities")
    levels = len(probs)
    if levels < 2:
        raise ValueError("needs at least two levels")
    cdf = 0.0
    total = 0.0
    for i in range(levels - 1):
        cdf += probs[i]
        y = 1.0 if i >= gold_level else 0.0
        total += (cdf - y) ** 2
    return total / (levels - 1)