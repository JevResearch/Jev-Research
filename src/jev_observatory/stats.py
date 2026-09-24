"""Cluster-aware statistics and sampling-design estimators (DESIGN.md §6).

Everything here is deterministic given its seed and unit-tested against
hand-checked fixtures.  Repeated prompts of one base item are NOT independent
facts: cluster resampling exists precisely to keep such items from manufacturing
significance.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Sequence

WILSON_Z = 1.959963984540054


def chance_adjusted(accuracy: float, chance: float) -> float:
    """(a - c) / (1 - c) for accuracy only; can be negative. Not a general metric."""
    if not 0.0 <= chance < 1.0:
        raise ValueError("chance must be in [0, 1)")
    return (accuracy - chance) / (1.0 - chance)


@dataclass(frozen=True)
class BootstrapResult:
    estimate: float
    ci_low: float
    ci_high: float
    n_resamples: int
    method: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimate": round(self.estimate, 6),
            "ci95_low": round(self.ci_low, 6),
            "ci95_high": round(self.ci_high, 6),
            "n_resamples": self.n_resamples,
            "method": self.method,
        }


def cluster_bootstrap_accuracy(
    correct: Sequence[bool],
    clusters: Sequence[str],
    *,
    n_resamples: int = 2000,
    seed: int = 0,
    z: float = WILSON_Z,
) -> BootstrapResult:
    """Percentile bootstrap over clusters (whole clusters drawn with replacement).

    The point estimate stays the plain mean; the interval reflects cluster-level
    dependence.  Fewer than two clusters cannot support a resampling interval.
    """
    if len(correct) != len(clusters):
        raise ValueError("correct and clusters must align")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    data = list(zip(clusters, correct))
    point = sum(1 for _, c in data if c) / len(data)
    by_cluster: dict[str, list[bool]] = {}
    for cluster, c in data:
        by_cluster.setdefault(cluster, []).append(c)
    names = sorted(by_cluster)
    if len(names) < 2:
        raise ValueError("cluster bootstrap needs at least two clusters")

    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_resamples):
        acc = 0
        n_drawn = 0
        for _ in range(len(names)):
            members = by_cluster[rng.choice(names)]
            acc += sum(1 for c in members if c)
            n_drawn += len(members)  # resampled size varies with unequal clusters
        means.append(acc / n_drawn)
    means.sort()
    lo_index = int(0.025 * n_resamples)
    hi_index = min(n_resamples - 1, int(0.975 * n_resamples))
    return BootstrapResult(
        estimate=point,
        ci_low=means[lo_index],
        ci_high=means[hi_index],
        n_resamples=n_resamples,
        method="cluster_percentile_bootstrap",
    )


def paired_difference_cluster_bootstrap(
    correct_a: Sequence[bool],
    correct_b: Sequence[bool],
    clusters: Sequence[str],
    *,
    n_resamples: int = 2000,
    seed: int = 0,
) -> BootstrapResult:
    """CI for accuracy(A) - accuracy(B) resampling shared clusters.

    Requires the two series to be aligned item-for-item (same base items).
    """
    if not (len(correct_a) == len(correct_b) == len(clusters)):
        raise ValueError("series must align item-for-item")
    diffs_by_cluster: dict[str, list[float]] = {}
    for ca, cb, cluster in zip(correct_a, correct_b, clusters):
        diffs_by_cluster.setdefault(cluster, []).append((1.0 if ca else 0.0) - (1.0 if cb else 0.0))
    names = sorted(diffs_by_cluster)
    if len(names) < 2:
        raise ValueError("paired cluster bootstrap needs at least two clusters")

    n_items = len(correct_a)
    point = (sum(1 for c in correct_a if c) - sum(1 for c in correct_b if c)) / n_items
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_resamples):
        total = 0.0
        n_drawn = 0
        for _ in range(len(names)):
            members = diffs_by_cluster[rng.choice(names)]
            total += sum(members)
            n_drawn += len(members)  # resampled size varies with unequal clusters
        means.append(total / n_drawn)
    means.sort()
    lo_index = int(0.025 * n_resamples)
    hi_index = min(n_resamples - 1, int(0.975 * n_resamples))
    return BootstrapResult(
        estimate=point,
        ci_low=means[lo_index],
        ci_high=means[hi_index],
        n_resamples=n_resamples,
        method="paired_cluster_percentile_bootstrap",
    )


def stratified_accuracy(
    correct: Sequence[bool],
    strata: Sequence[str],
    population_sizes: dict[str, int],
) -> dict[str, Any]:
    """Weighted population estimate from a stratified pilot.

    Weight of stratum h is N_h / n_h (its inverse selection probability), so the
    estimate targets the full-set accuracy, not the sample's own mix.  This
    deliberately differs from naive macro/micro averages when the sample is
    imbalanced relative to the population.
    """
    if len(correct) != len(strata):
        raise ValueError("correct and strata must align")
    sums: dict[str, tuple[int, int]] = {}
    for c, stratum in zip(correct, strata):
        if stratum not in population_sizes:
            raise KeyError(f"stratum {stratum!r} missing from population_sizes")
        k, n = sums.get(stratum, (0, 0))
        sums[stratum] = (k + (1 if c else 0), n + 1)
    if not sums:
        return {"estimate": None, "weights": {}, "note": "empty sample"}
    estimate = 0.0
    weights: dict[str, float] = {}
    total_weight = sum(population_sizes[stratum] for stratum in sums)
    for stratum, (k, n) in sorted(sums.items()):
        if n == 0:
            raise ValueError(f"stratum {stratum!r} has no sample rows")
        weight = population_sizes[stratum] / total_weight
        weights[stratum] = weight
        estimate += weight * (k / n)
    return {
        "estimate": estimate,
        "weights": {s: round(w, 6) for s, w in weights.items()},
        "strata": {s: {"n_sample": n, "n_population": population_sizes[s], "accuracy": k / n}
                   for s, (k, n) in sorted(sums.items())},
        "note": "weights are population shares; see DESIGN.md §6 for pilot-vs-full distinction",
    }


def effective_sample_size_for_clusters(clusters: Sequence[str]) -> float:
    """Kish ESS under cluster weighting: m^2 / sum(m_i^2) per cluster sizes."""
    counts: dict[str, int] = {}
    for cluster in clusters:
        counts[cluster] = counts.get(cluster, 0) + 1
    if not counts:
        return 0.0
    m = len(clusters)
    return m * m / sum(c * c for c in counts.values())