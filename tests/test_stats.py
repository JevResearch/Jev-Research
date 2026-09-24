"""Stats tests: hand-checked estimators, cluster effects, stratified weighting."""

import pytest

from jev_observatory.stats import (
    chance_adjusted,
    cluster_bootstrap_accuracy,
    effective_sample_size_for_clusters,
    paired_difference_cluster_bootstrap,
    stratified_accuracy,
)


def test_chance_adjusted():
    assert chance_adjusted(0.8, 0.25) == pytest.approx(0.7333333)
    assert chance_adjusted(0.1, 0.25) < 0  # below chance is negative, not clamped
    with pytest.raises(ValueError):
        chance_adjusted(0.5, 1.0)


def test_cluster_bootstrap_contains_point_estimate_and_is_deterministic():
    correct = [True] * 60 + [False] * 40
    clusters = [f"c{i // 2}" for i in range(100)]  # pairs share a cluster
    a = cluster_bootstrap_accuracy(correct, clusters, n_resamples=500, seed=1)
    b = cluster_bootstrap_accuracy(correct, clusters, n_resamples=500, seed=1)
    assert a.estimate == pytest.approx(0.6)
    assert a.ci_low <= a.estimate <= a.ci_high
    assert (a.ci_low, a.ci_high) == (b.ci_low, b.ci_high)


def test_clustered_errors_widen_interval_vs_iid():
    """Correlated errors must widen the CI — that is the point of clustering."""
    iid = cluster_bootstrap_accuracy([True] * 50 + [False] * 50,
                                     [f"i{i}" for i in range(100)], n_resamples=800, seed=5)
    # same 50% accuracy, but errors perfectly correlated within clusters of 4
    clustered = cluster_bootstrap_accuracy(
        [True] * 50 + [False] * 50,
        [f"c{i // 4}" for i in range(100)], n_resamples=800, seed=5)
    assert (clustered.ci_high - clustered.ci_low) >= (iid.ci_high - iid.ci_low) - 0.02


def test_cluster_bootstrap_refuses_single_cluster():
    with pytest.raises(ValueError, match="two clusters"):
        cluster_bootstrap_accuracy([True, False], ["only", "only"])


def test_paired_difference_detects_perfect_shift():
    # A is right exactly where B is wrong: difference 1.0 on every cluster.
    correct_a = [True, True, True, True]
    correct_b = [False, False, False, False]
    clusters = ["c1", "c2", "c3", "c4"]
    result = paired_difference_cluster_bootstrap(correct_a, correct_b, clusters, n_resamples=400, seed=0)
    assert result.estimate == pytest.approx(1.0)
    assert result.ci_low == pytest.approx(1.0) and result.ci_high == pytest.approx(1.0)


def test_paired_difference_zero_when_identical():
    correct = [True, False, True, False]
    result = paired_difference_cluster_bootstrap(correct, correct, ["c1", "c2", "c3", "c4"], seed=0)
    assert result.estimate == 0.0
    assert result.ci_low <= 0.0 <= result.ci_high


def test_stratified_estimate_differs_from_naive_on_imbalanced_fixture():
    """The M1 acceptance case: population weighting ≠ naive averaging."""
    # Population: 900 easy-stratum items at 90% acc, 100 hard-stratum at 20%.
    # Pilot oversamples hard: 10 from each stratum.
    correct = ([True] * 9 + [False]) + ([True] * 2 + [False] * 8)
    strata = ["easy"] * 10 + ["hard"] * 10
    result = stratified_accuracy(correct, strata, population_sizes={"easy": 900, "hard": 100})
    naive_mean = sum(correct) / len(correct)  # 0.55
    assert result["estimate"] == pytest.approx(0.9 * 0.9 + 0.1 * 0.2)  # 0.83
    assert abs(result["estimate"] - naive_mean) > 0.2
    assert result["weights"]["easy"] == pytest.approx(0.9)
    assert result["weights"]["hard"] == pytest.approx(0.1)


def test_stratified_refuses_unknown_stratum():
    with pytest.raises(KeyError):
        stratified_accuracy([True], ["ghost"], population_sizes={"known": 1})


def test_effective_sample_size_kish():
    # 4 items in 2 clusters of sizes 3 and 1: ESS = 16 / (9+1) = 1.6
    assert effective_sample_size_for_clusters(["a", "a", "a", "b"]) == pytest.approx(1.6)
    # perfectly balanced clusters keep full ESS
    assert effective_sample_size_for_clusters(["a", "b", "c", "d"]) == pytest.approx(4.0)
    assert effective_sample_size_for_clusters([]) == 0.0