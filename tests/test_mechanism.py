"""Mechanism analysis tests: isolation, odds, latency summaries, claims labels."""

import math

from jev_observatory.mechanism import (
    isolation_summary,
    latency_by_condition,
    odds_summary,
    rename_sensitivity,
    total_variation_distance,
)


def _anchor_record(cluster, sibling_set, probs, repeat=0, status="ok"):
    return {
        "logical_request_id": f"anchor-{cluster}-{sibling_set}",
        "cluster": cluster,
        "condition": f"role=anchor;sibling_set={sibling_set};repeat={repeat}",
        "status": status,
        "predictions": {"anchor": {"type": "choice", "usable": True,
                                   "choice": max(probs, key=probs.get), "probabilities": probs,
                                   "p_max": max(probs.values())}},
    }


def test_isolation_equivalence_when_anchor_stable():
    results = []
    for repeat in range(3):
        probs = {"yes": 0.7 + 1e-9 * repeat, "no": 0.3}
        for sibling_set in (0, 1, 8):
            results.append(_anchor_record(f"r{repeat}", sibling_set, dict(probs)))
    summary = isolation_summary(results, margin=0.05)
    assert summary["claim_type"] == "exploratory"
    assert summary["all_within_margin"] is True
    assert summary["verdict"].startswith("equivalence")
    # the caveat must stay: equivalence is not proof of independence
    assert "does not prove" in summary["caveat"]


def test_isolation_detects_instability_beyond_margin():
    results = []
    for sibling_set, p_yes in ((0, 0.7), (1, 0.7), (8, 0.3)):  # 8 siblings move the anchor
        results.append(_anchor_record("r0", sibling_set, {"yes": p_yes, "no": 1 - p_yes}))
    summary = isolation_summary(results, margin=0.05)
    assert summary["all_within_margin"] is False
    assert summary["verdict"].startswith("instability")
    worst = summary["clusters"]["r0"]["worst_tvd"]
    assert math.isclose(worst, 0.4)


def test_isolation_skips_unusable_records():
    results = [_anchor_record("r0", 0, {"yes": 0.7, "no": 0.3}, status="error")]
    summary = isolation_summary(results)
    assert summary["n_clusters_compared"] == 0 and summary["verdict"] is None


def test_tvd_hand_computed():
    assert total_variation_distance({"a": 0.5, "b": 0.5}, {"a": 0.5, "b": 0.5}) == 0.0
    assert math.isclose(total_variation_distance({"a": 1.0}, {"a": 0.0, "b": 1.0}), 1.0)
    assert math.isclose(total_variation_distance({"a": 0.6, "b": 0.4}, {"a": 0.4, "b": 0.6}), 0.2)


def _odds_record(variant, probs, repeat=0):
    return {
        "logical_request_id": f"odds-{variant}-{repeat}",
        "cluster": f"odds-r{repeat}",
        "condition": f"role=odds;odds_variant={variant};tracked_pair=A,B;repeat={repeat}",
        "status": "ok",
        "predictions": {"verdict": {"type": "choice", "usable": True,
                                    "choice": max(probs, key=probs.get),
                                    "probabilities": probs, "p_max": max(probs.values())}},
    }


def test_odds_invariance_under_irrelevant_distractors():
    base = {"A": 0.6, "B": 0.3, "C": 0.06, "D": 0.04}
    # candidate-local scoring: appending unrelated options rescales all base probs
    # by (1 - irrelevant_mass); the A:B odds ratio is preserved exactly.
    scaled = {"A": 0.48, "B": 0.24, "C": 0.048, "D": 0.032,
              "E": 0.08, "F": 0.08, "G": 0.02, "H": 0.02}
    results = [_odds_record("base", base), _odds_record("irrelevant", scaled)]
    summary = odds_summary(results)
    assert summary["available"] is not False
    assert summary["base_odds_median"] == 2.0
    variant = summary["variants"]["irrelevant"]
    assert variant["ratio_median"] == pytest.approx(1.0)
    assert "does not identify" in summary["note"]


def test_odds_duplicate_splits_mass_and_is_documented():
    base = {"A": 0.6, "B": 0.3, "C": 0.1}
    duplicate = {"A": 0.3, "A2": 0.3, "B": 0.3, "C": 0.1}
    summary = odds_summary([_odds_record("base", base), _odds_record("duplicate", duplicate)])
    variant = summary["variants"]["duplicate"]
    assert variant["ratio_median"] == pytest.approx(0.5)
    assert "split" in variant["expected"]


import pytest  # noqa: E402


def test_odds_requires_base():
    summary = odds_summary([_odds_record("irrelevant", {"A": 0.5, "B": 0.5})])
    assert summary["available"] is False


def test_latency_by_condition_summary_and_insufficient_n():
    records = []
    for i in range(12):
        records.append({
            "logical_request_id": f"r{i}", "cluster": f"c{i}",
            "condition": "L=512;Q=4", "status": "ok",
            "latency_ms_first_attempt": 100.0 + i,
        })
    records.append({"logical_request_id": "bad", "cluster": "cx", "condition": "L=512;Q=4",
                    "status": "http_error", "latency_ms_first_attempt": 900.0})
    records.append({"logical_request_id": "few1", "cluster": "cy", "condition": "L=64;Q=1",
                    "status": "ok", "latency_ms_first_attempt": 50.0})
    records.append({"logical_request_id": "few2", "cluster": "cy", "condition": "L=64;Q=1",
                    "status": "ok", "latency_ms_first_attempt": 60.0})
    summary = latency_by_condition(records)
    big = summary["conditions"]["L=512;Q=4"]
    assert big["n_total"] == 13 and big["n_failures"] == 1
    assert math.isclose(big["failure_rate"], round(1 / 13, 4))  # summary rounds to 4dp
    assert big["median_ms"] == 105.5
    assert big["p90_ms"] != "insufficient_n"
    small = summary["conditions"]["L=64;Q=1"]
    assert small["p90_ms"] == "insufficient_n (<10)"


def test_rename_sensitivity_flags_only_beyond_jitter():
    base = {"q": {"A": 0.9, "B": 0.1}}
    identical = {"q": {"A": 0.9, "B": 0.1}}
    moved = {"q": {"A": 0.6, "B": 0.4}}
    quiet = rename_sensitivity(base, identical)
    assert quiet["ids_visible_beyond_jitter"] is False
    loud = rename_sensitivity(base, moved)
    assert loud["ids_visible_beyond_jitter"] is True
    assert "falsification" in loud["note"]


def test_all_mechanism_outputs_carry_claim_labels():
    assert "claim_type" in isolation_summary([])
    assert "claim_type" in odds_summary([])
    assert "claim_type" in latency_by_condition([])