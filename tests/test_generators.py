"""Generator tests: determinism, independent label verification, spec hygiene."""

import json

import pytest

from jev_observatory.datasets.generators import (
    build_synthetic_spec,
    gen_base_rate,
    gen_policy_routing,
    gen_relation_lookup,
    gen_unanswerable,
    solve_relation_lookup,
)
from jev_observatory.experiment import SpecError, logical_requests


def test_generators_are_deterministic_per_seed():
    a = gen_relation_lookup(7, 20)
    b = gen_relation_lookup(7, 20)
    c = gen_relation_lookup(8, 20)
    assert [i["id"] for i in a] == [i["id"] for i in b]
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert json.dumps(a, sort_keys=True) != json.dumps(c, sort_keys=True)


def test_relation_lookup_labels_verified_independently():
    """Labels re-derived by parsing the rendered text, not the RNG path."""
    for seed in (0, 1, 42):
        items = gen_relation_lookup(seed, 12)
        for item in items:
            assert solve_relation_lookup(item) == item["gold"]["city" if "city" in item["gold"] else "entity"]["value"]


def test_policy_labels_match_hand_computed_boundaries():
    items = gen_policy_routing(0, 8)
    by_id = {i["_params"]["order_id"] if False else i["id"]: i for i in items}
    expected = [
        ("policy-00000", "approve"),  # 10d / 50 / standard
        ("policy-00001", "deny"),     # 40d / 50 / standard (age)
        ("policy-00002", "deny"),     # 10d / 300 / standard (amount)
        ("policy-00003", "approve"),  # 40d / 300 / gold
        ("policy-00004", "deny"),     # 90d / 100 / gold (age)
        ("policy-00005", "escalate"), # damaged
        ("policy-00006", "approve"),  # boundary 30d/200
        ("policy-00007", "deny"),     # boundary 31d/200
    ]
    for item_id, decision in expected:
        assert by_id[item_id]["gold"]["decision"]["value"] == decision, item_id


def test_unanswerable_gold_is_not_stated():
    for seed in (0, 3, 9):
        for item in gen_unanswerable(seed, 5):
            assert item["gold"]["material"]["value"] == "not_stated"
            # the document genuinely omits any material fact
            for material in ("bronze", "granite", "cedar"):
                assert material not in item["state"].lower()


def test_base_rate_gold_matches_counts():
    for item in gen_base_rate(4, 12):
        params = item["_params"]
        expected = "yes" if params["red"] > params["blue"] else "no"
        assert item["gold"]["likelier_red"]["value"] == expected
        assert str(params["red"]) in item["state"] and str(params["blue"]) in item["state"]


def test_synthetic_spec_strips_params_and_passes_guards():
    spec = build_synthetic_spec(
        [("relation", gen_relation_lookup), ("policy", gen_policy_routing),
         ("unanswerable", gen_unanswerable), ("base_rate", gen_base_rate)],
        seed=11, n_per_generator=8,
    )
    assert len(spec["items"]) == 32
    assert all("_params" not in item for item in spec["items"])
    assert spec["seeds"]["generator_base"] == 11
    requests = logical_requests(spec)  # raises on leakage or schema problems
    assert len(requests) == 32


def test_synthetic_leakage_guard_catches_leaked_state():
    """The guard stays ON for unanswerable/base-rate items and must fire."""
    spec = build_synthetic_spec([("unanswerable", gen_unanswerable)], seed=2, n_per_generator=2)
    logical_requests(spec)  # clean items pass
    leaked = json.loads(json.dumps(spec))
    leaked_item = leaked["items"][0]
    leaked_item["state"] = f"{leaked_item['state']} The material is not_stated."
    with pytest.raises(SpecError, match="leak"):
        logical_requests(leaked)