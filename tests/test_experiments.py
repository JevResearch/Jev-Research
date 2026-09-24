"""M2 experiment-builder tests: sweeps, scheduler, isolation, odds designs."""

import json

import pytest

from jev_observatory.experiments import (
    DEFAULT_L_LEVELS,
    SweepCondition,
    build_sweep_items,
    condition_id,
    condition_param,
    isolation_spec,
    odds_spec,
    one_factor_sweep,
    randomize_blocks,
    rename_question_ids,
    reorder_questions,
    screening_cross,
    sweep_spec,
)
from jev_observatory.experiment import SpecError, logical_requests


def test_one_factor_sweep_includes_baseline_and_variations():
    conditions = one_factor_sweep()
    ids = {condition_id(c.params()) for c in conditions}
    assert condition_id({**dict(BASELINE_PARAMS)} ) in ids  # baseline present
    # each factor varies alone: every non-baseline condition differs in exactly one field
    baseline_key = ("L", "Q", "K", "primitive", "qlen", "content", "cold")
    for condition in conditions:
        params = condition.params()
        diffs = sum(1 for key in baseline_key if params[key] != BASELINE_PARAMS[key])
        assert diffs <= 1, f"{condition_id(params)} changes {diffs} factors at once"


BASELINE_PARAMS = {"L": 512, "Q": 4, "K": 8, "primitive": "choice", "qlen": "short",
                   "content": "natural", "cold": 0}


def test_sweep_respects_headroom_and_caps():
    with pytest.raises(ValueError, match="headroom"):
        SweepCondition(L=40_000, Q=4, K=8, primitive="choice", qlen="short",
                       content="natural", cold=0).validate()
    with pytest.raises(ValueError, match="question bound"):
        SweepCondition(L=512, Q=100, K=8, primitive="choice", qlen="short",
                       content="natural", cold=0).validate()
    with pytest.raises(ValueError, match="option cap"):
        SweepCondition(L=512, Q=4, K=300, primitive="choice", qlen="short",
                       content="natural", cold=0).validate()


def test_sweep_spec_passes_schema_and_has_block_layout():
    conditions = one_factor_sweep(l_levels=(128, 512), q_levels=(1, 8), k_levels=(2, 8))
    spec = sweep_spec(conditions, repeats=3, seed=7)
    assert spec["shuffle"] is False
    assert spec["claim_type"] == "confirmatory"
    # per block: 1 sentinel + len(conditions)
    assert len(spec["items"]) == 3 * (1 + len(conditions))
    # sentinel first in every block, identical content, distinct ids
    block_starts = [0, 1 + len(conditions), 2 * (1 + len(conditions))]
    for start in block_starts:
        assert spec["items"][start]["group"] == "sentinel"
    assert spec["items"][0]["state"] == spec["items"][block_starts[1]]["state"]
    assert spec["items"][0]["id"] != spec["items"][block_starts[1]]["id"]
    requests = logical_requests(spec)  # raises on schema problems
    assert len(requests) == 3 * (1 + len(conditions))


def test_sweep_conditions_encoded_in_items():
    conditions = one_factor_sweep(l_levels=(128, 2048))  # baseline (512) is included too
    spec = sweep_spec(conditions, repeats=1, seed=1)
    l_values = {condition_param(i["condition"], "L") for i in spec["items"] if i["group"] != "sentinel"}
    assert l_values == {"128", "2048", "512"}


def test_screening_cross_is_small_and_labelled():
    conditions = screening_cross((128, 512, 2048, 8192), (1, 4, 16, 64), (2, 8, 64, 255))
    # aligned (4) + anti-aligned (4) + Q×K (4) = 12, deduped
    assert len(conditions) <= 12
    spec = sweep_spec(conditions, repeats=2, seed=0)
    assert "NOT a full factorial" in spec["dataset"]["sampling"]["design_note"]


def test_randomize_blocks_deterministic_and_local():
    units = list(range(12))
    a = randomize_blocks(units, block_size=4, seed=42)
    b = randomize_blocks(units, block_size=4, seed=42)
    assert a == b
    assert sorted(a) == units  # permutation only
    # every block of 4 stays within the same original block
    for block_index in range(3):
        original = set(units[block_index * 4:(block_index + 1) * 4])
        shuffled = a[block_index * 4:(block_index + 1) * 4]
        assert set(shuffled) == original


def test_rename_and_reorder_keep_content():
    item = {
        "id": "x", "group": "g", "cluster": "c", "condition": "role=probe",
        "state": "s", "gold": {},
        "questions": {"b_second": {"type": "noul", "instructions": "B"},
                      "a_first": {"type": "noul", "instructions": "A"}},
    }
    renamed = rename_question_ids(item, style="opaque")
    assert set(renamed["questions"]) == {"q_00000", "q_00001"}
    # sorted order preserved through renaming
    assert renamed["questions"]["q_00000"]["instructions"] == "A"
    assert renamed["id"] == "x:opaque"
    assert condition_param(renamed["condition"], "renamed") == "opaque"
    reordered = reorder_questions(item, seed=3)
    assert set(reordered["questions"]) == {"a_first", "b_second"}  # same questions, new order
    assert list(reordered["questions"]) != list(item["questions"]) or len(item["questions"]) == 1


def test_isolation_spec_identical_states_and_clusters():
    spec = isolation_spec(sibling_levels=(0, 1, 8), repeats=3, seed=2)
    assert spec["claim_type"] == "exploratory"
    anchors = [i for i in spec["items"] if condition_param(i["condition"], "role") == "anchor"]
    # every anchor sees the byte-identical state and the same anchor question
    states = {i["state"] for i in anchors}
    assert len(states) == 1
    anchor_questions = {json.dumps(i["questions"]["anchor"], sort_keys=True) for i in anchors}
    assert len(anchor_questions) == 1
    # sibling sets differ as designed, same cluster within a repeat
    by_cluster: dict[str, set] = {}
    for item in anchors:
        by_cluster.setdefault(item["cluster"], set()).add(
            condition_param(item["condition"], "sibling_set"))
    assert all(levels == {"0", "1", "8"} for levels in by_cluster.values())
    assert len(by_cluster) == 3
    assert len(logical_requests(spec)) == 9


def test_odds_spec_variants_and_tracked_pair():
    spec = odds_spec(repeats=4, seed=0)
    variants = {condition_param(i["condition"], "odds_variant") for i in spec["items"]}
    assert variants == {"base", "irrelevant", "duplicate"}
    duplicate = next(i for i in spec["items"]
                     if condition_param(i["condition"], "odds_variant") == "duplicate")
    criteria = duplicate["questions"]["verdict"]["criteria"]
    # duplicate variant keeps A and adds A2 with the same description
    assert criteria["A"] == criteria["A2"]
    irrelevant = next(i for i in spec["items"]
                      if condition_param(i["condition"], "odds_variant") == "irrelevant")
    assert set(irrelevant["questions"]["verdict"]["criteria"]) >= set(BASE_KEYS)
    assert condition_param(spec["items"][0]["condition"], "tracked_pair") == "A,B"
    assert len(logical_requests(spec)) == 12


BASE_KEYS = {"A", "B", "C", "D"}
