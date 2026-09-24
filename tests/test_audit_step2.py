"""Audit follow-up tests: block shuffle, fresh generators, paired scorer."""

import json
import math
from collections import Counter
from pathlib import Path

import pytest

from jev_observatory.datasets.generators import (
    GENERATOR_VERSION,
    build_synthetic_spec,
    gen_date_window,
    gen_multihop,
    gen_policy_routing,
    solve_date_window,
    solve_multihop,
)
from jev_observatory.experiment import logical_requests
from jev_observatory.experiments import build_sweep_items, one_factor_sweep, condition_param
from jev_observatory.paired import paired_encoding_comparison


# ------------------------------------------------------------- block shuffle
def test_sweep_conditions_shuffled_within_blocks_and_reproducible():
    conditions = one_factor_sweep(l_levels=(128, 512, 2048, 8192), q_levels=(1, 4))
    spec_items_a = build_sweep_items(conditions, repeats=3, seed=99)
    spec_items_b = build_sweep_items(conditions, repeats=3, seed=99)
    spec_items_c = build_sweep_items(conditions, repeats=3, seed=100)
    assert [i["id"] for i in spec_items_a] == [i["id"] for i in spec_items_b]
    assert [i["id"] for i in spec_items_a] != [i["id"] for i in spec_items_c]
    # each block: sentinel first, then every condition exactly once, order varies
    block_size = 1 + len(conditions)
    for block in range(3):
        members = spec_items_a[block * block_size + 1:(block + 1) * block_size]
        orders = [tuple(int(condition_param(m["condition"], "c_index") or -1) for m in members)]
        assert len({m["id"].split("-c")[1] for m in members}) == len(conditions)
    # the *positions* of conditions differ between blocks (actual shuffling)
    positions = []
    for block in range(3):
        members = spec_items_a[block * block_size + 1:(block + 1) * block_size]
        positions.append([m["id"].split("-c")[1] for m in members])
    assert positions[0] != positions[1], "conditions must be reordered between blocks"


def test_sweep_items_carry_block_index():
    conditions = one_factor_sweep(l_levels=(128, 512))
    spec_items = build_sweep_items(conditions, repeats=2, seed=5)
    for item in spec_items:
        block = condition_param(item["condition"], "block")
        assert block is not None, "every sweep item must record its block"
        expected = f"b{int(block):02d}"
        assert expected in item["id"], f"{item['id']} does not match its block {block}"


def test_planned_to_executed_block_preservation(tmp_path):
    """plan → reconstructed spec → logical_requests must keep block order."""
    from jev_observatory.runner import load_manifest, plan_run

    conditions = one_factor_sweep(l_levels=(128, 512), q_levels=(1, 4))
    spec = {
        "experiment": "block-order",
        "model": "jev-1.13.0",
        "shuffle": False,
        "seeds": {"order": 3},
        "items": build_sweep_items(conditions, repeats=2, seed=3),
    }
    run_id, _ = plan_run(spec, provider="mock", root=str(tmp_path))
    from jev_observatory.runner import load_run_store

    # reconstruct via the CLI path and check order survives
    manifest = load_manifest(run_id, str(tmp_path))
    assert manifest.shuffle is False
    items_path = tmp_path / run_id / "items.jsonl"
    reconstructed = {
        "experiment": manifest.experiment,
        "model": manifest.model_requested,
        "items": [json.loads(l) for l in items_path.read_text().splitlines() if l.strip()],
        "seeds": manifest.seeds,
        "shuffle": manifest.shuffle,
    }
    requests = logical_requests(reconstructed)
    assert [r.item.item_id for r in requests] == [i["id"] for i in spec["items"]], \
        "reconstructed dispatch order must equal the designed block order"


# ------------------------------------------------------------- fresh generators
def test_multihop_labels_verified_independently():
    for seed in (0, 4, 9):
        for item in gen_multihop(seed, 12):
            assert solve_multihop(item) == item["gold"][list(item["gold"])[0]]["value"]


def test_multihop_binary_labels_balanced():
    items = gen_multihop(1, 40)
    binary = [i for i in items if i["group"] == "multihop_binary"]
    values = Counter(i["gold"]["manages"]["value"] for i in binary)
    assert abs(values[True] - values[False]) <= 2


def test_date_window_balanced_and_boundary_covered():
    items = gen_date_window(2, 80)
    values = Counter(i["gold"]["eligible"]["value"] for i in items)
    assert values[True] == values[False]
    gaps = {i["_params"]["gap"] for i in items}
    assert 30 in gaps and 31 in gaps  # both sides of the boundary present
    assert all(solve_date_window(i) == i["gold"]["eligible"]["value"] for i in items)


def test_policy_labels_balanced_v2():
    items = gen_policy_routing(5, 100)
    counts = Counter(i["gold"]["decision"]["value"] for i in items)
    assert counts["approve"] == counts["deny"]
    assert counts["escalate"] > 0


def test_fresh_spec_records_generator_version():
    spec = build_synthetic_spec(
        [("multihop", gen_multihop), ("datewin", gen_date_window), ("policy", gen_policy_routing)],
        seed=31, n_per_generator=30)
    assert spec["generator_version"] == GENERATOR_VERSION
    assert len(logical_requests(spec)) == 90


# ------------------------------------------------------------- paired scorer
def _paired_fixture():
    """Six base records, two encodings, hand-computed metrics."""
    items = {}
    results = []
    golds = [True, False, True, True, False, False]
    choice_p = [0.9, 0.6, 0.8, 0.3, 0.2, 0.05]
    noul_p = [0.7, 0.5, 0.6, 0.4, 0.3, 0.1]
    for i, (gold, cp, np_) in enumerate(zip(golds, choice_p, noul_p)):
        items[f"item-{i}:choice"] = {"id": f"item-{i}:choice", "gold": {"answer": {"value": "yes" if gold else "no"}}}
        items[f"item-{i}:noul"] = {"id": f"item-{i}:noul", "gold": {"answer": {"value": gold}}}
        choice_choice = "yes" if cp >= 0.5 else "no"
        results.append({
            "logical_request_id": f"c{i}", "item_id": f"item-{i}:choice",
            "cluster": f"base-{i}", "condition": "boolq_choice",
            "predictions": {"answer": {"type": "choice", "usable": True, "choice": choice_choice,
                                       "probabilities": {"yes": cp, "no": round(1 - cp, 6)}, "p_max": max(cp, 1 - cp)}},
        })
        results.append({
            "logical_request_id": f"n{i}", "item_id": f"item-{i}:noul",
            "cluster": f"base-{i}", "condition": "boolq_noul",
            "predictions": {"answer": {"type": "noul", "usable": True, "noul": np_}},
        })
    return results, items


def test_paired_scorer_hand_computed():
    results, items = _paired_fixture()
    out = paired_encoding_comparison(results, items, n_bootstrap=400, seed=0)
    assert out["available"] is True
    assert out["n_pairs"] == 6
    choice, noul = out["per_condition"]["boolq_choice"], out["per_condition"]["boolq_noul"]
    # hand-check choice: golds yes,no,yes,yes,no,no; P(yes)=.9,.6,.8,.3,.2,.05
    # correct: yes(0.9->yes ok), no(0.6->choice yes wrong), yes(0.8->yes), yes(0.3->no wrong), no(0.2->no), no(0.05->no)
    assert choice["accuracy"] == pytest.approx(4 / 6)
    # noul: 0.7 yes ok, 0.5 yes wrong, 0.6 yes ok, 0.4 no wrong, 0.3 no ok, 0.1 no ok
    assert noul["accuracy"] == pytest.approx(4 / 6)
    brier_choice = ((0.9 - 1) ** 2 + (0.6 - 0) ** 2 + (0.8 - 1) ** 2 + (0.3 - 1) ** 2
                    + (0.2 - 0) ** 2 + (0.05 - 0) ** 2) / 6
    assert choice["binary_brier_p_yes"] == pytest.approx(brier_choice)
    brier_noul = ((0.7 - 1) ** 2 + (0.5 - 0) ** 2 + (0.6 - 1) ** 2 + (0.4 - 1) ** 2
                  + (0.3 - 0) ** 2 + (0.1 - 0) ** 2) / 6
    assert noul["binary_brier_p_yes"] == pytest.approx(brier_noul)
    assert out["mcnemar_exact_p"] == pytest.approx(1.0)  # 2 vs 2 discordant
    assert -0.5 <= out["accuracy_difference"] <= 0.5


def test_paired_scorer_reproduces_saved_live_artifacts():
    """Regression: the corrected scorer must reproduce the audited numbers on
    the saved paired BoolQ run (skipped if artifacts are absent)."""
    run_dir = Path("runs_live/2026-09-18-e1a2252d3ec8")
    if not (run_dir / "results.jsonl").exists():
        pytest.skip("saved paired BoolQ run not present")
    results = [json.loads(l) for l in (run_dir / "results.jsonl").read_text().splitlines() if l.strip()]
    items = {json.loads(l)["id"]: json.loads(l) for l in (run_dir / "items.jsonl").read_text().splitlines() if l.strip()}
    out = paired_encoding_comparison(results, items)
    assert out["available"] and out["n_pairs"] == 500
    assert out["per_condition"]["boolq_choice"]["accuracy"] == pytest.approx(0.898)
    assert out["per_condition"]["boolq_noul"]["accuracy"] == pytest.approx(0.896)
    assert out["per_condition"]["boolq_choice"]["binary_brier_p_yes"] == pytest.approx(0.0898544)
    assert out["per_condition"]["boolq_noul"]["binary_brier_p_yes"] == pytest.approx(0.0783124)
    assert out["mcnemar_exact_p"] == pytest.approx(1.0)
    assert out["brier_difference_first_minus_second"] == pytest.approx(0.011542)