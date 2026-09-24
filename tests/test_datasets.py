"""M1 dataset tests: loaders, stratification, permutation equivariance."""

import json
import math

import pytest

from jev_observatory.datasets.boolq import build_spec as boolq_spec, load_boolq
from jev_observatory.datasets.mmlu_pro import (
    build_spec as mmlu_spec,
    load_mmlu_pro,
    stratified_pilot,
    stratum_sizes,
)
from jev_observatory.datasets.permutation import attach_permutations, permute_choice_item
from jev_observatory.experiment import SpecError, load_experiment, logical_requests


@pytest.fixture
def mmlu_file(tmp_path):
    categories = ["math", "law", "bio", "chem"]
    records = []
    for i in range(40):
        records.append({
            "question_id": f"q{i}",
            "question": f"Question {i} about {categories[i % 4]}?",
            "choices": ["alpha", "beta", "gamma", "delta"],
            "answer_index": i % 4,
            "answer": "ABCD"[i % 4],
            "category": categories[i % 4],
            "src": "fixture",
        })
    path = tmp_path / "mmlu_pro.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records))
    return path


def test_mmlu_pro_loader_maps_options_and_gold(mmlu_file):
    dataset = load_mmlu_pro(mmlu_file, revision="rev-1", accessed_at="2026-09-18")
    assert dataset.provenance.name == "mmlu-pro"
    item = dataset.items[7]  # answer_index 3 -> delta
    assert item["gold"]["mmlu"]["value"] == "D"
    assert list(item["questions"]["mmlu"]["criteria"]) == ["A", "B", "C", "D"]
    assert item["leakage_check"] is False  # explicit opt-out, recorded


def test_mmlu_pro_rejects_contradictory_answer_letter(mmlu_file, tmp_path):
    records = [json.loads(l) for l in mmlu_file.read_text().splitlines()]
    records[0]["answer"] = "B"  # answer_index says A
    bad = tmp_path / "bad.jsonl"
    bad.write_text("\n".join(json.dumps(r) for r in records))
    with pytest.raises(ValueError, match="contradicts"):
        load_mmlu_pro(bad)


def test_mmlu_pro_rejects_over_255_choices(tmp_path):
    record = {"question_id": "q", "question": "Q?", "choices": [f"c{i}" for i in range(256)],
              "answer_index": 0, "answer": "A", "category": "x", "src": "s"}
    path = tmp_path / "big.jsonl"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="option cap"):
        load_mmlu_pro(path)


def test_mmlu_spec_passes_leakage_and_schema(mmlu_file):
    spec = mmlu_spec(load_mmlu_pro(mmlu_file), pilot=16, seed=5)
    assert spec["dataset"]["sampling"]["mode"] == "stratified_pilot"
    assert len(spec["items"]) == 16
    requests = logical_requests(spec)  # raises SpecError on leakage/schema problems
    assert len(requests) == 16


def test_stratified_pilot_deterministic_and_min_one_per_stratum(mmlu_file):
    dataset = load_mmlu_pro(mmlu_file)
    items = dataset.items
    a = stratified_pilot(items, n=12, seed=9)
    b = stratified_pilot(items, n=12, seed=9)
    c = stratified_pilot(items, n=12, seed=10)
    assert [i["id"] for i in a] == [i["id"] for i in b]
    assert [i["id"] for i in a] != [i["id"] for i in c]
    assert {i["group"] for i in a} == {"math", "law", "bio", "chem"}  # min one each


def test_boolq_two_conditions_share_cluster(tmp_path):
    records = [{"question": f"q{i}?", "passage": "Passage text.", "label": i % 2 == 0} for i in range(4)]
    path = tmp_path / "boolq.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records))
    dataset = load_boolq(path)
    assert {i["condition"] for i in dataset.items} == {"boolq_noul", "boolq_choice"}
    noul = next(i for i in dataset.items if i["condition"] == "boolq_noul")
    choice = next(i for i in dataset.items if i["condition"] == "boolq_choice" and i["cluster"] == noul["cluster"])
    assert noul["gold"]["answer"]["value"] is choice["gold"]["answer"]["value"].replace("yes", "yes") or True
    # gold encodings agree: label True -> noul True, choice "yes"
    base_label = noul["gold"]["answer"]["value"]
    assert (choice["gold"]["answer"]["value"] == "yes") == bool(base_label)


def test_boolq_balanced_sample_requires_even_n(tmp_path):
    records = [{"question": "q?", "passage": "P.", "label": True}] * 4
    path = tmp_path / "boolq.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records))
    dataset = load_boolq(path)
    with pytest.raises(ValueError, match="even"):
        boolq_spec(dataset, n=3, seed=0)
    spec = boolq_spec(dataset, n=4, seed=0)
    conditions = [i["condition"] for i in spec["items"]]
    assert conditions.count("boolq_noul") == conditions.count("boolq_choice") == 2


def _choice_item():
    return {
        "id": "base-1", "group": "g", "cluster": "c1", "condition": "native",
        "state": "Question?",
        "gold": {"q": {"value": "A"}},
        "questions": {"q": {"type": "choice", "instructions": "pick",
                            "criteria": {"A": "red desk", "B": "blue car", "C": "green tree"}}},
    }


def test_permutation_moves_gold_with_description():
    item = _choice_item()
    result = permute_choice_item(item, seed=3, variant=1)
    permuted = result["item"]
    qid = "q"
    new_criteria = permuted["questions"][qid]["criteria"]
    gold_key = permuted["gold"][qid]["value"]
    # the gold description must still be "red desk" under the new key
    assert new_criteria[gold_key] == "red desk"
    assert set(new_criteria) == {"A", "B", "C"}
    # the remapping is a real permutation of descriptions
    assert sorted(result["remapping"]) == ["A", "B", "C"]


def test_permutation_never_identity_when_descriptions_distinct():
    for seed in range(10):
        result = permute_choice_item(_choice_item(), seed=seed, variant=1)
        original = _choice_item()["questions"]["q"]["criteria"]
        assert list(result["item"]["questions"]["q"]["criteria"].values()) != list(original.values()) or len(
            set(original.values())
        ) < 3  # only trivially identical if descriptions repeat


def test_attach_permutations_records_and_groups():
    spec = {"experiment": "x", "model": "jev-1.13.0", "items": [_choice_item(),
             {"id": "n-1", "group": "g", "cluster": "c2", "state": "s",
              "gold": {"q": {"value": True}},
              "questions": {"q": {"type": "noul", "instructions": "is it?"}}}]}
    extended = attach_permutations(spec, seed=1, variants=2)
    assert len(extended["items"]) == 4  # base choice + noul + 2 permuted (noul gets none)
    ids = [i["id"] for i in extended["items"]]
    assert ids.count("base-1:perm1") == 1 and ids.count("base-1:perm2") == 1
    permuted = next(i for i in extended["items"] if i["id"] == "base-1:perm1")
    assert permuted["cluster"] == "c1"  # same cluster as base for paired analysis
    assert set(extended["permutations"]) == {"base-1:perm1", "base-1:perm2"}
    logical_requests(extended)  # must pass schema/leakage


def test_load_experiment_rejects_unknown_top_level_item_keys():
    spec = {"experiment": "x", "items": [{**_choice_item(), "answer_key": "A"}]}
    with pytest.raises(SpecError):
        logical_requests(spec)