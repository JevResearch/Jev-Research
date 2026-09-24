"""Full-benchmark suite builder tests (offline fixtures only, no network).

Covers: exact full-split count enforcement, gold isolation in outbound
payloads, option mapping after rotations, sidecar verification, subset
determinism, request-ceiling enforcement.
"""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from jev_observatory.benchmark_spec import (
    DEFAULT_EXPECTED,
    N_ARC_CHALLENGE_TEST,
    N_MMLU_PRO_TEST,
    N_ROTATION_ITEMS,
    REQUEST_CEILING,
    BenchmarkSpecError,
    build_benchmark_suite,
    build_rotation_items,
    subject_stratified_subset,
    verify_data_sidecar,
)
from jev_observatory.experiment import logical_requests


def _write_mmlu_fixture(path: Path, n_items: int = 20,
                        subjects: tuple[str, ...] = ("math", "law")) -> None:
    records = []
    letters = "ABCDEFGHIJ"
    for i in range(n_items):
        n_options = 4 + (i % 7)  # 4..10 options
        records.append({
            "question_id": f"q{i:04d}",
            "question": f"Fixture question {i}: which option is best?",
            "options": [f"option text {i}.{j}" for j in range(n_options)],
            "answer_index": i % n_options,
            "answer": letters[i % n_options],
            "category": subjects[i % len(subjects)],
            "src": "fixture",
        })
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


def _write_arc_fixture(path: Path, n_items: int = 6) -> None:
    records = []
    labels = ["A", "B", "C", "D"]
    for i in range(n_items):
        records.append({
            "id": f"arc{i:04d}",
            "question": f"ARC fixture question {i}?",
            "choices": {"text": [f"choice {i}.{j}" for j in range(4)], "label": labels},
            "answerKey": labels[i % 4],
        })
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


def _sidecar(path: Path, url: str = "fixture://source") -> None:
    Path(str(path) + ".sha256.json").write_text(json.dumps({
        "file": str(path), "sha256": sha256(path.read_bytes()).hexdigest(), "url": url,
    }) + "\n", encoding="utf-8")


@pytest.fixture
def fixture_data(tmp_path):
    mmlu = tmp_path / "mmlu_test.jsonl"
    arc = tmp_path / "arc_test.jsonl"
    _write_mmlu_fixture(mmlu, n_items=20)   # 2 subjects x 10 -> 10/subject subset
    _write_arc_fixture(arc, n_items=6)
    _sidecar(mmlu)
    _sidecar(arc)
    return {"mmlu": mmlu, "arc": arc}


EXPECTED = {"mmlu_full": 20, "arc_test": 6}


def test_constants_match_the_gate():
    assert DEFAULT_EXPECTED["mmlu_full"] == 12032
    assert DEFAULT_EXPECTED["arc_test"] == 1172
    assert DEFAULT_EXPECTED["arc_agi2"] == 120   # ARC-AGI-2 public eval (late-2026 ext)
    assert DEFAULT_EXPECTED["math500"] == 500    # MATH-500 (late-2026 ext)
    assert N_MMLU_PRO_TEST == 12032
    assert N_ARC_CHALLENGE_TEST == 1172
    assert N_ROTATION_ITEMS == 420
    assert REQUEST_CEILING == 15000
    assert N_MMLU_PRO_TEST + N_ROTATION_ITEMS + N_ARC_CHALLENGE_TEST <= REQUEST_CEILING


def test_full_suite_counts_and_structural_gold_isolation(fixture_data):
    suite = build_benchmark_suite(
        mmlu_path=fixture_data["mmlu"], arc_path=fixture_data["arc"],
        expected=EXPECTED)
    assert suite["counts"] == {"mmlu_full": 20, "option_rotations": 60, "arc_test": 6}
    assert suite["total_requests"] == 86

    for stage in ("mmlu_full", "option_rotations", "arc_test"):
        spec = suite[stage]["spec"]
        for lr in logical_requests(spec):
            payload = lr.request.to_payload()
            assert set(payload) == {"state", "model", "questions"}
            for question in payload["questions"].values():
                assert set(question) == {"type", "instructions", "criteria"}
            serialized = json.dumps(payload)
            for forbidden in ("answer_index", "answer", "cot_content", "answerKey",
                              "answer_key", "gold"):
                assert f'"{forbidden}"' not in serialized, (stage, lr.logical_request_id)


def test_option_mapping_after_rotations_is_canonical(fixture_data):
    suite = build_benchmark_suite(
        mmlu_path=fixture_data["mmlu"], arc_path=fixture_data["arc"],
        expected=EXPECTED)
    rotation_spec = suite["option_rotations"]["spec"]
    rotations = rotation_spec["rotations"]
    assert len(rotations) == 60
    base_ids = set(rotation_spec["protocol"]["base_item_ids"])
    assert len(base_ids) == 20  # 10 per subject x 2 subjects

    base_items = {it["id"]: it for it in suite["mmlu_full"]["spec"]["items"]}
    for rot_id, record in rotations.items():
        remapping = record["remapping"]
        assert sorted(remapping) == sorted(remapping.values())  # a true permutation
        base = base_items[record["base_item"]]
        base_criteria = base["questions"]["mmlu"]["criteria"]
        rot_item = next(it for it in rotation_spec["items"] if it["id"] == rot_id)
        rot_criteria = rot_item["questions"]["mmlu"]["criteria"]
        for key in base_criteria:
            assert rot_criteria[remapping[key]] == base_criteria[key]
        # gold follows its description to the new key
        gold = rot_item["gold"]["mmlu"]["value"]
        assert rot_criteria[gold] == base_criteria[base["gold"]["mmlu"]["value"]]


def _synthetic_items():
    return [
        {"id": f"q{i}", "group": f"s{i % 2}", "state": f"st{i}",
         "questions": {"mmlu": {"type": "choice", "instructions": "i",
                                "criteria": {k: f"o{idx}" for idx, k in
                                             enumerate("ABCD")}}},
         "gold": {"mmlu": {"value": "A"}}, "leakage_check": False}
        for i in range(10)]


def test_rotations_are_deterministic():
    a, records_a = build_rotation_items(
        subject_stratified_subset(_synthetic_items(), per_subject=5))
    b, records_b = build_rotation_items(
        subject_stratified_subset(_synthetic_items(), per_subject=5))
    assert [it["id"] for it in a] == [it["id"] for it in b]
    assert records_a == records_b


def test_subset_is_subject_stratified_and_deterministic(fixture_data):
    suite = build_benchmark_suite(
        mmlu_path=fixture_data["mmlu"], arc_path=fixture_data["arc"],
        expected=EXPECTED)
    items = suite["mmlu_full"]["spec"]["items"]
    subset = subject_stratified_subset(items, per_subject=2)
    by_subject: dict[str, int] = {}
    for it in subset:
        by_subject[it["group"]] = by_subject.get(it["group"], 0) + 1
    assert set(by_subject.values()) == {2}
    again = subject_stratified_subset(items, per_subject=2)
    assert [it["id"] for it in subset] == [it["id"] for it in again]


def test_sidecar_verification_and_tamper_refusal(fixture_data):
    record = verify_data_sidecar(fixture_data["mmlu"])
    assert record["sha256"] == sha256(fixture_data["mmlu"].read_bytes()).hexdigest()
    fixture_data["mmlu"].write_text("tampered\n", encoding="utf-8")
    with pytest.raises(BenchmarkSpecError, match="does not match sidecar"):
        verify_data_sidecar(fixture_data["mmlu"])


def test_wrong_full_split_count_is_refused(fixture_data):
    # production defaults demand the FULL splits: the 20-item fixture is refused
    with pytest.raises(BenchmarkSpecError, match="never a pilot"):
        build_benchmark_suite(mmlu_path=fixture_data["mmlu"],
                              arc_path=fixture_data["arc"])


def test_request_ceiling_enforced(fixture_data, monkeypatch):
    import jev_observatory.benchmark_spec as spec_module

    monkeypatch.setattr(spec_module, "REQUEST_CEILING", 50)
    with pytest.raises(BenchmarkSpecError, match="request ceiling"):
        build_benchmark_suite(mmlu_path=fixture_data["mmlu"],
                              arc_path=fixture_data["arc"], expected=EXPECTED)
