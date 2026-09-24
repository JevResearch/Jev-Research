"""Benchmark scorer tests: gold-source parity, honest accounting, robustness.

Offline only. Builds tiny run directories through the real executor + mock
provider, then checks hand-computed parity: accuracy, per-subject breakdown,
zero-probability gold count, proper scores and exclusions, strict-format
failure counting, rotation canonical-label restoration, finite-set caveat
presence, and refusal to score against missing or mutated gold.
"""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from jev_observatory.benchmark_exec import (
    BenchmarkExecutor,
    build_freeze,
    stage_run_id,
)
from jev_observatory.benchmark_score import (
    ScoringError,
    load_items_gold,
    score_run,
)
from jev_observatory.benchmark_spec import build_benchmark_suite
from jev_observatory.ledger import RunStore
from jev_observatory.providers import JevProvider, MockProvider, RetryPolicy
from jev_observatory.runner import plan_run
from jev_observatory.transport import ScriptedTransport, TransportResponse

EXPECTED = {"mmlu_full": 20, "arc_test": 6}

def _prepare_stage(root: Path, freeze: dict, suite: dict, stage: str) -> Path:
    run_id = stage_run_id(freeze, stage)
    directory = root / run_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "spec.json").write_text(
        json.dumps(suite[stage]["spec"], ensure_ascii=False) + "\n", encoding="utf-8")
    if stage == "option_rotations":
        (directory / "rotations.json").write_text(
            json.dumps(suite[stage]["spec"]["rotations"], ensure_ascii=False) + "\n",
            encoding="utf-8")
    plan_run(suite[stage]["spec"], provider="jev", root=str(root), run_id=run_id,
             retry_policy={"max_attempts": 1})
    return directory



def _mmlu_records(n: int = 20) -> list[dict]:
    letters = "ABCDEFGHIJ"
    return [{
        "question_id": f"q{i:04d}",
        "question": f"Fixture question {i}?",
        "options": [f"option {i}.{j}" for j in range(4 + (i % 7))],
        "answer_index": i % (4 + (i % 7)),
        "answer": letters[i % (4 + (i % 7))],
        "category": ("math", "law")[i % 2],
        "src": "fixture",
    } for i in range(n)]


def _arc_records(n: int = 6) -> list[dict]:
    labels = ["A", "B", "C", "D"]
    return [{
        "id": f"arc{i:04d}", "question": f"ARC fixture {i}?",
        "choices": {"text": [f"c{i}.{j}" for j in range(4)], "label": labels},
        "answerKey": labels[i % 4],
    } for i in range(n)]


@pytest.fixture
def scored_mmlu_run(tmp_path) -> tuple[Path, dict, dict]:
    """Prepare + dispatch (mock) the mmlu_full stage; return (run_dir, freeze, doc)."""
    data = tmp_path / "data"
    data.mkdir()
    mmlu_path = data / "mmlu_test.jsonl"
    mmlu_path.write_text(
        "\n".join(json.dumps(r) for r in _mmlu_records()) + "\n", encoding="utf-8")
    Path(str(mmlu_path) + ".sha256.json").write_text(json.dumps({
        "file": str(mmlu_path), "sha256": sha256(mmlu_path.read_bytes()).hexdigest(),
        "url": "fixture://source"}) + "\n", encoding="utf-8")
    root = tmp_path / "runs_benchmark"
    suite = build_benchmark_suite(mmlu_path=mmlu_path, arc_path=mmlu_path,
                                  expected={"mmlu_full": 20},
                                  skip_arc=True)
    freeze = build_freeze(suite, root=root)
    directory = _prepare_stage(root, freeze, suite, "mmlu_full")
    executor = BenchmarkExecutor(root, provider_factory=lambda: MockProvider(seed=13),
                                 concurrency=2, stages=("mmlu_full",))
    state = executor.run(dry_run=False)
    doc = score_run(directory, stage="mmlu_full", rotations={})
    return directory, freeze, doc


def test_gold_source_matches_frozen_items(scored_mmlu_run):
    directory, freeze, _doc = scored_mmlu_run
    gold = load_items_gold(directory)
    assert len(gold) == 20
    items = [json.loads(line) for line
             in (directory / "items.jsonl").read_text().splitlines() if line]
    for item in items:
        assert gold[item["id"]]["value"] == item["gold"]["mmlu"]["value"]


def test_accuracy_parity_with_hand_computed_values(scored_mmlu_run):
    directory, freeze, doc = scored_mmlu_run
    # MockProvider picks the argmax of seeded random probabilities per question;
    # recompute the expectation independently from the stored results.
    results = [json.loads(line) for line
               in (directory / "results.jsonl").read_text().splitlines() if line]
    gold = load_items_gold(directory)
    correct = 0
    by_group: dict[str, list[bool]] = {}
    for result in results:
        answer = next(iter(result["predictions"].values()))
        is_correct = answer["choice"] == gold[result["item_id"]]["value"]
        correct += is_correct
        by_group.setdefault(gold[result["item_id"]]["group"], []).append(is_correct)
    assert doc["n_scored"] == 20
    assert doc["accuracy"] == correct / 20
    assert doc["status_counts"]["ok"] == 20
    for group, flags in by_group.items():
        assert doc["by_group"][group]["n"] == len(flags)
        assert doc["by_group"][group]["accuracy"] == sum(flags) / len(flags)
    assert doc["finite_set_caveat"]


def test_proper_scores_and_zero_probability_gold(scored_mmlu_run):
    _directory, _freeze, doc = scored_mmlu_run
    proper = doc["proper_scores"]
    assert proper["n"] == 20  # mock answers carry valid probability vectors
    assert proper["excluded_no_valid_probabilities"] == 0
    assert proper["brier_mean"] is not None
    results = [json.loads(line) for line
               in (scored_mmlu_run[0] / "results.jsonl").read_text().splitlines() if line]
    gold = load_items_gold(scored_mmlu_run[0])
    zero = 0
    for result in results:
        answer = next(iter(result["predictions"].values()))
        if answer["probabilities"].get(gold[result["item_id"]]["value"]) == 0.0:
            zero += 1
    assert doc["zero_probability_gold"] == zero
    assert doc["zero_probability_gold_candidates"] == 20


def test_invalid_probabilities_are_excluded_not_repaired(tmp_path, scored_mmlu_run):
    directory, freeze, _ = scored_mmlu_run
    run_id = stage_run_id(freeze, "mmlu_full")
    results_path = directory / "results.jsonl"
    results = [json.loads(line) for line in results_path.read_text().splitlines() if line]
    # a probability vector that does not sum to ~1 must be EXCLUDED, counted
    answer = next(iter(results[0]["predictions"].values()))
    answer["probabilities"] = {k: 0.9 for k in answer["probabilities"]}
    results_path.write_text(
        "\n".join(json.dumps(r) for r in results) + "\n", encoding="utf-8")
    doc = score_run(directory, stage="mmlu_full", rotations={})
    assert doc["proper_scores"]["excluded_no_valid_probabilities"] == 1
    assert doc["proper_scores"]["n"] == 19


def test_strict_format_failures_are_counted(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    mmlu_path = data / "mmlu_test.jsonl"
    mmlu_path.write_text(
        "\n".join(json.dumps(r) for r in _mmlu_records()) + "\n", encoding="utf-8")
    Path(str(mmlu_path) + ".sha256.json").write_text(json.dumps({
        "file": str(mmlu_path), "sha256": sha256(mmlu_path.read_bytes()).hexdigest(),
        "url": "fixture://source"}) + "\n", encoding="utf-8")
    root = tmp_path / "runs_benchmark"
    suite = build_benchmark_suite(mmlu_path=mmlu_path, arc_path=mmlu_path,
                                  expected={"mmlu_full": 20}, skip_arc=True)
    freeze = build_freeze(suite, root=root)
    directory = _prepare_stage(root, freeze, suite, "mmlu_full")
    # a contract-invalid body: prose instead of an exact option key
    invalid_body = {"status_code": 200, "body": {"model": "jev-1.13.0", "answers": {
        "q": {"type": "choice", "choice": "the best option is A",
              "probabilities": {"A": 1.0}, "confidence": 0.0}},
        "usage": {"input_tokens": 100, "output_tokens": 10}}}
    transport = ScriptedTransport([invalid_body] * 20)
    provider = JevProvider(transport, retry_policy=RetryPolicy.none())
    BenchmarkExecutor(root, provider_factory=lambda: provider, concurrency=1,
                      consecutive_failure_limit=25,
                      stages=("mmlu_full",)).run(dry_run=False)
    doc = score_run(directory, stage="mmlu_full", rotations={})
    assert doc["strict_format_failures"] == 20
    assert doc["status_counts"]["contract_invalid"] == 20
    assert doc["n_scored"] == 0
    assert doc["accuracy"] is None  # no fake accuracy from unusable answers


def test_rotation_scoring_restores_canonical_labels(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    mmlu_path = data / "mmlu_test.jsonl"
    mmlu_path.write_text(
        "\n".join(json.dumps(r) for r in _mmlu_records()) + "\n", encoding="utf-8")
    Path(str(mmlu_path) + ".sha256.json").write_text(json.dumps({
        "file": str(mmlu_path), "sha256": sha256(mmlu_path.read_bytes()).hexdigest(),
        "url": "fixture://source"}) + "\n", encoding="utf-8")
    root = tmp_path / "runs_benchmark"
    suite = build_benchmark_suite(mmlu_path=mmlu_path, arc_path=mmlu_path,
                                  expected={"mmlu_full": 20}, skip_arc=True)
    freeze = build_freeze(suite, root=root)
    directory = _prepare_stage(root, freeze, suite, "option_rotations")
    BenchmarkExecutor(root, provider_factory=lambda: MockProvider(seed=17),
                      concurrency=2,
                      stages=("option_rotations",)).run(dry_run=False)
    doc = score_run(directory, stage="option_rotations")
    assert doc["n_scored"] == 60
    assert set(doc["by_rotation_variant"]) == {"perm1", "perm2", "perm3"}
    for variant, block in doc["by_rotation_variant"].items():
        assert block["n"] == 20
        assert block["finite_set_caveat"]
    # canonical restoration must agree with the direct rotated-gold comparison:
    # correct ⟺ canonical (restored) choice equals the restored base gold letter
    for record in doc["per_item"]:
        assert record["correct"] is not None
        assert (record["canonical_choice"] == record["gold_canonical"]) == \
            record["correct"]


def test_missing_terminal_results_are_counted_not_dropped(scored_mmlu_run):
    directory, freeze, _ = scored_mmlu_run
    run_id = stage_run_id(freeze, "mmlu_full")
    results_path = directory / "results.jsonl"
    results = [json.loads(line) for line in results_path.read_text().splitlines() if line]
    results_path.write_text(
        "\n".join(json.dumps(r) for r in results[:15]) + "\n", encoding="utf-8")
    doc = score_run(directory, stage="mmlu_full", rotations={})
    assert doc["n_terminal_results"] == 15
    assert doc["n_scored"] == 15
    assert doc["n_missing_terminal"] == 5
    assert len(doc["missing_logical_ids"]) == 5


def test_refuses_to_score_items_without_gold(scored_mmlu_run):
    directory, _freeze, _ = scored_mmlu_run
    items_path = directory / "items.jsonl"
    items = [json.loads(line) for line in items_path.read_text().splitlines() if line]
    del items[0]["gold"]
    items_path.write_text(
        "\n".join(json.dumps(r) for r in items) + "\n", encoding="utf-8")
    with pytest.raises(ScoringError, match="no gold value"):
        load_items_gold(directory)


def test_unknown_item_in_results_is_refused(scored_mmlu_run):
    directory, freeze, _ = scored_mmlu_run
    run_id = stage_run_id(freeze, "mmlu_full")
    results_path = directory / "results.jsonl"
    results = [json.loads(line) for line in results_path.read_text().splitlines() if line]
    rogue = dict(results[0])
    rogue["logical_request_id"] = "native:rogue-item"
    results_path.write_text(
        "\n".join(json.dumps(r) for r in results + [rogue]) + "\n", encoding="utf-8")
    with pytest.raises(ScoringError, match="not in the frozen gold set"):
        score_run(directory, stage="mmlu_full", rotations={})
