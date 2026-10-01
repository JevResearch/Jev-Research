"""Benchmark scorer tests: gold-source parity, honest accounting, robustness.

Offline only. Builds tiny run directories through the real executor + mock
provider, then checks hand-computed parity: accuracy, per-subject breakdown,
zero-probability gold count, proper scores and exclusions, strict-format
failure counting, rotation canonical-label restoration, finite-set caveat
presence, and refusal to score against missing or mutated gold.
"""

from __future__ import annotations

import json
import math
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
    assert doc["n_usable_responses"] == 20
    assert doc["accuracy"] == correct / 20
    assert doc["status_counts"]["ok"] == 20
    for group, flags in by_group.items():
        assert doc["by_group"][group]["n"] == len(flags)
        assert doc["by_group"][group]["accuracy"] == sum(flags) / len(flags)
    assert doc["n_requested"] == 20
    assert doc["n_usable_responses"] == 20
    assert doc["usable_response_accuracy"] == correct / 20
    assert doc["n_correct"] == correct
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
    assert doc["n_usable_responses"] == 0
    assert doc["n_correct"] == 0
    assert doc["accuracy"] == 0.0  # all-requested headline: invalid counts wrong
    assert doc["usable_response_accuracy"] is None  # no usable responses at all


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
    assert doc["n_requested"] == 60
    assert doc["n_usable_responses"] == 60
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
    assert doc["n_usable_responses"] == 15
    assert doc["n_missing_terminal"] == 5
    assert len(doc["missing_logical_ids"]) == 5
    # headline denominator is ALL requested: 5 missing items count as wrong
    assert doc["n_requested"] == 20
    assert doc["accuracy"] == doc["n_correct"] / 20
    assert doc["usable_response_accuracy"] == doc["n_correct"] / 15


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


def test_duplicate_terminal_logical_ids_are_refused(scored_mmlu_run):
    directory, _freeze, _doc = scored_mmlu_run
    results_path = directory / "results.jsonl"
    results = [json.loads(line) for line in results_path.read_text().splitlines() if line]
    results_path.write_text(
        "\n".join(json.dumps(r) for r in results + [results[0]]) + "\n", encoding="utf-8")
    with pytest.raises(ScoringError, match="duplicate terminal logical_request_id"):
        score_run(directory, stage="mmlu_full", rotations={})


def test_duplicate_terminal_item_ids_are_refused(scored_mmlu_run):
    directory, _freeze, _doc = scored_mmlu_run
    results_path = directory / "results.jsonl"
    results = [json.loads(line) for line in results_path.read_text().splitlines() if line]
    clone = dict(results[0])
    clone["logical_request_id"] = "other-condition:" + clone["logical_request_id"].partition(":")[2]
    results_path.write_text(
        "\n".join(json.dumps(r) for r in results + [clone]) + "\n", encoding="utf-8")
    with pytest.raises(ScoringError, match="duplicate terminal item_id"):
        score_run(directory, stage="mmlu_full", rotations={})


def test_headline_counts_invalid_and_missing_as_wrong(tmp_path):
    """Mixed run: 1 correct, 1 wrong, 1 invalid, 1 missing -> headline 1/4."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    items = [
        {"id": "i1", "group": "g", "gold": {"q": {"value": "A"}}},
        {"id": "i2", "group": "g", "gold": {"q": {"value": "A"}}},
        {"id": "i3", "group": "g", "gold": {"q": {"value": "A"}}},
        {"id": "i4", "group": "g", "gold": {"q": {"value": "A"}}},
    ]
    (run_dir / "items.jsonl").write_text(
        "\n".join(json.dumps(i) for i in items) + "\n", encoding="utf-8")
    (run_dir / "attempts.jsonl").write_text("", encoding="utf-8")

    def ok(lid, item, choice, usable):
        return {"logical_request_id": lid, "item_id": item, "terminal": True,
                "status": "ok", "predictions": {"q": {
                    "type": "choice", "usable": usable, "choice": choice,
                    "probabilities": {"A": 1.0, "B": 0.0}}}}
    results = [
        ok("native:i1", "i1", "A", True),          # correct
        ok("native:i2", "i2", "B", True),          # wrong
        ok("native:i3", "i3", "A", False),         # unusable -> wrong, not scored
        # i4: no terminal result -> wrong
    ]
    (run_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(r) for r in results) + "\n", encoding="utf-8")
    doc = score_run(run_dir, stage="fixture", rotations={})
    assert doc["n_requested"] == 4
    assert doc["n_correct"] == 1
    assert doc["accuracy"] == 0.25          # HEADLINE: all requested
    assert doc["n_usable_responses"] == 2
    assert doc["usable_response_accuracy"] == 0.5   # separate diagnostic
    assert doc["by_group"]["g"]["n"] == 4
    assert doc["by_group"]["g"]["accuracy"] == 0.25
    assert doc["wilson_95"] is not None      # interval on the all-requested base


def _rotation_run(tmp_path) -> Path:
    """Tiny rotated run: remapping swaps A/B (rotated keys hold old descriptions)."""
    run_dir = tmp_path / "rot"
    run_dir.mkdir()
    items = [
        {"id": "b1:perm1", "group": "g", "condition": "perm1",
         "gold": {"q": {"value": "B"}}},   # rotated gold: canonical gold A lives under B
        {"id": "b2:perm1", "group": "g", "condition": "perm1",
         "gold": {"q": {"value": "B"}}},
        {"id": "b3:perm1", "group": "g", "condition": "perm1",
         "gold": {"q": {"value": "B"}}},
    ]
    (run_dir / "items.jsonl").write_text(
        "\n".join(json.dumps(i) for i in items) + "\n", encoding="utf-8")
    (run_dir / "attempts.jsonl").write_text("", encoding="utf-8")
    rotations = {
        f"b{i}:perm1": {"base_item": f"b{i}", "variant": 1,
                        "remapping": {"A": "B", "B": "A"}}
        for i in (1, 2, 3)
    }
    (run_dir / "rotations.json").write_text(json.dumps(rotations), encoding="utf-8")

    def row(item, choice, probs):
        return {"logical_request_id": f"perm1:{item}:perm1", "item_id": f"{item}:perm1",
                "terminal": True, "status": "ok", "predictions": {"q": {
                    "type": "choice", "usable": True, "choice": choice,
                    "probabilities": probs}}}
    results = [
        # b1: correct. Rotated key B holds canonical gold A with displayed
        # probability 1.0; the OLD scorer read key A = 0.0 -> a spurious
        # zero-gold. Canonical p(gold) is 1.0, Brier 0.
        row("b1", "B", {"A": 0.0, "B": 1.0}),
        # b2: wrong (chose rotated A = canonical B). Canonical p(gold)=0.0 is
        # a GENUINE zero-gold (old scorer read key A = 1.0 and missed it).
        row("b2", "A", {"A": 1.0, "B": 0.0}),
        # b3: wrong. Canonical p(gold) = 0.75 (asymmetric fixture value).
        row("b3", "A", {"A": 0.25, "B": 0.75}),
    ]
    (run_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(r) for r in results) + "\n", encoding="utf-8")
    return run_dir


def test_rotation_probability_keys_are_canonicalized(tmp_path):
    run_dir = _rotation_run(tmp_path)
    doc = score_run(run_dir, stage="option_rotations")
    assert doc["n_correct"] == 1
    assert doc["accuracy"] == pytest.approx(1 / 3)
    # canonical p(gold) values are 1.0 / 0.0 / 0.75:
    #   Brier  = (0 + 2 + 0.125) / 3, log loss = (0 + 34.54 + 0.288) / 3
    assert doc["proper_scores"]["n"] == 3
    assert doc["proper_scores"]["brier_mean"] == pytest.approx(2.125 / 3, abs=1e-9)
    assert doc["proper_scores"]["log_loss_mean"] == pytest.approx(
        (-math.log(1e-15) - math.log(0.75)) / 3, abs=1e-6)
    assert doc["by_rotation_variant"]["perm1"]["n"] == 3


def test_rotation_zero_gold_is_not_spurious_on_canonical_keys(tmp_path):
    run_dir = _rotation_run(tmp_path)
    doc = score_run(run_dir, stage="option_rotations")
    # exactly ONE zero-gold (b2, genuine); the un-remapped scorer reported a
    # spurious zero on b1 and MISSED the genuine one on b2
    assert doc["zero_probability_gold"] == 1
    assert doc["zero_probability_gold_candidates"] == 3


def test_preserve_original_aggregate_writes_versioned_provenance(tmp_path):
    from jev_observatory.benchmark_score import preserve_original_aggregate
    agg = tmp_path / "derived" / "score.json"
    agg.parent.mkdir(parents=True)
    agg.write_text(json.dumps({"accuracy": 0.5}) + "\n", encoding="utf-8")
    entry = preserve_original_aggregate(agg)
    preserved = agg.parent / "preserved-pre-policy-v1" / "score.json"
    assert preserved.exists()
    assert json.loads(preserved.read_text()) == {"accuracy": 0.5}
    assert entry["sha256"]
    assert "pre-policy-v1" in entry["scoring_version"]
    provenance = json.loads((agg.parent / "aggregate_provenance.json").read_text())
    assert provenance["schema"] == "aggregate-provenance-1.0.0"
    assert provenance["entries"][0]["artifact"] == "score.json"
    # idempotent: preserving again never overwrites the original bytes
    preserved.write_text("{}\n", encoding="utf-8")
    preserve_original_aggregate(agg)
    assert preserved.read_text() == "{}\n"
