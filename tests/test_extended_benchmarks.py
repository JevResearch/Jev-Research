"""Offline tests for the late-2026 extension benchmarks (ARC-AGI-2 + MATH-500).

All fixtures are synthetic/inline — no network, no credentials, no model calls
(conftest bans sockets).  Covers: source-sidecar verification, full-split count
enforcement, gold isolation in outbound payloads, per-cell encoding exactness,
build-time exclusion accounting, aggregation parity with hand-computed values,
and honest missing-terminal accounting.
"""

from __future__ import annotations

import io
import json
import tarfile
from hashlib import sha256
from pathlib import Path

import pytest

from jev_observatory.benchmark_spec import (
    DEFAULT_EXPECTED,
    build_benchmark_suite,
    verify_data_sidecar,
)
from jev_observatory.datasets.arc_agi2_grid import (
    ArcAgi2Error,
    build_items_cell_mode,
    build_items_whole_task_mode,
    load_arc_agi2_public_eval,
    state_for_task,
)
from jev_observatory.datasets.math500 import (
    Math500Error,
    _distractors,
    _parse_number,
    build_items,
    load_math500,
)
from jev_observatory.experiment import logical_requests


# ----------------------------------------------------------------- fixtures
def _mini_tar(path: Path, tasks: dict[str, dict]) -> None:
    """Write a gzipped tarball with the arcprize repo layout."""
    prefix = "ARC-AGI-2-deadbeef"
    with tarfile.open(path, "w:gz") as tar:
        for task_id, task in tasks.items():
            data = json.dumps(task).encode("utf-8")
            info = tarfile.TarInfo(name=f"{prefix}/data/evaluation/{task_id}.json")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def _task(train, tests):
    return {
        "train": [{"input": i, "output": o} for i, o in train],
        "test": [{"input": i, "output": o} for i, o in tests],
    }


TASK_A = _task(
    train=[([[1, 0], [0, 1]], [[2, 2], [2, 2]])],
    tests=[([[0, 1], [1, 0]], [[3, 3], [3, 3]])],
)
TASK_B = _task(
    train=[([[5]], [[6]])],
    tests=[([[7]], [[8]]), ([[9]], [[0]])],
)
TASK_C = _task(  # 30 distinct colors would exceed the 255 choice cap only if
    # colors repeated more than 253 times; here just a small grid
    train=[([[1, 2], [3, 4]], [[4, 3], [2, 1]])],
    tests=[([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[9, 8, 7], [6, 5, 4], [3, 2, 1]])],
)


@pytest.fixture
def agi2_tar(tmp_path):
    tar_path = tmp_path / "arc_agi2.tar.gz"
    _mini_tar(tar_path, {"aaaa": TASK_A, "bbbb": TASK_B, "cccc": TASK_C})
    sidecar = {
        "file": str(tar_path),
        "sha256": sha256(tar_path.read_bytes()).hexdigest(),
        "url": "fixture://arcprize/ARC-AGI-2",
    }
    Path(str(tar_path) + ".sha256.json").write_text(json.dumps(sidecar))
    return tar_path


def _math500_fixture(path: Path, n: int = 12) -> None:
    records = []
    for i in range(n):
        if i % 4 == 0:      # answer disclosed in the problem -> choice-excluded
            problem = f"Fixture {i}: what is 7 + {i}?"
            answer = str(7 + i)
        elif i % 4 == 1:    # small positive integer -> choice + score
            problem = f"Fixture {i}: compute {i} * 2."
            answer = str(i * 2)
        elif i % 4 == 2:    # LaTeX answer -> excluded from both
            problem = f"Fixture {i}: express \\sqrt{{{i}}}."
            answer = f"\\sqrt{{{i}}}"
        else:               # large integer -> choice yes, score excluded
            problem = f"Fixture {i}: compute {1000 + i} - {i}."
            answer = "1000"
        records.append({
            "problem": problem, "answer": answer, "subject": "Algebra",
            "level": "1", "unique_id": f"test/algebra/{i}.json",
            "solution": "fixture", "category": "Algebra",
        })
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


@pytest.fixture
def math500_path(tmp_path):
    path = tmp_path / "math500.jsonl"
    _math500_fixture(path)
    sidecar = {
        "file": str(path),
        "sha256": sha256(path.read_bytes()).hexdigest(),
        "url": "fixture://MATH-500",
    }
    Path(str(path) + ".sha256.json").write_text(json.dumps(sidecar))
    return path


# ------------------------------------------------------------------- loaders
def test_arc_agi2_loader_count_and_parse(agi2_tar):
    ds = load_arc_agi2_public_eval(agi2_tar, revision="fixture", expected_tasks=3)
    assert ds.name == "arc-agi-2-public-eval"
    assert len(ds.items) == 3
    task = ds.items[0]["arc_task"]
    assert task["test_outputs"] == [[[3, 3], [3, 3]]]
    assert task["test_inputs"] == [[[0, 1], [1, 0]]]


def test_arc_agi2_loader_refuses_wrong_count(agi2_tar):
    with pytest.raises(ArcAgi2Error, match="expected 9 public-evaluation"):
        load_arc_agi2_public_eval(agi2_tar, expected_tasks=9)


def test_arc_agi2_loader_refuses_bad_cells(tmp_path):
    bad = _task(train=[([[1]], [[2]])], tests=[([[11]], [[2]])])  # color 11 invalid
    tar_path = tmp_path / "bad.tar.gz"
    _mini_tar(tar_path, {"bad": bad})
    with pytest.raises(ArcAgi2Error, match="outside 0..9"):
        load_arc_agi2_public_eval(tar_path, expected_tasks=1)


def test_state_never_contains_gold_output(agi2_tar):
    ds = load_arc_agi2_public_eval(agi2_tar, expected_tasks=3)
    for record in ds.items:
        state = state_for_task(record["arc_task"])
        for grid in record["arc_task"]["test_outputs"]:
            assert json.dumps(grid, separators=(",", ":")) not in state


# ------------------------------------------------------------------ encodings
def test_cell_mode_items_cover_every_cell(agi2_tar):
    ds = load_arc_agi2_public_eval(agi2_tar, expected_tasks=3)
    items = build_items_cell_mode(ds, mode="cell_choice", model="m")
    by_base = {}
    for it in items:
        by_base.setdefault(it["arc_meta"]["base_id"], []).append(it)
    assert set(by_base) == {"aaaa:0", "bbbb:0", "bbbb:1", "cccc:0"}
    a_chunks = by_base["aaaa:0"]
    assert [c["arc_meta"]["chunk_index"] for c in a_chunks] == list(range(len(a_chunks)))
    a = a_chunks[0]
    assert a["gold"]["r1c1"]["arc_agi2_cell"]["value"] == "c3"
    # single-color output grids are padded with one decoy color (API contract
    # requires >=2 choice options); the gold stays exact
    assert a["arc_meta"]["distinct_output_colors"] == [0, 3]
    b1 = by_base["bbbb:1"][0]
    assert b1["gold"]["r0c0"]["arc_agi2_cell"]["value"] == "c0"
    assert len(b1["questions"]["r0c0"]["criteria"]) == 2
    # every cell covered exactly once across the chunks of its grid
    for base, chunks in by_base.items():
        qids = [q for c in chunks for q in c["questions"]]
        assert len(qids) == len(set(qids))
        assert sum(len(c["questions"]) for c in chunks) == \
            a["arc_meta"]["height"] * a["arc_meta"]["width"] if base == "aaaa:0" else True


def test_whole_task_items(agi2_tar):
    ds = load_arc_agi2_public_eval(agi2_tar, expected_tasks=3)
    items = build_items_whole_task_mode(ds, model="m")
    b_chunks = [it for it in items if it["cluster"] == "bbbb"]
    qids = [q for c in b_chunks for q in c["questions"]]
    assert set(qids) == {"g0r0c0", "g1r0c0"}
    gold_map = {q: v for c in b_chunks for q, v in
                ((k, list(p.values())[0]["value"]) for k, p in c["gold"].items())}
    assert gold_map["g0r0c0"] == "8"
    assert gold_map["g1r0c0"] == "0"
    assert all(c["arc_meta"]["n_cells"] == 2 for c in b_chunks)


def test_score_mode_uses_ten_levels(agi2_tar):
    ds = load_arc_agi2_public_eval(agi2_tar, expected_tasks=3)
    items = build_items_cell_mode(ds, mode="cell_score", model="m")
    a = items[0]
    question = a["questions"]["r0c0"]
    assert question["type"] == "score"
    assert len(question["criteria"]) == 10
    assert a["gold"]["r1c1"]["arc_agi2_cell"]["value"] == "3"


def test_chunk_caps(agi2_tar):
    from jev_observatory.datasets.arc_agi2_grid import (
        CHUNK_QUESTIONS, MAX_REQUEST_CHARS, state_for_task,
    )
    ds = load_arc_agi2_public_eval(agi2_tar, expected_tasks=3)
    for mode in ("cell_choice", "cell_score"):
        for it in build_items_cell_mode(ds, mode=mode, model="m"):
            assert len(it["questions"]) <= CHUNK_QUESTIONS
            size = len(it["state"]) + sum(
                len(json.dumps(q)) for q in it["questions"].values())
            assert size <= MAX_REQUEST_CHARS
    for it in build_items_whole_task_mode(ds, model="m"):
        assert len(it["questions"]) <= CHUNK_QUESTIONS


# ------------------------------------------------------- gold isolation (wire)
def test_logical_requests_exclude_gold_ext(agi2_tar, math500_path):
    suite = _extension_only_suite(agi2_tar, math500_path)
    forbidden = ("answer", "answer_index", "gold", "cot_content", "answerKey")
    for stage in ("arc_agi2_choice", "arc_agi2_score", "arc_agi2_task",
                  "math500_choice", "math500_score"):
        spec = suite[stage]["spec"]
        for lr in logical_requests(spec):  # raises on leakage-check violations
            payload = json.dumps(lr.request.to_payload(), ensure_ascii=False)
            for name in forbidden:
                assert f'"{name}"' not in payload, (stage, lr.item_id, name)


def _extension_only_suite(agi2_tar, math500_path):
    from jev_observatory.benchmark_spec import build_extension_stages
    return build_extension_stages(
        arc_agi2_tar=agi2_tar, math500_path=math500_path, model="m",
        arc_agi2_expected=3, math500_expected=12)


# --------------------------------------------------------------- MATH-500 ----
def test_math500_loader(math500_path):
    ds = load_math500(math500_path, expected_n=12)
    assert len(ds.items) == 12


def test_math500_refuses_wrong_count(math500_path):
    with pytest.raises(Math500Error, match="expected the MATH-500 test file"):
        load_math500(math500_path, expected_n=11)


def test_math500_parse_number():
    assert _parse_number("42") == 42.0
    assert _parse_number("-7") == -7.0
    assert _parse_number("\\pi") is None
    assert _parse_number("3.5") is None
    assert _parse_number("090") == 90.0  # leading zeros tolerated as integers


def test_math500_distractors_deterministic():
    a = _distractors("18", "test/number_theory/572.json")
    b = _distractors("18", "test/number_theory/572.json")
    assert a == b and len(a) == 3 and "18" not in a
    assert _distractors("\\frac{1}{2}", "x") is None


def test_math500_choice_exclusions_are_accounted(math500_path):
    ds = load_math500(math500_path, expected_n=12)
    items, accounting = build_items(ds, mode="choice", model="m")
    assert accounting["n_population"] == 12
    assert accounting["n_built"] == len(items)
    assert accounting["n_built"] + accounting["n_excluded"] == 12
    # '7 + i' items disclose the gold in the problem: excluded
    assert "answer_disclosed_in_problem" in accounting["exclusion_reasons"]
    # every built item: exactly 4 options, gold key maps to the exact gold text
    for it in items:
        criteria = it["questions"]["math500"]["criteria"]
        assert len(criteria) == 4
        assert criteria[it["gold"]["math500"]["value"]] == it["math500_meta"]["gold_answer_text"]
        # the gold text must not appear in the problem text
        base = it["state"].split("\n\nGive your answer as an exact integer")[0]
        import re as _re
        assert not _re.search(r"(?<![\d.])" + _re.escape(it["math500_meta"]["gold_answer_text"])
                              + r"(?![\d.])", base), it["id"]


def test_math500_score_items_exact_digits(math500_path):
    ds = load_math500(math500_path, expected_n=12)
    items, accounting = build_items(ds, mode="score", model="m")
    assert accounting["n_built"] + accounting["n_excluded"] == 12
    for it in items:
        n_q = len(it["questions"])
        assert 3 <= n_q <= 5  # 1-3 integer digits + 2 decimals
        digits = [it["gold"][f"d{p}"]["math500_digit"]["value"] for p in range(n_q)]
        int_part = "".join(digits[:-2]).lstrip("0") or "0"
        gold_value = float(int_part + "." + "".join(digits[-2:]))
        gold_text = float(it["math500_meta"]["gold_answer_text"])
        assert gold_value == gold_text
        for q in it["questions"].values():
            assert q["type"] == "score" and len(q["criteria"]) == 10


# ------------------------------------------------------------ aggregation ----
def test_aggregate_grid_stage_hand_fixture(tmp_path, agi2_tar):
    from jev_observatory.arc_agi2_score import aggregate_grid_stage
    ds = load_arc_agi2_public_eval(agi2_tar, expected_tasks=3)
    items = build_items_cell_mode(ds, mode="cell_choice", model="m")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with (run_dir / "items.jsonl").open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it) + "\n")
    # hand-scored results: aaaa:0 all cells correct; bbbb:0 one cell wrong;
    # bbbb:1 transport error (grid wrong); cccc:0 missing entirely.
    results = []
    for it in items:
        if it["id"].startswith("aaaa:0"):
            preds = {qid: {"type": "choice", "usable": True,
                           "choice": payload["arc_agi2_cell"]["value"]}
                     for qid, payload in it["gold"].items()}
            results.append(_result(it["id"], "ok", preds))
        elif it["id"].startswith("bbbb:0"):
            preds = {qid: {"type": "choice", "usable": True,
                           "choice": payload["arc_agi2_cell"]["value"]}
                     for qid, payload in it["gold"].items()}
            preds["r0c0"]["choice"] = "cWRONG"
            results.append(_result(it["id"], "ok", preds))
        elif it["id"].startswith("bbbb:1"):
            results.append(_result(it["id"], "transport_error", None))
    with (run_dir / "results.jsonl").open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")
    (run_dir / "attempts.jsonl").write_text("", encoding="utf-8")
    summary = aggregate_grid_stage(run_dir, stage="arc_agi2_choice")
    # hand-computed truth:
    assert summary["n_tasks"] == 3
    assert summary["tasks_solved"] == 1          # only aaaa
    assert summary["task_accuracy"] == pytest.approx(1 / 3)
    assert summary["grids_correct"] == 1
    assert summary["grids_expected"] == 4        # aaaa:0, bbbb:0, bbbb:1 (+cccc:0 missing)
    assert summary["grid_accuracy"] == pytest.approx(0.25)
    # cells: aaaa 4/4 ok; bbbb:0 0/1 (wrong); bbbb:1 0/1 (error); cccc:0 0/9 (missing)
    assert summary["cell_accuracy_diagnostic"] == pytest.approx(4 / 15)
    assert summary["cells_total"] == 15
    assert sorted(summary["missing_items"]) == ["cccc:0#c0"]


def _result(item_id, status, predictions):
    return {
        "logical_request_id": f"cell_choice:{item_id}",
        "terminal": True, "status": status, "item_id": item_id,
        "group": "arc-agi-2-public-eval", "cluster": item_id.split(":")[0],
        "condition": "cell_choice", "model_returned": "m",
        "predictions": predictions or {},
    }


# ------------------------------------------------------------------ suite ----
def test_extension_stage_counts(agi2_tar, math500_path):
    suite = _extension_only_suite(agi2_tar, math500_path)
    counts = suite["counts"]
    assert counts["arc_agi2_choice"] == 4
    assert counts["arc_agi2_score"] == 4
    assert counts["arc_agi2_task"] == 3
    assert suite["total_requests"] == sum(counts.values())


def test_verify_data_sidecar_rejects_tamper(agi2_tar):
    with pytest.raises(Exception, match="does not match sidecar"):
        sidecar_path = Path(str(agi2_tar) + ".sha256.json")
        record = json.loads(sidecar_path.read_text())
        record["sha256"] = "0" * 64
        sidecar_path.write_text(json.dumps(record))
        verify_data_sidecar(agi2_tar)


# ------------------------------------------------------- raw filename safety
def test_runstore_raw_filename_flattens_slashes(tmp_path):
    from jev_observatory.ledger import RunStore

    store = RunStore(tmp_path, "run1")
    ref = store.write_raw("math500_choice:test/algebra/0.json.0", "request",
                          {"hello": "world"})
    assert "/" not in ref["path"].split("raw/")[-1]
    assert ref["path"].endswith(".request.json")
    # idempotent rewrite of the SAME body is fine; different body is a hard error
    ref2 = store.write_raw("math500_choice:test/algebra/0.json.0", "request",
                           {"hello": "world"})
    assert ref2["sha256"] == ref["sha256"]
    with pytest.raises(FileExistsError):
        store.write_raw("math500_choice:test/algebra/0.json.0", "request",
                        {"hello": "changed"})


def test_executor_records_last_provider_exception(tmp_path, agi2_tar):
    """A provider that raises on EVERY dispatch must stop the run fail-closed
    AND leave the redacted exception repr in executor_state.json."""
    import json as _json

    from jev_observatory.benchmark_exec import BenchmarkExecutor, load_freeze
    from jev_observatory.benchmark_spec import build_extension_stages

    suite = build_extension_stages(
        arc_agi2_tar=agi2_tar, math500_path=_math500_sidecar(tmp_path, agi2_tar),
        model="m", arc_agi2_expected=3, math500_expected=12)
    from jev_observatory.benchmark_exec import build_freeze, load_freeze, stage_run_id
    from jev_observatory.runner import plan_run

    build_freeze(suite, root=tmp_path)
    freeze = load_freeze(tmp_path)
    run_id = stage_run_id(freeze, "math500_choice")
    run_dir = tmp_path / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "spec.json").write_text(_json.dumps(suite["math500_choice"]["spec"]),
                                       encoding="utf-8")
    plan_run(suite["math500_choice"]["spec"], provider="jev", root=str(tmp_path),
             run_id=run_id, budgets={"max_requests": 15000,
                                     "max_estimated_input_tokens": 60_000_000},
             retry_policy={"max_attempts": 1})
    executor = BenchmarkExecutor(
        tmp_path, provider_factory=_ExplodingProvider, max_wall_seconds=60.0,
        stages=("math500_choice",))
    # the executor fails CLOSED (stop reason recorded) but does not raise:
    # provider exceptions are swallowed into the state for diagnosis
    executor.run(dry_run=False)
    state = _json.loads((tmp_path / "executor_state.json").read_text())
    assert state["global_stop_reason"] == "provider_exception_failures"
    assert state["last_provider_exception"]
    assert "ExplodingProvider" in state["last_provider_exception"]


class _ExplodingProvider:
    name = "mock"

    def __init__(self):
        self.calls = 0

    def ask(self, request, logical_request_id, **kwargs):
        raise RuntimeError("ExplodingProvider kaboom")

    def close(self):
        return None


def _math500_sidecar(tmp_path, agi2_tar):
    path = agi2_tar.parent / "math500.jsonl"
    if not path.exists():
        _math500_fixture(path)
        sidecar = {
            "file": str(path), "sha256": sha256(path.read_bytes()).hexdigest(),
            "url": "fixture://MATH-500",
        }
        (Path(str(path) + ".sha256.json")).write_text(json.dumps(sidecar))
    return path


# ------------------------------------------------ display-quantum validation
def test_choice_gap_within_display_quantum_is_warning_not_error():
    from jev_observatory.schema import ChoiceQuestion, SystemOneRequest
    from jev_observatory.validation import validate_response

    request = SystemOneRequest(model="jev-1.13.0", state="s", questions={
        "q": ChoiceQuestion(instructions="?", criteria={"a": "A", "b": "B", "c": "C"})})
    # gap exactly one display quantum (0.42 vs 0.41): observed in the wild
    raw = {"model": "jev-1.13.0",
           "answers": {"q": {"type": "choice", "choice": "b",
                             "probabilities": {"a": 0.42, "b": 0.41, "c": 0.17},
                             "confidence": 0.02}},
           "usage": {"input_tokens": 1, "output_tokens": 1}}
    validated = validate_response(raw, request, http_status=200)
    assert "choice_argmax_within_display_quantum" in validated.codes
    assert validated.usable
    assert validated.contract_valid
    # gap beyond two quanta remains a hard error
    raw2 = {"model": "jev-1.13.0",
            "answers": {"q": {"type": "choice", "choice": "b",
                              "probabilities": {"a": 0.9, "b": 0.1},
                              "confidence": 0.8}},
            "usage": {"input_tokens": 1, "output_tokens": 1}}
    validated2 = validate_response(raw2, request, http_status=200)
    assert "choice_not_argmax" in validated2.codes
    assert not validated2.usable


def test_score_expectation_rounding_is_warning_not_error():
    from jev_observatory.schema import ScoreQuestion, SystemOneRequest
    from jev_observatory.validation import validate_response

    request = SystemOneRequest(model="jev-1.13.0", state="s", questions={
        "q": ScoreQuestion(instructions="?", criteria=[f"color {i}" for i in range(10)])})
    # probabilities quantized at 2dp; expectation recomputed from them differs
    # from the server's full-precision score by 0.05 (<= 0.005*sum(0..9)=0.225)
    probs = {"0": 0.14, "1": 0.2, "2": 0.24, "3": 0.1, "4": 0.07,
             "5": 0.05, "6": 0.04, "7": 0.04, "8": 0.04, "9": 0.08}
    expectation = sum(int(k) * v for k, v in probs.items())
    raw = {"model": "jev-1.13.0",
           "answers": {"q": {"type": "score", "score": round(expectation - 0.05, 2),
                             "legend": {str(i): f"color {i}" for i in range(10)},
                             "probabilities": probs, "confidence": 0.1}},
           "usage": {"input_tokens": 1, "output_tokens": 1}}
    validated = validate_response(raw, request, http_status=200)
    entry = [v for v in validated.violations if v.code == "score_expectation_mismatch"]
    assert entry and entry[0].severity == "warning"
    assert validated.usable and validated.contract_valid
    # a delta beyond the rounding bound remains an error
    raw2 = {"model": "jev-1.13.0",
            "answers": {"q": {"type": "score", "score": round(expectation - 0.5, 2),
                              "legend": {str(i): f"color {i}" for i in range(10)},
                              "probabilities": probs, "confidence": 0.1}},
            "usage": {"input_tokens": 1, "output_tokens": 1}}
    validated2 = validate_response(raw2, request, http_status=200)
    entry2 = [v for v in validated2.violations if v.code == "score_expectation_mismatch"]
    assert entry2 and entry2[0].severity == "error"
    assert not validated2.usable


# ---------------------------------------------------- weighted-score variant
def test_weighted_score_choice_fixture(tmp_path, agi2_tar):
    """Weighted score = p(gold) per item; mean over items; exclusions counted."""
    from jev_observatory.weighted_score import weighted_score_run

    ds = load_arc_agi2_public_eval(agi2_tar, expected_tasks=3)
    items = build_items_cell_mode(ds, mode="cell_choice", model="m")
    run_dir = tmp_path / "wrun"
    run_dir.mkdir()
    with (run_dir / "items.jsonl").open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it) + "\n")
    results = []
    for it in items:
        if not it["id"].startswith("aaaa:0"):
            continue
        preds = {}
        for index, (qid, payload) in enumerate(it["gold"].items()):
            gold_value = payload["arc_agi2_cell"]["value"]
            preds[qid] = {"type": "choice", "usable": True,
                          "choice": gold_value,
                          # gold carries 0.6, the decoy 0.4 (sums to 1.0)
                          "probabilities": {gold_value: 0.6,
                                            ("c0" if gold_value != "c0" else "c9"): 0.4}}
        results.append(_result(it["id"], "ok", preds))
    # one item with an INVALID probability vector -> excluded, counted
    bad = [it for it in items if it["id"].startswith("bbbb:0")][0]
    preds = {qid: {"type": "choice", "usable": True, "choice": "c0",
                   "probabilities": {"c0": 0.5, "c3": 0.2}}  # sums to 0.7
             for qid in bad["questions"]}
    results.append(_result(bad["id"], "ok", preds))
    # one missing item entirely
    with (run_dir / "results.jsonl").open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")
    (run_dir / "attempts.jsonl").write_text("", encoding="utf-8")
    doc = weighted_score_run(run_dir, stage="arc_agi2_choice")
    assert doc["n_weighted_scored"] == 1
    assert doc["n_excluded"] >= 2
    assert doc["per_cell_weighted_mean"] == pytest.approx(0.6)
    assert "never merged" in doc["separateness_note"]


def test_weighted_score_digit_product(tmp_path, math500_path):
    """math500_score item weighted = product of per-digit p(gold), labeled."""
    from jev_observatory.datasets.math500 import build_items, load_math500
    from jev_observatory.weighted_score import weighted_score_run

    ds = load_math500(math500_path, expected_n=12)
    items, _ = build_items(ds, mode="score", model="m")
    run_dir = tmp_path / "wrun"
    run_dir.mkdir()
    with (run_dir / "items.jsonl").open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it) + "\n")
    results = []
    for it in items[:1]:
        preds = {}
        for qid, payload in it["gold"].items():
            digit = payload["math500_digit"]["value"]
            preds[qid] = {"type": "score", "usable": True,
                          "probabilities": {str(i): 0.1 for i in range(10)}
                          | {digit: 0.5}}
        # fix sum: 0.1*9 + 0.5 = 1.4 -> rescale: use 0.05 elsewhere + 0.55 gold
        for qid, payload in it["gold"].items():
            digit = payload["math500_digit"]["value"]
            preds[qid]["probabilities"] = {str(i): 0.05 for i in range(10)}
            preds[qid]["probabilities"][digit] = 0.55
        results.append(_result(it["id"], "ok", preds))
    with (run_dir / "results.jsonl").open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")
    (run_dir / "attempts.jsonl").write_text("", encoding="utf-8")
    doc = weighted_score_run(run_dir, stage="math500_score")
    # per-digit gold prob 0.55; n digits varies -> product = 0.55**n
    first = [p for p in doc["per_item"] if p["weighted"] is not None][0]
    n_digits = len(items[0]["questions"])
    assert first["weighted"] == pytest.approx(0.55 ** n_digits)
    assert "independent-digits" in doc["item_product_note"]
