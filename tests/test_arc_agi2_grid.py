"""ARC-AGI-2 cell-mode adapter regressions: test-grid identity, oracle protocol.

Offline only (conftest bans sockets).  Two different test grids of the same
task share the SAME state (all test inputs) — with row/column-only question
instructions and the same palette/shape, cells of different grids serialized
to byte-identical requests carrying different gold (7 duplicate-payload pairs
in the recorded choice run).  These tests pin the fix: every cell question
states its test-grid identity, outbound payloads are distinct per grid, and
the oracle-derived dimensions/palette are an explicitly recorded diagnostic
protocol, not pretend official evaluation.
"""

from __future__ import annotations

import io
import json
import tarfile
from hashlib import sha256
from pathlib import Path

import pytest

from jev_observatory.datasets.arc_agi2_grid import (
    ORACLE_DIAGNOSTIC_PROTOCOL,
    build_items_cell_mode,
    build_spec,
    load_arc_agi2_public_eval,
)


def _mini_tar(path: Path, tasks: dict[str, dict]) -> None:
    prefix = "ARC-AGI-2-deadbeef"
    with tarfile.open(path, "w:gz") as tar:
        for task_id, task in tasks.items():
            data = json.dumps(task).encode("utf-8")
            info = tarfile.TarInfo(name=f"{prefix}/data/evaluation/{task_id}.json")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


# one task, two test inputs: SAME input shape, gold grids share shape AND
# palette ({0,3}) but differ in layout -> same palette/shape, different gold
SAME_PALETTE_TASK = {
    "train": [{"input": [[1]], "output": [[3]]}],
    "test": [
        {"input": [[1, 1]], "output": [[0, 3]]},
        {"input": [[1, 1]], "output": [[3, 0]]},
    ],
}


@pytest.fixture
def same_palette_dataset(tmp_path):
    tar_path = tmp_path / "arc_agi2.tar.gz"
    _mini_tar(tar_path, {"tttt": SAME_PALETTE_TASK})
    sidecar = {"file": str(tar_path),
               "sha256": sha256(tar_path.read_bytes()).hexdigest(),
               "url": "fixture://arcprize/ARC-AGI-2"}
    Path(str(tar_path) + ".sha256.json").write_text(json.dumps(sidecar))
    return load_arc_agi2_public_eval(tar_path, revision="fixture", expected_tasks=1)


def _payload_key(item: dict) -> str:
    """The outbound request payload: state + questions (never gold)."""
    return json.dumps({"state": item["state"], "questions": item["questions"]},
                      sort_keys=True, ensure_ascii=False)


def _duplicate_payload_pairs(items: list[dict]) -> list[tuple[str, str]]:
    by_payload: dict[str, list[dict]] = {}
    for item in items:
        by_payload.setdefault(_payload_key(item), []).append(item)
    pairs = []
    for group in by_payload.values():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                pairs.append((group[i]["id"], group[j]["id"]))
    return pairs


@pytest.mark.parametrize("mode", ["cell_choice", "cell_score"])
def test_distinct_test_grids_never_share_outbound_payloads(same_palette_dataset, mode):
    items = build_items_cell_mode(same_palette_dataset, mode=mode, model="m")
    assert _duplicate_payload_pairs(items) == []
    # same palette AND shape is the interesting case: the fixtures differ ONLY
    # in which grid/cell holds which color
    golds = {it["arc_meta"]["grid_index"]: {
        q: v["arc_agi2_cell"]["value"] for q, v in it["gold"].items()}
        for it in items}
    assert golds[0] != golds[1]
    assert {it["arc_meta"]["height"] for it in items} == {1}
    assert {tuple(it["arc_meta"]["distinct_output_colors"]) for it in items} == {(0, 3)}


@pytest.mark.parametrize("mode", ["cell_choice", "cell_score"])
def test_cell_questions_state_test_grid_identity(same_palette_dataset, mode):
    items = build_items_cell_mode(same_palette_dataset, mode=mode, model="m")
    for item in items:
        grid_identity = item["arc_meta"]["grid_identity"]
        for question in item["questions"].values():
            assert grid_identity in question["instructions"]
            assert "row 0" in question["instructions"]
        # explicit identity: grid number and grid count both present
        assert f"grid {item['arc_meta']['grid_index'] + 1} of 2" in \
            next(iter(item["questions"].values()))["instructions"]


def test_oracle_palette_and_dims_are_recorded_diagnostic_protocol(same_palette_dataset):
    items = build_items_cell_mode(same_palette_dataset, mode="cell_choice", model="m")
    for item in items:
        assert item["arc_meta"]["diagnostic_protocol"] == ORACLE_DIAGNOSTIC_PROTOCOL
        assert "NOT official ARC-AGI-2" in ORACLE_DIAGNOSTIC_PROTOCOL["status"]
    spec = build_spec(same_palette_dataset, mode="cell_choice")
    assert spec["protocol"]["oracle_diagnostics"] == ORACLE_DIAGNOSTIC_PROTOCOL
    assert "grid i of n" in spec["protocol"]["grid_identity"]
    # the oracle-derived palette is visible in the item metadata, not hidden
    assert all(it["arc_meta"]["distinct_output_colors"] == [0, 3] for it in items)


def test_duplicate_payload_detection_catches_row_column_only_regressions(same_palette_dataset):
    """The detector that found the 7 recorded pairs must fire on the OLD shape."""
    items = build_items_cell_mode(same_palette_dataset, mode="cell_choice", model="m")
    for item in items:                       # strip grid identity = OLD adapter
        for question in item["questions"].values():
            question["instructions"] = question["instructions"].replace(
                f"test output grid {item['arc_meta']['grid_index'] + 1} of 2 "
                f"(the output of test input {item['arc_meta']['grid_index'] + 1}), ",
                "")
    pairs = _duplicate_payload_pairs(items)
    assert pairs != []        # identical requests with different gold are detectable