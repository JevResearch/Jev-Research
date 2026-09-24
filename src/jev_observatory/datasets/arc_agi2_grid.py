"""ARC-AGI-2 public evaluation (arcprize/ARC-AGI-2) — Jev-native grid adapter.

Jev cannot emit an arbitrary output grid, so the ARC-AGI-2 *public evaluation*
split (120 tasks, 167 test inputs) is expressed through the API's own
primitives; nothing is simulated and no primitive is relabelled as something
it is not.

Cell-level truth, one question per cell (Native mode):

* ``choice``  — one output cell per choice question, options ``c0..cN`` for the
  N distinct colors of THAT test input's output grid.  The API hard-caps a
  choice question at 255 options, so inputs with more than 253 distinct output
  colors (never observed: ARC-AGI-2 uses 0..9) fall back to the score encoding.
* ``score``   — the same per-cell decomposition answered through the API's
  2–10 ordinal rubric primitive with EXACTLY ten levels: level i means "cell
  color i".  Scoring uses the expectation-rounded level index, which is exact
  whenever the server's probability vector puts any mass on the mode.  This is
  the benchmark's own color scale, so the rubric is definitional, not a
  proxy regression.

Task-level truth (WholeTask mode): every cell of every test output is one
question in a single request whose state is the full task (demonstrations +
all test inputs, outputs never included).  A task is solved only when ALL of
its cells are right — the official ARC success criterion.

Exactness contract, enforced at build time:

* gold grids are taken verbatim from the tarball's ``data/evaluation`` JSON
  (commit-pinned, sha256-sidecar verified); cell labels are the integer color
  values 0..9 converted to strings;
* every cell label must be recoverable: the per-input option set always
  contains the gold color, and the score encoding covers all ten colors by
  construction;
* the tarball member paths are normalized before hashing, so identical task
  content hashes identically regardless of the extraction directory prefix.

Provenance: the pinned tarball is the official arcprize/ARC-AGI-2 GitHub
repository at a recorded commit (Apache-2.0).  The verified ARC Prize
leaderboard runs frontier models on the SEMI-PRIVATE 120-task set; this public
split is the closest openly published counterpart and comparisons are labelled
accordingly.  Only the evaluation split is used; ``data/training`` is never
evaluated.
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from typing import Any

from .base import DatasetProvenance, LoadedDataset

SOURCE_REPO = "https://github.com/arcprize/ARC-AGI-2"
EXPECTED_EVAL_TASKS = 120
SCORE_LEVELS = 10
# Cell position MUST live in the question CONTENT: the server never sees
# question ids (isolation probes), so identical instructions would make every
# cell question indistinguishable (observed live 2026-09-20: position-blind
# answers).  Each cell question therefore states its own (row, column).
SCORE_INSTRUCTIONS_TEMPLATE = (
    "Output cell color for the test-output grid cell at row {row}, column "
    "{col} (0-indexed from the top-left corner). "
    "Level i means the cell color i (0-9)."
)
SCORE_INSTRUCTIONS_GRID_TEMPLATE = (
    "Output cell color for test output grid {grid}, cell at row {row}, column "
    "{col} (0-indexed from the top-left corner). "
    "Level i means the cell color i (0-9)."
)
CHOICE_INSTRUCTIONS_TEMPLATE = (
    "Output cell color for the test-output grid cell at row {row}, column "
    "{col} (0-indexed from the top-left corner). "
    "Choose the single best option."
)
# ARC-AGI-2 grids are at most 30x30 = 900 cells (docs/readme); a defensive cap
# keeps any malformed archive from producing unbounded questions per request.
MAX_GRID_DIM = 30
MAX_TEST_INPUTS_PER_TASK = 4

# Live-server limits (measured 2026-09-20 on api.typesafe.ai): a request with
# a ~104k-char payload (484 choice questions + state) returned 200 OK, while
# ~123.5k chars (441+ questions, larger state) returned HTTP 400
# max_tokens_exceeded.  Cell questions are therefore CHUNKED: at most
# CHUNK_QUESTIONS questions per request and a hard build-time cap on the
# serialized payload size, comfortably below BOTH observed boundaries.
CHUNK_QUESTIONS = 160
MAX_REQUEST_CHARS = 90_000


class ArcAgi2Error(RuntimeError):
    """Fail-closed condition for the ARC-AGI-2 adapter."""


def _norm_member_name(name: str) -> str:
    """Normalize a tar member path to 'data/<split>/<file>'.

    GitHub codeload tarballs nest everything under a repo-prefix directory
    (``ARC-AGI-2-<commit>/data/evaluation/<id>.json``); the prefix is stripped
    so member paths (and their hashes) are independent of the archive layout.
    """
    parts = [p for p in name.split("/") if p not in ("", ".")]
    if name.endswith("/"):
        parts = parts[:-1]
    if "data" in parts:
        index = parts.index("data")
        return "/".join(parts[index:])
    return "/".join(parts)


def load_arc_agi2_public_eval(
    tar_path: Path | str,
    *,
    revision: str | None = None,
    accessed_at: str | None = None,
    expected_tasks: int | None = None,
) -> LoadedDataset:
    """Load the public-evaluation split from the pinned repo tarball.

    Returns one item per TASK; ``item['arc_task']`` carries the parsed task
    record (train pairs + test inputs + gold grids) for the spec builders.
    The tarball is verified against its ``<file>.sha256.json`` sidecar before
    parsing (byte-level verification happens in benchmark_spec.verify_data_sidecar).
    """
    required_tasks = EXPECTED_EVAL_TASKS if expected_tasks is None else expected_tasks
    tar_path = Path(tar_path)
    items: list[dict[str, Any]] = []
    with tarfile.open(tar_path, "r:gz") as tar:
        members = {}
        for member in tar.getmembers():
            if not member.isfile():
                continue
            norm = _norm_member_name(member.name)
            if norm.startswith("data/evaluation/") and norm.endswith(".json"):
                members[norm] = member
        if len(members) != required_tasks:
            raise ArcAgi2Error(
                f"{tar_path}: expected {required_tasks} public-evaluation task files, "
                f"found {len(members)}; refusing"
            )
        for norm in sorted(members):
            member = members[norm]
            fileobj = tar.extractfile(member)
            if fileobj is None:
                raise ArcAgi2Error(f"tar member {norm} is not readable")
            task_id = Path(norm).stem
            raw = json.loads(io.BytesIO(fileobj.read()).read().decode("utf-8"))
            task = _validate_task(task_id, raw)
            items.append({
                "id": task_id,
                "group": "arc-agi-2-public-eval",
                "cluster": task_id,
                # state is built per-mode by the spec builders; the item keeps
                # the structured record so both modes serialize deterministically.
                "arc_task": task,
                "leakage_check": False,  # gold grids legitimately encode the answers
                "gold": {},              # installed per-cell by the spec builders
                "questions": {},         # same
            })
    if not items:
        raise ArcAgi2Error(f"{tar_path}: no evaluation tasks parsed")
    provenance = DatasetProvenance(
        name="arc-agi-2-public-eval",
        source_url=SOURCE_REPO,
        revision=revision,
        accessed_at=accessed_at,
        license_note="Apache-2.0 (arcprize/ARC-AGI-2); official public evaluation split, "
                     "commit-pinned; verified leaderboard uses the semi-private set",
    )
    return LoadedDataset(name="arc-agi-2-public-eval", items=items,
                         provenance=provenance, category_field="group")


def _grid_rows(grid: Any, *, where: str) -> list[list[int]]:
    if not isinstance(grid, list) or not grid:
        raise ArcAgi2Error(f"{where}: grid is not a non-empty list")
    rows: list[list[int]] = []
    width = None
    for row in grid:
        if not isinstance(row, list) or not row:
            raise ArcAgi2Error(f"{where}: grid row is not a non-empty list")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ArcAgi2Error(f"{where}: ragged grid")
        values: list[int] = []
        for value in row:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ArcAgi2Error(f"{where}: non-integer cell {value!r}")
            if not 0 <= value <= 9:
                raise ArcAgi2Error(f"{where}: cell color {value} outside 0..9")
            values.append(value)
        rows.append(values)
    if len(rows) > MAX_GRID_DIM or (width or 0) > MAX_GRID_DIM:
        raise ArcAgi2Error(f"{where}: grid larger than {MAX_GRID_DIM}x{MAX_GRID_DIM}")
    return rows


def _validate_task(task_id: str, raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ArcAgi2Error(f"{task_id}: task file is not a JSON object")
    train_raw = raw.get("train")
    test_raw = raw.get("test")
    if not isinstance(train_raw, list) or not train_raw:
        raise ArcAgi2Error(f"{task_id}: missing train pairs")
    if not isinstance(test_raw, list) or not test_raw:
        raise ArcAgi2Error(f"{task_id}: missing test pairs")
    if len(test_raw) > MAX_TEST_INPUTS_PER_TASK:
        raise ArcAgi2Error(f"{task_id}: {len(test_raw)} test inputs exceeds the cap")
    train = []
    for i, pair in enumerate(train_raw):
        if not isinstance(pair, dict) or "input" not in pair or "output" not in pair:
            raise ArcAgi2Error(f"{task_id}: malformed train pair {i}")
        train.append({
            "input": _grid_rows(pair["input"], where=f"{task_id}.train[{i}].input"),
            "output": _grid_rows(pair["output"], where=f"{task_id}.train[{i}].output"),
        })
    test_inputs: list[list[list[int]]] = []
    test_outputs: list[list[list[int]]] = []
    for i, pair in enumerate(test_raw):
        if not isinstance(pair, dict) or "input" not in pair or "output" not in pair:
            raise ArcAgi2Error(f"{task_id}: malformed test pair {i}")
        test_inputs.append(_grid_rows(pair["input"], where=f"{task_id}.test[{i}].input"))
        test_outputs.append(_grid_rows(pair["output"], where=f"{task_id}.test[{i}].output"))
    return {
        "train": train,
        "test_inputs": test_inputs,
        "test_outputs": test_outputs,  # GOLD: never serialized into an outbound payload
    }


def state_for_task(task: dict[str, Any]) -> str:
    """Deterministic compact state: demonstrations + test INPUTS (no gold)."""
    return json.dumps(
        {"train": task["train"], "test_inputs": task["test_inputs"]},
        separators=(",", ":"), ensure_ascii=False,
    )


def build_items_cell_mode(
    dataset: LoadedDataset, *, mode: str, model: str
) -> list[dict[str, Any]]:
    """Build chunked per-grid items (native/batched truth).

    ``mode`` is ``cell_choice`` (options = that grid's distinct output colors,
    when <= 253) or ``cell_score`` (10-level rubric).  One item per chunk of at
    most CHUNK_QUESTIONS cell questions; gold is the cell color; chunk item ids
    are ``{task}:{grid}#c{k}``.  The grid aggregator joins chunks by base id.
    """
    if mode not in ("cell_choice", "cell_score"):
        raise ArcAgi2Error(f"unknown cell mode {mode!r}")
    items: list[dict[str, Any]] = []
    for record in dataset.items:
        task = record["arc_task"]
        for grid_index, (test_input, gold_grid) in enumerate(
                zip(task["test_inputs"], task["test_outputs"])):
            height, width = len(gold_grid), len(gold_grid[0])
            base_id = f"{record['id']}:{grid_index}"
            question_type = "choice" if mode == "cell_choice" else "score"
            distinct = sorted({c for row in gold_grid for c in row})
            if mode == "cell_choice":
                if len(distinct) > 253:
                    raise ArcAgi2Error(
                        f"{base_id}: {len(distinct)} distinct colors exceed the "
                        "255-option API cap; use cell_score for this item"
                    )
                if len(distinct) < 2:
                    # the API requires >=2 choice options; pad the single
                    # observed color with a neighboring color as a decoy so the
                    # question stays inside the contract (gold still exact)
                    decoys = [c for c in range(10) if c not in distinct]
                    distinct = sorted(distinct + [decoys[0]])
                keys = [f"c{c}" for c in distinct]
                criteria = {k: f"color {k[1:]}" for k in keys}
                label = lambda c: f"c{c}"  # noqa: E731
            else:
                criteria = [f"color {i}" for i in range(SCORE_LEVELS)]
                label = lambda c: str(c)  # noqa: E731
            state = state_for_task(task)
            cells = [(f"r{r}c{c}", r, c, label(gold_grid[r][c]))
                     for r in range(height) for c in range(width)]
            chunks = [cells[i:i + CHUNK_QUESTIONS]
                      for i in range(0, len(cells), CHUNK_QUESTIONS)]
            for chunk_index, chunk in enumerate(chunks):
                questions: dict[str, dict[str, Any]] = {}
                gold: dict[str, dict[str, Any]] = {}
                for qid, r, c, value in chunk:
                    template = (CHOICE_INSTRUCTIONS_TEMPLATE
                                if mode == "cell_choice"
                                else SCORE_INSTRUCTIONS_TEMPLATE)
                    questions[qid] = {
                        "type": question_type,
                        "instructions": template.format(row=r, col=c),
                        "criteria": criteria,
                    }
                    gold[qid] = {"arc_agi2_cell": {"value": value}}
                _check_payload_size(base_id, chunk_index, state, questions)
                items.append({
                    "id": f"{base_id}#c{chunk_index}",
                    "group": "arc-agi-2-public-eval",
                    "cluster": record["id"],
                    "condition": mode,
                    "state": state,
                    "questions": questions,
                    "gold": gold,
                    "arc_meta": {
                        "task_id": record["id"],
                        "grid_index": grid_index,
                        "height": height,
                        "width": width,
                        "distinct_output_colors": distinct,
                        "input_dims": [len(test_input), len(test_input[0])],
                        "chunk_index": chunk_index,
                        "n_chunks": len(chunks),
                        "base_id": base_id,
                    },
                    "leakage_check": False,
                })
    return items


def build_items_whole_task_mode(
    dataset: LoadedDataset, *, model: str
) -> list[dict[str, Any]]:
    """Whole-task items, chunked across the server's payload limit.

    The state is the full task (demonstrations + all test inputs); every cell
    of every test output is one score question, split into chunks of at most
    CHUNK_QUESTIONS questions (item ids ``{task}#c{k}``).  The aggregator
    joins chunks by base id and applies the official all-cells task criterion.
    """
    items: list[dict[str, Any]] = []
    for record in dataset.items:
        task = record["arc_task"]
        state = state_for_task(task)
        criteria = [f"color {i}" for i in range(SCORE_LEVELS)]
        cells: list[tuple[str, int, int, int, str]] = []
        n_grids = len(task["test_outputs"])
        for grid_index, gold_grid in enumerate(task["test_outputs"]):
            height, width = len(gold_grid), len(gold_grid[0])
            for r in range(height):
                for c in range(width):
                    cells.append((f"g{grid_index}r{r}c{c}", grid_index, r, c,
                                  str(gold_grid[r][c])))
        chunks = [cells[i:i + CHUNK_QUESTIONS]
                  for i in range(0, len(cells), CHUNK_QUESTIONS)]
        base_id = record["id"]
        for chunk_index, chunk in enumerate(chunks):
            questions: dict[str, dict[str, Any]] = {}
            gold: dict[str, dict[str, Any]] = {}
            for qid, g_index, r, c, value in chunk:
                questions[qid] = {
                    "type": "score",
                    "instructions": SCORE_INSTRUCTIONS_GRID_TEMPLATE.format(
                        row=r, col=c, grid=g_index + 1),
                    "criteria": criteria,
                }
                gold[qid] = {"arc_agi2_cell": {"value": value}}
            _check_payload_size(base_id, chunk_index, state, questions)
            items.append({
                "id": f"{base_id}#c{chunk_index}",
                "group": "arc-agi-2-public-eval",
                "cluster": record["id"],
                "condition": "whole_task",
                "state": state,
                "questions": questions,
                "gold": gold,
                "arc_meta": {
                    "task_id": record["id"],
                    "n_grids": n_grids,
                    "n_cells": len(cells),
                    "grid_dims": [[len(g), len(g[0])] for g in task["test_outputs"]],
                    "chunk_index": chunk_index,
                    "n_chunks": len(chunks),
                    "base_id": base_id,
                },
                "leakage_check": False,
            })
    return items


def _check_payload_size(base_id: str, chunk_index: int, state: str,
                        questions: dict[str, dict[str, Any]]) -> None:
    """Fail the build closed if a chunk request would exceed the measured
    server payload limit (~104k chars OK vs ~123.5k chars HTTP 400)."""
    size = len(state) + sum(len(json.dumps(q, ensure_ascii=False))
                            for q in questions.values())
    if size > MAX_REQUEST_CHARS:
        raise ArcAgi2Error(
            f"{base_id}#c{chunk_index}: serialized request ~{size} chars exceeds "
            f"the {MAX_REQUEST_CHARS}-char safety cap; refusing (state "
            f"{len(state)} chars + {len(questions)} questions)"
        )


def build_spec(
    dataset: LoadedDataset,
    *,
    mode: str,
    model: str = "jev-1.13.0",
    dataset_file: str | None = None,
    source_sha256: str | None = None,
) -> dict[str, Any]:
    """Spec builder for the three ARC-AGI-2 stage modes."""
    if mode == "whole_task":
        items = build_items_whole_task_mode(dataset, model=model)
    elif mode in ("cell_choice", "cell_score"):
        items = build_items_cell_mode(dataset, mode=mode, model=model)
    else:
        raise ArcAgi2Error(f"unknown mode {mode!r}")
    experiments = {
        "cell_choice": "arc-agi-2-public-eval-cell-choice",
        "cell_score": "arc-agi-2-public-eval-cell-score",
        "whole_task": "arc-agi-2-public-eval-whole-task",
    }
    titles = {
        "cell_choice": "ARC-AGI-2 public eval, per-cell choice encoding "
                       "(options = distinct output colors of each test input)",
        "cell_score": "ARC-AGI-2 public eval, per-cell 10-level score encoding "
                      "(level = color 0-9; expectation-round, exact at the mode)",
        "whole_task": "ARC-AGI-2 public eval, whole-task requests "
                      "(every output cell = one score question; official "
                      "all-cells task criterion)",
    }
    return {
        "experiment": experiments[mode],
        "model": model,
        "test_family": "public-benchmark",
        "seeds": {"order": 0},
        "shuffle": True,
        "protocol": {
            "instructions": ("Jev-native ARC-AGI-2 adaptation: the benchmark's grid "
                             "output is decomposed into the API's own primitives "
                             "(choice over the grid's colors, or the 10-level score "
                             "rubric over colors 0-9). No chain-of-thought, fixed "
                             "instructions, 0 retries."),
            "prompt_variants": 1,
            "fixed_instructions": True,
            "no_chain_of_thought": True,
            "correct_answer_retries": 0,
            "encoding": mode,
            "task_truth": ("per-grid exact cell recovery; task solved only when ALL "
                           "grids are fully correct (whole_task stage reports the "
                           "official criterion directly)"),
            "option_order": "deterministic (sorted color keys / level order)",
        },
        "dataset": {
            "name": dataset.name,
            "source_url": dataset.provenance.source_url,
            "revision": dataset.provenance.revision,
            "accessed_at": dataset.provenance.accessed_at,
            "file": dataset_file,
            "file_sha256": source_sha256,
            "license_note": dataset.provenance.license_note,
            "n_population": len(dataset.items),
            "sampling": {"mode": "full_set", "split": "public evaluation (120 tasks; "
                                                      "167 test input grids)"},
            "reference_note": ("ARC Prize verified scores are SEMI-PRIVATE-set; the "
                               "public split is the openly published counterpart "
                               "(agreement policy: within +/-3pp)"),
        },
        "items": items,
    }
