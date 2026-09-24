"""ARC-AGI-2 grid aggregation: chunked per-cell predictions -> grid/task truth.

Pure offline functions over scored run artifacts.  Arc-AGI-2 cell questions
are CHUNKED across the server's per-request payload limit, so one grid's cells
may span several chunk items (ids ``{task}:{grid}#c{k}`` or ``{task}#c{k}``).
Aggregation is exact and fail-closed:

* a GRID is solved only when EVERY one of its cells — across all of its
  chunks — has a usable, correct prediction; any missing/failed/wrong cell
  makes the grid wrong (never partially credited, never imputed);
* a TASK is solved only when ALL of its test-output grids are fully correct —
  the official ARC success criterion (pass@1; Jev's protocol has no retries);
* headline task accuracy uses ALL tasks as the denominator, including tasks
  with missing terminal chunks (counted unsolved, listed explicitly);
* cell-level accuracy is reported as a separate diagnostic over ALL defined
  cells (missing terminal chunks count as wrong cells), never as a substitute
  for task truth;
* deterministic: same artifacts in, same numbers out.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .benchmark_score import ScoringError, argmax_level, score_run

_CHUNK_SUFFIX = re.compile(r"#c(\d+)$")
_GRID_STAGE_QID = re.compile(r"^r(\d+)c(\d+)$")
_TASK_STAGE_QID = re.compile(r"^g(\d+)r(\d+)c(\d+)$")


def aggregate_grid_stage(run_dir: Path | str, *, stage: str) -> dict[str, Any]:
    """Aggregate one arc_agi2_* stage run into grid/task/cell accuracies."""
    run_dir = Path(run_dir)
    if not stage.startswith("arc_agi2"):
        raise ScoringError(f"aggregate_grid_stage: non-ARC-AGI-2 stage {stage!r}")
    whole_task = stage == "arc_agi2_task"
    doc = score_run(run_dir, stage=stage)
    metas = load_items_arc_meta(run_dir)
    gold_maps = load_gold_maps(run_dir)
    results = {str(r.get("item_id")): r for r in _read_results(run_dir)
               if r.get("terminal") is True}

    # expected cells per grid: {(task_id, grid_index): {qid: gold_value}}
    expected: dict[tuple[str, int], dict[str, str]] = {}
    for item_id, meta in metas.items():
        gkey = (meta["task_id"], int(meta.get("grid_index", 0)) if not whole_task else -1)
        if whole_task:
            for g_index, (h, w) in enumerate(meta["grid_dims"]):
                bucket = expected.setdefault((meta["task_id"], g_index), {})
                for r in range(h):
                    for c in range(w):
                        bucket[f"g{g_index}r{r}c{c}"] = None  # gold filled below
        else:
            h, w = meta["height"], meta["width"]
            bucket = expected.setdefault(gkey, {})
            for r in range(h):
                for c in range(w):
                    bucket[f"r{r}c{c}"] = None
    # fill gold values (fail-closed: every expected cell must have gold)
    for item_id, meta in metas.items():
        gkey = (meta["task_id"], int(meta.get("grid_index", 0)) if not whole_task else -1)
        gmap = gold_maps.get(item_id) or {}
        if whole_task:
            for qid, value in gmap.items():
                mm = _TASK_STAGE_QID.match(qid)
                if not mm:
                    raise ScoringError(f"{item_id}: malformed task-stage qid {qid!r}")
                g_index = int(mm.group(1))
                expected[(meta["task_id"], g_index)][qid] = value
        else:
            bucket = expected[gkey]
            for qid, value in gmap.items():
                if qid not in bucket:
                    raise ScoringError(
                        f"{item_id}: gold cell {qid!r} outside the item's grid")
                bucket[qid] = value
    for gkey, bucket in expected.items():
        missing_gold = [qid for qid, v in bucket.items() if v is None]
        if missing_gold:
            raise ScoringError(f"grid {gkey}: {len(missing_gold)} cells without gold")

    # score cells
    cells_total = sum(len(b) for b in expected.values())
    cells_correct = 0
    cells_unusable = 0
    grids: dict[tuple[str, int], dict[str, bool]] = {}
    for gkey, bucket in expected.items():
        verdicts = dict.fromkeys(bucket, False)
        grids[gkey] = verdicts
    for item_id, meta in metas.items():
        result = results.get(item_id)
        gmap = gold_maps.get(item_id) or {}
        if result is None or result.get("status") != "ok":
            continue  # all its cells stay False (missing/failed chunk = wrong)
        predictions = result.get("predictions") or {}
        if len(predictions) != len(gmap):
            raise ScoringError(
                f"{item_id}: expected {len(gmap)} cell predictions, got "
                f"{len(predictions)}; refusing to aggregate a partial chunk")
        for qid, answer in predictions.items():
            if qid not in gmap:
                raise ScoringError(f"{item_id}: predicted cell {qid!r} has no gold")
            usable = bool((answer or {}).get("usable"))
            if not usable:
                cells_unusable += 1
            if (answer or {}).get("type") == "score":
                choice = argmax_level((answer or {}).get("probabilities"))
            else:
                choice = (answer or {}).get("choice")
            correct = bool(usable and choice is not None
                           and str(choice) == str(gmap[qid]))
            if whole_task:
                mm = _TASK_STAGE_QID.match(qid)
                gkey = (meta["task_id"], int(mm.group(1))) if mm else None
            else:
                gkey = (meta["task_id"], int(meta.get("grid_index", 0)))
            if gkey is None or qid not in grids.get(gkey, {}):
                raise ScoringError(f"{item_id}: cell {qid!r} outside the frozen grids")
            grids[gkey][qid] = correct
    for bucket in grids.values():
        cells_correct += sum(1 for ok in bucket.values() if ok)

    grids_solved = sum(1 for bucket in grids.values() if all(bucket.values()))
    per_task: dict[str, dict[str, Any]] = {}
    tasks: dict[str, list[tuple[str, int]]] = {}
    for gkey in grids:
        tasks.setdefault(gkey[0], []).append(gkey)
    for task_id in sorted(tasks):
        gkeys = sorted(tasks[task_id])
        solved = all(all(grids[g].values()) for g in gkeys)
        g_correct = sum(1 for g in gkeys if all(grids[g].values()))
        per_task[task_id] = {
            "solved": solved,
            "grids_correct": g_correct,
            "grids_total": len(gkeys),
        }
    n_tasks = len(per_task)
    n_tasks_expected = len({meta["task_id"] for meta in metas.values()})
    missing_items = sorted(set(metas) - set(results))

    summary = {
        "stage": stage,
        "run_dir": str(run_dir),
        "official_task_criterion": ("task solved only when every output cell of "
                                    "every test input is exactly correct "
                                    "(pass@1; no retries by protocol)"),
        "n_tasks_expected": n_tasks_expected,
        "n_tasks": n_tasks,
        "tasks_solved": sum(1 for t in per_task.values() if t["solved"]),
        "task_accuracy": (sum(1 for t in per_task.values() if t["solved"]) / n_tasks)
        if n_tasks else None,
        "grids_expected": len(grids),
        "grids_correct": grids_solved,
        "grid_accuracy": (grids_solved / len(grids)) if grids else None,
        "cell_accuracy_diagnostic": (cells_correct / cells_total) if cells_total else None,
        "cells_correct": cells_correct,
        "cells_total": cells_total,
        "n_chunk_items_expected": len(metas),
        "n_chunk_items_terminal": len(results),
        "n_missing_terminal": doc["n_missing_terminal"],
        "missing_items": missing_items,
        "status_counts": doc["status_counts"],
        "cells_unusable": cells_unusable,
        "note_on_strict_format": ("per-item strict-format counts are not defined "
                                  "for multi-question chunk items; unusable CELLS "
                                  "are counted in cells_unusable and scored wrong"),
        "finite_set_caveat": doc["finite_set_caveat"],
        "base_score_doc": {
            "note": ("item-level choice accuracy is NOT defined for multi-question "
                     "chunk items; use the cell/grid/task metrics above"),
            "wilson_cell_diagnostic": doc.get("wilson_95"),
        },
        "per_task": per_task,
    }
    return summary


def load_items_arc_meta(run_dir: Path | str) -> dict[str, dict[str, Any]]:
    """item_id -> arc_meta from the run's frozen items.jsonl."""
    path = Path(run_dir) / "items.jsonl"
    metas: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        record = json.loads(line)
        meta = record.get("arc_meta")
        if meta is None:
            raise ScoringError(f"item {record.get('id')} has no arc_meta; refusing")
        metas[str(record["id"])] = meta
    return metas


def load_gold_maps(run_dir: Path | str) -> dict[str, dict[str, str]]:
    """item_id -> {qid: gold value} from the frozen items.jsonl."""
    path = Path(run_dir) / "items.jsonl"
    gold: dict[str, dict[str, str]] = {}
    for line in path.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        record = json.loads(line)
        values: dict[str, str] = {}
        for qid, payload in (record.get("gold") or {}).items():
            value = None
            if isinstance(payload, dict):
                if payload.get("value") is not None:      # {"value": v}
                    value = payload["value"]
                else:                                      # {group: {"value": v}}
                    for inner in payload.values():
                        if isinstance(inner, dict) and inner.get("value") is not None:
                            value = inner["value"]
                            break
            if value is not None:
                values[qid] = str(value)
        gold[str(record["id"])] = values
    return gold


def _read_results(run_dir: Path | str) -> list[dict[str, Any]]:
    path = Path(run_dir) / "results.jsonl"
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        if line.strip():
            records.append(json.loads(line))
    return records
