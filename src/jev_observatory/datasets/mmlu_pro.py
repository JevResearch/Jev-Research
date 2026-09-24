"""MMLU-Pro loader and spec builder.

Source schema (TIGER-Lab/MMLU-Pro): question_id, question, choices (list of
strings, variable length up to 10), answer_index, answer (letter), category, src.
Local files are JSONL with exactly those fields; `fetch-data` documents the
pinned source. License: the dataset is released under Apache-2.0 for the code
and carries its own data terms (see docs/data-sources.md); we record provenance
and never redistribute items.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from .base import DatasetProvenance, LoadedDataset, content_hash, read_jsonl

LETTERS = "ABCDEFGHIJ"
SOURCE_URL = "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro"
EXPECTED_FIELDS = {"question_id", "question", "answer_index", "answer", "category"}


def _read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        table = pq.read_table(str(path))
        return table.to_pylist()
    return read_jsonl(path)


def load_mmlu_pro(path: Path | str, *, revision: str | None = None,
                  accessed_at: str | None = None) -> LoadedDataset:
    records = _read_records(Path(path))
    if not records:
        raise ValueError(f"{path} contains no records")
    missing = EXPECTED_FIELDS - set(records[0])
    if missing:
        raise ValueError(f"{path}: records missing fields {sorted(missing)}")
    items: list[dict[str, Any]] = []
    for record in records:
        choices = record.get("choices") or record.get("options")  # HF parquet uses `options`
        answer_index = record["answer_index"]
        if not choices:
            raise ValueError(f"question {record['question_id']}: no options present")
        if not 0 <= answer_index < len(choices):
            raise ValueError(f"question {record['question_id']}: answer_index {answer_index} out of range")
        if len(choices) > len(LETTERS):
            raise ValueError(f"question {record['question_id']}: {len(choices)} choices exceed option cap")
        gold_letter = LETTERS[answer_index]
        if record.get("answer") not in (None, gold_letter):
            raise ValueError(
                f"question {record['question_id']}: answer letter {record['answer']!r} "
                f"contradicts answer_index {answer_index}"
            )
        items.append({
            "id": str(record["question_id"]),
            "group": str(record["category"]),
            "cluster": str(record["question_id"]),  # each question is its own cluster
            "state": str(record["question"]),
            # The answer text legitimately lives in the option list; the state
            # leakage guard is therefore explicitly opted out and recorded.
            "leakage_check": False,
            "gold": {"mmlu": {"value": gold_letter}},
            "questions": {
                "mmlu": {
                    "type": "choice",
                    "instructions": "Answer the multiple-choice question. Choose the single best option.",
                    "criteria": {LETTERS[i]: str(choice) for i, choice in enumerate(choices)},
                }
            },
            "_source": {k: record.get(k) for k in ("src", "category")},
        })
    provenance = DatasetProvenance(
        name="mmlu-pro",
        source_url=SOURCE_URL,
        revision=revision,
        accessed_at=accessed_at,
        license_note="dataset terms at source; do not redistribute items or reveal examples online",
    )
    return LoadedDataset(name="mmlu-pro", items=items, provenance=provenance, category_field="group")


def stratified_pilot(items: list[dict[str, Any]], *, n: int, seed: int,
                     category_field: str = "group") -> list[dict[str, Any]]:
    """Proportional allocation with a minimum of one per category; deterministic.

    Returns the sample. Selection probabilities are recoverable: the caller
    passes population category sizes (from the full loader) to the estimator.
    """
    if n <= 0:
        raise ValueError("pilot size must be positive")
    strata: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        strata.setdefault(str(item.get(category_field, "unknown")), []).append(item)
    for members in strata.values():
        members.sort(key=lambda it: it["id"])  # deterministic within stratum

    rng = random.Random(seed)
    allocation: dict[str, int] = {}
    for name in sorted(strata):
        allocation[name] = max(1, round(n * len(strata[name]) / len(items)))
    # Fix rounding drift by trimming or topping up, largest strata first.
    while sum(allocation.values()) > n:
        biggest = max(allocation, key=lambda k: allocation[k])
        if allocation[biggest] <= 1:
            break
        allocation[biggest] -= 1
    while sum(allocation.values()) < n:
        for name in sorted(strata):
            if sum(allocation.values()) >= n:
                break
            if allocation[name] < len(strata[name]):
                allocation[name] += 1

    sample: list[dict[str, Any]] = []
    for name in sorted(strata):
        members = strata[name]
        take = min(allocation[name], len(members))
        indices = rng.sample(range(len(members)), take)
        sample.extend(members[i] for i in indices)
    sample.sort(key=lambda it: (it.get(category_field, "unknown"), it["id"]))
    return sample


def stratum_sizes(items: list[dict[str, Any]], category_field: str = "group") -> dict[str, int]:
    sizes: dict[str, int] = {}
    for item in items:
        key = str(item.get(category_field, "unknown"))
        sizes[key] = sizes.get(key, 0) + 1
    return dict(sorted(sizes.items()))


def build_spec(
    dataset: LoadedDataset,
    *,
    pilot: int | None = None,
    seed: int = 0,
    model: str = "jev-1.13.0",
    dataset_file: str | None = None,
) -> dict[str, Any]:
    """Build an experiment spec; pilot mode records the sampling design."""
    items = dataset.items
    sample_meta: dict[str, Any] = {}
    if pilot is not None:
        items = stratified_pilot(items, n=pilot, seed=seed)
        sample_meta = {
            "mode": "stratified_pilot",
            "pilot_n": pilot,
            "seed": seed,
            "population_category_sizes": stratum_sizes(dataset.items),
            "note": "pilot is NOT the official aggregate; estimates are weighted by stratum sizes",
        }
    selected = [{k: v for k, v in item.items() if not k.startswith("_")} for item in items]
    return {
        "experiment": f"mmlu-pro-{'pilot' if pilot is not None else 'full'}",
        "model": model,
        "test_family": "public-benchmark",
        "seeds": {"order": seed},
        "dataset": {
            "name": dataset.name,
            "source_url": dataset.provenance.source_url,
            "revision": dataset.provenance.revision,
            "accessed_at": dataset.provenance.accessed_at,
            "file": dataset_file,
            "file_sha256": dataset.provenance.file_sha256,
            "license_note": dataset.provenance.license_note,
            "n_population": len(dataset.items),
            "sampling": sample_meta or {"mode": "full_set"},
        },
        "items": selected,
    }
