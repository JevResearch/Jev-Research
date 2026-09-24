"""ARC-Challenge (allenai/ai2_arc, ARC-Challenge config) loader and spec builder.

Only the official TEST split is ever used for evaluation; train/validation are
never scored (BENCHMARK-EXECUTION-GATE item 3).  The pinned file is the public
parquet export of that split; its sha256 is recorded in a sidecar next to the
file and verified before any spec is built.

Source schema: id, question, choices{text[], label[]}, answerKey.  Option keys
keep the dataset's own labels (A..E or 1..5), so the outbound option map and
the gold label use the canonical labels with no silent re-lettering.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import DatasetProvenance, LoadedDataset

SOURCE_URL = "https://huggingface.co/datasets/allenai/ai2_arc"
PINNED_FILE_URL = (
    "https://huggingface.co/datasets/allenai/ai2_arc/resolve/main/"
    "ARC-Challenge/test-00000-of-00001.parquet"
)
EXPECTED_N_TEST = 1172
INSTRUCTIONS = (
    "Answer the multiple-choice question. Reply with only the option key."
)


def _read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        return pq.read_table(str(path)).to_pylist()
    import json

    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_arc_challenge(path: Path | str, *, revision: str | None = None,
                       accessed_at: str | None = None,
                       expected_n: int | None = None) -> LoadedDataset:
    """``expected_n`` may be relaxed ONLY for small offline test fixtures;
    production use keeps the official full TEST-split count (1,172)."""
    required_n = EXPECTED_N_TEST if expected_n is None else expected_n
    records = _read_records(Path(path))
    if not records:
        raise ValueError(f"{path} contains no records")
    if len(records) != required_n:
        raise ValueError(
            f"{path}: expected the ARC-Challenge TEST split ({required_n} items), "
            f"got {len(records)}; refusing (train/validation are never used for evaluation)"
        )
    items: list[dict[str, Any]] = []
    for record in records:
        choices = record.get("choices") or {}
        labels = list(choices.get("label") or [])
        texts = list(choices.get("text") or [])
        answer_key = str(record.get("answerKey") or "")
        question_id = str(record.get("id") or "")
        if not question_id:
            raise ValueError("ARC record without id")
        if not labels or len(labels) != len(texts):
            raise ValueError(f"ARC item {question_id}: malformed choices")
        if len(set(labels)) != len(labels):
            raise ValueError(f"ARC item {question_id}: duplicate option keys")
        if answer_key not in labels:
            raise ValueError(
                f"ARC item {question_id}: answerKey {answer_key!r} not among option keys {labels}"
            )
        items.append({
            "id": question_id,
            "group": "arc-challenge",
            "cluster": question_id,
            "state": str(record["question"]),
            # The answer text is one of the option descriptions; the state
            # leakage guard is explicitly opted out and recorded (same policy
            # as the MMLU-Pro loader).
            "leakage_check": False,
            "gold": {"arc": {"value": answer_key}},
            "questions": {
                "arc": {
                    "type": "choice",
                    "instructions": INSTRUCTIONS,
                    "criteria": {str(label): str(text) for label, text in zip(labels, texts)},
                }
            },
        })
    provenance = DatasetProvenance(
        name="arc-challenge",
        source_url=SOURCE_URL,
        revision=revision,
        accessed_at=accessed_at,
        license_note="public dataset; terms recorded from the source card before any publication",
    )
    return LoadedDataset(name="arc-challenge", items=items, provenance=provenance, category_field="group")


def build_spec(
    dataset: LoadedDataset,
    *,
    model: str = "jev-1.13.0",
    dataset_file: str | None = None,
) -> dict[str, Any]:
    """Full TEST-split spec; no sampling, no prompt variation."""
    selected = [{k: v for k, v in item.items() if not k.startswith("_")} for item in dataset.items]
    return {
        "experiment": "arc-challenge-test-full",
        "model": model,
        "test_family": "public-benchmark",
        "seeds": {"order": 0},
        "shuffle": True,
        "protocol": {
            "instructions": INSTRUCTIONS,
            "prompt_variants": 1,
            "fixed_instructions": True,
            "no_chain_of_thought": True,
            "correct_answer_retries": 0,
            "option_order": "native (dataset labels preserved; never sorted)",
        },
        "dataset": {
            "name": dataset.name,
            "source_url": dataset.provenance.source_url,
            "revision": dataset.provenance.revision,
            "accessed_at": dataset.provenance.accessed_at,
            "file": dataset_file,
            "file_sha256": dataset.provenance.file_sha256,
            "license_note": dataset.provenance.license_note,
            "n_population": len(dataset.items),
            "sampling": {"mode": "full_set", "split": "test"},
        },
        "items": selected,
    }
