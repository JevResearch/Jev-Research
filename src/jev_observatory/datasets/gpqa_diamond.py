"""GPQA Diamond (Idavidrein/gpqa, gpqa_diamond.csv) — Jev-native loader.

The canonical graduate-level Google-proof Q&A benchmark: 198 expert-written
multiple-choice questions (4 options each), released CC-BY-4.0 behind an
accepted HF gate.  Access conditions were accepted 2026-09-20; the pinned CSV
is downloaded with the user's token and sha256-sidecarred (items are NOT
redistributed by this repo and the dataset's canary string is never sent to
any endpoint: the outbound state is ONLY the question text).

Encoding: native Choice, exactly like MMLU-Pro/ARC.  The CSV lists the correct
answer separately from the three incorrect answers (there is no native option
order), and the official protocol randomizes option positions per run — so
this adapter applies ONE deterministic seeded shuffle per item (recorded in
the spec; seed frozen in GPQA_SHUFFLE_SEED) and freezes the result.  Gold is
the option key holding the correct answer text.  Protocol: direct answer, no
chain-of-thought, fixed instructions, 0 retries — NOT the CoT protocols of
published reference scores (labelled historical_contextual).

Late-2026 reference scores (verified 2026-09-19, see research.md / canonical
records): Vals GPQA is ARCHIVED (saturated, no longer run on new models);
Artificial Analysis still tracks it: GPT-6 Astra 96.26 (xhigh) / 96.06 (max),
Gemini 3.8 Flash (high) 95.3, Sol (max) 94.14, Fable 5.1 93.74, Opus 5 (max)
93.23, Qwen3.8-Max 92.83, GLM-5.3 91.72, GLM-5.3-Flash 91.21, DeepSeek
V4-Flash 89.9 (Vals).
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

from .base import DatasetProvenance, LoadedDataset

SOURCE_URL = "https://huggingface.co/datasets/Idavidrein/gpqa"
EXPECTED_N_DIAMOND = 198
GPQA_SHUFFLE_SEED = 4471   # frozen; per-item rng = f"gpqa-shuffle:{seed}:{record_id}"
LETTERS = "ABCD"
INSTRUCTIONS = (
    "Answer the multiple-choice question. Choose the single best option.")
REQUIRED_FIELDS = ("Record ID", "Question", "Correct Answer",
                   "Incorrect Answer 1", "Incorrect Answer 2",
                   "Incorrect Answer 3")


class GpqaError(RuntimeError):
    """Fail-closed condition for the GPQA adapter."""


def load_gpqa_diamond(
    path: Path | str,
    *,
    revision: str | None = None,
    accessed_at: str | None = None,
    expected_n: int | None = None,
) -> tuple[LoadedDataset, dict[str, Any]]:
    """Load the pinned gpqa_diamond.csv. ``expected_n`` relaxed ONLY for fixtures."""
    required_n = EXPECTED_N_DIAMOND if expected_n is None else expected_n
    with open(path, encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    if len(records) != required_n:
        raise GpqaError(
            f"{path}: expected the GPQA Diamond split ({required_n} items), got "
            f"{len(records)}; refusing")
    items: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    accounting: dict[str, Any] = {
        "n_rows": len(records),
        "excluded_ids": [],
        "exclusion_reasons": {},
    }

    def exclude(item_id: str, reason: str) -> None:
        accounting["exclusion_reasons"][reason] = (
            accounting["exclusion_reasons"].get(reason, 0) + 1)
        accounting["excluded_ids"].append(item_id)

    for record in records:
        missing = [f for f in REQUIRED_FIELDS if not record.get(f)]
        if missing:
            raise GpqaError(f"GPQA record missing fields {missing}")
        record_id = str(record["Record ID"]).strip()
        if not record_id or record_id in seen_ids:
            raise GpqaError(f"GPQA record id missing or duplicate: {record_id!r}")
        seen_ids.add(record_id)
        correct = str(record["Correct Answer"]).strip()
        incorrect = [str(record[f"Incorrect Answer {i}"]).strip() for i in (1, 2, 3)]
        options = [correct] + incorrect
        if any(not o for o in options):
            exclude(record_id, "empty_option_text")
            continue
        if len(set(options)) != 4:
            # source data defect (observed in 2 published rows): a duplicated
            # option text makes the 4-way key assignment ambiguous for any
            # answerer — excluded and counted, never repaired silently
            exclude(record_id, "duplicate_option_text_in_source")
            continue
        rng = random.Random(f"gpqa-shuffle:{GPQA_SHUFFLE_SEED}:{record_id}")
        order = list(range(4))
        rng.shuffle(order)
        criteria = {LETTERS[i]: options[order[i]] for i in range(4)}
        gold_key = next(k for k, text in criteria.items() if text == correct)
        domain = str(record.get("High-level domain") or "unknown").strip() or "unknown"
        items.append({
            "id": record_id,
            "group": domain,
            "cluster": record_id,
            "state": str(record["Question"]).strip(),
            "leakage_check": False,  # option texts legitimately carry the answer text
            "gold": {"gpqa": {"value": gold_key}},
            "questions": {
                "gpqa": {
                    "type": "choice",
                    "instructions": INSTRUCTIONS,
                    "criteria": criteria,
                }
            },
            "_source": {k: record.get(k) for k in ("Subdomain",)},
            "gpqa_meta": {
                "shuffle_seed": GPQA_SHUFFLE_SEED,
                "option_order": [options[i] for i in order],  # audit copy
                "gold_answer_text": correct,
            },
        })
    accounting["n_built"] = len(items)
    provenance = DatasetProvenance(
        name="gpqa-diamond",
        source_url=SOURCE_URL,
        revision=revision,
        accessed_at=accessed_at,
        license_note="CC-BY-4.0; gated access accepted; items never redistributed; "
                     "canary string never transmitted",
    )
    return LoadedDataset(name="gpqa-diamond", items=items, provenance=provenance,
                         category_field="group"), accounting


def build_spec(
    dataset: LoadedDataset,
    *,
    model: str = "jev-1.13.0",
    dataset_file: str | None = None,
    source_sha256: str | None = None,
) -> dict[str, Any]:
    selected = [{k: v for k, v in item.items() if not k.startswith("_")}
                for item in dataset.items]
    return {
        "experiment": "gpqa-diamond-full",
        "model": model,
        "test_family": "public-benchmark",
        "seeds": {"order": 0, "shuffle_seed": GPQA_SHUFFLE_SEED},
        "shuffle": True,
        "protocol": {
            "instructions": INSTRUCTIONS,
            "prompt_variants": 1,
            "fixed_instructions": True,
            "no_chain_of_thought": True,
            "correct_answer_retries": 0,
            "option_order": ("one deterministic seeded shuffle per item (the CSV "
                             "has no native order; official protocol randomizes "
                             "positions per run); seed recorded in the spec"),
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
            "sampling": {"mode": "full_set", "split": "diamond"},
        },
        "items": selected,
    }
