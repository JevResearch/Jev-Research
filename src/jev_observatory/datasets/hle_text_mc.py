"""Humanity's Last Exam — text-only multiple-choice sub-track (cais/hle).

HLE is a 2,500-question expert-written benchmark (CAIS/Scale AI); access was
accepted 2026-09-20 and the pinned test parquet is downloaded with the user's
token and sha256-sidecarred.  The dataset card asks that items NOT be
re-distributed — this repo keeps the file local and never re-uploads items;
the canary string is never transmitted.

This adapter uses the TEXT-ONLY MULTIPLE-CHOICE sub-track ONLY:

* text-only: the ``image`` field is empty (2,158 of 2,500 rows);
* multiple-choice: ``answer_type == "multipleChoice"`` (591 rows, of which
  513 are text-only); the option block embedded in the question text
  (lines ``A. ...`` / ``A) ...``) is parsed into the Choice criteria with the
  dataset's OWN letters and native order preserved (never re-lettered);
* items whose gold letter does not parse into the extracted option set are
  EXCLUDED at build time and counted (recorded in the spec), never guessed;
* the 1,595+ text-only exactMatch (short free-form answer) questions are OUT
  OF SCOPE for native Jev (it cannot emit text): recorded, not converted.

Official protocol (Scale leaderboard): model answers with reasoning, graded
by exact match / letter match.  This run: direct answer, no CoT, no tools —
labelled as a distinct protocol; reference scores (e.g. Gemini 3.1 Pro 47.3,
GPT-5.4 Pro 45.3 on the text-only leaderboard) are historical_contextual only.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .base import DatasetProvenance, LoadedDataset, read_jsonl

SOURCE_URL = "https://huggingface.co/datasets/cais/hle"
INSTRUCTIONS = (
    "Answer the multiple-choice question. Choose the single best option.")
_OPTION_MARKER = re.compile(r"(?:^|\n)\s*([A-L])[).]\s")
_EXPECTED_ROWS = 2500


class HleError(RuntimeError):
    """Fail-closed condition for the HLE adapter."""


def parse_mc_options(question: str) -> dict[str, str] | None:
    """Extract the embedded option block from an HLE MC question.

    Returns {letter: option_text} with the dataset's native letters (A..L) in
    order of appearance, or None when no option lines are found.
    """
    matches = list(_OPTION_MARKER.finditer(question))
    if not matches:
        return None
    options: dict[str, str] = {}
    for index, match in enumerate(matches):
        letter = match.group(1)
        if letter in options:
            return None  # repeated letter block: not a clean option list
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(question)
        options[letter] = question[start:end].strip()
    return options


def _read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        return pq.read_table(str(path)).to_pylist()
    return read_jsonl(path)


def load_hle(
    path: Path | str,
    *,
    revision: str | None = None,
    accessed_at: str | None = None,
    expected_n: int | None = None,
    max_state_chars: int = 60_000,
) -> tuple[LoadedDataset, dict[str, Any]]:
    """Load the pinned HLE parquet; return (dataset, build accounting).

    The accounting records the sub-track selection honestly: total rows,
    text-only rows, text-only MC rows, parsed-with-gold rows (the stage
    population) and every exclusion reason with counts.  ``expected_n``
    relaxed ONLY for small fixtures.
    """
    if expected_n is not None and expected_n != _EXPECTED_ROWS:
        if expected_n < 10:  # fixture mode: caller passes row count
            required_n = expected_n
        else:
            required_n = _EXPECTED_ROWS
    else:
        required_n = _EXPECTED_ROWS
    records = _read_records(Path(path))
    if len(records) != required_n:
        raise HleError(
            f"{path}: expected the HLE test split ({required_n} rows), got "
            f"{len(records)}; refusing")
    accounting: dict[str, Any] = {
        "n_rows": len(records),
        "exclusion_reasons": {},
        "excluded_ids": [],
    }
    items: list[dict[str, Any]] = []

    def exclude(item_id: str, reason: str) -> None:
        accounting["exclusion_reasons"][reason] = (
            accounting["exclusion_reasons"].get(reason, 0) + 1)
        accounting["excluded_ids"].append(item_id)

    for record in records:
        item_id = str(record.get("id") or "")
        if not item_id:
            raise HleError("HLE row without id")
        accounting.setdefault("n_text_only", 0)
        if record.get("image"):
            exclude(item_id, "multimodal_row")
            continue
        accounting["n_text_only"] += 1
        if record.get("answer_type") != "multipleChoice":
            exclude(item_id, "free_form_answer_out_of_scope")
            continue
        question = str(record["question"])
        options = parse_mc_options(question)
        if options is None:
            exclude(item_id, "option_block_unparseable")
            continue
        gold = str(record.get("answer") or "").strip()
        if gold not in options:
            exclude(item_id, "gold_letter_not_in_parsed_options")
            continue
        if len(options) < 2 or len(options) > 253:
            exclude(item_id, "option_count_out_of_contract")
            continue
        if len(question) > max_state_chars:
            exclude(item_id, "state_over_documented_limit")
            continue
        items.append({
            "id": item_id,
            "group": str(record.get("category") or "unknown"),
            "cluster": item_id,
            "state": question,
            "leakage_check": False,  # option texts legitimately carry letters
            "gold": {"hle": {"value": gold}},
            "questions": {
                "hle": {
                    "type": "choice",
                    "instructions": INSTRUCTIONS,
                    "criteria": options,
                }
            },
            "hle_meta": {
                "gold_answer_text": gold,
                "n_options": len(options),
                "category": str(record.get("category") or ""),
                "raw_subject": str(record.get("raw_subject") or ""),
            },
        })
    accounting["n_built"] = len(items)
    accounting["note"] = (
        "text-only MC sub-track; exactMatch questions are out of scope for "
        "native Jev (no free-form output) and are recorded, never converted")
    provenance = DatasetProvenance(
        name="hle-text-only-mc",
        source_url=SOURCE_URL,
        revision=revision,
        accessed_at=accessed_at,
        license_note="HLE data terms at source; gated access accepted; items never "
                     "re-distributed; canary string never transmitted",
    )
    return LoadedDataset(name="hle-text-only-mc", items=items,
                         provenance=provenance, category_field="group"), accounting


def build_spec(
    dataset: LoadedDataset,
    accounting: dict[str, Any],
    *,
    model: str = "jev-1.13.0",
    dataset_file: str | None = None,
    source_sha256: str | None = None,
) -> dict[str, Any]:
    selected = [{k: v for k, v in item.items() if not k.startswith("_")}
                for item in dataset.items]
    return {
        "experiment": "hle-text-only-mc-full",
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
            "option_order": "native (dataset letters preserved; never re-lettered)",
        },
        "build_accounting": accounting,
        "dataset": {
            "name": dataset.name,
            "source_url": dataset.provenance.source_url,
            "revision": dataset.provenance.revision,
            "accessed_at": dataset.provenance.accessed_at,
            "file": dataset_file,
            "file_sha256": source_sha256,
            "license_note": dataset.provenance.license_note,
            "n_population": accounting.get("n_built"),
            "sampling": {"mode": "subtrack", "split": "test (text-only MC)"},
        },
        "items": selected,
    }
