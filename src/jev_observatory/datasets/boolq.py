"""BoolQ loader: passage entailment as Noul and as Choice, reported separately.

Source: SuperGLUE BoolQ (question, passage, label). License: the dataset ships
under its own terms (CC BY-SA 3.0 via the source card); record provenance and
do not redistribute items.

The two encodings are *different conditions*:
* `boolq_noul`   — one yes/no question per item;
* `boolq_choice` — a two-option Choice ("yes"/"no"), which exercises a
  different output path; DESIGN.md §3 requires they never be pooled.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import DatasetProvenance, LoadedDataset, read_jsonl

SOURCE_URL = "https://github.com/google-research-datasets/boolean-questions"
EXPECTED_FIELDS = {"question", "passage", "label"}

CHOICE_OPTIONS = {"yes": "The passage supports the question as true.",
                  "no": "The passage does not support the question as true."}


def _base_items(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items = []
    for index, record in enumerate(records):
        question = str(record["question"]).strip()
        passage = str(record["passage"]).strip()
        if not question or not passage:
            raise ValueError(f"record {index}: empty question or passage")
        label = record["label"]
        if not isinstance(label, bool):
            raise ValueError(f"record {index}: label must be boolean, got {label!r}")
        items.append({
            "id": str(record.get("id") or f"boolq-{index}"),
            "passage": passage,
            "question": question,
            "label": label,
        })
    return items


def _question(state_passage: str, question_text: str) -> str:
    return f"{state_passage}\n\nQuestion: {question_text}"


def _read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        table = pq.read_table(str(path))
        records = table.to_pylist()
        # HF's google/boolq uses `answer` for the label
        for record in records:
            if "label" not in record and "answer" in record:
                record["label"] = record.pop("answer")
        return records
    return read_jsonl(path)


def load_boolq(path: Path | str, *, revision: str | None = None,
               accessed_at: str | None = None) -> LoadedDataset:
    """Produce one item per record *per condition*, with distinct ids."""
    records = _read_records(Path(path))
    if not records:
        raise ValueError(f"{path} contains no records")
    missing = EXPECTED_FIELDS - set(records[0])
    if missing:
        raise ValueError(f"{path}: records missing fields {sorted(missing)}")
    base = _base_items(records)

    items: list[dict[str, Any]] = []
    for entry in base:
        shared = {
            "group": "boolq",
            "cluster": entry["id"],  # both conditions of one record share a cluster
            "leakage_check": False,
        }
        items.append({
            **shared,
            "id": f"{entry['id']}:noul",
            "condition": "boolq_noul",
            "state": _question(entry["passage"], entry["question"]),
            "gold": {"answer": {"value": entry["label"]}},
            "questions": {
                "answer": {
                    "type": "noul",
                    "instructions": "Based only on the passage, is the answer to the question yes?",
                }
            },
        })
        items.append({
            **shared,
            "id": f"{entry['id']}:choice",
            "condition": "boolq_choice",
            "state": _question(entry["passage"], entry["question"]),
            "gold": {"answer": {"value": "yes" if entry["label"] else "no"}},
            "questions": {
                "answer": {
                    "type": "choice",
                    "instructions": "Based only on the passage, does it support the question as true?",
                    "criteria": CHOICE_OPTIONS,
                }
            },
        })
    provenance = DatasetProvenance(
        name="boolq",
        source_url=SOURCE_URL,
        revision=revision,
        accessed_at=accessed_at,
        license_note="dataset terms at source (CC BY-SA 3.0 per source card); do not redistribute items",
    )
    return LoadedDataset(name="boolq", items=items, provenance=provenance, category_field="condition")


def build_spec(
    dataset: LoadedDataset,
    *,
    n: int | None = None,
    paired: bool = False,
    seed: int = 0,
    model: str = "jev-1.13.0",
    dataset_file: str | None = None,
) -> dict[str, Any]:
    """Build a BoolQ spec.

    * ``paired=True``: sample ``n`` *base records* and emit BOTH encodings for
      each — the paired design needed to attribute accuracy differences to the
      encoding rather than to different questions (audit finding).
    * otherwise: balanced sample across conditions with independent records
      (a between-items design; report accordingly).
    """
    import random

    items = dataset.items
    sample_meta: dict[str, Any] = {"mode": "full_set"}
    if paired:
        if n is None or n <= 0:
            raise ValueError("paired sampling requires n (number of base records)")
        by_condition: dict[str, list[dict[str, Any]]] = {}
        for item in items:
            by_condition.setdefault(item["condition"], []).append(item)
        if set(by_condition) != {"boolq_noul", "boolq_choice"}:
            raise ValueError("paired mode needs exactly the two boolq conditions")
        noul_by_base = {it["cluster"]: it for it in by_condition["boolq_noul"]}
        choice_by_base = {it["cluster"]: it for it in by_condition["boolq_choice"]}
        shared_bases = sorted(set(noul_by_base) & set(choice_by_base))
        rng = random.Random(seed)
        chosen = rng.sample(shared_bases, min(n, len(shared_bases)))
        items = [choice_by_base[base] for base in chosen] + [noul_by_base[base] for base in chosen]
        sample_meta = {
            "mode": "paired_conditions",
            "n_base_records": n,
            "n_requests": len(items),
            "seed": seed,
            "note": "same records in both encodings; conditions reported separately, differences paired",
        }
    elif n is not None:
        if n % 2 != 0:
            raise ValueError("n must be even: conditions are sampled in equal halves")
        per_condition = n // 2
        rng = random.Random(seed)
        by_condition: dict[str, list[dict[str, Any]]] = {}
        for item in items:
            by_condition.setdefault(item["condition"], []).append(item)
        sampled: list[dict[str, Any]] = []
        for condition in sorted(by_condition):
            members = sorted(by_condition[condition], key=lambda it: it["id"])
            sampled.extend(rng.sample(members, min(per_condition, len(members))))
        items = sampled
        sample_meta = {
            "mode": "balanced_conditions_independent_records",
            "n": n,
            "seed": seed,
            "note": "half noul, half choice from DIFFERENT records; between-subjects only",
        }
    selected = [{k: v for k, v in item.items() if k not in {"passage", "question", "label"}} for item in items]
    return {
        "experiment": f"boolq-{'sample' if n is not None else 'full'}",
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
            "sampling": sample_meta,
        },
        "items": selected,
    }
