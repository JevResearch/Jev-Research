"""Dataset layer: pinned loaders, solver-backed generators, permutation maps.

Rules (DESIGN.md §3, IMPLEMENTATION.md M1):
* loaders are read-only and never modify source files;
* every item gets a stable content hash, so analysis can verify what was run;
* gold labels stay out of outbound payloads (enforced upstream in experiment.py);
* public datasets opt OUT of the state-leakage guard by default (an answer text
  can legitimately appear in a reading passage), and this opt-out is recorded
  explicitly rather than silent;
* no dataset is fetched during tests: fixtures carry the schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class DatasetProvenance:
    """Everything needed to reproduce which revision of a dataset was used."""

    name: str
    source_url: str
    revision: str | None
    accessed_at: str | None
    license_note: str
    file_sha256: str | None = None
    n_items: int = 0

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def content_hash(obj: Any) -> str:
    return sha256(
        json_dumps(obj).encode("utf-8")
    ).hexdigest()


def json_dumps(obj: Any) -> str:
    import json

    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    import json

    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc
    return records


def write_jsonl(path: Path | str, records: list[dict[str, Any]]) -> None:
    import json

    with Path(path).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


@dataclass
class LoadedDataset:
    """A parsed dataset ready for spec building."""

    name: str
    items: list[dict[str, Any]]
    provenance: DatasetProvenance
    category_field: str | None = None

    def categories(self) -> dict[str, int]:
        if not self.category_field:
            return {}
        counts: dict[str, int] = {}
        for item in self.items:
            key = str(item.get(self.category_field, "unknown"))
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))

    def items_sha256(self) -> str:
        return content_hash(self.items)


Loader = Callable[..., LoadedDataset]