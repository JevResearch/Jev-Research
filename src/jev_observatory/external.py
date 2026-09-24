"""External benchmark scores: provenance-first import, no invented statistics.

A reference result is usable ONLY with its provenance (DESIGN.md §4).  The
comparison class controls what analyses may do with it:

* `matched_rerun`                — we ran the items ourselves; item ids required.
* `historical_protocol_compatible` — aggregate reuse allowed; item-level paired
  tests forbidden (no item predictions).
* `historical_contextual`        — display-only; must be visually separated.

Anything missing provenance is refused at import, not guessed.  Aggregates with
no `n` are kept but flagged; a missing uncertainty is rendered visibly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

COMPARISON_CLASSES = {
    "matched_rerun",
    "historical_protocol_compatible",
    "historical_contextual",
}
REQUIRED_SOURCE_FIELDS = {"source_url", "accessed_at", "model", "dataset"}


class ScoreImportError(ValueError):
    pass


@dataclass
class ExternalScore:
    model: str
    dataset: str
    metric: str
    value: float | None
    comparison_class: str
    source_url: str
    accessed_at: str
    n: int | None = None
    uncertainty: dict[str, Any] | None = None
    protocol_notes: str | None = None
    revision: str | None = None
    item_predictions_file: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def paired_capable(self) -> bool:
        """Only our own reruns with item ids may enter paired tests."""
        return self.comparison_class == "matched_rerun" and self.item_predictions_file is not None

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def import_scores(path: Path | str) -> list[ExternalScore]:
    """Load and validate an external-scores JSON document."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or "scores" not in doc:
        raise ScoreImportError("expected a top-level object with a 'scores' list")
    scores: list[ExternalScore] = []
    for index, raw in enumerate(doc["scores"]):
        try:
            scores.append(_validate_score(raw))
        except ScoreImportError as exc:
            raise ScoreImportError(f"scores[{index}]: {exc}") from exc
    return scores


def _validate_score(raw: Any) -> ExternalScore:
    if not isinstance(raw, dict):
        raise ScoreImportError("score entry must be an object")
    missing_source = REQUIRED_SOURCE_FIELDS - set(raw.get("source", {}))
    if missing_source:
        raise ScoreImportError(f"missing provenance fields {sorted(missing_source)}")
    source = raw["source"]
    comparison_class = raw.get("comparison_class")
    if comparison_class not in COMPARISON_CLASSES:
        raise ScoreImportError(
            f"comparison_class {comparison_class!r} not in {sorted(COMPARISON_CLASSES)}"
        )
    value = raw.get("value")
    if value is not None and not isinstance(value, (int, float)):
        raise ScoreImportError(f"value must be numeric or null, got {value!r}")
    n = raw.get("n")
    if n is not None and (not isinstance(n, int) or isinstance(n, bool) or n < 0):
        raise ScoreImportError(f"n must be a non-negative integer or null, got {n!r}")
    item_predictions_file = raw.get("item_predictions_file")
    if comparison_class == "matched_rerun" and not item_predictions_file:
        raise ScoreImportError("matched_rerun requires item_predictions_file (item ids)")
    return ExternalScore(
        model=str(source["model"]),
        dataset=str(source["dataset"]),
        metric=str(raw.get("metric", "accuracy")),
        value=None if value is None else float(value),
        comparison_class=comparison_class,
        source_url=str(source["source_url"]),
        accessed_at=str(source["accessed_at"]),
        n=n,
        uncertainty=raw.get("uncertainty"),
        protocol_notes=raw.get("protocol_notes"),
        revision=source.get("revision"),
        item_predictions_file=item_predictions_file,
        extra={k: v for k, v in raw.items() if k not in {
            "source", "comparison_class", "value", "n", "metric",
            "uncertainty", "protocol_notes", "item_predictions_file",
        }},
    )


# ------------------------------------------------------------------ heatmap
def build_heatmap(
    scores: list[ExternalScore],
    *,
    models: list[str] | None = None,
    datasets: list[str] | None = None,
    chance_adjust: bool = False,
) -> dict[str, Any]:
    """Heatmap data: datasets x models, raw metric labels, explicit missingness.

    Contextual comparators are hatched (`marker: "contextual"`), never silently
    mixed into comparable cells.
    """
    model_order = models or sorted({s.model for s in scores})
    dataset_order = datasets or sorted({s.dataset for s in scores})
    cells: dict[tuple[str, str], dict[str, Any]] = {}
    for score in scores:
        key = (score.dataset, score.model)
        existing = cells.get(key)
        if existing is not None and existing["value"] is not None and score.value is not None:
            raise ScoreImportError(
                f"conflicting values for {key}: {existing['value']} vs {score.value}; resolve before plotting"
            )
        cells[key] = _cell(score, chance_adjust=chance_adjust)
    grid = []
    for dataset in dataset_order:
        row = []
        for model in model_order:
            cell = cells.get((dataset, model))
            if cell is None or cell["value"] is None:
                row.append({"value": None, "display": "—", "missing": True})
            else:
                row.append(cell)
        grid.append({"dataset": dataset, "cells": row})
    return {
        "models": model_order,
        "datasets": dataset_order,
        "rows": grid,
        "chance_adjusted": chance_adjust,
        "note": "— marks absent results; contextual comparators carry ^ and must not be pooled",
    }


def _cell(score: ExternalScore, *, chance_adjust: bool) -> dict[str, Any]:
    value = score.value
    display_value: Any = value
    if value is not None and chance_adjust and score.metric == "accuracy":
        # Chance is option-count dependent and must arrive via protocol_notes;
        # without it we refuse to adjust rather than assume 1/K.
        chance = _chance_from_notes(score.protocol_notes)
        if chance is None:
            display_value = value
        else:
            display_value = round((value - chance) / (1.0 - chance), 6)
    marker = "" if score.comparison_class != "historical_contextual" else "^"
    suffix = "" if score.n is not None else "?n"
    if display_value is None:
        return {"value": None, "display": "—", "missing": True, "n": score.n}
    return {
        "value": display_value,
        "display": f"{display_value}{marker}{suffix}",
        "missing": False,
        "n": score.n,
        "metric": score.metric,
        "comparison_class": score.comparison_class,
        "uncertainty_missing": score.uncertainty is None,
        "source_url": score.source_url,
    }


def _chance_from_notes(notes: str | None) -> float | None:
    if not notes:
        return None
    marker = "chance="
    if marker not in notes:
        return None
    try:
        return float(notes.split(marker, 1)[1].split()[0])
    except (ValueError, IndexError):
        return None


def heatmap_markdown(heatmap: dict[str, Any]) -> str:
    """Render the grid as markdown; missing cells stay visibly empty."""
    lines: list[str] = []
    header = "| dataset | " + " | ".join(heatmap["models"]) + " |"
    separator = "|---" * (len(heatmap["models"]) + 1) + "|"
    lines.append(header)
    lines.append(separator)
    for row in heatmap["rows"]:
        cells = [cell["display"] for cell in row["cells"]]
        lines.append("| " + row["dataset"] + " | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(f"*{heatmap['note']}*")
    lines.append("")
    return "\n".join(lines)
