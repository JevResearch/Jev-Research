"""External-score schema + heatmap tests (provenance discipline)."""

import json

import pytest

from jev_observatory.external import (
    ScoreImportError,
    build_heatmap,
    heatmap_markdown,
    import_scores,
)


def _score_doc(**over):
    score = {
        "source": {
            "source_url": "https://example.org/leaderboard",
            "accessed_at": "2026-09-18",
            "model": "some-model",
            "dataset": "mmlu-pro",
        },
        "comparison_class": "historical_protocol_compatible",
        "metric": "accuracy",
        "value": 0.75,
        "n": 12032,
        "protocol_notes": "5-shot CoT; chance=0.1",
    }
    score.update(over)
    return {"scores": [score]}


def _write(tmp_path, doc):
    path = tmp_path / "scores.json"
    path.write_text(json.dumps(doc))
    return path


def test_import_valid_score(tmp_path):
    scores = import_scores(_write(tmp_path, _score_doc()))
    assert len(scores) == 1
    assert scores[0].paired_capable is False  # aggregate without item predictions


def test_missing_provenance_refused(tmp_path):
    doc = _score_doc()
    del doc["scores"][0]["source"]["source_url"]
    with pytest.raises(ScoreImportError, match="source_url"):
        import_scores(_write(tmp_path, doc))


def test_unknown_comparison_class_refused(tmp_path):
    with pytest.raises(ScoreImportError, match="comparison_class"):
        import_scores(_write(tmp_path, _score_doc(comparison_class="best_effort")))


def test_matched_rerun_requires_item_predictions(tmp_path):
    with pytest.raises(ScoreImportError, match="item_predictions_file"):
        import_scores(_write(tmp_path, _score_doc(comparison_class="matched_rerun")))
    ok = _score_doc(comparison_class="matched_rerun", item_predictions_file="preds.jsonl")
    assert import_scores(_write(tmp_path, ok))[0].paired_capable is True


def test_bad_n_and_value_types_refused(tmp_path):
    with pytest.raises(ScoreImportError, match="n must be"):
        import_scores(_write(tmp_path, _score_doc(n="many")))
    with pytest.raises(ScoreImportError, match="value must be numeric"):
        import_scores(_write(tmp_path, _score_doc(value="high")))


def test_heatmap_missingness_and_contextual_marker(tmp_path):
    scores = import_scores(_write(tmp_path, _score_doc()))
    grid = build_heatmap(scores, models=["some-model", "other-model"], datasets=["mmlu-pro", "boolq"])
    rows = {r["dataset"]: r["cells"] for r in grid["rows"]}
    assert rows["mmlu-pro"][0]["display"] == "0.75"
    assert rows["mmlu-pro"][1]["missing"] is True and rows["mmlu-pro"][1]["display"] == "—"
    assert rows["boolq"][0]["missing"] is True


def test_heatmap_contextual_cells_marked(tmp_path):
    doc = _score_doc(comparison_class="historical_contextual")
    scores = import_scores(_write(tmp_path, doc))
    grid = build_heatmap(scores)
    cell = grid["rows"][0]["cells"][0]
    assert cell["display"].endswith("^")
    assert cell["comparison_class"] == "historical_contextual"


def test_heatmap_chance_adjust_requires_explicit_chance(tmp_path):
    scores = import_scores(_write(tmp_path, _score_doc()))
    adjusted = build_heatmap(scores, chance_adjust=True)["rows"][0]["cells"][0]
    assert adjusted["value"] == pytest.approx((0.75 - 0.1) / 0.9)
    no_notes = _score_doc(protocol_notes="5-shot CoT")  # no chance= anywhere
    scores2 = import_scores(_write(tmp_path, no_notes))
    refused = build_heatmap(scores2, chance_adjust=True)["rows"][0]["cells"][0]
    assert refused["value"] == 0.75  # unadjusted rather than a guessed chance


def test_conflicting_scores_for_same_cell_refused(tmp_path):
    doc = _score_doc()
    doc["scores"].append(json.loads(json.dumps(doc["scores"][0])))
    doc["scores"][1]["value"] = 0.99
    scores = import_scores(_write(tmp_path, doc))
    with pytest.raises(ScoreImportError, match="conflicting"):
        build_heatmap(scores)


def test_markdown_renders_missing_and_notes(tmp_path):
    scores = import_scores(_write(tmp_path, _score_doc()))
    md = heatmap_markdown(build_heatmap(scores, models=["some-model", "other"], datasets=["mmlu-pro"]))
    assert "| dataset | some-model | other |" in md
    assert "0.75" in md and "—" in md
    assert "contextual" in md or "not be pooled" in md