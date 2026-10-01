"""Source-to-plot contract for the matched-baseline charts.

Regression for the P0-1 publication blocker: the capability/pareto charts must
be generated from the authoritative v4r1 active summary
(data_report/baselines/v4r1/public_summary.json + active_source_map.json), not
from the stale pre-repair runs_matched_cheap/v3/summary_v3.json. Contract:
  * every one of the 42 complete cells appears, with its all-requested
    accuracy over the FULL sampled denominator and the current cost definition;
  * all 17 partial cells are excluded (never partial_settled_accuracy);
  * the corrected Mistral Small MATH-500 figure is the 217/261 tie with Jev,
    not the legacy 2.7% parser artifact;
  * missing authoritative artifacts fail closed (no legacy fallback);
  * no legacy accuracy_recovered preference anywhere in the chart data path;
  * rendered chart rows equal the authoritative source rows (source-to-plot).
"""
from __future__ import annotations

import html as _html
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "report"))

import chartkit                      # noqa: E402
import comparison_graphs as cg       # noqa: E402
import pareto_graphs as pg           # noqa: E402

DATASETS = ["mmlu", "arc", "gpqa", "math500_choice", "hle_text_mc"]


def _cells(status):
    pub = json.loads(
        (ROOT / "data_report/baselines/v4r1/public_summary.json").read_text())
    return {k: v for k, v in pub["cells"].items()
            if (v["status"] == "complete") == (status == "complete")}


def _rows():
    summary = chartkit.load_matched(ROOT)
    return {r["cell"]: r for ds in DATASETS
            for r in chartkit.matched_points(summary, ds)}


def test_every_complete_cell_charts_source_values():
    rows = _rows()
    comp = _cells("complete")
    assert len(comp) == 42
    assert set(rows) == set(comp), "charted cells must equal the 42 complete cells"
    for key, r in rows.items():
        e = comp[key]
        assert r["accuracy"] == pytest.approx(e["accuracy_all_requested"] * 100)
        assert r["n"] == e["n_requested"]            # FULL sampled denominator
        usd = ((e["cost"]["original_reported_usd"] or 0)
               + (e["cost"]["replacement_reported_usd"] or 0))
        assert r["cost"] == pytest.approx(usd / e["n_requested"])
        jp = e["jev_paired"]
        assert r["jev_join"]["jev_accuracy_on_subset"] == pytest.approx(
            jp["jev_correct_on_subset"] / jp["n_paired"])
        assert r["wire"] == e["original_wire_protocol"]


def test_all_partial_cells_excluded_and_partial_accuracy_unused():
    rows = set(_rows())
    partial = set(_cells("partial"))
    assert len(partial) == 17
    assert not rows & partial
    src = (ROOT / "scripts/report/chartkit.py").read_text()
    assert '.get("partial_settled_accuracy"' not in src
    assert '["partial_settled_accuracy"]' not in src
    assert '.get("accuracy_recovered"' not in src      # no legacy recovery preference
    assert '["accuracy_recovered"]' not in src
    for e in _cells("partial").values():
        assert e["accuracy_all_requested"] is None


def test_mistral_small_math_is_the_current_tie_not_legacy_2_7():
    rows = chartkit.matched_points(chartkit.load_matched(ROOT), "math500_choice")
    m = next(r for r in rows
             if r["id"] == "mistralai/mistral-small-3.2-24b-instruct")
    jev = chartkit.load_jev_scores(ROOT)["math500_mcq_adapted"]["greedy"] * 100
    assert m["n_correct"] == 217 and m["n"] == 261
    assert m["accuracy"] == pytest.approx(217 / 261 * 100, abs=1e-3)
    assert m["accuracy"] == pytest.approx(jev, abs=1e-3)   # tie, not a lead
    assert m["accuracy"] > 80                              # not stale 2.7


def test_missing_active_artifacts_fail_closed_no_legacy_fallback(tmp_path):
    leg = tmp_path / "runs_matched_cheap/v3"
    leg.mkdir(parents=True)
    (leg / "summary_v3.json").write_text(json.dumps({"models": {}}))
    with pytest.raises(FileNotFoundError):
        chartkit.load_matched(tmp_path)


def test_no_legacy_accuracy_recovered_preference(tmp_path):
    pub = {"schema": "active-public-summary-1.0.0", "cells": {
        "m/model-a:mmlu": {
            "model": "m/model-a", "dataset": "mmlu", "status": "complete",
            "complete": True, "n_requested": 10, "n_correct": 5,
            "accuracy_all_requested": 0.5, "counts": {},
            "cost": {"original_reported_usd": 1.0,
                     "replacement_reported_usd": 0.0},
            "effects": {"unrecovered_settled": 0},
            "original_wire_protocol": "v3-provider-defaults",
            "jev_paired": {"n_paired": 10, "jev_correct_on_subset": 6},
        }}}
    bd = tmp_path / "data_report/baselines/v4r1"
    bd.mkdir(parents=True)
    (bd / "public_summary.json").write_text(json.dumps(pub))
    (bd / "active_source_map.json").write_text(json.dumps({"sources": {}}))
    leg = tmp_path / "runs_matched_cheap/v3"
    leg.mkdir(parents=True)
    (leg / "summary_v3.json").write_text(json.dumps(
        {"models": {"m/model-a": {"mmlu": {"accuracy_recovered": 0.9}}}}))
    rows = chartkit.matched_points(chartkit.load_matched(tmp_path), "mmlu")
    assert [r["accuracy"] for r in rows] == [50.0]   # 0.9 legacy never read


def _tips(text):
    return [_html.unescape(m.group(1))
            for m in re.finditer(r'data-tip="([^"]*)"', text)]


def test_rendered_chart_rows_equal_source_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(cg, "OUT", tmp_path / "cmp.html")
    assert cg.main() == 0
    monkeypatch.setattr(pg, "OUT", tmp_path / "par.html")
    pg.main()
    cmp_h = (tmp_path / "cmp.html").read_text()
    tips = _tips(cmp_h)
    matched_tips = [t for t in tips
                    if "measured cost $" in t and "/question" in t]
    # exactly the 42 complete cells plotted; the 17 partial cells absent
    assert len(matched_tips) == 42
    rows = _rows()
    assert len(rows) == 42
    for key, r in rows.items():
        score = f"score {r['accuracy']:.1f}%"
        assert any(r["name"] in t and score in t for t in matched_tips), \
            (key, score)
    for key, e in _cells("partial").items():
        name = chartkit.CHEAP_META.get(
            e["model"], (e["model"].split("/")[-1],))[0]
        bad = f"score {e['partial_settled_accuracy'] * 100:.1f}%"
        assert not any(name in t and bad in t for t in matched_tips), key
    # current Mistral MATH tie on the plot, never the legacy 2.7 artifact
    ms = [t for t in matched_tips if "Mistral Small 3.2" in t]
    assert any("score 83.1%" in t for t in ms)
    assert not any("score 2.7%" in t for t in ms)
