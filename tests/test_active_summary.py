"""ACTIVE summary aggregator tests (focused: aggregator + cap only).

Validates the integration seam against the real artifacts: full-denominator
accuracy (never rerun-subset), honest completeness, salvage inclusion, gold
join, cost provenance, and license-safe output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "benchmark"))
sys.path.insert(0, str(ROOT / "src"))

import build_active_summary as bas  # noqa: E402

OUT = ROOT / "data_report" / "baselines" / "v4r1"
SUMMARY = json.loads((OUT / "public_summary.json").read_text(encoding="utf-8"))
SOURCE_MAP = json.loads((OUT / "active_source_map.json").read_text(encoding="utf-8"))


def test_complete_iff_no_pending_and_no_provisional():
    for cell, b in SUMMARY["cells"].items():
        assert b["complete"] == (b["n_pending_replacements"] == 0
                                 and b["n_provisional_transport"] == 0), cell
        if not b["complete"]:
            assert b["accuracy_all_requested"] is None, cell   # never charted
            assert "exclusion_reason" in b, cell
        else:
            assert b["partial_settled_accuracy"] is None if "partial_settled_accuracy" in b else True


def test_accuracy_uses_full_sampled_denominator():
    for cell, b in SUMMARY["cells"].items():
        assert b["n_requested"] > 0
        if b["complete"]:
            expect = round(b["n_correct"] / b["n_requested"], 6)
            assert b["accuracy_all_requested"] == expect, cell
        # row classes partition the FULL sample exactly
        assert sum(b["counts"].values()) == b["n_requested"], cell


def test_mimo_pro_math_completeness_is_count_derived():
    """Regression against the stale checkpoint that called this cell complete
    at 108/139: completeness must be DERIVED from counts (all rows settled),
    never asserted.  Now fully settled: complete with the full denominator."""
    b = SUMMARY["cells"]["xiaomi/mimo-v2.6-pro:math500_choice"]
    assert b["n_pending_replacements"] == 0
    assert b["n_provisional_transport"] == 0
    assert b["complete"] is True
    assert b["accuracy_all_requested"] == round(b["n_correct"] / b["n_requested"], 6)
    assert sum(b["counts"].values()) == b["n_requested"] == 261


def test_salvage_rows_included():
    total_salvage = sum(b["counts"].get("salvage", 0)
                        for b in SUMMARY["cells"].values())
    assert total_salvage == 20     # frozen-plan salvage count, all included


def test_gold_join_never_mismatches():
    assert sum(b["n_gold_join_mismatch"] for b in SUMMARY["cells"].values()) == 0


def test_totals_are_consistent():
    t = SUMMARY["totals"]
    assert t["n_requested"] == sum(b["n_requested"] for b in SUMMARY["cells"].values())
    assert t["n_complete_cells"] + t["n_partial_cells"] == t["n_cells"] == 59
    assert t["n_pending_replacements"] == sum(
        b["n_pending_replacements"] for b in SUMMARY["cells"].values())


def test_costs_carry_provenance_and_bill_counts():
    for cell, b in SUMMARY["cells"].items():
        c = b["cost"]
        assert "original_n_billed" in c and "original_n_unknown" in c
        assert "replacement_n_billed" in c and "replacement_n_unknown" in c
        assert "bill_note" in c
    # typesafe accounting distinguishes tariff-derived from plan worst-case
    ct = SUMMARY["cost_totals"]
    assert ct["typesafe_arc_tariff_derived_usd"] == pytest.approx(1.160391372)
    assert ct["typesafe_arc_plan_worstcase_usd"] == 2.09


def test_effects_distinguish_format_and_cap_finish():
    for cell, b in SUMMARY["cells"].items():
        e = b["effects"]
        assert "strict_format_failures_recovered" in e
        assert "cap_finish_length_replacements" in e
        assert "finish_reason_unknown_original_rows" in e


def test_outputs_are_license_safe():
    for path in (OUT / "public_summary.json", OUT / "active_source_map.json"):
        text = path.read_text(encoding="utf-8")
        assert '"gold"' not in text
        assert '"question"' not in text and '"content"' not in text
        assert '"reasoning"' not in text
    items_text = (OUT / "active_items.jsonl").read_text(encoding="utf-8")
    assert '"gold"' not in items_text and '"content"' not in items_text


def test_source_map_points_to_versioned_evidence():
    sm = SOURCE_MAP
    assert sm["sources"]["replacements"].endswith("v4r1/results.jsonl")
    assert "d98f20eb693e" in sm["sources"]["arc_corrected"]
    assert sm["per_item_file"].endswith("active_items.jsonl")
    assert set(sm["cells"]) == set(SUMMARY["cells"])


def test_builder_rebuild_is_deterministic():
    out = bas.build()
    assert out["summary"]["totals"] == SUMMARY["totals"]
    for cell, b in out["summary"]["cells"].items():
        assert b["counts"] == SUMMARY["cells"][cell]["counts"]
        assert b["n_correct"] == SUMMARY["cells"][cell]["n_correct"]
