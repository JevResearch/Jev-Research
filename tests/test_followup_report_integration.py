"""Follow-up report integration boundaries (2026-10-03).

Locks the data-to-prose boundaries of the follow-up round:
  * no May-2025 cliff / hard cutoff language, no "new foundation" ancestry
    claim, no Qwen-base identification, no o200k "wrong sign" claim;
  * corrected false-control count 16/16 (not the old mixed 21/16);
  * the upstream-header claim is September-dated (current deployment has none);
  * counterpoint links resolve to real exported artifacts and carry no gold;
  * knowledge exports: 24 unique items, model-facing file has no answers.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "report"))

import build_report                     # noqa: E402
import build_counterpoint               # noqa: E402

HTML = (ROOT / "report" / "index.html").read_text(encoding="utf-8")
CP = (ROOT / "counterpoint" / "index.html").read_text(encoding="utf-8")
ARCH_MD = (ROOT / "ARCHITECTURE-ANALYSIS.md").read_text(encoding="utf-8")
DIAG = (ROOT / "scripts" / "report" / "arch_diagram.py").read_text(encoding="utf-8")
METHODS = (ROOT / "docs" / "modern-comparison" / "FOLLOWUP-METHODS.md").read_text(
    encoding="utf-8")


# ------------------------------------------------------------ boundaries
def test_no_may_cliff_or_hard_cutoff_claim():
    for text in (HTML, ARCH_MD, DIAG):
        assert "partial to ~May" not in text and "partial to May 2025" not in text
        assert "gone by" not in text and "nothing after June 2025" not in text
        assert "May 2025" not in text and "June 2025" not in text
    # the author cut removed the retrospective changelog line entirely
    assert "partial through May 2025" not in CP
    assert "reading was overstated" not in CP
    assert "not a declared training" in HTML
    assert "strongest on 2024-era facts" in HTML


def test_no_new_foundation_or_qwen_identification():
    for text in (HTML, ARCH_MD, DIAG):
        assert "new foundation" not in text
        assert "relabeled open model" not in text
        assert "No discernible preexisting lineage" not in text
    assert "new foundation" not in CP and "too strong" not in CP
    assert "Qwen-base classification" not in HTML       # never a positive claim


def test_no_o200k_wrong_sign_claim():
    # digits are NOT the fewest-token o200k class; no wrong-sign prose anywhere
    for text in (HTML, CP, ARCH_MD, METHODS):
        assert "fewest tokens" not in text
        assert not re.search(r"o200k.{0,80}wrong sign", text, re.I | re.S)


def test_false_controls_use_corrected_16_of_16():
    assert "16/16 invented events" in HTML
    assert "21/16" not in HTML and "21 of 16" not in HTML
    key = build_report.P5C["per_kind"]["fictional"]
    assert (key["n"], key["gold_hits"]) == (16, 16)


def test_header_claim_is_september_dated():
    assert "In our September timing campaign, responses reported" in HTML
    assert "Every response reports an upstream-service timing header" not in HTML
    assert "twofold cost of question text" in re.sub(r"\s+", " ", CP)


def test_padding_conclusion_not_absolute():
    assert "essentially absolutely" not in HTML
    visible = re.sub(r"\s+", " ", HTML)
    assert "Those filler options did not distract Jev from easy factual answers." in visible
    assert "Our independent tests below show that simply reordering the real choices can change the answer." in visible


def test_counterpoint_is_short_and_uninflated():
    words = re.sub(r"<[^>]+>", " ", build_counterpoint.BODY).split()
    # Author requested the political/language comparison and a central
    # ancestry assessment after the original short draft; keep it readable.
    assert len(words) <= 800
    for anchor in ('counter-ancestry', 'counter-language', 'counter-politics'):
        assert f'id="{anchor}"' in CP


# ------------------------------------------------------------- exports
def _qitems():
    return json.loads((ROOT / "data_report/followup_20261003/"
                       "knowledge-questions.json").read_text(encoding="utf-8"))


def _kitems():
    return json.loads((ROOT / "data_report/followup_20261003/"
                       "knowledge-answer-key.json").read_text(encoding="utf-8"))


def test_knowledge_exports_24_items_no_outbound_gold():
    q, k = _qitems(), _kitems()
    assert q["n_unique_items"] == 24 and len(q["items"]) == 24
    assert len(k["items"]) == 24
    assert {i["id"] for i in q["items"]} == {i["id"] for i in k["items"]}
    assert q["n_variants"] == 48 and q["rotations_per_item"] == 2
    frozen = json.loads((ROOT / "runs_archprobe/followup_20261003/"
                         "plan_frozen.json").read_text())["stimuli"]["fu_horizon"]
    for item, original in zip(q["items"], frozen["items"], strict=True):
        assert set(item) == {"id", "question", "variants"}   # no gold field
        assert re.fullmatch(r"q\d{2}", item["id"])  # no control-kind hints
        assert [v["rotation"] for v in item["variants"]] == frozen["rotations"]
        for variant in item["variants"]:
            offset = variant["rotation"]
            options = original["options"]
            assert variant["options"] == options[offset:] + options[:offset]
            assert len(variant["options"]) >= 4
    q_blob = json.dumps(q).lower()
    for banned in ("\"gold\"", "\"answer\"", "sha256", "gold_anchor"):
        assert banned not in q_blob        # options repeat the gold text, but nothing marks it
    assert "repeated measures" in q["note"]
    assert "NOT a known-pretraining-cutoff test" in q["note"]


def test_answer_key_has_provenance_not_page_copies():
    k = _kitems()
    sources = json.loads((ROOT / "runs_archprobe/followup_20261003/v2/"
                          "sources_v2.json").read_text())["items"]
    questions = {r["id"]: r for r in _qitems()["items"]}
    for row in k["items"]:
        assert row["answer"] and row["date"]
        assert isinstance(row["source"], list)   # empty only for invented events
        for source in row["source"]:
            support = sources[row["source_id"]]
            assert source["url"] == support["source_url"]
            assert source["sha256"] == support["source_sha256"]
            assert support["verified"]
        for selected in row["variants"]:
            variant = next(v for v in questions[row["id"]]["variants"]
                           if v["rotation"] == selected["rotation"])
            assert variant["options"][int(selected["choice"][1:])] == row["answer"]
    blob = json.dumps(k)
    assert "excerpt" not in blob          # no third-party page text shipped


def test_arena_replay_exports_both_rotations_without_kind_hints():
    text = (ROOT / "data_report/followup_20261003/"
            "knowledge-arena-replay.txt").read_text()
    headers = re.findall(r"^q\d{2} / rotation [03]$", text, re.M)
    assert len(headers) == 48 and len(set(headers)) == 48
    assert "hs_fake" not in text and "hs_" not in text


# ------------------------------------------------------------- links
def test_counterpoint_links_resolve_to_real_files():
    for href in re.findall(r'href="([^"]+)"', CP):
        if href.startswith("http") or href.startswith("#"):
            continue
        assert (ROOT / "counterpoint" / href).resolve().exists(), href
    assert "https://jevresearch.github.io/Jev-Research/report/" in CP
    assert "archerhume.com/posts/jevs-architecture-unmasked" in CP


def test_main_top_link_points_at_counterpoint_absolute_url():
    assert 'href="https://jevresearch.github.io/Jev-Research/counterpoint/"' in HTML


def test_no_literal_fn_placeholders_anywhere():
    for text in (HTML, CP):
        assert "fn(" not in text and "KNOWLEDGE_QUESTIONS_LINK" not in text


def test_methods_appendix_within_word_budget():
    assert len(METHODS.split()) <= 1300
    assert "978 recorded requests (834 Jev + 144 reference-model)" in METHODS
