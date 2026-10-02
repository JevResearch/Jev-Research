"""Author copyedit pass: chart encoding + report structure regressions.

Codifies the author's requested presentation decisions so they cannot drift:
  * reasoning markers are epistemic: solid dot (•) = thinking UNSPECIFIED,
    ring (⊙) = explicit NONE; a missing config is never tiered "none" or
    "high" by model brand;
  * the matched Qwen3.8 Max MMLU cell (v2 wire, effort low) is labeled LOW;
  * ARC-AGI-2 is ONE combined chart: official task rows + off-protocol Jev
    per-cell diagnostics, the latter italic-labeled and crosshatched;
  * pareto hit-target circles are fully transparent (no solid black discs);
  * the standalone latency section is merged into the architecture section as
    one subsection titled "What the milliseconds say", with #latency kept as
    an alias and the TOC renumbered/relaid.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "report"))

import chartkit                      # noqa: E402
import comparison_graphs as cg       # noqa: E402
import pareto_graphs as pg           # noqa: E402


# ------------------------------------------------- unknown • vs explicit ⊙
def test_unspecified_marker_is_bare_dot_none_is_ring():
    unknown = chartkit.symbol(10, 10, "unspecified", "#888888", 4.2)
    none = chartkit.symbol(10, 10, "none", "#888888", 4.2)
    assert 'r="1.8"' in unknown                 # the underlying dot
    assert "<circle" in unknown and unknown.count("<") == 1   # bare dot only
    assert 'fill="none"' in none                # ring drawn around the dot
    assert chartkit.TIER_LABEL["unspecified"] == "thinking unspecified"
    assert chartkit.TIER_TIP["unspecified"] == "thinking unspecified"
    assert chartkit.TIER_TIP["none"] == "none"


def test_vals_tier_missing_config_is_unspecified_not_none_or_high():
    silent = {"reasoning_effort": None, "compute_effort": None}
    assert chartkit.vals_tier("alibaba/qwen3.8-max", silent) == "unspecified"
    assert chartkit.vals_tier("anthropic/claude-fable-5", silent) == "unspecified"
    assert chartkit.vals_tier("x/y", {"reasoning_effort": "none"}) == "none"
    assert chartkit.vals_tier("x/y-non-reasoning", silent) == "none"
    assert chartkit.vals_tier("x/y", {"reasoning_effort": "high"}) == "high"
    assert chartkit.vals_tier("grok/grok-3-mini-fast-low-reasoning", silent) == "low"


# ------------------------------------------------------ matched wire effort
def test_matched_max_labels_low_and_defaults_are_unspecified():
    assert chartkit.matched_tier("qwen/qwen3.8-max-0902",
                                 "v2-temperature0-effort-low") == "low"
    # v3 provider defaults: effort unknown by name -> unspecified (never high)
    assert chartkit.matched_tier("z-ai/glm-5.3", "v3-provider-defaults") == "unspecified"
    assert chartkit.matched_tier("qwen/qwen3.8-flash", "v3-provider-defaults") == "unspecified"
    # documented no-effort models keep explicit none
    assert chartkit.matched_tier("google/gemma-3-4b-it", "v3-provider-defaults") == "none"
    assert chartkit.matched_tier("mistralai/mistral-small-3.2-24b-instruct",
                                 "v3-provider-defaults") == "none"
    rows = chartkit.matched_points(chartkit.load_matched(ROOT), "mmlu")
    by_id = {r["id"]: r for r in rows}
    assert by_id["qwen/qwen3.8-max-0902"]["tier"] == "low"
    assert by_id["qwen/qwen3.8-flash"]["tier"] == "unspecified"


# ----------------------------------------- ARC-AGI-2 combined chart + hatch
def test_arc_combined_chart_crosshatches_offprotocol_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(cg, "OUT", tmp_path / "cmp.html")
    assert cg.main() == 0
    cmp_h = (tmp_path / "cmp.html").read_text()
    assert len(re.findall(r'aria-label="ARC-AGI-2"', cmp_h)) == 1
    svg = re.search(r'<svg[^>]*aria-label="ARC-AGI-2"[^>]*>.*?</svg>', cmp_h, re.S).group(0)
    # single title, no subtitle lines under it
    assert "task level" not in svg and "per-cell" not in svg.split("data-tip")[0]
    assert 'font-size="11.0"' not in svg      # wrap_subtitle lines never render
    # official task rows and the diagnostic rows share the chart
    assert "Jev exact-grid" in svg and "Jev per-cell choice" in svg \
        and "Jev per-cell score" in svg
    # crosshatch: pattern ids unique per chart and referenced by the bar fills
    pids = re.findall(r'<pattern id="(xh-[^"]+)"', svg)
    assert len(pids) == 2 and len(set(pids)) == 2
    assert all(f'url(#{pid})' in svg for pid in pids)
    assert svg.count("patternUnits") == 2
    # off-protocol labels are italic (name + value for both per-cell rows)
    assert svg.count('font-style="italic"') == 4
    # no changelog-style defect prose in the chart or its captions
    assert "duplicate request-payload" not in cmp_h


# ------------------------------------------- pareto hit targets, no black
def test_pareto_hit_targets_are_transparent_not_black(tmp_path, monkeypatch):
    monkeypatch.setattr(pg, "OUT", tmp_path / "par.html")
    pg.main()
    par_h = (tmp_path / "par.html").read_text()
    assert ".fatten{fill:transparent;stroke:none}" in par_h   # global rule
    assert "g.ptrow .fatten" not in par_h                     # old narrow rule
    assert 'class="fatten"' in par_h                          # hit area kept
    assert "fill=\"black\"" not in par_h and "fill=\"#000" not in par_h
    # intentional Jev enhancements stay: teal halo on pareto, white bar ring
    assert 'r="8" fill="#38e1c8"' in par_h
    monkeypatch.setattr(cg, "OUT", tmp_path / "cmp.html")
    assert cg.main() == 0
    assert 'stroke="#ffffff"' in (tmp_path / "cmp.html").read_text()


# --------------------------- merged latency subsection, ids and TOC links
def test_one_merged_latency_subsection_ids_and_toc_links(tmp_path, monkeypatch):
    import build_report as br
    monkeypatch.setattr(br, "SITE", tmp_path)
    br.main()
    h = (tmp_path / "index.html").read_text()
    # exactly one subsection with the merged title, inside the architecture
    assert h.count('<h3 id="arch-millis">What the milliseconds say</h3>') == 1
    assert "<section id=\"latency\">" not in h           # no standalone section
    assert 'id="latency"' in h and 'id="arch-onepass"' in h   # aliases kept
    # TOC: the subsection is linked from the architecture block only, and the
    # remaining top-level sections are renumbered 4..8 with no latency entry
    toc = h[h.find('<nav class="toc">'):h.find('</nav>')]
    assert 'href="#arch-millis">What the milliseconds say</a>' in toc
    assert 'href="#latency"' not in toc
    assert 'href="#arch-onepass"' not in toc
    nums = re.findall(r'toc-sec"><a href="#[a-z]+">(\d+) &middot;', toc)
    assert nums == ["1", "2", "3", "4", "5", "6", "7", "8"]
    secs = re.findall(r'toc-sec"><a href="#([a-z]+)">', toc)
    assert secs == ["pitch", "method", "arch", "benchmarks", "pareto",
                    "probes", "keynotes", "conclusion", "refs"]
    # the old latency section text now lives in the merged subsection
    assert "self-batching" in h and "In short, what moves the measured upstream" in h
