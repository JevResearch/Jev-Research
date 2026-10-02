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


# =================== final readability pass (author's authorized 6 edits) ===
import math                                   # noqa: E402

import pytest                                 # noqa: E402


def _pareto_svgs(par_h):
    out = {}
    for m in re.finditer(r'<svg[^>]*aria-label="([^"]+)"[^>]*>.*?</svg>',
                         par_h, re.S):
        out[m.group(1).split(":")[0].lower().replace("-", "_")] = m.group(0)
    return out


def test_pareto_y_floor_same_height_and_point_math(tmp_path, monkeypatch):
    """Y-floor 30/25, both charts at the same 1280x1025, and the rebased
    point math: grid, marks and the Jev halo share one transform; X is
    untouched (log-cost marks/labels stay accurate)."""
    monkeypatch.setattr(pg, "OUT", tmp_path / "par.html")
    pg.main()
    par_h = (tmp_path / "par.html").read_text()
    assert pg.SVG_W == 1280 and pg.SVG_H == 1025        # 820 -> 1025 (+25%)
    assert par_h.count('viewBox="0 0 1280 1025"') == 2  # same size for both
    assert pg.Y_FLOOR == {"mmlu_pro": 30.0, "gpqa": 25.0}
    svgs = _pareto_svgs(par_h)
    for bkey, _title, _sub, jglob in pg.CHARTS:
        svg = svgs[bkey]
        yfloor = pg.Y_FLOOR[bkey]
        # rendered ticks: the floor tick label (30 / 25) is visible, first
        ticks = [(int(gy), int(y) - 4) for y, gy in re.findall(
            r'<text x="80" y="(\d+)" fill="#9aa2b6" font-size="12" '
            r'text-anchor="end">(\d+)</text>', svg)]
        assert ticks[0] == (yfloor, 957)          # floor label at H - B
        assert all(0 <= y <= 957 for _, y in ticks)
        (g0, y0), (g1, y1) = ticks[0], ticks[-1]
        k = (y0 - y1) / (g1 - g0)

        def Y(v):
            return y0 - (v - g0) * k
        # rebased mapping: floor at the axis edge, monotone, never clamped
        assert Y(yfloor) == 957 == pg.SVG_H - 68
        assert Y(yfloor - 10) > 957
        # point math: the Jev halo sits at exactly (X(jc), Y(jev_g))
        halo = re.search(r'<circle cx="([\d.]+)" cy="([\d.]+)" r="8" '
                         r'fill="#38e1c8"', svg)
        cx, cy = float(halo.group(1)), float(halo.group(2))
        js = pg.JEV_SCORES[pg._JEV_KEY[bkey]]
        assert abs(cy - Y(js["greedy"] * 100)) <= 1.0
        # X unchanged: read the rendered log-cost axis, check the same point
        xlab = [(int(x), float(c)) for x, c in re.findall(
            r'<text x="(\d+)" y="975" fill="#9aa2b6" font-size="12" '
            r'text-anchor="middle">\$([0-9.]+)</text>', svg)]
        (xa, ca), (xb, cb) = xlab[0], xlab[-1]
        b = (xa - xb) / (math.log10(ca) - math.log10(cb))
        a = xa - b * math.log10(ca)
        assert abs(cx - (a + b * math.log10(
            pg.jev_cost_per_question(jglob)))) <= 1.0


def test_pareto_below_floor_data_clipped_not_clamped(tmp_path, monkeypatch):
    monkeypatch.setattr(pg, "OUT", tmp_path / "par.html")
    pg.main()
    svgs = _pareto_svgs((tmp_path / "par.html").read_text())
    for bkey, svg in svgs.items():
        yf = pg.Y_FLOOR[bkey]
        # clip rect is exactly the plot band, ending on the floor line
        rect = re.search(r'<clipPath id="plotclip-[^"]+">'
                         r'<rect x="88" y="(\d+)" width="1144" height="(\d+)"',
                         svg)
        top, hgt = int(rect.group(1)), int(rect.group(2))
        assert top + hgt == 957 == pg.SVG_H - 68
        # marks and the frontier projection render inside the clipped group
        g_open = svg.index('<g clip-path=')
        g_close = svg.index('</g></g>')
        assert g_open < svg.index('<polyline') < g_close
        assert g_open < svg.index('class="mrow isorow"') < g_close
        ys = [float(y) for y in re.findall(
            r'<polyline points="([^"]+)"', svg)[0].replace(",", " ").split()[1::2]]
        assert max(ys) < pg.SVG_H          # never a giant line across canvas
        # below-floor marks are CLIPPED at the edge and keep their true values
        # (clamped marks would sit on the axis line at cy == 957.0)
        below = re.findall(
            r'<circle class="fatten" cx="[\d.]+" cy="([\d.]+)" r="14"/>', svg)
        edge = [cy for cy in below if float(cy) == 957.0]
        assert not edge, "no mark may be clamped onto the floor line"
        hidden = [float(cy) for cy in below if float(cy) > 957.0]
        rows = re.findall(r'<g class="mrow isorow" data-tip="([^"]*)"', svg)
        scores = [float(m.group(1)) for r in rows
                  for m in [re.search(r"score ([\d.]+)%", r)] if m]
        if bkey == "gpqa":     # current data: one matched row at 23.0% < 25
            assert hidden, "below-floor row must exist and be clipped, not dropped"
            assert min(scores) < yf, "its recorded score stays true (no clamp)"
        else:                  # MMLU-Pro: nothing below the 30 floor today
            assert not hidden
            assert min(scores) >= yf
        # values unchanged: no tooltip ever reports the floor as the score
        assert not [s for s in scores if s == yf and s != min(scores)]


def test_shared_legend_group_padding_and_fit():
    """Metric layout: symbol LEFT of label, balanced padding, 16-20px group
    margins, and the row (dagger label included) fits every chart width."""
    lay = chartkit.tier_legend_layout(0.0)
    assert [g["label"] for g in lay] == [
        "Unspecified", "None", "Low \u2192 Max", "\u2020 our matched run"]
    for g in lay[:3]:                      # symbol left, balanced padding
        assert g["symbols"][-1] < g["label_x"]
        assert abs(g["label_x"] - (g["symbols"][-1] + chartkit.LEGEND_SYM_R)
                   - chartkit.LEGEND_PAD) <= 0.01
    for a, b in zip(lay, lay[1:]):         # group margins ~16-20px, no overlap
        gap = b["x"] - (a["x"] + a["w"])
        assert 16 <= gap <= 20
        assert b["x"] >= a["x"] + a["w"]
    # fit at the actual call sites: pareto 1280-wide and benchmark 1180-wide
    assert chartkit.legend_end_x((88 + 1280 - 48) // 2 - 300) <= 1280 - 24
    assert chartkit.legend_end_x((292 + 1180 - 128) // 2 - 300) <= 1180 - 24


def test_shared_legend_renders_computed_positions(tmp_path, monkeypatch):
    monkeypatch.setattr(pg, "OUT", tmp_path / "par.html")
    monkeypatch.setattr(cg, "OUT", tmp_path / "cmp.html")
    assert pg.main() is None and cg.main() == 0
    for f in ("par.html", "cmp.html"):
        h = (tmp_path / f).read_text()
        assert "shape = thinking" in h
        for g in chartkit.tier_legend_layout(
                chartkit._tier_legend_start(360 if f == "par.html" else 372)):
            label = (g["label"].replace("\u2192", "&rarr;")
                     .replace("\u2020", "&dagger;"))
            assert f'<text x="{g["label_x"]:.1f}" ' in h
            assert f'>{label}</text>' in h


def test_pareto_unknown_dot_radius_factor_only_on_pareto_points(
        tmp_path, monkeypatch):
    """The +60% thinking-unspecified dot is a circle radius on the pareto
    plot points only; the shared legend dot and every other marker keep 1.8."""
    assert pg.UNK_DOT_R == pytest.approx(1.8 * 1.6)
    base = chartkit.symbol(0, 0, "unspecified", "#888888", 4.2)
    big = chartkit.symbol(0, 0, "unspecified", "#888888", 4.2,
                          unk_r=pg.UNK_DOT_R)
    assert 'r="1.8"' in base and "<circle" in base
    assert 'r="2.88"' in big
    none = chartkit.symbol(0, 0, "none", "#888888", 4.2, unk_r=pg.UNK_DOT_R)
    assert 'r="1.8"' in none and 'r="2.88"' not in none   # ring center kept
    monkeypatch.setattr(pg, "OUT", tmp_path / "par.html")
    monkeypatch.setattr(cg, "OUT", tmp_path / "cmp.html")
    pg.main()
    assert cg.main() == 0
    par_h = (tmp_path / "par.html").read_text()
    cmp_h = (tmp_path / "cmp.html").read_text()
    assert 'r="2.88"' in par_h              # pareto actual points
    assert 'r="2.88"' not in cmp_h          # not the benchmark chart/legend
    assert 'r="1.8"' in cmp_h               # shared legend dot unchanged
    assert 'fill="black"' not in par_h and 'fill="#000' not in par_h
