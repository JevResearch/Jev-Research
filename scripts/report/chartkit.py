#!/usr/bin/env python3
"""Shared chart furniture for comparison_graphs.py and pareto_graphs.py.

Era gradient (audit pass 2): red (<= 2023) -> magenta -> purple -> blue
(2026). The hue route deliberately avoids green/teal so Jev's teal/amber can
never be mistaken for a year. Undated models render neutral gray and say so
in their tooltips.

Reasoning-tier symbols: rounder = less thinking, pointier = more:
circle none, hexagon low, pentagon medium, square high, triangle xhigh,
plus max; an underlying dot is always drawn. Jev gets a white-ringed
diamond, unique on every chart.

Tooltips/hover: every row is a <g class="mrow" data-tip="...">; one shared
script (UI_BLOCK, embedded in each SVG so it survives lifting into the
report) shows a floating tooltip. On charts marked data-isolate="1"
(pareto), hovering a point also hides every label except Jev's and the
hovered one.
"""
from __future__ import annotations

import colorsys
import html as _html
import json
import math
import textwrap

BG = "#0e1117"; PANEL = "#161b27"; TEXT = "#e6e9f2"; MUTED = "#9aa2b6"
GRID = "#242a38"; JEV_G = "#38e1c8"; JEV_W = "#f5b342"; UNK = "#7d8590"

ERA_MIN = (2023, 1)   # everything <= 2023 clamps to the reddest stop
ERA_MAX = (2026, 9)

TIER_SIDES = {"none": 0, "low": 6, "medium": 5, "high": 4, "xhigh": 3, "max": -1}
TIER_LABEL = {"none": "no reasoning", "low": "low reasoning",
              "medium": "medium reasoning", "high": "high reasoning",
              "xhigh": "very high reasoning", "max": "max reasoning"}


def _months(ym: str) -> int:
    y, m = ym.split("-")
    return int(y) * 12 + int(m)


def era_t(released: str | None) -> float | None:
    if not released:
        return None
    lo, hi = _months("%d-%02d" % ERA_MIN), _months("%d-%02d" % ERA_MAX)
    return max(0.0, min(1.0, (_months(released) - lo) / (hi - lo)))


def era_color(released: str | None) -> str:
    t = era_t(released)
    if t is None:
        return UNK
    # red (hue 4) -> blue (hue 228) the LONG way, through magenta/purple:
    # hue = 4 - 136t (mod 360). No green/teal anywhere on this route.
    h = ((4.0 - t * 136.0) % 360.0) / 360.0
    r, g, b = colorsys.hls_to_rgb(h, 0.62, 0.60)
    return "#%02x%02x%02x" % (round(r * 255), round(g * 255), round(b * 255))


def symbol(cx: float, cy: float, tier: str, color: str, r: float = 5.0) -> str:
    out = [f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="1.8" fill="{color}"/>']
    sides = TIER_SIDES.get(tier, 0)
    if sides == -1:
        out.append(f'<path d="M {cx-r:.1f} {cy:.1f} L {cx+r:.1f} {cy:.1f} '
                   f'M {cx:.1f} {cy-r:.1f} L {cx:.1f} {cy+r:.1f} '
                   f'M {cx-r*0.7:.1f} {cy-r*0.7:.1f} L {cx+r*0.7:.1f} {cy+r*0.7:.1f} '
                   f'M {cx-r*0.7:.1f} {cy+r*0.7:.1f} L {cx+r*0.7:.1f} {cy-r*0.7:.1f}" '
                   f'stroke="{color}" stroke-width="1.4" fill="none"/>')
    elif sides == 0:
        out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="none" '
                   f'stroke="{color}" stroke-width="1.4"/>')
    else:
        pts = " ".join(
            f"{cx + r * math.cos(-math.pi / 2 + i * 2 * math.pi / sides):.1f},"
            f"{cy + r * math.sin(-math.pi / 2 + i * 2 * math.pi / sides):.1f}"
            for i in range(sides))
        out.append(f'<polygon points="{pts}" fill="none" stroke="{color}" stroke-width="1.4"/>')
    return "".join(out)


def jev_diamond(cx: float, cy: float, color: str, r: float = 5.6) -> str:
    return (f'<polygon points="{cx:.1f},{cy-r:.1f} {cx+r:.1f},{cy:.1f} '
            f'{cx:.1f},{cy+r:.1f} {cx-r:.1f},{cy:.1f}" fill="{color}" '
            f'stroke="#ffffff" stroke-width="1.1"/>')


def tip(s: str) -> str:
    return 'data-tip="' + _html.escape(s, quote=True) + '"'


def wrap_subtitle(x: int, y: int, text: str, width: int = 118,
                  size: float = 11.0, fill: str = MUTED,
                  anchor: str = "start") -> tuple[str, int]:
    lines = textwrap.wrap(text, width)
    out = [f'<text x="{x}" y="{y + i * 14:.0f}" fill="{fill}" font-size="{size}" '
           f'text-anchor="{anchor}">{_html.escape(ln)}</text>'
           for i, ln in enumerate(lines)]
    return "".join(out), y + len(lines) * 14


UI_BLOCK = '''<style>
svg.isolate .mlabel{opacity:0;transition:opacity .12s}
svg.isolate g.hot .mlabel{opacity:1}
svg.isolate .jevlabel{opacity:1}
g.mrow{cursor:default}
g.ptrow .fatten{fill:transparent;stroke:none}
</style>
<script type="text/javascript"><![CDATA[
(function(){
 if (window.__jevUI) return; window.__jevUI = 1;
 var tip = document.createElement('div');
 tip.style.cssText = 'position:fixed;pointer-events:none;background:#161b27;border:1px solid #242a38;color:#e6e9f2;font:12px/1.55 system-ui,sans-serif;padding:8px 11px;border-radius:9px;max-width:330px;z-index:999;display:none;box-shadow:0 6px 18px rgba(0,0,0,.55)';
 document.addEventListener('DOMContentLoaded', function(){ document.body.appendChild(tip); });
 if (document.body && !tip.parentNode) document.body.appendChild(tip);
 document.addEventListener('mousemove', function(e){
   var g = e.target && e.target.closest ? e.target.closest('g[data-tip]') : null;
   if (g) {
     tip.innerHTML = g.getAttribute('data-tip');
     tip.style.display = 'block';
     var x = e.clientX + 16, y = e.clientY + 16;
     if (x + 340 > window.innerWidth) x = Math.max(8, e.clientX - 345);
     if (y + 140 > window.innerHeight) y = Math.max(8, e.clientY - 145);
     tip.style.left = x + 'px'; tip.style.top = y + 'px';
     var svg = g.ownerSVGElement;
     if (svg && svg.getAttribute('data-isolate') === '1') {
       svg.classList.add('isolate');
       var rows = svg.querySelectorAll('g.isorow');
       for (var i = 0; i < rows.length; i++) rows[i].classList.toggle('hot', rows[i] === g);
     }
   } else {
     tip.style.display = 'none';
     var ss = document.querySelectorAll('svg.isolate');
     for (var j = 0; j < ss.length; j++) {
       ss[j].classList.remove('isolate');
       var hot = ss[j].querySelectorAll('g.hot');
       for (var k = 0; k < hot.length; k++) hot[k].classList.remove('hot');
     }
   }
 });
 // keyboard + touch parity: every tooltip row is focusable and tappable
 function place(g){
   var r = g.getBoundingClientRect();
   tip.innerHTML = g.getAttribute('data-tip');
   tip.style.display = 'block';
   var x = r.right + 8, y = r.top;
   if (x + 340 > window.innerWidth) x = Math.max(8, r.left - 348);
   if (y + 140 > window.innerHeight) y = Math.max(8, window.innerHeight - 150);
   tip.style.left = x + 'px'; tip.style.top = y + 'px';
   var svg = g.ownerSVGElement;
   if (svg && svg.getAttribute('data-isolate') === '1') {
     svg.classList.add('isolate');
     var rr = svg.querySelectorAll('g.isorow');
     for (var i = 0; i < rr.length; i++) rr[i].classList.toggle('hot', rr[i] === g);
   }
 }
 function arm(){
   var rr = document.querySelectorAll('g[data-tip]');
   for (var i = 0; i < rr.length; i++) {
     if (rr[i].getAttribute('tabindex') === null) {
       rr[i].setAttribute('tabindex', '0');
       rr[i].setAttribute('role', 'img');
       rr[i].setAttribute('aria-label',
         rr[i].getAttribute('data-tip').replace(/<br[^>]*>/gi, '; ').replace(/<[^>]+>/g, ''));
     }
   }
 }
 if (document.readyState !== 'loading') arm();
 document.addEventListener('DOMContentLoaded', arm);
 document.addEventListener('focusin', function(e){
   var g = e.target && e.target.closest ? e.target.closest('g[data-tip]') : null;
   if (g) place(g);
 });
 document.addEventListener('focusout', function(){ tip.style.display = 'none'; });
 document.addEventListener('click', function(e){
   var g = e.target && e.target.closest ? e.target.closest('g[data-tip]') : null;
   if (g) { place(g); }
 });
})();
]]></script>'''


def era_legend(x: int, y: int) -> str:
    """Gradient strip + tier shapes + Jev diamond legend row."""
    p = []
    lo_m, hi_m = ERA_MIN[0] * 12 + ERA_MIN[1], ERA_MAX[0] * 12 + ERA_MAX[1]
    gx = x + 46
    for i in range(44):
        ym = lo_m + int(i / 43 * (hi_m - lo_m))
        rel = f"{ym // 12}-{ym % 12:02d}"
        p.append(f'<rect x="{gx + i * 3}" y="{y - 9}" width="3.4" height="9" fill="{era_color(rel)}"/>')
    p.append(f'<text x="{gx - 6}" y="{y}" fill="{MUTED}" font-size="9.5" text-anchor="end">&le;2023</text>')
    p.append(f'<text x="{gx + 136}" y="{y}" fill="{MUTED}" font-size="9.5">2026</text>')
    p.append(f'<rect x="{gx + 168}" y="{y - 9}" width="10" height="9" fill="{UNK}"/>')
    p.append(f'<text x="{gx + 182}" y="{y}" fill="{MUTED}" font-size="9.5">date n/a</text>')
    sx = gx + 240
    p.append(f'<text x="{sx}" y="{y}" fill="{MUTED}" font-size="10.5" font-weight="600">shape = reasoning</text>')
    sx += 122
    for tier in ("none", "low", "medium", "high", "xhigh", "max"):
        p.append(symbol(sx, y - 4, tier, MUTED, 4.2))
        sx += 13
    p.append(f'<text x="{sx + 14}" y="{y}" fill="{MUTED}" font-size="9.5">&dagger; our '
             'matched run</text>')
    return "".join(p)


# ---------------------------------------------------------------- matched runs
# Cheap-model matched baselines (runs_matched_cheap/, OpenRouter, identical
# frozen items, direct answers, reasoning_effort=low where supported,
# strict parsing, format failures counted as wrong).
CHEAP_META = {
    "meta-llama/llama-3.1-8b-instruct": ("Llama 3.1 8B", "2024-07", "none", "documented"),
    "mistralai/mistral-nemo": ("Mistral Nemo", "2024-07", "none", "documented"),
    "openai/gpt-oss-20b": ("gpt-oss-20b", "2025-08", "medium", "documented"),
    "openai/gpt-oss-120b": ("gpt-oss-120b", "2025-08", "medium", "documented"),
    "ibm-granite/granite-4.0-h-micro": ("Granite 4.0 Micro", "2025-06", "medium", "estimated"),
    "google/gemma-3-4b-it": ("Gemma 3 4B", "2025-03", "none", "documented"),
    "qwen/qwen3.7-flash": ("Qwen3.7 Flash", "2026-06", "high", "estimated"),
    "qwen/qwen3.7-flash@think": ("Qwen3.7 Flash (thinking)", "2026-06", "low", "estimated"),
    "qwen/qwen3.8-max-0902": ("Qwen3.8 Max", "2026-08", "high", "documented"),
    "xiaomi/mimo-v2.6-pro": ("MiMo v2.6 Pro", "2026-05", "high", "documented"),
    "xiaomi/mimo-v2.6-flash": ("MiMo v2.6 Flash", "2026-05", "high", "documented"),
    "mistralai/mistral-small-3.2-24b-instruct": ("Mistral Small 3.2", "2025-09", "none", "estimated"),
    "z-ai/glm-5.3-flash": ("GLM-5.3 Flash", "2026-08", "high", "documented"),
    "qwen/qwen3.8-flash": ("Qwen3.8 Flash", "2026-08", "high", "estimated"),
    "qwen/qwen3.8-flash@think": ("Qwen3.8 Flash (thinking)", "2026-08", "low", "estimated"),
    "deepseek/deepseek-v4-flash-0731": ("DeepSeek V4 Flash", "2025-07", "high", "documented"),
    "z-ai/glm-5.3": ("GLM-5.3", "2026-08", "max", "documented"),
}


def exactly_one(root, pattern: str) -> Path:
    """Require exactly one glob hit (no arbitrary first-match selection)."""
    hits = sorted(root.glob(pattern))
    if len(hits) != 1:
        raise RuntimeError(
            f"expected exactly 1 hit for {pattern!r}, found {len(hits)}: "
            f"{[str(h) for h in hits]}")
    return hits[0]


def load_matched(root) -> dict:
    """The SOLE authoritative matched-baseline source: the v4r1 active summary
    (data_report/baselines/v4r1/public_summary.json), whose cells carry
    all-requested accuracy over full sampled denominators, complete/partial
    status and paired Jev metrics. Missing artifacts FAIL CLOSED; there is no
    legacy v2/v3 summary fallback (those summaries are unrepaired legacy and
    are never presented as current)."""
    for need in ("data_report/baselines/v4r1/public_summary.json",
                 "data_report/baselines/v4r1/active_source_map.json"):
        p = root / need
        if not p.exists():
            raise FileNotFoundError(
                f"authoritative baseline artifact missing (fail closed, no "
                f"legacy fallback): {need}")
    return json.loads(
        (root / "data_report/baselines/v4r1/public_summary.json").read_text())


def load_jev_scores(root) -> dict:
    """Jev's own scores, read from the regenerated score.json artifacts through
    the recorded source selectors (active_source_map.json). Nothing is
    hardcoded; greedy is the ALL-REQUESTED headline accuracy and weighted is
    the mean displayed probability on the gold option (descriptive only)."""
    amap = json.loads(
        (root / "data_report/baselines/v4r1/active_source_map.json").read_text())
    sel = amap["sources"]["jev_scores"]

    def one(pattern: str) -> dict:
        return json.loads(exactly_one(root, pattern).read_text())

    def stage(pattern: str) -> dict:
        s = one(pattern)
        w = one(pattern.replace("score.json", "weighted_score.json"))
        return {"greedy": s["accuracy"],
                "weighted_mean_p_gold": w["weighted_accuracy_mean"],
                "n": s.get("n_requested") or s.get("n_expected")}

    out = {
        "mmlu_pro": stage(sel["mmlu"]),
        "arc_challenge": stage(sel["arc"]),
        "gpqa_diamond": stage(sel["gpqa"]),
        "math500_mcq_adapted": stage(sel["math500_choice"]),
        "hle_text_only_mc": stage(sel["hle_text_mc"]),
        "math500_digit_readout": stage(
            "runs_benchmark_ext/bench-math500_score-*/derived/score.json"),
    }
    corr = amap["sources"]["arc_corrected"]
    ch = one(corr.replace("{choice,score}", "choice") + "/derived/score.json")
    sc = one(corr.replace("{choice,score}", "score") + "/derived/score.json")
    tk = one("runs_benchmark_ext/bench-arc_agi2_task-*/derived/score.json")
    out["arc_agi2_public_eval"] = {
        "cell_accuracy_choice": ch["cell_accuracy_diagnostic"],
        "cell_accuracy_score": sc["cell_accuracy_diagnostic"],
        "cells_total": ch["cells_total"],
        "exact_grid": tk["task_accuracy"],
        "n_tasks": tk.get("n_tasks_expected") or tk.get("n_tasks"),
    }
    return out


def paired_mmlu_matched(root) -> dict:
    """The paired matched-subset MMLU figure (repaired diagnostics)."""
    d = json.loads(
        (root / "data_report/benchmark_diagnostics/paired_diagnostics.json").read_text())
    return d["mmlu_matched_subset"]


def rotation_diagnostics(root) -> dict:
    """Cluster-aware option-rotation diagnostics (repaired diagnostics)."""
    d = json.loads(
        (root / "data_report/benchmark_diagnostics/paired_diagnostics.json").read_text())
    return d["rotation_clusters"]


def matched_points(summary: dict, ds: str) -> list[dict]:
    """Chart-ready rows from the v4r1 active summary cells:
    {id,name,short,released,tier,basis,accuracy,cost,n,...}.

    Contract (regression-tested in tests/test_report_chart_sources.py):
      * only cells with status == 'complete' are charted; partial cells are
        skipped even when their original rows looked complete, and
        partial_settled_accuracy is NEVER used;
      * accuracy is the all-requested accuracy_all_requested over the FULL
        sampled denominator (n_requested);
      * cost is the current definition: provider-reported original +
        replacement spend over the same denominator;
      * wire = the cell's original wire protocol (v2 carryovers visible);
      * jev_join carries the repaired paired Jev metric
        (jev_correct_on_subset / n_paired)."""
    out = []
    for key, e in (summary.get("cells") or {}).items():
        if e.get("dataset") != ds:
            continue
        if e.get("status") != "complete" or not e.get("complete"):
            continue                       # 17 partial cells: never charted
        acc = e.get("accuracy_all_requested")
        if acc is None:
            continue                       # fail closed; never legacy values
        mid = e["model"]
        short, rel, tier, basis = CHEAP_META.get(
            mid, (mid.split("/")[-1], None, "none", None))
        cost = e.get("cost") or {}
        usd = ((cost.get("original_reported_usd") or 0.0)
               + (cost.get("replacement_reported_usd") or 0.0))
        n = e["n_requested"]
        jp = e.get("jev_paired") or {}
        n_pair = jp.get("n_paired") or 0
        unre = (e.get("effects") or {}).get("unrecovered_settled", 0)
        out.append({
            "id": mid, "name": short, "short": short, "released": rel,
            "tier": tier, "date_basis": basis,
            "accuracy": acc * 100,
            "cost": usd / n if n else None,
            "n": n,
            "n_correct": e.get("n_correct"),
            "fmt_pct": 100 * unre / max(n, 1),
            "jev_join": ({"jev_accuracy_on_subset":
                          jp["jev_correct_on_subset"] / n_pair}
                         if n_pair else None),
            "wire": e.get("original_wire_protocol"),
            "cell": key,
        })
    out.sort(key=lambda r: -r["accuracy"])
    return out
