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
                  size: float = 11.0, fill: str = MUTED) -> tuple[str, int]:
    lines = textwrap.wrap(text, width)
    out = [f'<text x="{x}" y="{y + i * 14:.0f}" fill="{fill}" font-size="{size}">'
           f'{_html.escape(ln)}</text>' for i, ln in enumerate(lines)]
    return "".join(out), y + len(lines) * 14


UI_BLOCK = '''<style>
svg.isolate .mlabel{opacity:0;transition:opacity .12s}
svg.isolate g.hot .mlabel{opacity:1}
svg.isolate .jevlabel{opacity:1}
g.mrow{cursor:default}
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
})();
]]></script>'''


def era_legend(x: int, y: int) -> str:
    """Gradient strip + tier shapes + Jev diamond legend row."""
    p = [f'<text x="{x}" y="{y}" fill="{MUTED}" font-size="10.5" font-weight="600">'
         'color = release era</text>']
    lo_m, hi_m = ERA_MIN[0] * 12 + ERA_MIN[1], ERA_MAX[0] * 12 + ERA_MAX[1]
    gx = x + 108
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
    p.append(jev_diamond(sx + 4, y - 4, JEV_G, 5))
    p.append(f'<text x="{sx + 14}" y="{y}" fill="{MUTED}" font-size="9.5">Jev</text>')
    return "".join(p)
