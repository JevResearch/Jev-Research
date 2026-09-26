#!/usr/bin/env python3
"""Pareto frontier charts: score vs cost per question (log x-axis).

Data basis (revised 2026-09-26 after the audit): every external point is a
vals.ai platform row - accuracy AND cost_per_test measured by the vals
harness, which means any chain-of-thought a model burns is included in its
cost (it counts against the models that use it). Jev's points are measured
from our own billing: reported input tokens x $0.042/M / n questions; its
output is billed at zero.

This replaces the old list-price-estimate basis, whose calibrated guesses put
Jev and GLM-5.3-Flash at the same clamped x position on the GPQA chart -
nonsense that the audit caught. Measured per-test costs cannot produce that
artifact: Jev sits four to five orders of magnitude left of every vals row.

Bar color = release era (red 2022 -> blue 2026); marker shape = reasoning
tier (rounder = less thinking); underlying dot always drawn.

Output: docs/modern-comparison/pareto-frontiers.html
  python scripts/report/pareto_graphs.py
"""
from __future__ import annotations

import colorsys
import glob
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/modern-comparison/pareto-frontiers.html"
VALS = json.loads((ROOT / "docs/modern-comparison/canonical/vals-leaderboards-20260926.json").read_text())
COSTS = json.loads((ROOT / "data_report/costs.json").read_text())

BG = "#0e1117"; TEXT = "#e6e9f2"; MUTED = "#9aa2b6"; GRID = "#242a38"
JEV_G = "#38e1c8"; JEV_W = "#f5b342"
ERA_MIN, ERA_MAX = (2022, 11), (2026, 9)

TIER_SIDES = {"none": 0, "low": 6, "medium": 5, "high": 4, "xhigh": 3, "max": -1}
TIER_LABEL = {"none": "no reasoning", "low": "low", "medium": "medium",
              "high": "high", "xhigh": "very high", "max": "max"}


def _months(ym):
    y, m = ym.split("-")
    return int(y) * 12 + int(m)


def era_color(released):
    t = 0.5
    if released:
        lo, hi = _months("%d-%02d" % ERA_MIN), _months("%d-%02d" % ERA_MAX)
        t = max(0.0, min(1.0, (_months(released) - lo) / (hi - lo)))
    h = (4 + t * (228 - 4)) / 360.0
    r, g, b = colorsys.hls_to_rgb(h, 0.60, 0.62)
    return "#%02x%02x%02x" % (round(r * 255), round(g * 255), round(b * 255))


def symbol(cx, cy, tier, color, r=5.0):
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
        pts = []
        for i in range(sides):
            a = -math.pi / 2 + i * 2 * math.pi / sides
            pts.append(f"{cx + r * math.cos(a):.1f},{cy + r * math.sin(a):.1f}")
        out.append(f'<polygon points="{" ".join(pts)}" fill="none" '
                   f'stroke="{color}" stroke-width="1.4"/>')
    return "".join(out)


def jev_cost_per_question(stage_glob):
    hit = sorted(glob.glob(str(ROOT / stage_glob)))[0]
    d = json.loads(Path(hit).read_text())
    u = d["attempts_summary"]
    price = COSTS["prices_usd_per_M"]["Jev"]["input"]
    return u["usage_input_tokens_sum"] * price / 1e6 / u["n_attempts"]


CHARTS = [
    ("mmlu_pro", "MMLU-Pro: score vs cost per question",
     "vals.ai measured per-test costs (chain-of-thought included, so it counts "
     "against the models that use it); Jev from our billing: $0.042/M input, "
     "output free.",
     "runs_benchmark/bench-mmlu_full-*/derived/score.json", 82.8, 74.0),
    ("gpqa", "GPQA: score vs cost per question",
     "Same basis. Jev: Diamond subset, direct answers.",
     "runs_benchmark_ext2/bench-gpqa_diamond-*/derived/score.json", 76.5, 63.0),
    ("math500", "MATH-500: score vs cost per question",
     "vals rows solve free-form with reasoning; Jev's point is the 4-option "
     "MCQ adaptation - a protocol conversion, not a matched race.",
     "runs_benchmark_ext/bench-math500_choice-*/derived/score.json", 83.1, 69.5),
    ("hle", "Humanity's Last Exam: score vs cost per question",
     "vals rows: full text-only set. Jev: multiple-choice subset only - an "
     "easier format with a ~25% guessing floor; do not rank-read this chart.",
     "runs_benchmark_ext2/bench-hle_text_mc-*/derived/score.json", 21.9, 21.2),
]


def frontier(pts):
    """Upper envelope: sweep cost ascending, keep accuracy maxima."""
    pts = sorted(pts, key=lambda p: p[0])
    out = []
    best = -1
    for x, y, *_ in reversed(pts):
        if y > best:
            out.append((x, y))
            best = y
    return list(reversed(out))


def chart(bkey, title, subtitle, jev_glob, jev_g, jev_w):
    models = VALS["benchmarks"][bkey]["models"]
    pts = []
    for slug, m in models.items():
        if m.get("accuracy") is None or not m.get("cost_per_test"):
            continue
        if m["cost_per_test"] <= 0:
            continue
        pts.append((m["cost_per_test"], m["accuracy"], m["name"],
                    era_color(m.get("released")), m.get("tier", "none")))
    jc = jev_cost_per_question(jev_glob)
    W, H = 940, 560
    L, R, T, B = 74, 150, 96, 58
    xs = [p[0] for p in pts] + [jc]
    lo = math.floor(math.log10(min(xs))) - 0.3
    hi = math.ceil(math.log10(max(xs))) + 0.3
    ymax = max(max(p[1] for p in pts), jev_g) * 1.04

    def X(c):
        return L + (math.log10(c) - lo) / (hi - lo) * (W - L - R)

    def Y(v):
        return H - B - v / ymax * (H - T - B)

    p = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
         f'font-family="system-ui,sans-serif" role="img" aria-label="{title}">',
         f'<rect width="{W}" height="{H}" fill="{BG}" rx="14"/>',
         f'<text x="24" y="30" fill="{TEXT}" font-size="17" font-weight="700">{title}</text>',
         f'<text x="24" y="50" fill="{MUTED}" font-size="11">{subtitle}</text>']
    # legend line: era gradient + tier shapes
    lx = 24
    p.append(f'<text x="{lx}" y="72" fill="{MUTED}" font-size="10.5" font-weight="600">'
             'color = era</text>')
    lo_m, hi_m = 2022 * 12 + 11, 2026 * 12 + 9
    for i in range(40):
        ym = lo_m + int(i / 39 * (hi_m - lo_m))
        p.append(f'<rect x="{lx + 68 + i * 3}" y="64" width="3.4" height="9" '
                 f'fill="{era_color(f"{ym // 12}-{ym % 12:02d}")}"/>')
    p.append(f'<text x="{lx + 62}" y="72" fill="{MUTED}" font-size="9.5" text-anchor="end">2022</text>')
    p.append(f'<text x="{lx + 194}" y="72" fill="{MUTED}" font-size="9.5">2026</text>')
    sx = lx + 230
    p.append(f'<text x="{sx}" y="72" fill="{MUTED}" font-size="10.5" font-weight="600">shape = reasoning</text>')
    sx += 118
    for tier in ("none", "low", "medium", "high", "xhigh", "max"):
        p.append(symbol(sx, 68, tier, MUTED, 4.2))
        sx += 14
    # gridlines
    e0 = math.ceil(lo)
    e = e0
    while e <= hi:
        c = 10.0 ** e
        x = X(c)
        p.append(f'<line x1="{x:.0f}" y1="{T - 10}" x2="{x:.0f}" y2="{H - B}" stroke="{GRID}"/>')
        p.append(f'<text x="{x:.0f}" y="{H - B + 16}" fill="{MUTED}" font-size="10" '
                 f'text-anchor="middle">${c:g}</text>')
        e += 1
    gy = 0
    while gy <= ymax:
        y = Y(gy)
        p.append(f'<line x1="{L}" y1="{y:.0f}" x2="{W - R}" y2="{y:.0f}" stroke="{GRID}"/>')
        p.append(f'<text x="{L - 8}" y="{y + 3:.0f}" fill="{MUTED}" font-size="10" '
                 f'text-anchor="end">{gy:.0f}</text>')
        gy += 20
    # frontier polyline
    fr = frontier([(x, y) for x, y, *_ in pts])
    p.append('<polyline points="' + " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in fr)
             + f'" fill="none" stroke="{MUTED}" stroke-width="1.5" stroke-dasharray="4 4" opacity="0.75"/>')
    # points
    labeled = set()
    for x, y, name, color, tier in pts:
        p.append(symbol(X(x), Y(y), tier, color, 4.6))
    # label frontier vertices + extremes
    for x, y in fr:
        nm = next((n for xx, yy, n, _, _ in pts if abs(xx - x) < 1e-12 and yy == y), None)
        if nm and nm not in labeled:
            labeled.add(nm)
            p.append(f'<text x="{X(x) + 8:.0f}" y="{Y(y) - 6:.0f}" fill="{MUTED}" '
                     f'font-size="9.8">{nm}</text>')
    # Jev points
    p.append(f'<circle cx="{X(jc):.1f}" cy="{Y(jev_g):.1f}" r="6.5" fill="{JEV_G}" opacity="0.25"/>')
    p.append(f'<circle cx="{X(jc):.1f}" cy="{Y(jev_g):.1f}" r="4" fill="{JEV_G}"/>')
    p.append(f'<text x="{X(jc) + 9:.0f}" y="{Y(jev_g) + 4:.0f}" fill="{JEV_G}" '
             f'font-size="11" font-weight="600">Jev (greedy)</text>')
    p.append(f'<circle cx="{X(jc):.1f}" cy="{Y(jev_w):.1f}" r="3.2" fill="{JEV_W}"/>')
    p.append(f'<text x="{X(jc) + 9:.0f}" y="{Y(jev_w) + 4:.0f}" fill="{JEV_W}" '
             f'font-size="10">Jev (weighted)</text>')
    p.append(f'<text x="{(W - L - R) / 2 + L:.0f}" y="{H - 14}" fill="{MUTED}" '
             f'font-size="11" text-anchor="middle">cost per question (USD, log)</text>')
    p.append(f'<text x="18" y="{(H - T - B) / 2 + T:.0f}" fill="{MUTED}" font-size="11" '
             f'text-anchor="middle" transform="rotate(-90 18 {(H - T - B) / 2 + T:.0f})">score (%)</text>')
    p.append("</svg>")
    return "".join(p), jc


def main():
    charts = []
    summary = {}
    for bkey, title, subtitle, jglob, jg, jw in CHARTS:
        svg, jc = chart(bkey, title, subtitle, jglob, jg, jw)
        charts.append(svg)
        summary[bkey] = {"jev_cost_per_question_usd": round(jc, 8)}
        print(f"{bkey}: Jev ${jc:.6f}/question")
    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Jev Pareto frontiers</title><style>
 body {{ background:{BG}; color:{TEXT}; font-family: system-ui, sans-serif;
        max-width:1000px; margin:2rem auto; padding:0 1rem; line-height:1.5; }}
 h1 {{ font-size:24px; }}
 svg {{ width:100%; height:auto; display:block; margin:.4rem 0 1.4rem; }}
 p.lead {{ color:{MUTED}; font-size:14px; max-width:80ch; }}
</style></head><body>
<h1>The frontier the cost numbers actually draw</h1>
<p class="lead">Every external point is a vals.ai platform row: accuracy and
cost per test measured by the vals harness, chain-of-thought included in the
cost. Jev's points are measured from our billing ($0.042/M input tokens,
output free). Color is release era; marker shape is the row's reasoning
configuration. Dashed line: the cost/score frontier. Jev's per-question costs:
{json.dumps(summary)}.</p>
{''.join(charts)}
</body></html>
"""
    OUT.write_text(html, encoding="utf-8")
    print(f"[ok] wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size:,} bytes); 4 charts")


if __name__ == "__main__":
    main()
