#!/usr/bin/env python3
"""Pareto frontier charts: score vs measured cost per question.

Only the two benchmarks whose protocols line up honestly are charted:
MMLU-Pro and GPQA (Jev answers both natively, in format, direct; the vals
rows answer the same items with the platform harness). HLE and MATH-500 are
deliberately absent - Jev's rows there are a multiple-choice subset and an
MCQ conversion, and no fair cost/score visual exists against free-form or
full-set references. Their numbers live in the report table with caveats.

Every external point is a vals.ai platform row: accuracy AND cost_per_test
measured by the same harness - so any chain-of-thought a model burns is
included in its cost (it counts against the models that use it). Jev's points
are measured from our own billing: reported input tokens x $0.042/M divided
by questions; its output is billed at zero.

The frontier is the true Pareto envelope: sweep cost ascending and keep each
point that is more accurate than everything CHEAPER than it - the polyline
therefore runs from the upper right DOWN TO THE LEFT (cheaper-and-worse at
each step down), which is the direction a cost/score frontier must have.

All models are labeled (small type, every point); hovering a point hides all
labels except Jev's and the hovered one and shows a tooltip with name, score
+ rank, cost + rank, release date, reasoning tier. Jev: white-ringed
diamonds.

Output: docs/modern-comparison/pareto-frontiers.html
  python scripts/report/pareto_graphs.py
"""
from __future__ import annotations

import glob
import html as _html
import json
import math
from pathlib import Path

from chartkit import (BG, GRID, JEV_G, JEV_W, MUTED, TEXT, UI_BLOCK, UNK,
                      TIER_TIP, exactly_one, load_matched, load_jev_scores,
                      matched_points, vals_tier,
                      TIER_LABEL, era_color, era_legend, jev_diamond, symbol,
                      tip, wrap_subtitle)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/modern-comparison/pareto-frontiers.html"
VALS = json.loads((ROOT / "docs/modern-comparison/canonical/vals-leaderboards-20260926.json").read_text())
COSTS = json.loads((ROOT / "data_report/costs.json").read_text())


def jev_cost_per_question(stage_glob):
    d = json.loads(Path(exactly_one(ROOT, stage_glob)).read_text())
    u = d["attempts_summary"]
    price = COSTS["prices_usd_per_M"]["Jev"]["input"]
    return u["usage_input_tokens_sum"] * price / 1e6 / u["n_attempts"]


CHARTS = [
    ("mmlu_pro", "MMLU-Pro: score vs measured cost per question",
     "Vals.ai platform measurements (chain-of-thought included in the cost, so reasoning "
     "counts against the models that use it), plus our own matched runs (dagger); "
     "Jev from our billing: $0.042/M input tokens, "
     "output free. Hover a point to isolate it.",
     "runs_benchmark/bench-mmlu_full-*/derived/score.json"),
    ("gpqa", "GPQA: score vs measured cost per question",
     "Same basis. Jev: Diamond subset (196 items), direct answers, seeded option shuffle.",
     "runs_benchmark_ext2/bench-gpqa_diamond-*/derived/score.json"),
]

# Authoritative Jev scores (regenerated score artifacts via recorded selectors):
# nothing about Jev is hardcoded in this chart code.
JEV_SCORES = load_jev_scores(ROOT)
_JEV_KEY = {"mmlu_pro": "mmlu_pro", "gpqa": "gpqa_diamond"}


def frontier(pts):
    """Cost/accuracy envelope of this mixed view: cost ascending, keep accuracy
    record-breakers. A point survives iff no cheaper point is at least as
    accurate - i.e. each kept point beats the running max of everything
    cheaper. The result rises to the right (falls to the left), the correct
    frontier direction. Positioning only: the rows mix publisher protocols,
    our protocol conversions and our matched runs.
    """
    out = []
    best = -1.0
    for x, y in sorted(pts, key=lambda p: p[0]):
        if y > best:
            out.append((x, y))
            best = y
    return out


def _rank(v, vals, reverse=False):
    # rank of v among vals (1 = best); reverse=True -> smaller is better
    if reverse:
        return 1 + sum(1 for w in vals if w < v)
    return 1 + sum(1 for w in vals if w > v)


def chart(bkey, title, subtitle, jev_glob, jev_g, jev_w, n_items):
    models = VALS["benchmarks"][bkey]["models"]
    pts = []
    for slug, m in models.items():
        if m.get("accuracy") is None or not m.get("cost_per_test"):
            continue
        if m["cost_per_test"] <= 0:
            continue
        pts.append((m["cost_per_test"], m["accuracy"],
                    {**m, "slug": slug, "tier": vals_tier(slug, m)}))
    scores = [a for _, a, _ in pts]
    costs = [c for c, _, _ in pts]
    jc = jev_cost_per_question(jev_glob)
    # matched cheap-model baselines (this study): identical items, measured cost
    _ds = {"mmlu_pro": "mmlu", "gpqa": "gpqa"}.get(bkey, bkey)
    for r in matched_points(load_matched(ROOT), _ds):
        if not r.get("cost"):
            continue
        mm = dict(r)
        mm["name"] = r["name"] + " \u2020"
        pts.append((r["cost"], r["accuracy"], mm))
    n_all = len(pts)

    W, H = 1280, 820
    L, R, B = 88, 48, 68
    xs = costs + [jc]
    lo = math.floor(math.log10(min(xs))) - 0.35
    hi = math.ceil(math.log10(max(xs))) + 0.35
    ymax = max(max(scores), jev_g) * 1.06

    def X(c):
        return L + (math.log10(c) - lo) / (hi - lo) * (W - L - R)

    def Y(v):
        return H - B - v / ymax * (H - top - B)

    sub, sub_end = wrap_subtitle((L + W - R) // 2, 54, subtitle, width=175, size=12, anchor="middle")
    top = sub_end + 54
    p = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
         f'font-family="system-ui,sans-serif" role="img" data-isolate="1" '
         f'aria-label="{_html.escape(title, quote=True)}">',
         f'<rect width="{W}" height="{H}" fill="{BG}" rx="14"/>',
         f'<text x="{(L + W - R) // 2}" y="32" fill="{TEXT}" font-size="20" '
         f'font-weight="700" text-anchor="middle">'
         f'{_html.escape(title)}</text>',
         sub, era_legend((L + W - R) // 2 - 300, sub_end + 26)]
    # gridlines
    e = math.ceil(lo)
    while e <= hi:
        c = 10.0 ** e
        x = X(c)
        p.append(f'<line x1="{x:.0f}" y1="{top}" x2="{x:.0f}" y2="{H - B}" stroke="{GRID}"/>')
        p.append(f'<text x="{x:.0f}" y="{H - B + 18}" fill="{MUTED}" font-size="12" '
                 f'text-anchor="middle">${c:g}</text>')
        e += 1
    gy = 0
    while gy <= ymax:
        y = Y(gy)
        p.append(f'<line x1="{L}" y1="{y:.0f}" x2="{W - R}" y2="{y:.0f}" stroke="{GRID}"/>')
        p.append(f'<text x="{L - 8}" y="{y + 4:.0f}" fill="{MUTED}" font-size="12" '
                 f'text-anchor="end">{gy:.0f}</text>')
        gy += 20
    # frontier
    fr = frontier([(c, a) for c, a, _ in pts] + [(jc, jev_g)])
    p.append('<polyline points="' + " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in fr)
             + f'" fill="none" stroke="{MUTED}" stroke-width="1.6" '
             f'stroke-dasharray="6 5" opacity="0.85"/>')
    # model points, ALL labeled
    scores = [a for _, a, _ in pts]
    costs = [c for c, _, _ in pts]
    for c, a, m in pts:
        color = era_color(m.get("released"))
        tier = m.get("tier", "unspecified")
        sr = _rank(a, scores)
        cr = _rank(c, costs, reverse=True)
        rel = m.get("released") or "unknown"
        basis = m.get("date_basis") or "n/a"
        _tl = TIER_TIP.get(tier, "thinking unspecified")
        if m.get("jev_join") is not None or m.get("fmt_pct") is not None:
            tt = (f"<b>{_html.escape(m['name'])} ({_tl})</b><br>"
                  f"score {a:.1f}% &mdash; #{sr} of {n_all}<br>"
                  f"cost ${c:.6f}/question &mdash; #{cr} of {n_all}<br>"
                  f"released {rel}" + (" (estimated)" if basis == "estimated" else ""))
        else:
            tt = (f"<b>{_html.escape(m['name'])} ({_tl})</b><br>"
                  f"score {a:.1f}% &mdash; #{sr} of {n_all}<br>"
                  f"cost ${c:.5f}/question &mdash; #{cr} of {n_all}<br>"
                  f"released {rel}" + (" (estimated)" if basis == "estimated" else ""))
        p.append(f'<g class="mrow isorow" {tip(tt)}>')
        p.append(f'<circle class="fatten" cx="{X(c):.1f}" cy="{Y(a):.1f}" r="14"/>')
        p.append(symbol(X(c), Y(a), tier, color, 5.2))
        p.append(f'<text class="mlabel" x="{X(c) + 9:.1f}" y="{Y(a) - 7:.1f}" '
                 f'fill="{color}" font-size="8.4" opacity="0.92">'
                 f'{_html.escape(m["name"])}</text>')
        p.append('</g>')
    # Jev points
    jsr = _rank(jev_g, scores + [jev_g])
    jcr = _rank(jc, costs + [jc], reverse=True)
    jtt_g = (f"<b>Jev, greedy (this study)</b><br>score {jev_g:.1f}% &mdash; "
             f"#{jsr} of {n_all + 1} here<br>cost ${jc:.8f}/question &mdash; "
             f"#{jcr} of {n_all + 1}<br>measured billing: "
             f"${jc * n_items:.4f} for the whole {n_items}-question run")
    jtt_w = (f"<b>Jev, probability-weighted (this study)</b><br>mean probability on the "
             f"gold option: {jev_w:.1f}%<br>same cost: ${jc:.8f}/question")
    p.append(f'<g class="mrow isorow" {tip(jtt_g)}>')
    p.append(f'<circle cx="{X(jc):.1f}" cy="{Y(jev_g):.1f}" r="8" fill="{JEV_G}" opacity="0.22"/>')
    p.append(symbol(X(jc), Y(jev_g), "none", JEV_G, 5.2))
    p.append(f'<text class="mlabel jevlabel" x="{X(jc) + 11:.1f}" y="{Y(jev_g) + 4:.1f}" '
             f'fill="{JEV_G}" font-size="11" font-weight="700">Jev (greedy)</text>')
    p.append('</g>')
    p.append(f'<g class="mrow isorow" {tip(jtt_w)}>')
    p.append(symbol(X(jc), Y(jev_w), "none", JEV_W, 4.4))
    p.append(f'<text class="mlabel jevlabel" x="{X(jc) + 11:.1f}" y="{Y(jev_w) + 4:.1f}" '
             f'fill="{JEV_W}" font-size="10">Jev (weighted)</text>')
    p.append('</g>')
    p.append(f'<text x="{(W - L - R) / 2 + L:.0f}" y="{H - 16}" fill="{MUTED}" '
             f'font-size="13" text-anchor="middle">Cost per question (USD, log scale)</text>')
    p.append(f'<text x="20" y="{(H - top - B) / 2 + top:.0f}" fill="{MUTED}" font-size="13" '
             f'text-anchor="middle" transform="rotate(-90 20 {(H - top - B) / 2 + top:.0f})">'
             f'Score (%)</text>')
    p.append(UI_BLOCK)
    p.append("</svg>")
    return "".join(p), jc

def main():
    charts = []
    summary = {}
    for bkey, title, subtitle, jglob in CHARTS:
        js = JEV_SCORES[_JEV_KEY[bkey]]
        svg, jc = chart(bkey, title, subtitle, jglob,
                        js["greedy"] * 100, js["weighted_mean_p_gold"] * 100, js["n"])
        charts.append(svg)
        summary[bkey] = {"jev_cost_per_question_usd": round(jc, 8),
                         "jev_run_cost_usd": round(jc * js["n"], 4)}
        print(f"{bkey}: Jev ${jc:.6f}/question (${jc * js['n']:.4f}/run)")
    lead = ("Every external point is a vals.ai platform row: accuracy and cost per "
            "question measured by the same harness, chain-of-thought included in the "
            "cost. Jev's diamonds are measured from our billing ($0.042/M input "
            "tokens, output free) and its scores read from the regenerated score "
            "artifacts (all-requested). Color is release era (red &le;2023 &rarr; purple "
            "&rarr; blue 2026; gray = date not established); marker shape is the "
            "row's reasoning tier (solid dot = thinking unspecified, ring = none). "
            "All models are labeled; hover or focus any point to isolate it and see "
            "score rank, cost rank, release date and tier. "
            "Dagger-marked points are our own matched runs of a dozen models "
            "on the identical frozen items (direct answers, one attempt, strict "
            "parsing, provider-reported costs) - they fill the commodity end of the "
            "market that the vals boards do not cover, and only complete v4r1 cells "
            "(full sampled denominators) are charted. "
            "HLE and MATH-500 are not charted here: Jev's rows on those benchmarks "
            "are a multiple-choice subset and an MCQ conversion, and no fair "
            "cost/score comparison exists against free-form or full-set references. "
            "Jev's measured costs: " + json.dumps(summary) + ".")
    style = ("body{background:" + BG + ";color:" + TEXT + ";font-family:system-ui,sans-serif;"
             "max-width:1020px;margin:2rem auto;padding:0 1rem;line-height:1.5}"
             "h1{font-size:24px}svg{width:100%;height:auto;display:block;margin:.4rem 0 1.4rem}"
             "p.lead{color:" + MUTED + ";font-size:14px;max-width:82ch}")
    html = ('<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            '<title>Jev Pareto frontiers</title><style>' + style + '</style></head><body>\n'
            '<h1>The frontier the cost numbers actually draw</h1>\n'
            '<p class="lead">' + lead + '</p>\n'
            + "".join(charts) + '\n</body></html>\n')
    OUT.write_text(html, encoding="utf-8")
    print(f"[ok] wrote {OUT} ({OUT.stat().st_size:,} bytes); "
          f"{len(charts)} charts")


if __name__ == "__main__":
    main()

