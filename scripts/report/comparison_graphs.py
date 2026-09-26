#!/usr/bin/env python3
"""Render comparison graphs (self-contained SVG/HTML, viewable offline).

One bar chart per benchmark. Every non-Jev bar is a published value: for
MMLU-Pro / GPQA / MATH-500 / HLE the rows come from the vals.ai platform
extract (docs/modern-comparison/canonical/vals-leaderboards-20260926.json,
133/133/52/69 models); for ARC-Challenge and ARC-AGI-2 they come from the
hand-collected canonical references (comparable-scores.json), because no
platform leaderboard covers Jev's encodings there.

Encoding (requested by the audit):
  bar color   release era - red = oldest (2022), blue = newest (2026)
  tip symbol  reasoning effort - rounder = less thinking, pointier = more:
              circle none, hexagon low, pentagon medium, square high,
              triangle xhigh, plus max; an underlying dot is always drawn
  Jev         teal (greedy) / amber (probability-weighted), as before

Display selection per vals chart: top-10 by score, a priority list of models
named in the audit, the bottom-4, then date-stratified fill to ~32 bars -
chosen so the full range (frontier to floor, 2022 to 2026) is covered. The
full extract stays on disk; the chart is a curated view of it.

Output: docs/modern-comparison/comparison-graphs.html
  python scripts/report/comparison_graphs.py
"""

from __future__ import annotations

import colorsys
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/modern-comparison/comparison-graphs.html"

BG = "#0e1117"; PANEL = "#161b27"; TEXT = "#e6e9f2"; MUTED = "#9aa2b6"
GRID = "#242a38"; JEV_G = "#38e1c8"; JEV_W = "#f5b342"

VALS = json.loads((ROOT / "docs/modern-comparison/canonical/vals-leaderboards-20260926.json").read_text())
REF = json.loads((ROOT / "docs/modern-comparison/canonical/comparable-scores.json").read_text())
import glob as _glob
_agi_w = json.loads(Path(_glob.glob(str(
    ROOT / "runs_benchmark_ext/bench-arc_agi2_choice-*/derived/weighted_score.json"))[0]
).read_text())
AGI_W = _agi_w["per_cell_weighted_mean"]

ERA_MIN = (2022, 11)   # GPT-3.5 Turbo
ERA_MAX = (2026, 9)    # GPT-6 generation


def _months(ym: str) -> int:
    y, m = ym.split("-")
    return int(y) * 12 + int(m)


def era_t(released: str | None) -> float:
    if not released:
        return 0.5
    lo = _months("%d-%02d" % ERA_MIN); hi = _months("%d-%02d" % ERA_MAX)
    return max(0.0, min(1.0, (_months(released) - lo) / (hi - lo)))


def era_color(t: float) -> str:
    # red (h=4deg) -> blue (h=228deg), saturation/lightness gentle on dark bg
    h = (4 + t * (228 - 4)) / 360.0
    r, g, b = colorsys.hls_to_rgb(h, 0.60, 0.62)
    return "#%02x%02x%02x" % (round(r * 255), round(g * 255), round(b * 255))


TIER_ORDER = ["none", "low", "medium", "high", "xhigh", "max"]
TIER_SIDES = {"none": 0, "low": 6, "medium": 5, "high": 4, "xhigh": 3, "max": -1}
TIER_LABEL = {"none": "no reasoning", "low": "low", "medium": "medium",
              "high": "high", "xhigh": "very high", "max": "max"}


def symbol(cx: float, cy: float, tier: str, color: str, r: float = 5.0) -> str:
    """Underlying dot + tier outline shape (rounder = less thinking)."""
    out = [f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="1.8" fill="{color}"/>']
    sides = TIER_SIDES.get(tier, 0)
    if sides == -1:  # plus / asterisk for "max"
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


def legend(width: int, y: int) -> str:
    p = [f'<text x="24" y="{y}" fill="{MUTED}" font-size="11.5" font-weight="600">'
         'bar color = release era</text>']
    x0 = 210
    for i in range(60):
        t = i / 59
        p.append(f'<rect x="{x0 + i * 3}" y="{y - 9}" width="3.4" height="10" '
                 f'fill="{era_color(t)}"/>')
    p.append(f'<text x="{x0 - 6}" y="{y}" fill="{MUTED}" font-size="10.5" '
             f'text-anchor="end">2022</text>')
    p.append(f'<text x="{x0 + 184}" y="{y}" fill="{MUTED}" font-size="10.5">2026</text>')
    x = x0 + 240
    p.append(f'<text x="{x}" y="{y}" fill="{MUTED}" font-size="11.5" '
             f'font-weight="600">tip shape = reasoning</text>')
    x += 168
    for tier in TIER_ORDER:
        p.append(symbol(x, y - 4, tier, MUTED, 4.6))
        p.append(f'<text x="{x + 8}" y="{y}" fill="{MUTED}" font-size="9.6">'
                 f'{TIER_LABEL[tier]}</text>')
        x += 8 + len(TIER_LABEL[tier]) * 5.6 + 16
    return "".join(p)


PRIORITY = [
    "fireworks/gpt-oss-120b", "fireworks/gpt-oss-20b",
    "nvidia/nemotron-3-ultra-550b-a55b", "minimax/MiniMax-M3",
    "mistralai/mistral-medium-3.5", "alibaba/qwen3.8-27b",
    "google/gemini-3.8-flash", "thinkingmachines/inkling-small",
    "kimi/kimi-k3", "openai/gpt-5.6-luna", "openai/gpt-5.6-terra",
    "grok/grok-4.6", "grok/grok-4.7", "meta/muse_spark_1_3",
    "anthropic/claude-opus-4-8", "anthropic/claude-haiku-4-5-20251001-thinking",
    "anthropic/claude-opus-4-1-20250805", "zai/glm-4.7",
    "xiaomi/mimo-v2-flash", "grok/grok-4-0709", "fireworks/deepseek-v3p2",
    "google/gemini-2.5-flash-lite-preview-09-2025", "google/gemini-2.0-flash-001",
    "mistralai/mistral-large-2411", "fireworks/deepseek-v3",
    "anthropic/claude-3-5-sonnet-20241022", "openai/gpt-4o-2024-08-06",
    "openai/gpt-3.5-turbo", "openai/o3-2025-04-16", "openai/gpt-5-2025-08-07",
    "fireworks/qwen3-235b-a22b", "fireworks/llama4-maverick-instruct-basic",
    "alibaba/qwen3-max", "zai/glm-4.5", "cohere/command-r-plus",
    "deepseek/deepseek-v4-flash-0731", "google/gemini-1.5-pro-002",
    "anthropic/claude-3-7-sonnet-20250219",
]


def select(models: dict, cap: int = 32) -> list[str]:
    have = {s: m for s, m in models.items() if m.get("accuracy") is not None}
    by_score = sorted(have, key=lambda s: -have[s]["accuracy"])
    chosen = set(by_score[:10])
    chosen |= {s for s in PRIORITY if s in have}
    chosen |= set(by_score[-4:])
    if len(chosen) < cap:
        rest = sorted((s for s in have if s not in chosen),
                      key=lambda s: have[s].get("released") or "2025-01")
        need = cap - len(chosen)
        if rest:
            step = max(1, len(rest) // max(need, 1))
            chosen |= set(rest[::step][:need])
    return sorted(chosen, key=lambda s: -have[s]["accuracy"])


def bar_chart(key: str, title: str, subtitle: str, models: dict,
              jev: tuple[float, float | None]) -> str:
    sel = select(models)
    rows = []
    for s in sel:
        m = models[s]
        rows.append((m["name"], m["accuracy"], era_color(era_t(m.get("released"))),
                     m.get("tier", "none")))
    g, w = jev
    # insert Jev bars at score position
    def insert(rows, label, score, color, tier=None):
        i = next((i for i, r in enumerate(rows) if r[1] < score), len(rows))
        rows.insert(i, (label, score, color, tier))
    if w is not None:
        insert(rows, "Jev (weighted)", w, JEV_W)
    insert(rows, "Jev (greedy)", g, JEV_G)

    W = 980
    row_h = 22
    top = 96
    legend_y = 66
    H = top + len(rows) * row_h + 66
    L = 268
    R = 92
    xmax = max(r[1] for r in rows) * 1.02
    p = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
         f'font-family="system-ui,sans-serif" role="img" aria-label="{title}">',
         f'<rect width="{W}" height="{H}" fill="{BG}" rx="14"/>',
         f'<text x="24" y="30" fill="{TEXT}" font-size="17" font-weight="700">{title}</text>',
         f'<text x="24" y="48" fill="{MUTED}" font-size="11.5">{subtitle}</text>',
         legend(W, legend_y)]
    for gx in range(0, int(xmax) + 1, 10):
        x = L + gx / xmax * (W - L - R)
        p.append(f'<line x1="{x:.0f}" y1="{top - 8}" x2="{x:.0f}" y2="{H - 52}" stroke="{GRID}"/>')
        p.append(f'<text x="{x:.0f}" y="{H - 38}" fill="{MUTED}" font-size="10" '
                 f'text-anchor="middle">{gx}</text>')
    y = top
    for name, score, color, tier in rows:
        bw = score / xmax * (W - L - R)
        is_jev = name.startswith("Jev")
        fs = 11.5 if is_jev else 10.5
        weight = ' font-weight="600"' if is_jev else ""
        p.append(f'<text x="{L - 12}" y="{y + 12}" fill="{TEXT if is_jev else MUTED}" '
                 f'font-size="{fs}"{weight} text-anchor="end">{name}</text>')
        h = 15 if is_jev else 12
        op = "" if is_jev else ' opacity="0.92"'
        p.append(f'<rect x="{L}" y="{y + 4}" width="{bw:.1f}" height="{h}" rx="3" '
                 f'fill="{color}"{op}/>')
        p.append(f'<text x="{L + bw + 8:.1f}" y="{y + 12}" fill="{color}" '
                 f'font-size="10.5" font-weight="600">{score:.1f}</text>')
        p.append(symbol(L + bw + 34 + len(f"{score:.1f}") * 2.2, y + 8.5,
                        tier if tier else "none", color if not is_jev else TEXT, 4.4))
        y += row_h
    p.append(f'<text x="{(W - L - R) / 2 + L:.0f}" y="{H - 16}" fill="{MUTED}" '
             f'font-size="11" text-anchor="middle">score (%)</text>')
    p.append("</svg>")
    return "".join(p)


# ARC charts keep the hand-collected canonical rows (no vals coverage for
# Jev's encodings); era/tier metadata for those few models is curated here.
ARC_META = {
    "Llama 3.1 405B": ("2024-07", "none"), "GPT-4o": ("2024-05", "none"),
    "Claude 3 Opus": ("2024-03", "none"), "GPT-4": ("2023-03", "none"),
    "Nemotron-H 56B": ("2025-06", "none"), "Llama 3.1 70B": ("2024-07", "none"),
    "Llama 3.1 8B (base model)": ("2024-07", "none"),
    "GPT-6 Astra": ("2026-09", "max"), "GPT-5.6 Sol": ("2026-07", "max"),
    "Claude Opus 5": ("2026-08", "high"), "Claude Fable 5.1": ("2026-08", "high"),
    "Claude Fable 5": ("2026-07", "high"), "Gemini 3.7 Flash": ("2026-07", "high"),
    "Grok 4.6": ("2026-05", "high"), "DeepSeek V4 Flash 0731": ("2025-07", "high"),
    "Inkling Small": ("2026-04", "medium"),
    "Claude 3.7 Sonnet (thinking 16K)": ("2025-02", "high"),
    "GPT-4.5": ("2025-02", "none"), "o3 (low)": ("2025-04", "low"),
    "DeepSeek V4 Pro 0813": ("2025-08", "high"),
    "Gemini 3.6 Flash": ("2026-06", "high"),
    "Kimi K3": ("2026-06", "high"),
    "GPT-5.6 Luna": ("2026-07", "max"),
    "GPT-5.2 (Dec 2025)": ("2025-12", "high"),
    "Llama 3.1 8B (base, zero-shot)": ("2024-07", "none"),
    "Gemma 4 E4B": ("2026-03", "medium"),
    "Gemma 4 E2B": ("2026-03", "medium"),
    "Qwen 3.5 9B": ("2026-01", "high"),
    "Claude 3.7 Sonnet (no thinking)": ("2025-02", "none"),
    "Claude 3.7 Sonnet (thinking)": ("2025-02", "high"),
    "Qwen3.8 Max": ("2026-08", "high"),
    "DeepSeek R1 Distill Qwen 14B": ("2025-01", "high"),
    "DeepSeek R1 Distill Llama 8B": ("2025-01", "high"),
    "GPT-5 (high)": ("2025-08", "high"),
    "o3": ("2025-04", "high"),
    "Gemini 3 Pro": ("2025-11", "high"),
    "GPT-4o (chatgpt-latest 2025-03)": ("2025-03", "none"),
    "Llama 3.1 70B": ("2024-07", "none"),
    "Qwen3-235B-A22B-Thinking-2507": ("2025-07", "high"),
    "DeepSeek-R1-0528": ("2025-05", "high"),
    "Claude 3.7 Sonnet": ("2025-02", "high"),
}


def ref_chart(key: str, title: str, subtitle: str, jev: tuple) -> str:
    rows_ref = [r for r in REF["benchmarks"][key]["rows"] if r.get("score") is not None]
    models = {}
    for r in rows_ref:
        rel, tier = ARC_META.get(r["model"], (None, "none"))
        models[r["model"]] = {"name": r["model"], "accuracy": r["score"] * 100,
                              "released": rel, "tier": tier}
    g = jev[0]
    w = jev[1] if len(jev) > 1 else None
    return bar_chart_manual(title, subtitle, models, (g, w))


def bar_chart_manual(title, subtitle, models, jev) -> str:
    return bar_chart(None, title, subtitle, models, jev)


def main() -> int:
    js = REF["jev_scores"]
    charts = []

    def vals_chart(bkey, title, subtitle, jev):
        return bar_chart(bkey, title, subtitle,
                         VALS["benchmarks"][bkey]["models"], jev)

    charts.append(vals_chart(
        "mmlu_pro", "MMLU-Pro - broad knowledge (vals.ai, 12,032 items)",
        "Jev: direct one-shot answers (teal greedy, amber probability-weighted). "
        "vals rows use the platform harness; per-row configs shown as tip shapes.",
        (js["mmlu_pro"]["greedy"] * 100, js["mmlu_pro"]["weighted_mean_p_gold"] * 100)))
    charts.append(vals_chart(
        "gpqa", "GPQA - graduate science (vals.ai)",
        "vals retired GPQA in Sep 2026 as saturated; rows preserved. Jev: "
        "Diamond subset, direct, seeded option shuffle.",
        (js["gpqa_diamond"]["greedy"] * 100, js["gpqa_diamond"]["weighted_mean_p_gold"] * 100)))
    charts.append(ref_chart(
        "arc_challenge", "ARC-Challenge - elementary science (canonical refs)",
        "No vals coverage for this encoding; bars are the hand-collected "
        "canonical rows (mostly 25-shot CoT, mostly older models). Saturated "
        "for everyone - Jev's best-looking chart.",
        (js["arc_challenge"]["greedy"] * 100, js["arc_challenge"]["weighted_mean_p_gold"] * 100)))
    charts.append(vals_chart(
        "math500", "MATH-500 (vals.ai, free-form reasoning protocol)",
        "External rows solve free-form with reasoning; Jev's bars are the "
        "4-option MCQ adaptation (greedy) - a protocol conversion, labeled as "
        "such everywhere in this report.",
        (js["math500_mcq_adapted"]["greedy"] * 100, js["math500_mcq_adapted"]["weighted_mean_p_gold"] * 100)))
    charts.append(ref_chart(
        "arc_agi2", "ARC-AGI-2 - abstract puzzles (canonical refs, pass@2 semi-private)",
        "Jev cannot emit grids: scored per cell (53.5%) - shown for shape, not "
        "rank. External bars are grid-production pass@2 with reasoning.",
        (js["arc_agi2_public_eval"]["cell_accuracy_choice"] * 100,
         AGI_W * 100)))
    charts.append(vals_chart(
        "hle", "Humanity's Last Exam (vals.ai, full text-only set)",
        "Jev's bar is the multiple-choice subset only - an easier format, with "
        "a ~25% guessing floor of its own. The vals rows are the full text-only "
        "set, where guessing is near-zero. Do not rank-read this chart across "
        "the two protocols; both say the same thing: expert-frontier material "
        "is far out of reach.",
        (js["hle_text_only_mc"]["greedy"] * 100, js["hle_text_only_mc"]["weighted_mean_p_gold"] * 100)))

    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Jev benchmark comparisons</title><style>
 body {{ background:{BG}; color:{TEXT}; font-family: system-ui, sans-serif;
        max-width:1000px; margin:2rem auto; padding:0 1rem; line-height:1.5; }}
 h1 {{ font-size:24px; }}
 svg {{ width:100%; height:auto; display:block; margin:.4rem 0 1.4rem; }}
 p.lead {{ color:{MUTED}; font-size:14px; max-width:80ch; }}
</style></head><body>
<h1>Where Jev lands, across the whole field</h1>
<p class="lead">Every non-Jev bar is a published value: vals.ai platform rows
for MMLU-Pro / GPQA / MATH-500 / HLE (fetched 2026-09-26 into
canonical/vals-leaderboards-20260926.json), hand-collected canonical rows for
the two ARC encodings. Bar color is release era (red 2022 &rarr; blue 2026;
estimated months are used only for the gradient and labeled in the extract);
the tip shape is the row's reasoning configuration (rounder = less thinking).
Charts show a curated ~32-bar view (top, bottom, and the models named in the
audit); the full 133-model extract stays on disk.</p>
{''.join(charts)}
</body></html>
"""
    OUT.write_text(html, encoding="utf-8")
    print(f"[ok] wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size:,} bytes); "
          f"{len(charts)} charts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
