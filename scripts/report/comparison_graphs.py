#!/usr/bin/env python3
"""Render comparison graphs (self-contained SVG/HTML, viewable offline).

One bar chart per benchmark whose protocols can be put side by side
honestly: MMLU-Pro, GPQA (vals.ai rows, 133 models each), ARC-Challenge and
ARC-AGI-2 (hand-collected canonical rows), plus the option-rotation audit as
a Jev-only consistency chart.

NOT charted, by audit decision: HLE (Jev's row is the multiple-choice
subset; every published reference is the full text-only set, several with
tools/reasoning - no fair visual exists) and MATH-500 (Jev's row is an MCQ
conversion; external rows are free-form reasoning). Both stay in the report's
results table with their caveats.

Encoding: bar color = release era (red <=2023 -> magenta -> purple -> blue
2026; the hue route avoids green/teal so Jev cannot be mistaken for a year;
gray if undated); tip shape = reasoning tier (rounder = less thinking);
Jev = white-ringed diamonds, teal (greedy) / amber (probability-weighted),
bold labels. Every bar carries a hover tooltip (name, score + rank, release
date and its basis, tier, measured cost where available).

Output: docs/modern-comparison/comparison-graphs.html
  python scripts/report/comparison_graphs.py
"""

from __future__ import annotations

import glob
import html as _html
import json
from pathlib import Path

from chartkit import (BG, GRID, JEV_G, JEV_W, MUTED, TEXT, UI_BLOCK, UNK,
                      CHEAP_META, load_matched, matched_points,
                      TIER_LABEL, era_color, era_legend, jev_diamond, symbol,
                      tip, wrap_subtitle)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/modern-comparison/comparison-graphs.html"

VALS = json.loads((ROOT / "docs/modern-comparison/canonical/vals-leaderboards-20260926.json").read_text())
REF = json.loads((ROOT / "docs/modern-comparison/canonical/comparable-scores.json").read_text())


def _gload(pat):
    return json.loads(Path(glob.glob(str(ROOT / pat))[0]).read_text())


JEV = REF["jev_scores"]
AGI_W = _gload("runs_benchmark_ext/bench-arc_agi2_choice-*/derived/weighted_score.json")["per_cell_weighted_mean"]
MMLU_ITEMS = _gload("runs_benchmark/bench-mmlu_full-*/derived/score.json")["per_item"]
ROT_ITEMS = _gload("runs_benchmark/bench-option_rotations-*/derived/score.json")["per_item"]

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


def select(models, cap=34):
    have = {k: m for k, m in models.items() if m.get("accuracy") is not None}
    order = sorted(have, key=lambda k: -have[k]["accuracy"])
    chosen = set(order[:10]) | {k for k in PRIORITY if k in have} | set(order[-4:])
    if len(chosen) < cap:
        rest = sorted((k for k in have if k not in chosen),
                      key=lambda k: have[k].get("released") or "2025-01")
        need = cap - len(chosen)
        if rest and need > 0:
            step = max(1, len(rest) // need)
            chosen |= set(rest[::step][:need])
    return order, sorted(chosen, key=lambda k: -have[k]["accuracy"])


def _tt_model(m, rank, nfull, ndisp, source="vals.ai"):
    rel = m.get("released")
    basis = m.get("date_basis") or "n/a"
    tier = TIER_LABEL.get(m.get("tier", "none"), "unknown")
    cost = m.get("cost_per_test")
    parts = [f"<b>{_html.escape(m['name'])}</b>",
             f"score {m['accuracy']:.1f}% &mdash; #{rank} of {nfull}",
             f"released {rel or 'unknown'}"
             + (" (estimated)" if basis == "estimated" else "")]
    if cost:
        parts.append(f"measured cost ${cost:.4f}/test")
    return "<br>".join(parts)


def _tt_jev(label, score, note):
    return (f"<b>{label} (this study)</b><br>score {score:.1f}%<br>"
            f"protocol: direct one-shot answers, no reasoning, no tools<br>"
            f"cost: $0.042/M input tokens, output free (measured billing)<br>"
            f"{note}")


def bar_chart(title, subtitle, entries, unit="%"):
    W, L, R, row_h = 1180, 292, 128, 25
    sub, sub_end = wrap_subtitle(24, 50, subtitle)
    leg_y = sub_end + 20
    top = leg_y + 22
    H = top + len(entries) * row_h + 62
    xmax = max(max(e["score"] for e in entries), 1) * 1.03
    p = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
         f'font-family="system-ui,sans-serif" role="img" '
         f'aria-label="{_html.escape(title, quote=True)}">',
         f'<rect width="{W}" height="{H}" fill="{BG}" rx="14"/>',
         f'<text x="24" y="30" fill="{TEXT}" font-size="16.5" font-weight="700">'
         f'{_html.escape(title)}</text>',
         sub, era_legend(24, leg_y)]
    gx = 0
    step = 10 if xmax <= 100 else 20
    while gx <= xmax:
        x = L + gx / xmax * (W - L - R)
        p.append(f'<line x1="{x:.0f}" y1="{top - 10}" x2="{x:.0f}" y2="{H - 46}" stroke="{GRID}"/>')
        p.append(f'<text x="{x:.0f}" y="{H - 32}" fill="{MUTED}" font-size="10" '
                 f'text-anchor="middle">{gx}</text>')
        gx += step
    y = top
    for e in entries:
        bw = max(e["score"] / xmax * (W - L - R), 0.0)
        jev = e.get("jev")
        lab_cls = "mlabel jevlabel" if jev else "mlabel"
        stroke = ' stroke="#ffffff" stroke-opacity="0.6" stroke-width="1"' if jev else ''
        hgt = 15 if jev else 12
        p.append(f'<g class="mrow" {tip(e["tooltip"])}>')
        p.append(f'<text class="{lab_cls}" x="{L - 10}" y="{y + 12}" '
                 f'fill="{TEXT if jev else MUTED}" font-size="{11.5 if jev else 10.5}"'
                 f'{" font-weight='700'" if jev else ""} text-anchor="end">'
                 f'{_html.escape(e["name"])}</text>')
        p.append(f'<rect x="{L}" y="{y + 4}" width="{bw:.1f}" height="{hgt}" rx="3" '
                 f'fill="{e["color"]}"{stroke} opacity="{1 if jev else 0.93}"/>')
        vx = L + bw + 8
        p.append(f'<text class="{lab_cls}" x="{vx:.1f}" y="{y + 12}" fill="{e["color"]}" '
                 f'font-size="10.5" font-weight="600">{e["score"]:.1f}</text>')
        mx = vx + 30
        if jev:
            p.append(jev_diamond(mx, y + 8.5, e["color"]))
        else:
            p.append(symbol(mx, y + 8.5, e.get("tier", "none"), e["color"], 4.4))
        p.append('</g>')
        y += row_h
    p.append(f'<text x="{(W - L - R) / 2 + L:.0f}" y="{H - 12}" fill="{MUTED}" '
             f'font-size="11" text-anchor="middle">score ({unit})</text>')
    p.append(UI_BLOCK)
    p.append("</svg>")
    return "".join(p)


def _matched_entries(mrows, jev_score_for_join):
    out = []
    for r in mrows:
        tt = (f"<b>{r['name']}</b> &mdash; our matched run<br>"
              f"score {r['accuracy']:.1f}%<br>"
              f"released {r['released'] or 'unknown'}"
              + (" (estimated)" if r.get("date_basis") == "estimated" else "") + "<br>"
              f"measured cost ${r['cost']:.6f}/question")
        out.append({"name": r["name"] + " \u2020", "score": r["accuracy"],
                    "color": era_color(r.get("released")), "tier": r.get("tier", "none"),
                    "jev": False, "tooltip": tt})
    return out


def jev_bars(pairs):
    out = []
    for label, score, kind, note in pairs:
        out.append({"name": label, "score": score,
                    "color": JEV_G if kind == "greedy" else JEV_W,
                    "jev": True, "tooltip": _tt_jev(label, score, note)})
    return out


def vals_chart(bkey, title, subtitle, jevs, unit="%", matched=None):
    models = VALS["benchmarks"][bkey]["models"]
    order, sel = select(models)
    rank = {k: i + 1 for i, k in enumerate(order)}
    entries = _matched_entries(matched or [], None)
    for k in sel:
        m = models[k]
        entries.append({"name": m["name"], "score": m["accuracy"],
                        "color": era_color(m.get("released")),
                        "tier": m.get("tier", "none"), "jev": False,
                        "tooltip": _tt_model(m, rank[k], len(order), len(sel))})
    entries.sort(key=lambda e: -e["score"])
    for je in sorted(jevs, key=lambda j: j["score"]):
        i = next((i for i, e in enumerate(entries) if e["score"] < je["score"]),
                 len(entries))
        entries.insert(i, je)
    return bar_chart(title, subtitle, entries, unit)

ARC_META = {
    "Llama 3.1 405B": ("2024-07", "none"), "GPT-4o": ("2024-05", "none"),
    "Claude 3 Opus": ("2024-03", "none"), "GPT-4": ("2023-03", "none"),
    "Nemotron-H 56B": ("2025-06", "none"), "Llama 3.1 70B": ("2024-07", "none"),
    "Llama 3.1 8B (base, zero-shot)": ("2024-07", "none"),
    "Llama 3.1 8B (base model)": ("2024-07", "none"),
    "GPT-6 Astra": ("2026-09", "max"), "GPT-5.6 Sol": ("2026-07", "max"),
    "Claude Opus 5": ("2026-08", "high"), "Claude Fable 5.1": ("2026-08", "high"),
    "Claude Fable 5": ("2026-07", "high"), "Gemini 3.7 Flash": ("2026-07", "high"),
    "Grok 4.6": ("2026-05", "high"), "DeepSeek V4 Flash 0731": ("2025-07", "high"),
    "DeepSeek V4 Pro 0813": ("2025-08", "high"),
    "Gemini 3.6 Flash": ("2026-06", "high"), "Kimi K3": ("2026-06", "high"),
    "GPT-5.6 Luna": ("2026-07", "max"), "GPT-5.2 (Dec 2025)": ("2025-12", "high"),
    "Inkling Small": ("2026-04", "medium"),
    "Claude 3.7 Sonnet (thinking 16K)": ("2025-02", "high"),
    "Claude 3.7 Sonnet (thinking)": ("2025-02", "high"),
    "Claude 3.7 Sonnet (no thinking)": ("2025-02", "none"),
    "Claude 3.7 Sonnet": ("2025-02", "high"),
    "GPT-4.5": ("2025-02", "none"), "o3 (low)": ("2025-04", "low"),
    "o3": ("2025-04", "high"),
    "Gemma 4 E4B": ("2026-03", "medium"), "Gemma 4 E2B": ("2026-03", "medium"),
    "Qwen 3.5 9B": ("2026-01", "high"),
    "Qwen3.8 Max": ("2026-08", "high"),
    "DeepSeek R1 Distill Qwen 14B": ("2025-01", "high"),
    "DeepSeek R1 Distill Llama 8B": ("2025-01", "high"),
    "GPT-5 (high)": ("2025-08", "high"),
    "Gemini 3 Pro": ("2025-11", "high"),
    "GPT-4o (chatgpt-latest 2025-03)": ("2025-03", "none"),
    "Qwen3-235B-A22B-Thinking-2507": ("2025-07", "high"),
    "DeepSeek-R1-0528": ("2025-05", "high"),
}


def ref_chart(bkey, title, subtitle, jevs, unit="%", matched=None):
    rows = [r for r in REF["benchmarks"][bkey]["rows"] if r.get("score") is not None]
    order = sorted(rows, key=lambda r: -r["score"])
    rank = {r["model"]: i + 1 for i, r in enumerate(order)}
    entries = _matched_entries(matched or [], None)
    for r in order:
        rel, tier = ARC_META.get(r["model"], (None, "none"))
        m = {"name": r["model"], "accuracy": r["score"] * 100, "released": rel,
             "date_basis": "curated" if rel else None, "tier": tier,
             "cost_per_test": None}
        proto = r.get("protocol") or "unknown"
        tt = (f"<b>{_html.escape(r['model'])}</b><br>score {r['score']*100:.1f}% "
              f"&mdash; #{rank[r['model']]} of {len(order)}<br>"
              f"released {rel or 'unknown'}<br>"
              f"protocol: {proto} &middot; source type: {r.get('source_type', 'n/a')}")
        entries.append({"name": r["model"], "score": r["score"] * 100,
                        "color": era_color(rel), "tier": tier, "jev": False,
                        "tooltip": tt})
    entries.sort(key=lambda e: -e["score"])
    for je in sorted(jevs, key=lambda j: j["score"]):
        i = next((i for i, e in enumerate(entries) if e["score"] < je["score"]),
                 len(entries))
        entries.insert(i, je)
    return bar_chart(title, subtitle, entries, unit)


def rotation_chart():
    from collections import defaultdict
    native = {it["item_id"].split(":")[0]: it for it in MMLU_ITEMS}
    agg = defaultdict(lambda: {"nat": [0, 0], "rot": [0, 0]})
    flips = 0
    npair = 0
    for it in ROT_ITEMS:
        base = it["item_id"].split(":")[0]
        if base not in native or it.get("status") != "ok" or native[base].get("status") != "ok":
            continue
        g = it.get("group") or "other"
        n_ok = bool(native[base]["correct"])
        r_ok = bool(it["correct"])
        agg[g]["nat"][0] += n_ok; agg[g]["nat"][1] += 1
        agg[g]["rot"][0] += r_ok; agg[g]["rot"][1] += 1
        npair += 1
        flips += (n_ok != r_ok)
    tot_n = [sum(a["nat"][0] for a in agg.values()), npair]
    tot_r = [sum(a["rot"][0] for a in agg.values()), npair]
    rows = [("ALL PAIRED ITEMS", tot_n, tot_r)]
    rows += [(g.title(), a["nat"], a["rot"])
             for g, a in sorted(agg.items(),
                                key=lambda kv: -(kv[1]["nat"][0] / kv[1]["nat"][1]))]
    entries = []
    for name, nat, rot in rows:
        n_acc = 100 * nat[0] / nat[1]
        r_acc = 100 * rot[0] / rot[1]
        base_tt = (f"<b>{_html.escape(name)}</b><br>{nat[1]} paired items<br>"
                   f"native order: {nat[0]}/{nat[1]} correct ({n_acc:.1f}%)<br>"
                   f"rotated order: {rot[0]}/{rot[1]} correct ({r_acc:.1f}%)")
        entries.append({"name": f"{name} - native", "score": n_acc, "color": JEV_G,
                        "jev": True, "tooltip": base_tt + "<br>condition: original option order"})
        entries.append({"name": f"{name} - rotated", "score": r_acc, "color": JEV_W,
                        "jev": True, "tooltip": base_tt + "<br>condition: shuffled option order"})
    subtitle = ("The same 420 MMLU-Pro items re-run with their option order shuffled. "
                "A consistency check, not a comparison benchmark: no other model publishes "
                "scores under this protocol, so every bar is Jev (teal = native order, "
                "amber = rotated). Shuffling flipped "
                f"{100*flips/npair:.1f}% of the {npair} paired answers with no net direction "
                "(exact McNemar p = 0.83): content dominates position wherever there is "
                "content.")
    return bar_chart("Option-rotation audit - Jev vs Jev (consistency, not comparison)",
                     subtitle, entries)


def matched_only_chart(title, subtitle, mrows, jevs, unit="%"):
    entries = _matched_entries(mrows, None)
    for je in sorted(jevs, key=lambda j: j["score"]):
        i = next((i for i, e in enumerate(entries) if e["score"] < je["score"]),
                 len(entries))
        entries.insert(i, je)
    return bar_chart(title, subtitle, entries, unit)


def main() -> int:
    summary = load_matched(ROOT)
    mm_mmlu = matched_points(summary, "mmlu")
    mm_arc = matched_points(summary, "arc")
    mm_gpqa = matched_points(summary, "gpqa")
    mm_math = matched_points(summary, "math500_choice")
    mm_hle = matched_points(summary, "hle_text_mc")
    charts = []
    charts.append(vals_chart(
        "mmlu_pro", "MMLU-Pro - broad knowledge (12,032 items)",
        "Jev: direct one-shot answers (diamonds; teal greedy, amber probability-weighted). "
        "Vals rows use the platform harness with per-row reasoning configs (tip shapes).",
        jev_bars([("Jev (greedy)", JEV["mmlu_pro"]["greedy"] * 100, "greedy",
                   "12,032 graduate-level multiple-choice items"),
                  ("Jev (weighted)", JEV["mmlu_pro"]["weighted_mean_p_gold"] * 100,
                   "weighted", "mean probability placed on the gold option")]),
        matched=mm_mmlu))
    charts.append(vals_chart(
        "gpqa", "GPQA - graduate science (Diamond for Jev)",
        "Vals retired GPQA in Sep 2026 as saturated; rows preserved. Jev: Diamond subset, "
        "direct answers, seeded option shuffle.",
        jev_bars([("Jev (greedy)", JEV["gpqa_diamond"]["greedy"] * 100, "greedy",
                   "196 Diamond items"),
                  ("Jev (weighted)", JEV["gpqa_diamond"]["weighted_mean_p_gold"] * 100,
                   "weighted", "mean probability on the gold option")]),
        matched=mm_gpqa))
    charts.append(ref_chart(
        "arc_challenge", "ARC-Challenge - elementary science (canonical refs)",
        "No vals coverage for this encoding; bars are hand-collected canonical rows "
        "(mostly 25-shot CoT, mostly older models). Saturated for everyone.",
        jev_bars([("Jev (greedy)", JEV["arc_challenge"]["greedy"] * 100, "greedy",
                   "1,172 items"),
                  ("Jev (weighted)", JEV["arc_challenge"]["weighted_mean_p_gold"] * 100,
                   "weighted", "mean probability on the gold option")]),
        matched=mm_arc))
    charts.append(ref_chart(
        "arc_agi2", "ARC-AGI-2 - abstract puzzles (mixed encodings - read the caption)",
        "External bars: whole-grid pass@2, semi-private set, reasoning on. Jev cannot emit "
        "grids. Its exact-grid bars are the protocol-matched pair (0 of 120 tasks solved); "
        "the per-cell bars are a diagnostic encoding, not comparable to grid-level rows.",
        jev_bars([("Jev per-cell (greedy)", JEV["arc_agi2_public_eval"]["cell_accuracy_choice"] * 100,
                   "greedy", "diagnostic: 70,100 individual cell decisions"),
                  ("Jev per-cell (weighted)", AGI_W * 100, "weighted",
                   "diagnostic: mean probability on the gold cell color"),
                  ("Jev exact-grid (greedy)", 0.0, "greedy",
                   "protocol-matched: 0 of 120 tasks with every cell correct"),
                  ("Jev exact-grid (weighted)", 0.0, "weighted",
                   "whole-grid probability is the product of cell probabilities: median "
                   "~1e-63, indistinguishable from zero")])))
    # rotation audit deliberately not charted: position bias lives in the
    # architecture section ("The order of the options matters"), per audit.
    jev_math = JEV["math500_mcq_adapted"]["greedy"] * 100
    jev_hle = JEV["hle_text_only_mc"]["greedy"] * 100
    _math_beats = mm_math and max(r["accuracy"] for r in mm_math) > jev_math
    _n_math_beats = sum(1 for r in mm_math if r["accuracy"] > jev_math)
    _n_hle_beats = sum(1 for r in mm_hle if r["accuracy"] > jev_hle)
    if _math_beats:
        charts.append(matched_only_chart(
            "MATH-500 as multiple choice - matched runs (this study)",
            "Our runs of cheap OpenRouter models on Jev's exact 261-item MCQ "
            "conversion - same items, same format, direct answers. The vals "
            "free-form reasoning rows are not shown (protocol mismatch). "
            f"{_n_math_beats} of {len(mm_math)} matched models beat Jev here.",
            mm_math,
            jev_bars([("Jev (greedy)", jev_math, "greedy", "261 encodable items, MCQ"),
                      ("Jev (weighted)",
                       JEV["math500_mcq_adapted"]["weighted_mean_p_gold"] * 100,
                       "weighted", "mean probability on the gold option")])))
    if mm_hle and max(r["accuracy"] for r in mm_hle) > jev_hle:
        charts.append(matched_only_chart(
            "Humanity's Last Exam (MC subset) - matched runs (this study)",
            "Our runs of cheap OpenRouter models on Jev's exact 494-item MC "
            "subset - same items, same format, direct answers. The vals "
            "full-text-set rows are not shown (protocol mismatch). "
            f"{_n_hle_beats} of {len(mm_hle)} matched models beat Jev here.",
            mm_hle,
            jev_bars([("Jev (greedy)", jev_hle, "greedy", "494 MC items"),
                      ("Jev (weighted)",
                       JEV["hle_text_only_mc"]["weighted_mean_p_gold"] * 100,
                       "weighted", "mean probability on the gold option")])))

    html = ("<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
            "<title>Jev benchmark comparisons</title><style>\n"
            f" body {{ background:{BG}; color:{TEXT}; font-family: system-ui, sans-serif;\n"
            "        max-width:1000px; margin:2rem auto; padding:0 1rem; line-height:1.5; }}\n"
            " h1 { font-size:24px; }\n"
            " svg { width:100%; height:auto; display:block; margin:.4rem 0 1.4rem; }\n"
            f" p.lead {{ color:{MUTED}; font-size:14px; max-width:80ch; }}\n"
            "</style></head><body>\n"
            "<h1>Where Jev lands, across the whole field</h1>\n"
            "<p class=\"lead\">Every non-Jev bar is a published value: vals.ai platform rows "
            "for MMLU-Pro and GPQA (fetched 2026-09-26 into "
            "canonical/vals-leaderboards-20260926.json), hand-collected canonical rows for "
            "the two ARC encodings. Bar color is release era (red &le;2023 &rarr; purple "
            "&rarr; blue 2026; the hue route avoids teal so Jev's diamonds cannot be "
            "mistaken for a year; gray = date not established). Tip shapes are reasoning "
            "tiers, rounder = less thinking. Hover any bar for details. Charts show a "
            "curated view (top, bottom, and audit-named models); the full extract stays on "
            "disk. Dagger-marked bars are our own matched runs of cheap OpenRouter "
            "models on the identical frozen items under Jev's protocol (direct "
            "answers, strict parsing, measured costs). HLE and MATH-500 are "
            "deliberately NOT charted: Jev's rows there are a "
            "multiple-choice subset and an MCQ conversion respectively, and no published "
            "rows share those protocols - their numbers live in the report table with "
            "caveats instead.</p>\n"
            + ("<p class=\"lead\">Update: the matched cheap-model runs (dagger "
               "bars, this study) DO beat Jev on one of those subsets, so that "
               "chart is revived above as a matched-only comparison - our runs, "
               "Jev's exact items and format, no external protocol mixing.</p>\n"
               if (mm_math and max(r["accuracy"] for r in mm_math) > JEV["math500_mcq_adapted"]["greedy"] * 100)
               or (mm_hle and max(r["accuracy"] for r in mm_hle) > JEV["hle_text_only_mc"]["greedy"] * 100)
               else "")
            + "".join(charts) + "\n</body></html>\n")
    OUT.write_text(html, encoding="utf-8")
    print(f"[ok] wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size:,} bytes); "
          f"{len(charts)} charts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

