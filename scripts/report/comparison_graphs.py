#!/usr/bin/env python3
"""Render comparison graphs (self-contained SVG/HTML, viewable offline).

One bar chart per benchmark. Each chart shows Jev twice - its greedy score and
its probability-weighted score - alongside published scores for other models.

The model numbers are curated by hand into BENCHMARKS below and kept in sync
with docs/modern-comparison/canonical/comparable-scores.json (the fetched,
source-labeled reference data). We do not benchmark any other model here; every
non-Jev bar is a published value read from a leaderboard or vendor report.

Output: docs/modern-comparison/comparison-graphs.html

No JavaScript, no external assets, no network calls at render time.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/modern-comparison/comparison-graphs.html"

# Each benchmark: a one-line description of how Jev was run ("jev_note"), a
# short plain-language paragraph about the comparison ("note"), and the bars as
# (model, score_percent[, flags]). "jev" + kind=greedy/weighted marks Jev's two
# highlighted bars. Scores are percentages.
BENCHMARKS: dict[str, dict] = {
    "MMLU-Pro": {
        "jev_note": "Jev gave one direct answer per question, with no step-by-step reasoning.",
        "note": ("Most other scores come from the Vals AI leaderboard, where models "
                 "were shown examples and asked to reason step by step, so the two "
                 "were not run under identical conditions. We could not find a "
                 "published MMLU-Pro score for GPT-6 Astra or Qwen3.8 Flash."),
        "bars": [
            ("Claude Fable 5.1", 92.4), ("Claude Opus 5", 91.6),
            ("GPT-5.6 Sol", 89.1), ("Qwen3.8 Max", 88.6),
            ("GLM-5.3", 86.8), ("DeepSeek V4 Flash 0731", 86.2),
            ("GLM-5.3 Flash", 86.1), ("Qwen3.8-27B", 84.3),
            ("Jev (greedy)", 82.8, {"jev": True, "kind": "greedy"}),
            ("Qwen 3.5 9B", 82.5), ("Claude 3.7 Sonnet (no thinking)", 80.7),
            ("GPT-4o", 74.7),
            ("Jev (weighted)", 74.0, {"jev": True, "kind": "weighted"}),
            ("Gemma 4 E4B", 69.4), ("Llama 3.1 8B", 48.3),
        ],
    },
    "GPQA Diamond": {
        "jev_note": "Jev answered each graduate-level science question directly, "
                    "with the answer options shuffled.",
        "note": ("The other scores come from Artificial Analysis with reasoning "
                 "turned on, so they are not an exact match for how Jev was run. "
                 "Vals AI retired GPQA in September 2026 because the top models had "
                 "nearly saturated it. DeepSeek V4 Flash scored 89.9 on Vals and "
                 "reported 88.1 for itself."),
        "bars": [
            ("GPT-6 Astra", 96.3), ("GPT-5.6 Sol", 95.2),
            ("Claude Fable 5.1", 93.7), ("Claude Opus 5", 93.2),
            ("Qwen3.8 Max", 92.8), ("GLM-5.3", 91.7),
            ("GLM-5.3 Flash", 91.2), ("DeepSeek V4 Flash 0731", 89.9),
            ("Qwen3.8-27B", 82.2), ("Qwen 3.5 9B", 77.6),
            ("Claude 3.7 Sonnet (no thinking)", 76.8),
            ("Jev (greedy)", 76.5, {"jev": True, "kind": "greedy"}),
            ("Jev (weighted)", 63.0, {"jev": True, "kind": "weighted"}),
            ("Gemma 4 E4B", 58.6), ("GPT-4o", 54.3), ("Llama 3.1 8B", 27.0),
        ],
    },
    "ARC-Challenge": {
        "jev_note": "Jev answered the full 1,172-question test directly, one answer "
                    "per question.",
        "note": ("This elementary-science test dates to 2018 and today's frontier "
                 "models score at the top of it and no longer publish it, so the "
                 "comparison set is mostly older models. Jev's 97.9% is the highest "
                 "score we could find on this benchmark."),
        "bars": [
            ("Jev (greedy)", 97.9, {"jev": True, "kind": "greedy"}),
            ("Jev (weighted)", 96.9, {"jev": True, "kind": "weighted"}),
            ("Llama 3.1 405B", 96.9), ("GPT-4o", 96.7),
            ("Claude 3 Opus", 96.4), ("GPT-4", 96.3),
            ("Nemotron-H 56B", 95.0), ("Llama 3.1 70B", 94.8),
            ("Llama 3.1 8B (base model)", 74.7),
        ],
    },
    "MATH-500": {
        "jev_note": "Jev cannot write out a worked solution, so each problem was "
                    "turned into a four-option multiple-choice question built from "
                    "its exact numeric answer.",
        "note": ("Every other model here was scored on the original open-response "
                 "version with reasoning, so Jev's bars sit outside that comparison "
                 "and are shown only for rough placement, not as a like-for-like "
                 "ranking. On a stricter digit-by-digit version of the same test, "
                 "Jev scored 13.6%."),
        "bars": [
            ("GPT-5 (high)", 99.4), ("o3", 99.2), ("Gemini 3 Pro", 96.4),
            ("DS R1 Distill Qwen 14B", 94.9),
            ("Claude 3.7 Sonnet (thinking)", 94.7),
            ("GPT-4o (chatgpt-latest)", 89.3), ("Gemma 4 E2B", 86.0),
            ("DS R1 Distill Llama 8B", 85.9),
            ("Jev (greedy)", 83.1, {"jev": True, "kind": "greedy"}),
            ("Llama 3.1 405B", 71.4), ("Llama 3.1 70B", 65.1),
            ("Jev (weighted)", 69.5, {"jev": True, "kind": "weighted"}),
        ],
    },
    "ARC-AGI-2": {
        "jev_note": "Jev cannot produce a finished grid, so it was measured a "
                    "different way: how often it picks the right color for each "
                    "individual cell of a puzzle.",
        "note": ("The other models are scored on whether they reproduce the whole "
                 "completed grid, on a harder private set, with reasoning and two "
                 "attempts each. Because the two numbers measure different things, "
                 "Jev's bars are placed here for context rather than as a direct "
                 "ranking. Jev's cell scores are from the public 120-task set."),
        "bars": [
            ("GPT-6 Astra", 95.0), ("GPT-5.6 Sol", 92.5),
            ("Claude Opus 5", 90.4), ("Claude Fable 5.1", 90.0),
            ("Gemini 3.7 Flash", 84.6), ("Grok 4.6", 67.1),
            ("DeepSeek V4 Flash 0731", 61.4),
            ("Jev (greedy, per cell)", 53.5, {"jev": True, "kind": "greedy"}),
            ("Inkling Small", 40.1),
            ("Jev (weighted, per cell)", 43.4, {"jev": True, "kind": "weighted"}),
            ("Claude 3.7 Sonnet (thinking)", 28.6), ("GPT-4.5", 10.3),
            ("o3 (low)", 4.0), ("GPT-4o", 0.0),
        ],
    },
    "Humanity's Last Exam": {
        "jev_note": "Jev answered only the multiple-choice questions (494 of them), "
                    "one direct answer each. Random guessing here would score about 17%.",
        "note": ("Other models are scored on the much larger full text-only set, "
                 "usually with reasoning and sometimes with tools, which are marked "
                 "with (tools) on the bar. Because Jev answered a narrower set of "
                 "questions in a different way, its number is not directly "
                 "comparable to the others."),
        "bars": [
            ("GLM-5.3 (tools)", 62.5), ("Claude Fable 5.1", 60.9),
            ("Qwen3.8 Max (tools)", 56.2), ("Claude Opus 5", 54.9),
            ("GPT-6 Astra", 54.2), ("GLM-5.3 Flash (tools)", 50.2),
            ("GPT-5.6 Sol", 49.5),
            ("DeepSeek V4 Flash 0731", 34.8), ("GPT-5 Pro (2025)", 33.3),
            ("GPT-5 (2025)", 26.3),
            ("Jev (greedy)", 21.9, {"jev": True, "kind": "greedy"}),
            ("Jev (weighted)", 21.2, {"jev": True, "kind": "weighted"}),
            ("Qwen3-235B-A22B (thinking)", 15.4), ("DeepSeek-R1-0528", 14.0),
        ],
    },
}

# ---------------------------------------------------------------- palette
JEV_GREEDY = "#38e1c8"      # teal
JEV_WEIGHTED = "#f5b342"    # amber
BG = "#0e1117"
PANEL = "#161b27"
TEXT = "#e6e9f2"
MUTED = "#9aa2b6"
GRID = "#242a38"


def _interp(c1: tuple, c2: tuple, t: float) -> str:
    return "#%02x%02x%02x" % tuple(round(a + (b - a) * t) for a, b in zip(c1, c2))


def _bar_color(rank: int, n: int, flags: dict) -> str:
    if flags.get("jev"):
        return JEV_GREEDY if flags.get("kind") == "greedy" else JEV_WEIGHTED
    return _interp((0x4a, 0x5b, 0xd4), (0x7d, 0x8c, 0xff), rank / max(1, n - 1))


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _wrap(text: str, width: int) -> list[str]:
    """Greedy word wrap so the SVG subtitle breaks onto lines."""
    words, lines, current = text.split(), [], ""
    for word in words:
        candidate = (current + " " + word).strip()
        if len(candidate) <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def bar_chart_svg(title: str, meta: dict, *, show_title: bool = True) -> str:
    bars = sorted(meta["bars"], key=lambda b: -b[1])
    W = 980
    left, right = 268, 90
    plot_w = W - left - right
    scale = plot_w / 100.0
    row_h = 30

    note_lines = _wrap(meta["note"], 150)
    top = (46 + 15 * len(note_lines)) if show_title else 22
    bottom = 40
    H = top + row_h * len(bars) + bottom

    parts = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
             f'role="img" aria-label="{_esc(title)}">',
             f'<rect x="0" y="0" width="{W}" height="{H}" fill="{PANEL}" rx="14"/>',
             ]
    if show_title:
        parts.append(f'<text x="24" y="30" fill="{TEXT}" font-size="18" font-weight="600">'
                     f'{_esc(title)}</text>')
        for i, line in enumerate(note_lines):
            parts.append(f'<text x="24" y="{48 + i * 15}" fill="{MUTED}" '
                         f'font-size="11.5">{_esc(line)}</text>')

    for tick in range(0, 101, 25):
        x = left + tick * scale
        parts.append(f'<line x1="{x:.1f}" y1="{top - 4}" x2="{x:.1f}" '
                     f'y2="{H - bottom + 2}" stroke="{GRID}" stroke-width="1"/>')
        parts.append(f'<text x="{x:.1f}" y="{H - bottom + 18}" fill="{MUTED}" '
                     f'font-size="10.5" text-anchor="middle">{tick}</text>')

    y = top + 4
    for rank, (name, value, *rest) in enumerate(bars):
        flags = rest[0] if rest else {}
        is_jev = bool(flags.get("jev"))
        color = _bar_color(rank, len(bars), flags)
        h = 20 if is_jev else 16
        w = max(1.5, value * scale)
        parts.append(f'<text x="{left - 12}" y="{y + h / 2 + 4}" fill="'
                     f'{TEXT if is_jev else MUTED}" '
                     f'font-size="{11.5 if is_jev else 10.5}" '
                     f'font-weight="{"600" if is_jev else "400"}" '
                     f'text-anchor="end">{_esc(name)}</text>')
        if is_jev:
            parts.append(f'<rect x="{left}" y="{y}" width="{plot_w}" height="{h}" '
                         f'fill="{color}" opacity="0.10" rx="3"/>')
        parts.append(f'<rect x="{left}" y="{y}" width="{w:.1f}" height="{h}" '
                     f'fill="{color}" rx="3"/>')
        parts.append(f'<text x="{left + w + 6:.1f}" y="{y + h / 2 + 4}" fill="'
                     f'{color}" font-size="10.5" font-weight="600">{value:.1f}</text>')
        y += (H - top - bottom) / len(bars)
    parts.append("</svg>")
    return "".join(parts)


def main() -> None:
    legend = (
        '<div class="legend">'
        f'<span class="chip" style="background:{JEV_GREEDY}"></span> Jev greedy '
        '&mdash; the single best answer Jev picked &nbsp;&nbsp; '
        f'<span class="chip" style="background:{JEV_WEIGHTED}"></span> Jev weighted '
        '&mdash; the average chance Jev itself gave to the right answer'
        '</div>')
    sections = "".join(
        f'<section><h2>{_esc(title)}</h2>'
        f'<div class="jevnote">{_esc(meta["jev_note"])}</div>'
        f'{bar_chart_svg(title, meta, show_title=False)}</section>'
        for title, meta in BENCHMARKS.items())
    intro = (
        "This page places Jev's own measured scores next to published scores for "
        "other models. Jev was run live, answering each question in a single direct "
        "shot with no step-by-step reasoning and no tools. Every other bar is a "
        "number we looked up from a public leaderboard or a vendor report, not a "
        "model we ran. Because the external numbers use different setups, these "
        "charts are for context, not a single controlled comparison; each chart "
        "says below its title how its conditions differ.")
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Jev against published benchmark scores</title>
<style>
  body {{ background:{BG}; color:{TEXT}; font-family: system-ui, sans-serif;
         max-width: 1000px; margin: 2rem auto; padding: 0 1rem; line-height: 1.5; }}
  h1 {{ font-size: 25px; margin-bottom: .3rem; }}
  h2 {{ font-size: 17px; margin: 2.2rem 0 .2rem; }}
  section svg {{ width: 100%; height: auto; display: block; }}
  p.intro {{ color:{MUTED}; font-size: 14px; max-width: 78ch; }}
  .legend {{ margin: 1rem 0 0; font-size: 13px; color:{TEXT}; }}
  .chip {{ display: inline-block; width: 14px; height: 14px; border-radius: 3px;
          vertical-align: -2px; margin-right: 6px; }}
  .jevnote {{ color:{MUTED}; font-size: 12px; margin: 0 0 .5rem; max-width: 84ch; }}
  footer {{ color:{MUTED}; font-size: 12px; margin-top: 2.5rem; }}
  code {{ background:{PANEL}; padding: 1px 5px; border-radius: 4px; }}
</style></head><body>
<h1>Jev against published benchmark scores</h1>
<p class="intro">{intro}</p>
{legend}
{sections}
<footer>Each figure's underlying scores, sources, and protocols are listed in
<code>COMPARABLE-SCORES.md</code> and the machine-readable
<code>canonical/comparable-scores.json</code>. External rows come from Vals AI,
Artificial Analysis, ARC Prize, Scale AI, model vendors, and a few aggregator
sites, each labeled there. No other model was run for this page.</footer>
</body></html>"""
    OUT.write_text(html, encoding="utf-8")
    print(f"[ok] wrote {OUT} ({OUT.stat().st_size} bytes); "
          f"{len(BENCHMARKS)} bar charts")


if __name__ == "__main__":
    main()
