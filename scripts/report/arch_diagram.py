#!/usr/bin/env python3
"""Generate the architecture diagram page: docs/modern-comparison/architecture-diagram.html

A clean transformer-style schematic: the canonical left-to-right stack
(embeddings -> N x [attention, MLP] -> hidden state) with only Jev's two real
modifications drawn at the ends - the serving-path input stage and the option
read-out / display pipeline that replaces the language head. Short labels
only; the measurements live in the report prose and ARCHITECTURE-ANALYSIS.md,
and the few numbers shown here render from on-disk artifacts at build time.

Honesty styling, matching ARCHITECTURE-ANALYSIS.md:
  solid teal   = confident / measured
  dashed amber = plausible (best explanation / estimate)
  dotted grey  = NOT IDENTIFIED (we do not guess)

One <svg>; build_report.py lifts it into the report's architecture section.

  python scripts/report/arch_diagram.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/modern-comparison/architecture-diagram.html"

BG = "#0e1117"
PANEL = "#161b27"
INK = "#e6e9f2"
MUT = "#9aa2b6"
TEAL = "#38e1c8"
AMBER = "#f5b342"
PERI = "#7d8cff"

W, H = 980, 620


def jload(rel: str):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def box(x, y, w, h, fill=PANEL, stroke=TEAL, sw=1.6, dash=None, rx=10):
    da = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x:.0f}" y="{y:.0f}" width="{w:.0f}" height="{h:.0f}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{da}/>')


def txt(x, y, s, fill=MUT, size=11.5, anchor="middle", weight=None):
    w = f' font-weight="{weight}"' if weight else ""
    return (f'<text x="{x:.0f}" y="{y:.0f}" fill="{fill}" font-size="{size}"{w} '
            f'text-anchor="{anchor}">{esc(s)}</text>')


def arrow(x1, y1, x2, y2, color=PERI, sw=2.0):
    """Straight arrow whose head points along (x1,y1)->(x2,y2)."""
    dx, dy = x2 - x1, y2 - y1
    n = math.hypot(dx, dy) or 1.0
    ux, uy = dx / n, dy / n          # unit vector
    px, py = -uy, ux                 # perpendicular
    bx, by = x2 - ux * 9, y2 - uy * 9
    head = (f'{x2:.0f},{y2:.0f} {bx + px * 5:.0f},{by + py * 5:.0f} '
            f'{bx - px * 5:.0f},{by - py * 5:.0f}')
    return (f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{bx:.0f}" y2="{by:.0f}" '
            f'stroke="{color}" stroke-width="{sw}"/>'
            f'<polygon points="{head}" fill="{color}"/>')


def build_svg() -> str:
    A = jload("runs_archprobe/analysis.json")
    SIZE = jload("data_report/size_estimate.json")

    pf = A["prefill"]
    floor = f"{pf['fixed_floor_ms']:.0f}"
    slope = f"{pf['ms_per_1k_input_tokens']:.1f}"
    mq = A["headcount"]["marginal_ms_per_question"]
    mo = A["optioncount"]["marginal_ms_per_option"]
    conv = SIZE["reconciliation"]["converged_statement"]
    # pull the "dense-equivalent order X-YB" span out of the converged text
    import re as _re
    m = _re.search(r"dense-equivalent order (\d+)-(\d+)B", conv)
    est_lo, est_hi = (m.group(1), m.group(2)) if m else ("4", "9")

    cx = 470            # trunk centerline
    tw = 250            # trunk inner width
    ty0 = 92            # trunk top
    p = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
         f'font-family="system-ui,sans-serif" role="img" '
         f'aria-label="Hypothesized Jev architecture: transformer schematic with '
         f'modified input stage and option read-out">',
         f'<rect width="{W}" height="{H}" fill="{BG}" rx="14"/>',
         txt(W // 2, 30, "Hypothesized Architecture", INK, 16, weight="700"),
         txt(W // 2, 48, "A transformer stack, drawn plain - with the two modifications the evidence actually shows",
             MUT, 10.5)]

    # ---------------- INPUT column (left) ----------------
    ix, iw = 24, 208
    p.append(box(ix, 78, iw, 118))
    p.append(txt(ix + iw / 2, 98, "REQUEST", TEAL, 11, weight="700"))
    p.append(txt(ix + iw / 2, 120, "State + questions", INK, 11.5))
    p.append(txt(ix + iw / 2, 138, "Options: 255 or fewer each", MUT, 11))
    p.append(txt(ix + iw / 2, 156, "Choice / score / noul", MUT, 11))
    p.append(txt(ix + iw / 2, 178, "POST /v1/systemone", MUT, 9.5))

    p.append(arrow(ix + iw / 2, 196, ix + iw / 2, 224))

    p.append(box(ix, 226, iw, 150))
    p.append(txt(ix + iw / 2, 246, "SERVING PATH", TEAL, 11, weight="700"))
    p.append(txt(ix + iw / 2, 268, "Whitespace normalizer", INK, 11.5))
    p.append(txt(ix + iw / 2, 286, "Fixed template, ~316 tok", MUT, 11))
    p.append(txt(ix + iw / 2, 306, "Vendor's own tokenizer", INK, 11.5))
    p.append(txt(ix + iw / 2, 324, "Latin-centric BPE,", MUT, 11))
    p.append(txt(ix + iw / 2, 340, "byte-level fallback,", MUT, 11))
    p.append(txt(ix + iw / 2, 356, "~1 tok per non-Latin char", MUT, 11))

    p.append(arrow(ix + iw + 2, 301, cx - tw / 2 - 18, 301))

    # ---------------- TRUNK (center) ----------------
    p.append(box(cx - tw / 2 - 16, ty0 - 14, tw + 32, 402, sw=1.4))
    p.append(txt(cx, ty0 + 6, "ONE FORWARD PASS  (prefill only)", TEAL, 11.5, weight="700"))

    p.append(box(cx - tw / 2, ty0 + 22, tw, 40))
    p.append(txt(cx, ty0 + 47, "Token embeddings", INK, 12))

    p.append(arrow(cx, ty0 + 62, cx, ty0 + 84))

    # N x [attention, feed-forward]
    p.append(box(cx - tw / 2, ty0 + 86, tw, 132, sw=1.2))
    p.append(box(cx - tw / 2 + 14, ty0 + 100, tw - 28, 44))
    p.append(txt(cx, ty0 + 127, "Attention", INK, 12))
    p.append(box(cx - tw / 2 + 14, ty0 + 158, tw - 28, 44))
    p.append(txt(cx, ty0 + 185, "Feed-forward", INK, 12))
    p.append(txt(cx + tw / 2 + 24, ty0 + 157, "\u00d7 N", INK, 13, anchor="start", weight="700"))

    # loop-back arrow (residual repetition of the block)
    lx = cx - tw / 2 - 8
    p.append(f'<path d="M {cx - tw/2:.0f} {ty0+152} L {lx:.0f} {ty0+152} L {lx:.0f} {ty0+42:.0f} '
             f'L {cx - tw/2:.0f} {ty0+42:.0f}" fill="none" stroke="{MUT}" stroke-width="1.4"/>'
             f'<polygon points="{cx - tw/2 + 1:.0f},{ty0+42:.0f} {cx - tw/2 - 8:.0f},{ty0+37.5:.0f} '
             f'{cx - tw/2 - 8:.0f},{ty0+46.5:.0f}" fill="{MUT}"/>')

    p.append(arrow(cx, ty0 + 218, cx, ty0 + 244))
    p.append(box(cx - tw / 2, ty0 + 246, tw, 40))
    p.append(txt(cx, ty0 + 271, "Final hidden states h", INK, 12))
    p.append(txt(cx, ty0 + 310, "Every question + option reads from this one pass",
                 MUT, 10.5))
    p.append(txt(cx, ty0 + 326, f"Compute: {floor} ms floor + {slope} ms per 1k tokens",
                 MUT, 10.5))
    p.append(txt(cx, ty0 + 342, "Linear to 29k; deciding adds <= 0.11 ms/question",
                 MUT, 10.5))
    p.append(txt(cx, ty0 + 358, "Heavily batched (options \u00d7 parallel runs)",
                 MUT, 10.5))
    p.append(txt(cx, ty0 + 380, f"Est. active size: ~{est_lo} to {est_hi}B params",
                 AMBER, 10.2))

    # hidden states -> read-out head: elbow out of the frame and up
    p.append(f'<path d="M {cx + tw / 2:.0f} 358 L 672 358 L 672 202 L 738 202" '
             f'fill="none" stroke="{PERI}" stroke-width="2"/>')
    p.append(arrow(738, 202, 746, 202))

    # ---------------- READ-OUT column (right) ----------------
    rx0, rw = 748, 208
    p.append(box(rx0, 118, rw, 168))
    p.append(txt(rx0 + rw / 2, 138, "READ-OUT HEAD", TEAL, 11, weight="700"))
    p.append(txt(rx0 + rw / 2, 160, "Replaces the language head", MUT, 11))
    p.append(txt(rx0 + rw / 2, 182, "One distribution per question,", INK, 11.5))
    p.append(txt(rx0 + rw / 2, 198, "over the caller's options", INK, 11.5))
    p.append(txt(rx0 + rw / 2, 226, f"+{mq:.2f} ms / question", MUT, 10.5))
    p.append(txt(rx0 + rw / 2, 242, f"+{mo:.2f} ms / option", MUT, 10.5))
    p.append(txt(rx0 + rw / 2, 260, "Each pays only for its own tokens", MUT, 10.5))

    p.append(arrow(rx0 + rw / 2, 286, rx0 + rw / 2, 314))

    p.append(box(rx0, 316, rw, 110))
    p.append(txt(rx0 + rw / 2, 336, "DISPLAY PIPELINE", TEAL, 11, weight="700"))
    p.append(txt(rx0 + rw / 2, 358, "Argmax decided before rounding", INK, 11.5))
    p.append(txt(rx0 + rw / 2, 378, "Probabilities rounded to", MUT, 11))
    p.append(txt(rx0 + rw / 2, 394, "the 0.01 grid; returned sums", MUT, 11))
    p.append(txt(rx0 + rw / 2, 410, "are 0.99 or 1.00, never above", MUT, 11))

    p.append(arrow(rx0 + rw / 2, 426, rx0 + rw / 2, 484))

    p.append(box(rx0, 486, rw, 66))
    p.append(txt(rx0 + rw / 2, 508, "RESPONSE JSON", TEAL, 11, weight="700"))
    p.append(txt(rx0 + rw / 2, 530, "Serialized vectors", INK, 11.5))

    # ---------------- bottom band ----------------
    yb = 486
    p.append(box(24, yb, 424, 118, stroke=AMBER, dash="6 4"))
    p.append(txt(36, yb + 20, "WHAT MADE THE WEIGHTS (inferred - plausible)", AMBER,
                 10.8, anchor="start", weight="700"))
    p.append(txt(36, yb + 42, "English-dominant pretraining; knowledge horizon",
                 INK, 11, anchor="start"))
    p.append(txt(36, yb + 58, "solid to late 2024, partial to May 2025",
                 INK, 11, anchor="start"))
    p.append(txt(36, yb + 76, "Judgement-format post-training (vendor: RLCD);",
                 MUT, 11, anchor="start"))
    p.append(txt(36, yb + 92, "No discernible preexisting lineage;",
                 MUT, 11, anchor="start"))
    p.append(txt(36, yb + 108, "Frontier-teacher contribution: none identifiable, not excluded",
                 MUT, 11, anchor="start"))

    p.append(box(466, yb, 250, 118, stroke=MUT, dash="2 3"))
    p.append(txt(478, yb + 20, "NOT IDENTIFIED", MUT, 10.8, anchor="start", weight="700"))
    p.append(txt(478, yb + 44, "Dense vs MoE; attention type", MUT, 11, anchor="start"))
    p.append(txt(478, yb + 64, "Teacher-distilled vs trained", MUT, 11, anchor="start"))
    p.append(txt(478, yb + 80, "on its own data", MUT, 11, anchor="start"))
    p.append(txt(478, yb + 100, "Exact rule behind the 0.01 rounding", MUT, 11, anchor="start"))

    p.append("</svg>")
    return "".join(p)


def main() -> int:
    svg = build_svg()
    html = (
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '<title>Jev architecture diagram</title><style>\n'
        f' body {{ background:{BG}; color:{INK}; font-family: system-ui, sans-serif;\n'
        '        max-width:1000px; margin:2rem auto; padding:0 1rem; line-height:1.5; }}\n'
        ' h1 { font-size:24px; }\n'
        ' svg { width:100%; height:auto; display:block; margin:.4rem 0 1rem; }\n'
        f' p.lead {{ color:{MUT}; font-size:14px; max-width:80ch; }}\n'
        '</style></head><body>\n'
        '<h1>Hypothesized Architecture</h1>\n'
        '<p class="lead">The canonical transformer stack, drawn plain, with the '
        'two modifications the evidence actually shows: a serving-path input '
        'stage (normalizer, fixed template, vendor tokenizer) and an option '
        'read-out with a quantized display pipeline in place of the language '
        'head. Solid teal = measured; dashed amber = inferred; dotted grey = '
        'not identified. Detail and derivation: the report\u2019s architecture '
        'section and ARCHITECTURE-ANALYSIS.md. Generated by '
        'scripts/report/arch_diagram.py from the published artifacts.</p>\n'
        + svg + '\n'
        '</body></html>\n'
    )
    OUT.write_text(html, encoding="utf-8")
    print(f"[ok] wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
