#!/usr/bin/env python3
"""Generate the architecture diagram page: docs/modern-comparison/architecture-diagram.html

A clean transformer-style schematic: the canonical left-to-right stack
(embeddings -> N x [attention, MLP] -> read-out) with only Jev's two real
modifications drawn at the ends - the serving-path input stage and the
option read-out / display pipeline that replaces the language head. Short
labels only; every measurement lives in the report prose and
ARCHITECTURE-ANALYSIS.md, and the few numbers shown here render from the
on-disk artifacts at build time.

Honesty styling, matching ARCHITECTURE-ANALYSIS.md:
  solid teal   = confident / measured
  dashed amber = plausible (best explanation)
  dotted grey  = NOT IDENTIFIED (we do not guess)

One <svg>; build_report.py lifts it into the report's architecture section.

  python scripts/report/arch_diagram.py
"""

from __future__ import annotations

import json
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
    return (f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" stroke="{color}" '
            f'stroke-width="{sw}"/>'
            f'<polygon points="{x2:.0f},{y2:.0f} {x2-9:.0f},{y2-5:.0f} {x2-9:.0f},{y2+5:.0f}" fill="{color}"/>')


def build_svg() -> str:
    A = jload("runs_archprobe/analysis.json")
    LF = jload("data_report/lattice_forensics.json")

    pf = A["prefill"]
    floor = f"{pf['fixed_floor_ms']:.0f}"
    slope = f"{pf['ms_per_1k_input_tokens']:.1f}"
    mq = A["headcount"]["marginal_ms_per_question"]
    mo = A["optioncount"]["marginal_ms_per_option"]
    n_val = sum(c["n_values"] for c in LF["corpora"].values())
    n_vec = sum(c["n_vectors"] for c in LF["corpora"].values())
    n_mis = sum(c["n_choice_not_table_argmax"] for c in LF["corpora"].values())

    cx = 470            # trunk centerline
    tw = 250            # trunk inner width
    ty0 = 92            # trunk top
    p = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
         f'font-family="system-ui,sans-serif" role="img" '
         f'aria-label="Jev architecture: transformer schematic with modified '
         f'input stage and option read-out">',
         f'<rect width="{W}" height="{H}" fill="{BG}" rx="14"/>',
         txt(W // 2, 28, "One request through Jev", INK, 15.5, weight="700"),
         txt(W // 2, 46, "a transformer stack, drawn plain - with the two modifications the evidence actually shows",
             MUT, 10.5)]

    # ---------------- INPUT column (left) ----------------
    ix, iw = 24, 208
    p.append(box(ix, 78, iw, 118))
    p.append(txt(ix + iw / 2, 98, "REQUEST", TEAL, 11, weight="700"))
    p.append(txt(ix + iw / 2, 120, "state + questions", INK, 11.5))
    p.append(txt(ix + iw / 2, 138, "options: 255 or fewer each", MUT, 11))
    p.append(txt(ix + iw / 2, 156, "choice / score / noul", MUT, 11))
    p.append(txt(ix + iw / 2, 178, "POST /v1/systemone", MUT, 9.5))

    p.append(arrow(ix + iw / 2, 196, ix + iw / 2, 224))

    p.append(box(ix, 226, iw, 150))
    p.append(txt(ix + iw / 2, 246, "SERVING PATH", TEAL, 11, weight="700"))
    p.append(txt(ix + iw / 2, 268, "whitespace normalizer", INK, 11.5))
    p.append(txt(ix + iw / 2, 286, "fixed template, ~316 tok", MUT, 11))
    p.append(txt(ix + iw / 2, 306, "vendor's own tokenizer", INK, 11.5))
    p.append(txt(ix + iw / 2, 324, "Latin-centric BPE,", MUT, 11))
    p.append(txt(ix + iw / 2, 340, "byte-level fallback,", MUT, 11))
    p.append(txt(ix + iw / 2, 356, "~1 tok per non-Latin char", MUT, 11))

    # serving path -> trunk: clean horizontal into the pass frame
    p.append(arrow(ix + iw + 2, 301, cx - tw / 2 - 18, 301))

    # ---------------- TRUNK (center) ----------------
    p.append(box(cx - tw / 2 - 16, ty0 - 14, tw + 32, 386, sw=1.4))
    p.append(txt(cx, ty0 + 6, "ONE FORWARD PASS  (prefill only)", TEAL, 11.5, weight="700"))

    p.append(box(cx - tw / 2, ty0 + 22, tw, 40))
    p.append(txt(cx, ty0 + 47, "token embeddings", INK, 12))

    p.append(arrow(cx, ty0 + 62, cx, ty0 + 84))

    # N x [attention, feed-forward]
    p.append(box(cx - tw / 2, ty0 + 86, tw, 132, sw=1.2))
    p.append(box(cx - tw / 2 + 14, ty0 + 100, tw - 28, 44))
    p.append(txt(cx, ty0 + 127, "attention", INK, 12))
    p.append(box(cx - tw / 2 + 14, ty0 + 158, tw - 28, 44))
    p.append(txt(cx, ty0 + 185, "feed-forward", INK, 12))
    p.append(txt(cx + tw / 2 + 24, ty0 + 157, "\u00d7 N", INK, 13, anchor="start", weight="700"))

    # loop-back arrow (residual repetition of the block), drawn in the
    # corridor between the inner block and the pass frame
    lx = cx - tw / 2 - 8
    p.append(f'<path d="M {cx - tw/2:.0f} {ty0+152} L {lx:.0f} {ty0+152} L {lx:.0f} {ty0+42:.0f} '
             f'L {cx - tw/2:.0f} {ty0+42:.0f}" fill="none" stroke="{MUT}" stroke-width="1.4"/>'
             f'<polygon points="{cx - tw/2 + 1:.0f},{ty0+42:.0f} {cx - tw/2 - 8:.0f},{ty0+37.5:.0f} '
             f'{cx - tw/2 - 8:.0f},{ty0+46.5:.0f}" fill="{MUT}"/>')

    p.append(arrow(cx, ty0 + 218, cx, ty0 + 244))
    p.append(box(cx - tw / 2, ty0 + 246, tw, 40))
    p.append(txt(cx, ty0 + 271, "final hidden states h", INK, 12))
    p.append(txt(cx, ty0 + 310, "every question + option reads from this one pass",
                 MUT, 10.5))
    p.append(txt(cx, ty0 + 326, f"compute: {floor} ms floor + {slope} ms per 1k tokens",
                 MUT, 10.5))
    p.append(txt(cx, ty0 + 342, "linear to 29k; deciding adds <= 0.11 ms/question",
                 MUT, 10.5))

    # not-identified inset (dotted grey), inside the trunk frame
    p.append(box(cx - tw / 2 + 8, ty0 + 352, tw - 16, 30, fill=BG, stroke=MUT,
                 sw=1.1, dash="2 3", rx=8))
    p.append(txt(cx, ty0 + 371, "N, width, size, dense/MoE: NOT IDENTIFIED", MUT, 9.8))

    # final hidden states -> read-out head: elbow out of the frame and up
    p.append(f'<path d="M {cx + tw / 2:.0f} 358 L 672 358 L 672 202 L 738 202" '
             f'fill="none" stroke="{PERI}" stroke-width="2"/>')
    p.append(arrow(738, 202, 746, 202))

    # ---------------- READ-OUT column (right) ----------------
    rx0, rw = 748, 208
    p.append(box(rx0, 118, rw, 168))
    p.append(txt(rx0 + rw / 2, 138, "READ-OUT HEAD", TEAL, 11, weight="700"))
    p.append(txt(rx0 + rw / 2, 160, "replaces the language head", MUT, 11))
    p.append(txt(rx0 + rw / 2, 182, "scores caller options", INK, 11.5))
    p.append(txt(rx0 + rw / 2, 200, "distribution per question", MUT, 11))
    p.append(txt(rx0 + rw / 2, 226, f"+{mq:.2f} ms / question", MUT, 10.5))
    p.append(txt(rx0 + rw / 2, 242, f"+{mo:.2f} ms / option", MUT, 10.5))
    p.append(txt(rx0 + rw / 2, 260, "= their own tokens' prefill", MUT, 10.5))

    p.append(arrow(rx0 + rw / 2, 286, rx0 + rw / 2, 314))

    p.append(box(rx0, 316, rw, 120))
    p.append(txt(rx0 + rw / 2, 336, "DISPLAY PIPELINE", TEAL, 11, weight="700"))
    p.append(txt(rx0 + rw / 2, 358, "argmax before rounding", INK, 11.5))
    p.append(txt(rx0 + rw / 2, 376, "probs onto the 0.01 grid", MUT, 11))
    p.append(txt(rx0 + rw / 2, 394, "sums 0.99 / 1.00, never >1", MUT, 11))
    p.append(txt(rx0 + rw / 2, 412, "confidence: shape-derived", MUT, 11))
    p.append(txt(rx0 + rw / 2, 428, f"{n_mis} choice-vs-table seams, all 1q", MUT, 9.8))

    p.append(arrow(rx0 + rw / 2, 436, rx0 + rw / 2, 484))

    p.append(box(rx0, 486, rw, 76))
    p.append(txt(rx0 + rw / 2, 506, "RESPONSE JSON", TEAL, 11, weight="700"))
    p.append(txt(rx0 + rw / 2, 528, "serialized vectors;", INK, 11.5))
    p.append(txt(rx0 + rw / 2, 546, "output billed $0", MUT, 11))

    # ---------------- bottom band ----------------
    yb = 486
    p.append(box(24, yb, 424, 104, stroke=AMBER, dash="6 4"))
    p.append(txt(36, yb + 20, "WHAT MADE THE WEIGHTS (inferred - plausible)", AMBER,
                 10.8, anchor="start", weight="700"))
    p.append(txt(36, yb + 42, "English-dominant pretraining; horizon late 2024",
                 INK, 11, anchor="start"))
    p.append(txt(36, yb + 60, "judgement-format post-training (vendor: RLCD);",
                 MUT, 11, anchor="start"))
    p.append(txt(36, yb + 76, "OpenAI-shaped brand prior = learned text, not lineage",
                 MUT, 11, anchor="start"))
    p.append(txt(36, yb + 94, "frontier-teacher contribution: not identifiable",
                 MUT, 11, anchor="start"))
    # dashed arrow up into the trunk
    p.append(f'<line x1="236" y1="{yb}" x2="374" y2="{ty0 + 392}" stroke="{AMBER}" '
             f'stroke-width="1.3" stroke-dasharray="6 4"/>')
    p.append(f'<polygon points="378,{ty0 + 388} 366,{ty0 + 388} 372,{ty0 + 397}" fill="{AMBER}"/>')

    p.append(box(466, yb, 250, 104, stroke=MUT, dash="2 3"))
    p.append(txt(478, yb + 20, "NOT IDENTIFIED", MUT, 10.8, anchor="start", weight="700"))
    p.append(txt(478, yb + 42, "parameter count (banded in report)", MUT, 11, anchor="start"))
    p.append(txt(478, yb + 60, "dense vs MoE; attention type", MUT, 11, anchor="start"))
    p.append(txt(478, yb + 78, "distillation vs on-policy", MUT, 11, anchor="start"))
    p.append(txt(478, yb + 96, "exact lattice rounding rule", MUT, 11, anchor="start"))

    # legend row: between trunk bottom (~464) and bottom band (486)
    ly = 478
    p.append(f'<line x1="238" y1="{ly}" x2="266" y2="{ly}" stroke="{TEAL}" stroke-width="2.4"/>')
    p.append(txt(274, ly + 4, "measured / confident", MUT, 10, anchor="start"))
    p.append(f'<line x1="424" y1="{ly}" x2="452" y2="{ly}" stroke="{AMBER}" stroke-width="2.4" stroke-dasharray="6 4"/>')
    p.append(txt(460, ly + 4, "plausible (inferred)", MUT, 10, anchor="start"))
    p.append(f'<line x1="608" y1="{ly}" x2="636" y2="{ly}" stroke="{MUT}" stroke-width="2.4" stroke-dasharray="2 3"/>')
    p.append(txt(644, ly + 4, "not identified", MUT, 10, anchor="start"))

    # provenance, in the left column's dead space above the amber panel
    p.append(txt(28, 408, f"lattice: {n_val:,} values / {n_vec:,} vectors,",
                 MUT, 9.3, anchor="start"))
    p.append(txt(28, 422, "zero off-grid (lattice_forensics.json)",
                 MUT, 9.3, anchor="start"))

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
        '<h1>What Jev appears to be, drawn</h1>\n'
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
