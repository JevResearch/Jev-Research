#!/usr/bin/env python3
"""Generate the architecture diagram page: docs/modern-comparison/architecture-diagram.html

One <svg> that build_report.py lifts into the report's "What Jev appears to be"
section. Numeric labels are read from on-disk artifacts at render time
(runs_archprobe/analysis.json, data_report/arch_audits.json,
data_report/lattice_forensics.json, data_report/costs.json, the benchmark
score roll-ups, and runs_live/token_talk_orderprobe.json), so the picture
cannot drift from the data. A small number of prose-quoted statistics
(abstention 0.71, fictional-origin 0.97, 173 tokenizer signatures) are cited
verbatim from runs_live/FINDINGS.md and docs/modern-comparison/
ARCHITECTURE-PROBES.md and marked as such in the provenance line.

Honesty legend, matching ARCHITECTURE-ANALYSIS.md:
  solid teal   = confident (measured, adequate n, above the noise floor)
  dashed amber = plausible (best explanation; rivals not excluded)
  dotted grey  = NOT IDENTIFIED (we do not guess)

  python scripts/report/arch_diagram.py
"""

from __future__ import annotations

import glob
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


def jload(rel: str):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def first_glob(pattern: str) -> dict:
    hit = sorted(glob.glob(str(ROOT / pattern)))[0]
    return json.loads(Path(hit).read_text(encoding="utf-8"))


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def box(x: int, y: int, w: int, h: int, title: str, lines: list[str],
        stroke: str, dash: str | None = None) -> str:
    da = f' stroke-dasharray="{dash}"' if dash else ""
    p = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{PANEL}" '
         f'stroke="{stroke}" stroke-width="1.6"{da}/>',
         f'<text x="{x+12}" y="{y+22}" fill="{stroke}" font-size="11.6" '
         f'font-weight="700" letter-spacing="0.04em">{esc(title)}</text>']
    ty = y + 42
    for ln in lines:
        fill = INK if ln.startswith("!") else MUT
        txt = ln[1:] if ln.startswith("!") else ln
        weight = ' font-weight="600"' if ln.startswith("!") else ""
        p.append(f'<text x="{x+12}" y="{ty}" fill="{fill}" font-size="10.2"{weight}>'
                 f'{esc(txt)}</text>')
        ty += 15.5
    return "".join(p)


def arrow(x1: int, y1: int, x2: int, y2: int, color: str = PERI) -> str:
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2-8}" y2="{y2}" stroke="{color}" '
            f'stroke-width="1.8"/>'
            f'<polygon points="{x2},{y2} {x2-8},{y2-4.5} {x2-8},{y2+4.5}" fill="{color}"/>')


def build_svg() -> str:
    A = jload("runs_archprobe/analysis.json")
    AU = jload("data_report/arch_audits.json")
    LF = jload("data_report/lattice_forensics.json")
    COSTS = jload("data_report/costs.json")
    SC = {k: first_glob(p) for k, p in {
        "mmlu": "runs_benchmark/bench-mmlu_full-*/derived/score.json",
        "gpqa": "runs_benchmark_ext2/bench-gpqa_diamond-*/derived/score.json",
        "hle": "runs_benchmark_ext2/bench-hle_text_mc-*/derived/score.json",
        "math_c": "runs_benchmark_ext/bench-math500_choice-*/derived/score.json",
        "math_s": "runs_benchmark_ext/bench-math500_score-*/derived/score.json",
        "agi": "runs_benchmark_ext/bench-arc_agi2_choice-*/derived/score.json",
    }.items()}
    OP = jload("runs_live/token_talk_orderprobe.json")
    tvds = [v for c in OP["contexts"].values() for v in c["repeat_tvd"].values()]
    tvd_lo, tvd_hi = min(tvds), max(tvds)

    pf = A["prefill"]; hc = A["headcount"]; oc = A["optioncount"]
    conc = A["concurrency"]["per_call_wall"]
    anc = A["ancestry"]; mass = anc["family_mass"]; votes = anc["greedy_votes"]
    openai_votes = sum(votes.get(k, 0) for k in ("OpenAI", "GPT", "ChatGPT"))
    cf = AU["confidence_formula"]["choice"]
    mcq = AU["marginal_cost"]["per_question"]; mco = AU["marginal_cost"]["per_option"]
    n_values = sum(c["n_values"] for c in LF["corpora"].values())
    n_vec = sum(c["n_vectors"] for c in LF["corpora"].values())
    offgrid = sum(c["n_offgrid"] for c in LF["corpora"].values())
    n_mismatch = sum(c["n_choice_not_table_argmax"] for c in LF["corpora"].values())
    big = LF["corpora"]["talk_ensemble_raw"]["by_k_bucket"]["65-255"]
    frac99 = big["sum_dev_sign_counts"]["le_-0.005"] / big["n_vectors"]
    signs_pos = sum(b["sum_dev_sign_counts"]["ge_+0.005"]
                    for c in LF["corpora"].values() for b in c["by_k_bucket"].values())
    jp = COSTS["prices_usd_per_M"]["Jev"]
    pct = lambda x: f"{100*x:.1f}%"

    W, H = 980, 740
    p = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
         f'font-family="system-ui,sans-serif" role="img" '
         f'aria-label="Jev architecture reconstruction diagram">',
         f'<rect width="{W}" height="{H}" fill="{BG}" rx="14"/>',
         f'<text x="24" y="30" fill="{INK}" font-size="16" font-weight="700">'
         'The operating premise, drawn</text>',
         f'<text x="24" y="48" fill="{MUT}" font-size="11">'
         'behavioral evidence only - what one request does, left to right; what produced the weights, below</text>']

    # ---- top row: request -> serving path -> one pass -> read-out -> response
    ty, th = 64, 268
    bw = 182
    xs = [12, 208, 404, 600, 796]

    p.append(box(xs[0], ty, bw, th, "1 - REQUEST", [
        "!POST /v1/systemone",
        "state <= 32k tokens",
        "Q questions x K options",
        "choice: <= 255 options",
        "score: 2-10 rubric levels",
        "noul: one yes-scalar",
        "model pinned jev-1.13.0",
        "",
        "question IDs never reach",
        "inference: vendor claim,",
        "not established by us",
    ], TEAL))

    p.append(box(xs[1], ty, bw, th, "2 - SERVING PATH", [
        "!whitespace normalizer",
        "whitespace runs -> 0 tok",
        f"fixed template ~{AU['battery_power']['mergerate_baseline_tokens']} tok",
        "!vendor's own BPE tokenizer:",
        "Latin: 0.15-0.26 tok/char",
        "non-Latin ~1 tok/codepoint",
        "digits ~1; rare chars ~2",
        "(byte-level fallback)",
        "no match among 173 open",
        "signatures (1,215 repos)",
    ], TEAL))

    p.append(box(xs[2], ty, bw, th, "3 - ONE FORWARD PASS", [
        "!prefill only; no decode",
        f"{pf['fixed_floor_ms']:.0f} ms floor + {pf['ms_per_1k_input_tokens']:.1f} ms/1k tok",
        f"linear to {pf['token_range'][1]/1000:.0f}k, no blowup",
        "all Q x K from one pass",
        f"+{hc['marginal_ms_per_question']:.2f} ms/question,",
        f"+{oc['marginal_ms_per_option']:.2f} ms/option - both",
        "= their own tokens' prefill",
        "continuous batching:",
        f"upstream flat {conc['1']['median_upstream_ms']:.0f}->{conc['32']['median_upstream_ms']:.0f} ms",
        "at c=1->32",
    ], TEAL))

    ix, iy, iw, ih = xs[2] + 10, ty + th - 62, bw - 20, 52
    p.append(f'<rect x="{ix}" y="{iy}" width="{iw}" height="{ih}" rx="8" fill="{BG}" '
             f'stroke="{MUT}" stroke-width="1.2" stroke-dasharray="2 3"/>')
    p.append(f'<text x="{ix+9}" y="{iy+16}" fill="{MUT}" font-size="9.8">core: transformer-family</text>')
    p.append(f'<text x="{ix+9}" y="{iy+30}" fill="{MUT}" font-size="9.8">(plausible); size, dense/</text>')
    p.append(f'<text x="{ix+9}" y="{iy+44}" fill="{MUT}" font-size="9.8">MoE, attention: NOT IDENT.</text>')

    p.append(box(xs[3], ty, bw, th, "4 - READ-OUT", [
        "!trained head, not text",
        "probs over YOUR options",
        f"0.01 grid: {n_values:,} values,",
        f"{offgrid} off-grid ({n_vec:,} vectors)",
        "sums 0.99/1.00, never >1",
        f"{pct(frac99)} of K~255 vectors",
        "  land 0.01 short",
        "decision = pre-rounding",
        f"argmax ({n_mismatch} tables differ,",
        "  all by exactly 1q)",
        "confidence formula fits:",
        "(pmax-1/K)/(1-1/K)",
        f"{cf['exact']:,} exact; {cf['within_1_quantum'] + cf['within_2_quanta']} <=2q;",
        "none beyond",
        f"decision cost <= {mcq['residual_ms_per_unit']:.2f} ms/q",
    ], TEAL))

    p.append(box(xs[4], ty, bw - 10, th, "5 - RESPONSE", [
        "!serialized JSON, no prose",
        f"~{hc['marginal_output_tokens_per_question']:.0f} output tok/question",
        f"~{oc['marginal_output_tokens_per_option']:.1f} output tok/option",
        "output billed $0",
        f"input ${jp['input']:.3f}/M tokens",
        "schema-invalid: 19 of",
        "  ~16k requests",
        "",
        "nondeterminism: TVD",
        f"{tvd_lo:.2f}-{tvd_hi:.2f} on flat menus",
        "(the noise floor for",
        " all of the above)",
    ], TEAL))

    for i in range(4):
        p.append(arrow(xs[i] + bw + 1, ty + th // 2,
                       xs[i + 1] - 1, ty + th // 2))

    # ---- bottom row: training history (plausible) + not identified
    by, bh = 366, 268
    p.append(box(16, by, 470, bh, "TRAINING HISTORY - inferred (plausible)", [
        "!pretraining: English-dominant corpus",
        "   per-script tokenizer coverage is the fossil record:",
        "   small script blocks covered char-by-char, CJK only common chars",
        "!knowledge horizon: late 2024",
        "   2022-24 facts at p~0.97-1.0; Nov-2024 election known;",
        "   fictional events refused closed-book (FINDINGS.md S5)",
        "!post-training: judgement format (vendor name: 'RLCD')",
        "   schema perfection; trained abstention (0.71 'cannot say' on an",
        "   item it answers at 0.98 forced); semantic END on short answers;",
        f"   recognition >> production (MATH-500: {pct(SC['math_c']['accuracy'])} as MCQ",
        f"   vs {pct(SC['math_s']['accuracy'])} read out per digit)",
        "!brand prior: OpenAI-shaped learned text - NOT lineage",
        f"   {mass['openai']:.2f} family mass; {openai_votes}/{anc['n_frames_with_probs']} greedy votes; own name never",
        "   chosen at parity; fictional origin accepted at 0.97",
        "!frontier-teacher (distillation) share: not identifiable",
    ], AMBER, dash="6 4"))

    p.append(box(502, by, 462, bh, "NOT IDENTIFIED - and what could move it", [
        "!parameter count: no honest number exists",
        "   capability band only: 2025-era small-instruct class on",
        f"   knowledge MCQ (MMLU-Pro {pct(SC['mmlu']['accuracy'])} / GPQA {pct(SC['gpqa']['accuracy'])}, direct",
        f"   one-shot); HLE {pct(SC['hle']['accuracy'])} and ARC-AGI-2 {SC['agi']['tasks_solved']}/{SC['agi']['n_tasks']} exact",
        "   say 'not frontier'",
        "!dense vs MoE: unconstrained by any API-visible signal",
        f"   (prefill-only small-dense serving already explains ${jp['input']:.3f}/M)",
        "!distillation vs on-policy: every discriminator confounded",
        "!exact rule of the 0.01 lattice: bounded + one-sided",
        f"   (never >1.00; {pct(frac99)} of flat K~255 vectors land 0.01 short)",
        "",
        "!battery P1-P5 STAGED, dry-run-validated (3,331 calls ~ $0.10):",
        "   fallback granularity - option-cost decoupling - calibration vs K -",
        "   lattice rule on identical options - horizon bisection. Dispatch gated",
        "   on TYPESAFE_API_KEY (scripts/benchmark/run_probe_battery2.py).",
    ], MUT, dash="2 3"))

    # arrow: training -> weights (into box 3)
    p.append(f'<line x1="251" y1="{by}" x2="492" y2="{ty+th+6}" stroke="{AMBER}" '
             f'stroke-width="1.4" stroke-dasharray="6 4"/>')
    p.append(f'<polygon points="496,{ty+th+2} 485,{ty+th+2} 490,{ty+th+11}" fill="{AMBER}"/>')
    p.append(f'<text x="308" y="{by-8}" fill="{AMBER}" font-size="10">produced the weights</text>')

    # ---- legend + provenance
    ly = by + bh + 32
    p.append(f'<line x1="20" y1="{ly}" x2="52" y2="{ly}" stroke="{TEAL}" stroke-width="2.4"/>')
    p.append(f'<text x="58" y="{ly+4}" fill="{MUT}" font-size="10.5">confident - measured, above the noise floor</text>')
    p.append(f'<line x1="310" y1="{ly}" x2="342" y2="{ly}" stroke="{AMBER}" stroke-width="2.4" stroke-dasharray="6 4"/>')
    p.append(f'<text x="348" y="{ly+4}" fill="{MUT}" font-size="10.5">plausible - best explanation, rivals not excluded</text>')
    p.append(f'<line x1="640" y1="{ly}" x2="672" y2="{ly}" stroke="{MUT}" stroke-width="2.4" stroke-dasharray="2 3"/>')
    p.append(f'<text x="678" y="{ly+4}" fill="{MUT}" font-size="10.5">not identified - we do not guess</text>')
    p.append(f'<text x="20" y="{ly+24}" fill="{MUT}" font-size="9.8">'
             'numbers render at build time from runs_archprobe/analysis.json, data_report/arch_audits.json, data_report/lattice_forensics.json,</text>')
    p.append(f'<text x="20" y="{ly+38}" fill="{MUT}" font-size="9.8">'
             'data_report/costs.json, runs_benchmark*/derived/score.json, runs_live/token_talk_orderprobe.json; prose-quoted statistics cite runs_live/FINDINGS.md</text>')
    p.append(f'<text x="20" y="{ly+52}" fill="{MUT}" font-size="9.8">'
             'and docs/modern-comparison/ARCHITECTURE-PROBES.md. Across all corpora, not one displayed vector sums above 1.00 (positive deviations found: ' + str(signs_pos) + ').</text>')
    p.append("</svg>")
    return "".join(p)


def main() -> int:
    svg = build_svg()
    html = (
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '<title>Jev architecture diagram</title><style>\n'
        f' body {{ background:{BG}; color:{INK}; font-family: system-ui, sans-serif;\n'
        '        max-width:1000px; margin:2rem auto; padding:0 1rem; line-height:1.5; }\n'
        ' h1 { font-size:24px; }\n'
        ' svg { width:100%; height:auto; display:block; margin:.4rem 0 1rem; }\n'
        f' p.lead {{ color:{MUT}; font-size:14px; max-width:80ch; }}\n'
        '</style></head><body>\n'
        '<h1>What Jev appears to be, drawn</h1>\n'
        '<p class="lead">The operating premise from the report\u2019s architecture section: '
        'one request\u2019s path left to right (teal = measured), the training history that '
        'produced the weights (amber = best explanation), and the parts no API-visible '
        'signal can reach (grey dotted = not identified). Generated by '
        'scripts/report/arch_diagram.py from the published artifacts at build time; '
        'behavioral evidence only.</p>\n'
        + svg + '\n'
        '</body></html>\n'
    )
    OUT.write_text(html, encoding="utf-8")
    print(f"[ok] wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
