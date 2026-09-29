#!/usr/bin/env python3
"""Build the public report: report/index.html (single page, GitHub-Pages-ready).

Every figure is read from on-disk artifacts at render time (nothing hand-typed);
chart SVGs are lifted from the generated standalone pages so the report never
duplicates a heading or drifts from the underlying data.

  python scripts/report/build_report.py

Footnote prose lives in NOTES below; inline markers are only ever `{fn('key')}`
(putting long quoted prose inside f-string braces is a syntax error and was the
bug this layout avoids).
"""

from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path

# JEVO_ROOT lets the publish bundle re-render and verify the page against the
# aggregates it ships (scripts/report/build_site.py runs exactly this).
ROOT = Path(os.environ.get("JEVO_ROOT") or Path(__file__).resolve().parents[2])
SITE = ROOT / "report"
GH = "https://github.com/JevResearch/Jev-Research/blob/main"


def jload(rel: str):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def first_glob(pattern: str) -> str:
    hits = sorted(Path(p) for p in glob.glob(str(ROOT / pattern)))
    if not hits:
        raise FileNotFoundError(pattern)
    return str(hits[0].relative_to(ROOT))


def svgs_of(rel: str) -> list[str]:
    return [m.group(0) for m in
            re.finditer(r"<svg.*?</svg>", (ROOT / rel).read_text(encoding="utf-8"), re.S)]


def score(pattern: str) -> dict:
    return jload(first_glob(pattern))


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def wil(d: dict) -> str:
    w = d.get("wilson_95") or {}
    return f"[{100*w['low']:.1f}, {100*w['high']:.1f}]" if w.get("low") is not None else ""


# ---------------------------------------------------------------- data loading
SC = {s: score(p) for s, p in {
    "mmlu": "runs_benchmark/bench-mmlu_full-*/derived/score.json",
    "arc": "runs_benchmark/bench-arc_test-*/derived/score.json",
    "rot": "runs_benchmark/bench-option_rotations-*/derived/score.json",
    "math_c": "runs_benchmark_ext/bench-math500_choice-*/derived/score.json",
    "math_s": "runs_benchmark_ext/bench-math500_score-*/derived/score.json",
    "gpqa": "runs_benchmark_ext2/bench-gpqa_diamond-*/derived/score.json",
    "hle": "runs_benchmark_ext2/bench-hle_text_mc-*/derived/score.json",
}.items()}
WD = {s: score(p) for s, p in {
    "mmlu": "runs_benchmark/bench-mmlu_full-*/derived/weighted_score.json",
    "arc": "runs_benchmark/bench-arc_test-*/derived/weighted_score.json",
    "rot": "runs_benchmark/bench-option_rotations-*/derived/weighted_score.json",
    "math_c": "runs_benchmark_ext/bench-math500_choice-*/derived/weighted_score.json",
    "math_s": "runs_benchmark_ext/bench-math500_score-*/derived/weighted_score.json",
    "gpqa": "runs_benchmark_ext2/bench-gpqa_diamond-*/derived/weighted_score.json",
    "hle": "runs_benchmark_ext2/bench-hle_text_mc-*/derived/weighted_score.json",
    "agi_choice": "runs_benchmark_ext/bench-arc_agi2_choice-*/derived/weighted_score.json",
    "agi_task": "runs_benchmark_ext/bench-arc_agi2_task-*/derived/weighted_score.json",
}.items()}
ARCAGI = score("runs_benchmark_ext/bench-arc_agi2_choice-*/derived/score.json")
ARCAGI_T = score("runs_benchmark_ext/bench-arc_agi2_task-*/derived/score.json")
ARCAGI_S = score("runs_benchmark_ext/bench-arc_agi2_score-*/derived/score.json")
# Cell-level contract failures in the multi-question ARC-AGI stages (the
# per-item strict_format_failure keys there are definitional, not violations;
# see the note_on_strict_format fields and ARCHITECTURE-ANALYSIS.md §11.15).
AGI_UNUSABLE = sum(s.get("cells_unusable", 0) for s in (ARCAGI, ARCAGI_T, ARCAGI_S))
try:
    BILLING = jload("data_report/billing_totals.json")
except FileNotFoundError:
    raise SystemExit("data_report/billing_totals.json missing - run "
                     "scripts/report/aggregate_billing.py first")
# Total planned benchmark requests across the three freezes (the "questions
# we ran" figure; published as aggregates + hashes, item text stays licensed).
TOTAL_REQUESTS = sum(jload(f"runs_{d}/freeze/frozen.json").get("total_requests", 0)
                     for d in ("benchmark", "benchmark_ext", "benchmark_ext2"))
ARCH = jload("runs_archprobe/analysis.json")
PCC = jload("runs_archprobe/tokens_per_char_compare.json")
FINGER = jload("runs_archprobe/tokenizer_fingerprint.json")
COSTS = jload("data_report/costs.json")
VALS = jload("docs/modern-comparison/canonical/vals-leaderboards-20260926.json")
MATCHED = {}
for _mp in ("runs_matched_cheap/v3/summary_v3.json",
            "runs_matched_cheap/v2/summary_v2.json"):
    try:
        MATCHED = jload(_mp)
        break
    except Exception:
        continue
try:
    AUDITS = jload("data_report/arch_audits.json")
except FileNotFoundError:
    raise SystemExit("data_report/arch_audits.json missing - run "
                     "scripts/report/arch_audits.py first")
try:
    LATTICE = jload("data_report/lattice_forensics.json")
except FileNotFoundError:
    raise SystemExit("data_report/lattice_forensics.json missing - run "
                     "scripts/report/lattice_forensics.py first")
try:
    PROBEPLAN = jload("data_report/probe2_plan.json")
except FileNotFoundError:
    raise SystemExit("data_report/probe2_plan.json missing - run "
                     "scripts/benchmark/run_probe_battery2.py --plan first")
P2RUN = jload("runs_archprobe/probe2/probe2_analysis.json")
P2BILL = jload("runs_archprobe/probe2/BILLING-probe2.json")
JEVBOT = jload("data_report/jevbot_examples.json")
try:
    SIZE = jload("data_report/size_estimate.json")
except FileNotFoundError:
    raise SystemExit("data_report/size_estimate.json missing - run "
                     "scripts/report/size_estimate.py first")

# Jev per-benchmark cost: measured from the runs' own billing usage summaries
JEV_COST = {b: SC[s]["attempts_summary"]["usage_input_tokens_sum"] * 0.042 / 1e6
            for b, s in {"mmlu_pro": "mmlu", "gpqa": "gpqa", "arc": "arc",
                         "math500": "math_c", "hle": "hle"}.items()}

_BENCH_SVGS = svgs_of("docs/modern-comparison/comparison-graphs.html")
_CH_KEYS = [("MMLU-Pro -", "mmlu_pro"), ("GPQA -", "gpqa"),
            ("ARC-Challenge", "arc"), ("ARC-AGI-2", "arc_agi2"),
            ("MATH-500 as multiple choice - matched", "math_m"),
            ("Humanity", "hle_m")]
CHARTS = {}
for _svg in _BENCH_SVGS:
    _m = re.search(r'aria-label="([^"]+)"', _svg)
    _lab = _m.group(1) if _m else ""
    for _pref, _key in _CH_KEYS:
        if _lab.startswith(_pref):
            CHARTS[_key] = _svg
            break
PARETO = svgs_of("docs/modern-comparison/pareto-frontiers.html")
EVID = svgs_of("docs/modern-comparison/architecture-evidence.html")
ARCHDIAG = svgs_of("docs/modern-comparison/architecture-diagram.html")




# Chart titles/subtitles are the only <text> elements anchored at x="24" with
# a small y; data labels never use x="24", so this removes exactly the headings.
_HDR_TEXT = re.compile(r'<text x="24" y="(?:[1-9][0-9]|1[0-1][0-9])"[^>]*>.*?</text>')


def strip_titles(svg: str) -> str:
    """Drop a chart's baked-in title/subtitle texts (y in the header band).

    The report supplies its own single heading per figure, so the embedded chart
    must not repeat it - that duplication was the bug being fixed here.
    """
    return _HDR_TEXT.sub("", svg)


def fig(key: str, caption: str) -> str:
    return f'<figure>{CHARTS[key]}<figcaption>{caption}</figcaption></figure>'

# ------------------------------------------------------------------- footnotes
NOTES: dict[str, str] = {
    "almeida": "InstructGPT, of which Almeida was a coauthor, is best known "
        "for applying RLHF to GPT-3 and establishing a standard three-stage "
        "training pipeline, subsequently used in ChatGPT. However, RLHF did "
        "not originate with InstructGPT. It was first formulated by Russell "
        "and Ng (1998-2000), demonstrated through TAMER (2008-2012), PbRL "
        "(2011-2012), advanced with the 2017 seminal &ldquo;Deep "
        "Reinforcement Learning from Human Preferences&rdquo; (Paul "
        "Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and "
        "Dario Amodei), transitioning to natural language in Zeigler et al "
        "(2019) and Stiennon et al (2020), before being applied to GPT-3 "
        "with InstructGPT. GPT-3 in turn was enabled by numerous "
        "technologies: dense vector representations (Word2Vec, GloVe), "
        "subword tokenization / BPE, contextualized embeddings (ELMo), "
        "recurrent neural networks themselves, LSTMs and GRUs to solve the "
        "vanishing gradient problem, seq2seq, additive and scaled attention, "
        "the seminal work on Transformers (2017), autoregressive decoder "
        "models, self-supervised pretraining, transfer learning, empirical "
        "scaling laws, in-context learning / few-shot prompting, SFT, "
        "mixed-precision training, distributed "
        "parallelism frameworks (Megatron-LM, pipeline parallelism, ZeRO / "
        "DeepSpeed), and many more. &ldquo;I co-invented&rdquo; implies one "
        "of a small subset; ChatGPT has thousands of fathers.",
    "jevfree": "Jev's responses report an &ldquo;output&rdquo; token count in "
        "usage - the serialized response JSON, counted but never billed; every "
        "output token is priced at zero. Every Jev dollar figure in this "
        "report is measured from billing usage in the published run artifacts, "
        "not estimated.",
    "matchedbase": "Matched cheap-model baselines: "
        "scripts/benchmark/run_matched3.py (v3, spec-compliant) over "
        "run_cheap_matched2.py (v2); artifacts in runs_matched_cheap/ "
        "(v3/ preferred, v2/ kept for provenance). Identical frozen items to "
        "the Jev runs (MMLU-Pro 1,000-item stratified subset, seed 20260920; "
        "GPQA 196; MATH-500 choice 261; HLE MC 494; ARC-Challenge 1,172 for "
        "several models), one attempt per item, transport-level backoff only, "
        "provider-reported costs. v3 wire policy: NO temperature, top_p, seed "
        "or reasoning parameters - each model runs at its documented provider "
        "defaults, per the audit's out-of-spec warning; max_tokens is a bill "
        "guard, not a thinking budget. (v2 pinned temperature 0 + effort low, "
        "which is out of spec for reasoning models and depressed their scores "
        "and costs; v2 cells survive only where v3 did not re-run - Qwen3.8 "
        "Max MMLU - and are labeled in the summary.) Answers are parsed "
        "strictly and, failing that, recovered by a documented deterministic "
        "parser applied identically to every model; unrecovered answers count "
        "as wrong. The cost gap versus vals.ai rows for the same model is "
        "protocol, not metering: vals runs 5-shot chain-of-thought at "
        "max effort (2-50x the tokens), ours are one-shot at defaults - both "
        "figures are what the providers actually billed. Budget $80 total "
        "authorized by the project owner; spend in the live_summary files. "
        "Era colors use curated/estimated release dates, labeled in tooltips.",
    "costest": "External costs are the vals.ai platform's measured cost per "
        "test - the same harness that measured the scores, chain-of-thought "
        "tokens included, so reasoning counts against the models that use it. "
        "Jev's costs are measured from our billing (reported input tokens x "
        "$0.042/M; output free). The earlier draft's list-price estimate "
        "method and its arithmetic remain in data_report/costs.json for "
        "provenance; the charts no longer use it.",
    "protocol": "External scores use their publishers' protocols, which are "
        "not Jev's: Vals MMLU-Pro is 5-shot with chain-of-thought; Artificial "
        "Analysis GPQA runs with reasoning enabled; ARC-AGI-2 rows are "
        "grid-production pass@2 on the semi-private set; Scale HLE rows cover "
        "the full text-only subset, partly with tools. Jev answered directly, "
        "with no reasoning and no tools. The charts show position, not a "
        "matched race.",
    "mathadapt": "MATH-500 and ARC-AGI-2 are protocol conversions, not native "
        "runs: Jev cannot generate a worked solution or a completed grid, so "
        "those problems were re-encoded as multiple-choice and per-cell "
        "questions. Those numbers say what the model knows, not what it can "
        "produce.",
    "greedyw": "Two scorings of the same responses. Greedy takes the model's "
        "argmax option. The probability-weighted column is the average "
        "probability Jev itself put on the correct answer, so it reads as the "
        "model's own expected accuracy. Greedy exceeding weighted nearly "
        "everywhere (measured on every benchmark) says Jev's distributions "
        "spread mass onto wrong options more than their accuracy warrants.",
    "rotations": "A robustness audit: the same items re-run with option order "
        "rotated, to check the headline numbers are not an artifact of where "
        "the answer sits in the list.",
    "noversioning": "Every run used the pinned model string reported by the "
        "service; TypeSafe AI does not publish per-version behavioral notes, "
        "so all measurements describe whatever served that string during our "
        "runs in September 2026.",
    "behavioronly": "No attempt was made to inspect, extract, or reconstruct "
        "the model itself - no weights, no gradients, no serving internals. "
        "Everything here comes from what the public API returns: its answers, "
        "its probability vectors, its token counts, its latency headers, and "
        "its billing usage.",
    "quantization": "Every probability returned across all runs landed exactly "
        "on a 0.01 grid - thousands of values, zero off-grid. That is a "
        "fixed-precision read-out, not raw token-level logits; it also means "
        "any reasoning built on tiny probability differences is reading noise.",
    "ordering": "Ordering numbers: <a href='https://github.com/JevResearch/Jev-Research/blob/main/runs_live/token_talk_orderprobe.json'>runs_live/token_talk_orderprobe.json</a> "
        "(11 orderings x 6 repeats of a frozen 254-option step) and "
        "docs/token-talk-findings.md sections 11-12; the rotation-ensemble "
        "result (K>=6 cancels the bias, Spearman 0.91-1.00 against the "
        "12-rotation reference) is section 12. The repeat-noise band "
        "(TVD 0.03-0.12) is measured in the same file. The 419-item "
        "shuffle audit is runs_benchmark/bench-option_rotations-* paired "
        "against the native run (data_report/arch_audits.json).",
    "readout": "The probability surface is post-processed, not raw. Every "
        "value sits on the 0.01 grid; nine published vectors return a choice "
        "that is not the argmax of their own displayed table, always at "
        "exactly one quantum - round-to-nearest cannot reorder options, so "
        "the decision is computed at pre-display precision; and across 2,289 "
        "published choice vectors the confidence field equals the "
        "chance-corrected top probability (p_max - 1/K) / (1 - 1/K) to within "
        "two quanta (1,405 exactly, none beyond). Score answers use a "
        "different shape statistic; noul answers carry no confidence at all. "
        "Re-derived from the published raw responses by "
        "scripts/report/arch_audits.py -&gt; data_report/arch_audits.json.",
    "sizeest": "The size-estimate machinery lives in "
        "data_report/size_estimate.json (scripts/report/size_estimate.py): "
        "both angles, the full assumption grid (MFU 0.25-0.45; the late-2026 "
        "fleet's BF16 nameplate ~400-2250 TFLOPS per device times a "
        "quantization multiplier of 2-4x (fp8/int4-FP4) = ~800-9000 "
        "effective; 1-4 shards; FLOPs = 2*N_active per token, attention "
        "adding ~15% or less at these lengths; single-stream conservative), "
        "the reconciling readings (quantized dense / MoE / distilled small "
        "dense), and what would tighten each. Two earlier passes were "
        "wrong, both labeled in the file: one assumed bf16 on A100-class "
        "hardware (~2-4x too low); the second treated the fleet's BF16 "
        "numbers as already effective (~2-4x too low again) - the audit "
        "caught it. The MFU/peak ranges are industry-standard "
        "serving assumptions, not repo measurements.",
    "lattice": "Full lattice forensics: <a href='https://github.com/JevResearch/Jev-Research/blob/main/scripts/report/lattice_forensics.py'>scripts/report/lattice_forensics.py</a> "
        "-&gt; <a href='https://github.com/JevResearch/Jev-Research/blob/main/data_report/lattice_forensics.json'>data_report/lattice_forensics.json</a>, over 704,277 values in 7,887 "
        "published vectors at K=2..255 (probe rows, phase-1 raws, and the "
        "Talk program's per-rotation large-menu distributions). Zero "
        "off-grid values; displayed sums are only ever 0.99 or 1.00 and "
        "never exceed 1.00, against the +-0.04 scatter independent per-value "
        "rounding would give at K=255 - so the display pipeline applies a "
        "bounded one-sided correction (or apportions integer hundredths), "
        "leaving a 0.01 shortfall on ~46% of flat large-K vectors. Every "
        "two-option vector sums to exactly 1.000 (complement emission). The grid "
        "is a rounding step, not a floor: probabilities below 0.005 display "
        "as exactly 0.00, and they are common. The live identical-option "
        "probe (runs_archprobe/probe2/) confirmed sums of exactly 1.000 at "
        "every option count up to 12 and 0.99-1.00 at 255; the exact "
        "apportionment rule at the boundaries remains open.",
    "notwrapper": "Each negation is a measurement, not a vibe. Compute grows "
        "linearly with prompt tokens - a cache would not prefill - and the "
        "knowledge horizon stops at mid-2025 (dated-bisection probe), which a live "
        "retriever would "
        "not. Repeats of identical payloads differ (TVD 0.03-0.12), which a "
        "lookup would not. Server compute stays 73-86 ms from concurrency "
        "1-32, and four batched questions answer in ~261 ms vs ~1,117 ms "
        "separately - no upstream API round trip fits inside either. The "
        "usage counter follows a tokenizer matching none of 173 open "
        "signatures, including both OpenAI encodings. And $0.042/M with free "
        "output sits one to two orders of magnitude below every 2026 "
        "flagship's input list price (roughly 30-240x, before counting "
        "output they bill at 4-5x their input); reselling frontier inference "
        "on those terms does not survive. Paths: "
        "<a href='https://github.com/JevResearch/Jev-Research/blob/main/runs_archprobe'>runs_archprobe/</a>, runs_live/FINDINGS.md, "
        "data_report/costs.json.",
}
_NUM: dict[str, int] = {}
_OCC: dict[str, int] = {}


def fn(key: str) -> str:
    """Numbered footnote marker; unique anchor id on every use of the same note."""
    if key not in _NUM:
        _NUM[key] = len(_NUM) + 1
    _OCC[key] = _OCC.get(key, 0) + 1
    anchor = f' id="fnref-{key}"' if _OCC[key] == 1 else f' id="fnref-{key}-{_OCC[key]}"'
    return f'<sup><a href="#ref-{key}"{anchor}>{_NUM[key]}</a></sup>'


def refs() -> str:
    items = "".join(
        f'<li id="ref-{k}">{NOTES[k]} <a href="#fnref-{k}">[back]</a></li>'
        for k, n in sorted(_NUM.items(), key=lambda kv: kv[1]))
    return f'<h2 id="refs">Notes</h2><ol>{items}</ol>'

# ---------------------------------------------------------------- presentation
CSS = """
:root{--bg:#0e1117;--panel:#161b27;--ink:#e6e9f2;--mut:#9aa2b6;--line:#242a38;
--teal:#38e1c8;--amber:#f5b342;--peri:#7d8cff}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);margin:0;font:16px/1.68 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:880px;margin:0 auto;padding:0 1.4rem 6rem}
h1{font-size:2.3rem;line-height:1.15;margin:3.4rem 0 .35rem;letter-spacing:-.015em}
h2{font-size:1.45rem;margin:3.6rem 0 .6rem;border-bottom:1px solid var(--line);padding-bottom:.35rem}
h3{font-size:1.1rem;margin:2rem 0 .4rem}
p,li{max-width:none}a{color:var(--peri)}
.sub{color:var(--mut);font-size:1.05rem;max-width:none}
.chip{display:inline-block;font-size:.72rem;letter-spacing:.05em;text-transform:uppercase;color:var(--teal);border:1px solid var(--teal);border-radius:999px;padding:.1rem .6rem;margin-bottom:1.1rem}
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:1.05rem 1.3rem;margin:1.1rem 0}
.pitch{font-size:1.16rem;font-style:italic;border-left:3px solid var(--amber);padding-left:1.1rem;margin:1.1rem 0}
.pitch b{font-style:normal}
svg{width:100%;height:auto;display:block;margin:1rem 0}
figure{margin:1.6rem 0;width:100%}
figure svg{margin:.4rem 0}
figcaption{color:var(--mut);font-size:.8rem;font-style:italic;margin:.3rem 0 0;line-height:1.45}
@media(min-width:900px){
figure{width:min(96vw,1120px);margin-left:calc(50% - min(48vw,560px));margin-right:calc(50% - min(48vw,560px))}
figcaption{max-width:72ch;margin-left:auto;margin-right:auto}
}
table{border-collapse:collapse;width:100%;font-size:.92rem;margin:1rem 0}
th,td{text-align:left;padding:.45rem .6rem;border-bottom:1px solid var(--line)}
th{color:var(--mut);font-weight:600;font-size:.78rem;text-transform:uppercase;letter-spacing:.04em}
td.n{text-align:right;font-variant-numeric:tabular-nums}
.cap{color:var(--mut);font-size:.88rem;max-width:none}
.note{border-left:3px solid var(--amber);background:var(--panel);border-radius:0 12px 12px 0;padding:.9rem 1.1rem;margin:1.2rem 0}
sup a{text-decoration:none;color:var(--amber);font-weight:600}
#refs li{margin:.4rem 0;font-size:.88rem;color:var(--mut)}
footer{color:var(--mut);font-size:.85rem;border-top:1px solid var(--line);margin-top:4rem;padding-top:1.2rem}
code{background:var(--panel);border-radius:4px;padding:.05rem .35rem;font-size:.86em}
.scroll{overflow-x:auto}
.herorot{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:1.25rem 1.4rem 1rem;margin:1.3rem 0}
.rot-slide{display:none}
.rot-slide.active{display:block;animation:rotin .6s cubic-bezier(.4,0,.2,1)}
@keyframes rotin{from{opacity:0;transform:translateX(26px)}to{opacity:1;transform:none}}
.rot-name{color:var(--mut);font-size:.85rem;letter-spacing:.06em;text-transform:uppercase}
.rot-score{font-size:2.7rem;font-weight:700;color:var(--teal);line-height:1.15;font-variant-numeric:tabular-nums}
.rot-note{color:var(--mut);font-size:.9rem;max-width:70ch}
.rot-frontier{color:var(--peri);font-size:.88rem;margin-top:.3rem}
.rot-ctl{display:flex;gap:.4rem;align-items:center;margin-top:.75rem;opacity:.45;transition:opacity .25s}
.herorot:hover .rot-ctl{opacity:1}
.rot-btn{background:none;border:1px solid var(--line);color:var(--mut);border-radius:8px;width:30px;height:26px;cursor:pointer;font-size:.95rem;line-height:1;padding:0}
.rot-btn:hover{color:var(--ink);border-color:var(--mut)}
.rot-count{color:var(--mut);font-size:.8rem;margin-left:.35rem;font-variant-numeric:tabular-nums}
nav.toc{margin:1.5rem 0 .3rem;font-size:.98rem;color:var(--mut);border-top:1px solid var(--line);padding-top:1rem}
nav.toc a{color:var(--peri);text-decoration:none}
nav.toc a:hover{text-decoration:underline}
.toc-h{font-weight:700;color:var(--ink);font-size:1.02rem;margin-bottom:.55rem;letter-spacing:.02em}
.toc-sec{margin:.42rem 0}
.toc-sub{font-size:.85rem;color:var(--mut);margin:.15rem 0 .1rem 1.1rem;line-height:1.6}
.toc-sub a{color:var(--mut)}
.toc-sub a:hover{color:var(--peri)}
.tldr{border-left:3px solid var(--teal)}
table.plain th{text-transform:none;font-size:.82rem;letter-spacing:.01em}
table.wide{font-size:.78rem;min-width:720px}
table.wide th{font-size:.66rem}
"""


COMPS = jload("docs/modern-comparison/canonical/comparable-scores.json")

# hero rotator: every benchmark stage we ran, one at a time, with the best
# published reference per benchmark (model, score, protocol) from the
# canonical comparable-scores file. Stage order matches the benchmark table.
_ROT_STAGES = [
    ("mmlu", "MMLU-Pro", "12,032 graduate-level multiple-choice questions", "mmlu_pro"),
    ("arc", "ARC-Challenge", "1,172 grade-school science items - a saturated field", "arc_challenge"),
    ("gpqa", "GPQA Diamond", "196 graduate-level science questions", "gpqa_diamond"),
    ("hle", "Humanity's Last Exam (MC)", "494 expert-exam multiple-choice items", "hle_text_only"),
    ("math_c", "MATH-500 as multiple choice", "261 encodable items, answer as an option", "math500"),
    ("math_s", "MATH-500 digit read-out", "273 items, every digit right or wrong", None),
    (None, "ARC-AGI-2, per cell", "70,100 abstract-puzzle cells - a diagnostic encoding", None),
    (None, "ARC-AGI-2, exact grid", "120 tasks, all cells of all 167 grids correct", "arc_agi2"),
]


_PROTO_HUMAN = {"cot_5shot": "5-shot CoT", "cot_25shot": "25-shot CoT",
                "reasoning": "reasoning enabled", "tools": "with tools",
                "direct": "direct answers", "self_reported": "self-reported",
                "aggregator": "aggregator import",
                "reasoning_pass2_semiprivate": "reasoning, pass@2, semi-private set"}


def _frontier(comp_key: str | None) -> str:
    if not comp_key:
        return ""
    rows = [r for r in COMPS["benchmarks"].get(comp_key, {}).get("rows", [])
            if r.get("score") is not None]
    if not rows:
        return ""
    top = max(rows, key=lambda r: r["score"])
    proto = _PROTO_HUMAN.get(top.get("protocol") or "",
                             top.get("protocol") or "protocol unknown")
    return f"{top['model']} {top['score']*100:.1f}% ({proto})"


def hero() -> str:
    hle_w = pct(WD["hle"]["weighted_accuracy_mean"])
    slides = []
    agi_c = pct(ARCAGI["cell_accuracy_diagnostic"])
    agi_t = pct(ARCAGI["task_accuracy"])
    for i, (key, name, note, comp) in enumerate(_ROT_STAGES):
        if key is not None:
            score = pct(SC[key]["accuracy"])
        elif "per cell" in name:
            score = agi_c
        else:
            score = agi_t
        fr = _frontier(comp)
        fr_html = (f'<div class="rot-frontier">best published: {fr}</div>'
                   if fr else '<div class="rot-frontier">no protocol-matched reference</div>')
        slides.append(f"""<div class="rot-slide{' active' if i == 0 else ''}">
<div class="rot-name">{name}</div>
<div class="rot-score">{score}</div>
<div class="rot-note">{note}</div>
{fr_html}
</div>""")
    toc = """
<nav class="toc">
<div class="toc-h">In this report</div>
<div class="toc-sec"><a href="#pitch">1 &middot; The pitch, and the parts that survive contact</a></div>
<div class="toc-sec"><a href="#method">2 &middot; What we did, and what we could not do</a></div>
<div class="toc-sec"><a href="#arch">3 &middot; What Jev appears to be</a>
<div class="toc-sub"><a href="#arch-onepass">One pass, many read-outs</a> &middot; <a href="#arch-fingerprints">Fingerprints on the read-out</a> &middot; <a href="#arch-order">The order of the options matters</a> &middot; <a href="#arch-tokenizer">A tokenizer nobody recognizes</a> &middot; <a href="#arch-capacity">Limited capacity, in a particular way</a> &middot; <a href="#arch-not">What it is not</a> &middot; <a href="#arch-open">What stays open</a></div></div>
<div class="toc-sec"><a href="#latency">4 &middot; What the milliseconds say</a></div>
<div class="toc-sec"><a href="#benchmarks">5 &middot; Benchmarks: where it actually lands</a>
<div class="toc-sub"><a href="#bench-charts">The charts</a></div></div>
<div class="toc-sec"><a href="#pareto">6 &middot; The frontier the cost numbers actually draw</a></div>
<div class="toc-sec"><a href="#probes">7 &middot; What is it? Three attempts to ask</a>
<div class="toc-sub"><a href="#probes-talk">Talking to it</a> &middot; <a href="#probes-name">Asking it to choose a name</a> &middot; <a href="#probes-tokens">Counting tokens</a></div></div>
<div class="toc-sec"><a href="#keynotes">8 &middot; Key notes for Jev users</a></div>
<div class="toc-sec"><a href="#conclusion">9 &middot; So, is it worth your attention?</a></div>
<div class="toc-sec"><a href="#refs">Notes</a></div>
</nav>"""
    return f"""
<div class="chip">An independent, hands-on evaluation</div>
<h1>Jev: Not Frontier, But Still Worth Your Attention</h1>
<p class="sub">TypeSafe AI sells Jev as a frontier-class reasoner that cannot
hallucinate, built by the co-inventor of ChatGPT - fast, and almost free. We ran it
live on {TOTAL_REQUESTS:,} benchmark requests, measured its latency and billing,
and probed what it is underneath. The result is a smaller, humbler model that
is nonetheless genuinely useful for a job that nobody else serves quite this way.</p>
<div class="herorot" id="herorot" aria-label="Jev benchmark results, rotating">
{''.join(slides)}
<div class="rot-ctl">
<button class="rot-btn" id="rot-prev" aria-label="previous benchmark">&lsaquo;</button>
<button class="rot-btn" id="rot-pause" aria-label="pause rotation">&#10074;&#10074;</button>
<button class="rot-btn" id="rot-next" aria-label="next benchmark">&rsaquo;</button>
<span class="rot-count" id="rot-count">1 / {len(slides)}</span>
</div>
</div>
<script>
(function(){{
  var box=document.getElementById('herorot');
  var s=box.querySelectorAll('.rot-slide'), i=0, paused=false;
  function show(n){{ i=((n%s.length)+s.length)%s.length;
    for(var k=0;k<s.length;k++) s[k].classList.toggle('active',k===i);
    document.getElementById('rot-count').textContent=(i+1)+' / '+s.length; }}
  setInterval(function(){{ if(!paused) show(i+1); }},6500);
  document.getElementById('rot-prev').addEventListener('click',function(){{show(i-1)}});
  document.getElementById('rot-next').addEventListener('click',function(){{show(i+1)}});
  var pb=document.getElementById('rot-pause');
  pb.addEventListener('click',function(){{ paused=!paused;
    pb.innerHTML=paused?'&#9654;':'&#10074;&#10074;';
    pb.setAttribute('aria-label',paused?'resume rotation':'pause rotation'); }});
  box.addEventListener('mouseenter',function(){{box.dataset.hover='1'}});
  box.addEventListener('mouseleave',function(){{box.dataset.hover='0'}});
}})();
</script>
{toc}
<div class="card tldr"><b>TL;DR.</b> Jev is not a frontier
model; for example, it misses about a quarter of graduate science questions,
and scores a mere {hle_w} on HLE. It nonetheless is also not a toy:
{pct(SC['mmlu']['accuracy'])} MMLU-Pro, {pct(SC['gpqa']['accuracy'])} GPQA,
~{ARCH['prefill']['fixed_floor_ms']:.0f} ms of server compute per question, and
a full graduate-scale benchmark run for cents. The honest category is
<i>cheap real-time sub-frontier judgement</i> - routing, rubric grading,
control-plane decisions - where nothing else on the market combines this
latency, this price, and probability vectors that never drift schema.</div>"""


# ------------------------------------------------------- sections A: method/arch
def methodology() -> str:
    n = SC["mmlu"]["n_expected"]
    mmlu_pct = pct(SC["mmlu"]["accuracy"])
    arc_n = SC["arc"]["n_expected"]; gpqa_n = SC["gpqa"]["n_expected"]
    hle_n = SC["hle"]["n_expected"]
    mc_n = SC["math_c"]["n_expected"]; ms_n = SC["math_s"]["n_expected"]
    agi_grids = ARCAGI["grids_expected"]; agi_tasks = ARCAGI_T["n_tasks_expected"]
    note_text = fn("behavioronly")
    return f"""
<section id="method">
<h2>What we did, and what we could not do</h2>
<p>We ran Jev live against public benchmarks through its API: {n:,} MMLU-Pro
questions ({mmlu_pct} accuracy), {arc_n:,} ARC-Challenge, {gpqa_n} GPQA
Diamond, {hle_n} Humanity's Last Exam multiple-choice items, the encodable
MATH-500 subsets ({mc_n} as option MCQ, {ms_n} under a per-digit rubric), and
the ARC-AGI-2 public set ({agi_grids} grids, re-encoded per-cell and as
{agi_tasks} whole-task requests) - {TOTAL_REQUESTS:,} planned requests in
three frozen suites. This repository publishes the derived aggregates and the
complete Talk-to-Jev traces. The benchmark item text itself is third-party
licensed content and is not republished here. The analysis results are
generated from these artifacts.{note_text}</p>
<p>Three constraints shape everything below. First, Jev answers directly: one
shot, no chain of thought, no tools, no retries - which is not how the frontier
scores we compare against are produced with modern models, although it is how
older models in the benchmarks functioned, and is automatically factored into
the Pareto frontier comparisons. Second, the API is closed: we never saw
weights, gradients, or anything internal to its design, so the architecture discussion is
inference from observable behavior.{fn('behavioronly')} Third, we chose benchmarks the model could
answer natively - selecting between visible options or grading fixed rubrics -
because converting them into something else would measure our conversion, not
the model.</p>
</section>"""


# Claim-area -> on-disk artifact map for the architecture section. The full
# ledger (alternatives, probabilities, falsifiers, next probes) lives in
# ARCHITECTURE-ANALYSIS.md at the repo root; this table is the short form.
ARCH_EVIDENCE = [
    ("Serving shape and latency", "runs_archprobe/analysis.json + rows.jsonl (987 calls, "
     "0 errors); charts: docs/modern-comparison/architecture-evidence.html"),
    ("Read-out post-processing", "data_report/arch_audits.json (re-derived offline by "
     "scripts/report/arch_audits.py); src/jev_observatory/validation.py"),
    ("Tokenizer profile", "runs_archprobe/tokenizer_fingerprint.json, tokenizer_perscript.json, "
     "tokenizer_broadscan.json, tokens_per_char_compare.json, cleanrun_jev.json; "
     "docs/modern-comparison/ARCHITECTURE-PROBES.md"),
    ("Capability and calibration", "runs_benchmark*/bench-*/derived/score.json + "
     "weighted_score.json, freeze manifests; "
     "docs/modern-comparison/canonical/comparable-scores.json"),
    ("Identity, horizon, abstention", "runs_archprobe/analysis.json (ancestry); "
     "runs_live/FINDINGS.md; runs_live/identity_parity_*.json; "
     "docs/token-talk-findings.md"),
    ("Generation ladder (Jev-only rungs)", "docs/jev-talk-program-report.md; "
     "docs/token-talk-findings.md; runs_live/ traces"),
    ("Lattice forensics + staged probe battery", "scripts/report/lattice_forensics.py -> "
     "data_report/lattice_forensics.json; scripts/benchmark/run_probe_battery2.py "
     "(P1-P5, dry-run validated; live dispatch gated on the API key)"),
    ("Economics", "data_report/costs.json, data_report/billing_totals.json"),
    ("Full ledger and next probes", "ARCHITECTURE-ANALYSIS.md"),
]


def architecture(v: dict) -> str:
    p = ARCH["prefill"]; h = ARCH["headcount"]; o = ARCH["optioncount"]
    floor = p["fixed_floor_ms"]; slope = p["ms_per_1k_input_tokens"]
    mq = h["marginal_ms_per_question"]; mo = o["marginal_ms_per_option"]
    fq = h["marginal_output_tokens_per_question"]
    fo = o["marginal_output_tokens_per_option"]
    c1 = ARCH["concurrency"]["per_call_wall"]["1"]
    c32 = ARCH["concurrency"]["per_call_wall"]["32"]
    anc = ARCH["ancestry"]; mass = anc["family_mass"]; votes = anc["greedy_votes"]
    openai_votes = sum(votes.get(k, 0) for k in ("OpenAI", "GPT", "ChatGPT"))
    qa = AUDITS["quantization"]["archprobe_rows"]
    ql = AUDITS["quantization"]["runs_live_raw"]
    n_values = qa["n_values"] + ql["n_values"]
    n_mismatch = qa["n_choice_not_table_argmax"] + ql["n_choice_not_table_argmax"]
    cfc = AUDITS["confidence_formula"]["choice"]
    mcq = AUDITS["marginal_cost"]["per_question"]
    mco = AUDITS["marginal_cost"]["per_option"]
    bp = AUDITS["battery_power"]
    rp = AUDITS["rotation_pairs"]
    lat_vals = sum(c["n_values"] for c in LATTICE["corpora"].values())
    lat_vecs = sum(c["n_vectors"] for c in LATTICE["corpora"].values())
    lat_big = LATTICE["corpora"]["talk_ensemble_raw"]["by_k_bucket"]["65-255"]
    lat_frac99 = (lat_big["sum_dev_sign_counts"]["le_-0.005"]
                  / max(lat_big["n_vectors"], 1))
    fmt_total = sum((SC[s].get("strict_format_failures") or 0) for s in SC)
    wmmlu = WD["mmlu"]
    jp = COSTS["prices_usd_per_M"]["Jev"]
    probe_calls = sum(f["calls"] for f in PROBEPLAN["plan"]["per_family"].values())
    probe_cost = PROBEPLAN["plan"]["est_cost_usd_at_0p042_per_M_in"]
    probe_tok = PROBEPLAN["plan"]["est_input_tokens_total"]
    sza = SIZE["prefill_angle"]; szc = SIZE["capability_angle"]
    sz_lo, sz_hi = sza["active_params_B_range"]
    szc_lo, szc_hi = szc["band_dense_equivalent_b"]
    sz_R = sza["marginal_prefill_rate_tok_s"]
    ev = "".join(f"<tr><td>{a}</td><td class='cap'>{b}</td></tr>"
                 for a, b in ARCH_EVIDENCE)
    p1 = P2RUN["p1_fallback"]["classes"]
    p2g = P2RUN["p2_grid"]
    p3 = P2RUN["p3_kcal"]["by_K"]
    p4 = P2RUN["p4_lattice"]
    p5 = P2RUN["p5_horizon"]["by_month"]
    ident = p4["identical_options"]
    n4 = sum(v["n"] for v in ident.values())
    sums_1 = sum(1 for v in ident.values() for _ in range(v["n"])
                 if v["sum_min"] == 1.0 and v["sum_max"] == 1.0)
    nt = p4["near_tie_choice_counts"]
    nt_n = sum(nt.values())
    nt_top = max(nt, key=lambda k: nt[k]) if nt else "-"
    nt_share = nt.get(nt_top, 0) / nt_n if nt_n else 0
    known = [m for m, b in p5.items() if b["gold_rate"] and b["gold_rate"] >= 0.99
             and "post" in b["kinds"]]
    partial = [m for m, b in p5.items()
               if b["kinds"].get("post") and 0.05 < b["gold_rate"] < 0.99]
    zero = [m for m, b in p5.items()
            if b["kinds"].get("post") and b["gold_rate"] <= 0.05 and m >= "2025-06"]
    fic = [b for m, b in p5.items() if b["kinds"].get("fictional")]
    fic_ok = sum(b["did_not_occur"] for b in fic)
    fic_n = sum(b["kinds"]["fictional"] for b in fic)
    k2, k255 = p3["2"], p3["255"]
    byte_classes = [c for c in ("cjk_ext_a", "cherokee", "yi_syllables", "cjk_ext_b")
                    if c in p1 and p1[c].get("tokens_per_utf8_byte")]
    byte_lo = min(p1[c]["tokens_per_utf8_byte"] for c in byte_classes)
    byte_hi = max(p1[c]["tokens_per_utf8_byte"] for c in byte_classes)
    _mn = {"01": "January", "02": "February", "03": "March", "04": "April",
           "05": "May", "06": "June", "07": "July", "08": "August",
           "09": "September", "10": "October", "11": "November", "12": "December"}
    def _mname(ym: str) -> str:
        y, m = ym.split("-")
        return f"{_mn[m]} {y}"
    hz_solid = _mname(partial[-1]) if partial else "late 2024"
    hz_gone = _mname(zero[0]) if zero else "mid-2025"
    p5_n = sum(b["n"] for b in p5.values())
    hc_n = ARCH["headcount"]["n_configs"]
    sz_pk_lo, sz_pk_hi = sza["assumptions"]["peak_effective_TFLOPS_per_device"]
    moe = next(r for r in SIZE["reconciliation"]["readings"]
               if "moe_total_B_range" in r)
    sz_moe_lo, sz_moe_hi = moe["moe_total_B_range"]
    sz_act_lo, sz_act_hi = moe["activation_pct_range"]
    sz_dense_hi = min(round(sz_hi), 9)
    return f"""
<section id="arch">
<h2>What Jev appears to be</h2>
<p>What follows is our <i>operating premise</i>: every other section reads
Jev's behavior through the model stated here. The full ledger - alternative
hypotheses with probabilities and a falsifier per load-bearing claim - lives
in <a href="{GH}/ARCHITECTURE-ANALYSIS.md"><code>ARCHITECTURE-ANALYSIS.md</code></a>.</p>
<div class="card">In one sentence: Jev looks like a <b>small,
English-centric transformer language model</b>, post-trained for judgement
rather than conversation, served with its <b>generation head replaced by a
probability read-out</b> over caller-supplied options - every question in a
request scored from <b>one batched prefill pass</b>, which is why it is fast,
why its output is free, and why it cannot write you a poem.</div>
<figure>{strip_titles(ARCHDIAG[0])}</figure>
<table class="plain">
<tr><th>Component</th><th>Best guess</th><th>Confidence</th></tr>
<tr><td>Output side</td><td class='cap'>A trained probability read-out over the caller's
options - a discriminative head where a language head usually goes, with
choice / score / noul as its three exposed shapes</td><td>Confident</td></tr>
<tr><td>Serving</td><td class='cap'>One forward pass per request; all questions and options scored from it; continuous batching across tenants (server compute stays flat whether we send 1 or 32 requests at once, and is unchanged by option count); no decode loop</td><td>Confident</td></tr>
<tr><td>Compute profile</td><td class='cap'>Fixed ~{floor:.0f} ms floor + ~{slope:.0f} ms per 1k input tokens, linear to 29k tokens, with no large quadratic signature. Marginal cost of deciding: +{mq:.2f} ms per extra question and +{mo:.2f} ms per extra option - both about what the added tokens cost.</td><td>Confident (measured)</td></tr>
<tr><td>Input pipeline</td><td class='cap'>A whitespace normalizer (ASCII runs collapse; NBSP, ZWJ and BOM do not) plus a fixed ~{bp['mergerate_baseline_tokens']}-token template, ahead of the model's own tokenizer</td><td>Confident / plausible</td></tr>
<tr><td>Tokenizer</td><td class='cap'>The vendor's own English/Latin-centric BPE: heavy Latin merges, ~1 token per codepoint for the major non-Latin scripts, and a byte-level fallback for uncovered characters ({byte_lo:.2f}-{byte_hi:.2f} tokens per UTF-8 byte on the rarest blocks). It matches no known preexisting tokenizer signature (173 tested) and appears to be new.</td><td>Plausible (strong)</td></tr>
<tr><td>Core</td><td class='cap'>Transformer-family decoder LM; dense vs MoE unknown -
with quantized-serving assumptions the two size angles overlap, so neither
is forced and MoE stays possible; attention variant unknowable at these
context lengths</td><td>Plausible / open</td></tr>
<tr><td>Size</td><td class='cap'>No point estimate. Latency suggests ~{sz_lo:.1f} to {sz_hi:.0f}B active params from prefill throughput (with quantized-serving assumptions); capability suggests {szc_lo:.0f} to {szc_hi:.0f}B dense-equivalent, which does the narrowing. A quantized dense ~{szc_lo:.0f} to {sz_dense_hi:.0f}B is the parsimonious joint reading; a MoE (~{sz_moe_lo:.0f} to {sz_moe_hi:.0f}B total) stays possible, not favored</td><td>Plausible</td></tr>
<tr><td>Training</td><td class='cap'>English-dominant pretraining; knowledge horizon solid to late 2024, partial to ~{hz_solid}, gone by {hz_gone}; judgement-format assistant post-training; OpenAI-flavored brand prior inherited from training text; frontier-teacher contribution: none identifiable, not excluded</td><td>Plausible</td></tr>
<tr><td>What it is not</td><td class='cap'>Not frontier, not retrieval- or cache-assisted, not a wrapper around another vendor's API, not a relabeled open model</td><td>Confident</td></tr>
</table>
<p class="cap">See <a href="{GH}/ARCHITECTURE-ANALYSIS.md">ARCHITECTURE-ANALYSIS.md</a>
&sect;1 for details.</p>

<h3 id="arch-onepass">One pass, many read-outs</h3>
<p>The latency section below carries the measurements: server compute is a
fixed floor plus a linear prefill term (~{slope:.0f} ms per 1k input tokens), and the
decisive numbers are the marginals. Packing one more question into a request
costs +{mq:.2f} ms. That question brings ~{mcq['input_tokens_per_unit']:.0f} input tokens with it, and
reading those tokens at the prefill slope is itself worth
+{mcq['prefill_predicted_ms_per_unit']:.2f} ms - which leaves under {mcq['residual_ms_per_unit']:.2f} ms for the act of
deciding. An extra option behaves the same way: +{mo:.2f} ms measured, against
the +{mco['prefill_predicted_ms_per_unit']:.2f} ms that its own ~{mco['input_tokens_per_unit']:.0f} tokens already predict. In
both cases the read-out's own compute is indistinguishable from zero.</p>
<p>The battery re-confirmed this live: across a grid of option counts (2-255)
crossed with option lengths (2-64 tokens), nothing scaled with the number
of options beyond the tokens they add
({p2g['ols_ms_per_100_options_beyond_tokens']:+.1f} ms per 100 options, inside the &plusmn;{p2g['residual_sd_ms']:.0f} ms
residual noise). Billed output grows ~{fq:.0f} tokens per question
and ~{fo:.1f} per option - that is the response JSON serializing itself, which
is exactly why output can be free: nothing is ever decoded. Upstream compute
stays flat ({c1['median_upstream_ms']:.0f} ms at concurrency 1, {c32['median_upstream_ms']:.0f} ms at 32) while our own wall
time bends, so independent requests share a batched path. A four-question
batch answers in ~261 ms where the same four questions separately take
~1,117 ms: there is no hidden per-question round trip anywhere.
&ldquo;System one&rdquo;, under this reading, is as much a serving description
as a psychological one - prefill, read out, done.</p>
<figure>{strip_titles(EVID[1])}<figcaption>Server compute vs the number of
questions packed into one request ({hc_n} probe calls). The dashed line is
what each question's own added input tokens predict at the prefill slope; the
points sit on it, which is the whole finding: deciding is free, reading is
not.</figcaption></figure>
<p class="cap">In short, what moves server compute:</p>
<ul class="cap">
<li><b>Prompt tokens</b> - the only thing that scales it, and linearly.</li>
<li><b>Extra questions</b> in a request - only through the tokens they add.</li>
<li><b>Extra options</b> - only through the tokens they add; the deciding itself is free.</li>
<li><b>Concurrent requests</b> - flat on the server (shared batching); only our own wall time queues.</li>
<li><b>Repeats</b> - every call re-reads the prompt; nothing is cached away.</li>
</ul>

<h3 id="arch-fingerprints">Fingerprints on the read-out</h3>
<p>Every probability Jev returns is rounded to two decimals: across
{lat_vals:,} values in {lat_vecs:,} published vectors, from 2-option questions up to
255-option menus, not one value ever landed off that 0.01
grid.{fn('quantization')} A probability of exactly 0.00 is common, so there is
no floor - genuinely unlikely options are shown as zero, not as 0.01. What
that costs you is resolution: two options at 0.17 and 0.174 are
indistinguishable, so any conclusion resting on a difference below 0.01 is
reading noise.</p>
<p>The rounding has more structure than &ldquo;rounded&rdquo;. Sums of all
probabilities returned are either 0.99 or 1.00, and never exceed 1.00 - in
any corpus, at any option count up to 255. Independent per-value rounding
would scatter sums by &plusmn;0.04 at 255 options. So a bounded correction
runs <b>after</b> rounding, and it only ever takes mass away:
{pct(lat_frac99)} of flat 255-option vectors land exactly 0.01 short, while
every sharp vector and every two-option vector sums to exactly
1.000.{fn('lattice')}</p>
<p>Two more seams show. In {n_mismatch} vectors the returned choice is not the
argmax of its own displayed table. Each such gap is exactly one quantum (e.g., 0.01),
which monotone rounding cannot produce - so the decision is computed at
pre-display precision while the table is rounded separately. And the choice
<code>confidence</code> field is roughly recoverable. It is, to within a
quantum or two, the chance-corrected top probability:
<code>(p_max &minus; 1/K) / (1 &minus; 1/K)</code>, where <code>p_max</code>
is the highest probability in the vector and <code>1/K</code> is what
guessing would earn across K options. Of {cfc['n']:,} published choice
vectors, {cfc['exact']:,} match that formula exactly against the displayed
table, {cfc['within_1_quantum']} are within one quantum, and {cfc['within_2_quanta']} within two. None is further
off.{fn('readout')} Buyer's note: choice confidence is a
deterministic function of the vector you were already handed, not a second
opinion. Score answers use a different shape statistic and noul answers carry
no confidence at all.</p>
<p>The follow-up battery ({P2BILL['n_calls']:,} live calls) put the read-out
on a degenerate case: identical option texts, repeated. Across {n4}
such vectors at option counts 2 to 255, sums were exactly 1.000 at every size
up to 12 and {ident['255']['sum_min']:.2f}-{ident['255']['sum_max']:.2f} at 255 - the bounded-correction reading, confirmed
live. Identical texts did <i>not</i> get identical probabilities: they were
scored by position, not content, which is the purest form of the ordering
effect described below.</p>
<p>To test whether dummy filler options dilute the answer, we re-ran 200
gold-labeled synthetic questions at every
option count from 2 to 255, padding with inert filler. The probability on
the right answer did not move - {pct(p3['255']['mean_p_gold'])} at 255
options against {pct(p3['2']['mean_p_gold'])} at two - and accuracy stayed perfect at every
size, with the filler options pinned at 0.00. Apart from the ordering effects
below, options are scored on their content, essentially absolutely; the vector is then normalized over the set.
One consequence for the benchmark tables below: the weighted scores are not
an artifact of menu size.</p>

<h3 id="arch-order">The order of the options matters</h3>
<p>Jev reads the option list as a list, in context, and position is a
first-order factor wherever content is weak. On a flat creative step with 254
candidate continuations, eleven different orderings of the identical option
set produced ten different winners, each internally stable across six repeats;
rank agreement with the native order fell to Spearman 0.26-0.44, against
0.42-0.88 on a sharp factual step. The serial-position curve is U-shaped: the
first decile of positions carries ~4&times; the mean probability of the middle
deciles, and the last decile is elevated too. Mean p_max for the same options
ranged 0.26-0.54 depending purely on their order. These effects are 3-10&times;
the measured repeat-noise band (TVD 0.03-0.12), so they are
real.{fn('ordering')}</p>
<p>Two consequences for anyone using this API. First, rotate: our own
measurement protocol ensembles K&ge;6 cyclic rotations and averages, which
cancels the bias (Spearman 0.91-1.00 against the 12-rotation reference) - at
the cost of flattening the aggregate, so per-answer temperature must drop to
compensate. Second, do not read a menu's ordering as neutral. Where content
is strong the bias washes out: re-running 419 MMLU-Pro items with shuffled
option order flips {pct(rp['flip_rate'])} of answers with no net accuracy change (exact
McNemar p = {rp['mcnemar_exact_p']:.2f}), and the ancestry probe below cancels position by
construction. Where content is weak - a routing menu of near-synonyms, a
flat rubric - order will move your answer.</p>

<h3 id="arch-tokenizer">A tokenizer nobody recognizes</h3>
<p>Probing tells the full story, including after discounting false leads.
Jev's counter merges English text and punctuation hard, spends ~1 token per
codepoint on Cyrillic, Greek, Arabic, Hebrew, Thai, Devanagari, Hangul, Kana
and common CJK, and ~1 per digit. Uncovered characters fall back below
codepoint granularity: the rare blocks cost {byte_lo:.2f}-{byte_hi:.2f} tokens per UTF-8 byte
(3-4 tokens per character), which is byte-level fallback. A lone surrogate
escape is rejected outright with <code>invalid Unicode text</code>, so the
pipeline is UTF-8, not UTF-16. Decomposed and precomposed accented text cost
the same token count, so Unicode normalization runs before tokenization.
Whitespace behaves by character class: ASCII space runs collapse to nearly
nothing, tabs and newlines to ~10-15% of their length, while NBSP, ZWJ, ZWNJ,
soft hyphen and BOM each cost a full token - the normalizer's definition of
whitespace is ASCII-only.</p>
<p>Early attempts to let Jev <a href="#probes-talk">talk</a> so it could
describe itself led to Jev identifying itself with an OpenAI-family name
{openai_votes} times out of {anc['n_frames_with_probs']}, but this appears to be contamination from training on a
broad, English-dominated vocabulary - not a signature of authorship.</p>

<h3 id="arch-capacity">Limited capacity, in a particular way</h3>
<p>The capability profile has a particular shape. On one-shot knowledge
multiple-choice Jev lands in the band of 2025-era small instruct models:
{pct(SC['mmlu']['accuracy'])} on MMLU-Pro and {pct(SC['gpqa']['accuracy'])} on GPQA Diamond, against 82.5 and
77.6 for Qwen 3.5 9B and 80.7 and 76.8 for Claude 3.7 Sonnet without
thinking.{fn('protocol')} On anything multi-step it falls off a cliff:
{pct(SC['hle']['accuracy'])} on HLE, zero exact grids on ARC-AGI-2, and the generation tax -
{pct(SC['math_c']['accuracy'])} on MATH-500 items when the answer is an option to pick,
{pct(SC['math_s']['accuracy'])} when the same answers must be read off digit by digit.
Recognition far exceeds production, which is what judgement-heavy
post-training on a small model produces. The distributions are shaped the
same way: the median MMLU-Pro item carries p(gold) = {wmmlu['weighted_accuracy_p50']:.2f} against a
mean of {pct(wmmlu['weighted_accuracy_mean'])}, and {SC['mmlu']['zero_probability_gold']} items put exactly 0.00 on the gold answer -
near-decisive where it knows, confidently wrong on a hard tail.</p>
<p>The benchmark section below shows two scorings of every stage: greedy
(take the top option) and probability-weighted (average the probability Jev
put on the right answer). Greedy beats weighted almost everywhere, by
{pct(SC['mmlu']['accuracy'] - WD['mmlu']['weighted_accuracy_mean'])} points on MMLU-Pro and {pct(SC['gpqa']['accuracy'] - WD['gpqa']['weighted_accuracy_mean'])} on GPQA. That gap is the
over-dispersion: Jev's argmax is right more often than its own distribution
predicts, so its probabilities understate its accuracy. Treat the argmax as
the answer and the vector as a ranking signal, not as a calibrated
confidence.</p>
<p><b>How big is it?</b> A slope alone is not a size - the server batches our
tokens with everyone else's - but two angles converge on a band, for
reasonable assumptions. From the throughput side: the marginal prefill rate is
~{sz_R:,} tokens/s. Batched prefill is compute-bound and costs about
2&middot;N<sub>active</sub> FLOPs per token, so
N<sub>active</sub> &le; MFU &times; effective peak &times; shards &divide; 2R.
Assuming typical late-2026 hardware (H100/H200/B200/TPU-v6/MI325X-class,
~400-2250 BF16 TFLOPS per device) and FP4-FP8 quantization - which doubles
(fp8) to quadruples (fp4) effective throughput - the honest peak range is
~{sz_pk_lo:.0f}-{sz_pk_hi:.0f} effective TFLOPS per device, not the 250-500 of a 2020-era bf16 A100.
With 25-45% utilization over 1-4 devices that bounds the <i>active</i>
footprint at ~{sz_lo:.1f} to {sz_hi:.1f}B parameters (central case ~{sza['central_case_B']:.1f}B), and sharing the machine
with other tenants only lowers the bound. From the capability side: the
similar-performing models put it at {szc_lo:.0f} to {szc_hi:.0f}B dense-equivalent. The two are
consistent - but honesty requires noting that the throughput bound, computed
this way, is loose: it is the capability angle doing the real work. The
parsimonious reading: a quantized dense model of roughly
{szc_lo:.0f} to {sz_dense_hi:.0f}B active parameters fits both angles without strain - an entirely
ordinary object at this size in late 2026. A MoE of ~{sz_moe_lo:.0f} to {sz_moe_hi:.0f}B total at
{sz_act_lo:.0f}-{sz_act_hi:.0f}% activation fits just as well and would explain the top of the
capability band with less compute per token, but nothing observable prefers
it over the dense reading. Total parameters stay unconstrained either way:
nothing this API returns can see expert structure.{fn('sizeest')}</p>

<h3 id="arch-not">What it is not</h3>
<p>Four negations, each a measurement rather than a vibe.{fn('notwrapper')}
Not retrieval or cache: compute grows linearly in prompt tokens, and a cache
would not prefill. Repeats of identical payloads differ, and {SC['mmlu']['zero_probability_gold']}
zero-probability gold answers on <i>public</i> benchmark text is not what a
lookup produces. The knowledge horizon also behaves like weights, not like a
query: dated-event bisection ({p5_n} live calls) shows solid answers through
late 2024, partial through {hz_solid}, and none by {hz_gone} - a fixed cutoff, with
{fic_ok}/{fic_n} invented events correctly called &ldquo;did not occur&rdquo; and
abstention rising exactly where accuracy falls. Not a wrapper around a
frontier API: {c1['median_upstream_ms']:.0f}-{c32['median_upstream_ms']:.0f} ms of upstream compute leaves no room inside for
anyone else's round trip, and the price sits one to two orders of magnitude
below flagship input lists. Not a relabeled and slightly-restructured open
model: the unusual tokenizer seems quite clear on this point. And not
frontier: the benchmark section says so six ways.</p>

<h3 id="arch-open">What stays open</h3>
<p>Three things this API cannot tell us, and we do not guess them. Whether the
transformer is dense or mixture-of-experts - the capability/latency ratio
favors MoE, but nothing observable separates them. Whether a frontier teacher
produced any of the training signal: every discriminator we can construct is
confounded, including the OpenAI-shaped brand prior, which the open
assistant-text ecosystem produces on its own. And any exact parameter count
or hidden dimension. The two-angle estimate above bands the active size at
~{sz_lo:.1f} to {sz_hi:.1f}B, but identification is beyond this API, which documents nothing past
32k-token state, 64k total, &le;255 options and 2-10 rubric levels.</p>
</section>"""


def pitch() -> str:
    arc_pct = pct(SC["arc"]["accuracy"])
    return f"""
<section id="pitch">
<h2>The pitch, and the parts that survive contact</h2>
<p class="pitch">&ldquo;A model that <b>cannot hallucinate</b>, at <b>frontier-level
performance</b>, built by <b>the co-inventor of ChatGPT</b> - incredibly fast,
incredibly cheap, with <b>free output</b>.&rdquo;</p>
<p>Each clause is technically defensible in a narrow sense and misleading in the
sense a buyer will hear. Take them one at a time.</p>
<p><b>Cannot hallucinate.</b> What Jev actually cannot do is emit free text. It
returns one option from a fixed menu, inside a schema that is always
well-formed. But a schema-valid answer is not a true answer. Jev is frequently
wrong (see <a href="#benchmarks">our benchmarks</a>), and every one of those
wrong answers arrives beautifully formatted. A model that cannot write prose
has not solved hallucination; it has made hallucination hard to notice.</p>
<p><b>Fast and cheap.</b> Both true, and explainable in one sentence: Jev never
runs a decode loop. It reads the prompt once and reads off a
vector.{fn('jevfree')} Speed and price are properties of the <i>task</i> being
prefill-only, not of frontier economics.</p>
<p><b>Frontier-level.</b> This one simply does not survive. Jev is good, and
&ldquo;good&rdquo; will turn out to mean something genuinely useful here - but
it is not a frontier model by any late-2026 standard, and where it looks
frontier-like ({arc_pct} on ARC-Challenge), that's only on a race that
finished years ago. Everything below is the evidence.</p>
<p><b>The co-inventor of ChatGPT.</b> Diogo Almeida certainly deserves credit,
but saying &ldquo;I co-invented ChatGPT&rdquo; overassigns it. Almeida was one
of the eight primary authors on InstructGPT, worked on learned
optimizers, and was also one of 88 people thanked in the initial release of
ChatGPT. But ChatGPT was born from the work of thousands of people; see the
footnote.{fn('almeida')}</p>
</section>"""


def benchmarks() -> str:
    mmlu, arc, rot = SC["mmlu"], SC["arc"], SC["rot"]
    gpqa, hle = SC["gpqa"], SC["hle"]
    mc, ms = SC["math_c"], SC["math_s"]
    ag, at = ARCAGI, ARCAGI_T
    wmmlu = WD["mmlu"]["weighted_accuracy_mean"]
    warc = WD["arc"]["weighted_accuracy_mean"]
    wgpqa = WD["gpqa"]["weighted_accuracy_mean"]
    whle = WD["hle"]["weighted_accuracy_mean"]
    wmc = WD["math_c"]["weighted_accuracy_mean"]
    wms = WD["math_s"]["weighted_accuracy_mean"]
    rows = [
        ("MMLU-Pro", mmlu["n_expected"], pct(mmlu["accuracy"]), wil(mmlu),
         pct(wmmlu), "12k graduate-level MC questions, 10-19 options"),
        ("ARC-Challenge", arc["n_expected"], pct(arc["accuracy"]), wil(arc),
         pct(warc), "grade-school science, 4 options"),
        ("GPQA Diamond", gpqa["n_expected"], pct(gpqa["accuracy"]), wil(gpqa),
         pct(wgpqa), "graduate science, 4 options, seeded shuffle"),
        ("HLE, multiple-choice", hle["n_expected"], pct(hle["accuracy"]), wil(hle),
         pct(whle), "expert exam, MC subset only"),
        ("MATH-500 (as MCQ)", mc["n_expected"], pct(mc["accuracy"]), wil(mc),
         pct(wmc), "numeric answers, 4 options"),
        ("MATH-500 (digit read-out)", ms["n_expected"], pct(ms["accuracy"]), wil(ms),
         pct(wms),
         "every digit right, or it is wrong"),
        ("ARC-AGI-2, per cell", ag["cells_total"], pct(ag["cell_accuracy_diagnostic"]), "",
         pct(WD["agi_choice"]["per_cell_weighted_mean"]),
         "public eval, color-per-cell diagnostic"),
        ("ARC-AGI-2, exact grid", ag["n_tasks"], pct(ag["task_accuracy"]), "", "-",
         "all cells of all 167 grids"),
    ]
    body = "".join(
        f"<tr><td>{n}</td><td class='n'>{c:,}</td><td class='n'><b>{a}</b></td>"
        f"<td class='n'>{w}</td><td class='n'>{ww}</td><td class='cap'>{t}</td></tr>"
        for n, c, a, w, ww, t in rows)
    fig_list = [
        fig("mmlu_pro", f"MMLU-Pro. Jev's best-aligned benchmark: same items, same "
            f"format, direct answers. External rows come from the fetched vals.ai "
            f"extract (133 models); dagger bars are our matched runs on the "
            f"identical 1,000-item subset.{fn('protocol')}"),
        fig("gpqa", "GPQA. Jev ran the Diamond subset (196 items); external rows are "
            "the vals.ai platform's GPQA set, retired as saturated in Sep 2026."),
        fig("arc", "ARC-Challenge. Saturated for everyone. Jev's bar is 4-choice; "
            "external bars are hand-collected canonical rows (mostly 25-shot CoT, "
            "mostly older models) plus our matched runs of six modern models "
            "(dagger)."),
        fig("arc_agi2", "ARC-AGI-2. Mixed encodings, flagged: external bars are "
            "whole-grid pass@2 with reasoning; Jev cannot emit grids. Its exact-grid "
            "bars are the protocol-matched pair (0 of 120 tasks); the per-cell bars "
            "are a diagnostic encoding and do not compete with the grid rows."),
    ]
    if "math_m" in CHARTS:
        fig_list.append(fig("math_m", "MATH-500 (as MCQ). Two benchmarks that we ran "
                            "posed challenges for comparing to public results; this "
                            "one compares only our matched runs over Jev's exact "
                            "items and format - the free-form public rows are not "
                            "shown."))
    if "hle_m" in CHARTS:
        fig_list.append(fig("hle_m", "HLE (MC subset), matched runs. Same logic: our "
                            "runs of cheap models on Jev's exact 494-item subset; a "
                            "matched row beats Jev, so the chart is fair. The full "
                            "text-only set rows are not shown."))
    figs = "".join(fig_list)
    return f"""
<section id="benchmarks">
<h2>Benchmarks: where it actually lands</h2>
<p>Jev ran <b>directly</b>: one shot, no chain of thought, no tools, no retries,
failures counted against it. Most published comparison numbers use reasoning
enabled and few-shot prompts (though older models lack reasoning), so the
columns below may be considered more as positioning than as a matched
race.{fn('protocol')} By contrast, in the Pareto frontier graphs, chain of
thought is a cost that counts against the models that employ it.</p>
<table>
<tr><th>Benchmark</th><th class="n">Items</th><th class="n">Jev</th>
<th class="n">95% CI</th><th class="n">Weighted</th><th>Notes</th></tr>
{body}
</table>
<p class="cap">The <b>Weighted</b> column is the average probability Jev itself
put on the right answer; it sits below accuracy almost everywhere, which says
Jev's confidence spreads onto wrong options more than its accuracy war-
rants.{fn('greedyw')} MATH-500 and ARC-AGI-2 rows are conversions, not native
runs.{fn('mathadapt')} The rotation audit re-ran items with option order
shuffled.{fn('rotations')}</p>
<h3 id="bench-charts">The charts</h3>
<p>Teal is Jev's greedy score (the highest-probability choice is chosen),
while amber is Jev's probability-weighted score, and every other bar is a
published number we fetched - not a model we ran - except the dagger-marked
bars, which are our own matched runs of cheap models over the identical
frozen items.{fn('matchedbase')} Bar color is release era (red &le;2023
&rarr; purple &rarr; blue 2026), and the shape at each bar tip is that row's
reasoning configuration - rounder means less thinking, pointier means more.
Each chart is a curated view (top, bottom, and audit-named models) of the
full fetched extract - 133 models for MMLU-Pro and GPQA - kept on disk in
<a href="{GH}/docs/modern-comparison/canonical/vals-leaderboards-20260926.json"><code>canonical/vals-leaderboards-20260926.json</code></a>.</p>
{figs}

</section>"""


def pareto() -> str:
    vm = VALS["benchmarks"]["mmlu_pro"]["models"]
    vg = VALS["benchmarks"]["gpqa"]["models"]
    fable = vm["anthropic/claude-fable-5-1"]
    fable_run = fable["cost_per_test"] * SC["mmlu"]["n_expected"]
    fable_gpqa = vg["anthropic/claude-fable-5-1"]["cost_per_test"]
    n_mmlu = SC["mmlu"]["n_expected"]
    n_gpqa = SC["gpqa"]["n_expected"]
    jq = JEV_COST["mmlu_pro"] / n_mmlu
    jgq_run = JEV_COST["gpqa"]
    jgq = jgq_run / n_gpqa
    ratio = fable_gpqa / jgq
    figs = "".join(f"<figure>{svg}</figure>" for svg in PARETO)
    pareto_extra = ""
    if MATCHED.get("models"):
        _jcq = JEV_COST["mmlu_pro"] / SC["mmlu"]["n_expected"]
        _rows = []
        for _mid, _dss in MATCHED["models"].items():
            _e = _dss.get("mmlu") or {}
            _a = _e.get("accuracy_recovered")
            _c = _e.get("cost_per_question_usd")
            _jj = _e.get("jev_join") or {}
            if _a is not None and _c and _e.get("n_terminal") == _e.get("n_requested"):
                _rows.append((_mid.split("/")[-1], _a * 100.0, _c,
                              (_jj.get("jev_accuracy_on_subset") or 0.0) * 100.0))
        if _rows:
            _rows.sort(key=lambda r: -r[1])
            _js = _rows[0][3]
            _above = [r for r in _rows if r[1] > _js]
            _cheaper = [r for r in _rows if r[2] < _jcq]
            _cb = max(_cheaper, key=lambda r: r[1]) if _cheaper else None
            bits = [" The dagger points fill in the commodity end, and they "
                    "refine the claim."]
            if _above:
                _names = ", ".join(f"{r[0]} ({r[1]:.1f}%)" for r in _above[:3])
                _mults = [r[2] / _jcq for r in _above]
                bits.append(
                    f" {len(_above)} matched models outscore Jev on these "
                    f"identical items - {_names} - at {min(_mults):.0f} to "
                    f"{max(_mults):.0f} times its per-question cost, because "
                    f"every thinking token gets billed.")
            else:
                bits.append(" No matched model outscores Jev on these "
                            "identical items.")
            if _cb is not None:
                bits.append(
                    f" Below Jev's price the best matched score is "
                    f"{_cb[1]:.1f}% ({_cb[0]}), and several commodity models "
                    f"bill less per question than Jev does.")
            bits.append(
                f" The frontier therefore runs: "
                + (f"{_cb[1]:.0f}% at commodity prices, " if _cb else "")
                + f"a jump to Jev's {_js:.1f}% at ${_jcq:.6f} per question, "
                f"then a long climb through the thinking tier to the "
                f"flagships. Jev owns the low-cost rung; it does not own the "
                f"top.")
            pareto_extra = "<p>" + "".join(bits) + "</p>"


    return f"""
<section id="pareto">
<h2>The frontier the cost numbers actually draw</h2>
<p>TypeSafe AI publishes a capability-versus-cost graphic with Jev radically
redefining the Pareto frontier to the top left, with performance equivalent
to frontier models. That framing only works with whatever their
&ldquo;custom&rdquo; rubric is - not against any standard benchmarks (it
should be clear by now why they discourage users from benchmarking the
model).</p>
<p>Our version uses the same accuracies as the benchmark section and measured
money on the horizontal axis: what one question costs, start to finish. Every
external point is either our own run (mostly of small models) or a vals.ai
platform measurement. Chain-of-thought models are thus inherently punished
for their billed intermediary tokens.{fn('costest')} Jev's points come from
our own billing.</p>
<p>Jev does land on the visible frontier - and then some. At
{pct(SC['mmlu']['accuracy'])} on MMLU-Pro its whole 12,032-question run cost
${JEV_COST['mmlu_pro']:.2f}, about ${jq:.6f} per question: four to five orders
of magnitude left of every vals-measured flagship. The same run at Claude
Fable 5.1's measured per-test cost runs about ${fable_run:,.0f} - and Fable
scores {pct(fable['accuracy']/100)}, not {pct(SC['mmlu']['accuracy'])}. On
GPQA, Jev's whole 196-question run cost ${jgq_run:.4f}; per question that is
roughly {ratio:,.0f} times cheaper than Fable 5.1, and about
{(fable['accuracy'] - SC['gpqa']['accuracy']*100):.0f} points less accurate.
Cheap and mid-tier can be the same sentence - and on these axes,
&ldquo;off the chart&rdquo; is a position, not an excuse.</p>
{pareto_extra}
{figs}
</section>"""


def latency() -> str:
    p = ARCH["prefill"]; h = ARCH["headcount"]; o = ARCH["optioncount"]
    cw = ARCH["coldwarm"]
    floor = p["fixed_floor_ms"]; slope = p["ms_per_1k_input_tokens"]
    mq = h["marginal_ms_per_question"]; mo = o["marginal_ms_per_option"]
    fq = h["marginal_output_tokens_per_question"]; fo = o["marginal_output_tokens_per_option"]
    c1 = ARCH["concurrency"]["per_call_wall"]["1"]
    c32 = ARCH["concurrency"]["per_call_wall"]["32"]
    return f"""
<section id="latency">
<h2>What the milliseconds say</h2>
<p>Every response carries server-side compute timing in a proxy header, which
strips out network and queueing - the cleanest architecture signal a public API
leaks. Sequential probes across prompt sizes from ~0.5k to ~29k tokens
fit:</p>
<div class="card"><b>~{floor:.0f} ms fixed floor + ~{slope:.0f} ms per 1k input
tokens</b>, with a negligible quadratic term. A flat base plus a linear prefill
term is what a transformer's forward pass looks like from outside; the floor is
serving overhead, not the model.</div>
<figure>{strip_titles(EVID[0])}<figcaption>Server compute (the queue-free
proxy header) vs input tokens across {p['n_configs']} probe configurations: a flat
floor plus a straight line, with no attention-blowup curvature at these
lengths.</figcaption></figure>
<figure>{strip_titles(EVID[2])}<figcaption>Where a typical call's milliseconds
go: the floor is ~95% of it. Prefill is the model reading; the decision itself
is a rounding error on the chart.
</figcaption></figure>
<p>The more telling experiment is <i>self-batching</i>. Packing 192 questions
into one request barely moves compute time (+{mq:.2f} ms per extra question),
while the reported output grows about {fq:.0f} tokens per question - each one's
full probability vector, serialized. Scoring 255 options instead of 2 costs
+{mo:.2f} ms of compute and {fo:.1f} reported output tokens per option. Those
marginals are fully explained by the input tokens each question (~55) and each
option (~18) adds to the shared prompt; the decision step itself is invisible
at this precision. The reported output tokens are billed at zero: the response
JSON is counted and priced at nothing, because nothing was generated.</p>
<p>Running up to 32 requests concurrently keeps server compute flat
({c1['median_upstream_ms']:.0f} ms at c=1 vs {c32['median_upstream_ms']:.0f} ms at c=32) while client wall time bends -
that bend is our own connection pool, not the server. The flat header is the
evidence of continuous batching. Cold-connection overhead was
~{cw['connection_overhead_ms']:.0f} ms of TLS setup - a reminder to trust the header, not the wall
clock.</p>
<figure>{strip_titles(EVID[4])}<figcaption>Concurrency 1&rarr;32: client wall
time bends at 32 (our connection pool); server compute does not move
(their continuous batching).</figcaption></figure>
</section>"""


def probes() -> str:
    anc = ARCH["ancestry"]
    mass = anc["family_mass"]
    openai = mass.get("openai", 0); tsf = mass.get("typesafe", 0)
    votes = anc["greedy_votes"]; idx0 = anc["choice_at_index_0_frac"]
    jev_p = anc["mean_p_jev"]; ts_p = anc["mean_p_typesafe"]
    jevbot_n = JEVBOT["n_unique_posts_collected"]
    tbl = PCC.get("table", PCC)          # tolerate either nesting
    cols = tbl["cols"]
    head = "".join(f"<th class='n'>{c}</th>" for c in cols)
    rws = tbl["rows"]
    rows = "".join(
        "<tr><td>" + ("Jev (measured)" if name == "JEV" else name) + "</td>" + "".join(
            f"<td class='n'>{rws[name][i]:.2f}</td>" for i in range(len(cols))) + "</tr>"
        for name in rws)
    return f"""
<section id="probes">
<h2>What is it? Three attempts to ask</h2>
<p>Jev cannot tell us what it is in prose, so we built three indirect ways to
ask, in increasing order of how much we trust the answers.</p>

<h3 id="probes-talk">1. Talking to it</h3>
<p>The first idea was to give Jev a text box by hand: offer it candidate
continuations and let it choose, one step at a time. We ran that two ways. In
the first, Jev drives alone: the menu is built from English vocabulary
statistics (letters, common letter-combos, whole words) with no other model in
the loop. In the second, a small local language model (Mellum2-12B-A2.5B)
proposes the candidate next tokens and Jev re-scores them - Jev holds the
wheel, with a driving instructor beside him holding it too. The instructor's
prior leaks into the result (including through option position), so that mode
is a collaboration, and we report it as one.</p>
<p>Alone at the character level, Jev fails outright: its per-character
distributions are dominated by spaces and <code>a</code>, and greedy decoding
produces <code>&ldquo;Geeee&rdquo;</code> and <code>&ldquo;A&nbsp;&nbsp;&nbsp;&nbsp;&rdquo;</code> no
matter what guards we add. Alone with a vocabulary menu it produces 86-100%
genuine words and zero syntax - <i>&ldquo;They areas s aren'ts area aren't
arenas are s&rdquo;</i>, <i>&ldquo;My american s can't cannots s
she&rdquo;</i> - word salad that stops politely. Two quirks show up in every
mode: whitespace-blindness (it prefers the bare token over the leading-space
variant even mid-sentence, so &ldquo;The capital of France is&hellip;&rdquo;
greedily decodes to <code>&ldquo;ThecapitalisParis.&rdquo;</code>) and repetition
attractors (a <code>was was was</code> loop that only structural bans stop;
temperature and nucleus sampling cool it somewhat but never cure it - on
creative prompts its step distributions are so flat that any honest sampling
is dominated by its own noise). With the instructor aboard, coherent text
appears: <i>&ldquo;The capital of France is Paris.&rdquo;</i>, and at the best
long-form configuration the program produced <i>&ldquo;In the outer rim of
galaxy where nebulae paint the void in hues ofviolet andgold cos cosmic
rabbits hop through the interstellar aether and meet Elvis Presley who was
beenhad&rdquo;</i> - fluent, on-prompt, and corrupted exactly where Jev's own
surface-form quirks show through (<code>&ldquo;ofviolet&rdquo;</code>, <code>&ldquo;andgold&rdquo;</code>).</p>
<p>We are not the first to give Jev a mouth. <a
href="https://famelos.com/jev-chat/watch/">Jev Chat</a> - which appears to
power <a href="https://bsky.app/profile/jevbot.bsky.social">Jev Bot</a> on
Bluesky - grows every reply one word at a time from a Markov model, a simpler
guide than our local LM. Its public feed reads exactly like our unguided
vocabulary menu: <i>&ldquo;Wow interesting actually yeah anyway? Well about
this thing? Yours opinion speaking again?&rdquo;</i>, <i>&ldquo;Well im
steve.&rdquo;</i>, <i>&ldquo;I am jeff smith.&rdquo;</i>, <i>&ldquo;I think
that the sea is ocean. It typically seems open sea, and specifically atlantic
ocean.&rdquo;</i> (we collected {jevbot_n} unique posts and published a nine-post sample in
<a href="{GH}/data_report/jevbot_examples.json"><code>data_report/jevbot_examples.json</code></a>).
An independent implementation with an independent guide reaching the same
result is worth something: general-internet continuation, invented personas,
no knowledge of what it is.</p>
<p>The program's summary: unguided Jev is a word-salad generator with
excellent stopping behavior. It can, however, turn the wheel for you while
you're driving.</p>

<h3 id="probes-name">2. Asking it to choose a name</h3>
<p>Because free text is off the table, we made the identity question a multiple
choice - and to keep it honest, the option list <i>always</i> included
TypeSafe and Jev alongside the usual suspects, and every frame was shown under
all 24 cyclic orderings so the <a href="#arch-order">serial-position bias
we measured</a> could not manufacture the result (the winning name landed at index 0 only
{pct(idx0)} of the time - below uniform, so this is content, not position).
Across {anc['n_frames_with_probs']} calls with fifteen differently-worded
prompts:</p>
<div class="card">Jev picked an <b>OpenAI-family name</b>
{sum(v for k, v in votes.items() if k in ('OpenAI','GPT','ChatGPT'))} times out
of {anc['n_frames_with_probs']} ({pct(openai)} of probability mass as a family) -
over &ldquo;Anthropic&rdquo; {pct(mass.get('anthropic',0))}, Google
{pct(mass.get('google',0))}, and its actual maker Typesafe
{pct(tsf)}. Its own name &ldquo;Jev&rdquo; averaged {jev_p:.3f} probability and
&ldquo;Typesafe&rdquo; sat at the {ts_p:.3f} quantization floor.</div>
<p>The sane reading is not that Jev is secretly a GPT. A closed model's
self-report is learned text, and assistant training data is saturated with
OpenAI-shaped self-descriptions; injected contrary options got flat 0.01
treatment and were never chosen in earlier token-steering runs too. It is a
brand prior. It is not evidence about the weights.</p>

<h3 id="probes-tokens">3. Counting tokens</h3>
<p>The API reports an input-token count for every request, and that is not a
self-report - it is an instrument, which we can use to probe the underlying
tokenizer. The decisive test removes everything variable: long,
whitespace-free, per-script samples in which the marginal cost <i>per
character</i> can be read directly, scored against every reference
vocabulary we could fetch:</p>
<div class="scroll"><table class="wide"><tr><th>tokens / character</th>{head}</tr>{rows}</table></div>
<p>Jev's shape is: heavy merging of Latin text and punctuation, but <b>about one
token per character for every other script we tried</b> - Cyrillic, Chinese,
Korean, Greek, Arabic, Hebrew, Thai, Devanagari all sit at 0.92-0.96 - with
astral emoji at ~2 and pure whitespace collapsing to zero (so the server runs a
normalizer before tokenizing). No reference vocabulary reproduces that profile.
Across roughly 128 distinct tokenizers spanning ~30 organizations'
releases, nothing reproduced the profile.</p>
<p>The answer is, frankly, the boring one: the tokenizer, and model, is most
plausibly <b>the vendor's own</b> English/Latin-centric one, with code-point
fallback for everything else - i.e., a new foundation, not a relabeled one.
The OpenAI identity answer was an artifact: a trained prior, well known as
dataset contamination. It is also not at all unreasonable that a startup like
TypeSafe would be capable of training a model of this small size with limited
compute. If you came here looking for a scandal relating to a stolen model
being frankensteined into becoming Jev, we're sorry to have to
disappoint!</p>
</section>"""


def keynotes() -> str:
    return f"""
<section id="keynotes">
<h2>Key notes for Jev users</h2>
<p>The cliff's-notes version: every gotcha a user will actually hit, roughly
in the order you will hit them.</p>
<ul>
<li><b>Order your options deliberately.</b> Position is a first-order factor
wherever content is weak: a U-shaped serial-position curve (~4&times; primacy,
plus a recency bump), and on flat menus the ordering alone can change the
winner. Where content is strong it washes out (~5% of answers flip under
shuffling, with no net direction). For measurement-grade use, rotate the list
and average over 6+ rotations.</li>
<li><b>Probabilities come back on a 0.01 grid.</b> Nothing finer is ever
shown, so differences below 0.01 are invisible. An option can sit at exactly
0.00 - there is no floor - but low-probability answers get drowned in
rounding noise. The vector sums to 0.99 or 1.00, never above. Set your
thresholds accordingly.</li>
<li><b><code>confidence</code> is not new information.</b> For choice it is
the chance-corrected top probability of the same vector, rounded. If you want
that statistic, compute it yourself; do not give it independent trust.</li>
<li><b>Trust the argmax more than the distribution.</b> Greedy accuracy beats
probability-weighted accuracy nearly everywhere ({pct(SC['mmlu']['accuracy'])} vs
{pct(WD['mmlu']['weighted_accuracy_mean'])} on MMLU-Pro): the vector understates how often Jev is
right. Use the choice as the answer and the vector as a ranking signal:
treat probabilities as a ranker, not a well-calibrated distribution.</li>
<li><b>It is not deterministic.</b> Repeats of identical requests differ (TVD
0.03-0.12 on flat menus; less on sharp ones). Do not build logic on one
call's tail probabilities; average repeats where stability matters.</li>
<li><b>Normalize option surface forms.</b> Leading spaces and casing change
scores - the bare token beats its leading-space variant even mid-sentence.
Present options the way you want them judged.</li>
<li><b>Whitespace is nearly free; digits and non-Latin scripts are not.</b>
The server collapses ASCII whitespace runs before tokenizing, while digits
cost ~1 token each and the major non-Latin scripts ~1 token per character
(the rarest blocks ~1 token per byte): number-heavy or CJK/Cyrillic-heavy
states cost several times what the same volume of English text would.</li>
<li><b>It does not see whitespace.</b> Options differing only in spacing or
case score near-identically - a leading space can change the winner - and
its word-menu speech runs words together. Never make whitespace the only
difference between two options.</li>
<li><b>Pack questions into one request.</b> Every question rides the same
forward pass: +{ARCH['headcount']['marginal_ms_per_question']:.2f} ms per question, +{ARCH['optioncount']['marginal_ms_per_option']:.2f} ms per option. A
192-question request costs ~153 ms of server compute; the same four questions
sent separately cost ~280 ms of wall each. Put shared context in
<code>state</code> once. Extra options and rubric levels are essentially
free: they cost only their own tokens' prefill.</li>
<li><b>The envelope is solid.</b> Across ~16k benchmark requests: 19
contract-invalid responses and zero schema drift. Limits: 255 options, 2-10
levels, 32k-token state, 64k total; oversized requests fail with
<code>max_tokens_exceeded</code>.</li>
<li><b>Its knowledge stops around mid-2025.</b> Solid to late 2024, partial
to ~May 2025, nothing after June 2025 in our dated bisection. Do not ask it
about last week - and if you want its abstention, offer an explicit
&ldquo;cannot say&rdquo; option; it uses one.</li>
<li><b>It knows nothing about who it is.</b> It has undergone no SFT in
that regard; ask its origin and training-text contamination answers - it
will say OpenAI, and it accepts a fictional origin story at 0.97 if you
assert one in the prompt.</li>
<li><b>And the headline: it is not frontier.</b> Respectable 2025-era
small-model knowledge ({pct(SC['mmlu']['accuracy'])} MMLU-Pro, {pct(SC['gpqa']['accuracy'])} GPQA Diamond),
{pct(SC['hle']['accuracy'])} on HLE's multiple-choice subset, zero exact grids on
ARC-AGI-2. The <a href="#benchmarks">benchmark section</a> has the full
picture.</li>
</ul>
</section>"""


def conclusion() -> str:
    mmlu = pct(SC["mmlu"]["accuracy"]); gpqa = pct(SC["gpqa"]["accuracy"])
    arc = pct(SC["arc"]["accuracy"]); jm = JEV_COST["mmlu_pro"]
    return f"""
<section id="conclusion">
<h2>So, is it worth your attention?</h2>
<p>It is not what the landing page says. TL/DR, Jev is a small, new,
English-centric foundation model with a
probability read-out bolted where a language head usually goes - respectable
general knowledge ({mmlu} on MMLU-Pro, {gpqa} on GPQA Diamond, {arc} on the
saturated ARC-Challenge), no chance against a 2026 frontier model, and a
self-narrative inherited from internet text rather than from its own
lineage. If you buy it expecting the frontier, you will be disappointed, and
you should not have to read an independent report to find that out.</p>
<p>But here is the part the framing buries, and it is why we kept going: what
Jev <i>is</i> is genuinely nice. There is a real product shape that this
occupies well - fast, deterministic-shaped, cheap structured judgement. Pick a
routing decision from a menu of a hundred intents; grade an answer against a
rubric; score which of twenty rewrites a user most likely meant; gate a
pipeline on whether a document is about X; give an ordinal quality signal on
an agent's last step. It answers in tens of milliseconds of compute, in a
schema that never drifts, with probabilities attached, and it costs
${jm:.2f} per ten-thousand-odd hard questions - numbers a reasoning model
cannot approach because it is doing a different job. The prefill-only economics
we measured are not a trick; they are a genuine consequence of deciding to
trade generation for judgement, and for control-plane uses that trade is often
the right one.</p>
<p>Our own favorite evidence that this is a real thing and not a demo: the
model that <i>cannot hallucinate</i> still confidently picks wrong answers
about a quarter of the time on physics, still wants you to believe it came from
OpenAI, and turned out to be a tokenizer fingerprint nobody could match to an
existing model. It earns attention by being useful, not by being frontier. That
is a perfectly good thing to be. Just say so.</p>
</section>
<footer>
<p><b>Reproducibility.</b> This report was created from the run artifacts in this
repository by <code>scripts/report/build_report.py</code>; every number is read
from disk at render time, and the charts are produced by
<code>comparison_graphs.py</code>, <code>pareto_graphs.py</code> and
<code>arch_evidence.py</code>. Our own probing transcripts (the Talk-to-Jev
traces and the architecture-probe rows) are published verbatim; benchmark raw
logs are held locally because they embed licensed dataset text.</p>
</footer>
{refs()}"""


def main() -> None:
    SITE.mkdir(exist_ok=True)
    body = "".join([hero(), pitch(), methodology(), architecture(None),
                    latency(), benchmarks(), pareto(), probes(), keynotes(),
                    conclusion()])
    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Jev: Not Frontier, But Still Worth Your Attention</title>
<style>{CSS}</style></head><body><main>{body}</main></body></html>"""
    out = SITE / "index.html"
    out.write_text(html, encoding="utf-8")
    print(f"[ok] wrote {out.relative_to(ROOT.parent)} ({out.stat().st_size:,} bytes), "
          f"{len(_NUM)} notes")


if __name__ == "__main__":
    main()
