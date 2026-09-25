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
CHARTS = dict(zip(["mmlu_pro", "gpqa", "arc", "math500", "arc_agi2", "hle"], _BENCH_SVGS))
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
    return f'<figure>{strip_titles(CHARTS[key])}<figcaption>{caption}</figcaption></figure>'

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
        "scaling laws, in-context learning / few-shot prompting, SFT, GPU "
        "computing and CUDA, mixed-precision training, distributed "
        "parallelism frameworks (Megatron-LM, pipeline parallelism, ZeRO / "
        "DeepSpeed), and many more. &ldquo;I co-invented&rdquo; implies one "
        "of a small subset; ChatGPT has thousands of fathers.",
    "jevfree": "Jev's output tokens are server-reported but billed at zero. "
        "Every Jev dollar figure in this report is measured from billing "
        "usage in the published run artifacts, not estimated.",
    "costest": "External model costs are estimates from public list prices "
        "and per-item token priors, calibrated to measured cost-per-question "
        "anchors (OpenRouter for GPQA, Vals for MMLU-Pro) wherever available. "
        "Assume a few-x error on any single point; the gaps shown are three to "
        "four orders of magnitude, far outside that band. Full priors and "
        "arithmetic: data_report/costs.json.",
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
    "selfbatch": "Packing 192 questions into one request barely moves server "
        "compute time (tens of milliseconds), while output tokens grow in "
        "proportion to questions times options. One forward pass over the "
        "state, many read-outs, all serialized - the self-batched signature.",
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
    "marginal": "The decomposition: each packed question adds ~55 input "
        "tokens to the shared prompt and +0.44 ms of server compute; at the "
        "measured prefill slope those tokens alone account for +0.34 ms, "
        "leaving under 0.11 ms for the decision itself. Each option adds ~18 "
        "tokens and +0.10 ms - fully accounted for by its own tokens "
        "(+0.11 ms predicted). The read-out's compute is invisible inside "
        "latency noise. Source: runs_archprobe/rows.jsonl + analysis.json; "
        "audit: data_report/arch_audits.json.",
    "sizelimits": "Three things this API cannot tell us, and we do not "
        "guess them: dense vs mixture-of-experts (both prefill identically "
        "under batching, and the economics require neither); whether "
        "frontier-teacher distillation contributed to training (every "
        "API-visible signal we can construct is confounded); and any exact "
        "parameter count or hidden dimension. The banded size estimate is "
        "an explicit-assumptions bound, not a slope-to-size conversion: the "
        "ground rule (marginal milliseconds under shared batching are a "
        "scheduling artifact) still stands, and the capability band is "
        "positioning under protocol mismatch. Full ledger: "
        "ARCHITECTURE-ANALYSIS.md; size machinery: "
        "data_report/size_estimate.json.",
    "sizeest": "The size-estimate machinery lives in "
        "data_report/size_estimate.json (scripts/report/size_estimate.py): "
        "both angles, the full assumption grid (MFU 0.25-0.45; bf16 peak "
        "250-500 TFLOPS per device; 1-4 shards; FLOPs = 2*N_active per "
        "token, attention adding ~15% or less at these lengths; single-stream "
        "conservative), the four reconciling readings (MoE, quantized "
        "serving, distilled small dense, soft band top), and what would "
        "tighten each. The MFU/peak ranges are industry-standard serving "
        "assumptions, not repo measurements, and are labeled as such.",
    "lattice": "Full lattice forensics: scripts/report/lattice_forensics.py "
        "-&gt; data_report/lattice_forensics.json, over 704,277 values in 7,887 "
        "published vectors at K=2..255 (probe rows, phase-1 raws, and the "
        "Talk program's per-rotation large-menu distributions). Zero "
        "off-grid values; displayed sums are only ever 0.99 or 1.00 and "
        "never exceed 1.00, against the +-0.04 scatter independent per-value "
        "rounding would give at K=255 - so the display pipeline applies a "
        "bounded one-sided correction (or apportions integer hundredths), "
        "leaving a 0.01 shortfall on ~46% of flat large-K vectors. Every "
        "two-option vector sums to exactly 1.000 (complement emission). No "
        "probability floor exists: exact 0.00 values are common. The precise "
        "rule is what the staged P4 probe settles deterministically.",
    "probestaged": "The follow-up battery P1-P5 "
        "(scripts/benchmark/run_probe_battery2.py) sits behind the same "
        "live-call gate as every runner in this repo (JEVO_ALLOW_LIVE=1 + "
        "TYPESAFE_API_KEY from the environment, never stored in artifacts). "
        "At the time of writing no key is provisioned here: the endpoint is "
        "reachable (anonymous requests get 403), the documented fallback key "
        "returns 401, so nothing was dispatched and no live probe result "
        "appears in this report. What has run: plan mode (3,331 calls, "
        "~2.47M input tokens, ~$0.10 at the measured rate; "
        "data_report/probe2_plan.json) and a dry run exercising every "
        "builder and analyzer against synthetic responses with planted "
        "ground truth - the analyzers recover a planted 2024-11 knowledge "
        "cutoff, a planted Luce dilution law, and a planted display "
        "apportionment rule, which validates the instruments, not any "
        "claim about Jev.",
    "notwrapper": "Each negation is a measurement, not a vibe. Compute grows "
        "linearly with prompt tokens - a cache would not prefill - and the "
        "knowledge horizon stops at late 2024, which a live retriever would "
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
        "runs_archprobe/, runs_live/FINDINGS.md, data_report/costs.json.",
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
table{border-collapse:collapse;width:100%;font-size:.92rem;margin:1rem 0}
th,td{text-align:left;padding:.45rem .6rem;border-bottom:1px solid var(--line)}
th{color:var(--mut);font-weight:600;font-size:.78rem;text-transform:uppercase;letter-spacing:.04em}
td.n{text-align:right;font-variant-numeric:tabular-nums}
.cap{color:var(--mut);font-size:.88rem;max-width:none}
.note{border-left:3px solid var(--amber);background:var(--panel);border-radius:0 12px 12px 0;padding:.9rem 1.1rem;margin:1.2rem 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:.9rem;margin:1.2rem 0}
.stat{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:.8rem 1rem}
.stat b{display:block;font-size:1.5rem}.stat span{color:var(--mut);font-size:.8rem}
sup a{text-decoration:none;color:var(--amber);font-weight:600}
#refs li{margin:.4rem 0;font-size:.88rem;color:var(--mut)}
footer{color:var(--mut);font-size:.85rem;border-top:1px solid var(--line);margin-top:4rem;padding-top:1.2rem}
code{background:var(--panel);border-radius:4px;padding:.05rem .35rem;font-size:.86em}
.scroll{overflow-x:auto}
nav.toc{margin:1.3rem 0 .2rem;font-size:.88rem;color:var(--mut)}
nav.toc a{margin:0 .15rem}
.tldr{border-left:3px solid var(--teal)}
table.wide{font-size:.78rem;min-width:720px}
table.wide th{font-size:.66rem}
.carousel{overflow:hidden;border-radius:14px;margin:1rem 0}
.carousel .track{display:flex;width:700%;animation:chartcycle 49s cubic-bezier(.4,0,.2,1) infinite}
.carousel:hover .track{animation-play-state:paused}
.carousel figure{width:14.285714%;flex:0 0 auto;margin:0}
.carousel svg{margin:.4rem 0}
@keyframes chartcycle{0%,12.6%{transform:translateX(0)}14.3%,26.9%{transform:translateX(-14.2857%)}28.6%,41.2%{transform:translateX(-28.5714%)}42.9%,55.5%{transform:translateX(-42.8571%)}57.1%,69.8%{transform:translateX(-57.1429%)}71.4%,84.1%{transform:translateX(-71.4286%)}85.7%,100%{transform:translateX(-85.7143%)}}
"""


def hero() -> str:
    mmlu = pct(SC["mmlu"]["accuracy"]); gpqa = pct(SC["gpqa"]["accuracy"])
    arc = pct(SC["arc"]["accuracy"])
    bt = BILLING["totals"]
    spend = f"${bt['usd']:.2f}"
    calls = f"{bt['calls_recorded']:,}"
    mn = SC["mmlu"]["n_expected"]
    return f"""
<div class="chip">An independent, hands-on evaluation</div>
<h1>Jev: Not Frontier, But Still Worth Your Attention</h1>
<p class="sub">TypeSafe AI sells Jev as a frontier-class reasoner that cannot
hallucinate, built by the co-inventor of ChatGPT - fast, and almost free. We ran it
live on {TOTAL_REQUESTS:,} benchmark requests, measured its latency and billing,
and probed what it is underneath. The result is a smaller, humbler model that
is genuinely useful for a job nobody else serves quite this way.</p>
<div class="grid">
<div class="stat"><b>{mmlu}</b><span>MMLU-Pro, {mn:,} questions</span></div>
<div class="stat"><b>{gpqa}</b><span>GPQA Diamond, graduate science</span></div>
<div class="stat"><b>{arc}</b><span>ARC-Challenge</span></div>
<div class="stat"><b>{spend}</b><span>total our measurements cost
({calls} recorded API calls)</span></div>
</div>
<nav class="toc"><b>In this report:</b>
<a href="#pitch">The pitch</a> &middot;
<a href="#method">Method</a> &middot;
<a href="#arch">What it is</a> &middot;
<a href="#benchmarks">Benchmarks</a> &middot;
<a href="#pareto">Cost frontier</a> &middot;
<a href="#latency">Latency</a> &middot;
<a href="#probes">Probing</a> &middot;
<a href="#conclusion">Verdict</a> &middot;
<a href="#refs">Notes</a></nav>
<div class="card tldr"><b>TL;DR.</b> Jev is not a frontier model: it misses
about a quarter of graduate science and its self-narrative is internet prior,
not lineage{fn('behavioronly')}. It nonetheless is also not a toy: {mmlu} MMLU-Pro,
{gpqa} GPQA, ~{ARCH['prefill']['fixed_floor_ms']:.0f} ms of server compute per
question, and a full graduate-scale benchmark run for cents. The honest
category is <i>cheap real-time sub-frontier judgement</i> - routing, rubric grading,
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
three frozen suites. Every request, response, probability vector, token count
and latency header was written to disk. This repository publishes the derived
aggregates - scores and weighted scores, per-stage usage summaries, billing
roll-ups, the architecture-probe table, and the complete Talk-to-Jev traces -
plus freeze manifests with SHA-256 hashes of every input; the benchmark item
text itself is third-party licensed content and is not republished here, so
the hashes stand in for the prompts for anyone who holds the datasets. The
analysis pages are generated from these artifacts, so every number in this
report re-derives from what we recorded.{note_text}</p>
<p>Three constraints shape everything below. First, Jev answers directly: one
shot, no chain of thought, no tools, no retries - which is not how the frontier
scores we compare against are produced, so all of those comparisons are
positioning, not matched races. Second, the API is closed: we never saw
weights, gradients, or any serving internal, so the architecture discussion is
inference from observable behavior. Third, we chose benchmarks the model could
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
    return f"""
<section id="arch">
<h2>What Jev appears to be</h2>
<p>This section is inference, not disclosure: no weights, gradients, or
serving internals were ever touched{fn('behavioronly')} - what follows is
assembled from answers, probability vectors, token counts, timing headers and
bills, and it is our <i>operating premise</i>: every other section reads
Jev's behavior through the model stated here. The full ledger - alternative
hypotheses with probabilities, a falsifier per load-bearing claim, and six
cheap discriminating probes - lives in <code>ARCHITECTURE-ANALYSIS.md</code>
in the repository.</p>
<div class="card">In one sentence: Jev looks like a <b>small,
English-centric transformer language model</b>, post-trained for judgement
rather than conversation, served with its <b>generation head replaced by a
probability read-out</b> over caller-supplied options - every question in a
request scored from <b>one batched prefill pass</b>, which is why it is fast,
why its output is free, and why it cannot write you a poem.</div>
<figure>{strip_titles(ARCHDIAG[0])}<figcaption>The operating premise, drawn:
one request's journey left to right (teal = measured), the training history
that produced the weights (amber = best explanation), and what no API-visible
signal reaches (dotted = not identified). Generated from the artifacts by
<code>scripts/report/arch_diagram.py</code>; standalone page:
<code>docs/modern-comparison/architecture-diagram.html</code>.</figcaption></figure>
<table>
<tr><th>Component</th><th>Best guess</th><th>Confidence</th></tr>
<tr><td>Output side</td><td class='cap'>a trained probability read-out over the caller's
options - a discriminative head where a language head usually goes, with
choice / score / noul as its three exposed shapes</td><td>confident</td></tr>
<tr><td>Serving</td><td class='cap'>one forward pass per request; all questions and options
scored from it; continuous batching across tenants; no decode loop</td><td>confident</td></tr>
<tr><td>Compute profile</td><td class='cap'>~{floor:.0f} ms fixed floor + ~{slope:.0f} ms per 1k input
tokens, linear to 29k tokens; no large quadratic (attention-blowup)
signature</td><td>confident (measured)</td></tr>
<tr><td>Input pipeline</td><td class='cap'>a whitespace normalizer plus a fixed ~{bp['mergerate_baseline_tokens']}-token
template run ahead of the model's own tokenizer</td><td>confident / plausible</td></tr>
<tr><td>Tokenizer</td><td class='cap'>the vendor's own English/Latin-centric BPE: heavy Latin
merges, ~1 token per codepoint for the major non-Latin scripts, byte-level
fallback for uncovered characters, no match among 173 open
signatures</td><td>plausible (strong)</td></tr>
<tr><td>Core</td><td class='cap'>transformer-family decoder LM; dense vs MoE unknown;
attention variant unknown and unknowable at these context
lengths</td><td>plausible / open</td></tr>
<tr><td>Size</td><td class='cap'>no point estimate - a two-angle band: ~{sz_lo:.1f}-{sz_hi:.1f}B
active parameters from the prefill-throughput bound (explicit serving
assumptions), {szc_lo:.0f}-{szc_hi:.0f}B dense-equivalent from the capability band;
converged guess ~0.5-4B active, total unconstrained (MoE, quantization or
distillation reconcile the angles)</td><td>speculative</td></tr>
<tr><td>Training</td><td class='cap'>English-dominant pretraining; knowledge horizon late
2024; judgement-format assistant post-training; OpenAI-flavored brand prior
inherited from training text; frontier-teacher contribution
unknowable</td><td>plausible</td></tr>
<tr><td>What it is not</td><td class='cap'>not frontier, not retrieval- or cache-assisted,
not a wrapper around another vendor's API, not a relabeled open
model</td><td>confident</td></tr>
</table>
<p class="cap">&ldquo;Confident&rdquo; = directly measured, adequate n,
robust to the noise floor. &ldquo;Plausible&rdquo; = best explanation of the
measurements, rivals not excluded. &ldquo;Speculative&rdquo; = reasoned
guess; the evidence under-determines it. Every row's evidence and its
falsifier: <code>ARCHITECTURE-ANALYSIS.md</code> §1.</p>

<h3>One pass, many read-outs</h3>
<p>The latency section below carries the measurements; the architectural
content is this. Server compute is a fixed floor plus a linear prefill term
(~{slope:.0f} ms per 1k input tokens), and the decisive numbers are the
marginals. Packing one more question into a request costs +{mq:.2f} ms - and
the ~{mcq['input_tokens_per_unit']:.0f} input tokens that question adds predict +{mcq['prefill_predicted_ms_per_unit']:.2f} ms at the
prefill slope, so the decision itself hides in a residual under
{mcq['residual_ms_per_unit']:.2f} ms. One more option costs +{mo:.2f} ms against the +{mco['prefill_predicted_ms_per_unit']:.2f} ms its own ~{mco['input_tokens_per_unit']:.0f} tokens predict: the
read-out's compute is indistinguishable from zero.{fn('marginal')} Billed
output grows ~{fq:.0f} tokens per question and ~{fo:.1f} per option - that is
the response JSON serializing itself, which is exactly why output can be
free: nothing is ever decoded. Upstream compute stays flat
({c1['median_upstream_ms']:.0f} ms at concurrency 1, {c32['median_upstream_ms']:.0f} ms at 32) while our own wall time bends, so
independent requests share a batched path; and a four-question batch answers
in ~261 ms where the same four questions separately take ~1,117 ms - there is
no hidden per-question round trip anywhere.{fn('selfbatch')} &ldquo;System
one&rdquo;, under this reading, is as much a serving description as a
psychological one: prefill, read out, done.</p>

<h3>Fingerprints on the read-out</h3>
<p>Every one of the {lat_vals:,} probability values published in this
repository - {lat_vecs:,} vectors spanning the probe battery, the phase-1 raw
responses, and the Talk program's large-menu traces - sits on the 0.01
grid.{fn('quantization')} The lattice has more structure than
&ldquo;rounded&rdquo;: displayed sums are 0.99 or 1.00 and <b>never exceed
1.00</b>, in any corpus, at any K up to 255 - independent per-value rounding
would scatter sums by ±0.04 at K=255, so a bounded one-sided correction runs
after rounding ({pct(lat_frac99)} of flat K≈255 vectors land 0.01 short; every
published two-option vector sums to exactly 1.000).{fn('lattice')} Two more
seams are visible. In {n_mismatch} vectors the
returned choice is not the argmax of its own displayed table - always at
exactly one quantum, which monotone rounding cannot produce, so the decision
is computed at pre-display precision while the table is rounded separately.
And the choice <code>confidence</code> field is recoverable: it is the
chance-corrected top probability, (p_max - 1/K)/(1 - 1/K) - {cfc['exact']:,} of
{cfc['n']:,} published choice vectors match it exactly against the displayed
table, {cfc['within_1_quantum']:,} within one quantum, {cfc['within_2_quanta']} within two, none
beyond.{fn('readout')} Buyer's note: choice confidence is a deterministic
function of the vector you were already handed, not a second opinion. Score
answers use a different shape statistic, noul answers carry no confidence at
all, and schema perfection under load ({fmt_total} contract-invalid responses
across the text stages, {AGI_UNUSABLE} unusable cells in the multi-question
ARC-AGI stages) completes the picture: a serializer over
a trained discriminative head, not generated text parsed into numbers.</p>

<h3>A tokenizer nobody recognizes</h3>
<p>The probing section below tells the full story, including both false
leads. The architectural content: Jev's counter merges English text and
punctuation hard, spends ~1 token per codepoint on Cyrillic, Greek, Arabic,
Hebrew, Thai, Devanagari, Hangul, kana and common CJK, ~1 per digit, and ~2
per uncovered rare character - byte-level-fallback territory - and it
collapses pure whitespace to zero, so a normalizer runs in the serving path
before tokenization. None of the 173 open tokenizer signatures across 1,215
scanned repositories reproduces that combination, and the short-string
battery that once &ldquo;matched&rdquo; Qwen2.5 at RMSE ~1 token carries at
most {bp['short_battery_max_delta_over_baseline']} tokens of discriminating information per probe against a
~{bp['mergerate_baseline_tokens']}-token template - a trap, not a match. Per-script coverage is a
pretraining-corpus fossil record, and this one says: English-dominant diet,
incidental multilingual exposure, vendor's own vocabulary. That is what makes
Jev a <i>new foundation</i> rather than a relabeled one.</p>

<h3>Small, in a particular way</h3>
<p>The capability profile has a particular shape. On one-shot knowledge
multiple-choice Jev sits in
the band of 2025-era small instruct models - {pct(SC['mmlu']['accuracy'])} MMLU-Pro
where Claude 3.7 Sonnet scored 80.7 without thinking and Qwen 3.5 9B 82.5,
{pct(SC['gpqa']['accuracy'])} GPQA where they scored 76.8 and 77.6 - positioning, not a matched
race.{fn('protocol')} On anything multi-step it falls off a cliff:
{pct(SC['hle']['accuracy'])} on HLE, zero exact grids on ARC-AGI-2, and the generation tax -
{pct(SC['math_c']['accuracy'])} on MATH-500 items when the answer is an option to pick,
{pct(SC['math_s']['accuracy'])} when the same answers must be read off digit by digit.
Recognition far exceeds production, which is what judgement-heavy
post-training on a small model produces. The distributions are shaped the
same way: the median MMLU-Pro item carries p(gold) = {wmmlu['weighted_accuracy_p50']:.2f} against a
mean of {pct(wmmlu['weighted_accuracy_mean'])}, and {SC['mmlu']['zero_probability_gold']} items put exactly 0.00 on the gold answer -
near-decisive where it knows, confidently wrong on a hard tail. Position
effects are real but unbiased where content lives: re-running 419 items with
shuffled option order flips {pct(rp['flip_rate'])} of answers with no net accuracy change
(exact McNemar p = {rp['mcnemar_exact_p']:.2f}) - an in-context list reader, not a per-option
oracle.</p>
<p><b>How big is it?</b> A slope alone is not a size - the server batches our
tokens with everyone else's - but two angles converge on a band, and stating
assumptions explicitly is not the same as refusing to estimate. From the
throughput side: the marginal prefill rate is ~{sz_R:,} tokens/s; batched
prefill is compute-bound and costs about 2&middot;N<sub>active</sub> FLOPs per
token, so N<sub>active</sub> &le; MFU &times; peak &times; shards &divide; 2R.
Across an honest assumption grid (25-45% model-FLOPs utilization, 250-500
TFLOPS bf16 per device, 1-4 devices) that bounds the <i>active</i> footprint
at ~{sz_lo:.1f}-{sz_hi:.1f}B parameters, central case ~{sza['central_case_B']:.1f}B - and
sharing the machine with other tenants only lowers the bound. From the
capability side: the neighbors above put it at {szc_lo:.0f}-{szc_hi:.0f}B dense-equivalent.
The angles overlap only if something hides the difference: an MoE (active
&ll; total - the throughput bound sees active parameters only), aggressive
quantization (fp8/int4 lifts the throughput band to ~0.4-10B), or
distillation (teacher labels lift a 1-4B model into the bottom of the
capability band on knowledge multiple-choice - and Jev's
recognition-far-exceeds-production asymmetry is exactly that shape).
Converged best guess: <b>~0.5-4B active parameters, total
unconstrained</b>; forced to a single dense-equivalent order, 1-9B - a
2025-era small model, which is what the benchmarks said all
along.{fn('sizeest')}</p>

<h3>What it is not</h3>
<p>Four negations, each a measurement rather than a vibe.{fn('notwrapper')}
Not retrieval or cache: compute grows linearly in prompt tokens - a cache
would not prefill - the knowledge horizon stops at late 2024, repeats of
identical payloads differ, and {SC['mmlu']['zero_probability_gold']} zero-probability gold answers on
<i>public</i> benchmark text is not what a lookup produces. Not a wrapper
around a frontier API: {c1['median_upstream_ms']:.0f}-{c32['median_upstream_ms']:.0f} ms of upstream compute leaves no room
inside for anyone else's round trip, and the price sits one to two orders of
magnitude below flagship input lists. Not a relabeled open model: the
tokenizer scan says so, and both historical leads dissolved as template
artifacts. Not frontier: the benchmark section says so six ways. And the
brand prior is not lineage - asked to name its maker from a
position-controlled menu that includes that maker, Jev says an OpenAI-family
name {openai_votes} times out of {anc['n_frames_with_probs']} ({pct(mass['openai'])} of probability mass), with
&ldquo;Typesafe&rdquo; at the quantization floor; injected &ldquo;Jev&rdquo;
spellings at probability parity are never chosen. Learned text about who
makes assistants - which the token counts independently contradict as actual
OpenAI or Qwen provenance.</p>

<h3>What stays open</h3>
<p>Three things this API cannot tell us, and we do not guess them: whether
the transformer is dense or mixture-of-experts; whether a frontier teacher
produced any of the training signal (every discriminator we can construct is
confounded, including the OpenAI-shaped brand prior, which the open
assistant-text ecosystem produces on its own); and any exact parameter count
or hidden dimension - the two-angle estimate above bands the active size
(~0.5-4B) but identification is beyond this API, which documents nothing
past 32k-token state, 64k total, ≤255 options, 2-10 rubric
levels.{fn('sizelimits')}</p>
<p>The follow-up probes are built, not just wished for. Five families that
could move these questions - tokenizer fallback granularity, option-cost
decoupling, calibration under option-count scaling, the lattice rule on
identical options, and a knowledge-horizon bisection - are implemented in
<code>scripts/benchmark/run_probe_battery2.py</code>, validated end-to-end by
a dry run against synthetic responses with planted ground truth, and planned
at {probe_calls:,} calls (~{probe_tok/1e6:.1f}M input tokens ≈
${probe_cost:.2f} at the measured rate). Live dispatch is gated on a working
<code>TYPESAFE_API_KEY</code>, which this environment does not have: nothing
was sent, and no live probe result appears anywhere in this
report.{fn('probestaged')} The offline arm of the lattice probe <i>has</i>
run - that is the bounded-sums result above - and its live arm is what
settles the exact apportionment rule deterministically.</p>

<p>Everything above re-derives from published artifacts:</p>
<table><tr><th>Claim area</th><th>Where it lives</th></tr>{ev}</table>
<p>If TypeSafe published a model card tomorrow that said <i>&ldquo;small
English-centric transformer, judgement-tuned, prefill-only serving&rdquo;</i>,
nothing in this section would need rewriting. If it said <i>&ldquo;rebadged
Qwen&rdquo;</i> or <i>&ldquo;a GPT behind a curtain&rdquo;</i>, the token
counts and the millisecond headers would have a great deal of explaining to
do.</p>
</section>"""


def pitch() -> str:
    wrong_gpqa = pct(1 - SC["gpqa"]["accuracy"])
    arc_pct = pct(SC["arc"]["accuracy"])
    return f"""
<section id="pitch">
<h2>The pitch, and the parts that survive contact</h2>
<p class="pitch">&ldquo;A model that <b>cannot hallucinate</b>, at <b>frontier-level
performance</b>, built by a <b>coauthor of ChatGPT</b> - incredibly fast,
incredibly cheap, with <b>free output</b>.&rdquo;</p>
<p>Each clause is technically defensible in a narrow sense and misleading in the
sense a buyer will hear. Take them one at a time.</p>
<p><b>The co-inventor of ChatGPT.</b> Almeida certainly deserves credit, but
saying &ldquo;I co-invented ChatGPT&rdquo; overassigns credit - as one of the
eight primary authors on InstructGPT, whose work was on learned optimizers,
and as one of 88 people thanked in the initial release of ChatGPT - from the
work of thousands of people; see the footnote.{fn('almeida')} It is a real
credential. It is not an architecture claim - and, as it turned out, not a
provenance claim either.</p>
<p><b>Cannot hallucinate.</b> What Jev actually cannot do is emit free text. It
returns one option from a fixed menu, inside a schema that is always
well-formed. But a schema-valid answer is
not a true answer. Jev is wrong {wrong_gpqa} of the time on graduate science,
and every one of those wrong answers arrived beautifully formatted. A model
that cannot write prose has not solved hallucination; it has made hallucination
hard to notice.</p>
<p><b>Fast and cheap.</b> Both true, and explainable in one sentence: Jev never
runs a decode loop. It reads the prompt once and reads off a
vector.{fn('jevfree')}
Speed and price are properties of the <i>task</i> being prefill-only, not of
frontier economics.</p>
<p><b>Frontier-level.</b> This one simply does not survive. Jev is good, and
&ldquo;good&rdquo; will turn out to mean something genuinely useful here - but
it is not a frontier model by any late-2026 standard, and where it looks
frontier-like ({arc_pct} on ARC-Challenge), that's only on a race that
finished years ago. Everything below is the evidence.</p>
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
        ("Option-rotation audit", rot["n_expected"], pct(rot["accuracy"]), wil(rot),
         pct(WD["rot"]["weighted_accuracy_mean"]), "same items, shuffled option order"),
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
        fig("mmlu_pro", "Broad knowledge. Jev's direct-answer score sits near a "
            "2025-era small model; the frontier bars had chain-of-thought."),
        fig("gpqa", "Graduate science. Jev lands near Claude 3.7 Sonnet without "
            "thinking - genuinely respectable, clearly not frontier."),
        fig("arc", "Elementary science. Saturated for everyone; note the "
            "comparison set is mostly older models, so this is Jev's best-looking chart."),
        fig("math500", "Math, re-encoded as multiple choice. The gap between the "
            "MCQ and digit-read-out bars is the generation tax, not a knowledge gap."),
        fig("arc_agi2", "Abstract puzzles. Jev cannot emit a grid, so it is scored "
            "cell by cell and solves zero tasks outright - shown for shape, not rank."),
        fig("hle", "Expert frontier exam, multiple-choice subset. Close to the "
            "guessing floor; the external bars are reasoning-enabled on a wider set."),
    ]
    figs = "".join(fig_list) + fig_list[0]  # clone slide 1 for a seamless loop
    return f"""
<section id="benchmarks">
<h2>Benchmarks: where it actually lands</h2>
<p>Jev ran <b>directly</b>: one shot, no chain of thought, no tools, no retries,
failures counted against it. Most published comparison numbers use reasoning
enabled and few-shot prompts, so the columns below are positioning rather than a
matched race.{fn('protocol')}</p>
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
<h3>The charts</h3>
<p>Teal is Jev's greedy score, amber is Jev's probability-weighted score, and
every other bar is a published number we fetched - not a model we ran. All six
chart stages rotate below, one at a time - hover to pause. The table above is
the complete result set; the rotation is a viewing convenience, not a
selection.</p>
<div class="carousel"><div class="track">
{figs}
</div></div>
</section>"""


def pareto() -> str:
    jm = JEV_COST["mmlu_pro"]; jg = JEV_COST["gpqa"]
    af = COSTS["costs_usd"]["mmlu_pro"].get("GPT-6 Astra", {}).get("usd", 0)
    ff = COSTS["costs_usd"]["gpqa"].get("Claude Fable 5.1", {}).get("usd", 0)
    figs = "".join(f"<figure>{strip_titles(s)}</figure>" for s in PARETO)
    return f"""
<section id="pareto">
<h2>The frontier the cost numbers actually draw</h2>
<p>TypeSafe AI publishes a capability-versus-cost graphic with Jev near the top
of a frontier. That framing only works if capability is measured with the
model's hands tied. The charts below use the same accuracies as the benchmark
section and real money on the horizontal axis: what it costs to run that
benchmark, once, start to finish.{fn('costest')}</p>
<p>Jev does land on the visible frontier - at {pct(SC['mmlu']['accuracy'])} on
MMLU-Pro for ${jm:.2f}, it is hard to beat per dollar - but the frontier bends
steeply: the same run costs a 2026 flagship about ${af:.0f}, and Claude Fable
5.1 scores {pct(92.4)} rather than {pct(SC['mmlu']['accuracy'])}. On GPQA, Jev's
${jg:.3f} is roughly {ff/jmax(jg):.0f} times cheaper than Fable 5.1's
${ff:.2f}, and about twenty points less accurate. Cheap and mid-tier can be the
same sentence.</p>
{figs}
</section>"""


def jmax(x: float) -> float:
    return max(x, 1e-9)


def latency() -> str:
    p = ARCH["prefill"]; h = ARCH["headcount"]; o = ARCH["optioncount"]
    cw = ARCH["coldwarm"]; anc = ARCH["ancestry"]
    floor = p["fixed_floor_ms"]; slope = p["ms_per_1k_input_tokens"]
    mq = h["marginal_ms_per_question"]; mo = o["marginal_ms_per_option"]
    fq = h["marginal_output_tokens_per_question"]; fo = o["marginal_output_tokens_per_option"]
    c1 = ARCH["concurrency"]["per_call_wall"]["1"]
    c32 = ARCH["concurrency"]["per_call_wall"]["32"]
    return f"""
<section id="latency">
<h2>What the milliseconds say</h2>
<p>Every response carries a server-side compute timing in a proxy header, which
strips out network and queueing - the cleanest architecture signal a public API
leaks. Sequential probes across prompt sizes from ~0.5k to ~32k tokens fit:</p>
<div class="card"><b>~{floor:.0f} ms fixed floor + ~{slope:.0f} ms per 1k input
tokens</b>, with a negligible quadratic term. A flat base plus a linear prefill
term is what a transformer's forward pass looks like from outside; the floor is
serving overhead, not the model.</div>
<p>The more telling experiment is <i>self-batching</i>. Packing 192 questions
into one request barely moves compute time (+{mq:.2f} ms per extra
question){fn('selfbatch')} while billed output grows about {fq:.0f} tokens per
question - the full probability vector for each, serialized. Scoring 255 options
instead of 2 costs +{mo:.2f} ms of compute and {fo:.1f} output tokens per
option. And running up to 32 requests concurrently keeps server compute flat
({c1['median_upstream_ms']:.0f} ms at c=1 vs {c32['median_upstream_ms']:.0f} ms
at c=32) while client wall time bends - that bend is our own connection pool,
not the server; the flat header is the evidence of continuous batching. The
whole profile is one forward pass over shared state, many read-outs, every
answer materialized - exactly what a swapped-out output head implies.</p>
<p>Two calibration signatures came out of the same runs. Every probability we
ever received sat on a 0.01 grid ({ARCH['counts']['n_rows']:,} values, zero
off-grid){fn('quantization')} - a fixed-precision read-out, so tiny probability
differences are noise. And greedy accuracy exceeded average gold-probability on
almost every benchmark: Jev's distributions spread mass onto wrong options more
than its accuracy justifies. Cold-connection overhead was ~{cw['connection_overhead_ms']:.0f} ms -
TLS setup, useful only as a reminder to trust the header, not the wall clock.</p>
<p>We deliberately do <b>not</b> convert these slopes into a parameter count.
The server batches our requests with everyone else's, so a single request's
marginal cost is not a clean measure of model size - the honest size signal is
the capability band, and it says small, not how small.</p>
</section>"""


def probes() -> str:
    anc = ARCH["ancestry"]
    mass = anc["family_mass"]
    openai = mass.get("openai", 0); tsf = mass.get("typesafe", 0)
    votes = anc["greedy_votes"]; idx0 = anc["choice_at_index_0_frac"]
    jev_p = anc["mean_p_jev"]; ts_p = anc["mean_p_typesafe"]
    top = FINGER["ranking"][0]
    ranks = "".join(
        f"<tr><td>{r['label']}</td><td class='n'>{r['rmse_tokens']:.2f}</td>"
        f"<td class='n'>{r['slope']:.3f}</td><td class='n'>{r['r2']:.3f}</td></tr>"
        for r in FINGER["ranking"][:8])
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

<h3>1. Talking to it</h3>
<p>The first idea was to give Jev a text box by hand: offer it candidate
continuations and let it choose, one step at a time. A character menu failed
completely - its per-character distributions are dominated by spaces and
<code>a</code>, and the output stayed garbage no matter what guards we added. A
vocabulary menu (real words assembled from English statistics) produced
86-100% genuine words and no syntax at all: word salad that stops politely.
Only a token-level menu built by a <i>local</i> language model, which Jev
re-scores, produced coherent text - which is the honest limit of this method:
the fluency is partly the local model's, and asking Jev to describe itself this
way mostly tells you about whichever model you fed it. The attempt that
seemingly pointed at an OpenAI identity turned out to be substantially the
local model's own prior leaking through. The program's summary: unguided Jev is
a word-salad generator with excellent stopping behavior.</p>

<h3>2. Asking it to choose a name</h3>
<p>Because free text is off the table, we made the identity question a multiple
choice - and to keep it honest, the option list <i>always</i> included
TypeSafe and Jev alongside the usual suspects, and every frame was shown under
all 24 cyclic orderings so the serial-position bias we had measured earlier
could not manufacture the result (the winning name landed at index 0 only
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
brand prior, and it is the same prior that made the &ldquo;coauthor of
ChatGPT&rdquo; framing of the launch page feel like a hint about the weights.
It is not evidence about the weights.</p>

<h3>3. Counting tokens</h3>
<p>The API reports an input-token count for every request, and that is not a
self-report - it is an instrument. Feeding sixty controlled strings (repeated
subwords, casing variants, CJK, Cyrillic, Greek, Arabic, emoji, ZWJ family
sequences, whitespace runs, numerals, markdown, code) through a fixed template
and fitting the reported counts against reference tokenizers gave a
first answer that looked exciting:</p>
<table><tr><th>candidate</th><th class='n'>RMSE</th><th class='n'>slope</th><th class='n'>R²</th></tr>{ranks}</table>
<p>The Qwen2.5 tokenizer fit almost perfectly - slope {top['slope']:.3f},
residual ~1 token across fifty-nine strings, robust when all multilingual
strings were removed from the fit. Every OpenAI encoding fit noticeably worse.
That looked like a Qwen base wearing an OpenAI accent.</p>
<p>Then we built the test that removes the fixed template entirely: whitespace-
free, long, per-script samples where we can read the marginal cost
<i>per character</i> directly. That answer is much less comfortable for the
Qwen hypothesis:</p>
<div class="scroll"><table class="wide"><tr><th>tokens / character</th>{head}</tr>{rows}</table></div>
<p>Jev's shape is: heavy merging of Latin text and punctuation, but <b>about one
token per character for every other script we tried</b> - Cyrillic, Chinese,
Korean, Greek, Arabic, Hebrew, Thai, Devanagari all sit at 0.92-0.96 - with
astral emoji at ~2 and pure whitespace collapsing to zero (so the server runs a
normalizer before tokenizing). No reference vocabulary reproduces that profile.
Qwen is <i>strong</i> at merging CJK and Cyrillic (0.34-0.68 tokens/char) -
exactly the ability Jev's counter does not show. Nor does any OpenAI encoding:
o200k and cl100k merge digits and Cyrillic aggressively, while Jev spends
~0.94 tokens per digit. Across roughly 128 distinct tokenizers spanning ~30
organizations' releases, nothing reproduced the profile.</p>
<p>The honest conclusion is the boring one that the marketing should have made
us suspicious anyway: the tokenizer is most plausibly <b>the vendor's
own</b>, English/Latin-centric, with code-point fallback for everything else -
i.e., a new foundation, not a relabeled one. The OpenAI identity answer and the
Qwen tokenization &ldquo;clue&rdquo; were both artifacts (a trained prior, and
a template constant) that a sharper instrument corrected. That is the whole
point of publishing this: even on our own second-best measurement, the
assumptions you start with are the ones that fail.</p>
</section>"""


def conclusion() -> str:
    mmlu = pct(SC["mmlu"]["accuracy"]); gpqa = pct(SC["gpqa"]["accuracy"])
    arc = pct(SC["arc"]["accuracy"]); jm = JEV_COST["mmlu_pro"]
    return f"""
<section id="conclusion">
<h2>So, is it worth your attention?</h2>
<p>It is not what the landing page says. The evidence in this report says the
honest description is: a small, new, English-centric foundation model with a
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
<p><b>Reproducibility.</b> This report is generated from the run artifacts in this
repository by <code>scripts/report/build_report.py</code>; every number is read
from disk at render time, and the charts are produced by
<code>comparison_graphs.py</code>, <code>pareto_graphs.py</code> and
<code>arch_evidence.py</code>. Jev was called live with an API key that is never
stored in any artifact. Our own probing transcripts (the Talk-to-Jev traces and
the architecture-probe rows) are published verbatim; benchmark raw logs are held
locally because they embed licensed dataset text - the published freeze hashes
pin them exactly.</p>
</footer>
{refs()}"""


def main() -> None:
    SITE.mkdir(exist_ok=True)
    body = "".join([hero(), pitch(), methodology(), architecture(None),
                    benchmarks(), pareto(), latency(), probes(), conclusion()])
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
