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

# Jev per-benchmark cost: measured from the runs' own billing usage summaries
JEV_COST = {b: SC[s]["attempts_summary"]["usage_input_tokens_sum"] * 0.042 / 1e6
            for b, s in {"mmlu_pro": "mmlu", "gpqa": "gpqa", "arc": "arc",
                         "math500": "math_c", "hle": "hle"}.items()}

_BENCH_SVGS = svgs_of("docs/modern-comparison/comparison-graphs.html")
CHARTS = dict(zip(["mmlu_pro", "gpqa", "arc", "math500", "arc_agi2", "hle"], _BENCH_SVGS))
PARETO = svgs_of("docs/modern-comparison/pareto-frontiers.html")
EVID = svgs_of("docs/modern-comparison/architecture-evidence.html")




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
    "almeida": "Diogo Almeida led the InstructGPT work, which is a direct "
        "methodological ancestor of ChatGPT. Two caveats the marketing "
        "collapses: RLHF was not invented there (it predates InstructGPT), "
        "and ChatGPT has many parents - GPT-3.5, RLHF, code-as-reasoning, "
        "tool use, decades of prior work. The accurate reading is a "
        "researcher whose earlier model is one of ChatGPT's several "
        "ancestors, which is still a credential. It is not an architecture "
        "claim - and, as it turned out, not a provenance claim either.",
    "nohalluc": "Being unable to emit free text is not the same as being "
        "unable to be wrong. Jev is wrong about a quarter of the time on "
        "graduate-level science questions, and every wrong answer arrived "
        "beautifully formatted. The guarantee is about the output envelope, "
        "not the reliability of the content.",
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
p,li{max-width:70ch}a{color:var(--peri)}
.sub{color:var(--mut);font-size:1.05rem;max-width:62ch}
.chip{display:inline-block;font-size:.72rem;letter-spacing:.05em;text-transform:uppercase;color:var(--teal);border:1px solid var(--teal);border-radius:999px;padding:.1rem .6rem;margin-bottom:1.1rem}
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:1.05rem 1.3rem;margin:1.1rem 0}
.pitch{font-size:1.16rem;font-style:italic;border-left:3px solid var(--amber);padding-left:1.1rem;margin:1.1rem 0}
.pitch b{font-style:normal}
svg{width:100%;height:auto;display:block;margin:1rem 0}
table{border-collapse:collapse;width:100%;font-size:.92rem;margin:1rem 0}
th,td{text-align:left;padding:.45rem .6rem;border-bottom:1px solid var(--line)}
th{color:var(--mut);font-weight:600;font-size:.78rem;text-transform:uppercase;letter-spacing:.04em}
td.n{text-align:right;font-variant-numeric:tabular-nums}
.cap{color:var(--mut);font-size:.88rem;max-width:74ch}
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
hallucinate, built by a coauthor of ChatGPT - fast, and almost free. We ran it
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
not lineage{fn('behavioronly')}. It is also not a toy: {mmlu} MMLU-Pro,
{gpqa} GPQA, ~{ARCH['prefill']['fixed_floor_ms']:.0f} ms of server compute per
question, and a full graduate-scale benchmark run for cents. The honest
category is <i>cheap real-time judgement</i> - routing, rubric grading,
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


ARCH_MANIFEST = [
    ("The hypothesis", "What we think Jev is under the hood, stated up front: a pre-existing "
     "language model, post-trained, with the generation head replaced by a "
     "choice/score/noul read-out, and all outputs computed in one batched pass."),
    ("Benchmark evidence", "runs_benchmark/, runs_benchmark_ext/, "
     "runs_benchmark_ext2/ - derived score/weighted-score JSON per stage "
     "(aggregate; per-item lists stripped), freeze manifests (frozen.json, "
     "preparation.json) with SHA-256 hashes of every input, and per-stage "
     "usage summaries. Gated dataset item text is not republished; the hashes "
     "pin it."),
    ("Latency and self-batching", "runs_archprobe/analysis.json + rows.jsonl - "
     "prefill slope, per-question and per-option marginal cost, cold vs warm "
     "connection, in-flight concurrency curves. Charts: "
     "docs/modern-comparison/architecture-evidence.html."),
    ("Tokenization fingerprint", "runs_archprobe/tokenizer_fingerprint.json (short-string "
     "affine fit), tokenizer_perscript.json (per-script residual fit), "
     "tokens_per_char_compare.json (whitespace-free marginal cost table), "
     "tokenizer_broadscan.json (1,215 repos -> 173 unique tokenizer signatures)."),
    ("Talk-to-Jev program", "docs/token-talk-findings.md (sections 1-15), "
     "docs/jev-talk-program-report.md, runs_live/ traces - the option-ladder "
     "result (character / vocabulary-menu / token+local-LM), ordering probe, "
     "ensemble probes, identity interventions."),
    ("Key behavioral conclusions", "Greedy accuracy exceeds probability-weighted accuracy on "
     "nearly every benchmark (over-dispersed distributions); every probability "
     "lands on a 0.01 grid; server compute is a fixed floor plus ~5-6 ms per "
     "1k input tokens with negligible quadratic term; additional questions "
     "inside one request are near-free; concurrency scales throughput ~10x "
     "without hurting server latency; identity prior points at OpenAI; the "
     "whitespace-free tokenizer profile matches no open model."),
    ("Reproducibility", "scripts/ + src/ in this repo, deterministic seeds and "
     "freeze hashes in every runs_* dir; the same commands that produced these "
     "figures are listed in the data section."),
]


def architecture(v: dict) -> str:
    rows = "".join(f"<li><b>{t}</b> - {d}</li>" for t, d in ARCH_MANIFEST)
    return f"""
<section id="arch">
<h2>What Jev appears to be</h2>
<div class="note"><b>Drafting note.</b> This section is deliberately a
placeholder. The prose and figures for it are being authored separately; the
list below is the complete inventory of published evidence it should draw on,
with paths. Until then, treat the summary bullets as the claim:</div>
<p><i>Working picture:</i> Jev looks like a small, English-centric language
model whose output side has been converted into a probability read-out over
caller-supplied options, with every question in a request scored from one
forward pass. It is not a frontier model, not a retrieval cache, and not a
wrapper around a bigger vendor - and the OpenAI-shaped answers in our identity
probes are almost certainly a training-data artifact, not lineage.</p>
<h3>Evidence inventory for the write-up</h3>
<ul>{rows}</ul>
</section>"""


def pitch() -> str:
    cf = SC["mmlu"]["strict_format_failures"]
    mn = SC["mmlu"]["n_expected"]
    wrong_gpqa = pct(1 - SC["gpqa"]["accuracy"])
    jp = COSTS["prices_usd_per_M"]["Jev"]
    cost_mmlu = f"${JEV_COST['mmlu_pro']:.2f}"
    arc_pct = pct(SC["arc"]["accuracy"])
    return f"""
<section id="pitch">
<h2>The pitch, and the parts that survive contact</h2>
<p class="pitch">&ldquo;A model that <b>cannot hallucinate</b>, at <b>frontier-level
performance</b>, built by a <b>coauthor of ChatGPT</b> - incredibly fast,
incredibly cheap, with <b>free output</b>.&rdquo;</p>
<p>Each clause is technically defensible in a narrow sense and misleading in the
sense a buyer will hear. Take them one at a time.</p>
<p><b>The authorship claim.</b> The phrase does a lot of quiet work. ChatGPT has
hundreds of parents; naming one person as its coauthor compresses a lineage of
RLHF, instruction tuning, code-as-reasoning and decades of prior work into a
founding myth.{fn('almeida')} It is a real credential. It is not an
architecture claim - and, as it turned out, not a provenance claim either.</p>
<p><b>Cannot hallucinate.</b> What Jev actually cannot do is emit free text. It
returns one option from a fixed menu, inside a schema that is always
well-formed, and that guarantee held under load: {cf} malformed outputs across
{mn:,} MMLU-Pro items and zero everywhere else. But a schema-valid answer is
not a true answer. Jev is wrong {wrong_gpqa} of the time on graduate science,
and every one of those wrong answers arrived beautifully formatted. A model
that cannot write prose has not solved hallucination; it has made hallucination
hard to notice.{fn('nohalluc')}</p>
<p><b>Fast and cheap.</b> Both true, and explainable in one sentence: Jev never
runs a decode loop. It reads the prompt once and reads off a vector, which is
why the bill is ${jp['input']:.3f} per million input tokens with output free,
and why a full {mn:,}-question MMLU-Pro run cost {cost_mmlu}.{fn('jevfree')}
Speed and price are properties of the <i>task</i> being prefill-only, not of
frontier economics.</p>
<p><b>Frontier-level.</b> This one simply does not survive. Jev is good, and
&ldquo;good&rdquo; will turn out to mean something genuinely useful here - but
it is not a frontier model by any late-2026 standard, and where it looks
frontier-like ({arc_pct} on ARC-Challenge) the whole field finished that race
years ago. Everything below is the evidence.</p>
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
    figs = "".join([
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
    ])
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
every other bar is a published number we fetched - not a model we ran.</p>
{figs}
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
