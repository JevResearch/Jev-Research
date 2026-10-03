import json, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
h = open(str(ROOT / 'report/index.html')).read()
cmp_h = open(str(ROOT / 'docs/modern-comparison/comparison-graphs.html')).read()
par_h = open(str(ROOT / 'docs/modern-comparison/pareto-frontiers.html')).read()
dia = open(str(ROOT / 'docs/modern-comparison/architecture-diagram.html')).read()
fails = []
def must(cond, name):
    print(('OK   ' if cond else 'FAIL ') + name)
    if not cond: fails.append(name)

# --- user review items (this pass) ---
must('mdash; our matched run' not in cmp_h and 'mdash; our matched run' not in par_h, '1 no matched-run tooltip spam')
_TIER_TIP_RE = r'\((none|low|medium|high|xhigh|max|thinking unspecified)\)&lt;/b&gt;'
must(re.search(_TIER_TIP_RE, cmp_h) and re.search(_TIER_TIP_RE, par_h),
     '1b tooltips render (tier parenthetical in both pages)')
must('GLM-5.3' in cmp_h, '2 glm on charts (inversion check is data-side)')
must(re.search(_TIER_TIP_RE, par_h), '3 tier parenthetical in tooltips')
must('cheapest = #1' not in par_h, '1c no cheapest-rank noise')
must('date basis' not in cmp_h and 'date basis' not in par_h, '4 no date-basis suffix')
must('color = release era' not in cmp_h and 'color = release era' not in par_h, '5 no era label')
must('&dagger; our' in cmp_h or 'our matched' in cmp_h, '6 legend defines dagger')
must('figure{width:min(96vw,1120px)' in h, '7 figures widened')
must('figcaption{color:var(--mut);font-size:.8rem;font-style:italic' in h, '8 captions italic+smaller')
must('is nonetheless genuinely useful' in h, '9 nonetheless')
tldr = h[h.find('class="card tldr"'):h.find('herorot')]
must('<sup' not in tldr, '10 TL;DR has no footnote')
must('&ldquo;output&rdquo; token count' in h, '11 jevfree air quotes')
must('InstructGPT, worked on learned' in h, '12 also->worked')
must('anything internal to its design' in h, '13 serving internal')
must('An exposed probability distribution' in h
     and 'Confident (exposed' in h and '<td>Plausible' in h
     and 'leading theory (trained replacement)' not in h
     and 'leading theory, not a measured fact' not in h,
     '14 output side: confident distribution, no leading-theory hedge')
must('(see report)' not in dia, '15 no see-report')
must('~4 to 9B params' in dia, '16 diagram range not 4-9B')
must('Each pays only for its own tokens' in dia, '17 prefill phrasing')
must('stays possible, not favored' not in h, '17b no not-favored')
must('Reproducibility' not in h, '17c footer dropped')
must('beaten to the punch' in h, '17d mouth sentence')
must('move the frontier' in h, '17e frontier sentence')
must('war- rants' not in h and 'warrants' in h, '17f warrants')
_approved_ancestry_dash = 'possibility—not a supported identification or a default coin flip.'
_dash_checked = h.replace(_approved_ancestry_dash, '')
must('&mdash;' not in _dash_checked and '\u2014' not in _dash_checked
     and h.count(_approved_ancestry_dash) == 1,
     '17g no unapproved emdash (author-approved ancestry wording excepted)')
must('we estimate the peak' in h, '17h peak wording')
must('250-500 of a 2020' not in h, '17i a100 clause gone')
must('capability/latency ratio' not in h, '17j caplat gone')
must('0.01 floor is standard' in h, '17k floor wording')
must('none identifiable, not excluded' in dia, '18 not excluded')
must('Questions packed into one request' in (ROOT / 'docs/modern-comparison/architecture-evidence.html').read_text(), '19 axis capitalization')
must('flat in the header' in h and 'not distinguished' in h,
     '20 concurrency: flatness kept, mechanism claim removed')
must('added tokens' in h and 'not isolated decision-head timings' not in h
     and 'not proof of one shared' not in h,
     '21 added tokens kept, decision-head hedges removed')
must('In short, what moves the measured upstream service time' in h, '22 one-pass TLDR bullets')
must('one quantum (e.g., 0.01)' in h, '23 quantum gloss')
must('(p_max &minus; 1/K) / (1 &minus; 1/K)' in h, '24 formula')
must('within one quantum, and' in h, '25 oxford comma')
must('roughly recoverable' in h, '26 roughly recoverable')
must(', $0.24' not in h, '27 no battery cost')
must('The battery re-confirmed this live' in h and '</p>\n<p>The battery re-confirmed' in h, '28 battery new paragraph')
must('(2-255)' in h and '(2-64 tokens)' in h, '29 parenthesized ranges')
must('To test whether dummy filler options dilute' in h, '30 filler framing')
must('Those filler options did not distract Jev from easy factual answers.' in re.sub(r'\s+', ' ', h)
     and 'Our independent tests below show that simply reordering the real choices can change the answer.' in re.sub(r'\s+', ' ', h),
     '31 padding and independent ordering tests distinguished')
must('Hangul, Kana' in h, '32 Kana capitalization')
must('Assuming typical late-2026 hardware' in h, '33 tflops sentence')
must('800-9000 effective' in h, '34 corrected quantized range')
must('similar-performing models' in h, '35 neighbors')
must('Distillation is orthogonal' not in h, '36 distillation removed')
must('What could be tested from the outside' not in h, '37 meta para removed')
must(re.search(r'reported an upstream-service\s+timing header', h)  # phrase updated: author copyedit "responses reported" (was "reports")
     and 'proxy-reported upstream service time' in h
     and 'strips out network and queueing' not in h, '38 timing phrasing (envoy-labeled)')
must('raw shape' not in h, '39 raw shape removed')
must('because it is one on the server' not in h, '40 pie caption')
must('routed away from teal' not in h, '41 teal clause removed')
must('white-ringed diamonds' not in h, '42 diamonds sentence removed')
must('Hover any bar' not in h, '43 hover sentence removed')
must('Same encoding and hover' not in h, '44 gpqa caption trimmed')
must('Revived because' not in h, '45 revived-because removed')
must('Two benchmarks that we ran posed challenges' in h, '46 math caption')
must('Two table rows are deliberately' not in h, '47 exclusion para removed')
must('Measured spend' not in h, '48 spend sentence removed')
must('either our own run (mostly of small models)' in h, '49 pareto external rewrite')
must('from a word table' not in h and 'Markov model' in h, '50 markov')
must('&ldquo;Geeee&rdquo;' in h, '51 quoted surface forms')
must('nine-post sample' in h, '52 jevbot sample')
must('coauthor of' not in h, '53 brand clause removed')
must('It answers; it does not write' not in h, '54 keynote bullet removed')
must('drowned in' in h and 'rounding noise' in h, '55 floor wording')
must('It knows nothing about who it is' in h, '56 identity bullet')
must('Believe its probabilities' not in h, '57 biography line removed')
must('ranker, not a well-calibrated distribution' in h, '58 ranker bullet')
must('essentially nothing beyond' in h or 'essentially free' in h, '59 options marginal-cost bullet')
must('non-Latin scripts' in h, '60 non-latin bullet')
must('It does not see whitespace' in h, '61 whitespace bullet')
must('was created from the run artifacts' not in h, '62 footer dropped (per latest review)')
must('never stored in any artifact' not in h, '63 key line removed')
must('freeze hashes' not in h, '64 freeze hashes removed')
must('Cost per question (USD, log scale)' in par_h, '65 axis capitalization')
must('class="fatten"' in par_h, '66 fat hover targets')
must('Option-rotation audit' not in h, '67 rotation gone from report')
must('Jev (greedy)' in par_h, '68 jev on pareto')
# --- integration/corrections pass (review fixes); each can fail ---
must('the queue-free' not in h and 'queue-free server time' not in h
     and 'queue-free x-envoy' not in h, '69 no queue-free header claim')
must('probabilities understate its accuracy' not in h
     and 'That gap is the' not in h, '70 no over-dispersion inference')
must('leaves under' not in h and '0.11 ms/question' not in h
     and '0.11 ms/question' not in dia, '71 no sub-0.11ms head bound')
must('the floor is ~95%' not in h, '72 no 95%-overhead pie claim')
must('illustrative' in (ROOT / 'docs/modern-comparison/architecture-evidence.html').read_text().lower(),
     '73 pie labeled illustrative')
must('has undergone no SFT' not in h and 'never drift schema' not in h,
     '74 no SFT / no absolute schema claim')
must('a fixed cutoff' not in h and 'the same answers must be read off' not in h,
     '75 horizon sampled / no same-answers claim')
must('all-requested' in h and 'bench-calib' in h, '76 all-requested + calibration table')
must('419 scored observations' in h and 'equivalence' in h
     and '140 base items x 3 variants' not in h,
     '77 rotation clustering kept in note, parenthetical cut from main text')
must('0 of 120' in h and 'oracle' in h, '78 arc task criterion + oracle protocol')
must('82.7%' in h, '79 mmlu all-requested headline')
must('84.0%' in h and 'matched subset' in h, '80 matched subset shown distinctly')
_pub = json.loads((ROOT / 'data_report/baselines/v4r1/public_summary.json').read_text())
_t = _pub['totals']
must(f"{_t['n_complete_cells']} of {_t['n_cells']}" in h
     and f"{_t['n_requested']:,}" in h
     and '42 complete cells' not in h and '17 partial cells' not in h,
     '81 coverage notice derives current cell counts (no stale 42/17)')
must('proxy-reported upstream service time' in cmp_h.lower()
     or 'proxy-reported' in (ROOT / 'docs/modern-comparison/architecture-evidence.html').read_text().lower(),
     '82 envoy label in evidence pages')
must('tabindex' in cmp_h, '83 keyboard-focusable chart tooltips')
must('independent re-scoring' in h, '84 rebuild-vs-rescore distinction')
must('not a single undifferentiated' not in par_h
     and 'not a single undifferentiated' not in h, '85 true-pareto hedge removed')
# --- P0-1 fix: charts consume the authoritative v4r1 active summary ---
must('Mistral Small 3.2' in cmp_h and 'score 83.1%' in cmp_h,
     '86 current mistral math tie charted (217/261)')
must('score 2.7%' not in cmp_h and 'score 2.7%' not in par_h,
     '87 no legacy 2.7% parser-artifact bar')
import re as _re
_tips = [_re.findall(r'data-tip="([^"]*)"', t) for t in (cmp_h, par_h)]
_mt = [x for lst in _tips for x in lst
       if 'measured cost $' in x and '/question' in x]
must(len(_mt) == _t['n_complete_cells'],
     f"88 exactly {_t['n_complete_cells']} matched rows charted (got {len(_mt)})")
must('accuracy_recovered' not in (ROOT / 'scripts/report/chartkit.py').read_text()
     or '.get("accuracy_recovered"' not in (ROOT / 'scripts/report/chartkit.py').read_text(),
     '89 no legacy accuracy_recovered preference')
# --- parent final prose review: semantic contradictions on generated text ---
must('server compute' not in h, '90 no server-compute phrasing')
must('every measured response' not in h and 'schema-valid in' not in h,
     '91 no universal output-validity promise')
_srow = h[h.find('<tr><td>Serving</td>'):h.find('<tr><td>Service-time profile</td>')]
must('Plausible' in _srow and 'Confident' not in _srow
     and 'leading theory' not in _srow,
     '92 serving row is Plausible, not a confident fact')
_hn = re.sub(r'\s+', ' ', h)
must('19 contract-invalid' in _hn and '18 MMLU-Pro, 1 rotation' in _hn
     and 'not necessarily malformed JSON' not in _hn,
     '93 measured 19 contract-invalid / ~16k counts kept, hedge removed')
must('pricing decision, not proof' not in h
     and 'every output token is priced at zero' in h,
     '94 output-free billing caveat cut; zero price fact kept in the note')
must('disfavor a conventional frontier-API' in h
     or 'disfavors a conventional frontier-API' in h,
     '95 wrapper disfavored, not categorically excluded')
must('the exact rule is unidentified' not in h,
     '96 unidentified-rule caveat removed')
must('input-token scaling' in h and 'not proof of one shared' not in h,
     '97 increments stay input-token scaling; shared-pass hedge removed')

# --- author copyedit pass: structural / chart-encoding checks ---
must(h.count('<h3 id="arch-millis">What the milliseconds say</h3>') == 1
     and 'id="latency"' in h and 'id="arch-onepass"' in h
     and '<section id="latency">' not in h,
     '98 one merged latency subsection with #latency/#arch-onepass aliases')
must('href="#arch-millis">What the milliseconds say</a>' in h
     and 'href="#latency"' not in h,
     '99 TOC links the merged subsection, no stale #latency TOC entry')
must('thinking unspecified' in cmp_h and 'thinking unspecified' in par_h
     and 'shape = thinking' in cmp_h and 'shape = thinking' in par_h,
     '100 shared legend maps solid dot (unspecified) vs ring (none)')
must('.fatten{fill:transparent;stroke:none}' in par_h
     and 'g.ptrow .fatten' not in par_h
     and 'stroke="#000' not in par_h,
     '101 pareto hit-targets fully transparent, no black circles')
_arc = re.findall(r'<svg[^>]*aria-label="ARC-AGI-2"[^>]*>', cmp_h)
must(len(_arc) == 1 and 'task level' not in cmp_h
     and 'per-cell diagnostic - adapter' not in cmp_h,
     '102 ARC-AGI-2 is ONE chart, no split/pedantic titles')
_arcsvg = re.search(r'<svg[^>]*aria-label="ARC-AGI-2"[^>]*>.*?</svg>', cmp_h, re.S).group(0)
_pat = re.findall(r'<pattern id="(xh-[^"]+)"', _arcsvg)
must(len(_pat) == 2 and all(f'url(#{p})' in _arcsvg for p in _pat)
     and _arcsvg.count('font-style="italic"') == 4
     and 'Jev per-cell choice' in _arcsvg and 'Jev per-cell score' in _arcsvg
     and 'Jev exact-grid' in _arcsvg,
     '103 ARC per-cell rows crosshatched + italic on the combined chart')
_arcnote = h[h.find('<li id="ref-arcproto">'):h.find('<li id="ref-arcproto">') + 2000]
must('duplicate request-payload' not in cmp_h
     and h.count('duplicate request-payload') == _arcnote.count('duplicate request-payload'),
     '104 payload-defect detail only in footnote 18, never in chart captions')
must('Qwen3.8 Max (low)&lt;/b&gt;' in cmp_h, '105 matched Qwen Max labels LOW (v2 effort-low wire)')
must('Qwen3.8 Max (thinking unspecified)&lt;/b&gt;' in cmp_h,
     '106 vals Qwen Max (null config) is unspecified, never none')
must('cheap OpenRouter' not in h and 'cheap OpenRouter' not in cmp_h
     and 'cheap OpenRouter' not in par_h,
     '107 no cheap-OpenRouter attribution phrasing')
must('figcaption{max-width:70%}' in h, '108 captions narrowed to 70% on desktop')
_mmlu_svg = re.search(r'<svg[^>]*aria-label="MMLU-Pro[^"]*"[^>]*>.*?</svg>', cmp_h, re.S)
_gpqa_svg = re.search(r'<svg[^>]*aria-label="GPQA[^"]*"[^>]*>.*?</svg>', cmp_h, re.S)
must(_gpqa_svg and '(thinking unspecified)&lt;/b&gt;&lt;br&gt;score 93.7%' in _gpqa_svg.group(0)
     and '(thinking unspecified)&lt;/b&gt;&lt;br&gt;score 88.6%' not in _gpqa_svg.group(0),
     '109 Qwen Max 93.7 renders on the GPQA chart, not as an MMLU score')
print('FAILS:', len(fails))
sys.exit(1 if fails else 0)
