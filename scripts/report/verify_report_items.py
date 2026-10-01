import re, sys
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
must(re.search(r'\((none|low|medium|high|xhigh|max)\)&lt;/b&gt;', cmp_h)
     and re.search(r'\((none|low|medium|high|xhigh|max)\)&lt;/b&gt;', par_h),
     '1b tooltips render (tier parenthetical in both pages)')
must('GLM-5.3' in cmp_h, '2 glm on charts (inversion check is data-side)')
must(re.search(r'\((none|low|medium|high|xhigh|max)\)&lt;/b&gt;', par_h), '3 tier parenthetical in tooltips')
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
must('An exposed probability distribution' in h and 'leading theory, not a measured fact' in h
     and '<td>Confident' in h and '<td>Plausible' in h, '14 output side: measured vs leading theory')
must('(see report)' not in dia, '15 no see-report')
must('~4 to 9B params' in dia, '16 diagram range not 4-9B')
must('Each pays only for its own tokens' in dia, '17 prefill phrasing')
must('stays possible, not favored' not in h, '17b no not-favored')
must('Reproducibility' not in h, '17c footer dropped')
must('beaten to the punch' in h, '17d mouth sentence')
must('move the frontier' in h, '17e frontier sentence')
must('war- rants' not in h and 'warrants' in h, '17f warrants')
must('&mdash;' not in h and '\u2014' not in h, '17g no emdash')
must('we estimate the peak' in h, '17h peak wording')
must('250-500 of a 2020' not in h, '17i a100 clause gone')
must('capability/latency ratio' not in h, '17j caplat gone')
must('0.01 floor is standard' in h, '17k floor wording')
must('none identifiable, not excluded' in dia, '18 not excluded')
must('Questions packed into one request' in (ROOT / 'docs/modern-comparison/architecture-evidence.html').read_text(), '19 axis capitalization')
must('flat in the header' in h and 'does not identify' in h, '20 concurrency: flatness not mechanism')
must('added tokens' in h and 'not isolated decision-head timings' in h, '21 added tokens / increments not decision')
must('In short, what moves the measured upstream service time' in h, '22 one-pass TLDR bullets')
must('one quantum (e.g., 0.01)' in h, '23 quantum gloss')
must('(p_max &minus; 1/K) / (1 &minus; 1/K)' in h, '24 formula')
must('within one quantum, and' in h, '25 oxford comma')
must('roughly recoverable' in h, '26 roughly recoverable')
must(', $0.24' not in h, '27 no battery cost')
must('The battery re-confirmed this live' in h and '</p>\n<p>The battery re-confirmed' in h, '28 battery new paragraph')
must('(2-255)' in h and '(2-64 tokens)' in h, '29 parenthesized ranges')
must('To test whether dummy filler options dilute' in h, '30 filler framing')
must('Apart from the ordering effects' in h, '31 ordering caveat')
must('Hangul, Kana' in h, '32 Kana capitalization')
must('Assuming typical late-2026 hardware' in h, '33 tflops sentence')
must('800-9000 effective' in h, '34 corrected quantized range')
must('similar-performing models' in h, '35 neighbors')
must('Distillation is orthogonal' not in h, '36 distillation removed')
must('What could be tested from the outside' not in h, '37 meta para removed')
must('reports an upstream-service timing header' in h
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
must('140 base items' in h and 'equivalence' in h, '77 rotation clustering stated')
must('0 of 120' in h and 'oracle' in h, '78 arc task criterion + oracle protocol')
must('82.7%' in h, '79 mmlu all-requested headline')
must('84.0%' in h and 'matched subset' in h, '80 matched subset shown distinctly')
must('42 complete cells' in h and '17 partial cells' in h, '81 coverage notice')
must('proxy-reported upstream service time' in cmp_h.lower()
     or 'proxy-reported' in (ROOT / 'docs/modern-comparison/architecture-evidence.html').read_text().lower(),
     '82 envoy label in evidence pages')
must('tabindex' in cmp_h, '83 keyboard-focusable chart tooltips')
must('independent re-scoring' in h, '84 rebuild-vs-rescore distinction')
must('not a single undifferentiated' in par_h, '85 no true-pareto claim')
# --- P0-1 fix: charts consume the authoritative v4r1 active summary ---
must('Mistral Small 3.2' in cmp_h and 'score 83.1%' in cmp_h,
     '86 current mistral math tie charted (217/261)')
must('score 2.7%' not in cmp_h and 'score 2.7%' not in par_h,
     '87 no legacy 2.7% parser-artifact bar')
import re as _re
_tips = [_re.findall(r'data-tip="([^"]*)"', t) for t in (cmp_h, par_h)]
_mt = [x for lst in _tips for x in lst
       if 'measured cost $' in x and '/question' in x]
must(len(_mt) == 42, f'88 exactly 42 matched rows charted (got {len(_mt)})')
must('accuracy_recovered' not in (ROOT / 'scripts/report/chartkit.py').read_text()
     or '.get("accuracy_recovered"' not in (ROOT / 'scripts/report/chartkit.py').read_text(),
     '89 no legacy accuracy_recovered preference')
# --- parent final prose review: semantic contradictions on generated text ---
must('server compute' not in h, '90 no server-compute phrasing')
must('every measured response' not in h and 'schema-valid in' not in h,
     '91 no universal output-validity promise')
_srow = h[h.find('<tr><td>Serving</td>'):h.find('<tr><td>Service-time profile</td>')]
must('Plausible' in _srow and 'leading' in _srow and 'Confident' not in _srow,
     '92 serving row is Plausible leading theory, not confident fact')
must('19 contract-invalid' in h and '18 MMLU-Pro, 1 rotation' in h,
     '93 explicit 19 contract-invalid / ~16k counts')
must('pricing decision, not proof' in h, '94 output-free is billing policy')
must('disfavor a conventional frontier-API' in h
     or 'disfavors a conventional frontier-API' in h,
     '95 wrapper disfavored, not categorically excluded')
must('the exact rule is' in h and 'unidentified' in h,
     '96 pre-display decision rule left unidentified')
must('not an isolated' in h and 'not proof of one shared' in h,
     '97 increments consistent with input scaling, no head bound/shared-pass proof')
print('FAILS:', len(fails))
