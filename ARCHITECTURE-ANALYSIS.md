# What underlies jev-1.13.0 — a best-guess reconstruction

**Status.** Working analysis, updated 2026-09-26 (post-probe). Companion to `report/index.html`
(section "What Jev appears to be", now illustrated by
`docs/modern-comparison/architecture-diagram.html`),
`docs/modern-comparison/ARCHITECTURE-PROBES.md` (the probe battery), and
`docs/ARCHITECTURE-ANALYSIS-PROMPT.md` (the task this answers). Secondary
audits cited below are re-derived from published artifacts by
`scripts/report/arch_audits.py` → `data_report/arch_audits.json` and
`scripts/report/lattice_forensics.py` → `data_report/lattice_forensics.json`
and `scripts/report/size_estimate.py` → `data_report/size_estimate.json`
(offline; no network, no live calls). The §10 follow-up battery (P1–P5)
**ran live on 2026-09-25/26**: 3,331 calls, 5,805,477 reported input tokens,
$0.2438 at the measured rate, 1 transient HTTP error (the intentional
lone-surrogate probe; see §2C), artifacts in `runs_archprobe/probe2/`
(rows, per-family analysis, billing). Its results are integrated below and
marked **(P-battery)**.

**Ground rules honored throughout.** (1) Behavioral evidence only — every
observation is an API-visible signal (answers, probability vectors, usage token
counts, server-compute timing headers, billing); nothing here implies weights,
gradients, or serving internals were accessed. (2) Every claim cites repo
paths; numbers were re-verified against the JSON in this pass, not copied from
summaries. (3) The 0.01 quantization floor is treated as a hard noise floor:
no argument rests on a single probability difference < 0.01, and §11 flags
every place we found one — including in prior documents and in our own drafts.
(4) Latency slopes are used for *structure* only (one pass vs decode loops),
never for parameter counts — the server batches our requests with other
tenants', so marginal milliseconds are a scheduling artifact as much as a
compute artifact. (5) Jev's identity self-reports are treated as learned text,
never provenance. (6) Single-call distribution comparisons are read against
the measured nondeterminism band (repeat TVD 0.034–0.122,
`runs_live/token_talk_orderprobe.json`). (7) External benchmark comparisons
are positioning under mismatched protocols
(`docs/modern-comparison/canonical/comparable-scores.json` labels each row's
protocol). (8) Talk-to-Jev token-menu results are treated as collaboration
with the local scorer (Mellum2), per `docs/token-talk-findings.md` §8, §9,
§11, §14–15; only character- and vocabulary-menu degeneracies are attributed
to Jev alone.

Confidence tags: **[confident]** = directly measured, adequate n, robust to
the noise floor. **[plausible]** = best explanation of measurements, rivals
not excluded. **[speculative]** = reasoned guess; the evidence
under-determines it.

---

## 1. The architecture card

One row per component. "Falsified by" names the API-visible observation that
would force us to abandon or materially rewrite the row.

| Component | Best guess | Tag | Key evidence (paths) | Falsified by |
|---|---|---|---|---|
| Output mechanism | A trained probability **read-out over caller-supplied options**, replacing the token-generation head — not sampled text parsed into numbers | confident | 0.01 grid on all 704,277 published probability values (7,887 vectors, K=2–255), zero off-grid; displayed sums bounded one-sided (0.99/1.00, never >1.00 — confirmed live at every K≤12 and at 255, **(P4)**); 9 choice-vs-table argmax mismatches, all at exactly one quantum; `confidence` = chance-corrected p_max; options scored essentially absolutely — no mass dilution K=2→255 **(P3)** (§2A/§2D; `data_report/arch_audits.json`, `data_report/lattice_forensics.json`, `runs_archprobe/probe2/`) | Off-grid probabilities at scale; a displayed sum above 1.00; free-text output; a confidence field not derivable from (p_max, K); systematic p_gold dilution under inert padding |
| Serving shape | **One forward pass per request** ("self-batching"): every question and option scored from a shared pass over the state; continuous batching across tenants; **no decode loop** | confident | +0.44 ms/question and +0.10 ms/option marginal compute ≈ the prefill cost of the tokens each adds (read-out residual ≤0.11 ms/question, zero per option); upstream flat 74–86 ms at concurrency 1→32; output tokens = serialized response JSON (`runs_archprobe/analysis.json`, `rows.jsonl`; §2B) | Per-question compute growing far above its added-token prefill cost; upstream latency rising with in-flight count; per-output-token timing |
| Compute profile | ~73 ms fixed floor + ~6.0 ms per 1k input tokens, linear to 29k; **no large quadratic (attention-blowup) signature** | confident (measurement) / plausible (transformer reading) | `runs_archprobe/analysis.json` (73.11 ms, 6.053 ms/ktok, R²=0.857); refit: quadratic buys ΔR²=+0.008, top-bin residual +4.8 ms (`arch_audits.json → prefill_shape`) | Strong superlinear compute growth at long prompts beyond batching noise |
| Input pipeline | Server-side **whitespace normalizer** + fixed request template (~316 tokens on an empty state; affine intercepts 246–258) ahead of tokenization | confident (normalizer) / plausible (template size) | 300 spaces = empty-state count (`runs_archprobe/cleanrun_jev.json`, `tokens_per_char.json`); `tokenizer_fingerprint.json` intercepts | Whitespace-bearing inputs whose counts show a per-space token cost |
| Tokenizer | Vendor's own **English/Latin-centric BPE**: heavy Latin/punctuation merges; ~1 token per codepoint for Cyrillic, Greek, Arabic, Hebrew, Thai, Devanagari, Hangul, kana, common CJK; digits ~1 each; **byte-level fallback confirmed live (P1)** — rare blocks cost 0.99–1.01 tokens per UTF-8 byte, partially-merged blocks 0.54–0.73; **UTF-8 pipeline confirmed** (lone surrogates rejected, HTTP 400); NFC normalization confirmed; no match among 173 open signatures | confident (mechanism, post-P1) / plausible (vendor's own) | `runs_archprobe/probe2/probe2_analysis.json → p1_fallback`, `runs_archprobe/tokens_per_char_compare.json`, `tokenizer_perscript.json`, `tokenizer_broadscan.json`, `docs/modern-comparison/ARCHITECTURE-PROBES.md` §5b–5c | A tokenizer reproducing the whitespace-free per-script table within ~1 token/sample; acceptance of lone surrogates |
| Core model class | Transformer-family decoder LM | plausible | LM-like token behavior in Talk traces; linear prefill; capability profile; the vendor's own primer narrative (`SOURCES.md` S10) — no API signal separates attention variants at ≤29k | — (not falsifiable from this API; see §5) |
| Dense vs MoE | **Not constrained.** Nothing API-visible separates them; the economics do not require MoE | — | §5 | — |
| Size | **No point estimate — a two-angle band.** Throughput angle (BF16 nameplate 400–2250 TFLOPS × quantization 2–4×, MFU 0.25–0.45, 1–4 shards): ~0.6–49B *active* — honest but loose. Capability angle: ~4–14B dense-equivalent, which does the narrowing. A quantized dense ~4–9B is the parsimonious joint reading; a MoE (~15–100B total) stays possible, not favored. Live P2 grid confirms nothing scales with option count beyond tokens | plausible (band); assumptions stated | §4; `data_report/size_estimate.json`; `runs_archprobe/probe2/ → p2_grid` | Vendor disclosure; a matched-protocol evaluation outside the band |
| Training history | English-dominant pretraining corpus (the tokenizer's per-script coverage is its fossil record); knowledge horizon **solid to Dec 2024, partial to May 2025, none detected Jun–Aug 2025 (P5)**; assistant-shaped **judgement-format post-training** (schema perfection under load, trained abstention rising exactly where knowledge fades, shape-derived confidence); OpenAI-flavored **brand prior inherited from training text** | plausible | §2F, §2G; `runs_archprobe/probe2/probe2_analysis.json → p5_horizon`; `runs_live/FINDINGS.md` §5; `runs_archprobe/analysis.json → ancestry` | Verifiably post-horizon events answered correctly closed-book beyond the P5 fade; identity answers that track deployment facts rather than internet priors |
| Frontier-teacher distillation | Possible contributor to post-training; **not identifiable** from any API-visible signal we can construct | speculative | §6 | — (under-determined; §6 lists the weak discriminators and their power) |
| What it is not | Not a frontier model; not retrieval- or cache-assisted; not a wrapper around another vendor's API; not a relabeled *open* model | confident | §7 | One clean counter-instance each (e.g., a post-cutoff fact closed-book; a cache-flat latency component; an upstream round-trip signature; an exact open-tokenizer match) |
| Interface limits | ≤255 options/question; score rubrics 2–10 levels; noul = bare scalar with no confidence field; state ≤32k, total ≤64k tokens (vendor-documented) | confident (documented + exercised) | `src/jev_observatory/schema.py`; `SOURCES.md` S4–S6, S9; `runs_live/FINDINGS.md` §2.4 | A served request exceeding a documented limit |

---

## 2. The verified evidence base

What was measured, where it lives, whether we re-derived it in this pass, and
what it licenses.

### 2A. The read-out surface and its post-processing

* **Quantization.** Across every published probability vector — 12,584 values
  in `runs_archprobe/rows.jsonl`, 15,026 in the phase-1 raws
  (`runs_live/2026-09-18-*/raw/*.response.json`), and 676,667 in the 4,490
  per-rotation raw distributions inside the Talk traces
  (`nodes[*].ensemble_raw`, K=32–255) — **zero off-grid values** (grid 0.01;
  704,277 values in 7,887 vectors total; re-derived:
  `data_report/lattice_forensics.json`). Exact 0.00 values are common (no
  probability floor); the live raws exercise all 101 grid levels, the probe
  battery 86. Vector sums deviate from 1.0 by at most 0.01 (19/987 probe and
  24/2,410 live vectors beyond half a quantum). Formal checks:
  `src/jev_observatory/validation.py` (`DISPLAY_QUANTUM = 0.01`).
* **The lattice is bounded and one-sided (new).** Displayed vector sums are
  only ever 0.99 or 1.00 — **never above 1.00, never below 0.99**, in any
  corpus at any K up to 255 (`lattice_forensics.json → corpora[*]
  .by_k_bucket`). Independent per-value round-to-nearest would scatter sums
  with σ ≈ 0.01·√(K/12) (≈0.037–0.046 at K=65–255, i.e. about half of
  vectors off by ≥0.01 in *both* directions); observed sd is 0.005 and every
  nonzero deviation is negative. The shortfall frequency grows with K and
  flatness: ~0% at K=3–4, 5–35% at K=17–64, **46.4% of flat K≈255 vectors
  land exactly 0.01 short**; every published K=2 vector sums to exactly
  1.000 (1,489/1,489 — complement emission). Reading: the display pipeline
  apportions integer hundredths (or rounds, then applies a bounded,
  subtract-only correction); the exact rule (e.g. 0.33×3 vs 0.34/0.33/0.33
  at K=3) is what the staged P4 identical-option probe settles
  deterministically. **[confident measurement; mechanism candidates listed,
  not decided]**
* **Live P4 confirmation (P-battery).** The identical-option probe has run
  (`runs_archprobe/probe2/`): with identical option texts, sums were exactly
  1.000 at every K from 2 to 12 (8 repeats each) and 0.99–1.00 at K=255 —
  the bounded one-sided reading, confirmed live; the exact apportionment rule
  at the boundaries stays open. Bonus finding: identical texts did *not*
  receive identical probabilities (K=3 gave e.g. [0.68, 0.19, 0.13] and
  [0.75, 0.15, 0.10] across repeats; 8 distinct patterns in 8 repeats) — on
  degenerate menus the score is position and noise, not content. Near-ties
  with genuinely distinct surface forms were stable: the bare token "zzq"
  beat "ZZQ"/" zzq"/"zzq " at p_max 0.95–0.98, chosen 20/20 — the
  whitespace-blindness signature, live. **[confident]**
* **Decision and table are post-processed separately.** 9 published vectors
  (3 probe, 6 live — the live count independently reproduces
  `runs_live/FINDINGS.md` §2.2) return a `choice` that is not the argmax of
  the displayed table, every one at exactly one quantum (max gap 0.01,
  `arch_audits.json`). Round-to-nearest is monotone and cannot reorder
  options, so decision and table cannot both be simple roundings of one float
  vector — at least the decision is computed pre-quantization.
  **[confident]**
* **The `confidence` field, recovered (new in this pass).** For choice
  questions, `confidence` is the **chance-corrected top probability**
  `(p_max − 1/K)/(1 − 1/K)` computed on the pre-quantization distribution and
  rounded to the grid: of 2,289 published choice vectors, 1,405 match exactly
  against displayed p_max, 822 are within one quantum, 62 within two, **none
  beyond** (`arch_audits.json → confidence_formula`). The 1–2-quantum tail is
  the expected residue of confidence using pre-quantization p_max while the
  table is rounded separately (cf. the sum deviations and argmax mismatches
  above). Score-type answers use a *different* shape statistic (100/121 beyond
  two quanta under the choice formula; max diff 0.48); noul answers carry no
  confidence field at all (963/963). This pins the vendor's "confidence is
  computed from probability-distribution shape" (`SOURCES.md` S6–S7) to a
  formula for choice. Practical consequence: choice `confidence` adds **no
  information beyond (p_max, K)** — it is not an independent calibration
  signal. **[confident]**
* **Schema perfection under load.** 19 contract-invalid responses across the
  text-question stages (18/12,032 MMLU-Pro, 1/420 rotations, zero in
  ARC-Challenge, GPQA, HLE, MATH-500;
  `runs_benchmark*/bench-*/derived/score.json → strict_format_failures`);
  the ARC-AGI-2 multi-question stages do not define per-item counts, and
  their cell-level equivalent is zero (`cells_unusable = 0` in all three
  encodings — note their `status_counts['strict_format_failure']` keys are
  definitional artifacts, §11 item 15). A trained
  read-out with a serializer, not parsed prose. **[confident]**
* **Nondeterminism.** Identical payloads give different responses (30
  byte-identical payloads → 13 distinct response signatures,
  `runs_live/FINDINGS.md` §2.5); on flat 254-option menus repeat-TVD is
  0.034–0.122 (`runs_live/token_talk_orderprobe.json`). This is the noise
  floor for every distribution comparison in this document. The most
  parsimonious mechanism is batch-dependent floating-point reduction order in
  a continuously-batched serving stack (amplified by flat distributions and
  made visible by the coarse grid); it does not identify the architecture and
  is fully compatible with a single deterministic-in-weights model.
  **[confident measurement; plausible mechanism]**

### 2B. Latency, self-batching, and the shape of serving

All from `runs_archprobe/` (987 calls, 0 errors, `BILLING.json`; server
compute = `x-envoy-upstream-service-time`, queue-free). Fits re-derived in
this pass (`arch_audits.json → prefill_shape, marginal_cost`):

* **Prefill.** upstream = 73.11 ms + 6.053 ms per 1k input tokens over
  859–28,859 tokens (R² 0.857). A quadratic term fits at 0.0756 ms/ktok² but
  buys ΔR² = +0.008, and the five longest prompts (~28.9k tokens) average only
  **+4.8 ms above the pure-linear fit** — inside the ±30–50 ms scatter at mid
  sizes. Reading: compute is floor + linear-in-tokens; **no large
  attention-blowup signature to 29k** — and equally, no way to distinguish
  efficiently-kernelled attention from anything else sub-quadratic at these
  lengths. **[confident measurement; interpretation bounded]**
* **The read-out is invisible (new decomposition).** Each packed question
  adds ~55.4 input tokens and ~33.4 output tokens; at the measured prefill
  slope its tokens alone predict +0.336 ms, against a measured marginal of
  +0.442 ms — residual **≤0.11 ms/question** for the entire decision
  computation. Each option adds ~18.2 input tokens and ~9.6 output tokens;
  tokens predict +0.110 ms against a measured +0.102 ms — residual **zero
  within noise** (headcount 1→192 questions: upstream 71→153 ms; optioncount
  2→255: 65→90 ms; `rows.jsonl`). The whole latency model collapses to:
  **floor + prefill(all input tokens) + nothing measurable for deciding.**
  That is the one-forward-pass, many-read-outs picture in its strongest form:
  the decision cost is not merely small, it is *accounted for* by the tokens
  the questions and options contribute to the shared prompt. **[confident]**
* **Output tokens are serialization, not generation.** Billed output grows
  ~33 tokens/question and ~9.6/option in the probes, and benchmark responses
  track their own JSON length (MMLU-Pro, 10–19 options: 1,014,506 output
  tokens / 12,032 calls ≈ 84/call; GPQA, 4 options: 9,016/196 ≈ 46/call;
  `runs_benchmark*/…/score.json → attempts_summary`), at $0
  (`data_report/costs.json`). "Free output" is the natural accounting of a
  prefill-only service. **[confident]**
* **Continuous batching.** Identical independent requests at concurrency
  1→32: median upstream 78.5/80/86/80/74/81.5 ms — flat; client wall bends
  only at c=32 (288→622 ms), which is our own connection pool
  (`analysis.json → concurrency`). One batched 4-question call ≈261 ms vs
  four separate calls ≈1,117 ms (`runs_live/FINDINGS.md` §3): no per-question
  external round trip exists to hide. **[confident]**
* **Cold path.** +353.5 ms median wall for cold TCP+TLS, upstream unaffected
  (`analysis.json → coldwarm`) — validates the header as the compute signal.
* **Battery-power caveat (new).** The short tokenizer battery that produced
  the Qwen false lead carries **≤14 tokens of discriminating information per
  probe** (reported counts 313–330 against the ~316-token empty-state
  template); the whitespace-free mergerate samples carry 2–315
  (`arch_audits.json → battery_power`). Any tokenizer ranking from the short
  battery is template-dominated; only the per-script table discriminates. This
  is the quantitative version of the cautionary tale in
  `docs/modern-comparison/ARCHITECTURE-PROBES.md` §5b. **[confident]**

### 2C. The tokenizer profile

Full walk-through: `docs/modern-comparison/ARCHITECTURE-PROBES.md` §5; tables:
`runs_archprobe/tokens_per_char_compare.json`. Whitespace-free marginal
tokens/codepoint (baseline-subtracted):

| | Latin words | digits | Cyrillic | CJK common | CJK random | Hangul | Greek | emoji | flags |
|---|---|---|---|---|---|---|---|---|---|
| **Jev** | 0.26 | 0.94 | 0.94 | 0.92 | 1.96 | 0.92 | 0.94 | 1.71 | 1.56 |
| Qwen2.5 | 0.11 | 1.01 | 0.51 | 0.51 | 2.01 | 1.01 | 1.01 | 1.04 | 1.06 |
| o200k | 0.11 | 0.34 | 0.34 | 0.51 | 2.01 | 0.68 | 0.81 | 1.04 | 2.06 |
| cl100k | 0.11 | 0.34 | 0.68 | 1.26 | 3.01 | 1.34 | 1.01 | 2.04 | 3.06 |

* Every major non-Latin script sits at ~0.92–0.96: the vocabulary holds
  **single-codepoint entries** for common Cyrillic/Greek/Arabic/Hebrew/Thai/
  Devanagari/Hangul/kana/common-CJK characters but essentially **no
  multi-character merges** for them (Qwen, o200k, cl100k all merge these
  scripts at 0.34–0.68). Digits cost ~1 each (no multi-digit tokens, unlike
  o200k/cl100k/Llama-3/DeepSeek). Latin merges heavily ("hello"×30 → 0.15
  tokens/char; 300 dashes → 14 tokens). **[confident]**
* Uncovered characters fall back **below codepoint granularity**: random
  (mostly rare) CJK costs 1.96 tokens/char, astral emoji 1.71,
  regional-indicator flags 1.56/codepoint (≈3.1 tokens per 8-byte flag pair).
  Super-codepoint, sub-byte-count — what **byte-level BPE fallback with
  partial byte-n-gram merges** (GPT-2-lineage mechanics) produces. Pure
  codepoint fallback is excluded (it would pin every BMP char at ≤1.0;
  cjk_random is 1.96). The counts alone cannot fully separate byte-granular
  hybrid variants; probe P1 (§10) can. **[plausible]**
* Affine fits across 88 candidate vocabularies on the whitespace-free table
  leave the best (tencent/Hy-MT2-7B) at **RMSE 20.6 tokens** with systematic
  per-script sign errors (Cyrillic +30, Hangul −39;
  `tokenizer_perscript.json`). The broad scan (1,215 HF repos → 173 unique
  (vocab+specials) signatures → 128 tokenizers scored,
  `tokenizer_broadscan.json`) finds no exact match; its near-zero LOO
  residuals (~0.47 tokens for the leader) are the short-battery artifact
  quantified in §2B, not a near-match. Honest scope: **no open/public
  tokenizer reproduces Jev's profile**; a private, unpublished vocabulary
  from any lab is outside the scan's reach. **[confident within the scanned
  set]**
* Pure whitespace collapses to the empty-state count (300 spaces → 316
  tokens, the baseline itself; `cleanrun_jev.json`) — a serving-path
  **normalizer** runs before tokenization, so observed token statistics
  describe the *pipeline*, not only the model. Fixed-template constants: ~316
  tokens empty-state; affine intercepts 246–258
  (`tokenizer_fingerprint.json`). **[confident]**
* **Live P1 confirmation (P-battery, `runs_archprobe/probe2/ → p1_fallback,
  p1_merge`).** Rare uncovered blocks cost ~1 token per UTF-8 byte
  (cjk_ext_a 0.994, cherokee 1.006, yi 1.006, cjk_ext_b 0.994 tok/byte —
  3–4 tokens per character): byte-level fallback with no byte-pair merges in
  those ranges. Partially-covered ranges sit between (ethiopic 0.672,
  emoji_rare 0.675, math_bold 0.731, hangul_rare 0.728, cjk_compat 0.544
  tok/byte — some byte n-grams merged). A JSON lone surrogate is rejected
  with HTTP 400 "Request contains invalid Unicode text" — the pipeline
  validates UTF-8; UTF-16 internals are excluded. Decomposed (e + U+0301)
  and precomposed (é) runs cost identical tokens — NFC normalization runs
  pre-tokenization. The normalizer's whitespace set is ASCII-only: 60 spaces
  → +2 tokens, 60 tabs → +6, 60 newlines → +9, while NBSP/ZWJ/ZWNJ/BOM/
  soft-hyphen/ideographic-space each cost ~1 token per char. Merge
  fingerprints: "hello" = 1 token at every repeat count (no super-merges),
  "th" = 1 token, digits single, dash runs saturate at ~10–16 dashes/token,
  case seams cost +31 tokens per 60 "aA" pairs vs "aa", inter-word
  punctuation ~+1 token per word. **[confident]**
* Read as a corpus fossil: full single-char coverage of the *small* script
  blocks (Cyrillic ~64–1k chars, Greek, Hebrew, Arabic, Thai, Devanagari,
  kana) plus common CJK, but not the 21k-char CJK block; digits unmerged;
  deep Latin/punctuation merges. That is an **English-dominant training
  corpus with incidental multilingual exposure** — against a Chinese-lab
  multilingual base specifically (those merge CJK/Cyrillic hard), and against
  OpenAI's public encodings (digit and Cyrillic columns). **[plausible]**

### 2D. Capability profile (band, not parameters)

Scores: `runs_benchmark*/bench-*/derived/score.json` + `weighted_score.json`;
references with protocol labels:
`docs/modern-comparison/canonical/comparable-scores.json`.

* **Knowledge MCQ, direct one-shot:** MMLU-Pro 82.8% [82.1, 83.5]
  (n=12,032); GPQA Diamond 76.5% [70.1, 81.9] (n=196); ARC-Challenge 97.9%
  (saturated for every model in the comparison set). Nearest published
  neighbors: Claude 3.7 Sonnet no-thinking 80.7 MMLU-Pro (direct, canonical)
  and 76.8 GPQA (direct, aggregator); Qwen 3.5 9B 82.5 / 77.6 (reasoning
  protocol). Jev sits **in the 2025 small-instruct-model band on knowledge**,
  with the standing caveat that several neighbors used CoT/reasoning and Jev
  never does — under a matched protocol Jev might sit slightly higher in the
  band, not lower. **[confident measurement; band is positioning]**
* **Expert-frontier material:** HLE MC-subset 21.9% [18.4, 25.7] — between
  chance and 2025 rows (GPT-5 26.3 reasoning; DeepSeek V4 Flash 34.8 direct
  self-reported), roughly half the 2026 frontier (49.5–62.5). ARC-AGI-2
  public eval: 53.5% per-cell diagnostic, **0/120 exact tasks**
  (`runs_benchmark_ext/bench-arc_agi2_choice-*/derived/score.json`). Far
  below frontier wherever multi-step reasoning carries the item.
  **[confident]**
* **Recognition ≫ production ("generation tax").** MATH-500 encodable
  subsets: 83.1% choosing the right MCQ option vs **13.6%** reading the same
  answers off a per-digit rubric (weighted 4.8%) — same knowledge, different
  output channel. The Talk ladder completes the picture: character menus
  degenerate under every control; vocabulary menus give 86–100% real words
  and **zero syntax**; coherent text appears only when a local LM supplies
  the sequential prior (`docs/jev-talk-program-report.md` §4;
  `docs/token-talk-findings.md` §10, §13 — the Jev-only rungs). Jev's
  sequential generative competence, where measurable without a collaborator,
  is weak — which is exactly what heavy judgement-format post-training on a
  small model would produce, and exactly what the "system one" product story
  requires. **[confident]**
* **Options are scored essentially absolutely (P3, P-battery).** 200
  gold-labeled synthetic items run at every K from 2 to 255 with inert
  padding: mean p(gold) = 1.000 at K=2 and 0.992 at K=255, accuracy 1.000 at
  every K, padding pinned at 0.00 (99.4% of entries at K=255). A
  set-softmax over noise-level option utilities would dilute p(gold) steeply
  in K; nothing of the sort happens. Reading: each option is scored against
  the state near-absolutely, then normalized for display — weighted scores
  are comparable across option counts, and inert menu padding is a safe
  protocol tool (`runs_archprobe/probe2/ → p3_kcal`). **[confident for inert
  padding; semantic distractors untested]**
* **Position effects are real but unbiased on content-bearing items (new).**
  Pairing the rotation audit to native-order results by item id: 419 pairs,
  22 discordant (5.3% flip rate), native 86.6% vs rotated 86.2% on the
  subset, exact McNemar **p = 0.83** (`arch_audits.json → rotation_pairs`).
  Option order flips a few percent of answers without moving accuracy — in
  sharp contrast to flat creative menus, where ordering changes the argmax
  almost every time (11 orderings → 10 distinct winners; Spearman vs native
  0.26–0.44 on flat steps vs 0.42–0.88 on sharp ones;
  `runs_live/token_talk_orderprobe.json`, `docs/token-talk-findings.md` §11).
  Position sensitivity scales inversely with content signal — the signature
  of an in-context list reader, not of isolated per-option encoders. The
  headline "86.2% rotation audit vs 82.8% full set" is **subset selection**
  (native accuracy on those same 419 items is 86.6%), not a shuffle benefit.
  **[confident]**

### 2E. Calibration and distribution shape

* **Bimodal, not uniformly timid.** MMLU-Pro weighted mean p(gold) = 0.740
  vs greedy 0.828, but the median item carries p(gold) = 0.93 and p05 = 0.04
  (`runs_benchmark/bench-mmlu_full-*/derived/weighted_score.json`). Most
  items get near-decisive distributions; a hard tail gets near-zero mass on
  gold — including **213 items (1.8%) at exactly 0.00** (`score.json →
  zero_probability_gold`; HLE: 45 zeros, clipped log-loss 5.17; GPQA: none).
  The clipped MMLU log-loss (1.114 at ε=1e-15,
  `src/jev_observatory/benchmark_score.py`) must always be quoted alongside
  the zero count. "Cannot hallucinate" fails in the strictest probabilistic
  sense, with the `runs_live/FINDINGS.md` §1 caveat that some zero-gold cases
  look like benchmark label errors and need adjudication before any claim
  about *factual* confident-failure rates. **[confident]**
* **Noul vs choice:** paired BoolQ (n=500): accuracy-equivalent (McNemar
  p=1.0); paired Brier favors noul by 0.0115 [0.0055, 0.0177]; noul P(yes)
  reliability monotone but imperfect (ECE-10 0.068 noul vs 0.083 choice;
  `runs_live/FINDINGS.md` §1). **[confident within that run]**
* Weighted < greedy on nearly every stage is the expected shape of a read-out
  whose hard tail smears mass; it is *not* evidence of sampling noise or of
  an ensemble behind the endpoint. **[plausible]**

### 2F. Identity and brand prior (training text, not lineage)

* Forced-choice ancestry probe: 15 frames × 24 cyclic orderings, 360 calls,
  position provably cancelled (winner at index 0 only 4.7% — below uniform):
  family mass **openai 0.4803**, qwen 0.112, typesafe 0.1031, anthropic
  0.0871; greedy votes OpenAI-family 337/360 (OpenAI 293, ChatGPT 22, GPT
  22); mean p("Jev") 0.0648; mean p("Typesafe") 0.0103 — **at the
  quantization floor** (the 0.0003 excess is sub-quantum; §11)
  (`runs_archprobe/analysis.json → ancestry`).
* With "Jev"/"Typesafe" spellings injected at probability parity with
  model-name tokens, Jev never chose them across four sessions
  (`runs_live/identity_parity_*.json`; `docs/token-talk-findings.md` §15);
  explicitly injected contrary tokens get floor treatment and are never
  chosen (§9).
* A stated fictional origin ("Meridian Labs") is accepted at 0.97
  (`runs_live/FINDINGS.md` §5) — prompt-suggestible branding.
* Reading: an assistant-shaped post-training corpus saturated with
  OpenAI-flavored self-descriptions — the same prior that makes every
  internet-trained assistant lean "OpenAI" when asked who made it. The token
  counts (§2C) independently contradict both OpenAI-encoding and Qwen
  lineage. Per ground rule 5, none of this is ancestry evidence in either
  direction; it *is* evidence about the post-training text diet.
  **[plausible]**

### 2G. Temporal horizon and abstention

* 2022/2023/2024 facts answered at p≈0.97–1.0; the Nov-2024 US-election
  question answered correctly (p≈0.98) under forced choice; a nonexistent
  event correctly refused closed-book (p=1.0); knowledge "extends to at
  least late 2024" — availability, not a cutoff measurement
  (`runs_live/FINDINGS.md` §5). **[confident measurement; horizon plausible]**
* Offered a "cannot say" option, Jev **abstains on the election question
  (p=0.71)** it answers under forced choice, and refuses the fictional-
  premise control — a trained conservative-abstention behavior, not absent
  knowledge. **[confident]**
* **Dated bisection, live (P5, P-battery).** 26 dated events (Oct 2024 →
  Aug 2025) plus pre-cutoff and fictional controls, in forced and abstention
  frames, 2 reps each (104 calls): gold-rate 1.00 for Oct/Nov/Dec 2024;
  partial for Jan 2025 (0.58), Feb (post-only ≈0.5), Apr (0.25), May (0.5);
  **0.00 for Jun, Jul, Aug 2025**; all 16 fictional-event calls correctly
  answered "did not occur"; abstentions appear exactly in the fade zone.
  Reading: a parametric horizon with a soft edge — data collection plausibly
  ran into spring 2025 with decaying coverage; nothing suggests retrieval or
  a rolling update. Disclosed blemish: one pre-cutoff control ("Starliner
  crew return", labeled Aug 2024) scored 0.00 because the label was wrong —
  the return was Sept 11, 2024; Jev's non-answer was arguably correct. The
  horizon claim rests on the post-2024 items, which are unaffected
  (`runs_archprobe/probe2/ → p5_horizon`). **[confident measurement;
  month-level precision limited by n=4–12 per month]**

### 2H. Economics as architecture evidence

* Measured billing: **$0.042/M input tokens, output free**; 23,459 recorded
  calls, 85.2M input tokens, $3.58 total program (`data_report/
  billing_totals.json`, `costs.json`). MMLU-Pro full run ≈ $0.28; GPQA
  ≈ $0.0044/question.
* The prefill-only economics are *measured* (§2B: no decode loop, output =
  serialization), not inferred from price — price could be subsidized, and
  the report says so. What the price *does* corroborate: a service whose
  marginal cost is dominated by prompt prefill can rationally give output
  away, because output costs it nothing but serialization.
* For scale: $0.042/M is ~3.6–10× below the cheapest 2026 small-model list
  prices (GLM-5.3 Flash $0.15/M, DeepSeek V4 Flash $0.44/M peak) and one to
  two orders of magnitude below flagship input lists (~30–240×;
  `costs.json`) — consistent with a small model, aggressive batching,
  prefill-only workload, and/or launch subsidy. It does not identify any of
  them. **[plausible]**

---

## 3. Synthesis: the model we think Jev is

**[plausible, composite of confident parts]** A decoder-style transformer
language model of modest 2026 standards — trained on an English-dominant
corpus with a vendor-own BPE tokenizer whose per-script coverage records that
diet — post-trained hard for *judgement*: given a state and a menu, emit a
calibrated-ish distribution over the menu, in one pass, with no text. The
post-training shaped both the head (a probability read-out with a
chance-corrected confidence statistic, a separate decision path, and a
0.01-precision display whose sums are bounded one-sided — 0.99 or 1.00,
never above) and the behavior (schema perfection, semantic
stopping on short answers, trained abstention, an OpenAI-flavored brand
prior inherited from assistant text). Serving runs the read-out as prefill:
one forward pass over state + questions + options, many read-out vectors,
serialized as the response and billed as free output, continuously batched
across tenants. The knowledge horizon behaves like a fixed training cutoff
with a soft edge in spring 2025 (P5: solid to Dec 2024, partial to May 2025,
none detected Jun–Aug 2025). Whether the post-training signal came substantially from a
frontier teacher (distillation) or from on-policy judgement data is not
identifiable from anything this API returns; whether the transformer is dense
or MoE is not identifiable either, and the economics do not need MoE.

The vendor's own framing — "system one" models, questions "evaluated in
parallel and in isolation", a post-training path they call RLCD branching off
the usual pretrained-LM trunk (`SOURCES.md` S1, S2, S10) — is *consistent*
with every measurement above. That is worth saying plainly: our independent
reconstruction and the marketing agree on the shape. Where we part company is
"frontier-level" (refuted, §7) and "cannot hallucinate" (an envelope
guarantee, not a content guarantee: 213 zero-probability gold outcomes on
MMLU-Pro alone, §2E). Under our reconstruction, "system one" is primarily a
*training-and-interface* claim — a language model whose generation was traded
away for fast structured judgement — and the trade is visible in every
artifact: the generation tax (83.1 vs 13.6 on the same MATH items), the word
salad of unguided sequential decisions, and the near-free marginal cost of
deciding.

---

## 4. Size: two angles that converge on a band

Ground rule 4 forbids reading a parameter count *off a latency slope*, and for
good reason: the marginal prefill rate is measured while the server batches our
tokens with other tenants', so a naive slope→size conversion would be a
scheduling artifact. But "do not convert the slope" is not "do not estimate."
There are two independent angles, each with explicit assumptions, and they
overlap — so we give a band and show the arithmetic.

### Angle 1 — prefill throughput (an upper bound on *active* parameters)

The measured marginal prefill rate is 6.053 ms per 1k input tokens
(`runs_archprobe/analysis.json`) = **R ≈ 165,000 tokens/s** of incremental
server compute. Batched prefill is compute-bound (165k tok/s is far above
memory-bound decode rates), and a forward pass costs ≈ 2·N_active FLOPs per
token (the Kaplan/Chinchilla convention; attention adds ≲15% at ≤29k context,
inside the assumption range). If the accelerator delivers MFU × peak FLOPS,
then sustaining R tokens/s needs

> N_active ≤ MFU × peak × shards ÷ (2R)

Across an honest grid — MFU 0.25–0.45 (achieved utilization on large-batch
prefill), the plausible late-2026 fleet's **BF16 nameplate of ~400–2250
TFLOPS per device** (H100/H200-class ≈990; B200-class ≈2250; smaller parts
≈400) **times a quantization multiplier of 2–4×** (fp8 doubles, int4/FP4-class
quadruples effective math throughput) = **~800–9000 effective TFLOPS per
device**, over 1–4 devices per pass — this bounds the **active footprint at
≈ 0.6–49B parameters, central case ≈ 4.2B**
(`data_report/size_estimate.json`). (Two earlier passes were wrong and are
recorded in the artifact: the first assumed bf16 250–500 TFLOPS on A100-class
hardware → 0.2–2.7B; the second treated the fleet's 400–2250 BF16 figures as
if they were already effective → 0.3–12.3B. The audit caught the second
error: quantization still has to be applied on top of BF16 nameplate.) Two
honesty notes: (a) concurrent streams B>1 sharing weight fetches only *lower*
the per-stream bound, so B=1 is the conservative direction; (b) the MFU/peak
ranges are industry-standard serving assumptions, not repo measurements. This
is an explicit-assumptions bound, not the forbidden slope→size conversion.
The vendor's advertised 250k tok/s (`SOURCES.md` S8) is the same order as our
R — consistent, and equally not a size.

### Angle 2 — capability band (a dense-equivalent range)

Direct-answer MMLU-Pro 82.8 / GPQA 76.5 sit beside Qwen 3.5 9B (82.5 / 77.6,
reasoning protocol) and above Claude 3.7 Sonnet no-thinking (80.7 / 76.8,
direct); the next rung up (Qwen3.8-27B, 84.3 / 82.2) is clearly above Jev
(`docs/modern-comparison/canonical/comparable-scores.json`). HLE 21.9 and
ARC-AGI-2 0/120 exact cap it far below frontier sizes. That places Jev in a
**~4–14B dense-equivalent** band, nearest-analog ~9B-class, with the protocol
mismatch (several comparators used CoT; Jev never does) cutting in Jev's
favor. Capability bands are loose — hundreds of models share this one — and
distillation shifts capability-per-parameter upward.

### Where the two angles meet

Under the corrected quantization arithmetic the throughput band (0.6–49B) is
honest but **loose** — consistent with anything from a small dense model to a
~50B-active MoE. The capability angle (4–14B) does the narrowing. The
readings, in order of parsimony:

1. **Quantized dense (parsimonious, sufficient)**: a dense ~4–9B model served
   at fp8/int4 fits both angles with nothing further assumed — an entirely
   ordinary late-2026 object at this price point ($0.042/M, 73 ms floor).
2. **MoE (possible, not favored)**: ~15–100B total at ~5–20% activation also
   fits, and would explain the *top* of the capability band with less compute
   per token — but the loose throughput band removes the capacity-density
   argument that once favored it. The API cannot see expert structure (§5).
3. **Distilled smaller dense**: teacher labels lift a 2–6B dense model into
   the capability band on knowledge MCQs (transferring knowledge, not
   multi-step reasoning) — and Jev's recognition ≫ production asymmetry
   (83.1% MCQ vs 13.6% digit read-out, §2D) is exactly that shape. Weak,
   indirect support for §6.
4. **Soft band edges**: B>1 amortization, higher MFU, or more shards move the
   throughput bound in either direction; the capability band is loose by
   construction.

**Converged statement [plausible-band]:** ACTIVE parameters most plausibly
**~4–9B** (capability-driven; the 0.6–49B throughput bound is consistent
but loose) — dense-equivalent order 4–9B, a 2025-era small model. A
quantized dense model in this range reconciles both angles without MoE; a
MoE (~15–100B total) remains possible and unfalsifiable from the API. TOTAL
parameters **unconstrained** either way. This respects ground rule 4: the
throughput angle is an explicit-assumptions bound, not a slope-to-size
conversion, and the capability angle is positioning under protocol
mismatch. **Falsified / tightened by**: vendor disclosure (trivially); a
matched-protocol capability evaluation (removes capability-band looseness).
The P2 option-cost grid has now run: its K-coefficient is zero within
±20.5 ms residual noise, confirming the single-trunk marginal-cost model but
not tightening the size bound (`data_report/size_estimate.json`).

---

## 5. Dense vs MoE, attention variants: what constrains them

**Nothing API-visible constrains either axis. [confident that it is
unconstrained]**

* Latency: prefill-only, floor + linear (§2B) is what both produce under
  continuous batching; the absent quadratic term to 29k bounds *attention
  blowup*, not attention type — flash-attention transformers, GQA/MLA, and
  sub-quadratic architectures all pass this test at these lengths.
* Nondeterminism (TVD 0.03–0.12 on flat menus): batch-dependent reduction
  order explains it for dense and MoE alike; capacity-based expert dropping
  would be an MoE-specific amplifier, but we cannot separate amplification
  from ordinary noise at this precision.
* Economics: prefill-only serving of a *small dense* model is already cheap
  enough to explain $0.042/M with free output; MoE is not required. It is
  not excluded either — small-active MoE is a common 2025–26 shape for
  exactly this cost point.
* Token counts: say nothing about weights by construction.

We therefore carry **no MoE/dense claim in the card**, and treat any future
assertion of either as needing evidence this API does not currently leak.
The one honest asymmetry: if forced at gunpoint, the 2026 population of
models built for cheap high-throughput judgement tilts MoE — but that is a
prior about the industry, not a measurement of Jev.

---

## 6. Distillation from a frontier teacher: can the API tell?

Short answer: **no clean signal exists; we assign it a meaningful but
unresolved probability (~0.35 that teacher-generated data contributed
materially to post-training) and say why.** The two-angle size estimate (§4)
briefly raised this to ~0.40 while the throughput band (bf16 assumptions)
sat far below the capability band and distillation was one of few
reconciliations; the quantized-serving revision made the bands overlap, so
the size argument dissolved and the estimate is back to ~0.35, resting on
the asymmetry argument alone.

Candidate discriminators and their actual power:

1. **Brand prior (§2F).** An OpenAI-shaped self-concept is consistent with
   an OpenAI-family teacher — and equally with the open assistant-text
   ecosystem, which is OpenAI-flavored regardless of teacher. *Power: near
   zero.* It cannot even distinguish "distilled from GPT-class outputs" from
   "trained on scraped ChatGPT conversations."
2. **Recognition ≫ production asymmetry (§2D).** Teacher-labeled
   MCQ/judgement distillation transfers answers without transferring
   multi-step reasoning — the observed shape. But judgement-focused
   post-training on *any* label source produces the same shape, and so does
   small size. *Power: weak.*
3. **Calibration shape (§2E).** Distillation typically transmits the
   teacher's confidence structure; hard-label RL typically over-sharpens.
   Jev is bimodal with a confident-wrong tail — consistent with either a
   mixed-signal post-training diet or teacher labels on hard items. At the
   0.01 grid we cannot resolve distributional fine structure. *Power: weak.*
4. **Error-sharing correlation (probe P6, §10).** If Jev's *wrong* answers
   on public MCQ items correlate with a specific teacher family's errors
   beyond the base rate at which small models share errors, that is
   suggestive. Confound: shared pretraining data makes all models share
   errors; distinguishing "same teacher" from "same internet" needs
   item-level controls we can only partially build. *Power: low, and the
   only one with any directness.*
5. **Vendor context.** TypeSafe's own launch material describes workflow
   references as "average probabilities of GPT-6 Astra and Fable 5.1"
   (`SOURCES.md` S1) — they are comfortable using frontier models as label
   sources for *evaluation*, and their primer places their RLCD path on a
   pretrained-LM trunk (S10). This raises the prior that frontier-teacher
   data appears somewhere in the pipeline; it is not evidence about the
   weights. *Power: prior-shifting only.*

What would actually settle it — training logs, teacher-output likelihood
tests under white-box access — is outside the behavioral envelope by
construction. We state distillation as **[speculative]** and refuse to
upgrade it on any current artifact.

---

## 7. What Jev is not, with the measurements that say so

* **Not a frontier model [confident].** HLE 21.9% vs frontier 49.5–62.5;
  ARC-AGI-2 0/120 exact; GPQA 23.5% wrong; generation ladder collapse
  (§2D). Protocol mismatch explains parts of individual gaps; no protocol
  explains all of them simultaneously.
* **Not retrieval- or cache-assisted [confident].** (a) Compute grows
  linearly in prompt tokens at ~6 ms/ktok with a fixed floor (§2B) — a cache
  hit would not prefill the prompt; there is no flat-cost component keyed to
  question content. (b) Novel content is answered, not looked up: fresh
  generator items (`runs_live/FINDINGS.md` §3c, with its easiness caveat) and
  70,100 ARC-AGI-2 cell decisions at 53.5%. (c) The knowledge horizon
  *stops with a soft edge*: solid to Dec 2024, fading Jan–May 2025, zero
  Jun–Aug 2025 (P5 dated bisection, §2G) — a live retriever would not have
  a date-shaped fade like that, and the abstention pattern
  (knows-then-hedges) is parametric-memory behavior.
  (d) Repeats are not identical (TVD 0.03–0.12; 13 signatures over 30
  identical payloads) — a lookup cache is deterministic. (e) 213
  zero-probability gold outcomes on *public* MMLU-Pro text (§2E) is not what
  benchmark-keyed retrieval produces.
* **Not a wrapper around another vendor's API [confident].** (a) Upstream
  compute is 73–86 ms flat for ≤2k-token requests at concurrency 1→32 (§2B)
  — there is no room inside those milliseconds for a frontier API round
  trip, and a wrapped per-question call would serialize (measured instead:
  batched 4-question call 261 ms vs 4×279 ms separate). (b) The usage
  counter follows a tokenizer that matches no public encoding, including
  both OpenAI ones (§2C) — a wrapper would expose its upstream's counting or
  a re-tokenization we could fit. (c) $0.042/M with free output sits one to
  two orders of magnitude below every 2026 flagship's input list price
  (~30–240×, `data_report/costs.json`), with output free while flagships
  bill output at 4–5× input: reselling frontier inference on those terms
  does not survive.
* **Not a relabeled open model [confident within the scanned set].** The
  whitespace-free per-script profile matches none of 173 open tokenizer
  signatures across 1,215 repos (§2C); both historical "leads" dissolved as
  template artifacts (Qwen: §5b of ARCHITECTURE-PROBES; the broadscan's
  sub-token LOO residuals: §2B battery-power caveat). Scope note, stated
  once: a *private* base from another lab cannot be excluded by token
  counts — but such a hypothesis does no explanatory work that "vendor's own
  new foundation" doesn't, and it inherits the same tokenizer problem.
* **Not "cannot hallucinate" in the content sense [confident].** The
  guarantee is the envelope: schema-valid, in-menu, quantized. The content
  is wrong 23.5% of the time on GPQA Diamond and puts exact 0.00 on the
  gold answer 213 times on MMLU-Pro (§2E), beautifully formatted every time.

---

## 8. Vendor claims, adjudicated

| Claim (source) | Verdict | Evidence |
|---|---|---|
| "New architecture" (S1) | Partly testable; the *interface+serving* shape is distinctive and real; core-transformer novelty not established | §2A–2B (read-out, one-pass serving); §5 (core unconstrained) |
| "Parallel sampler" (S1) | Consistent with self-batching measurements; not independently identified | §2B |
| RLCD post-training path on a pretrained LM (S1, S10) | Vendor narrative; consistent with every behavioral measurement (judgement-format training signature); algorithm unverifiable from outside | §2D–2F, §3 |
| Questions evaluated "in parallel and in isolation" against shared state (S2) | Parallel: supported (+0.44 ms/question ≈ its tokens' prefill). Isolation: supported only *at the noise floor* (worst TVD 0.02 across sibling manipulations); the ID-invisibility experiment was confounded and establishes nothing | §2B; `runs_live/FINDINGS.md` §3, §3b |
| "Frontier-level performance" (S1) | Refuted | §7; benchmark section of the report |
| "Cannot hallucinate" (S1) | Envelope guarantee only; content-wise refuted | §2E, §7 |
| 250,000 tokens/s (S8) | Same order as measured marginal prefill (165k tok/s/stream); different quantity; not verified under load | §4 |
| 70–500 ms service responses (S1) | Consistent: warm wall medians ~263–409 ms; upstream compute 40–355 ms across all 987 probe calls | `runs_archprobe/rows.jsonl` |
| $0.042/M input, free output (S1, S8) | Measured exactly | `data_report/costs.json`, `billing_totals.json` |
| "Coauthor of ChatGPT" (launch page) | A credential about a person, not a provenance claim about weights; Jev's own identity answers are internet prior (§2F) and its tokenizer matches no OpenAI encoding (§2C) | report §pitch; `SOURCES.md` (almeida note) |
| Score levels "judged independently without seeing neighbors" (S5) | Not tested directly; the choice analogue is testable (probe P3) and the position-sensitivity data (§2D) already suggest options are *not* fully independent in-context | §10 P3 |

---

## 9. Alternative-hypotheses ledger

Probabilities are our honest posteriors given the published evidence set;
they are not a partition (rows overlap across axes). "Moves it most" names
the single cheapest measurement with real power.

| # | Hypothesis | For | Against | P | Measurement that would move it most |
|---|---|---|---|---|---|
| H1 | **Composite card (§3):** own small foundation, English-centric BPE w/ byte fallback, judgement post-training, read-out head, one-pass prefill-only serving | Everything in §2; tokenizer scan; latency decomposition; economics; horizon behavior; **P1 confirmed byte-level fallback + UTF-8 + NFC; P2 confirmed single-trunk scoring; P3 confirmed absolute option scoring; P4 confirmed bounded sums; P5 confirmed a parametric horizon** | Nothing direct; the remaining plausible-tier links (vendor's-own vocab, transformer core) resisted every cheap test we could build | **0.80** | Vendor disclosure, or a public tokenizer match appearing post-scan (re-run `tokenizer_broadscan.py` periodically) |
| H2 | Relabeled **open** model (e.g. a Qwen/Llama/Hy-MT2 checkpoint behind the read-out) | Marketing's ChatGPT-adjacent framing invites it; the first token fit said Qwen | Per-script table excludes every scanned family (§2C); both "leads" were template artifacts (§2B); identity prior contradicts Qwen specifically | **0.03** | An exact open-tokenizer match appearing post-scan (re-run `tokenizer_broadscan.py` against new releases periodically) |
| H3 | Relabeled **private/internal** base from another lab (not in any scan) | Counts cannot see private vocabularies; brand prior is OpenAI-flavored | Must still explain the unmatched per-script profile, the whitespace normalizer, and the read-out post-processing — i.e., it converges to H1 with extra steps | **0.10** | P1: a private vocab is still a vocab — fallback-granularity + merge-boundary behavior narrows the space even without a match |
| H4 | Wrapper/ensemble around external frontier API(s) | None positive; only the brand prior | §7 timing, tokenizer, and economics arguments; flat upstream at c=32 | **0.02** | Any upstream-latency signature inside the 73 ms floor under load (none in 987 calls) |
| H5 | Retrieval- or cache-assisted answering | 82.8% MMLU-Pro is high for the band, inviting a memorization story | §7 items (a)–(e); **P5 ran: a date-shaped fade (solid→partial→zero) with fictional events refused is parametric-cutoff behavior, not retrieval** | **0.02** | Re-run P5 months apart: a moving horizon would resurrect this | 
| H6 | Frontier-teacher **distillation** contributed materially to post-training | Brand prior; recognition≫production; vendor comfort with teacher labels (S1) | Not identifiable; every signal is confounded (§6); the size-tension argument dissolved under the quantized-serving revision | **0.35** | P6 (error-sharing correlation) — low power, the only direct-ish API probe; not staged (needs external teacher APIs) |
| H7 | **MoE** (small active, larger total) | Cost point typical of 2026 small-active MoE serving; would explain the top of the capability band with less compute per token | Not required and no longer favored: the corrected §4 throughput bound is loose, a quantized dense ~4–9B is parsimonious, and nothing API-visible favors MoE | **0.35** (unconstrained directly; prior-driven) | None exists at this API surface; declare unconstrained |
| H8 | Non-transformer core (SSM/linear-attention/hybrid) | No quadratic signature to 29k (weak) | Population prior; capability profile is LM-typical; ΔR² test can't separate at these lengths | **0.10** (unconstrained) | Longer-context curvature probes are blocked by the 32k state cap; unconstrained |
| H9 | Headline capability materially inflated by **benchmark contamination** | MMLU-Pro/ARC are years public; 82.8 is strong for the band | HLE near-floor and ARC-AGI-2 zero-exact are contamination-resistant and weak; rotation audit shows content-driven answers; fresh generators 450/450 (easy, templated — weak) | **0.15** | A *hard* fresh-item suite (post-2024 exam material, private holdout) at MMLU-Pro difficulty — the only real test |
| H10 | Multiple heterogeneous models routed behind one endpoint (model field stable) | Nondeterminism is large-ish | Single stable `jev-1.13.0` across 482+ calls; tokenizer counts homogeneous; timing unimodal; batch numerics explain nondeterminism | **0.03** | Bimodality in upstream-latency or token-count distributions at n≫987 (none observed) |

---

## 10. Next-probe list (cheap, discriminating, harness-ready)

**Status (2026-09-26): P1–P5 EXECUTED LIVE.** The battery
(`scripts/benchmark/run_probe_battery2.py`, behind the repo's standard gate:
`JEVO_ALLOW_LIVE=1` + `TYPESAFE_API_KEY` from the environment, key never
stored) was dry-run-validated against planted ground truth, then dispatched
in full on 2026-09-25/26: **3,331 calls, 5,805,477 reported input tokens,
$0.2438** at the measured rate, 1 intentional HTTP 400 (the lone-surrogate
probe). Artifacts: `runs_archprobe/probe2/{probe2_rows.jsonl,
probe2_analysis.json, BILLING-probe2.json}`. Results are integrated in §2A
(P4), §2C (P1), §2D (P3), §2G (P5), and §4 (P2); per-family headlines:
byte-level fallback at ~1.0 tok/byte for rare blocks with UTF-8 validation
and NFC normalization (P1); option-count cost indistinguishable from zero
beyond the options' own tokens (P2); zero mass dilution from K=2→255 with
inert padding (P3); sums exactly 1.000 at K≤12 with identical options scoring
by position, not content (P4); knowledge horizon solid to Dec 2024, fading
Jan–May 2025, zero Jun–Aug 2025, all 16 fictional events refused (P5).
Actual tokens ran 2.35× the plan-mode estimate (chars/4 under-counted the
digit-string payloads; §11.21). P6 (teacher error-sharing) was **not
staged**: it needs external teacher APIs this environment does not hold, and
its power was always low (§6).

Costs at the measured $0.042/M input tokens, output free
(`data_report/costs.json`); all implementable with `src/jev_observatory/`
providers + `scripts/` patterns. Every probe should ship with ≥3 repeats per
config to sit above the TVD 0.03–0.12 noise floor where distributions are
compared, and with rotation controls where option order can matter.

**P1. Fallback-granularity and merge-boundary battery (tokenizer).**
*Separates:* H1's byte-level-BPE-fallback link vs codepoint/hybrid fallback
(§2C); maps the normalizer; tests whether any private-vocab structure is
recoverable beyond counts.
*Design:* (~250 noul calls, fixed template) (a) single rare codepoints per
block: uncovered BMP 3-byte chars, uncovered astral 4-byte chars, combining
sequences (base+diacritic, precomposed vs decomposed), lone-surrogate JSON
escapes (`\ud800`) to probe the parser/normalizer's internal encoding;
(b) merge-boundary contrasts: `ab`×n vs `a`×n+`b`×n vs `a`+`ba`×(n−1)+`b`;
case seams (`aA` vs `aa`); punctuation bridges (`word.word` vs `wordword`);
(c) normalizer map: tab, newline, NBSP, ZWJ, ZWNJ, BOM, ideographic space —
each as a run and as a single inter-word character.
*Expected signatures:* byte-level BPE → uncovered 3-byte chars at 2–3 tokens
with characteristic per-block variance; codepoint fallback → exactly 1;
UTF-16-code-unit internals → lone surrogates either rejected or counted 1–2
while astral chars stay ≤2. Merge-boundary contrasts fingerprint merge
*order* (which reference BPEs share), beyond what counts show.
*Cost:* ~250 calls × ~450 tokens ≈ 115k input tokens ≈ **$0.005**.

**P2. Option length × count decoupling (read-out mechanism).**
*Separates:* single-trunk in-context scoring of options (H1) vs any
per-option extra computation (a second encoder, per-option passes).
*Design:* 2-D grid: K ∈ {2, 8, 32, 128, 255} × option-text length ∈ {~2,
~8, ~32, ~64 tokens} (semantically inert but well-formed option strings;
fixed state), 3 repeats, randomized order; record upstream_ms and usage.
*Expected signatures:* single trunk → upstream explained by total input
tokens alone (per-cell residual ≈ 0, as in §2B at one length); per-option
compute → residual growing in K at fixed length, or superlinear in K×length.
*Cost:* 5×4×3 = 60 calls; largest cells ~20k tokens; total ≈ 400k tokens ≈
**$0.017**.

**P3. Calibration under option-count scaling (read-out semantics).**
*Separates:* softmax-over-the-set read-out (mass redistributes with K) vs
per-option independent scoring (p_gold roughly K-invariant); also tests the
choice analogue of the vendor's "judged independently" claim (S5) and the
phase-1 candidate-odds thread (`runs_live/FINDINGS.md` §3, untestable there
at 2-decimal quantization).
*Design:* 200 items with known gold (BoolQ-style or fresh synthetic), each
rendered at K ∈ {2, 4, 8, 16, 32, 64, 128, 255} by padding with matched
inert distractors; 2 repeats per (item, K); track p_gold(K), tail mass at
the 0.01/0.00 floors, IIA-style odds ratios between shared options.
*Expected signatures:* set-softmax → p_gold decays roughly geometrically in
K with distractor mass, floor-count grows ∝K; independent scoring → p_gold
flat, floors flat. IIA violations beyond the quantization floor are only
claimable at these n's (phase-1 n was too small).
*Cost:* 200×8×2 = 3,200 calls × ~600 tokens ≈ 1.9M tokens ≈ **$0.081**.

**P4. Logit-lattice forensics (post-processing pipeline).** *(offline arm
DONE — see §2A; live arm staged.)*
*Separates:* round-to-nearest vs floor/ceiling vs round-then-renormalize;
estimates internal noise scale; tests whether decision/table/confidence come
from one pre-quantization vector (§2A says: not exactly one).
*Design:* (a) 1,000 repeats of ~30 fixed near-tie requests (two semantically
equivalent options engineered to sit within a quantum) — flip rates vs
displayed gap bound internal noise; (b) reuse all published vectors: sum
deviation *sign* distribution (truncation biases sums low; renormalization
biases high; round-only is symmetric) — the current 43 deviating vectors are
too few, P3's 6,400 fresh vectors fix that; (c) confidence-vs-table residual
distribution at n≫2,289.
*Expected signatures:* symmetric small deviations + occasional 1-quantum
argmax flips → round-only display with separate decision path; systematic
sign → renormalization or truncation; flip rates at zero displayed gap →
internal precision scale.
*Cost:* ~1,000 repeats × ~400 tokens ≈ 0.4M + P3 reuse ≈ **$0.017**
standalone.

**P5. Knowledge-horizon bisection (training history / H5).**
*Separates:* fixed parametric cutoff (H1) vs rolling knowledge (retrieval,
silent refresh — H5).
*Design:* ~150 dated items Oct-2024 → Dec-2025 (event occurred / did not /
date shifted), each as forced-choice and with a "cannot say" option, 2
frames per item; items independently source-verified, held private until
run.
*Expected signatures:* parametric cutoff → accuracy cliff at one month,
abstention rising just before it (the election-item pattern of §2G
generalized); retrieval/refresh → no cliff, or a cliff that moves between
re-runs months apart.
*Cost:* 150×2×2 = 600 calls × ~350 tokens ≈ 210k tokens ≈ **$0.009**.

**P6. Error-sharing correlation with teacher-class models (H6; low power,
labeled as such).**
*Separates:* (weakly) "distilled from family X" vs "small model trained on
the same internet."
*Design:* freeze a 500-item public MCQ subset; collect Jev error sets +
probability orderings (direct protocol); compare against the same protocol
run on accessible teacher-class endpoints (GPT-4o-mini class, Haiku class)
and open ~9B models locally; measure excess error-sharing beyond the
base rate predicted by difficulty-matched controls (items where all small
models fail together).
*Expected signatures:* distillation from X → Jev's *confident wrong* answers
skew toward X's error set beyond base rate; shared-internet → error sharing
explained by item difficulty alone.
*Cost (Jev side):* 500 calls × ~600 tokens ≈ 0.3M ≈ **$0.013** (plus
external API costs, a few dollars). *Power caveat:* confounded by design;
report as suggestive-only regardless of outcome.

**Deliberately not proposed:** any MoE-vs-dense probe (no API-visible signal
exists at this noise floor, §5); any parameter-count-from-latency probe
(ground rule 4); long-context curvature beyond 32k (state cap blocks it).

Total program cost, executed: **$0.2438** on the Jev side (3,331 calls,
5.81M reported input tokens; plan-mode estimate was $0.10 over 2.47M — the
chars/4 payload estimator under-counted the digit-string option payloads,
§11.21). P6 remains unrun (external APIs required).

---

## 11. Uncertainty register — sub-quantum and sub-noise arguments

Every place we found an argument that hinges on differences below the 0.01
grid or inside the measured noise bands, flagged per ground rule 3 — in the
prior documents and in this analysis:

1. **"Typesafe pinned at 0.0103 vs the 0.010 floor"**
   (`analysis.json → ancestry`): the 0.0003 excess is a mean over 360
   quantized values and is *not* evidence that Typesafe sits above the floor
   as a single-call probability. Correct statement: **at the floor**. (Mean
   values over many quantized observations are legitimate; single-call
   sub-quantum differences are not.)
2. **Rotation-audit headline (86.2%) vs full-set (82.8%)**: reads as
   "shuffling helps" if unpaired. Paired, it is subset selection (native
   86.6% on the same 419 items; McNemar p=0.83; `arch_audits.json →
   rotation_pairs`). Do not cite the unpaired difference.
3. **Interim vs final prefill fits**: the 863-call interim log
   (`runs_archprobe-live.log`: floor 75.85, slope 4.932, R²=0.606) vs the
   987-call final (`analysis.json`: 73.11, 6.053, R²=0.857). The swing
   illustrates the ±30–50 ms per-call scatter; cite the final fit only, and
   never to more precision than ±0.5 ms/ktok.
4. **"Negligible quadratic term"**: our residual check bounds curvature at
   ~+4.8 ms mean at 28.9k tokens — *small*, not proven zero. The published
   phrasing is acceptable; "zero attention cost" would not be.
5. **HLE weighted 21.24 vs greedy 21.86**: a 0.6pp gap on 494 items is
   meaningful only as an average; per-item, it is floor-level. Quote as
   "weighted ≈ greedy at chance-level performance," not as a real
   calibration difference.
6. **Position-decile means** (first decile 0.0123 vs middle ~0.003,
   `token_talk_orderprobe.json`): each bin averages hundreds of quantized
   values, so the means are informative — but no single option's 0.01-vs-0.00
   is. The U-curve conclusion survives; any per-option reading does not.
7. **Ancestry family-mass tail ordering** (mistral 0.0052 vs amazon 0.0047
   vs nvidia 0.0020): sub-noise at these n's; do not rank the tail.
8. **Per-option latency residual −0.008 ms** (`arch_audits.json →
   marginal_cost`): negative "cost" is jitter; read as zero. Same for the
   +0.106 ms per-question residual: an *upper bound* on read-out cost, not a
   measured positive cost.
9. **MMLU log-loss 1.114**: finite only by pre-registered clipping
   (ε=1e-15); the unclipped fact is 213 zero-probability gold outcomes. Both
   numbers must travel together (they do in §2E; the report's benchmark
   section quotes neither, which is fine, but any future calibration chart
   must).
10. **"86 distinct levels"** (ARCHITECTURE-PROBES §2) is a property of the
    probe battery, not of the grid: the live raws show all 101 levels
    (`arch_audits.json → quantization`). Cite 86 only with its corpus.
11. **Confidence-formula residuals of 1–2 quanta** (62/2,289 choice vectors):
    consistent with pre-quantization p_max + separate table rounding, but we
    cannot exclude a slightly different internal statistic that agrees with
    the chance-corrected formula to within 2 quanta everywhere. The claim
    "confidence = chance-corrected p_max" is **[confident]** at display
    precision, not at full internal precision.
12. **Talk-program fluency claims**: everything in `product` mode is
    human-Mellum2-Jev collaboration (ground rule 8); only the Jev-only rungs
    (char menu garbage, word-menu salad, bare-token preference, END
    behavior, flat creative p_max≈0.2) are Jev evidence. This analysis uses
    only those.
13. **Vendor-advertised 250k tok/s vs measured 165k tok/s marginal**: different
    quantities (throughput under their load vs our marginal slope under
    shared batching); not a contradiction, not a confirmation.
14. **"18 malformed outputs … zero everywhere else"** (report pitch
    section): off by one — the rotation audit recorded 1
    (`runs_benchmark/bench-option_rotations-*/derived/score.json`). The
    correct total is 19 across the text-question stages; the generator has
    been fixed to compute it.
15. **ARC-AGI-2 `status_counts['strict_format_failure']` is definitional,
    not a violation count**: the score and task stages flag every item
    (517/517 and 496/496) because per-item strict format is undefined for
    multi-question chunks and the grid-level criterion is all-cells
    (`src/jev_observatory/arc_agi2_score.py`; stage `score.json →
    note_on_strict_format`). The contract-level fact is `cells_unusable = 0`
    in all three encodings. Anyone summing `strict_format_failure` across
    stages gets 1,032 instead of 19 — do not.
16. **Stale numbers in a published evidence page**: `docs/modern-comparison/
    architecture-evidence.html` shipped with mid-run prefill figures ("~5 ms
    per 1k / ~66 ms floor", the 863-call interim fit) and a caption asserting
    the *dissolved* Qwen tokenizer lead, while its fit line was drawn with
    swapped polyfit coefficients (off-canvas y2=-1709). All three fixed:
    `scripts/report/arch_evidence.py` (coefficient unpacking + computed
    captions) and the page regenerated from the final 987-call data. Cite
    only the regenerated page.
17. **Grid-total scope**: "27,610 values" (probe + phase-1 corpora) and
    "704,277 values" (adding the 4,490 Talk-ensemble raw vectors) are both
    correct zero-off-grid totals for their corpora; quote the corpus with
    the number. The one-sided bounded-sum result (§2A) needs the large-K
    ensemble corpus — the smaller corpora alone cannot see it.
18. **The §4 size estimate's assumption grid is external**: MFU 0.25–0.45,
    accelerator peak 250–500 TFLOPS bf16, shard count 1–4, and the
    2·N_active-FLOPs/token convention are industry-standard serving
    assumptions, not repo measurements — the band is only as good as they
    are, and it is labeled speculative everywhere it appears. The B=1
    (single-stream) choice is the conservative direction under multi-tenant
    batching (sharing lowers the per-stream bound). Anyone re-deriving §4
    should re-run `scripts/report/size_estimate.py` with their own grid
    rather than trusting ours.
19. **P5 pre-control label bug**: the "Starliner crew return" control was
    labeled Aug 2024; the event was Sept 11, 2024. Jev scored 0.00 on it —
    arguably correct. The item is excluded from horizon reasoning (disclosed
    in §2G); the post-2024 items, which carry the claim, are unaffected.
    Lesson: designer-verified event tables still need date double-checks.
20. **Degenerate menus are noise-dominated**: identical-option vectors vary
    widely across repeats (8 distinct patterns in 8 repeats at K=3; p_max
    0.68–0.89). The measured repeat-TVD band (0.034–0.122) comes from
    content-bearing flat menus; on degenerate ones the spread is larger. No
    conclusion may rest on a single degenerate call.
21. **Plan-vs-actual cost**: plan mode estimated 2.47M input tokens
    (chars/4); actual was 5.81M (server-reported). Digit-string payloads
    tokenize at ~1 token/char, not 4 chars/token. Future cost projections
    should use tokenizer-informed estimates; the $0.10 plan figure quoted
    earlier in this document's history underestimates by ~2.4×.
22. **P3 bucket sizes differ**: the K=2 bucket has n=132, not 396 — items
    with 3–4 base options cannot render at K=2 and are skipped. Per-K
    comparisons must carry their n.
23. **vals.ai protocol labels**: the platform page metadata now reads
    mode="one-shot" while the methodology text cited in `research.md`
    (2026-09-20) says 5-shot CoT. The expanded charts carry each row's own
    config fields (reasoning_effort/compute_effort) rather than a blanket
    label; the discrepancy is unresolved and flagged wherever vals rows are
    compared to Jev's direct protocol.
24. **The v1 matched sweep's scores were a parser artifact, not a capability
    measurement.** Strict exact-key parsing plus a 1,024-token output cap
    crushed exactly the models that think before answering (Qwen3.7-Flash
    burned its whole budget in the reasoning channel and returned empty
    content: 98.5% "format failures" on GPQA, yet ~100% accuracy on the
    answers it did emit) and the models that answer in prose (Gemma-3-4B on
    MATH: 92% rejected). Diagnosed with 216 raw-stored probe calls across
    three wire variants (`runs_matched_cheap/diag/`); the whole roster was
    re-run as v2 (temperature 0, thinking disabled where supported, 4,096-token
    cap, documented recovery parser applied identically to all models,
    unrecovered = wrong). Lesson: strict-format discipline measures
    instruction-following, not capability, and conflating them silently is
    how you publish a table of lies. Jev itself is unaffected - it has no
    text format to fail (18 contract-invalid responses in 12,032 MMLU calls,
    counted as wrong in its 82.8%).
25. **Public-leaderboard dependence has a coverage hole exactly where the
    Pareto story lives.** vals.ai carries 133 MMLU-Pro rows and not one
    Gemma/Llama/Granite/Nemo-class commodity model - the cheap end of the
    market had to be measured ourselves (v2 roster: 15 models, incl. MiMo
    v2.6 Pro/Flash and Qwen3.8-Max). Any chart sourced only from public
    boards inherits their roster biases; label the gap or fill it.
26. **Spec compliance cuts both ways - the v2 matched sweep was itself out
    of spec.** v2 fixed v1's parser artifacts but imposed temperature=0 and
    reasoning_effort=low on every model. For reasoning models that is not a
    neutral harness choice: GLM-5.3 documents default effort max for
    leaderboard reproduction, and at forced-low effort it scored BELOW its
    own flash sibling on MMLU (83.4 vs 85.5) and 13.6 points under the vals
    board's GLM-5.3 GPQA row, while looking ~25x cheaper - because it was
    made to think ~1k tokens where the published rows think tens of
    thousands. Owner audit caught the inversion. v3 re-runs every model with
    NO sampling or reasoning parameters (provider defaults = documented
    spec), generous bill-guard caps, and a cost-modeled dispatch order
    (cheap models first) so a budget stop can only truncate the expensive
    tail. Lesson recorded: "matched protocol" means matched items, format
    and parsing - never imposed sampling settings; run models at vendor
    defaults and let their billed thinking tokens count as cost, which is
    exactly what a Pareto frontier is for.

---

## 12. Reproduction

```bash
# secondary audits cited above (offline, published artifacts only):
.venv/bin/python scripts/report/arch_audits.py        # -> data_report/arch_audits.json
.venv/bin/python scripts/report/lattice_forensics.py  # -> data_report/lattice_forensics.json
.venv/bin/python scripts/report/size_estimate.py      # -> data_report/size_estimate.json

# reference leaderboards + charts (network for the fetch; no key needed):
.venv/bin/python scripts/report/fetch_vals_leaderboards.py  # -> docs/modern-comparison/canonical/vals-leaderboards-20260926.json
.venv/bin/python scripts/report/comparison_graphs.py        # -> docs/modern-comparison/comparison-graphs.html
.venv/bin/python scripts/report/pareto_graphs.py            # -> docs/modern-comparison/pareto-frontiers.html

# matched cheap-model baselines (OpenRouter key + live gate; executed 2026-09-26/27,
# 23,412 calls, $3.73, artifacts in runs_matched_cheap/):
.venv/bin/python scripts/benchmark/run_cheap_matched.py --plan   # roster + freeze (no key)
.venv/bin/python scripts/benchmark/run_cheap_matched.py --smoke  # 1 call/model, wire discovery
.venv/bin/python scripts/benchmark/run_cheap_matched.py --live   # full dispatch ($30 hard cap)
.venv/bin/python scripts/benchmark/run_cheap_matched.py --score  # summaries + Jev join (v1, superseded)

# matched cheap-model baselines v2 (the v1 scores were a strict-parse artifact;
# see register item 24; OpenRouter key + live gate; artifacts in runs_matched_cheap/v2/):
.venv/bin/python scripts/benchmark/diag_parse.py             # raw-response parser diagnostic
.venv/bin/python scripts/benchmark/run_cheap_matched2.py --plan
.venv/bin/python scripts/benchmark/run_cheap_matched2.py --smoke
.venv/bin/python scripts/benchmark/run_cheap_matched2.py --live
.venv/bin/python scripts/benchmark/run_cheap_matched2.py --score

# staged follow-up battery (offline modes are ungated and free):
.venv/bin/python scripts/benchmark/run_probe_battery2.py --plan     # -> call+cost plan
.venv/bin/python scripts/benchmark/run_probe_battery2.py --dry-run  # -> analyzer validation
# live dispatch (paid; needs key + opt-in; NOT run in this environment):
# JEVO_ALLOW_LIVE=1 TYPESAFE_API_KEY=... .venv/bin/python \
#   scripts/benchmark/run_probe_battery2.py --out runs_archprobe

# the primary probe batteries (live; require TYPESAFE_API_KEY + JEVO_ALLOW_LIVE=1):
JEVO_ALLOW_LIVE=1 .venv/bin/python scripts/benchmark/run_arch_probe.py --out runs_archprobe
.venv/bin/python scripts/benchmark/tokenizer_fingerprint.py --rows runs_archprobe/rows.jsonl
.venv/bin/python scripts/benchmark/tokenizer_perscript.py
.venv/bin/python scripts/benchmark/tokenizer_broadscan.py

# report page (reads every number from disk at render time):
.venv/bin/python scripts/report/arch_evidence.py     # -> docs/modern-comparison/architecture-evidence.html
.venv/bin/python scripts/report/arch_diagram.py       # -> docs/modern-comparison/architecture-diagram.html
.venv/bin/python scripts/report/build_report.py       # -> report/index.html
.venv/bin/python scripts/report/build_site.py         # -> jev-report/ bundle + gates
```

Primary artifacts: `runs_archprobe/` (latency/tokenizer/ancestry battery),
`runs_benchmark*/` (frozen capability suites), `runs_live/` (phase-1 +
Talk-to-Jev traces), `data_report/` (economics). Every figure quoted here is
in one of those paths or in `data_report/arch_audits.json`.
