# Prompt: Determine the underlying architecture of "Jev" (best guess, evidence-ranked)

> Paste this prompt, with the repository below available to the model (clone it,
> or load it from https://github.com/JevResearch/Jev-Research — the full report
> is rendered at https://jevresearch.github.io/Jev-Research/). Everything you
> need is in the repo; the whole bundle is public domain (Unlicense).

## Your mission

Produce a **best-guess reconstruction of the underlying architecture and
training of `jev-1.13.0`**, TypeSafe AI's "system one" model — with confidence
tiers, an explicit alternative-hypotheses ledger, and a list of the cheapest
discriminating follow-up probes. You are the designated author of the report's
architecture section ("What Jev appears to be" in `report/index.html`, currently
a placeholder with an evidence inventory).

The vendor sells Jev as "frontier-level", "cannot hallucinate", built by a
"coauthor of ChatGPT", "incredibly fast/cheap, free output". Treat each as a
hypothesis to adjudicate against evidence, not as background lore. Vendor
marketing and reference material: `SOURCES.md`, `README.md`, `research.md`,
`DESIGN.md`.

## Ground rules (non-negotiable)

1. **Behavioral evidence only.** Nothing was reverse-engineered: no weights,
   gradients, or serving internals were accessed. Every observation is an
   API-visible signal — answers, probability vectors, usage token counts,
   server-compute timing headers, billing. Do not write any claim that implies
   internals were inspected.
2. **Every claim cites repo paths.** The numeric summary below is a *map*, not
   a source of truth: verify against the JSON before using it, and cite the
   file for every figure you state. Do not invent numbers; do not smooth over
   numbers that disagree with your preferred hypothesis.
3. **Quantization floor.** All returned probabilities land exactly on a 0.01
   grid (measured across ~11k+ values; `src/jev_observatory/validation.py`
   encodes the checks). Any argument that hinges on differences < 0.01 is
   reading noise; flag such arguments wherever you find them — including in
   your own draft.
4. **Latency slopes do NOT identify parameter count.** The server batches our
   requests with other tenants'; marginal cost per token is a scheduling
   artifact as much as a compute artifact. Use slopes for *structural*
   signatures (one forward pass vs decode loops), never for "Jev is N params".
5. **Self-reports are learned text, not provenance.** Jev's identity answers
   are training-data priors (see the identity probes below and `docs/`
   findings); the token-counting evidence contradicts the self-report. Never
   treat "I am OpenAI's model" as evidence of OpenAI lineage.
6. **Nondeterminism is measured, not assumed.** Repeats of identical requests
   show TVD 0.03–0.12 (`scripts/token_talk_orderprobe.py` outputs in
   `runs_live/token_talk_orderprobe.json`). Ordering effects 3–10× larger than
   that band are real; anything within it is not.
7. **Protocol mismatch in comparisons.** External leaderboard scores use other
   protocols (reasoning models, few-shot, tools). Our Jev numbers are one-shot,
   direct, no CoT. `docs/modern-comparison/canonical/comparable-scores.json`
   labels each fetched reference with its protocol; comparisons are
   positioning, not matched races.
8. **The generation experiments are confounded by design — two readings.**
   In the Talk-to-Jev program, character- and vocabulary-menu outputs are
   Jev-only (unguided, and they degenerate); token-menu outputs are a
   *collaboration with a local scorer model* (Mellum2) and partly that
   model's prior — including, as the findings document shows, the local
   model's preferences leaking into "agreement" through option position
   (native order = descending local probability; position boosts are large).
   Do not attribute token-mode fluency, identity, or "OpenAI" claims to Jev
   without checking `docs/token-talk-findings.md` §8, §9, §11, §14–§15 caveats; do
   attribute character/word-level degeneracies to Jev (those had no local
   model in the loop).

## Evidence map (what's in the repo, and what produced it)

### The API contract and harness
- `src/jev_observatory/schema.py` — the typed contract we coded against: one
  endpoint (`POST /v1/systemone`), a `state` payload with the question text,
  and per-question outputs of exactly three kinds — `choice` (≤255 options,
  returns a probability vector + argmax + confidence), `score` (2–10 ordinal
  levels, probability vector over levels), `noul` (a single scalar
  yes-likelihood). State ≤32k tokens, total ≤64k (vendor-documented).
- `src/jev_observatory/` — the whole harness (providers, transports, budgets,
  validation, redaction, ledger/manifests). `validation.py` is the formal
  statement of response-shape expectations (argmax-vs-probability consistency,
  sum-to-1, the 0.01 grid, `confidence` required for choice/score).

### Capability measurements (three frozen suites; 16,379 planned requests)
- `runs_benchmark*/bench-*/derived/score.json` — greedy accuracy, Wilson CIs,
  Brier / log-loss, calibration summaries, strict-format failure counts, and
  per-stage `attempts_summary` (actual server-reported token usage).
- `runs_benchmark*/bench-*/derived/weighted_score.json` — the average
  probability Jev itself put on the correct option, per stage.
- `runs_benchmark*/freeze/{frozen.json,preparation.json}` — SHA-256 pinning of
  every input and request plan (item text is licensed and withheld; the hashes
  let dataset holders verify identity).
- Headline numbers to verify and build on (not to trust blindly):
  MMLU-Pro 12,032 items **82.8%** (option-rotation audit 420 items, 86.2%);
  ARC-Challenge 1,172 items **97.9%**; GPQA Diamond 196 items **76.5%**;
  HLE multiple-choice subset 494 items **21.9%**; MATH-500 encodable subsets:
  **83.1%** picking the right MCQ option vs **13.6%** reading the answer off a
  per-digit rubric (the "generation tax" contrast — same knowledge, different
  output channel); ARC-AGI-2 public eval: **53.5%** per-cell diagnostic vs
  **0** exact grids.
- Weighted < greedy accuracy on nearly every stage (`docs/modern-comparison/*.html`
  charts; `scripts/report/comparison_graphs.py`) — over-dispersed
  distributions relative to their own calibration.

### Latency / self-batching / serving structure
- `scripts/benchmark/run_arch_probe.py` produced `runs_archprobe/`:
  987 calls, 0 errors (`BILLING.json`), per-call rows with server-compute
  headers in `rows.jsonl`.
- `runs_archprobe/analysis.json` — prefill fit ≈ **73 ms fixed floor +
  ~6.0 ms per 1k input tokens**, negligible quadratic term (R² ≈ 0.86);
  packing +1 question into the same request costs **+0.44 ms** compute while
  adding ~33 serialized output tokens; +1 option costs +0.10 ms / ~9.6 tokens;
  cold TLS +353 ms; **upstream compute stays flat (74→82 ms) from
  concurrency 1→32** while client wall time bends.
- `runs_archprobe/cleanrun_jev.json` — the mergerate family: whitespace and
  formatting behavior probes (see tokenization below).
- Reading: one forward pass over the shared state per request, *many* read-out
  vectors serialized per question×option — consistent with a read-out head
  replacing autoregression, and with continuous batching on the serving side.

### Tokenization fingerprint (the strongest single-architecture signal here)
- Produced by `scripts/benchmark/tokenizer_*.py`; results in
  `runs_archprobe/tokenizer_{fingerprint,perscript,merge_score,broadscan}.json`
  and `tokens_per_char{,_compare}.json`. Method: the API's reported
  `input_tokens` is an instrument, not a self-report; we fed controlled
  strings and fit reported counts against reference tokenizers.
- First result was a **false Qwen2.5 clue** (slope 1.001, RMSE ~1 token over
  59 strings) that dissolved under the template-free test: the fixed chat
  template contributed a constant offset (documented walk-through in
  `docs/modern-comparison/ARCHITECTURE-PROBES.md` — a cautionary tale to
  include in your methods section).
- Decisive table (`tokens_per_char_compare.json`, whitespace-free per-script
  samples, marginal tokens/character vs 11 reference tokenizers incl.
  Qwen2.5, Llama, Mistral, Gemma, GLM, DeepSeek, Hy-MT2, cl100k, o200k,
  gpt-oss): Jev merges Latin/punctuation heavily but spends **~0.92–0.96
  tokens per codepoint on every non-Latin script tested** (Cyrillic, Chinese,
  Korean, Greek, Arabic, Hebrew, Thai, Devanagari), ~0.94 per digit, astral
  emoji ≈ 2. Qwen/o200k/cl100k merge CJK and Cyrillic aggressively (0.34–0.68)
  — they do *not* match.
- `tokenizer_perscript.json`: affine residual fits across ~128 distinct
  open vocabularies; the leader is at RMSE ~20 and nobody is close.
  `tokenizer_broadscan.json`: 1,215 HF repos scanned → 173 unique tokenizer
  signatures → no exact match.
- **Pure whitespace collapses to zero tokens** (`cleanrun_jev.json` /
  mergerate) — the server runs a *pre-tokenizer normalizer*, which is itself
  an architectural fact: observed token statistics partly describe the
  pipeline, not only the model.

### Identity / ancestry (forced choice, ordering-controlled)
- `runs_archprobe/analysis.json → ancestry`: 15 frames × 24 cyclic option
  orderings, 360 calls. Probability mass by family: **openai 0.4803**,
  qwen 0.112, **typesafe 0.1031**, anthropic 0.0871, google 0.0548,
  deepseek 0.0463, …; greedy votes: OpenAI 293/360, Jev 6, Qwen 15. Mean
  p("Jev") 0.0648; p("Typesafe") pinned at the 0.010 floor. The chosen name
  landed at index 0 only 4.7% of the time — *below* uniform, so the result is
  content, not position.
- Independent corroboration in the generation program
  (`docs/token-talk-findings.md` §5 item 2, §9, §15; traces
  `runs_live/identity_parity_*.json`): brand prior is OpenAI-shaped; even with
  "Jev"/"Typesafe" spellings injected at probability-parity with model-name
  tokens, Jev did not choose them.
- Phase-1 probes (earlier session, raws withheld locally; aggregates in
  `runs_live/FINDINGS.md`): identity multiple-choice leans "unknown"/"openai",
  never "typesafe"; accepts a fictional-origin prompt at 0.97 —
  prompt-suggestible branding, explicitly *not* ancestry evidence.

### Generation / "Talk-to-Jev" program (distribution-quality at sequential decisions)
- Full write-up: `docs/jev-talk-program-report.md` + incremental findings
  `docs/token-talk-findings.md` (§1–§15); ~22k calls of traces in
  `runs_live/*.json`; code in `src/jev_observatory/{talk,token_talk,char_talk,word_talk,vocab,local_lm}.py`;
  probe scripts `scripts/token_talk_*.py`.
- Protocol ladder and what each level isolates:
  - *Character menu* (98 options): degenerate under every control (floor
    subtraction, temperature, nucleus, run bans) — `Geeee`, `A    `; per-char
    distributions are intrinsically weak. This is Jev alone.
  - *Vocabulary menu* (letters/combos/whole words built from English
    frequency statistics, stuck-escape, backspace): 86–100% real words, zero
    syntax ("They areas s aren'ts area aren't arenas are s"). Jev alone.
  - *Token menu from a local LM* + Jev re-ranking: coherent paragraphs appear
    only in `product` (p_local × p_jev) mode, and only partly because option
    position (primacy/recency U-curve, `runs_live/token_talk_orderprobe.json`)
    broadcasts the local prior into Jev's choice. Treat as collaboration.
- Behavioral regularities established at token level (all with traces):
  candidate-local **whitespace/surface-form scoring** (bare-token preference,
  article drops), **END is used well** (semantic stopping on short answers;
  absent in long-form), **flat creative-mode distributions** (p_max ≈ 0.2),
  sharp short-step distributions, order-sensitivity of argmax identity, and
  the brand prior above.

### Cost / economics (context for "why is it this shape")
- `data_report/costs.json`, `data_report/billing_totals.json` (+
  `scripts/report/cost_model.py`, `aggregate_billing.py`): measured
  **$0.042/M input tokens, output free** (23,459 recorded calls, $3.58 total
  program cost; MMLU-Pro full 12k run measured at ~$0.28; GPQA per-question
  ~$0.0044). The prefill-only economics are *measured* (no decode loop), not
  inferred from price.
- External comparison costs in the Pareto charts are launch-era list-price
  estimates calibrated to measured anchors (method in
  `docs/modern-comparison/pareto-frontiers.html` + footnote in the report).

## The candidate hypotheses (adjudicate; add your own)

1. **Small (~sub-frontier) new foundation, English/Latin-centric BPE-ish
   tokenizer with codepoint fallback for other scripts, post-trained** for
   instruction-following on assistant/brand-saturated data, **whose generation
   head has been replaced by option-scoring read-outs** (the prefill-only,
   self-batched, one-forward-per-request picture). Working hypothesis of the
   project; the tokenizer result is its main load-bearing evidence.
2. **Relabeled open model** (the "coauthor of ChatGPT" marketing invites an
   OpenAI reading; the early token fit invited Qwen). What rules in/out each?
   (Note: tokenizer broadscan found no match among 173 open signatures;
   identity is a prior; the Qwen clue was an artifact — but argue, don't
   assert, that open vocabularies are excluded.)
3. **Distillation from a frontier teacher** (would explain brand priors +
   mid-tier capability + weirdly strong self-batching efficiency?); what
   evidence would distinguish distillation from on-policy training here — and
   can any API-visible signal do it at all?
4. **MoE vs dense** — does anything constrain this (activation-visible in
   latency/price; token counts cannot)? 5. **Retrieval-augmented or cached
   answering** — which results bear on it (e.g., temporal-knowledge behavior
   in `runs_live/FINDINGS.md`, abstention, exact-quantization of a
   discriminative head)?
6. The vendor's own framing: is "system one" (fast discriminative judgement,
   no generation) a *training* claim or an *interface* claim under your best
   architecture guess? What does "cannot hallucinate" mean given a read-out
   head, and what does the measured error rate (GPQA ~23.5% wrong,
   format-valid every time) license saying?

## Deliverables

1. **`ARCHITECTURE-ANALYSIS.md`** — the full reconstruction: a best-guess
   architecture card (size band *without* parameter claims, tokenizer profile,
   pre/post-training inferences, serving shape), each claim tagged
   **[confident / plausible / speculative]** with file-path citations and the
   specific statistic supporting it; an explicit "what would falsify this"
   line per load-bearing claim.
2. **Alternative-hypotheses ledger** — for hypotheses 1–6 above: evidence
   for, evidence against, current probability, and what single measurement
   would move it most. Include hypotheses *outside* the list if your reading
   of the data supports them.
3. **Report-ready prose** replacing the placeholder section in
   `report/index.html` (see its current voice in the published page): one
   up-front working picture, then the evidence-inventory structure it already
   lists, honoring every ground rule above. Match the report's tone: deadpan,
   numbers-on-the-table, no overclaiming, warm at the end. It must be safe to
   render as the report's central inference — the marketing-claim teardown
   already handles the rest.
4. **Next-probe list** — ≤6 *cheap* discriminating probes implementable with
   the existing harness (`src/jev_observatory/` + `scripts/`), each with:
   hypothesis discriminated, expected signature under each rival, and a cost
   estimate derived from measured $0.042/M input tokens. Good starting points
   if they survive your scrutiny: cross-tokenizer *sequence* probes (e.g.
   merge-boundary behavior on repeated substrings to fingerprint merges
   beyond counts), calibration under option-count scaling (already a phase-1
   thread in `runs_live/FINDINGS.md`), and logit-lattice forensics (the 0.01
   grid is a read-out *precision* fact — can its rounding direction leak
   pre-quantization comparisons?).

Work from the artifacts, argue from the statistics, and make the uncertainty
as informative as the conclusions.
