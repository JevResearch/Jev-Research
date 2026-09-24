# Canonical status — verified modern reference scores

**Status:** canonical verification lane output (read-only research; no live model calls, no images, no credentials). Companion data: [`verified-scores.json`](verified-scores.json). Input reviewed: [`../REFERENCE-MATRIX.md`](../REFERENCE-MATRIX.md), [`../reference-records.json`](../reference-records.json). Generated 2026-09-19.

## What changed vs. REFERENCE-MATRIX.md

1. **AA canonical pages are NOT fully JS-opaque.** `artificialanalysis.ai/evaluations/gpqa-diamond` embeds a complete JSON-LD `data[]` list of GPQA Diamond scores. All AA values can now be quoted from canonical text; the prior "AA is JS-rendered, verify before quoting" debt is **discharged** for GPQA Diamond.
2. **Vals canonical pages embed full SSR JSON.** `vals.ai/benchmarks/mmlu_pro` and `vals.ai/benchmarks/gpqa` contain per-model accuracy + stderr + effort config for 138 models. Vals MMLU-Pro now has **7 of 9 requested models** canonical-verified (was 6 mirrors + 1 partial). Vals GPQA has 7 of 9.
3. **Arithmetic fixed.**
   - GLM-5.3 evaluator gap: AA 91.72 vs Vals 88.132 = **3.6pp**, not the "7.2pp" written in the matrix (91.7 − 88.1 = 3.6). GLM-5.3-Flash gap: 91.21 vs 86.364 = **4.8pp**.
   - Binomial SE at n=2,000 is **p-dependent**: ≈0.8pp at p=0.85, ≈0.67pp at p=0.90. The matrix's universal "~0.67pp" was wrong as a constant. GPQA n=198 ≈ 2.1pp stands (at p≈0.9).
4. **Values corrected against canonical text** (mirrors were wrong):
   - Claude Opus 5 AA GPQA = **93.23%**, not 93.74 (93.74 is Fable 5.1; llmboard mirror misattributed).
   - Qwen3.8 Max AA GPQA (0902 snapshot) = **92.83%**; no "92.7 base" row on the canonical AA page (0803 slug deprecated).
   - GLM-5.3 Vals MMLU-Pro = **86.77** (±0.335), not "86.8" rounded from a mirror.
5. **New canonical rows found** (prior "unknown" gaps resolved):
   - DeepSeek V4 Flash 0731: Vals MMLU-Pro **86.206** (±0.339, effort high) and Vals GPQA **89.899** (±1.694).
   - Claude Fable 5.1: Vals GPQA **93.434** (±1.863) — the "archived, so no Fable 5.1 row" inference was wrong; the archived page still carries the row.
6. **Protocol compatibility (hard rule).** Vals MMLU-Pro is **5-shot CoT**; Vals GPQA "overall" is the **average of zero-shot CoT and few-shot CoT panels**; AA rows are **reasoning-enabled, effort-specific** (and some fallback-enabled). **Jev native direct-answer is NOT protocol-compatible with any of these.** External numbers may be imported only as `historical_protocol_compatible` with per-cell evaluator+protocol labels; the only protocol-compatible comparison vs. Jev direct is a matched frozen-subset run under one shared harness, with direct-answer and reasoning-enabled conditions reported separately.
7. **Language rule applied.** Absence of a row is recorded as **"not found on canonical page"** — never "no vendor publishes X" (we did not exhaustively audit all vendor pages; e.g., Anthropic's own announcement was not re-fetched this pass).
8. **Mirror-only cells stay unverified** (see `mirror_only_stays_unverified` in the JSON): Qwen3.8-Flash AA 92.3, Qwen3.8-Max vendor 92.6 (image table), Flash-Next vendor 91.7 (column caveat), Epoch 91, WaitWhichModel DeepSeek 90.8, DataLearner 86.40, llmboard 93.74.
9. **Additional finite-answer benchmarks (ARC-Challenge, classic MMLU, TruthfulQA): no canonical coverage found for any requested model** — only aggregator/mirror pages (pricepertoken, genailist, lmmarketcap, SOTA2) with pre-2026 or unattributed rows. No cells were filled from these.

## Verified score matrix (canonical only; % accuracy)

### GPQA Diamond (198 Q)

| Model | Vendor (OpenAI-run) | AA (canonical, config) | Vals (canonical, overall) |
|---|---|---|---|
| GPT-6 Astra | **96.0** | 96.06 (max) / 96.26 (xhigh) | not found (archived pre-release) |
| GPT-5.6 Sol | **94.6** | 94.14 (max) | 95.202 ± 1.074 |
| Claude Fable 5.1 | 93.7 (OpenAI-run) | 93.74 (max, with fallback) | 93.434 ± 1.863 |
| Claude Opus 5 | 93.7 (OpenAI-run) | **93.23** (max) | 93.434 ± 1.244 |
| GLM-5.3 | not found (HF README) | 91.72 (max) | 88.132 ± 1.851 |
| GLM-5.3 Flash | not found (HF README) | 91.21 | 86.364 ± 2.126 |
| Qwen3.8 Max | 92.6 ⚠ unverified (image) | **92.83** (0902 snapshot) | 93.686 ± 1.221 |
| Qwen3.8 Flash (Next) | 91.7 ⚠ column caveat | not found | not found |
| DeepSeek Flash 0731 | not found (HF README) | not found (slug deprecated) | 89.899 ± 1.694 |

### MMLU-Pro (12,032 Q, 10 options)

| Model | Vals (canonical, overall, 5-shot CoT) | Vendor | AA |
|---|---|---|---|
| Claude Fable 5.1 | **92.375** ± 0.266 (compute max) | not found | not found |
| Claude Opus 5 | **91.591** ± 0.276 (compute max; 91.58 w/ fallbacks as failures) | not found | not found |
| GPT-5.6 Sol | **89.1** ± 0.308 (effort max) | not found | not found |
| Qwen3.8 Max | **88.602** ± 0.313 | not found | not found |
| GLM-5.3 | **86.77** ± 0.335 (effort max) | not found (HF README) | not found |
| DeepSeek Flash 0731 | **86.206** ± 0.339 (effort high) — NEW | not found (HF README) | not found |
| GLM-5.3 Flash | **86.059** ± 0.341 (effort max) | not found | not found |
| GPT-6 Astra | not found (no Vals row; no vendor row in fetched text) | — | — |
| Qwen3.8 Flash | not found | not found | not found |

## Gaps remaining

- Qwen3.8-Max vendor GPQA 92.6: still bound only to an image-published table (tracker relay). Needs rendered-page/vendor PDF later phase.
- GPT-6 Astra MMLU-Pro and Qwen3.8-Flash rows: not found on any canonical page this pass.
- Anthropic/Alibaba/Z.ai/DeepSeek self-published GPQA or MMLU-Pro pages were not all re-fetched; "not found" is scoped to sources checked.
- AA MMLU-Pro canonical page not fetched this pass (no AA MMLU-Pro cell is used in the matrix).
- DataLearner DeepSeek MMLU-Pro 86.40 superseded by canonical Vals 86.206; do not import the DataLearner number.
