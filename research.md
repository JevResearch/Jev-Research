# Research: Canonical verification of modern requested-model reference scores (GPT-6 Astra … DeepSeek Flash 0731)

**Companion artifacts written by this task:**
- `docs/modern-comparison/canonical/verified-scores.json` — verified score records with exact supporting text + URL per value
- `docs/modern-comparison/canonical/CANONICAL-STATUS.md` — canonical status brief + corrections list

## Summary
All matrix scores were re-verified against canonical evaluator/vendor text: vals.ai (full SSR JSON on its MMLU-Pro and GPQA pages), artificialanalysis.ai (complete JSON-LD `data[]` GPQA Diamond list), and the OpenAI announcement tables. This fixed two wrong numbers propagated from mirrors (AA Opus 5 is 93.23 not 93.74; AA Qwen3.8-Max 0902 is 92.83 not 92.3/92.7), filled two "unknown" gaps with canonical rows (DeepSeek V4-Flash-0731 Vals MMLU-Pro 86.206 and Vals GPQA 89.899; Fable 5.1 Vals GPQA 93.434), corrected the arithmetic (91.7−88.1 = **3.6pp**, not 7.2pp) and the statistics (n=2,000 SE ≈ **0.8pp at p=0.85**, 0.67pp only at p≈0.90), and established that **Vals 5-shot CoT / AA reasoning-enabled rows are NOT protocol-compatible with Jev native direct-answer runs**. ARC-Challenge, classic MMLU, and TruthfulQA: no canonical coverage found for any requested model; cells stay empty.

## Findings

1. **AA canonical GPQA page is fully extractable.** `artificialanalysis.ai/evaluations/gpqa-diamond` embeds a JSON-LD `data[]` list ("Independent test run by Artificial Analysis on dedicated hardware") plus highlight text: "GPT-6 Astra (xhigh) scores the highest on GPQA with a score of 96.3%, followed by GPT-6 Astra (max) with a score of 96.1% and Gemini 3.8 Flash (high) with a score of 95.3%." Canonical AA values: Astra xhigh 96.26 / max 96.06 / high 94.95; Sol (max) 94.14; Fable 5.1 (max with fallback) 93.74; Opus 5 (max) 93.23; Qwen3.8 Max (0902) 92.83; GLM-5.3 (max) 91.72; GLM-5.3-Flash 91.21. [Source](https://artificialanalysis.ai/evaluations/gpqa-diamond)

2. **Mirror corrections.** llmboard's "Claude Opus 5 93.74%" was a misattribution — canonical AA is **93.23** (93.74 is Fable 5.1). BenchLeader's "92.3 (0902)" is wrong — canonical AA carries **92.83** for the 0902 snapshot and marks the 0803 base slug deprecated (no base row). Matrix GLM-5.3 Vals MMLU-Pro "86.8" is **86.77** ±0.335.

3. **Vals canonical pages embed complete SSR JSON.** Per-model accuracy, stderr, effort, temperature, provider: MMLU-Pro overall — Fable 5.1 92.375±0.266 (compute max), Opus 5 91.591±0.276 (compute max), GPT-5.6 Sol 89.1±0.308 (effort max), Qwen3.8-Max 88.602±0.313, GLM-5.3 86.77±0.335 (effort max), DeepSeek V4-Flash-0731 86.206±0.339 (effort high), GLM-5.3-Flash 86.059±0.341 (effort max). GPQA overall — Sol 95.202±1.074, Qwen3.8-Max 93.686±1.221, Fable 5.1 93.434±1.863, Opus 5 93.434±1.244, DeepSeek 89.899±1.694, GLM-5.3 88.132±1.851, GLM-5.3-Flash 86.364±2.126. [Sources](https://www.vals.ai/benchmarks/mmlu_pro), [GPQA](https://www.vals.ai/benchmarks/gpqa)

4. **Gap-fill: DeepSeek V4 Flash 0731 and Fable 5.1 Vals rows exist.** The prior gaps ("DeepSeek MMLU-Pro/GPQA unknown"; "no Fable 5.1 GPQA by design") are resolved by canonical rows above. Vals GPQA is archived (updated 9/1/2026: "Since performance on this benchmark has saturated, we no longer run this benchmark on new model releases") but preserves rows — only GPT-6 Astra (announced 9/3) has no row.

5. **"not found" cells (canonical absence, not "no vendor publishes"):** GPT-6 Astra MMLU-Pro — absent from Vals page data and no MMLU row in the fetched OpenAI Astra announcement text. Qwen3.8-Flash — no Vals/AA/vendor row found. DeepSeek/Qwen-Flash AA GPQA — not in canonical AA `data[]` (DeepSeek slug `deprecated:true`; WaitWhichModel's 90.8 unverifiable). GLM-5.3/-Flash vendor GPQA — absent from HF README tables (unchanged).

6. **Vendor GPQA verified from OpenAI announcement text.** Astra announcement Academic row: "GPQA Diamond 96.0% 94.6% 93.7% 92.6% 93.7% 95.3%" → [Astra, Sol, Fable 5.1, Fable 5, Opus 5, Gemini 3.8 Flash]. GPT-5.6 announcement: "GPQA Diamond 94.6% 92.9% 92.3% …" → Sol 94.6, Terra 92.9, Luna 92.3. [Sources](https://openai.com/index/gpt-6-astra/), [GPT-5.6](https://openai.com/index/gpt-5-6/)

7. **Arithmetic/statistics fixed.** GLM-5.3 AA-vs-Vals gap = 91.72 − 88.132 = **3.6pp** (matrix said "7.2pp" from the same numbers — wrong). GLM-5.3-Flash gap = **4.8pp**. SE at n=2,000: √(0.85·0.15/2000) = **0.80pp**; 0.67pp corresponds to p=0.90 — quote SE at the expected p per model, never as a universal constant. GPQA n=198 SE ≈ 2.1pp at p≈0.9 (matrix value correct).

8. **Protocol incompatibility is decisive.** Vals MMLU-Pro methodology (canonical text): "All reported results use the 5-shot Chain-of-Thought prompting method, which included 5 examples per category in the prompt…" Vals GPQA: "two evaluation approaches … Zero-shot chain-of-thought … Few-shot chain-of-thought", with overall = derived average of the two panels; AA rows are reasoning-enabled effort-specific configs (several fallback-enabled). **Jev native direct-answer runs match none of these protocols** — external cells may be cited only as `historical_protocol_compatible` context with per-cell evaluator/protocol/effort/fallback labels; matched comparisons require the shared-harness frozen-subset runs already planned, with direct and reasoning conditions reported separately.

9. **Additional finite-answer benchmarks: not found.** ARC-Challenge: canonical AI2-derived sources surface only pre-2026 models (Nemotron-H 56B 94.97, Llama 3.1 405B 96.9, GPT-4o 96.4 …); aggregator pages (pricepertoken, genailist, lmmarketcap) list no requested-model rows usable as verified references. Classic MMLU and TruthfulQA (MC1/MC2): no canonical rows found for any requested model. No cells were filled from these sources.

10. **Mirror-only cells stay flagged unverified** (Qwen3.8-Flash AA 92.3; Qwen3.8-Max vendor 92.6 image table; Flash-Next vendor 91.7 column-caveat; Epoch 91 rounded; WaitWhichModel DeepSeek 90.8; llmboard 93.74 contradicted; DataLearner 86.40 superseded by canonical Vals 86.206). No fabricated n/split anywhere: n=198 GPQA, n=12,032 MMLU-Pro, Vals panel sizes not republished by Vals text (no invented n).

## Sources
- Kept: vals.ai/benchmarks/mmlu_pro + /benchmarks/gpqa + /models pages (canonical evaluator SSR JSON with stderr/configs); artificialanalysis.ai/evaluations/gpqa-diamond (canonical AA JSON-LD data list); openai.com GPT-6 Astra + GPT-5.6 announcements (canonical vendor tables); benchlm.ai/benchleaders mirrors (only to locate values later confirmed/contradicted on canonical pages).
- Dropped: llmboard.ai Opus-5 93.74 (contradicted by canonical AA); benchleader "92.3 (0902)" (contradicted by canonical 92.83); DataLearner DeepSeek 86.40 (unverified variant; superseded); pricepertoken/genailist/lmmarketcap/SOTA2 ARC/TruthfulQA pages (no requested-model rows, aggregator quality); aimodelsnavi-style aggregators (per matrix, already dropped).

## Gaps
- Qwen3.8-Max vendor GPQA 92.6 still bound to an image-published table (needs later rendered-page/vendor-PDF phase).
- GPT-6 Astra MMLU-Pro and Qwen3.8-Flash: not found on any canonical page this pass.
- AA MMLU-Pro canonical page not fetched (no AA MMLU-Pro cell is used).
- Anthropic/Alibaba/Z.ai self-published benchmark pages not re-fetched; "not found" claims are scoped to sources checked (language kept as "not found", not "no vendor publishes").
- Exact Vals per-panel task counts for GPQA subpanels not republished in page text; no n invented for them.

## Supervisor coordination
No decision needed: the only conflict (task output path vs. runtime override) is resolved by writing both — canonical artifacts under `docs/modern-comparison/canonical/` and this brief at the authoritative `research.md` path. No blockers for the review lane.
