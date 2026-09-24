# Modern-model reference matrix — provenance-first draft

**Status:** research-lane output (immediate lane of `docs/review-gate/MODERN-COMPARISON-BRIEF.md`). Read-only research; no live model calls, no image viewing, no credential access. Machine-readable companion: [`reference-records.json`](reference-records.json). All numbers below carry an evaluator label and a provenance class; **`—` means unavailable/unknown, never zero**. Dates are 2026 UTC unless noted.

## 1. Resolved model IDs (primary sources)

| Requested name | Exact API/model ID | Vendor | Release | Resolved from (primary) |
|---|---|---|---|---|
| GPT-6 Astra | `gpt-6-astra` | OpenAI | System card 2026-09-03; announcement Sept 2026 | OpenAI API docs + announcement + system card [1][2][3] |
| GPT-5.6 Sol | `gpt-5.6-sol` (alias `gpt-5.6` → Sol) | OpenAI | GA 2026-07-09 | OpenAI API docs + announcement [4][5] |
| Claude Fable 5.1 | `claude-fable-5-1` | Anthropic | Sept 2026 (successor to Fable 5) | Claude platform docs + announcement [6][7] |
| Claude Opus 5.0 | `claude-opus-5` | Anthropic | 2026-07-24, $5/$25 per 1M | Anthropic announcement + system card [8][9] |
| GLM-5.3 | `glm-5.3` (Z.ai API); HF `zai-org/GLM-5.3` | Z.ai (Zhipu) | Epoch: 2026-08-14 ⚠ weights-date discrepancy | HF README + Epoch AI [10][11] |
| GLM-5.3 Flash | `glm-5.3-flash`; HF `zai-org/GLM-5.3-Flash` (MIT, 320B/18B) | Z.ai | late Aug 2026 (AA post 08-26) | HF README + Z.ai blog [12][13] |
| Qwen3.8 Max | `qwen3.8-max`; snapshot `qwen3.8-max-0902` (alias `qwen3.8-max-2026-09-02`) | Alibaba | preview 07-19 (WAIC), GA 2026-08-03 (Epoch: 08-02) | Alibaba Cloud Model Studio + Epoch [14][15] |
| Qwen3.8 Flash | `qwen3.8-flash` (production; open-weights preview = `Qwen3.8-Flash-Next`, 125B/6B) | Alibaba | Flash-Next blog 2026-08-27 | Model Studio + HF [16][17][18] |
| DeepSeek Flash 0731 | API `deepseek-flash`; checkpoint `DeepSeek-V4-Flash-0731` (HF `deepseek-ai/DeepSeek-V4-Flash-0731`) | DeepSeek | 2026-07-31 | DeepSeek Change Log + HF README [19][20] |

⚠ ID ambiguities that must be pinned before any run:
- **DeepSeek**: Change Log states legacy names (`deepseek-v4-flash`, `-vision-exp`) are retired and such requests are now served by **DeepSeek-V4.1-Flash**. Confirm at runtime whether `deepseek-flash` serves the 0731 checkpoint or V4.1-Flash; capture a wire sample.
- **Qwen3.8 Flash vs Flash-Next**: production `qwen3.8-flash` and open preview `Qwen3.8-Flash-Next` are different artifacts; AA's 92.3% row is tagged ambiguously between them.
- **GLM-5.3 weights status**: Epoch says "Closed weights"; an HF repo with a custom `glm-5.3` license exists. Release date also differs (Aug 14 vs Aug 25).
- **Access gating**: GPT-6 Astra rollout began "to a limited set of organizations" — API availability to our account is unverified.

## 2. Benchmark datasets (protocol facts from primary cards)

- **MMLU-Pro** (TIGER-Lab dataset card): **12,032 questions** ("**Total** **12032** 6810 5222"), typically **10 options**, 14 subjects; official protocol 5-shot CoT. Jev-compatible: 10 options ≤ 255 Choice cap. [21]
- **GPQA Diamond** (Epoch AI): **198 questions**, 4 options, random baseline 25%. "It contains 198 questions, for which both domain expert annotators got the correct answers, but which the majority of non-domain experts answered incorrectly." Canonical dataset is gated (Idavidrein/gpqa) — lawful gated access required. [22]

## 3. Reference score matrix

Provenance classes: **V** = official vendor-reported · **I** = independent evaluation · **C** = non-canonical mirror (discovery lead only, re-verify on canonical page before quoting in the final report) · **—** = unavailable/unknown (do not fill).

### GPQA Diamond (198 questions, 4 options, accuracy %)

| Model | Vendor (V) | Artificial Analysis (I/C) | Vals AI (I) | Epoch (I) |
|---|---|---|---|---|
| GPT-6 Astra | **96.0** (max at any effort; OpenAI-run table) [2] | 96.1 (96.26 per mirror row) [23][24] | — | — |
| GPT-5.6 Sol | **94.6** [5] | 94.1 [24][25] | 95.2 [26] | — |
| Claude Fable 5.1 | 93.7 (as scored in OpenAI-run table) [2] | 93.7 [23][24] | — | — |
| Claude Opus 5 | 93.7 (OpenAI-run table) [2] | 93.74 [23][24] | — | — |
| GLM-5.3 | — (not in HF README table) | 91.7 (`max`) [24][27] | 88.1 [28] | 91 (rounded) [11] |
| GLM-5.3 Flash | — | 91.2 [24][29] | 86.364 [30] | — |
| Qwen3.8 Max | 92.6 ⚠ self-reported; column attribution unresolvable from text; tracker: table "published as an image" [31] | 92.7 (base) / 92.3 (0902) [24][25] | 93.7 [28] | 93 (base; rounded) [15] |
| Qwen3.8 Flash | 91.7 for **Flash-Next** open preview, column attribution caveat [18] | 92.3 [32] | — | — |
| DeepSeek Flash 0731 | — ("DeepSeek published neither") [33] | 90.8 [33] | — | — |

Exact vendor table text (GPT-6 Astra announcement, Academic section): *"GPQA Diamond 96.0% 94.6% 93.7% 92.6% 93.7% 95.3%"* for columns [GPT‑6 Astra, GPT‑5.6 Sol, Claude Fable 5.1, Claude Fable 5, Claude Opus 5, Gemini 3.8 Flash], with the protocol marker *"Evaluation scores are the maximum at any effort."* [2]

### MMLU-Pro (12,032 questions, 10 options, accuracy %)

| Model | Vendor (V) | Vals AI 5-shot CoT (I) | Artificial Analysis (I) |
|---|---|---|---|
| GPT-6 Astra | — (no MMLU row in announcement or system card) | — | — |
| GPT-5.6 Sol | — | 89.10 (`max reasoning`) [26] | — |
| Claude Fable 5.1 | — | **92.38** ("MMLU Pro(92.38%)") [34] | — |
| Claude Opus 5 | — | 91.59 (fallback-adjusted 91.58) [21][35] | — |
| GLM-5.3 | — | 86.8 [28] | — |
| GLM-5.3 Flash | — | 86.059 [30] | — |
| Qwen3.8 Max | — | 88.6 [28] | — |
| Qwen3.8 Flash | — | — | — |
| DeepSeek Flash 0731 | — | — (DataLearner "86.40, 17/134" exists but variant/protocol unverified — not used) | — |

Vals protocol (exact text, vals.ai MMLU-Pro methodology): *"All reported results use the 5-shot Chain-of-Thought prompting method, which included 5 examples per category in the prompt, as well as encouraging the models to think step by step. Few-shot CoT prompting is the approach used in the original paper."* [21]

### Key protocol caveats (exact source statements)

1. **No vendor publishes MMLU-Pro for these models.** Text search of the fetched OpenAI GPT‑5.6 announcement, GPT‑6 Astra announcement + system card, Anthropic Opus 5 announcement, GLM-5.3/-Flash HF READMEs, DeepSeek 0731 README, and Alibaba Cloud Model Studio pages found **no MMLU-Pro rows**. Every MMLU-Pro cell above is an independent-evaluator number.
2. **Evaluator divergence is first-order, not noise:** GLM-5.3 GPQA Diamond = 91.7 (AA) vs 88.1 (Vals) for the same model — a 7.2pp gap. GLM-5.3-Flash: 91.2 vs 86.364. **Never average across evaluators; label every cell.**
3. **Refusal-fallback sensitivity:** vals.ai GPQA key takeaways: *"The published 93.18% for Claude Fable 5 counts refusal-triggered fallbacks as successes. When those fallbacks are counted as failures instead, its score falls to 55.56% (166 of 396 tasks, 41.92%). The site provides a toggle to apply this correction."* Opus 5 MMLU-Pro 91.59 → 91.58 with fallbacks as failures. Our Jev runs record invalid outputs, so any external comparison must state the fallback policy per source.
4. **Effort settings matter and differ per vendor:** OpenAI `reasoning.effort` up to `max`; scores are "the maximum at any effort" in OpenAI tables. GLM-5.3 defaults `reasoning_effort=max` ("For benchmark and leaderboard reproduction, keep the default max"). Anthropic Opus 5: thinking on by default, disable only at effort ≤ high. Qwen thinking mode with 262,144-token CoT budget. Any matched run must pin and record the effort setting per model.
5. **Vals GPQA Diamond is archived** ("Since performance on this benchmark has saturated, we no longer run this benchmark on new model releases", updated 9/1/2026) — so Vals coverage of the newest models (Astra, Fable 5.1 GPQA) is missing by design.

## 4. Comparability verdict for Jev

- **Protocol-compatible external comparisons (import via `external.py` as `historical_protocol_compatible`):** GPQA Diamond (198 Q, 4 options) — AA and vendor rows; MMLU-Pro — Vals 5-shot CoT rows only (protocol stated; still different harness → protocol marker, not matched).
- **Not comparable without matched runs:** all fresh Jev mechanism tasks (graph/registry/calendar families — no published modern-model scores exist); any AA Intelligence Index composite; anything published only as an image (Qwen3.8-Max vendor table).
- **BoolQ paired 500-item result** (choice 0.898 / noul 0.896): contextual comparison only (`historical_contextual`); BoolQ literature numbers come from heterogeneous zero/few-shot and fine-tuned protocols.

## 5. Minimal matched-run plan (execution-design input; no live calls yet)

1. **GPQA Diamond, full 198 items, all nine requested models + Jev** — shared identical items; two separately reported conditions: direct-answer (no CoT) and reasoning-enabled (documented effort per model). n=198/model/condition → binomial SE ≈ 2.1pp at p≈0.9; paired McNemar on shared items detects ≥~5pp gaps, not 1–2pp frontier distinctions. Permuted option-order variant on a fixed 20–40 item subset, reported separately.
2. **MMLU-Pro full on Jev (12,032; already scheduled — do not substitute a pilot) + frozen stratified subset for modern references** — n=2,000 stratified across 14 subjects (~143/subject, ~0.67pp SE) as the frozen matched set; n=1,000 floor for ≥3pp claims. Permuted variants (existing `datasets/permutation.py`) on ~10%, reported separately. Invalid outputs, token budgets, and refusal/fallback behavior recorded; **no retries on wrong answers**.
3. **Weaker reference points** (GPT-5.6 Terra/Luna GPQA 92.9/92.3 vendor; Qwen3.7-Max Vals MMLU-Pro 89.3/GPQA 90.2) usable as tier anchors if a Flash-tier comparison needs interpolation — clearly labeled as non-requested models.

**Bounded cost estimate (formula only):** subset 2,000 items × ~1.2k in + 0.5–2k out tokens per model-condition; at listed prices ($1.40–10/M in, $0.28–50/M out) ≈ **$5–60 per model per condition**, dominated by reasoning-enabled output length; GPQA full ≈ 10× cheaper. Jev-side cost is not the constraint. Budget must be re-estimated from wire samples before authorization.

## 6. Access / credential blockers (honest list)

- **Credentials:** repo credential surface defines only `TYPESAFE_API_KEY`, `TYPESAFE_BASE_URL`, `JEVO_ALLOW_LIVE` (`src/jev_observatory/redact.py`). No baseline-provider key names (OPENAI/ANTHROPIC/ZAI/DASHSCOPE/DEEPSEEK etc.) observed in inspected code paths; existence of any baseline credentials unverified — an execution-lane blocker, not silently assumable.
- **Gated dataset:** GPQA canonical questions are access-controlled (Idavidrein/gpqa); lawful gated access must be obtained; do not reproduce restricted items.
- **Gated model access:** GPT-6 Astra limited rollout; Mythos 5.1 is Project-Glasswing-only (not requested). Qwen3.8-Max closed weights → API only.
- **Terms:** README finding 4 — Master Customer Agreement §2.3 restricts publication of benchmark/performance info; publishing the final comparison report needs that resolved first.
- **Canonical-page verification debt:** AA leaderboard pages are JS-rendered; mirror values (BenchLeader/BenchLM/LLMBoard/BenchmarkList) are labeled C and must be re-verified on `artificialanalysis.ai` before entering the final report tables.

## Sources

Primary (kept): OpenAI GPT-6 Astra announcement, GPT-5.6 announcement, GPT-6 Astra system card, API model docs (gpt-6-astra, gpt-5.6-sol); Anthropic Opus 5 announcement + system card, Fable 5.1 platform docs + announcement; HF READMEs zai-org/GLM-5.3, zai-org/GLM-5.3-Flash, deepseek-ai/DeepSeek-V4-Flash-0731, Qwen/Qwen3.8-Flash-Next; Alibaba Cloud Model Studio (qwen3.8-max, qwen3.8-flash); DeepSeek API Change Log; Epoch AI model pages (glm-5-3, qwen-3-8-max) + GPQA Diamond benchmark page; TIGER-Lab/MMLU-Pro dataset card; vals.ai MMLU-Pro and GPQA pages (methodology + key takeaways).
Secondary mirrors (labeled C, discovery leads only): benchleader.com, benchlm.ai, benchmarklist.com, llmboard.ai, gradually.ai, waitwhichmodel.fyi, ai-atlas.co.
Dropped: aimodelsnavi.com, themodelbeat.com, aitooltier.com, seawork.ai, fitmyllm.com, datalearner.com, llmlearner.com, llmpodium.com, rankllms.com, goml.io, kie.ai, tokenharbor.ai, dev.to (blog-quality aggregators; contradicted or unverifiable against primary text), and all image-based vendor tables (not viewable this phase).

### Source URLs

1. https://developers.openai.com/api/docs/models/gpt-6-astra
2. https://openai.com/index/gpt-6-astra/
3. https://deploymentsafety.openai.com/gpt-6-astra
4. https://developers.openai.com/api/docs/models/gpt-5.6-sol
5. https://openai.com/index/gpt-5-6/
6. https://platform.claude.com/docs/en/models/fable-5-1/overview
7. https://www.anthropic.com/claude-fable-and-mythos-5-1
8. https://www.anthropic.com/news/claude-opus-5
9. https://www-cdn.anthropic.com/b514064af1408018e64b1ad24e7d5e75850b4ffd/claude%20opus%205%20system%20card.pdf
10. https://huggingface.co/zai-org/GLM-5.3/raw/main/README.md
11. https://epoch.ai/models/glm-5-3
12. https://huggingface.co/zai-org/GLM-5.3-Flash/raw/main/README.md
13. https://z.ai/blog/glm-5.3-flash
14. https://www.alibabacloud.com/help/en/model-studio/qwen3-8-max
15. https://epoch.ai/models/qwen-3-8-max
16. https://www.alibabacloud.com/help/en/model-studio/qwen3-8-flash
17. https://www.alibabacloud.com/blog/qwen-3-8-flash-next-a-new-architecture-towards-ultimate-cost-efficiency_603501
18. https://huggingface.co/Qwen/Qwen3.8-Flash-Next
19. https://api-docs.deepseek.com/
20. https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/raw/main/README.md
21. https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro
22. https://epoch.ai/benchmarks/gpqa-diamond
23. https://www.benchleader.com/benchmarks/aa_gpqa (mirror)
24. https://www.llmboard.ai/benchmarks/aa-gpqa-diamond-2026-09-08-744 (mirror)
25. https://benchmarklist.com/models/qwen-qwen3.8-max/ (mirror)
26. https://benchlm.ai/benchmarks/valsmmlupro (mirror of vals.ai/benchmarks/mmlu_pro)
27. https://benchlm.ai/models/glm-5-3 (mirror)
28. https://benchlm.ai/compare/glm-5-3-vs-qwen3-8-max (mirror)
29. https://benchlm.ai/models/glm-5-3-flash (mirror)
30. https://www.gradually.ai/en/llm-comparison/glm-5.3-flash-vs-glm-5.2/ (mirror of Vals data)
31. https://www.waitwhichmodel.fyi/models/qwen3-8-max (tracker; quotes Alibaba image table)
32. https://www.ai-atlas.co/models/qwen3.8-flash (mirror)
33. https://www.waitwhichmodel.fyi/models/deepseek-v4-flash-0731 (tracker relaying AA measurements)
34. https://www.vals.ai/ (Fable 5.1 evaluation page text)
35. https://www.vals.ai/benchmarks/mmlu_pro (methodology + key takeaways)
36. https://www.vals.ai/benchmarks/gpqa (archived-status statement + methodology)