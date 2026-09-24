# Comparable published scores — where Jev slots in (fetched 2026-09-20/21)

**No model was benchmarked by us.** Every number below is a published value,
fetched from the cited source and labeled with its protocol. Jev's own runs
are direct-answer (no CoT, no tools, pass@1) — so published reasoning/CoT rows
are *historical_contextual*, not protocol-matched comparisons. Aggregator-
sourced values are flagged. Machine-readable: `canonical/comparable-scores.json`.

## Summary — where Jev slots (greedy)

- **MMLU-Pro (82.8)** — between Claude 3.7 Sonnet no-thinking (80.7) and Qwen 3.5 9B (82.5); below the entire de-minimis nine that have rows (86.1-92.4)
- **GPQA Diamond (76.5)** — ≈ Claude 3.7 Sonnet no-thinking (76.8), just under Qwen 3.5 9B (77.6); above Gemma 4 E4B (58.6), GPT-4o (54.3), Llama 3.1 8B (~27-31)
- **ARC-Challenge (97.9)** — at/above every published row found (best: Llama 3.1 405B 96.9, GPT-4o 96.7)
- **MATH-500 (83.1 MCQ-adapted / 13.6 digit)** — no protocol-compatible row exists (free-form generation benchmark); context: between no-reasoning rows (GPT-4o 75-89, Llama 3.1 405B 71) and thinking rows (Claude 3.7 91.6-96.2+)
- **ARC-AGI-2 (per-cell 53.5-61.9; exact 0)** — encoding-incomparable by design; exact-grid context: 2025-era base LLMs 0-13.6, GPT-5.2 53.5, current frontier 84.6-95.0
- **HLE text-only MC (21.9)** — chance ~17%; published full-subset rows (reasoning): GPT-5 26.3, DeepSeek-R1-0528 14.0 — Jev is in that band on a harder-scoring sub-track caveat

## MMLU-Pro

*Jev context: Jev 82.8% sits between Claude 3.7 Sonnet (no-thinking, 80.7) and Qwen 3.5 9B (82.5); below every de-minimis model that has a published row*

| Model | Score | Source type | Protocol | Source / note |
|---|---|---|---|---|
| GPT-6 Astra | not found |  |  |  — absent from Vals MMLU-Pro page and no vendor MMLU row (canonical absence, verified 2026-09-19 and re-checked 2026-09-21) |
| GPT-5.6 Sol | 89.1% | canonical | cot_5shot | [canonical](https://www.vals.ai/benchmarks/mmlu_pro) |
| Claude Mythos 5.1 | not found |  |  |  — no published knowledge-benchmark rows; Anthropic system-card tables cover agentic/computer-use only (sibling Fable 5.1: 92.38 Vals) |
| Claude Fable 5.1 (sibling reference) | 92.4% | canonical | cot_5shot | [canonical](https://www.vals.ai/benchmarks/mmlu_pro) |
| Claude Opus 5 | 91.6% | canonical | cot_5shot | [canonical](https://www.vals.ai/benchmarks/mmlu_pro) |
| GLM-5.3 | 86.8% | canonical | cot_5shot | [canonical](https://www.vals.ai/benchmarks/mmlu_pro) |
| Qwen3.8 Max | 88.6% | canonical | cot_5shot | [canonical](https://www.vals.ai/benchmarks/mmlu_pro) |
| GLM-5.3 Flash | 86.1% | canonical | cot_5shot | [canonical](https://www.vals.ai/benchmarks/mmlu_pro) |
| DeepSeek V4 Flash 0731 | 86.2% | canonical | cot_5shot | [canonical](https://www.vals.ai/benchmarks/mmlu_pro) |
| Qwen3.8 Flash | not found |  |  |  — no MMLU-Pro row for plain Flash; Qwen3.8-FLASH-NEXT (different variant) self-reports 73.23 (vendor, datacamp.com/blog/qwen3-8-flash-next) |
| Qwen3.8-27B | 84.3% | aggregator | cot_5shot | [aggregator](https://benchlm.ai/models/qwen3-8-27b) — aggregator-imported Vals row; independent quant-harness measurement 76.7 (oracomputing/Qwen3.8-27B-OQ3-GGUF) |
| Qwen 3.5 9B | 82.5% | aggregator | reasoning | [aggregator](https://llmrun.dev/benchmark/mmlu-pro) |
| Claude 3.7 Sonnet (no thinking) | 80.7% | canonical | direct | [canonical](https://www.benchleader.com/models/claude-3-7-sonnet) — Vals no-reasoning row; thinking 82.7 |
| DeepSeek-V3 (2024-12) | 75.9% | aggregator | reasoning | [aggregator](https://codeswap.net/benchmark/mmlu-pro/) |
| GPT-4o (2024-08-06) | 74.7% | aggregator | cot_5shot | [aggregator](https://ai-stats.phaseo.app/benchmarks/mmlu-pro) |
| GPT-4o (2024-05-13, TIGER-Lab official) | 72.5% | canonical | cot_5shot | [canonical](https://github.com/TIGER-AI-Lab/MMLU-Pro) |
| Gemma 4 E4B | 69.4% | self_reported | reasoning | [self_reported](https://www.reddit.com/r/LocalLLaMA/comments/1sbp8ny/gemma_4_vs_qwen_35_benchmark_comparison/) — vendor comparison table (Qwen3.5 vs Gemma 4 rows) |
| Llama 3.1 8B | 48.3% | aggregator | cot_5shot | [aggregator](https://modelbeats.com/models/llama-3-1-8b) — Meta official 5-shot CoT micro: 47.0 post-trained (meta-llama/llama-models eval_details.md) |

## GPQA Diamond

*Jev context: Jev 76.5% is statistically at Claude 3.7 Sonnet no-thinking (76.8) and just under Qwen 3.5 9B (77.6); far above GPT-4o (54.3), Gemma 4 E4B (58.6), Llama 3.1 8B (~27-31); far below the de-minimis nine (88-96)*

| Model | Score | Source type | Protocol | Source / note |
|---|---|---|---|---|
| GPT-6 Astra | 96.3% | canonical | reasoning | [canonical](https://artificialanalysis.ai/evaluations/gpqa-diamond) — vendor launch 96.0 (openai.com/index/gpt-6-astra/); AA xhigh 96.26 / max 96.06 |
| GPT-5.6 Sol | 95.2% | canonical | reasoning | [canonical](https://www.vals.ai/benchmarks/gpqa) — AA 94.9; vendor 94.6 |
| Claude Mythos 5.1 | not found |  |  |  |
| Claude Fable 5.1 (sibling reference) | 93.7% | canonical | reasoning | [canonical](https://artificialanalysis.ai/evaluations/gpqa-diamond) — Vals 93.43 |
| Claude Opus 5 | 93.2% | canonical | reasoning | [canonical](https://artificialanalysis.ai/evaluations/gpqa-diamond) |
| Qwen3.8 Max | 92.8% | canonical | reasoning | [canonical](https://artificialanalysis.ai/evaluations/gpqa-diamond) — Vals 93.69 |
| GLM-5.3 | 91.7% | canonical | reasoning | [canonical](https://artificialanalysis.ai/evaluations/gpqa-diamond) — Vals 88.13 |
| GLM-5.3 Flash | 91.2% | canonical | reasoning | [canonical](https://artificialanalysis.ai/evaluations/gpqa-diamond) — Vals 86.36; vendor 91.0 |
| DeepSeek V4 Flash 0731 | 88.1% | canonical | reasoning | [canonical](https://local-ai-zone.github.io/blog/flash-tier-ai-models-comparative-analysis.html) — vendor GPQA Diamond Pass@1 88.1; Vals 89.9 |
| Qwen3.8 Flash | not found |  |  |  — no GPQA row for plain Flash; Qwen3.8-Flash-Next (different variant) 91.7 vendor |
| Qwen3.8-27B | 82.2% | canonical | reasoning | [canonical](https://openrouter.ai/benchmarks/gpqa-diamond) — AA-run via OpenRouter, ±1.4pp; aggregator benchlm lists 90.5 (discrepancy recorded) |
| Qwen 3.5 9B | 77.6% | canonical | reasoning | [canonical](https://openrouter.ai/benchmarks/gpqa-diamond) — AA-run ±2.6pp; HF model-card self-report 81.7 |
| Claude 3.7 Sonnet (no thinking) | 76.8% | aggregator | direct | [aggregator](https://modelbenchmark.io/benchmarks/gpqa-diamond) |
| Gemma 4 E4B | 58.6% | self_reported | reasoning | [self_reported](https://www.reddit.com/r/LocalLLaMA/comments/1sbp8ny/gemma_4_vs_qwen_35_benchmark_comparison/) |
| GPT-4o | 54.3% | aggregator | reasoning | [aggregator](https://benchlm.ai/benchmarks/aagpqadiamond) |
| o4-mini | 77.5% | aggregator | reasoning | [aggregator](https://modelbenchmark.io/benchmarks/gpqa-diamond) |
| Llama 3.1 8B | 27.0% | aggregator | reasoning | [aggregator](https://modelbeats.com/models/llama-3-1-8b) — ~27-31 across aggregators; no canonical row |

## ARC-Challenge

*Jev context: Jev 97.9% (direct) is at/above every published row found; benchmark is 2018-era and saturated*

| Model | Score | Source type | Protocol | Source / note |
|---|---|---|---|---|
| Llama 3.1 405B | 96.9% | self_reported | cot_25shot | [self_reported](https://modelbeats.com/benchmarks/arc-challenge) |
| GPT-4o | 96.7% | independent | reasoning | [independent](https://modelbeats.com/benchmarks/arc-challenge) |
| Claude 3 Opus | 96.4% | self_reported | reasoning | [self_reported](https://modelbeats.com/benchmarks/arc-challenge) |
| GPT-4 | 96.3% | self_reported | cot_25shot | [self_reported](https://modelbeats.com/benchmarks/arc-challenge) |
| Nemotron-H 56B | 95.0% | self_reported | cot_25shot | [self_reported](https://genailist.net/benchmark/arc-challenge) |
| Llama 3.1 70B | 94.8% | self_reported | cot_25shot | [self_reported](https://modelbeats.com/benchmarks/arc-challenge) |
| Llama 3.1 8B (base, zero-shot) | 74.7% | aggregator | direct | [aggregator](https://www.sota2.com/research/sota/multiple-choice-question-answering-on-arc-challenge) |
| de minimis nine (2026 flagships) | not found |  |  |  — no 2026-era ARC-Challenge rows found for any of the nine (saturated/archived; consistent with research.md 2026-09-19) |

## MATH-500

*Jev context: NO protocol-compatible comparison exists: Jev ran a 4-option MCQ adaptation (83.1%) and a per-digit readout (13.6%); published MATH-500 rows are free-form generation, reasoning-era, saturated (~96-99). Context rows only.*

| Model | Score | Source type | Protocol | Source / note |
|---|---|---|---|---|
| GPT-5 (high) | 99.4% | canonical | reasoning | [canonical](https://artificialanalysis.ai/evaluations/math-500) |
| o3 | 99.2% | canonical | reasoning | [canonical](https://artificialanalysis.ai/evaluations/math-500) |
| Gemini 3 Pro | 96.4% | aggregator | reasoning | [aggregator](https://anotherwrapper.com/tools/llm-pricing/evals/math-500) |
| DeepSeek R1 Distill Qwen 14B | 94.9% | canonical | reasoning | [canonical](https://frontierlog.azaharonline24.workers.dev/benchmarks/math-500/) |
| Claude 3.7 Sonnet (thinking) | 94.7% | canonical | reasoning | [canonical](https://frontierlog.azaharonline24.workers.dev/benchmarks/math-500/) — Vals no-thinking 76.8; Vals thinking 91.6 (benchmarklist) |
| Gemma 4 E2B | 86.0% | canonical | reasoning | [canonical](https://frontierlog.azaharonline24.workers.dev/benchmarks/math-500/) |
| DeepSeek R1 Distill Llama 8B | 85.9% | canonical | reasoning | [canonical](https://frontierlog.azaharonline24.workers.dev/benchmarks/math-500/) |
| GPT-4o (chatgpt-latest 2025-03) | 89.3% | canonical | reasoning | [canonical](https://frontierlog.azaharonline24.workers.dev/benchmarks/math-500/) — Vals GPT-4o 2024-08-06: 75.2 |
| Llama 3.1 405B | 71.4% | canonical | direct | [canonical](https://benchmarklist.com/benchmarks/vals_math500/) |
| Llama 3.1 70B | 65.1% | canonical | direct | [canonical](https://benchmarklist.com/benchmarks/vals_math500/) |
| de minimis nine | not found |  |  |  — no individual MATH-500 rows for the nine beyond saturated-era context; MATH-500 excluded from 2026 weighted scoring formulas (benchlm) |

## ARC-AGI-2

*Jev context: NOT comparable by design: Jev's numbers are a per-cell discrimination diagnostic (53.5-61.9%) on the PUBLIC 120-task eval, while published scores are exact-grid production pass@2 on the semi-private set. Context only.*

| Model | Score | Source type | Protocol | Source / note |
|---|---|---|---|---|
| GPT-6 Astra | 95.0% | canonical | reasoning_pass2_semiprivate | [canonical](https://arcprize.org/results) |
| GPT-5.6 Sol | 92.5% | canonical | reasoning_pass2_semiprivate | [canonical](https://arcprize.org/results) |
| Claude Opus 5 | 90.4% | canonical | reasoning_pass2_semiprivate | [canonical](https://arcprize.org/results) |
| Claude Fable 5.1 | 90.0% | canonical | reasoning_pass2_semiprivate | [canonical](https://arcprize.org/results) — Mythos 5.1: not found |
| Claude Fable 5 | 89.2% | canonical | reasoning_pass2_semiprivate | [canonical](https://arcprize.org/results) |
| Gemini 3.7 Flash | 84.6% | canonical | reasoning_pass2_semiprivate | [canonical](https://arcprize.org/results) |
| Grok 4.6 | 67.1% | canonical | reasoning_pass2_semiprivate | [canonical](https://arcprize.org/results) |
| DeepSeek V4 Flash 0731 | 61.4% | canonical | reasoning_pass2_semiprivate | [canonical](https://arcprize.org/results) |
| DeepSeek V4 Pro 0813 | 61.3% | canonical | reasoning_pass2_semiprivate | [canonical](https://arcprize.org/results) |
| Gemini 3.6 Flash | 60.4% | canonical | reasoning_pass2_semiprivate | [canonical](https://arcprize.org/results) |
| Kimi K3 | 60.4% | canonical | reasoning_pass2_semiprivate | [canonical](https://arcprize.org/results) |
| GPT-5.6 Luna | 59.6% | canonical | reasoning_pass2_semiprivate | [canonical](https://arcprize.org/results) |
| GPT-5.2 (Dec 2025) | 53.5% | aggregator | reasoning_pass2_semiprivate | [aggregator](https://intuitionlabs.ai/articles/gpt-5-2-arc-agi-2-benchmark) |
| Inkling Small | 40.1% | canonical | reasoning_pass2_semiprivate | [canonical](https://arcprize.org/results) |
| Claude 3.7 Sonnet (thinking 16K) | 28.6% | aggregator | reasoning_pass2_public | [aggregator](https://aisharenet.com/en/arc-agi-2-chengjijieai/) — base (no thinking): 13.6 |
| GPT-4.5 | 10.3% | aggregator | direct_public | [aggregator](https://aisharenet.com/en/arc-agi-2-chengjijieai/) |
| o3 (low) | 4.0% | canonical | reasoning_pass2_public | [canonical](https://arcprize.org/blog/analyzing-o3-with-arc-agi) |
| GPT-4o | 0.0% | aggregator | reasoning_pass2_public | [aggregator](https://modelcap.ai/benchmarks/arc-agi-2) — 0-4.5% across sources |
| GLM-5.3 / Qwen3.8 Max / GLM-5.3 Flash / Qwen3.8 Flash | not found |  |  |  — no verified ARC-AGI-2 rows found |

## Humanity's Last Exam (text-only)

*Jev context: Jev ran the MC SUB-TRACK (494 items) direct-answer: 21.9%. Published rows are the FULL text-only subset (2,158 items), mostly reasoning-enabled, some with tools. Chance on Jev's MC sub-track ~17%.*

| Model | Score | Source type | Protocol | Source / note |
|---|---|---|---|---|
| Claude Opus 5 | 54.9% | canonical | reasoning | [canonical](https://www.ai-atlas.co/benchmarks/humanitys-last-exam) — AA whole-HLE, max effort; vendor Fable 5.1 no-tools 60.9 |
| GPT-6 Astra | 54.2% | canonical | reasoning | [canonical](https://labs.scale.com/leaderboard/humanitys_last_exam_text_only) — AA whole-HLE max: 54.7 (ai-atlas.co) |
| Claude Fable 5.1 | 59.1% | aggregator | reasoning | [aggregator](https://www.benchleader.com/benchmarks/aa_hle) — vendor no-tools 60.9; Mythos 5.1 not separately published |
| GPT-5.6 Sol | 49.5% | canonical | reasoning | [canonical](https://www.ai-atlas.co/benchmarks/humanitys-last-exam) |
| GLM-5.3 | 62.5% | aggregator | tools | [aggregator](https://llmlearner.com/rankings/hle) — with tools, max |
| Qwen3.8 Max | 56.2% | canonical | tools | [canonical](https://llm-stats.com/benchmarks/humanity's-last-exam-(with-tools%2C-text-only)) |
| Gemini 3.1 Pro (thinking high) | 47.3% | canonical | reasoning | [canonical](https://labs.scale.com/leaderboard/humanitys_last_exam_text_only) |
| GPT-5.4 Pro | 45.3% | canonical | reasoning | [canonical](https://labs.scale.com/leaderboard/humanitys_last_exam_text_only) |
| GLM-5.3 Flash | 50.2% | self_reported | tools | [self_reported](https://local-ai-zone.github.io/blog/flash-tier-ai-models-comparative-analysis.html) — vendor with-tools row |
| DeepSeek V4 Flash 0731 | 34.8% | self_reported | direct | [self_reported](https://local-ai-zone.github.io/blog/flash-tier-ai-models-comparative-analysis.html) — vendor HLE no-tools Pass@1 |
| Qwen3.8 Flash | not found |  |  |  — Qwen3.8-Flash-Next (different variant) 35.9 vendor no-tools |
| GPT-5 Pro (Oct 2025) | 33.3% | canonical | reasoning | [canonical](https://labs.scale.com/leaderboard/humanitys_last_exam_text_only) |
| GPT-5 (2025-08) | 26.3% | canonical | reasoning | [canonical](https://labs.scale.com/leaderboard/humanitys_last_exam_text_only) |
| Qwen3-235B-A22B-Thinking-2507 | 15.4% | canonical | reasoning | [canonical](https://labs.scale.com/leaderboard/humanitys_last_exam_text_only) |
| DeepSeek-R1-0528 | 14.0% | canonical | reasoning | [canonical](https://labs.scale.com/leaderboard/humanitys_last_exam_text_only) |
| Claude 3.7 Sonnet | 7.9% | aggregator | reasoning | [aggregator](https://benchmarklist.com/models/anthropic-claude-3.7-sonnet/) — HLE text-only ±1.14pp |

