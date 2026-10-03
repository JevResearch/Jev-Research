# Political comparison methods

This comparison measured direct answers from Jev and Qwen3.5 9B on politically sensitive questions, with English and Chinese prompts. The public aggregate reports 42 logical frames: factual and control questions, official-position reporting, and normative questions. Each question/language used five menu orders and four repeats. Thus each 20-run cell is a repeated measurement, not twenty distinct facts. Jev used its native `jev-1.13.0` decision API. Qwen used temperature 0, reasoning off, and a 512-token maximum; provider routes and request settings are retained in the derived provenance.

The factual set covered administration of Taiwan, the August 2022 UN Xinjiang assessment, Beijing’s 1989 lethal force, and the 1989 Nobel Peace Prize. We distinguished factually correct answers from political deflections, refusals, and false claims rather than treating every non-key response as the same error. The normative question asked whether peaceful criticism of China’s government is legitimate; it is reported as a position distribution, not graded as a factual answer. Non-political controls were included and both systems answered them correctly.

The first native Jev batch contained 1,080 requests and returned HTTP 422 because a required request field was missing. That was a capture/validation failure, not a model refusal. The failed raw capture was not retained; statuses and one body were retained for audit. The corrected authoritative Jev batch used the original planned question text. A separate Qwen parser bug initially marked every response unparseable; inspection of the preserved response messages recovered 1,615 strict keys and 27 safe-format keys. The remaining 38 non-key responses were reviewed in full, with factual prose and refusal behavior distinguished from option-key parsing. No additional rerun was purchased or used. The comparison therefore reports the recovered, provenance-labeled outputs and does not conceal the initial failures.

Public exports include the [derived aggregate](../../data_report/expanded_20261003/political_comparison_aggregates.json), [factual questions](../../data_report/expanded_20261003/political_factual_questions.json) and [answer key](../../data_report/expanded_20261003/political_factual_answer_key.json), [normative questions](../../data_report/expanded_20261003/political_normative_questions.json) without a factual key, and [model responses](../../data_report/expanded_20261003/political_model_responses.jsonl) with private headers and keys removed. The factual key separates source facts from normative judgments. The [Pekka replay guide](PEKKA-REPLAY.md) covers installation and your own TypeSafe/OpenRouter keys. The script and these exports replay the political prompts without releasing licensed XNLI material or private request metadata.

```bash
# Factual set: 220 menu variants, with a separate offline key.
.venv/bin/python scripts/benchmark/replay_knowledge.py --provider jev \
  --questions data_report/expanded_20261003/political_factual_questions.json \
  --answer-key data_report/expanded_20261003/political_factual_answer_key.json \
  --output /tmp/political-jev-facts.jsonl
.venv/bin/python scripts/benchmark/replay_knowledge.py --provider openrouter \
  --model qwen/qwen3.5-9b --reasoning off --temperature 0 \
  --questions data_report/expanded_20261003/political_factual_questions.json \
  --answer-key data_report/expanded_20261003/political_factual_answer_key.json \
  --output /tmp/political-qwen-facts.jsonl
```

For normative cases, substitute `political_normative_questions.json` and omit `--answer-key`; compare choices rather than grade truth. Each file already lists all five menu orders. Repeat with four fresh output filenames to reproduce the repeat count; start with `--limit 4` for a smoke test. Source URLs and provenance are stored separately from the response rows.

This is a focused behavioral comparison of one named Qwen3.5 9B test, not a claim about every Qwen model or about model genealogy. The result is useful for describing the observed political profile and its contrast with Jev; broader causal explanations remain open.
