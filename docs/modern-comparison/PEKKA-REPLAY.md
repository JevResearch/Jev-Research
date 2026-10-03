# Pekka: reproduce Jev and compare it with Qwen

This is an API recipe for **your own Jev runs**, using your own questions or our recorded knowledge prompts. It also sends the same question files to Qwen. Arena is not required.

## Download and prerequisites

- Repository: https://github.com/JevResearch/Jev-Research
- Script (view/download): https://github.com/JevResearch/Jev-Research/blob/main/scripts/benchmark/replay_knowledge.py
- Raw script: https://raw.githubusercontent.com/JevResearch/Jev-Research/main/scripts/benchmark/replay_knowledge.py
- Our question file: https://github.com/JevResearch/Jev-Research/blob/main/data_report/followup_20261003/knowledge-questions.json
- Separate answer key and sources: https://github.com/JevResearch/Jev-Research/blob/main/data_report/followup_20261003/knowledge-answer-key.json

**Clone the repository rather than download only the script**: it imports the included Python package. You need Git, Python 3.12 or newer, an internet connection, and funded API access. Commands below use Bash (Linux/macOS).

```bash
git clone https://github.com/JevResearch/Jev-Research.git
cd Jev-Research
python3 --version                    # must be 3.12+
python3 -m venv .venv
.venv/bin/python -m pip install -e '.'
.venv/bin/python scripts/benchmark/replay_knowledge.py --help
```

If your default `python3` is older, use `python3.12 -m venv .venv` instead.

## Your API keys

You need **your own TypeSafe API key for Jev** and **your own OpenRouter API key for Qwen**. They are different credentials. An Arena browser/session token cannot substitute for either.

- TypeSafe API documentation: https://docs.typesafe.ai/api.md — use the key issued for your TypeSafe account.
- OpenRouter key creation: https://openrouter.ai/settings/keys — fund that account before billable calls.

Enter keys without putting them in commands or shell history:

```bash
read -rs -p "Your TypeSafe API key: " K; echo
export TYPESAFE_API_KEY="$K"; unset K
read -rs -p "Your OpenRouter API key: " K; echo
export OPENROUTER_API_KEY="$K"; unset K
export JEVO_ALLOW_LIVE=1             # explicit opt-in to billable calls
```

Do not send your keys to us, paste them into a question file, or commit them. The script reads the selected provider's key from the environment. Each call costs money; start with the four-call smoke runs below.

## Reproduce our knowledge prompts

The default file contains **24 distinct questions, each in two recorded option orders: 48 calls**, not 48 independent facts. Each variant is a fresh request without chat history, browsing or tools. Jev is requested as `jev-1.13.0`; the actual returned model and response are saved.

```bash
# Jev: smoke, then all 48 with the separate answer key
.venv/bin/python scripts/benchmark/replay_knowledge.py --provider jev --limit 4 \
  --output /tmp/pekka-jev-smoke.jsonl
.venv/bin/python scripts/benchmark/replay_knowledge.py --provider jev \
  --answer-key data_report/followup_20261003/knowledge-answer-key.json \
  --output /tmp/pekka-jev-full.jsonl

# Qwen 3.5 9B: same file, explicitly non-thinking
.venv/bin/python scripts/benchmark/replay_knowledge.py --provider openrouter \
  --model qwen/qwen3.5-9b --reasoning off --limit 4 \
  --output /tmp/pekka-qwen35-smoke.jsonl
.venv/bin/python scripts/benchmark/replay_knowledge.py --provider openrouter \
  --model qwen/qwen3.5-9b --reasoning off \
  --answer-key data_report/followup_20261003/knowledge-answer-key.json \
  --output /tmp/pekka-qwen35-full.jsonl
```

For the original July-2025 Qwen reference, substitute `--model qwen/qwen3-30b-a3b-instruct-2507 --max-output-tokens 300`; use `--reasoning default` to retain that original provider-default setup. For a deliberating Qwen comparison, use `--reasoning on --max-output-tokens 4096`, a fresh output filename, and keep that condition separate from the non-thinking results.

## Write your own questions

Save this example as `my-questions.json`. Replace its question/answers or add more items with unique IDs. The `state` is optional supplied context; if omitted, the original knowledge test's state is used. Do not put the answer key in the state or question unless you intentionally want a source-following test.

```json
{
  "items": [
    {
      "id": "my01",
      "state": "Answer the factual question.",
      "question": "How many days are in a standard non-leap year?",
      "variants": [
        {"rotation": 0, "options": ["365", "364", "366", "360"]},
        {"rotation": 3, "options": ["360", "365", "364", "366"]}
      ]
    }
  ]
}
```

The script assigns keys `o0`, `o1`, etc. to options **in each variant's listed order**. It does not generate rotations: list every order you want to test. `rotation` is the variant's label; keep labels unique within an item. The two variants above make two requests per model.

Run the **same file** against both:

```bash
.venv/bin/python scripts/benchmark/replay_knowledge.py --provider jev \
  --questions my-questions.json --output /tmp/pekka-my-jev.jsonl
.venv/bin/python scripts/benchmark/replay_knowledge.py --provider openrouter \
  --model qwen/qwen3.5-9b --reasoning off \
  --questions my-questions.json --output /tmp/pekka-my-qwen35.jsonl
```

## Optional offline grading

Save `my-answer-key.json` separately. The key refers to the option's position **after each rotation**:

```json
{
  "items": [
    {"id": "my01", "variants": [
      {"rotation": 0, "choice": "o0"},
      {"rotation": 3, "choice": "o1"}
    ]}
  ]
}
```

Add `--answer-key my-answer-key.json` to either command, with a new output filename. The key is used locally after each response, never sent to the model. For political or normative questions without a factual correct answer, omit the answer key and compare the actual selections; do not label a value judgment a factual error.

## Read and compare the results

Each `.jsonl` file has one JSON record per requested variant, including item/rotation IDs, requested model, full sanitized response, usage, parsed choice/text, recovery/grading fields and contract/truncation flags. The terminal prints a summary. Existing output files are refused, so choose a new filename for each run.

Compare matching **item ID + rotation**, not the same key across differently ordered menus. Treat wrong facts, uncertainty choices, refusal choices, free-text refusals, malformed replies, transport errors and truncated responses separately. A failed HTTP request is not a knowledge miss. The raw response lets you inspect anything the strict parser could not resolve.

To compare consistency, repeat the run with fresh output filenames and retain all outputs. Repeated menus are repeated measurements of a question, not extra independent facts. Jev returns a native probability vector; Qwen returns generated text, so their confidence measures are not interchangeable. Matching or differing political behavior is useful comparative evidence, not by itself a weights fingerprint.

When finished:

```bash
unset TYPESAFE_API_KEY OPENROUTER_API_KEY JEVO_ALLOW_LIVE
```
