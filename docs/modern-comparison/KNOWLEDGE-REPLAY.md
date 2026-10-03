# Replaying the knowledge probe (API instructions)

For a complete, forwardable Jev-versus-Qwen recipe, see [Pekka’s replay guide](PEKKA-REPLAY.md).

Reproduce our **Jev** results with your own TypeSafe API key, then run the same
questions against Qwen if desired. `scripts/benchmark/replay_knowledge.py` replays
the 48 public prompts (`data_report/followup_20261003/knowledge-questions.json`:
24 items × 2 option rotations) and also accepts your own question file.
Python 3.12+ required. Every API call is billable; nothing runs without the
opt-in below. Use `--limit 4` for a smoke run.

```bash
git clone https://github.com/JevResearch/Jev-Research.git
cd Jev-Research
python3 -m venv .venv
.venv/bin/pip install -e '.'
```

## API keys (Jev or OpenRouter)

Type the key at the prompt (it never enters shell history), then opt in:

```bash
read -rs -p "TypeSafe key: " K; export TYPESAFE_API_KEY="$K"; unset K
export JEVO_ALLOW_LIVE=1
```

Jev smoke, then the full 48 with offline grading against the local answer key:

```bash
.venv/bin/python scripts/benchmark/replay_knowledge.py --provider jev --limit 4 \
  --output /tmp/pekka-jev-smoke.jsonl
.venv/bin/python scripts/benchmark/replay_knowledge.py --provider jev \
  --answer-key data_report/followup_20261003/knowledge-answer-key.json \
  --output /tmp/pekka-jev-full.jsonl
```

The Qwen reference needs an **OpenRouter** key
(`read -rs -p "OpenRouter key: " K; export OPENROUTER_API_KEY="$K"; unset K`) —
not a TypeSafe key, and not an Arena account:

```bash
.venv/bin/python scripts/benchmark/replay_knowledge.py --provider openrouter \
  --model qwen/qwen3-30b-a3b-instruct-2507 --limit 4 \
  --output /tmp/pekka-qwen-smoke.jsonl
```

Each run writes fresh rows (requested/served model, status, full sanitized
response, usage, choice, strict/recovered/graded fields); existing output files
are refused. Output budget defaults to 512 tokens (for the original reference
cap use `--max-output-tokens 300`). See `--help`.

## Add your own questions

Copy `knowledge-questions.json` to `my-questions.json`, keeping its `items` /
`variants` structure. Each item needs a unique `id`, a `question`, and one or
more variants with a `rotation` label and ordered `options`:

```json
{"items":[{"id":"my01","question":"Your question here",
  "variants":[{"rotation":0,"options":["First answer","Second answer"]}]}]}
```

Run it directly against Jev:

```bash
.venv/bin/python scripts/benchmark/replay_knowledge.py --provider jev \
  --questions my-questions.json --output /tmp/pekka-jev-custom.jsonl
```

Use that same `--questions my-questions.json` with `--provider openrouter` and
`--model qwen/qwen3-30b-a3b-instruct-2507` for Qwen. A custom answer key is
optional: grade manually, or provide `--answer-key my-answer-key.json` using the
published key's structure and matching item IDs and rotation labels. The key
is never sent to either model.

## Arena web (optional Qwen comparison; no API key)

The web UI needs none of the keys above: paste each variant from
`knowledge-arena-replay.txt` into a **fresh** chat, pick
`qwen3-30b-a3b-instruct-2507`, disable web/tools, reply with only the option
key, save full replies, then grade against the answer key. A web session token
cannot drive the CLI.
