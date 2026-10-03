# Replaying the knowledge probe (API instructions)

Replay the 48 public prompts (`data_report/followup_20261003/knowledge-questions.json`:
24 items × 2 option rotations) with `scripts/benchmark/replay_knowledge.py`.
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

## Arena web (no API key)

The web UI needs none of the keys above: paste each variant from
`knowledge-arena-replay.txt` into a **fresh** chat, pick
`qwen3-30b-a3b-instruct-2507`, disable web/tools, reply with only the option
key, save full replies, then grade against the answer key. A web session token
cannot drive the CLI.
