# Talk to Jev, attempt 2: Jev steers a local LM's next-token distribution

Planning document for a **new** module. `talk.py` (attempt 1, character-level
autoregression) stays untouched; its live traces (`runs_live/talk_live_traces.json`)
are the evidence base for this redesign.

## 1. Why attempt 1 failed

Observed live behavior (greedy decoding, 98-option alphabet):

| session | prefix | stop reason |
|---|---|---|
| "Greet me..." | `Geeee` | repetition |
| reasoning prompt | `A    ` | repetition |
| identity prompt | `AAB` | end_token |

First-step distributions were *coherent* (G 0.28 / A 0.25 / H 0.23 for a "Greet
me" prompt), so Jev does condition on the transcript. The failure is structural:

1. **Flat candidate space.** Every one of the 98 characters is an equally
   "valid" continuation from a linguistic standpoint, so Jev's per-step
   distribution is jagged and tie-ish (max ≈ 0.3–0.4). Greedy argmax over
   near-ties compounds error: one bad character reconditions the next step,
   and the walk drifts into repetition.
2. **No language model anywhere in the loop.** We asked Jev to *be* the whole
   LM, one character at a time — the exact competence (a prior over fluent
   continuations) that makes text generation work was absent.
3. **Cost profile.** ≈950 reported input tokens per emitted *character*; the
   linguistic yield per request is one character.

## 2. The idea

Flip the roles. A small **local** LM supplies linguistic competence; Jev
supplies the *preference* over likely continuations. Per step:

1. Format the conversation with the local model's chat template + the partial
   assistant reply; run **one forward pass**; softmax over the vocabulary.
2. Take the top-K next tokens (K configurable; plus an explicit END option),
   decode them to visible text.
3. Send Jev one Choice question whose options are exactly these candidates,
   with opaque keys (`tok_000` … `tok_{K-1}`, `END`) and descriptions showing
   the decoded token text.
4. Jev returns a distribution over the candidate set.
5. Apply a decode rule (below), append the chosen token to the reply, repeat.

This is the same "explicit external autoregression" protocol, but the
candidate set is **contextual**: every option is a locally plausible
continuation of the conversation. Even a weak or noisy picker cannot fall out
of the language — worst case it random-walks *within* fluent English.

Honesty framing (DESIGN.md §9 unchanged in spirit): this is **not** "Jev
talking" and not native chat. It is *Jev steering a local model's next-token
distribution*. Both distributions (local softmax, Jev choice) are recorded raw
at every step.

## 3. Architecture

New module `src/jev_observatory/token_talk.py`. Reuse without modification:
`providers` (JevProvider/MockProvider, attempt ledger, retry policy),
`schema.SystemOneRequest`, `validation`, `guards` (budget), trace-export style
of `talk.py`.

### 3.1 LocalLM protocol

```python
class LocalLM(Protocol):
    name: str
    def candidates(self, context: Conversation) -> list[TokenCandidate]
        # full forward pass, softmax, top-K token ids + probs (never sampled locally)
    def advance(self, token_id: int) -> None      # append to KV cache
    def save_state(self) -> bytes                 # branching/backtrack support
    def load_state(self, state: bytes) -> None
```

Implementations:

- **`LlamaCppLM`** (primary): `llama-cpp-python` 0.3.35 (available for py3.14;
  CUDA build via `CMAKE_ARGS="-DGGML_CUDA=on"`). Model: **Mellum2-12B-A2.5B-Instruct,
  Q3_K_S GGUF** (`hf.co/bartowski/Mellum2-12B-A2.5B-Instruct-GGUF:Q3_K_S`) — MoE
  with ~2.5B active params, ≈5–6 GB at Q3_K_S, fits the 8 GB card with full GPU
  offload; one forward pass per step is cheap (~2.5B active). The GGUF is
  downloaded explicitly (hash recorded in the run manifest), gitignored, never
  bundled.
- **`ScriptedLM`** (tests): deterministic candidate tables — the *entire*
  session logic is testable offline with zero model download and MockProvider.
- `transformers`/`torch 2.14` is a possible alternate backend (cp314 wheels
  exist) but is heavy and not MVP; the Protocol keeps it swappable.

Local decoding temperature is **always 0** (pure scoring). The local LM never
samples; Jev (plus recorded, seeded transforms) is the only stochastic source.

### 3.2 Candidate builder

- **Candidate count: maximum, not curated.** Take **as many candidates as the
  contract allows**: 254 tokens + END = 255 options (contract limit). Fill
  from the model's top-K in its native yield order (descending local
  probability), stopping only at the vocabulary/logits limit. Realized candidate
  count recorded per step.
- **END is always Jev's choice, never the model's.** The model's own
  EOS / end-of-turn / template special tokens are **stripped from the
  candidate list** (they would also corrupt the chat template if re-fed);
  stopping is represented solely by the explicit `END` option we add.
- Filter candidates that decode to control characters or invalid text;
  descriptions use `repr`-style rendering (a token containing quotes must be
  inert data, never prompt-shaped injection — and any leakage effect is an
  *observed result*, recorded, per §9's stance).
- Option order: the model's native yield order (descending local
  probability), no shuffle. Recorded. (Position bias remains a known
  confound; the raw distributions in the trace are what let us see whether
  ordering skews Jev's picks — an empirical question, not a design knob.)

### 3.3 Request shape (per step)

Same state discipline as attempt 1 — distinct fields `user_prompt`,
`previous_turns`, `assistant_prefix` (text so far) — plus one Choice question
`next_token` with the candidate criteria. One sequential request per token:
parallel next-position questions remain non-equivalent (DESIGN.md §9).

### 3.4 Session

`TokenTalkSession` mirrors `TalkSession`: immutable tree of token nodes
(`token_id`, text, local_probs, jev_probs, decoder used, latency, usage,
`user_edited` flag), explicit stops only:

- END chosen; `max_tokens` (default 64); wall clock (default 240 s); provider
  failure; budget denial; user cancel; repetition guard (same conservative
  tail rule on decoded text, plus a same-token-id ×4 rule).

Decode modes, recorded per node (plus the local argmax is *always* recorded so
every trace shows what the local LM alone would have said):

| mode | rule | meaning |
|---|---|---|
| `local_greedy` | argmax local | control baseline; no Jev influence |
| `jev_greedy` | argmax Jev | direct analogue of attempt 1, over plausible candidates — headline mode |
| `jev_sample` | seeded sample of Jev distribution | temperature transform local, labeled, raw kept |
| `product` | renormalized `p_local × p_jev`, then greedy/sample | Jev as reranker with local prior as safety net |

Backtrack/fork via LM state save/restore; branches are new immutable nodes,
never rewrites (same as attempt 1). Multi-turn: a completed assistant reply is
appended to `previous_turns` and re-fed through the chat template.

### 3.5 CLI / web (phase 2)

`jevo talk-tokens --provider mock|jev --lm <gguf> --decode jev_greedy
--max-options 254 --max-tokens 64` with the same live gates as `jevo run`/`jevo
talk`
(`JEVO_ALLOW_LIVE`, loopback-only web UI, SSE per-step events). The
scientifically interesting UI view is **dual bars per step**: local-LM
distribution vs Jev's re-scoring — where they disagree is the measurement.

## 4. Why this should work better — and remaining failure modes

- Attempt 1's coherent first steps suggest Jev ranks candidates sensibly; the
  collapse came from argmax over a flat space with compounding errors.
  Constraining to locally likely tokens keeps greedy picks in-distribution.
- Remaining risks, all treated as results rather than bugs:
  - **Flat Jev distribution among candidates** → random walk over plausible
    tokens: fluent but possibly incoherent over long spans. Detectable from
    traces (mean p_max, entropy).
  - **Systematic token preferences** (e.g. always the most "generic" option)
    → degenerate rhythm; visible in per-token Jev-vs-local disagreement rates.
  - **END never chosen** → `max_tokens` stop; report honestly.
  - **Candidate text leakage** (token descriptions acting like injection into
    Jev's scoring) → observed, recorded, not silently patched.

## 5. Cost and latency estimate

- Observed ≈950–1,000 input tokens and ≈0.66 s per step at 98 options with
  short prefixes. 254 token options ≈ 2–3k criteria tokens, so a step costs
  ~3–4k reported input tokens including prefix growth (token-level prefixes
  grow ~4× slower than character-level: measured ~3.7 server-tokens/char).
- A 64-token reply ≈ 64 requests ≈ 200–250k input tokens ≈ **$0.008–0.011** at
  observed pricing; wall clock dominated by Jev latency, ≈ 40–60 s sequential
  (local forward passes are milliseconds on the GPU).
- Budget guard identical to attempt 1; smoke gate: one session, ≤16 tokens,
  hard cap 20 requests, explicit user go before any live call
  (`TYPESAFE_API_KEY` from env only — never in files, artifacts, or this repo).

## 6. Tests (offline only; no network, no model download)

`tests/test_token_talk.py` with `ScriptedLM` + `MockProvider`:

- candidate building: 254 tokens + END = 255 exactly at the contract cap
  when the model yields that many; EOS/special tokens stripped; control
  chars filtered; keys opaque; descriptions repr-safe; order recorded.
- decode modes incl. product-mode arithmetic; local argmax always recorded.
- stops: END, max_tokens, deadline, budget_exceeded, provider failure,
  repetition (text rule + token-id rule).
- backtrack → new immutable branch; LM state restore correctness (scripted).
- trace export: both raw distributions present; user-edited prefixes labeled;
  no option keys ever concatenated into output text.
- seeded determinism for sampling modes; candidate order always the model's
  native yield order.

## 7. Milestones — status

1. **Offline session core** (ScriptedLM + MockProvider) — done; 24 tests in
   `tests/test_token_talk.py`, full suite 341 passing.
2. **Local backend** — done (`local_lm.py`, CUDA llama-cpp-python 0.3.35, GPU
   offload validated); dry-run prints top-254 candidates for fixed prompts;
   KV-state save/restore and template-marker filtering verified.
3. **Live smoke** — done (`runs_live/token_talk_smoke_trace.json`), END chosen
   by Jev at step 6, no failures.
4. **Comparison mini-run** — done (`runs_live/token_talk_comparison.json`);
   findings in `docs/token-talk-findings.md`.

Remaining (phase 2): CLI/web wiring (`jevo talk-tokens` + SSE dual-bar UI),
whitespace-merge experiment, phrase candidates, larger prompt set.

## 8. Decisions (from planning review) and remaining open questions

Decided:

- **Candidate count: maximum.** 254 model tokens + END = 255, the contract
  cap. No artificial restriction. Whether Jev's distributions flatten at this
  option count (vs the earlier mechanism studies at smaller K) is measured
  from the recorded raw distributions, not pre-empted by curation.
- **Model**: Mellum2-12B-A2.5B-Instruct, Q3_K_S GGUF (fits the 8 GB card).
- **END**: always Jev's decision; the model's EOS/special tokens are stripped
  from candidates.
- **Ordering**: the model's native yield order (descending local probability);
  no shuffle.

Remaining:

- **Phrase candidates** (top 2–3-token continuations) would cut request count
  ~2–3× but interact with candidate decoding and the option cap; defer to a
  follow-up experiment, do not fold into MVP.
- Whether 254 options measurably changes Jev's sharpness vs a smaller-K run —
  an A/B worth one extra live session if traces look flat.
