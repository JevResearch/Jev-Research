# Talking to Jev: the complete program report

Everything attempted in this thread, all discoveries, and the current state.
Companion artifacts: `docs/token-talk-plan.md` (design),
`docs/token-talk-findings.md` (§1-15, incremental findings), all traces under
`runs_live/`, scripts under `scripts/`, tests at 367 passing.

## 1. Starting point

Jev (TypeSafe AI's classification endpoint) cannot emit text; it can only
answer Choice/Score/Noul questions over options we provide. Prior to this
thread, `talk.py` (attempt 1) implemented DESIGN.md §9 "external
autoregression": one character per request over a fixed 98-option alphabet.
Its live traces (`runs_live/talk_live_traces.json`) showed coherent first-step
distributions (G 0.28 / A 0.25 / H 0.23) but greedy degeneration: `Geeee`,
`A    `, `AAB`. Diagnosis: a flat candidate space + compounding argmax error +
no language model anywhere in the loop, at ~950 input tokens per emitted
character.

## 2. The redesign: Jev steers a local LM

New module `token_talk.py` (attempt 1 untouched). Per step: a local LM scores
its vocabulary (temperature 0, never sampled locally), the top-254 tokens plus
END become Jev's options (opaque `tok_NNN` keys, repr-quoted text), Jev returns
a distribution, a decode rule picks one token, it is fed back. Local
probabilities never leave the harness. Honest frame: Jev re-scores a local
model's contextual next-token distribution — not "Jev talking", not native
chat.

Backends: `local_lm.py` on CUDA-built llama-cpp-python with
**Mellum2-12B-A2.5B-Instruct Q3_K_S** (MoE, ~2.5B active) fully offloaded to
the 8 GB RTX 5050; ~10 ms/token forward. Non-trivial engineering found along
the way: logits only via the context buffer (`eval()` stores nothing without
`logits_all`), state save/restore must carry the logits buffer separately, chat
templates applied from GGUF metadata with a marker trick to open an assistant
turn at an exact partial prefix, template-marker fragments (`|<`, `im_end`)
blocked from ever being spoken, `reset()` before each session, and `_render_prompt`
marker handling that only worked for empty prefixes until the order probe
needed frozen ones.

## 3. Decode modes (recorded per node; raw distributions always kept)

- `local_greedy` — control; Jev's answer recorded but ignored.
- `jev_greedy` — argmax of Jev's distribution.
- `jev_sample` — seeded sampling of Jev's distribution.
- `product` — renormalized p_local x p_jev; Jev reranks, local prior as
  safety net. Repeatedly the only fluent configuration.

Sampling controls added over time: **floor subtraction** (the vendor
quantizes probabilities to ~0.01 steps; subtract the floor, renormalize —
note: this *amplifies* an above-floor END, causing the "empty response END"
failure), **temperature**, **nucleus top-p**, **top-k**.

## 4. The protocol ladder (the central result)

| level | lexicon | syntax | verdict |
|---|---|---|---|
| character menu (`char_talk.py`, attempt 1's alphabet + all fixes) | 0% | none | intrinsically broken: Jev's per-character distributions are weak (spaces + `a` dominate regardless of controls) |
| vocabulary menu (`word_talk.py` + `vocab.py`: 67k zipf words, prefix-mass combos, whole words, stuck escape "jzqg a", spelled-out punctuation incl. [backspace], in-context option texts) | 86-100% real words | none | word salad with real words: unigram/prefix statistics carry no syntax |
| token level + local LM (product) | real | real | the only coherent paragraphs |

The missing ingredient for coherent text is exactly the sequential prior the
local LM supplies. Unguided Jev — as free-form as menus allow — is a
word-salad generator with good stopping behavior.

## 5. Behavioral discoveries about Jev

1. **Whitespace-blind, surface-form re-ranking.** Prefers bare tokens over
   leading-space variants mid-sentence (`capital` 0.5-0.6 vs ` capital`
   0.13-0.16) → `ThecapitalisParis.`. Drops articles. This is the token-level
   signature of the candidate-local scoring documented in the repo's odds
   experiments.
2. **Jev has its own semantic prior.** It chose `is` against the local model's
   ` of/off` after a broken prefix and steered to `Paris` anyway; identity
   claims ("gpt", "OpenAI") arise even when the local distribution is nearly
   flat (8e-5 spread) — the brand prior is Jev's own, and it is
   OpenAI-shaped: never "typesafe", never Mellum (it dodged the local model's
   own `' Mell'` suggestion).
3. **END usage is genuinely good** — one-sentence and factual replies end
   semantically (END 0.68-0.93 after sentence-final punctuation). In long
   creative mode END stays low (never fires); in empty-response states it can
   fire immediately (3/4 sessions before the instruction fix).
4. **Identity through this protocol is mostly local-model property plus Jev
   brand prior**, not architecture evidence — measured causally via the
   option traces, not just inferred.
5. **Ordering is a first-order factor** (frozen-step probe, 11 orderings x 6
   repeats): 10 distinct argmax winners across orderings of identical
   options; Spearman vs native order 0.26-0.44 on flat steps vs 0.42-0.88 on
   sharp ones; U-shaped serial-position curve (primacy ~4x, recency);
   confidence order-dependent (p_max 0.26-0.54). Ordering effect is 3-10x the
   measured nondeterminism baseline (repeat-TVD 0.03-0.12). Implication: our
   native ordering (descending p_local) broadcasts the local prior into
   Jev's choice through position — part of the observed "agreement" is a
   position artifact.
6. **Injection**: `typesafe`/`JetBrains` tokens offered explicitly receive a
   uniform 0.01 floor and are never chosen.
7. **Vendor nondeterminism** is real and reproducible session-to-session.

## 6. Ensemble/order-cancellation

- Minimum ensemble for bias cancellation (frozen-step): **K=4 rough, K=6-8
  clean** (Spearman 0.91-1.0, TVD to 12-rotation reference 0.04-0.17), using
  cyclic rotations, per-answer floor+temperature, add+norm. Aggregation
  flattens (p_max 0.73 -> 0.46), so per-answer temperature must drop.
- Implemented as parallel `ensemble=K` in both token and word sessions
  (ThreadPool over the shared provider; per-ordering raws recorded).
- **End-to-end, ensembles improve measurement and factual decoding but hurt
  creative generation**: at flat steps the position-corrected signal is
  genuinely flat; add+norm yields mush. Verdict: use K>=6 ensembles for any
  mechanism claim about Jev's content preferences; use single native order
  (accepting the documented position artifact) for generation.

## 7. Expanded choices

`max_options` (e.g. 760) splits the local-LM pool into pages of <=254
(contract cap; END only on page 0), each page rotation-ensembled, **all
pages x rotations dispatched in parallel**, combined into one vector,
normalized, decoded. Combined with product mode and T=0.4 this produced the
best output of the program:

> "In the outer rim of galaxy where nebulae paint the void in hues ofviolet
> andgold cos cosmic rabbits hop through the interstellar aether and meet
> Elvis Presley who was beenhad"

Real narrative reaching Elvis; defects are Jev's missing spaces, one stray
fragment, and tail degradation. Cost ~2.0M input tokens for 60 tokens
(6 requests/step).

## 8. Anti-repetition, three generations

1. Post-hoc guards: tail unit x4, word x6 (hardened after ` was`/`was`
   alternation dodged the fixed-unit rule).
2. Structural immediate bans: triple letters impossible, recent words banned
   from menus, zero-emit options excluded (found live: `'an'` completed as
   `'n'` looped 398x as a no-op), no-progress stop, [space] only when a word
   exists to end.
3. **`creates_adjacent_repeat`**: menu-level ban on completing an adjacent
   doubled unit of period 3-24 chars, case-folded ("The the" caught), checked
   with/without a virtual trailing separator, in-progress exemption for
   legally built pairs. Measured effect: identical words cap at three;
   "was was"/"The the" unmakeable; "letter"/"hello" unaffected; guards remain
   as backstop.

Known gaps: multi-char tokens with internal repeats (`"\n\n"`) evade the
period check (newline runs in token_greedy_elvis); alliteration drift
(`she/she're/some/somehow`) is near-loop, not exact-loop, and passes.

## 9. The final matrix (all protocols x {greedy, sample} x 2 questions)

Best settings: token = expanded(760) x ensemble(2), floor 0.02, sample
T=0.4/p=0.9; word = ensemble(4) menu, sample T=0.7/p=0.9; char = sample
T=0.9/p=0.9. Results: token protocols produce grammatical fragments and
correct sentence openings (*"Iam an Assistant"*); word protocols produce real
words in non-grammatical order (86-100% real vocabulary); character
protocols produce spaces and letter runs. Sample mode ends itself more often
(5/6) than greedy (3/6); greedy drifts into structured junk. CJK drift
appears in the expanded pool at flat steps (an argument for content filtering
or product weighting). One transient provider_failure after ~640 requests.

## 10. Engineering notes

- The harness reuses the repo's provider/ledger/redaction/budget machinery;
  live runs gated by `JEVO_ALLOW_LIVE`, key only from the environment
  (never written to any file).
- Local transform discipline everywhere: raw distribution stored, every
  transform (floor, temperature, truncation, capitalization, ensembling)
  recorded next to it, never replacing it.
- Live bugs found and fixed via tests: END's implicit 1.0 prior in product
  mode; empty-prefix rendering in `_render_prompt`; ScriptedLM pool-vs-page
  answer accounting; zero-emit word options; bans tracking deltas instead of
  text; capitalization consulting committed-only text; case-sensitive word
  guards; space spam; immediate END (fixed by clarifying that the response
  is built from empty — an instruction change, recorded).
- Test suite: 367 passing (24 -> 25 -> 27 ... grown across attempts), fully
  offline via ScriptedLM/ScriptedTokenProvider/TextPicker fixtures.

## 11. Cost accounting (approximate, observed pricing ~$0.043/M input tokens)

Char attempt 1 (prior thread): 42k. Token comparison: 174k. Elvis challenge
rounds: ~2.9M. Identity probes: ~0.7M. Order probe: 0.79M. Ensemble probe:
0.79M. Ensemble end-to-end: ~4.0M. Word-talk development rounds: ~3.1M.
Matrix (12 sessions): 4.9M. **Thread total ~13-14M input tokens ≈ $0.60.**
The ensemble/expanded configurations are the first noticeable budget items
(request count multiplies per step).

## 12. Where this leaves us

Best configuration: token protocol, `product` decode, expanded pool (760),
ensemble 2, floor 0.02, T=0.4, anti-repeat on, Mellum2-12B-A2.5B as local
scorer, END always Jev's. Quality: fluent-on-prompt paragraphs corrupted
only by Jev's whitespace bias and late-text drift.

Next levers, in expected-value order: (a) content-filter the expanded pool
(drop CJK/no-alpha candidates); (b) recorded space-merge post-processing
(v1 naive merge failed — needs merge-only-when-doubled rules); (c) stronger
structural anti-repeat for multi-char whitespace tokens; (d) phrase-level
candidates (2-3 token units) to cut request count; (e) for mechanism claims,
always run native + shuffled replications (§5.5).