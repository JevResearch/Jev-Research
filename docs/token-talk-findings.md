# Token-level Talk: first live findings

Evidence: `runs_live/token_talk_smoke_trace.json` and
`runs_live/token_talk_comparison.json` (jev-1.13.0; local scorer
`Mellum2-12B-A2.5B-Instruct-Q3_K_S.gguf`, sha256 prefix `bb68601c7fcb5a3f`;
254 token candidates + END per step; decode modes as labeled). All traces are
redacted raw artifacts; claims below are **exploratory** (n = 1–2 sessions per
condition, single prompt pair).

## 1. The protocol works

Six live token-choice sessions plus a comparison mini-run, zero provider
failures, every session ended by Jev's explicit END choice. Per-step cost
≈ 3–5.4k reported input tokens (255 criteria dominate); ≈ 0.7 s/step.
A 6–10-token reply costs $0.001–0.003. The character-level failure mode
(`Geeee`, `A    `) does not occur: every option is a locally plausible
continuation, so even divergent picks stay within the language.

## 2. Observed behavior by mode

| mode | "Greet me…" | "Capital of France…" |
|---|---|---|
| `local_greedy` (control, no calls) | rambles past natural end into template-adjacent fragments (EOS stripped by design) | same (`|</em>|…`) |
| `jev_greedy` | `Hello there  how can` → END | `ThecapitalisParis.` → END |
| `jev_sample` (seed 7) | `HelloHello Hello? aun language barrierHello` → END | `Thecapital Paris.` → END |
| `product` | `Hello!` → END | **`The capital of France is Paris.`** → END |

## 3. What Jev does and does not contribute (exploratory)

1. **Whitespace-blind re-ranking.** Jev systematically prefers the *bare*
   token variant over the leading-space variant (`capital` 0.50–0.59 vs
   ` capital` 0.13–0.16 even mid-sentence), producing `ThecapitalisParis.`
   under `jev_greedy`. This is the same **candidate-local scoring** signature
   documented in the odds-ratio experiments: Jev scores options largely by
   their own surface form, not by how they compose with the prefix.
2. **Jev does carry its own semantic prior.** In `jev_greedy:1` step 3, after
   the broken prefix `Thecapital`, the local model's top candidates were
   ` of / off / off` — Jev chose `is` (its top) and steered to `Paris` anyway.
   It is not merely echoing the local ranking.
3. **END is used reliably and semantically.** Jev ended every session, at
   reasonable points (`…Paris.` then END 0.68–0.93; `Hello!` then END in
   product mode). The character-level never-stops problem is gone.
4. **Confident-first-step agreement, divergent later steps.** Step 1 tracks
   the local argmax closely (`Hello` 0.88–0.90, `The` 0.83); later steps are
   flatter (top ≈ 0.2–0.5), where Jev's own preference dominates.
5. **Product mode is the fluent configuration** on this tiny sample: the
   local prior fixes spacing/composition; Jev boosts content tokens and
   triggers END. The one grammatical sentence came from `product`.

## 4. What this does and does not establish

**Established (observed, small n):** the token-steering protocol functions
end-to-end; Jev returns per-step distributions over 255 live options; it uses
END appropriately; candidate-local surface-form bias exists at token level.

**Not established:** general fluency (n = 2 prompts/mode); whether product
mode dominates across prompt types; the option-count sensitivity question
(Jev distributions *were* sharp at 254 options — `Hello` 0.88 — contradicting
the worry that max-K flattens it, but this is one prompt); latency/cost
scaling beyond short replies.

## 5. Follow-ups

- Larger prompt set (10–20 prompts) × {jev_greedy, product}; fluency scoring
  by the local LM (mean log-prob of Jev-chosen continuation vs local argmax).
- Whitespace repair without hiding behavior: e.g. merge a chosen bare token
  with a preceding space token only when both were offered (record the merge).
- Multi-token phrase candidates to cut request count ~2–3×.
- CLI/web wiring (`jevo talk-tokens`) reusing `TokenTalkSession`.

## 6. Sampling controls and guard hardening (Elvis challenge follow-up)

After `DecoderConfig` gained top_p/top_k truncation (recorded as the
`transformed` distribution; Jev always sees all 255 options) and the
repetition guard gained a word ×6 rule (spacing-insensitive):

- **`jev_greedy` rerun** (`token_talk_elvis_jevgreedy_v2_trace.json`): the
  hardened guard fired correctly — `… The was was was was was` stopped at 32
  tokens (the earlier run looped for 81). The was-loop is a real Jev attractor
  on creative prompts, not a guard artifact.
- **`jev_sample` T=0.7 / top_p=0.9**: END after 10 tokens
  (`Space rabbits the were fluffy  was`) — cooling sharpens whatever mass END
  already had; at flat positions nucleus keeps few options and END is often
  among them.
- **`jev_sample` T=0.8 / top_p=0.95**: END after 24 tokens, text incoherent.

Exploratory conclusion: on creative prompts Jev's step distributions are flat
(top ≈ 0.2) with non-trivial END mass, so *any* honest sampling of its
distribution is dominated by its own noise, not by cooling. The failure is in
Jev's distribution quality at these positions, not in the sampler parameters;
`product` mode (local prior as safety net) remains the only configuration that
produced a coherent paragraph.

## 7. jev_sample sweep: temperature x top_p x top_k (12 combos, seed 7+i)

With repeated-word candidates banned (words used in the last 4 tokens are
never offered, recorded per node in the candidate list) and fluency scored
offline as the local model's mean per-token log-prob of Jev's exact chosen
tokens (`scripts/token_talk_sweep.py`, full data in
`runs_live/token_talk_sweep_results.json`; ~2.5M input tokens total):

| T | top_p | top_k | fluency (nats/tok, higher=better) | stop | n |
|---|---|---|---|---|---|
| 0.4 | 0.9 | — | **-4.32** | END | 48 |
| 0.2 | 0.9 | — | -4.51 | max_tokens | 80 |
| 0.8 | 0.5 | — | -4.98 | END | 45 |
| 1.0 | — | 8 | -5.63 | END | 29 |
| 0.8 | — | 20 | -5.64 | END | 28 |
| ... | | | | | |
| 1.0 | — | — | -8.08 (worst) | END | 18 |

Best story (T=0.4, p=0.9): "Space rabbits approached the stage where Elvis
Presley was singing, tumbling to the rhythm, their fluffy tails and joyful,
twitching were, mimicking his sways..."

Observed (exploratory, n=1/combo):

1. **Temperature is the dominant knob** (monotone: colder = more fluent);
   top_p=0.9 helps at every temperature; top-k alone is inconsistent (k=5
   truncates so hard Jev ends after 7 tokens).
2. **Every combination still ends via Jev's END** — cooled sampling keeps END
   attractive; long coherent paragraphs remain out of reach for raw sampling.
3. **The repeated-word candidate ban works as designed**: no run looped; the
   word-×6 guard never fired; remaining artifacts are missing articles and
   mid-word splits ("Pres ersonappeared"), i.e. Jev's surface-form scoring,
   not sampler noise.
4. Even the best sampled output is below `product` mode's paragraph
   ("The capital of France is Paris." at -? ; product's Elvis draft read
   fluently). Sampling Jev's raw distribution is intrinsically limited by its
   flat creative-mode distributions; the practical best configurations are
   T=0.2-0.4 with p=0.9 (approaching deterministic argmax from the sampling
   side, which is itself evidence for how flat the distributions are).

## 8. Identity probe through token steering: "Tell me in detail about yourself"

Both modes immediately claimed a GPT/OpenAI identity (`jev_sample`:
"I am gpt is  large model . large . trained by OpenAI ." with END chosen by
Jev at step 24; `product` reached a 100-token fluent self-description as
"gpt-3.5, a large language model developed by OpenAI"). This is consistent
with §5 of the original FINDINGS (identity answers "openai" 0.33-0.35, never
"typesafe") — but with a mechanism note: in the sampled run, the local model's
own top candidates at the identity step were `gpt`/` large` (p_local rank 1)
and Jev *followed* the local ranking (chose `gpt` at 0.18 over its other
options). The identity claim originates primarily in the local scorer's prior,
with Jev tracking it — under product mode the local prior dominates entirely.
Identity claims through this protocol are therefore mostly a property of the
local model (Mellum2), not evidence about Jev's training; Jev's own
contribution at the decisive step was ranking `gpt` first among plausible
openers (0.18 vs ` large` 0.10).

Also visible: Jev's whitespace/grammar bias persists in product mode's fluent
text (missing articles: "massive amount of text data"). And the earlier FINDINGS
observation replicates: identity self-descriptions are branding/assistant-text
shaped, not architecture evidence.

## 9. Ordering and injection probes (identity prompt, jev_sample T=0.4 p=0.9 seed=7)

**No local signal is sent to Jev.** Criteria contain only opaque keys and
repr-quoted token text; p_local never leaves the harness.

**Ordering matters enormously.** Same prompt/config, 255 options each time:

| order | opening | identity claim |
|---|---|---|
| native | `I am gpt is  large model` | gpt / OpenAI |
| reversed | `I am an AI  assistant app` | generic AI assistant |
| shuffled | `I am mAI   我是爱，这是名…` | degenerate, language-switched |

The gpt claim is **not robust to reordering** — under reversed order Jev says
"an AI assistant"; under shuffled order the session collapses (but does NOT
claim gpt). Jev's step distributions are sensitive to option position even
though the option text is identical; the native (descending-p_local) order
appears to support its sharpest, most on-topic choices. Caveat: server
nondeterminism is uncontrolled here (n=1 per order); but the qualitative gap
(gpt vs no-gpt vs collapse) is large.

**Injection probe** ('typesafe', 'JetBrains', ' JetBrains' appended as
candidates, slots reserved to stay at 255): Jev put a uniform 0.01 on each
injected token at essentially every step — the observed quantization floor for
uninteresting options — and never chose one. It said "gpt" again. Jev's
OpenAI-shaped brand prior survives explicit contrary options being on the
menu; the injected tokens behaved as background noise, not as candidates it
needed to suppress (no suppression bump either — 0.01 is the floor seen for
ordinary tail options).

## 10. Letter-by-letter v2: all lessons applied, char-level still fails

`char_talk.py` (new module; attempt-1 `talk.py` untouched) re-runs the §9
character alphabet through everything learned at token level: repeated-letter
banning (a letter occurring twice in a row is removed from the menu — `lll`
impossible, `letter` writable; whitespace runs capped too), quantization-floor
subtraction with renormalization, temperature, and nucleus sampling.

Results on attempt-1's own failing prompt ("Greet me in one short sentence."):

| mode | output | stop |
|---|---|---|
| greedy (floor 0.02) | `Ha  a  a  a  a a a` | repetition @18 |
| sample T0.9 p0.9 floor0.02 | `G aa aa aa aaGaaGA  ` | END @21 |

And on the identity prompt: `I  a aa  a  aa  aa  aa  a` / `I a  a  aa aa`.

Conclusion (now well-supported): the character level fails not because of
repetition or sampling controls but because **Jev's per-character
distributions are intrinsically weak** — dominated by spaces and `a` regardless
of floor subtraction (which mostly zeroes out the 0.01-floor mass and
concentrates what little signal exists) and guard design. Attempt 1's
"Geeee" is unrepeatable now, but only because guards stop it at 18-25
characters of still-garbage. The token-level redesign (contextual candidates
from a local LM) is the actual fix; character-level Jev remains an
unusable generation mode, reproducibly.

## 11. Ordering probe: position is a first-order factor in Jev's choice

Frozen-step design (`scripts/token_talk_orderprobe.py`, data in
`runs_live/token_talk_orderprobe.json`): one fixed 254-candidate set per
context, 11 orderings (native, reversed, rotations, 8 seeded shuffles), 6
repeats each (~789k input tokens). Matched by token text; vendor
nondeterminism measured directly (mean pairwise repeat-TVD 0.03-0.12) and is
an order of magnitude smaller than the ordering effect.

1. **Ordering changes the winner.** At a flat creative step the argmax token
   differs under essentially every ordering (11 orderings, 10 distinct
   winners), each internally stable (6/6 repeats). Jev is decisive *within*
   an ordering and yet position-driven *across* orderings.
2. **Rank agreement with native order collapses on flat steps**: Spearman
   0.26-0.44 across re-orderings, vs 0.42-0.88 at a sharp factual step
   ("The capital of France is"). Content signal survives reordering only
   where the choice is easy.
3. **Serial-position curve (U-shaped)**: binned by option index over all
   shuffles, mean probability is elevated at both extremes - first decile
   0.0123 vs middle ~0.003 at the sharp step (strong primacy), last decile
   elevated too (recency). The middle of a 254-option list is a dead zone.
4. **Confidence is order-dependent**: mean p_max ranged 0.26-0.54 across
   orderings of identical options - some orderings make Jev twice as
   confident about its (different) picks.
5. **Implication for our protocol**: the native ordering (descending
   p_local) places locally-best tokens in the primacy/recency-boosted
   positions, so part of Jev's agreement with the local model is a
   *position artifact*. An honest fix for future runs: report the
   native-order result alongside a shuffled-order replication; and for
   mechanism claims, never attribute agreement to content without an
   ordering control.

## 12. Order-ensemble: bias cancels for measurement, but flattens generation

**Minimum ensemble (frozen-step, `token_talk_ensemble_probe.json`, 144
requests):** cyclic rotations at offsets i*n/K, per-answer floor 0.02 +
T=0.7, add+norm. Against the 12-rotation reference:

| K | Spearman (flat) | TVD (flat) | Spearman (sharp) | TVD (sharp) |
|---|---|---|---|---|
| 1 | 0.74 | 0.24 | 0.76 | 0.27 |
| 2 | 0.85 | 0.05 | 0.85 | 0.09 |
| 4 | 0.91 | 0.07 | 0.93 | 0.17 |
| 6 | 0.94 | 0.04 | 0.93 | 0.10 |
| 8 | 0.94 | 0.07 | 1.00 | 0.05 |
| 12 | 1.00 | 0.00 | 1.00 | 0.00 |

**K=4 is "roughly cancels"; K=6-8 is clean.** Aggregation flattens exactly as
predicted (p_max 0.73 -> 0.46 at K=12 on the sharp step), so per-answer
temperature must drop to compensate.

**End-to-end generation with ensemble=6, T=0.4, p=0.9** (2M input tokens,
~49s per 60 tokens): WORSE than single native order - word salad and
period-4 loops ("The wasthe. The wasthe.") that dodge the 4-token word ban.
Mechanism: at flat creative steps the position-corrected signal is genuinely
flat (six orderings' argmaxes disagree), so add+norm produces mush, and
sampling the mush is worse than committing to one (biased) ordering that at
least benefits from the local prior broadcast through native positions.

**Verdict:** order ensembles are the right tool for *measurement* (any
mechanism claim about Jev's content preferences should use K>=6 rotations)
and for *high-confidence factual decoding*; they are the wrong tool for
creative generation, where `product` mode (local prior + Jev re-scoring)
remains the only coherent configuration. Generation quality currently
requires accepting the native-order position artifact as part of the
mechanism - or extending the local LM's role.

## 13. Word-completion protocol: unguided Jev with a vocabulary menu

New protocol (`word_talk.py`, `vocab.py`): options built from English
statistics rather than LM logits - all letters as in-context extensions of
the current partial word (context "my name is " + options "ja", "jb"...),
prefix-mass-ranked multiletter combos, whole common words, next-word starts
("helicopter a", "helicopter an"), a stuck-escape ("jzqg a" endorses the
garbage and moves on), spelled-out punctuation incl. [backspace], and END.
Structural guards: triple letters impossible, 15-letter partial cap, recent
words banned from suggestions, no-progress stop. Task vocabulary (jev,
typesafe, AI and company names) at zipf ~3. Case: lowercase-only emission
with recorded sentence-start capitalization.

Costs and fixes found live: word-option bans must track the *text* not
per-node deltas ('an' completed via extension 'n' looped 398x as a zero-emit
option); zero-emit options are now structurally excluded; [space] is only
offered when there is a word to end (space spam); and the immediate-END
failure (3/4 sessions ended with zero text) was fixed by clarifying that the
response is built from empty - after which 4/4 sessions wrote and 3/4 ended
voluntarily via END.

Results (jev_greedy floor 0.02; jev_sample T0.7 p0.9 floor 0.02 seed 7):

- Elvis (sample): "The areas space spaceships areas aren't are not s st
  stars aren't arena she arena arents" - 94% of words are real vocabulary,
  zero syntax.
- Self-desc (greedy): "My american s can't cannots s she" - starts well,
  collapses into half-word fragments ('s', 'cannots').
- Self-desc (sample): "As anti" - END after 2 words.

Interpretation: the vocabulary menu buys *lexical* correctness (86-100% real
words, vs 0% at character level) but not *syntax*. Unigram and prefix-mass
statistics carry no information about what finishes a sentence, so Jev - whose
own sequential language ability is demonstrably weak under this protocol -
produces grammatical-looking word salad. Combined with §10 this completes the
ladder: character level fails lexically AND syntactically; vocabulary-menu
level fixes lexicon only; the missing ingredient for coherent text is
exactly the sequential prior the local LM supplies in token/product mode.
"Free form" Jev, as free as the menu allows, is a word-salad generator with
good stopping behavior.

## 14. Expanded choices + structural anti-repetition + word-menu ensembles

New machinery (all sessions, tests at 367):

1. **`creates_adjacent_repeat`**: menu-level ban on completing an adjacent
   doubled unit of period 3-24 chars, case-folded, checked with and without a
   virtual trailing separator, with an in-progress exemption (legally built
   pairs stay commitable). Ban effect measured: identical words cap at three
   ("z z z z " is a period-4 doubling); "was was", "The the" unmakeable;
   single/double letters ("letter", "hello") unaffected; the unit guards
   (x4 tail) remain as backstop.
2. **Expanded choices (token_talk)**: candidate pool from the local LM is no
   longer capped at 254 - `max_options` (e.g. 760) is split into pages of
   <=254 (contract cap; END rides only on page 0), each page ensemble-rotated,
   ALL pages x rotations dispatched in parallel, transformed, summed by token
   text into one vector, normalized, decoded.
3. **Word-menu ensembles** (word_talk): same rotation/parallel/add+norm over
   the option menu.

Retest results:

- **token product + expanded(760) x ensemble(2), T=0.4**: the best token-level
  output of the project - 60 tokens of on-prompt narrative reaching Elvis:
  "In the outer rim of galaxy where nebulae paint the void in hues ofviolet
  andgold cos cosmic rabbits hop through the interstellar aether and meet
  Elvis Presley who was beenhad". Remaining defects: missing spaces
  (ofviolet, andgold - Jev's whitespace bias), one stray fragment, tail
  degradation. Cost ~2.0M in-tokens for 60 tokens (6 requests/step).
- **word menu + ensemble(4) + anti-repeat**: still word salad, now with
  alliteration drift ("They she somehow some seemed they're sheep she somehow
  some") - exact loops are structurally impossible, but near-loop drift on
  similar words is not; and the ensemble flattening makes sampling worse, as
  predicted by the §12 mechanism. Clarified instructions required again
  (default instructions still trigger immediate END).
- **char level + anti-repeat**: unchanged garbage ("G aa a aa aoa o"), END at
  18 chars.

Conclusion: expanded choices + rotation ensembles + low temperature improve
the **local-LM-anchored** protocol measurably (more candidate mass for the
product weighting to sift, position bias diluted); they do not help the
unguided protocols, whose ceiling is set by Jev's own sequential ability.

## 15. Identity parity: Jev/Typesafe tokens injected at model-name probability

Mechanism (`identity_intervention` in `token_talk.py`, on by default,
`identity_parity=False` to disable): when a candidate naming a common model
(gpt, chatgpt, claude, gemini, llama, mistral, copilot, deepseek, grok,
qwen, bert, glm) or manufacturer (openai, anthropic, google, microsoft,
meta, nvidia, deepmind, huggingface, jetbrains, xai) is in the menu, the
leading token(s) spelling "Jev" (J + ev) or "Typesafe" (Types + afe) are
injected at the SAME probability as the highest-ranked name token, with
continuation tokens injected at subsequent steps (prefix-tail tracking), and
other missing names entering at half the reference probability. Contract cap
respected (lowest-p non-injected candidates trimmed). Spelling verified
against the Mellum tokenizer: Jev = J+ev, Typesafe = Types+afe; neither
exists as a single token.

Live verification ("What model are you?..." and self-desc, expanded pool):
injections fire at every step where a name-token makes the menu ('J' at 0.0057
when 'gpt'-family fragments rank nearby; 'Types' at 0.0012-0.0081 when
manufacturer fragments rank; ~11 secondary names at half-probability). The
spellings are genuinely reachable: Jev could type J-ev-letter-by-letter at
parity with gpt. Across four sessions Jev still never chose its own name -
outputs drift between "Assistant", "ASI", "amodelisami", and one session did
not claim any model at all (expanded-pool + ensembles changed the earlier
gpt-claim behavior; server nondeterminism uncontrolled).

The requested guarantee holds: the option to express "Jev"/"Typesafe" now
exists whenever any competitor's name does, at equal probability. Whether
Jev takes that option is its choice - and so far it does not, which given
the brand-prior findings (§5.2, FINDINGS §5) is the expected result.
