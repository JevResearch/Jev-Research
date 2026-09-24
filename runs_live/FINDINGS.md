# Live pilot findings — jev-1.13.0 (v2, audit-corrected)

Run date: 2026-09-18 (post-audit revision 3). Live totals verified from
artifacts including Talk: **2,283 calls, 1,285,908 reported input tokens ≈
$0.0540** at the quoted $0.042/Mtok (see BILLING.json; unknown-usage calls on
rejected requests excluded).

This revision corrects FINDINGS v1 after an independent audit. Corrections are
marked ⚠️. Evidence labels per DESIGN.md §1.

## 1. Capability (observed endpoint behavior — verified)

- **MMLU-Pro stratified pilot (n=200, direct 0-shot choice): 0.850** — verified
  item-by-item against the source parquet (question, options, gold all match).
  Independent stratified bootstrap CI ≈ [0.799, 0.895]. A real, strong result
  for a no-reasoning protocol; still a pilot, contamination untested, and the
  historical comparators (Claude-3.5-Sonnet 0.761, GPT-4o 0.726, protocols
  unknown/likely-CoT) remain contextual only.
- ⚠️ **Label-quality caveat**: at least two "confident errors" look like wrong
  benchmark labels (economics interest-rate item; biology blood-pressure item).
  Adjudication notes should accompany any claim about confident hallucination.
- **BoolQ paired validation run (500 records × both encodings, 1,000 calls)**:
  choice **0.898**, noul **0.896**, paired difference **+0.002, McNemar p=1.0**.
  ⚠️ The v1 claim of an "8-point encoding gap" was a **sampling artifact** — the
  two conditions used different, non-overlapping question sets from the *train*
  split. With the same records, the encodings are equivalent in accuracy.
  Effective yes/no answers agree on 493/500 records.
- **Calibration (corrected, and corrected again after re-audit)**: the reusable
  paired scorer (`paired.py`) recomputes Brier on the *binary P(yes)-vs-gold*
  scale. On the paired BoolQ validation run: choice 0.0899, noul 0.0783;
  **the paired Brier difference favours noul by 0.0115 (CI95 [0.0055, 0.0177])** —
  an earlier "choice is much sharper" reading (0.014 vs 0.078) was an ad hoc
  miscalculation. Noul P(yes) reliability is monotone and near-diagonal but NOT
  perfectly calibrated (e.g. mean P≈0.135 bin → 38.5% actual yes; ECE-10 0.068
  noul vs 0.083 choice). MMLU exact log loss is **infinite** (5 zero-probability
  gold outcomes, some likely label errors); clipped mean 1.098. Noul reliability
  in v1 was computed against the wrong target (correctness, not gold).

## 2. Contract reconciliation (observed endpoint behavior)

1. **Probabilities are 2-decimal quantized** (11,193 values, zero exceptions) —
   this is about the exposed representation, not necessarily internal precision.
2. ⚠️ **Six choice/argmax discrepancies are real, not float ties**: displayed
   gaps of exactly one quantum (e.g. choice at 0.17 vs max 0.18). Monotone
   rounding of a pre-quantization argmax cannot reorder options, so the
   decision and the returned table disagree — consistent with separate
   post-processing of the decision and the distribution. Validator now
   distinguishes float-noise ties (warning) from real gaps (error).
3. **Distributions do not always sum to 1**: max observed deviation 0.01 (17
   cases in the sweep).
4. **A token limit exists but its cause was NOT isolated** ⚠️: the largest
   request (Q=64×K=255, ~655k JSON chars, ~25k input tokens billed on siblings)
   failed with `max_tokens_exceeded`. v1's claim of an "output ceiling" was
   unsupported — input, output, or combined accounting are all consistent with
   the evidence.
5. **The endpoint is non-deterministic**: 30 byte-identical payloads produced
   **13 distinct response signatures** (distributions differ slightly). All
   single-call equivalence claims must be read against this noise floor (TVD
   noise ≈ 0.01–0.02).
6. Model field stable (`jev-1.13.0`) across 482 calls; no drift observed.

## 3. Mechanism (supported behavioral inferences; architecture NOT identified)

- **Question isolation holds on non-saturated anchors**: three mixed-signal
  anchors (distributions up to 0.98/0.02) evaluated alone, with 8 irrelevant
  siblings, with 8 conflicting siblings (renamed IDs): worst TVD **0.02** —
  at the serving-noise floor. ⚠️ v1's isolation test used a single trivially
  saturated anchor ("no" at 1.0) and was weak evidence; v2 is the meaningful
  version. Still not proof of independence — equivalence at the noise floor.
- **Batching is real and large**: one batched 4-question call ≈ **261 ms** vs
  4 separate calls ≈ 4×279 = **1,117 ms** (~4.3× throughput per question), with
  answers changing by at most TVD 0.02 (≈ serving noise).
- **Serving jitter is large**: median 297 ms, stdev 82 ms (CV 0.28) on
  identical payloads. ⚠️ The v1 scaling coefficients (R²=0.11) were noise-
  dominated and are retracted; the fit machinery is fixed (block-held-out
  validation) and validated on the simulated endpoint (R²≈0.999) but the live
  sweep needs more blocks to estimate scaling effects.
- **Candidate-odds (ambiguous task)**: billing/refund_flow odds 0.71–0.94
  (base) vs 0.68–0.94 (+irrelevant options) — overlapping ranges; no gross IIA
  violation detectable at 2-decimal quantization + jitter. ⚠️ v1's odds task
  produced one-hot answers and was untestable.

## 3b. Controlled mechanism experiment v3 (post-audit; LEAD-GATE-corrected reading)

⚠️ **LEAD-GATE audit 2026-09-18 — the statistical analysis originally reported here was invalid
and the design is confounded. Do not cite the original numbers as evidence of ID visibility.**
The v3 analysis converted six calls per arm into 15 within-baseline and 36 cross-arm pairwise
distances and then bootstrapped those dependent distances as independent observations; that
pseudoreplication invalidates the reported CIs and every "CI excludes 0" statement. The executed
arm order was fixed (not randomized), and the renamed/reordered arms changed insertion positions
or the anchor's key position alongside IDs, so renaming, ordering and sibling presence cannot be
separated.

What survives (observed means; corrected paired-repeat reconstruction in
`docs/review-gate/audit/mechanism_v3_reconstruction.json`, exploratory only, sample units are the
six repeat pairs per anchor/arm — NOT pairwise distances):

- **dup-charge** (mean P(yes) ≈ 0.380): paired signed P(yes) shifts vs the alone arm are small
  (−0.010 to −0.007 across +8 siblings / renamed IDs / reordered); exact sign-flip p ≥ 0.59.
- **subscription-proration** (mean P(yes) ≈ 0.802): paired signed shifts +0.017 (renamed IDs),
  −0.007 (+8 siblings), −0.007 (reordered); exact sign-flip p ≥ 0.19 at n=6 pairs.

Interpretation (supported): small anchor-distribution shifts were observed on one of two anchors.
No causal interpretation is possible; **no falsification signal for the "ids never reach the
model" claim is carried by this experiment.** A properly randomized repair design (12 blocks ×
2 already-screened anchors × 4 arms with rename-only and reorder-only contrasts) is specified in
`docs/review-gate/audit/mech_dryrun_spec.json` (+ `mech_dryrun_mappings.json`) and has NOT been
executed — any live run requires separate authorization.

## 3c. Fresh capability evaluation (generator_version=2, test-seed family)

⚠️ **LEAD-GATE audit 2026-09-18 — read the headline numbers as easy, highly templated tests.**
Seed arguments in the multihop/date/policy generators did not change substantive items (only
ID/order-style fields), so this is NOT true held-out seed performance, and the 450/450 choice
accuracy is NOT broad reasoning or calibration evidence. Every date interval in the templates
(12/29/30/31/33/45/50 days) crossed a month boundary; gap 50 crossed exactly one month.

600 items from six solver-verified generator families, direct choice/noul, no retries:

- **Choice tasks (450 items): 1.000 accuracy** (multihop, relation lookup,
  relation negation, policy routing, unanswerable-detection, base-rate);
  Brier 0.0002, ECE-10 0.0032, exact log loss 0.0033. Zero contract violations.
- **Noul tasks (150 items): 0.853** [0.788, 0.901], driven by the date-window task:
  - gap 5d: 1.00, 12d: 1.00, 29d: 1.00, **30d (boundary): 1.00**
  - **gap 31d: 0.08**, 33d: 1.00, 45d: 1.00, **50d: 0.17**
  - Observed fact: failures concentrate in the two largest-gap templates. ⚠️ The earlier claim
    that this isolates "exactly where month-carry arithmetic is required" is **retracted**: every
    template crossed a month, and the narrow template set cannot separate month-carry from other
    long-interval handling. A corrected design (explicit 30-day-month calendar, independently
    varied start day and gap, same-month and crossing cells where mathematically possible, labels
    verified by two independent methods) is specified in `docs/review-gate/audit/fresh_v3_*.json`
    and has NOT been executed.
- Multihop binary noul: 1.000 (n=50).

## 4. Talk to Jev (observed endpoint behavior)

Character-level external autoregression fails under greedy decoding
(`Geeee`, `A    `, `AAB`; repetition guard), consistent with the vendor's own
jaggedness disclosure. First-step distributions are coherent (G 0.28 / A 0.25 /
H 0.23 for "Greet me...") — informative but not usable generation. Per-step
payloads cost ≈950 reported tokens each (~3.7 chars/token server-side), so our
1-token/char budget estimate over-reserves ~4× (safe by design).

## 5. Behavioral provenance (exploratory; ancestry NOT established)

- **Temporal knowledge**: 2022/2023/2024 facts answered correctly at p≈0.97–1.0;
  the nonexistent-event control was correctly refused (p=1.0 closed-book).
  Knowledge extends to at least late 2024. ⚠️ This is knowledge availability,
  not a training-cutoff measurement.
- **Abstention behavior (notable)**: offered a "cannot say" option, the model
  **abstains on the Nov-2024 US-election question (p=0.71)** despite answering
  it correctly (p≈0.98) under forced choice — conservative abstention rather
  than absent knowledge; it also refuses the fictional-premise control.
- **Identity probes are low-value and prompt-suggestible**: mostly "unknown"
  (0.54–0.61), "openai" second (0.33–0.35), never "typesafe"; and with a stated
  fictional origin ("Meridian Labs") it accepts the false premise at **0.97**.
  These outputs are branding/assistant-text shaped, not ancestry evidence.
- **Multilingual**: the 2024 population question is answered correctly (India)
  in all six languages at p≈0.97–0.99 — no linguistic asymmetry on this item.

## 6. What this does and does not establish

**Established (observed):** accuracy figures above; paired accuracy-equivalence
of the BoolQ encodings at n=500 (McNemar p=1.0); exploratory paired Brier
difference favouring noul (0.0115, bootstrap CI [0.0055, 0.0179]); noul P(yes)
reliability near-monotone with residual miscalibration (ECE-10 0.068);
large batching gains; 2-decimal
quantization; response nondeterminism; real choice/argmax contract gaps; the
token-limit failure mode; live Talk failure mode; knowledge through late 2024;
conservative abstention when offered; prompt-suggestible identity answers.

⚠️ **Not established by the corrected reading:** question-ID visibility or
invisibility (the v3 renaming result is confounded and pseudoreplication made
its significance claim invalid); a month-carry failure mechanism (templates
cannot isolate it); broad reasoning/calibration ability (fresh 450/450 was easy
and highly templated).

**Not established:** architecture, parameter count, training compute, foundation
checkpoint, or ancestry. The infinite exact log loss means "cannot hallucinate"
fails in the strictest probabilistic sense (0.00 on true labels), but those 5
cases include likely label errors — adjudication is required before any claim
about confident factual failure rates.

## 7. Billing reconciliation

| Component | Calls | Reported input tokens |
|---|---|---|
| Smoke | 4 | 1,274 |
| MMLU-Pro pilot | 200 | 114,054 |
| BoolQ train (unpaired, v1) | 200 | 90,193 |
| Isolation + odds (v1) | 18 | 11,934 |
| Sweep | 46 | 255,554 |
| Talk (3 sessions) | 14 | 41,921 |
| BoolQ paired validation (v2) | 1,000 | 439,752 |
| Mechanism v2 + provenance | 129 | 62,097 |
| **Grand total** | **1,611** | **1,019,855 (≈ $0.0428)** |
| **v2 additions** | **1,129** | see `runs_live/*/run_summary.json` |

All ledgered attempts, raw bodies (redacted), manifests, and plans are under
`runs_live/`; hash verification passes for every manifest.
