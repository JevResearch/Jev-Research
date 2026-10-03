# Follow-up probe round: method appendix (2026-10-03)

Companion to the [counterpoint](../../counterpoint/index.html) and the main
[report](../../report/index.html). Everything here is our own synthetic
probe material or derived statistics; no licensed benchmark item text is
reproduced.

## Design and sample units

978 recorded requests (834 Jev + 144 reference-model), 0 errors, in one
frozen follow-up battery (`runs_archprobe/followup_20261003/`,
`plan_frozen.json` pins the stimuli; per-call rows are `rows.jsonl`,
sanitized). Follow-up API spend $0.0953044 (usage/rate-derived ledger;
project exposure stays under the $80 owner cap). Valid evidence is v2
(96 relational + 64 isolation rows) and v3 (byte-matched timings: 8
calibration + 64 timed rows = 72 rows in 8 randomized blocks). v1
refcard/isolation results are invalid and are retained raw only: the card
leaked into the state for both placements (see
`data_report/followup_20261003/v2/analysis_v2.json`).

## Isolation and shared state (Archer replicated)

With a reference code supplied in `state` or the target question's own
instructions, Jev selected it 16/16 in each placement; the same code hidden
in a sibling question was never selected (0/16), and "no code" was selected
32/32 there and in the absent arm. Exact relational state was read 48/48,
a code buried in an option 39/48, balanced across the beta/gamma gold keys.
The customer/unknown log-odds shift from appending an irrelevant option is
-0.258 (10 blocks, duplicates pooled per the author's protocol; block
bootstrap 95% [-0.334, -0.178]) - matching Archer's reported -0.28. These
reproduce his behavioral findings; they do not identify the architecture.

**Cached-state route (theoretical note).** A block-masked encoder can allow
state↔state and question↔question attention both ways and question→state,
while blocking state→question and question→question cross-talk. The state
KV is then independent of the questions and shareable across them, giving
shared-state caching *without* a causal mask. This is a counterexample to
"only a causal decoder can do this" - it is **not** a claim about Jev's
actual mask, which is unobserved. A causal decoder remains a sensible
premise.

## Question tokens: cost, not identity

Total wall time for identical text placed in `state` vs in a question gave a
ratio of 0.989 (whole-block bootstrap 95% [0.915, 1.045]) - baseline-dominated
and not an added-token-cost statement. The incremental per-1k ratio is
1.112 with whole-block bootstrap 95% [0.589, 4.155], which **includes 2**:
a physical 2× cost for question tokens is not refuted. Current deployment
returns no upstream service-time header (header names verified absent), so
these are client wall times.

## Counter vs encoder (v3, byte-matched uploads)

Equal wire bytes, four styles (Latin words, letters, digits, CJK). At
24,000B the reported counter's long-token counts are Latin 3,846, CJK 5,366,
digits 8,033, letters 12,482: digits cost *more* tokens than Latin/CJK and
*fewer* than letters only. Wall time: digits ran +31 ms vs letters with
whole-block bootstrap 95% [-46.6, 75.3] - includes 0, not decisive.
Byte-matched long digits were ~96 ms slower than Latin at 24,000B (95%
[76.3, 115.0]), so processing differs beyond upload bytes. Held-out-style
RMSE: reported-count predictor 39.97 ms vs Qwen3 40.20 ms - a 0.23 ms
spread well inside block noise, so the encoder comparison is
**inconclusive** and never a positive Qwen-base classification. Long o200k
digit grouping is a property of that tokenizer only; nothing here rules it
in or out. The joint negative wire-bytes coefficient under collinearity is
not interpretable.

## Knowledge horizon (corrected)

Old P5 rows carried label blemishes. The corrected artifact
(`data_report/followup_20261003/p5_horizon_corrected.json`) preserves all
104 original rows and applies **4 relabels** (Anchorage summit: Jul→Aug
2025, with independent published photography coverage) and **4 exclusions**
(Starliner "crew return" rows: the crew returned Mar 2025 on Crew Dragon and
the correct month was never an option; NASA source pages). False controls
are 16/16 invented events refused (the old mixed count is superseded). The
fresh cohort is 24 unique dated-fact items × 2 option rotations = 48
repeated-measures rows: 4/16 recent facts correct in both rotations (three
Oct-2024 facts plus one Mar-2025 event), 1/12 of the 2025 facts. This is a
limited, news-heavy set: strongest on 2024-era facts, unreliable on 2025
news. **No training cutoff is asserted.** Candidate Qwen models show a
similar recent-facts weakness but are not unique as parents; their published
metadata cutoff is unverified. Reference models recovered 24/24 rows after
corrected handling (no cap errors).

## Provenance and third-party text

Source pages are recorded as URL + SHA-256 of the fetched page in
[`knowledge-answer-key.json`](../../data_report/followup_20261003/knowledge-answer-key.json);
no source page bodies are redistributed. Archer Hume's
[article](https://archerhume.com/posts/jevs-architecture-unmasked/) is
linked, not copied; his payout-probe prompt text is reproduced in
`scripts/benchmark/followup_battery.py` for exact replication and retains
his authorship (attribution in the file and the bundle NOTICE). Official
SDK reference for the published confidence rule:
[confidence_metrics.py](https://github.com/typesafe-ai/system-one-adapter-python/blob/fb52b1030b7fc1f4f1cf39910afa5da54f9835e3/src/system_one_adapter/_utils/confidence_metrics.py)
(pin fb52b103) - the adapter's published code, not proof of unseen server
code. TypeSafe's
[primer](https://docs.typesafe.ai/introduction/machine-learning-primer)
describes RLCD over pretrained models without naming a base.
