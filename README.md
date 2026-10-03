# Jev: Not Frontier, But Still Worth Your Attention

An independent, hands-on evaluation of TypeSafe AI's `jev-1.13.0` API:
16,379 live benchmark requests across three frozen suites, a 987-call
architecture probe, a multi-protocol Talk-to-Jev program, and a
token-counting study, and a matched cheap-model baseline sweep on OpenRouter
(12 models over the identical frozen items) - every figure re-derivable from
the published aggregates in this repository.

**The report page: <https://jevresearch.github.io/Jev-Research/report/>**
(rendered; the source is [`report/index.html`](report/index.html)).
Measured September 2026 against the service-reported `jev-1.13.0` model
string; report revision 3 (2026-10-03); an independent evaluation by
JevResearch, unaffiliated with TypeSafe AI.
Headline (all-requested scoring): MMLU-Pro 82.7%, GPQA Diamond 76.5%,
~73 ms fixed + ~6 ms/1k-token proxy-reported upstream service time, ~$0.28 per 12k-question
MMLU-Pro run - a small
model with a probability read-out in place of a generation
head, not a frontier system.

## What's here

| path | contents |
|---|---|
| [`report/`](report/) | the report (single page, figures embedded); rendered at <https://jevresearch.github.io/Jev-Research/report/> |
| [`counterpoint/`](counterpoint/) | a short counterpoint on the follow-up probe round (rendered at <https://jevresearch.github.io/Jev-Research/counterpoint/>) |
| [`assets/`](assets/) | standalone chart pages (same figures, un-embedded) |
| [`src/`](src/), [`scripts/`](scripts/), [`tests/`](tests/) | the harness that ran everything (public domain) |
| [`runs_benchmark*/`](runs_benchmark/) | per-stage derived aggregates (scores, weighted scores, calibration, usage) + freeze manifests with SHA-256 of every input |
| [`runs_archprobe/`](runs_archprobe/) | architecture-probe analysis, per-call rows, tokenizer studies, the probe2 follow-up battery, billing |
| [`runs_matched_cheap/`](runs_matched_cheap/) | matched cheap-model baselines (OpenRouter): freeze, smoke, per-call results, summaries, spend |
| [`runs_live/`](runs_live/) | Talk-to-Jev traces (character / vocabulary-menu / token programs), probes, billing, findings |
| [`data_report/`](data_report/) | cost model, billing roll-up, architecture + lattice audits, size estimate, probe-2 plan, benchmark calibration/paired diagnostics, raw-recoverability inventory and rerun plan, the v4r1 active baseline summary (complete/partial cell status), and the follow-up round's corrected statistics plus the public knowledge exports (questions without gold; answer key with source URL + SHA-256) |
| [`docs/`](docs/) | research write-ups: architecture probes, comparable scores, the vals.ai leaderboard extract, the Talk program |
| [`ARCHITECTURE-ANALYSIS.md`](ARCHITECTURE-ANALYSIS.md) | the full architecture reconstruction: card, evidence, alternatives ledger, next probes |

## Reproducing the page from the data

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python scripts/report/aggregate_billing.py   # -> data_report/billing_totals.json
.venv/bin/python scripts/report/build_report.py        # -> report/index.html
.venv/bin/pytest                                       # saved-run regressions skip in this checkout
```

The page embeds no hardcoded results: [`build_report.py`](scripts/report/build_report.py)
reads every number from the JSON aggregates at render time. Re-running the
generators needs the full working tree (the dated `runs_live/` raw response
dirs and benchmark per-item lists are not bundled). Re-running the *live*
experiments requires your own `TYPESAFE_API_KEY` (see `.env.example`) and hits
the paid API.

Re-rendering the page (the bundle's G4 gate proves the shipped page
byte-for-byte) is re-deriving prose and figures from the published
aggregates; it is **not** an independent re-scoring. Independently re-scoring
would require the private raw responses and the licensed item sets, which are
not redistributed here; the freeze manifests pin their identity by SHA-256.

## Data availability and licensing

Benchmark **item text is third-party licensed content** (MMLU-Pro is
CC-BY-NC-4.0; GPQA, HLE and ARC-AGI-2 are under their own terms, some gated)
and is **not republished here**. This repository publishes instead:

* derived aggregates per stage (accuracy, Wilson CIs, Brier / log-loss,
  weighted scores, usage roll-ups) with per-item lists stripped;
* freeze records (`frozen.json`, `preparation.json`) carrying SHA-256 hashes
  of every source file and request plan - holders of the datasets can verify
  item identity against those hashes;
* our own synthetic prompts verbatim (the Talk traces, the architecture-probe
  rows, the tokenizer samples - none is third-party licensed);
* short quoted third-party text used for exact replication (e.g., Archer
  Hume's payout-probe prompt, reproduced in
  `scripts/benchmark/followup_battery.py`) - such quotes retain their
  original authorship and are reproduced with attribution, not as our own;
* the complete cost model and billing roll-up;
* benchmark calibration and paired diagnostics aggregates
  (`data_report/benchmark_diagnostics/`, no item text) and the v4r1 active
  baseline summary: 59 of 59 (model, dataset) cells complete over their full sampled denominators and charted, under a versioned status (v4r1) - never presented as obsolete legacy scores;
* availability of per-item prediction evidence (item ids, correctness,
  displayed probabilities): it exists in the working tree
  (`runs_matched_cheap/v4r1/active_items.jsonl`, stage `per_item` lists) and
  is available to holders of the datasets on request, but is not
  redistributed here under the same item-id policy that strips `per_item`
  lists at publish time.

External comparison scores were fetched from public publisher pages; sources
and access dates are recorded in `docs/modern-comparison/canonical/` and
`docs/modern-comparison/COMPARABLE-SCORES.md`.

## License

Everything in this repository - code, report text, figures, and derived data
tables - is dedicated to the public domain under the Unlicense (see `LICENSE`).
No attribution is required or requested; no rights are reserved. Third-party
benchmark datasets referenced by our measurements remain under their own
licenses, and no dataset item text is redistributed here. Jev is a trademark of
TypeSafe AI; this project is independent and unaffiliated (see `NOTICE.md`).
