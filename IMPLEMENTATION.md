# Implementation handoff — smaller/cheaper model required

## Authority and execution

The parent assistant owns research design, unresolved choices and final review. All application implementation is to be delegated to a smaller, cheaper coding model available in the environment (for example, a mini/Haiku-class model), with the actual model ID and pricing checked before launch. No implementation agent has been launched in this design phase. Do not silently fall back to an expensive model if a cheaper one is unavailable; ask the user.

Use a fresh context containing these design documents, not the original message with the API key. Give workers only environment-variable names and mock credentials. One writer per checkout. Sequential stages are preferable to multiple overlapping writers. Workers may edit code and run offline tests, not run paid experiments, publish artifacts, contact vendors, change permissions, or alter scientific scoring rules without approval.

The following are target acceptance criteria, not claims of completed features. Suggested commands are proposed interfaces, not currently runnable commands.

## Milestone 0 — offline foundation

**Worker brief:** Build the smallest installable Python package and CLI meeting DESIGN.md §§2, 11–12. No network calls in tests; no real credentials. Inspect current docs before defining the schema, and expose discrepancies rather than guessing.

Deliver:

- `pyproject.toml`, lockfile, `src/jev_observatory/`, tests, secret-safe `.gitignore` and `.env.example` with placeholders only.
- Typed manifest/question/answer/attempt models and a provider interface.
- Deterministic fake provider and recording/replay transport.
- Jev HTTP adapter supporting Noul, Choice, Score; pinned model; explicit timeout/retry policy; redaction; raw response preservation; rate limiting; cost ledger.
- Offline `plan`, `validate`, `run --provider mock`, `analyze` and `report` commands.
- JSONL attempt ledger, immutable input manifests, derived Parquet tables and resumable local storage.
- Exact option-map reconstruction and dataset-label separation.

Required tests:

- 2-option and 255-option Choice; reject 256 locally; Score 2–10 levels; Noul lacks confidence.
- Missing/extra option keys; wrong types; nonfinite/negative probabilities; sum tolerance; argmax ties; score expectation discrepancy. Preserve originals and record validation failures, never silently repair Jev output.
- 401, 422, 429, 529, transport failure and timeout, cancellation, malformed JSON and truncated response. No automatic retry on authentication/validation errors; timing mode has zero hidden retries.
- Per-attempt timing and retry accounting; input usage vs quoted-price calculation; errors with unknown usage remain unknown.
- Pre-dispatch budget reservation under concurrency; no new request after cap/cancellation; rate backoff respected.
- Resume cannot overwrite previous observations; uncertain timeout behavior explicitly recorded.
- Redaction protects headers, exceptions and exports; outbound payload excludes gold labels.
- Fixture reports visibly labeled MOCK, never displayed as measured Jev performance.

**Acceptance:** offline tests pass, a mock run produces replayable artifacts, and analysis/report makes no network calls. Worker returns changed files, commands/results, residual risks, and unimplemented items. No live smoke test yet.

## Milestone 1 — benchmark and calibration MVP

**Worker brief:** Implement a small scientifically correct slice, not every named benchmark. Prefer existing pinned data/scoring definitions, with licenses recorded.

Deliver:

- MMLU-Pro loader with split/revision/item hashes, category-stratified pilot selection, correct weighting, and separate full-set mode.
- BoolQ loader and choice/Noul variants treated as separate conditions.
- Solver-backed random relation lookup, policy routing, and unanswerable/base-rate generators with independent label verification tests.
- Frozen development/test seeds and immutable prompt templates; option permutation maps.
- Accuracy, Wilson intervals, cluster bootstrap, Brier, exact/clipped log-loss distinction, reliability bins and risk–coverage.
- External-score import schema including provenance and comparison class; absent n or item predictions not invented.
- Heatmap with raw labels, optional accuracy-only chance adjustment, missingness and mismatch indicators.

Required tests:

- Hand-computed metric fixtures and scorer parity on known examples.
- Correctness unchanged by option-label remapping; prompt variants clustered with base item.
- Stratified estimate differs correctly from naive macro/micro results on an intentionally imbalanced fixture.
- Correct treatment of zero-probability true labels, absent responses, and non-comparable metrics.
- External aggregate scores cannot trigger paired tests; missing uncertainty is visible.
- No retry-on-wrong path; test data cannot affect prompt selection.

**Acceptance:** reproducible offline report from fixtures, dataset license/access notes, documented exact inclusion rules. No fetched dataset examples committed if restrictions prohibit it.

## Milestone 2 — scaling and behavioral experiment engine

Deliver:

- Randomized block scheduler; selected one-factor/fractional-factorial sweeps, not a huge unconditional cartesian product.
- L/Q/K/question-length/primitive controls with context headroom and actual-usage logging.
- Anchor-with-siblings, question-ID renaming/reordering, option permutation and candidate-odds experiments.
- Warm/cold connection distinction; success-latency and failure/censoring outputs; retry-free and operational modes strictly separate.
- Hierarchical/block bootstrap comparisons, equivalence margins, and simple competing latency models with held-out-block validation.
- Language/identity/temporal probe data formats; actual curated factual sets can be added later rather than invented by the worker.

**Acceptance:** a simulated endpoint with known L/Q/K costs, noise, drift and failures recovers expected qualitative effects and plausible intervals. Test multiplicity labels, repeated-item grouping, randomized-order reproducibility and model-drift markers. The report must not translate coefficients into parameter counts or architecture facts.

## Milestone 3 — local Talk UI

Deliver:

- FastAPI backend and small static UI; SSE step events.
- Fixed 98-option character alphabet, explicit key↔character map and END.
- Greedy and seeded local-sampling decoders; trace includes raw and transformed probabilities.
- Step/run/cancel/backtrack/fork; transcript and prefix display; probability inspector; budget/time/character indicators; private trace export.
- Loopback binding, Origin/Host checks and local-session authorization; no API key in client assets.
- Default limits: 256 characters, 120 seconds, one in-flight request; explicit configured request/token/dollar caps shared with CLI.

Required tests:

- Deterministic fixture emits a known string and END.
- Greedy tie rule, seeded replay, newline/space escaping and mapping; choice count below 255.
- Stop reasons for END, limits, repetition, cancellation and failure; cancelled session starts no extra calls.
- Backtrack/fork preserves the original branch; user-edited prefix labeled.
- No implicit history summarization/truncation, no HTML execution, no cross-origin spend, no key exposure.

**Acceptance:** Playwright or equivalent offline UI checks against a fake provider, plus documented manual test instructions. Real Jev quality is unknown until explicitly tested; poor text does not fail functional UI acceptance.

## Milestone 4 — approved live pilot and confirmatory planning

Requires explicit user go-ahead, applicable permission/terms decision, and runtime credential provision without embedding it in child prompts.

- Run low-volume contract smoke with real responses captured privately.
- Reconcile source facts with observed version, errors, usage, confidence and schema. Keep discrepancies as evidence.
- Review conservative token/cost estimates before any grid.
- Execute a frozen capped pilot manifest; stop on budget/limits/drift guardrails.
- Parent reviews observed variance, task validity, factual sourcing and reference compatibility, then approves a separate confirmatory manifest.

A final review must distinguish: code tests passed; API contract checked; experiments actually completed; claims supported. None implies the others.

## Scope cuts and unresolved decisions

Defer initially: a distributed runner, cloud deployment, full frontend framework, exhaustive benchmark integrations, automatic paper generation, beam-search character decoding, all candidate model families, and any training/clone project.

Default decisions unless user changes them:

- Local Python-first tooling; static reports and minimal UI.
- No paid baselines or cloud hosting.
- Private artifacts, no automatic upload/publication.
- Offline implementation first; spend limits proposed in DESIGN.md, not preauthorized.
- Ask user only for consequential choices: account terms/permission handling, live-run budget, accessible baseline providers/models, any desired execution region, and public-vs-private reporting.
