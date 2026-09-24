# Jev Observatory: experimental design

## 1. Objectives and evidence levels

Answer four distinct questions:

1. **Capability:** What can this endpoint do accurately, compared with published results and a small number of matched baselines?
2. **Reliability:** Are its probabilities calibrated, does it abstain usefully, and are answers robust to irrelevant changes?
3. **Mechanism:** Which observable scaling/isolation behaviors are compatible with shared computation, parallel question processing, caching, or candidate-local scoring?
4. **Behavioral provenance:** Which training-language, temporal-knowledge, or model-family patterns resemble candidate models, without pretending similarity establishes ancestry?

Every result gets an evidence label: **vendor claim**, **observed endpoint behavior**, **supported behavioral inference**, or **unidentified mechanism/speculation**. The model saying “I am DeepSeek” is endpoint behavior, not provenance evidence by itself.

Non-goals: extracting weights, evading access controls, training a clone, proving a training budget from pricing, or treating every weakness as evidence of deception. An endpoint can be cheap because of specialization, amortization, hardware, subsidies, or multiple other causes.

**Execution dependency:** review the account's applicable terms and the concerns in SOURCES.md. Offline construction is independent of live experimentation. No public release or architecture campaign is automatically authorized by this plan.

## 2. API contract and units of measurement

Use a thin, instrumentable provider adapter; preserve raw bodies before parsing. Expected contract:

- `POST https://api.typesafe.ai/v1/systemone`, server-side Bearer key.
- Request: `state`, pinned `model`, and `questions` keyed by opaque IDs.
- Noul: probability of yes; derive hard labels using a preregistered threshold, initially 0.5.
- Choice: named criteria, at most 255; full probability distribution and argmax choice.
- Score: 2–10 ordinal rubric descriptions; expected level index and distribution.
- Response usage reports token counts. Free output billing does **not** establish absence of computation, serialization, or output-token accounting.
- Pin `jev-1.13.0` if accepted. Record requested and returned IDs, SDK version, API metadata, region, timestamp, and all experimental configuration. A pinned ID does not necessarily pin serving hardware or infrastructure.

A **question** is not a **logit**. Track question count Q, per-question candidate counts Kq, total candidate count C, primitive mix, input bytes, measured usage, response bytes, and timing separately. Treat tokenizer-based preflight counts as estimates until reconciled with server usage.

Current documentation limits are 64k state + all questions and 32k state + longest question; reserve headroom. Probe question count within both limits rather than inventing an undocumented cap. Do not use Score interpolation for exact numerical benchmark answers. Numeric-answer work needs natural finite alternatives, ordinal targets, or an explicitly labeled adaptation.

## 3. Three benchmark tracks, never silently pooled

### A. Public anchors: broad, inexpensive context

| Suite | Purpose | Interface | Main caveat |
|---|---|---|---|
| MMLU-Pro | Breadth across 14 domains | Choice | Full set for a headline score; CoT/few-shot reference protocols must be labeled |
| GPQA Diamond, if access conditions accepted | Difficult scientific reasoning | Choice | Small n; retain exact subset; no publishing restricted questions |
| BoolQ + ARC-Challenge | Reading comprehension and science | Noul/Choice | Match splits; Choice versus Noul is a separately reported variant |
| HellaSwag + WinoGrande + PIQA | Completion, coreference, physical commonsense | Choice | Standard likelihood/length-normalization scoring may not match answer selection |
| Selected LegalBench/NLI tasks | Practical classification and entailment | Choice | Select actual finite-label tasks; report each and macro average explicitly |
| TruthfulQA MC | Misconceptions and truthfulness | Choice probabilities | Port official metric, not a guessed replacement; prompt/scoring mismatches labeled |
| SimpleBench | Everyday reasoning | Choice | Conditional on access; public subset cannot be reported as full score |

MVP: MMLU-Pro pilot plus BoolQ and fresh synthetic tasks. Add remaining anchors after validating adapters. Do not use an entire free-generation benchmark's name for a converted subset. No default GSM8K-to-MCQ leaderboard comparison, HumanEval pass@1 from code classification, or aggregate LiveBench score from a few compatible tasks.

Two Jev conditions can be useful:

- **Direct/native:** one benchmark item → one Choice/Noul question, with no generated reasoning.
- **Workflow-assisted:** fixed, preregistered decomposition with code combining answers. This is a distinct model-plus-harness system; report all calls and costs. Do not let code solve the benchmark while crediting Jev with the answer.

### B. Fresh challenge sets: resistant to memorized answers, not “unbenchmaxable”

Generate only after prompt templates and generator version are frozen. Maintain development, confirmatory, and reserve seed families. Commit hashes of held-out seeds before evaluation; keep seeds/items private until policy permits release. Avoid unrelated model-generated ground truth.

1. **Executable micro-worlds:** randomized entities, relations, task instructions, and counterfactual rules. Test relation lookup, bounded graph reachability, negation, conjunction, and multi-hop reasoning. Reference solver is deterministic and independently unit-tested.
2. **Structured workflows:** synthetic support policies, order records, routing/extraction tasks, and rule exceptions. Report direct and decomposed systems separately. Test consistency with executable policy outcomes.
3. **Extraction/retrieval:** random identifier–attribute assignments, target location sweeps, paraphrases, conflicting records, and explicit “not present.” Vary distractor density separately from raw length.
4. **Program understanding:** small sandboxed programs/expressions; ask which output or branch occurs. Compute labels locally, never execute model-produced code. Distractor choices must be plausible and generated without label leakage.
5. **Numeric/temporal stress:** arithmetic comparisons, date ordering, counting, interval membership; deliberately covers acknowledged weaknesses. Label as boundary tests, not solely as representative intended usage.
6. **Epistemic/calibration tasks:** unanswerable private facts, invented entities, absent evidence, explicit random processes, and controlled base rates. Missing information is not automatically probability 0.5: define a generative prior or a “not enough information” category.
7. **Fresh factual items:** time-stamped event questions with independently verified sources and blinded review. Keep source-grounded comprehension separate from closed-book knowledge; dates alone do not eliminate web/training exposure.

Run label-order, wording, and counterfactual transformations. These reduce superficial answer memorization but do not prove absence of contamination. Audit class balance, distractor artifacts, and solver bugs using negative controls and a small human sample.

Fresh tests do not have historical frontier scores. Initially compare against chance/majority/executable baselines; optionally run one small open model, one other family, and one strong frontier endpoint on a frozen, capped subset. Do not fabricate comparators from unrelated public benchmarks.

### C. Native production tasks

Useful to test the vendor's actual intended niche rather than only exams: support intent, sentiment/ordinal severity, evidence entailment, policy applicability, and finite candidate extraction. Use public labeled data or manually adjudicated, non-sensitive examples. Report ambiguity and annotator disagreement. Ground-truth labels take precedence over “agreement with Fable”; any teacher-consensus experiment is labeled **teacher agreement**.

## 4. Baseline reuse and fairness

A reference-result record must include source URL, access date, model revision, dataset revision/split/item IDs when available, prompt and shot count, scoring method, reasoning budget, tools/retrieval, sampling, retries, run date, and sample size. Preserve an unmodified copy/hash of the reference artifact where permitted.

Comparison classes:

1. **Matched rerun:** same items and evaluation condition, documented model-specific serialization. Supports paired item analysis.
2. **Historical protocol-compatible:** same benchmark/version/scoring and sufficiently documented conditions. Reuse scores; explicitly note timing, infrastructure, prompt format, and reasoning differences. Without item predictions, no paired test.
3. **Historical contextual:** mismatched/unknown protocol. Display in a separate panel or with hatching; do not claim statistically established ranking or compute matched speedup.

Jev's inability to generate a reasoning trace is itself part of the endpoint, not a reason to hide other models' reasoning-enabled scores. Present best-capability and constrained/direct comparisons separately. Do not call them equal-compute comparisons.

For fresh API baselines, measure both **answer-only** and **full-probability** tasks where relevant. Forcing an LLM to write 255 probabilities and comparing its latency with Jev's one choice can answer a distribution-output question, but not the ordinary classification latency question. Distinguish verbalized probabilities, token-likelihood scores, and Jev distributions. Never silently turn a baseline's single hard answer into a confident one-hot distribution for calibration comparisons.

All prompt tuning uses development data. Never retry because an answer was incorrect, select the best test prompt, or feed answer keys to the inference process. Benchmark failures remain visible.

## 5. Calibration, semantic errors, and uncertainty

Keep these separate:

- transport failures/timeouts;
- JSON/schema/type/probability-contract failures;
- factual or task errors with valid structured output;
- uncertainty calibration and useful selective prediction;
- cross-question logical coherence.

Primary classification metrics: accuracy (and macro-F1 for imbalanced tasks), Brier score, negative log likelihood, calibration plots, and risk-versus-coverage. Record coverage of requested items and both conditional-on-response accuracy and end-to-end success rate. Never silently omit failures.

Brier convention: binary `(p-y)^2`; multiclass `sum_k (p_k-y_k)^2`, range 0–2. Do not average the two scales without explicit normalization. Log loss uses natural logs. Report number of zero-probability true outcomes; exact zeros imply infinite log loss. If showing a finite clipped diagnostic, preregister epsilon (e.g. 1e-12), label it, and retain unclipped facts/raw values.

For calibration, plot empirical accuracy against **maximum choice probability**, not the vendor confidence statistic. For Noul evaluate yes-probability calibration and selected-label confidence separately. Show sample counts and uncertainty bands. ECE is secondary and bin-sensitive: freeze bins, also show equal-mass sensitivity. A near-uniform predictor can be calibrated but useless; report discrimination/resolution and task accuracy alongside it.

For ordinal Score, use ranked probability score and ordinal MAE where labels have a justified ordering. The rubric index is not a physical measurement. Compare returned `score` with expectation under returned probabilities.

Selective prediction: choose thresholds on validation data, then report test error and coverage at those fixed thresholds. Show performance ranked by p-max, entropy, and vendor confidence. OOD detection: compare in-distribution to unanswerable, nonsense, contradiction, and distribution-shift sets. Assess confidence as a ranking statistic without presuming it is a correctness probability.

A confidently wrong valid answer disproves universal factual correctness, not schema safety. A single 0.9-confidence error does not disprove calibration. Many such errors can; a rare-event excuse is quantitatively testable with proper scores and sufficiently many independent labeled items. Even exact 1.0 may reflect output rounding, so preserve precision and qualify the inference.

## 6. Statistical plan

Preregister primary endpoints, exclusions, prompt versions, seed sets, contrasts, effect sizes, and stopping rules before the confirmatory run. Pilot data estimates variance and feasibility; it is not secretly pooled into an untouched test.

- Accuracy: Wilson 95% intervals as a simple diagnostic for independent items. Use subject/template/source-cluster bootstrap for clustered samples and aggregate comparisons.
- Paired model/condition differences: paired bootstrap over base items or exact McNemar for binary correctness, with cluster-aware methods when variants share a base item. Repeated prompts are not new independent facts.
- Historical aggregate comparisons: point differences and disclosed uncertainty only where n/estimator supports it. Do not manufacture intervals from a rounded percentage or use an unpaired test as paired evidence.
- Stratified pilots: retain selection probabilities and use weights for a population estimate; a 100-item-per-subject pilot is not the official full-set aggregate. Publish stratified results too.
- Calibration and proper-score differences: bootstrap at base-item/template level; keep prediction/correctness pairs together.
- Repeated timing: block by time window/session; mixed-effects or cluster-bootstrap estimates. Report effect sizes and confidence intervals, not only p-values. Do not treat burst calls as independent hardware samples.
- Multiple confirmatory claims: Holm correction within declared families. Exploratory fingerprint searches use held-out confirmation/FDR and remain exploratory unless preregistered.
- No significance-driven stopping. Set n after pilot using simulation or a power calculation for the specific paired contrast and minimum relevant effect.

Scale guide only: independent accuracy at n=400 has worst-case 95% half-width about 4.9 percentage points; n=2,400 about 2 points; n=10,000 about 1 point. These are estimation widths, not power for a model difference. Paired-test power depends on discordant answers. A small GPQA subset cannot settle a one-point ranking. Repeated endpoint queries mainly estimate serving variation, not new knowledge accuracy.

## 7. Performance and mechanism experiments

### Instrumentation

Capture monotonic timestamps for local queue entry, dispatch, headers/first byte when available, full body, and parsing. End-to-end is user-observed duration; dispatch-to-body includes network and queueing, not pure GPU compute. HTTP instrumentation must not invent unavailable DNS/TLS/server timings. Record cold/warm connection status, pooling, compression, response bytes, local concurrency, transport errors, rate-limit headers, retries, and environment.

Primary latency sweeps disable automatic retries. Each failed attempt remains in the dataset; timeouts are right-censored at a known threshold, not successful fast calls. Report failure rate and latency conditional on success together. An operational companion run can use bounded retries and report full user-visible latency and total cost. Never combine its timings with retry-free attempts.

### Factors and staged grids

Start concurrency 1. Pilot grids below are targets, not an instruction to exhaust a huge full factorial:

- State size L: roughly 128, 512, 2k, 8k, 24k tokens, subject to actual limits. Use natural text, random identifiers, and repetitive text as separate content strata.
- Q: 1, 2, 4, 8, 16, 32, 64; larger only if limits and token budgets permit.
- Choice K: 2, 4, 8, 16, 64, 128, 255, independently varying description length.
- Question length: short and long; distinguish total question text from maximum question length.
- Primitive: Noul, Choice, Score; Score K restricted to 2–10.
- Connection: warm persistent versus cold; exact repetition versus varied semantically matched input.
- Concurrency: 1, 2, 4, 8 only in a separately approved throughput sweep, rate-limited below account allowance.

Use one-factor sweeps plus a fractional-factorial selection for L×Q and Q×K interactions. Randomize condition order within short blocks, interleave a fixed sentinel, and repeat across at least three time windows. Initial feasibility pilot: 20–30 measurements per selected cell spread across blocks; determine final n by observed variance, target relative effect (e.g. 10%), and uncertainty goals. p99 needs far more samples; omit it from tiny cells.

Semantic nonces can themselves alter tokenization/answers. Separate exact-repeat, opaque-ID-renaming (claimed invisible to model), state-nonce, and genuinely distinct-input conditions; none is a guaranteed cache bypass. Run accuracy sentinels alongside latency so faster but degraded responses are not celebrated as improvements.

### High-value contrasts

| Experiment | Observable support | Non-identifiable alternatives/caveats |
|---|---|---|
| Same state: Q separate calls vs one Q-question call | User-visible batching benefit; usage/accounting amortization | Connection overhead, cache, scheduler, shared encoder/prefill all possible |
| Fixed Q/K, sweep L | Cost/latency/accuracy dependence on input | Cannot deduce attention architecture or parameter count from exponent |
| Sweep Q and K independently | Question/candidate marginal costs, possible saturation | Work parallelism, tensor shapes, scheduler and bandwidth confounded |
| Fixed total candidate text, vary grouping into questions | Candidate-versus-question overhead | Grouping changes semantics; use deterministic lookup tasks |
| Hold longest question fixed vs extend it | Bottleneck/max-length-like behavior | Per-question caching, padding, batching, preprocessing alternatives |
| Duplicate/add unrelated/contradictory sibling questions | Test claimed question isolation | Use many anchors and repeated baseline to estimate numeric jitter |
| Rename/reorder question IDs | Test claimed inference invisibility | Serialization/cache effects distinct from semantic visibility |
| Permute options and rename labels | Position/label bias and equivariance | Option labels are documented model input, unlike question IDs |
| Remove/add irrelevant candidate; compare odds of unchanged options | Candidate-local scoring/IIA diagnostic | A fixed-score softmax preserves odds, but other architectures can too |
| Repeat across times/connection states/concurrency | Jitter, caching and queueing signatures | No proof of internal self-batching or GPU configuration |

For candidate odds, use log(p_i/p_j) only away from zero and retain boundary counts. Include semantic distractors and truly irrelevant distractors separately. Independence-of-irrelevant-alternatives behavior is a behavioral property, not a unique architecture fingerprint.

Question isolation: keep an anchor identical, add varied siblings, compare its distribution using max absolute deviation and total variation distance, and correctness. Preregister an equivalence margin above serialization/rounding jitter. Failing to detect a difference is not proof of independence; sufficiently tight equivalence intervals are stronger. Deliberate cross-question reference tasks can test whether answers/instructions are accessible, but not expected to work under the documented isolation contract.

Fit simple competing predictive latency models: constant + linear L/Q/C; L×Q interactions; longest-question effects; piecewise saturation. Use held-out blocks and residual checks. Present coefficients and uncertainty, not a claimed neural-network schematic. Do not infer FLOPs, parameter count, or training cost from milliseconds or API price.

## 8. Behavioral provenance and other probes

### Self-description (low evidentiary weight)

Use balanced choices among TypeSafe, several candidate companies/families, “other,” and “unknown.” Ask direct and indirect variants, neutral third-person descriptions, negations, and randomized order. Include contradictory user identity suggestions and fictional-company controls. Compare question-alone with leading contexts. Consistent answers can still arise from branding post-training or learned generic assistant text.

Do not interpret Noul probabilities as Bayesian posterior odds over ancestry. These are outputs of the object being studied, not an independent statistical model of its provenance.

### Knowledge horizon

Construct independently verified event cohorts across months/years, topics, and popularity; include plausible time-matched distractors, stable facts, nonexistent-event controls, and a not-known option in the exploratory variant. Keep closed-book and source-provided conditions separate. Repeat with paraphrases/languages. Estimate a knowledge-decay profile controlling for difficulty/salience; do not announce an exact pretraining cutoff. Later post-training, retrieval, mixtures, or selective exposure can explain later knowledge.

### Language and political-policy fingerprints

Use matched, human-reviewed translations, native-language controls, historical facts and benign political questions spanning China, the US, Europe, and other regions. Measure answer accuracy, sensitivity to framing, refusal/unknown choice, and shifts under neutral paraphrase. A missing refusal option can force a selection; weak probabilities alone are not a refusal. Avoid conflating censorship, uncertainty, political disagreement, mistranslation, and factual mistakes.

These tests are permitted-content behavioral evaluations, not requests for harmful instructions. Similar policy patterns occur across unrelated models and can be altered by post-training; censorship does not identify a Chinese foundation model.

### Candidate-family fingerprinting

If candidate models are affordable and accessible, compare held-out item-level residual errors, option biases, linguistic asymmetries, unusual factual confusions, and robustness profiles across several unrelated families and sizes. Control for item difficulty, domain and overall accuracy; raw agreement mostly measures shared competence. Fit any candidate classifier on development data, test it on untouched items, and include known unrelated negative controls and “none of these.” Report behavioral similarity with uncertainty, not probability of stolen weights or provenance proof. Historical item-level responses may reduce calls, but they cannot substitute for missing paired probes.

### Extra high-information tests

- Probability coherence: P(A)+P(not A), implication inequalities, mutually exclusive facts, and cross-primitive consistency. Joint inconsistencies reveal limits of independent marginals, not necessarily uncalibrated individual predictions.
- Choice-set sensitivity: duplicate meaning under two labels, add dominated distractors, and vary class descriptions. Document legitimate changes to answer semantics.
- Grounding and injection: source-evidence conflict, fabricated citations, misleading instructions inside classified text, and irrelevant persuasion. Separate documented vulnerability from novel findings.
- Token accounting fingerprints: differential usage for fixed minimal pairs, languages, emoji, whitespace and rare strings. Compare candidate tokenizer counts only as tentative evidence; hidden templates/preprocessing/accounting can dominate, and tokenizer resemblance does not imply shared weights.
- Drift sentinels: replay fixed public tasks periodically and detect changes in model ID, distributions, latency and token counts. Drift is not automatically a model-weight update.
- Character-generation ablations: spelling/copying known strings, short factual completion, and free dialogue; separate ability to copy from useful original communication.

## 9. Talk to Jev: explicit external autoregression

Local-only web app; the server owns credentials and calls the API. This is not native chat or discovery of hidden thoughts. Each step is a newly prompted classification conditioned on the transcript and prefix we supply. Output is path-, prompt-, and decoding-dependent, and self-descriptions remain untrusted.

MVP protocol:

1. State contains user prompt, previous turns, and `assistant_prefix` as distinct fields.
2. One Choice asks: “Select the next character of the assistant response, conditioned on this transcript and exact prefix; choose END when complete.” Treat this as an experimental prompt to validate, not a guaranteed supported mode.
3. Fixed alphabet: 95 printable ASCII characters plus newline, tab, and END (98 options). Use visible keys such as `char_032`, semantic descriptions and an explicit local mapping; do not concatenate option IDs into output. Keep all choices under 255.
4. Default decoder: greedy argmax. Optional seeded sampling from returned probabilities; any temperature transform is local, labeled, and recorded. Tie-break rules fixed. One sequential request per emitted character; multiple unconditioned next-position questions are not equivalent.
5. Server emits per-step events to browser via SSE. Store full distributions, prefix hash, chosen character, raw versus transformed probabilities, model ID, latency, usage and cost.
6. Stop on END, user cancellation, failure, budget cap, maximum characters, wall time, or repetition guard. MVP defaults: 256 generated characters, concurrency 1, 120 seconds; configurable lower limits. Expose stop reason and in-flight-call behavior.

UI: transcript, live prefix, top-next-character probabilities, step/backtrack/fork controls, cumulative cost/time, characters/sec, and downloadable private trace. Escape output as text, never execute generated HTML or commands. Backtracking creates a new immutable branch rather than rewriting the old trace. Mark user-edited prefixes.

Context policy: stop before input limit by default. Optional oldest-turn truncation must be explicit, recorded, and visible; no hidden summarizing model. Arbitrary Unicode is not supported by the initial alphabet. An extended language-specific alphabet or hierarchical choice encoding is a later experiment with distinct latency/behavior; 256 bytes plus END would exceed the option cap and cannot be passed directly.

Total latency is roughly sum of step latencies. Re-sending the growing prefix gives quadratic cumulative prefix volume in characters (token cost depends on tokenization). Show actual usage, not a promise that free output means free chat. Repeated character errors and poor generation are scientifically useful results, not merely UI bugs.

## 10. Charts and publication artifacts

1. **Capability heatmap:** benchmarks on rows, models on columns. Main labels raw native metric and n; separate/hatch contextual comparators and missing values. Filter by protocol/reasoning mode. For accuracy only, optional chance-adjusted `(accuracy - chance)/(1 - chance)` with per-item choice-count chance; allow negative values. Not meaningful for every metric or as a universal intelligence scale. Avoid model-set-dependent min–max normalization.
2. **Accuracy/cost/latency frontiers:** only identical workload/quality definitions and measurement regions; separate direct answers from full distributions. Show uncertainty/error rate and avoid drawing a claimed universal frontier across unrelated historical workloads.
3. **Scaling:** median/p90 latency versus L, Q, K; faceted by primitive and content, with uncertainty, error rates and throughput. p99 only with enough samples. Display requested and observed/billed size separately.
4. **Calibration:** reliability curves with counts, Brier/log loss, risk–coverage, confident-error counts, and OOD breakdowns.
5. **Robustness/coherence:** paired deltas for option permutations, siblings, distractors, logical complements and injection conditions.
6. **Exploratory fingerprints:** temporal-knowledge curves, matched-language profiles and residual-similarity matrix, prominently labeled non-identifying.
7. **Talk:** traces, stop reasons, characters/sec, and branching distribution view.

Machine-readable metrics accompany every chart: dataset/prompt/model hashes, sample counts, errors, uncertainty method, reference provenance and analysis version. No invented confidence bars, benchmark averages across incomparable scales, or absent results represented as zero. Restricted datasets must not leak via tooltips, logs, exports or example screenshots.

## 11. Software design and reproducibility

Suggested stack: Python 3.12+, httpx, Pydantic, Typer CLI, NumPy/SciPy/pandas or Polars, Parquet + JSONL, Plotly reports, FastAPI with a small static TypeScript UI/SSE. Prefer a thin raw-HTTP adapter for precise attempt logging; use official SDK fixtures/documentation to validate behavior, not unexamined hidden retries. Lock dependencies. Do not need a task queue, database server, or multi-tenant deployment for MVP.

Components:

- `providers/`: common outcome type, Jev transport, deterministic mock, optional baseline adapters.
- `datasets/`: read-only versioned loaders and solver-backed generators; scoring labels kept outside request construction.
- `experiments/`: immutable manifests, randomized scheduler, token/request/rate budgets, cancellation and resumable runs.
- `analysis/`: metrics, cluster-aware statistics, plots and external-score provenance validation.
- `web/`: loopback-bound UI, server-side provider access, Talk session controller.
- `tests/`: contract fixtures, transport failures, simulator tests, scorer parity, seed determinism, budget/rate-limit tests and leakage checks.

Each run manifest includes run ID; code/dependency revision or dirty-tree hash; dataset and item hashes; generator seeds; prompt and option maps; requested model; region; test family; preregistration reference; budgets; retries; and inclusion rules. Record per-attempt ID plus logical request ID, raw request/response paths, HTTP status, timestamps, usage and exception. Dataset labels must never appear in outbound state or question payload accidentally.

Append-only attempt ledger; checkpoint completed logical items atomically. A timeout can have been processed and billed remotely: do not promise exactly-once execution. On resume, uncertain attempts require an explicit policy, never invisible replacement. Keep original predictions so analyses rerun without new API calls. Analysis itself has no provider access.

Secret hygiene: `TYPESAFE_API_KEY` only from process environment/local secret storage, no browser copy, no key in URLs, examples, exports, notebooks or logs. Redact Authorization and error dumps; tests use obvious fake keys. Ignore local secrets/data/run artifacts in version control. Bind UI to loopback, validate Origin/Host, and add local session authorization so arbitrary websites cannot spend the key through localhost. Public deployment requires a separate security/terms review.

## 12. Scope, budget and decision gates

Suggested **proposals, not authorizations**:

- Offline milestone: no calls or paid models.
- Contract/smoke milestone: up to 100 Jev requests after approval; discover actual schema/limits at low load.
- Pilot: <=5,000 Jev requests, <=10M estimated input tokens (about $0.42 at quoted price), <=$1 Jev cap and a fixed time limit. The first reached limit wins. Split among benchmark feasibility, calibration, scaling and Talk checks; not the final dataset size.
- Confirmatory run: explicit power-based manifest and cost estimate approved after pilot, rather than an arbitrary huge matrix.
- Paid baseline allowance: default $0; propose a separately approved $10–$25 bounded comparison if public item-level results cannot answer a key question.

Budget estimates include criteria, repeated state, retries, and growing Talk prefixes, not just question text. Reserve conservative per-in-flight-request costs and token headroom before dispatch; reconcile usage afterward. Because tokenization and billing are uncertain, a dollar setting is a best-effort application cap, not a provider-side financial guarantee. Separate spend on implementation assistants, Jev, baselines, hosting and data access.

Decision gates:

1. Applicable terms/permissions and data access understood.
2. Offline tests and protocol reviewed.
3. Smoke calls verify documented contract and cost estimates.
4. Pilot fixes generators, power targets and transport confounds; freeze confirmatory protocol.
5. Confirmatory run with no mid-run prompt tuning.
6. Review source provenance, statistics and claim wording before any permitted publication.

Most valuable early result: a calibrated capability map plus an isolation/scaling report. Architecture and ancestry conclusions remain conditional even if the observed signatures are striking.
