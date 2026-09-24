# Evidence and source register

Retrieved 2026-09-17 UTC. These are public documentation observations, not independently verified measurements. Websites and aliases can change. Before executing experiments, archive permitted source text with fetch time, URL, and SHA-256; the design-phase notes below are not an immutable archive.

## Primary sources

| ID | Source | What it establishes / does not establish |
|---|---|---|
| S1 | https://typesafe.ai/blog/introducing-system-one-models-and-jev | Claims new architecture, parallel sampler, RLCD, $0.042/M input tokens, free output, and 70–500 ms service responses. Describes workflow reference as average probabilities of GPT-6 Astra and Fable 5.1, not independent labels. Says type-error rate is theoretical rather than empirical. West Coast client location is disclosed. Does not establish general frontier parity. |
| S2 | https://docs.typesafe.ai/introduction | Claims every question is evaluated in parallel and in isolation against shared state; adding questions barely changes response time. These are testable service/behavior claims, not architecture disclosure. |
| S3 | https://docs.typesafe.ai/api.md | POST `/v1/systemone`, Bearer authentication, request/response shapes, usage counts, and errors. Question IDs are said not to enter inference. |
| S4 | https://docs.typesafe.ai/primitives/choice | Up to 255 options; named option keys and descriptions are visible to the model; selected choice is maximum-probability option. |
| S5 | https://docs.typesafe.ai/primitives/score | 2–10 descriptive levels; score is expectation of level indices. Levels reportedly judged independently without seeing neighbors or their indices. Not an arbitrary real-valued regression head. |
| S6 | https://docs.typesafe.ai/primitives/noul | Probability of yes, with optional true/false criteria; no separate confidence property. |
| S7 | https://docs.typesafe.ai/confidence | Confidence is computed from probability-distribution shape. No statistical confidence interval. |
| S8 | https://docs.typesafe.ai/models.md | Lists `jev-1.13.0`, $0.042/M input tokens; aliases currently point to that version. Advertises 250,000 tokens/s and 1,200 requests/min but explicitly says limits change dynamically. Claims response reports versioned ID. Recheck account limits and runtime behavior. |
| S9 | https://docs.typesafe.ai/model-jaggedness/jev-1.13.md | Last reviewed 2026-09-16. Documents counting/math/date/indirection/adversarial-input errors; distractor-related context degradation; chained-choice text generation expected to be poor. Limits: 64k tokens total state + questions, 32k state + longest question. |
| S10 | https://docs.typesafe.ai/introduction/machine-learning-primer.md | Under “Three post-training approaches,” places RLCD alongside RLHF/RLVR. Diagram alt text: “Pretrained language models branch into muted RLHF and RLVR paths and an emphasized RLCD decision-model path.” Evidence for a language-model foundation narrative, not proof of checkpoint ancestry. |
| S11 | https://docs.typesafe.ai/concepts/system-one.md | Explicitly says calibration is measured across groups, not a guarantee of individual correctness. Text only; JSON objects/arrays supported as inputs. |
| S12 | https://evals.typesafe.ai/ | Public workflow-evaluation site linked by S1. Dynamic details were not comprehensively audited in this design phase. Do not describe it as an independent benchmark replication. |
| S13 | https://github.com/typesafe-ai/system-one-adapter-python | Vendor adapter supports both `discrete` and `probabilities`, structured outputs, normalization, and retries. Potential workflow replication tool, not a neutral default for every comparison. Its public confidence formulas are adapter behavior, not proof of Jev server internals. |
| S14 | https://typesafe.ai/legal/mca | §2.3(c) broadly restricts attempts to derive underlying algorithms/structure; §2.3(f) restricts publishing benchmarks/performance information; additional restrictions concern distillation, service resale, security testing, and limits. Check actual account agreement and permissions. |
| S15 | https://docs.typesafe.ai/legal.md | Links agreement, privacy policy, DPA; advertises a commitment against training on customer data and enterprise ZDR. Do not infer default zero retention. |

### Contractual dependency

S14 §2.3 says, in part, that the customer will not:

- “reverse engineer, decompile, disassemble, or attempt to access or derive the source code or underlying data ... including the underlying ideas, algorithms, structure, or organization”; or
- “publish benchmarks or performance information about the Services”.

We have not established which negotiated terms, exceptions, or legal rights apply to this account. Recommended next action: request written permission covering black-box behavioral/scaling research, aggregate benchmark publication, and the local communication demo. Obtain clarification about any later public hosting. Do not assume private architecture probing is exempt just because results are unpublished. No legal conclusion about enforceability is made here.

## Benchmark/reference candidates

| Source | Intended use and caveat |
|---|---|
| https://github.com/TIGER-AI-Lab/MMLU-Pro and https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro | Broad MCQ anchor, >12k questions, 14 subject areas, up to ten options. Official leaderboard: https://huggingface.co/spaces/TIGER-Lab/MMLU-Pro . CoT often improves scores. Audit harness commits: current README exposes retry-on-wrong options, which our confirmatory protocol must never use. |
| https://crfm.stanford.edu/helm/ and https://crfm.stanford.edu/2023/12/19/helm-lite.html | Reproducible scenario and model-result provenance; select only applicable tasks. HELM Lite includes both MCQ and free-generation tasks; do not claim its overall score is applicable. |
| https://huggingface.co/datasets/Idavidrein/gpqa and https://github.com/idavidrein/gpqa | Scientific MCQ stress test. Dataset access requires acceptance of conditions, including not posting examples online. Diamond is a specific subset; verify exact IDs and scoring. Small sample means wide uncertainty. |
| https://simple-bench.com/ | Commonsense/trick-question candidate; verify access to full set and evaluation protocol before scheduling. Public examples are not the entire benchmark. |
| https://livebench.ai/ and https://github.com/LiveBench/LiveBench/ | Freshness/versioning reference and possible compatible subtasks. Do not transform free-response tasks to MCQ and retain the original benchmark label. Release schedules in website and older descriptions differ; pin actual releases. |
| https://github.com/EleutherAI/lm-evaluation-harness | Candidate source for BoolQ, ARC-Challenge, HellaSwag, WinoGrande, PIQA, and other task definitions. Pin dataset and harness revisions; token-likelihood and length-normalization protocols are not equivalent to Jev choice selection. |
| https://github.com/sylinrl/TruthfulQA | Candidate misconception/epistemic test. Official MC metrics have particular scoring rules; evaluate from original scorer rather than treating every variant as top-1 accuracy. |
| https://github.com/nyu-mll/BBQ | Optional contextual/social-bias diagnostic; inspect license, task design, and ambiguous-context scoring before inclusion. Not an ancestry identifier. |

The last three repositories are candidate references, not fully audited implementations in this phase. Verify licenses, versions, split availability, benchmark integrity restrictions, and existing-result compatibility before importing data.

## Open questions to resolve by documentation/permission/small pilot

- Stable model pinning, actual model field, API response schema, effective rate limits, exact question cap, and token counting.
- Does a Choice criterion see other criteria? S5 explicitly describes isolation for Score levels; do not generalize it uncritically to Choice.
- Are probabilities quantized/rounded? How are exact 0/1 values produced? Does the confidence formula match the public adapter?
- Does billing count shared state once? Are repeated requests cached? Is response determinism promised? None assumed.
- Architecture, parameter count, training compute, foundation checkpoint, tokenizer lineage, retrieval use, and training cutoff remain unknown.

## Interpretation discipline

“Probability” is not “logit”: raw logits are not exposed. A JSON response reveals an interface, not a unique internal implementation. Do not turn company positioning, a model self-identification, output precision, or a timing curve into an unsupported factual claim about the foundation model. Evaluate claims rather than personalities.
