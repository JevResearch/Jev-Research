# Probing Jev's inner workings

Live probe battery run 2026-09-21 against `jev-1.13.0`. 987 calls, 0 errors,
about $0.038 (895k input + 200k output tokens). Charts:
`architecture-evidence.html`. Machine output: `runs_archprobe/`.
Every number here is **behavioral evidence consistent with an architecture, not
proof of one**, and none of it identifies the model's weights or provenance.
Claims are labeled **supported**, **suggested**, or **not established**.

## The question

Jev only emits choice/score/noul answers, never text. The working hypothesis is
that underneath it is a normal autoregressive LLM whose output head has been
swapped for a customizable logits/probability read-out, with all questions in a
request scored in one shared forward pass ("self-batched"). We tested that from
several independent angles.

## 1. Latency structure — **supported**

Server compute time is read from the queue-free `x-envoy-upstream-service-time`
response header (already logged on 19k earlier calls and re-measured here, so it
is not contaminated by network or our own concurrency).

| regime | finding |
|---|---|
| **fixed floor** | ~73 ms of per-request compute regardless of size (serving overhead, not the model); 500 tokens ~75 ms, 32k tokens ~252 ms |
| **prefill slope** | ~6.0 ms per 1k input tokens (linear; R²=0.86) |
| **quadratic term** | negligible at tested lengths — no attention blow-up signature |
| **headcount** | +0.44 ms marginal per extra question, but that is almost entirely the extra input *tokens* the question text adds (upstream stays 71→147 ms for 1→192 questions); read-out itself is near-free |
| **option count** | +0.10 ms per option — scoring 255 candidates is nearly free on the compute path |

**Interpretation.** One forward pass over the state, many cheap read-outs —
exactly what "self-batched" predicts. The cost that scales is (a) prompt length
(prefill) and (b) the size of the *output* we ask for (below), not the number of
decision heads.

## 2. The output is a probability vector, not text — **supported**

* output tokens grow **linearly with the number of options** (~9-10 tokens per
  option; and ~33 per question including its answer envelope), i.e. the response
  literally serializes a full distribution over candidates.
* **every probability is on a 0.01 grid** — 12,584/12,584 observed values, 86
  distinct levels (0.00-1.00). This is a fixed-precision softmax read-out, not a
  streaming token logit.

This is the clearest single piece of evidence for "generation head replaced by a
customizable probability head."

## 3. Continuous batching / shared compute — **supported**

We fired 1→32 identical independent requests concurrently and recorded both
client wall time and server compute time.

| in-flight c | median client wall | median server compute |
|---:|---:|---:|
| 1 | 288 ms | 79 ms |
| 4 | 288 ms | 86 ms |
| 8 | 280 ms | 80 ms |
| 16 | 282 ms | 74 ms |
| 32 | **622 ms** | **82 ms** |

Server compute stays essentially **flat** through c=32 while client wall bends up
only at 32. Two things follow: (i) independent requests share compute
(throughput rises ~14x from c=1 to c=16), and (ii) the c=32 wall bend is **our
own connection pool / TLS**, not a server limit — which is why the header-based
(upstream) signal is the trustworthy one. Consistent with an ordinary LLM
inference server doing continuous batching.

## 4. Connection vs compute — **supported**

Cold connection (fresh TCP+TLS) median 660 ms vs warm 307 ms → ~353 ms is pure
connection setup. Server compute (`x-envoy`) is unaffected by cold/warm, which
confirms it isolates compute and validates using it for §1/§3.

## 5. Where it points for ancestry - a dissociation, now with a broad net

### 5a. Identity prior - forced choice, rotation-balanced (supported)

15 neutral frames, each a single choice question over a 24-name parity list that
*always* includes Typesafe, Jev and Mellum, shown under all 24 cyclic rotations so
the serial-position bias provably cancels: the winning name sat at index 0 only
4.7% of the time (at/below the ~uniform 1/24 baseline), so the winner is
content-driven, not positional. Over 360 calls: OpenAI 293 votes (81%), ChatGPT
22, GPT 22, Qwen 15, Jev 6, Typesafe 0 (mean p 0.065 for Jev; Typesafe pinned at
the 0.010 quantization floor). Family mass: openai 0.48, qwen 0.11, typesafe 0.10,
anthropic 0.09. When Jev can only name a maker - its real maker on the menu, order
controlled - it says OpenAI. Unlike the earlier talk-to-Jev identity probes, this
forced-choice channel has no local model in the loop, so the OpenAI pull cannot be
a local-scorer artifact (that confound is gone).

### 5b. Tokenizer oracle, short battery - suggested, then revised below

Affine-fitting the server-reported input-token count of a fixed template against
reference tokenizers (HF_TOKEN so gated Llama/Gemma load): Qwen2.5 first (RMSE
1.05 tokens, slope 1.001, robust on ASCII-only subsets), then Mistral 1.36, Phi
1.44, Gemma 1.49, OpenAI o200k 1.84, Llama 1.89. That looked decisive - but the
strings are short and template-dominated, and a sharper test (5c) corrects it.

### 5c. Broad net + whitespace-free per-script rates (strong; revises 5b)

Broad net: 1,215 repos across ~30 orgs (Qwen, microsoft, google, meta-llama,
mistralai, deepseek, zai/THUDM, moonshotai, XiaomiMiMo, NVIDIA, tencent, 01-ai,
MiniMax, internlm, Cohere, SmolLM, upstage, allenai, BAAI, ...), clustered by
tokenizer.json bytes down to 173 distinct (vocab+specials) signatures; 128+ tokenizers
scored. No exact match exists among open models.

Tokens/char on whitespace-free samples (each system minus its own empty-state
baseline; ~1.0 = no merging, >1.0 = byte-level fallback, lower = merges):

| col | JEV | Qwen2.5 | Mistral | Llama-3.1 | Gemma-3 | Hy-MT2 | GLM-4.5 | DeepSeek | o200k | cl100k | gpt-oss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| latin_hello | 0.15 | 0.21 | 0.21 | 0.21 | 0.21 | 0.21 | 0.21 | 0.21 | 0.21 | 0.21 | 0.21 |
| latin_words | 0.26 | 0.11 | 0.16 | 0.11 | 0.11 | 0.11 | 0.11 | 0.11 | 0.11 | 0.11 | 0.11 |
| digits_single | 0.94 | 1.01 | 1.01 | 0.34 | 1.01 | 1.01 | 0.68 | 0.34 | 0.34 | 0.34 | 0.34 |
| cyrillic_common | 0.94 | 0.51 | 0.34 | 0.51 | 0.34 | 0.68 | 0.51 | 0.34 | 0.34 | 0.68 | 0.34 |
| cjk_common | 0.92 | 0.51 | 1.01 | 0.76 | 0.51 | 0.51 | 0.51 | 0.51 | 0.51 | 1.26 | 0.51 |
| cjk_random | 1.96 | 2.01 | 3.01 | 2.01 | 3.01 | 2.01 | 2.01 | 2.01 | 2.01 | 3.01 | 2.01 |
| hangul | 0.92 | 1.01 | 1.01 | 0.68 | 0.68 | 1.34 | 1.01 | 1.01 | 0.68 | 1.34 | 0.68 |
| greek | 0.94 | 1.01 | 1.01 | 0.81 | 0.81 | 1.01 | 0.81 | 0.81 | 0.81 | 1.01 | 0.81 |
| emoji_run | 1.71 | 1.04 | 1.04 | 2.04 | 1.04 | 2.04 | 2.04 | 2.04 | 1.04 | 2.04 | 1.04 |
| flags | 1.56 | 1.06 | 4.06 | 3.06 | 1.06 | 3.06 | 3.06 | 2.06 | 2.06 | 3.06 | 2.06 |
| repeated_pipe | 0.43 | 0.26 | 0.51 | 0.26 | 0.51 | 0.26 | 0.26 | 0.26 | 0.26 | 0.26 | 0.26 |

**Conclusion (revised).** The short-string "Qwen wins" was driven by the fixed
template constant, not the vocabulary. Under the whitespace-free per-script test,
Jev matches no open model. Its shape is a Latin-centric vocabulary with weak
non-Latin coverage (Cyrillic and CJK barely merged, about one token per code
point, astral pairs near two), plus an active whitespace-normalization pass in the
serving template. That is *consistent with* an English/Latin-dominant base (which
would agree with the OpenAI identity prior) rather than a Chinese-lab multilingual
base, but it is *not* any specific OpenAI tokenizer either: o200k, cl100k and
gpt-oss all fail the Cyrillic and flag columns. No weights, provenance, or
parameter count is established.

### 5d. What would actually settle it

A token-count oracle reads the serving vocabulary and its normalization, not the
weights. The decisive follow-up probes:

1. **Special-token probe.** Send the literal ChatML, Qwen-style and Phi-style
   chat markers (as plain text inside the state) and look for discontinuities in
   the reported count or in the answer distribution: a server that re-renders its
   own template may treat those byte sequences specially.
2. **Normalization probe.** Jev collapses whitespace runs to ~0 tokens. Send
   tab vs newline vs nbsp vs zero-width joiners to map the exact pre-tokenizer
   normalization; combine with per-script rates to search for a vocabulary that
   fits *after* that normalization is applied locally.
3. **Vocabulary-membership probe.** Single rare code points vs two-char
   sequences vs common words, per script: an exact vocabulary can be *pruned* by
   finding which specific character bigrams the server merges (rate exactly 0.5
   means the pair is one token). This narrows to a vocabulary, not a model.

Even all three together bound the base *tokenizer*; identifying the weights would
need behavior only a tokenizer cannot hide (e.g. cutoff-sensitive facts, or the
ordering/serial-position signature at matched option counts).

## 6. Model size from latency — **deliberately not estimated**

The observed marginal prefill rate (~200k tok/s) is far faster than a single
request would be on any plausible GPU, and §3 shows the server is batching our
tokens with other tenants'. That means per-request latency does **not** cleanly
map to active-parameter count; a naive back-out would badly *underestimate*. We
decline to state a parameter count from milliseconds (the same caution in
`docs/review-gate/batching-plan.md`). The honest size signal is the *benchmark
profile*: Jev's MMLU-Pro 82.8 / GPQA 76.5 sit in the capability band of small
(~4-9B-active) instruction models — but that band holds hundreds of models, so it
narrows *size*, not family. Note §5c: the per-script evidence argues against a
Chinese-lab multilingual vocabulary specifically, so a Qwen-class profile match
would be capability, not lineage.

## What this does and does not establish

* **Supported:** a fixed-floor + linear-prefill compute model; a
  probability-vector output head quantized to 0.01; self-batched multi-question
  read-outs from a shared forward pass; continuous batching on the server.
* **Suggested (strong):** an English/Latin-centric serving vocabulary with weak
  non-Latin merging, an OpenAI-shaped identity prior, and whitespace normalization
  in the serving template. NOT Qwen specifically — the whitespace-free per-script
  test (§5c) shows Qwen merges Cyrillic/CJK far more than Jev does.
* **Not established:** exact tokenizer match (none of ~173 open signatures fits
  Jev's per-script shape), weights, provenance, or parameter count. No
  architecture claim is proven from behavior alone.

## Reproduce

```
# inside the session holding TYPESAFE_API_KEY (sequential, randomized):
JEVO_ALLOW_LIVE=1 .venv/bin/python scripts/benchmark/run_arch_probe.py --out runs_archprobe
.venv/bin/python scripts/benchmark/tokenizer_fingerprint.py --rows runs_archprobe/rows.jsonl
.venv/bin/python scripts/benchmark/tokenizer_broadscan.py          # broad open-model net
.venv/bin/python scripts/benchmark/tokenizer_perscript.py          # per-script rates
.venv/bin/python scripts/report/arch_evidence.py
```
System code: `src/jev_observatory/arch_probe.py` + `arch_probe_mergerate.py` (pure
builders + analyzers), `run_arch_probe.py` (gated dispatcher),
`tokenizer_fingerprint.py` (short-string oracle), `tokenizer_perscript.py`
(template-free per-script rates), `tokenizer_broadscan.py` (open-model net),
15 offline arch-probe tests (463 total).
