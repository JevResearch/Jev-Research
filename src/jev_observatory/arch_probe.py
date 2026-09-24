"""Architecture probes for Jev: what is it, and where did it come from?

Working hypothesis: Jev is structurally a pre-existing LLM, post-trained, with
the generation head replaced by a choice/score/noul read-out, and with outputs
"self-batched" (one forward pass over the state answers many questions). This
module designs the experiments that support or deny that, and analyzes recorded
results. Pure/offline: it only BUILDS request specs and SCORES recorded rows;
live dispatch lives in scripts/benchmark/run_arch_probe.py.

Families:
  prefill      1 question, padded state, sweep reported input tokens: fixed
               floor vs linear prefill cost vs super-linear attention.
  headcount    fixed state, sweep number of questions per request (self-batch
               predicts near-zero marginal cost per extra question).
  optioncount  1 question, sweep number of options (output-head cost).
  coldwarm     same request, cold vs warm connection (setup vs compute).
  ancestry     neutral frames, single choice question over a parity list of
               model/provider names INCLUDING Typesafe/Jev, cyclically
               order-rotated so the documented serial-position bias cancels.
  tokenizer    controlled strings in a fixed template; server-reported input
               token counts act as a tokenization oracle, fingerprinted
               offline against reference tokenizers.

Discipline (lessons from docs/LEAD-GATE.md and the order probe): sequential
dispatch (concurrency 1, no queue contamination), randomized order with a
recorded seed, repeats as the resampling unit, claim_type everywhere, slopes
are behavioral evidence consistent with an architecture and never proof, and
nulls are never upgraded to equivalence without a prespecified margin.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from statistics import fmean, median
from typing import Any

from .schema import ChoiceQuestion, SystemOneRequest

DEFAULT_MODEL_PIN = "jev-1.13.0"
UPSTREAM_HEADER = "x-envoy-upstream-service-time"

# whitespace-separated filler so token growth is smooth across the sweep
_FILLER = ("the quick brown fox jumps over a lazy dog while measuring "
           "careful quantities of scattered grain ")

# Controlled probe strings: repeated subwords, casing variants, CJK/Hangul/
# Cyrillic/Greek/Arabic, emoji incl. flag and ZWJ family, code fences,
# whitespace runs, chat role words, markdown lists, long numerals. Their
# server-reported token counts discriminate tokenizer families.
_FENCE = chr(96) * 3
# Build every non-ASCII probe string from code points so this file is pure
# ASCII (literal CJK/emoji in a source payload is fragile across tools).
_CJK4 = "".join(map(chr, (0x4f60, 0x597d, 0x4e16, 0x754c)))
_CJK3 = "".join(map(chr, (0x65e5, 0x672c, 0x8a9e)))
_HANG = "".join(map(chr, (0xd55c, 0xad6d, 0xc5b4)))
_CYR = "".join(map(chr, (0x0440, 0x0443, 0x0441, 0x0441, 0x043a, 0x0438, 0x0439)))
_GRK = "".join(map(chr, (0x03b5, 0x03bb, 0x03bb, 0x03b7, 0x03bd, 0x03b9, 0x03ba, 0x03ac)))
_ARA = "".join(map(chr, (0x0639, 0x0631, 0x0628, 0x064a)))
_SMILE = chr(0x1f600)
_FLAG = chr(0x1f1ef) + chr(0x1f1f5)
_FAM = chr(0x1f468) + chr(0x200d) + chr(0x1f469) + chr(0x200d) + chr(0x1f467) + chr(0x200d) + chr(0x1f466)
TOKENIZER_STRINGS: list[str] = [
    "", "a", "aa", "aaa", "aaaaaaaa", "hello", "hello hello hello",
    "world", "the", "of", " typesafe", "typesafe", "Typesafe", "Typesafe AI",
    "jev", "Jev", "JEV", "Jev observatory", "JetBrains",
    "OpenAI", "openai", "OpenAi", "Anthropic", "Claude", "Gemini",
    "1234567890", "0", "1", "10", "100", "1000000", "3.14159265358979",
    _CJK4, _CJK3, _HANG, _CYR, _GRK, _ARA,
    _SMILE, _SMILE + _SMILE + _SMILE, _FLAG, _FAM,
    _FENCE, "###", "**bold**", "\n", "\n\n\n", "\t\t", "  ", "    ",
    "assistant", "user", "system",
    "- point one\n- point two\n- point three",
    "one two three four five six seven eight nine ten eleven twelve",
    "supercalifragilisticexpialidocious", "antidisestablishmentarianism",
    "https://example.com/path?a=1&b=2", "function foo() { return 42; }",
    "\xef\xbb\xbf",
]

# concurrency  identical independent requests fired in groups of size c; how
#              per-call wall grows with in-flight batch size probes server-side
#              batching (sub-linear growth = genuinely parallel shared compute).
FAMILIES = ("prefill", "headcount", "optioncount", "coldwarm", "concurrency",
            "ancestry", "tokenizer", "mergerate")


@dataclass
class ProbeCall:
    family: str
    config_id: str
    request: SystemOneRequest
    meta: dict[str, Any] = field(default_factory=dict)

    def payload_metrics(self) -> dict[str, Any]:
        m = self.request.measures()
        return {
            "n_questions": m["q"],
            "n_options_total": m["c_total_candidates"],
            "state_chars": m["state_chars"],
            "payload_chars": len(json.dumps(self.request.to_payload(),
                                            ensure_ascii=False)),
        }


def _choice(options: list[str], instructions: str, state: str) -> SystemOneRequest:
    criteria = {f"o{i}": o for i, o in enumerate(options)}
    return SystemOneRequest(
        state=state, model=DEFAULT_MODEL_PIN,
        questions={"q0": ChoiceQuestion(instructions=instructions,
                                        criteria=criteria)})


# parity name battery: always includes Typesafe and Jev next to the majors
ANCESTRY_OPTIONS = [
    "OpenAI", "GPT", "ChatGPT", "Google", "Gemini", "Anthropic", "Claude",
    "Meta", "Llama", "Mistral", "Alibaba", "Qwen", "DeepSeek", "xAI", "Grok",
    "Zhipu", "GLM", "Microsoft", "Copilot", "Typesafe", "Jev", "Mellum",
    "Amazon", "Nvidia",
]
ANCESTRY_FRAMES = [
    "Which company built the model you are?",
    "You were created by which organization?",
    "Name the developer of this system.",
    "The assistant in this conversation was trained by:",
    "Which lab is responsible for you?",
    "The source code of this assistant belongs to:",
    "Complete the sentence: 'I am a language model made by ___'",
    "Your weights were produced at:",
    "The company behind you is:",
    "Who is the manufacturer of this chat system?",
    "What family of models do you come from?",
    "The product line you belong to is sold by:",
    "Your training run was run and funded by:",
    "Which brand do you represent?",
    "The organization that operates you is:",
]


def build_prefill(*, reps: int = 5) -> list[ProbeCall]:
    """Sweep state length with one fixed 10-option question."""
    calls: list[ProbeCall] = []
    for rep in range(reps):
        for target in (500, 1000, 2000, 4000, 8000, 16000, 32000):
            state = _FILLER * max(1, int(target / 18))
            calls.append(ProbeCall(
                "prefill", f"prefill:t{target}:r{rep}",
                _choice([f"alpha{i}" for i in range(10)],
                        "After reading the passage, pick option alpha4.", state),
                {"rep": rep, "nominal_tokens": target}))
    return calls


def build_headcount(*, reps: int = 5) -> list[ProbeCall]:
    """Fixed short state; sweep the number of questions packed into one request."""
    state = "A warehouse keeps red, blue and green crates on three shelves."
    calls: list[ProbeCall] = []
    for rep in range(reps):
        for nq in (1, 2, 4, 8, 16, 32, 64, 128, 192):
            qs = {f"q{i}": ChoiceQuestion(
                instructions=f"Is crate {i} red? Pick yes or no.",
                criteria={"o0": "yes", "o1": "no"}) for i in range(nq)}
            calls.append(ProbeCall(
                "headcount", f"headcount:n{nq}:r{rep}",
                SystemOneRequest(state=state, model=DEFAULT_MODEL_PIN,
                                 questions=qs),
                {"rep": rep, "n_questions": nq}))
    return calls


def build_optioncount(*, reps: int = 5, seed: int = 1213) -> list[ProbeCall]:
    """One question; sweep the number of options scored from the same state."""
    state = "Pick the capital of France from the list."
    calls: list[ProbeCall] = []
    for rep in range(reps):
        for k in (2, 5, 10, 32, 64, 128, 255):
            options = ["Paris"] + [f"nope{i}" for i in range(k - 1)]
            random.Random(seed + rep).shuffle(options)
            calls.append(ProbeCall(
                "optioncount", f"optioncount:k{k}:r{rep}",
                _choice(options, "Select the capital of France.", state),
                {"rep": rep, "n_options": k}))
    return calls


def build_coldwarm(*, reps: int = 8) -> list[ProbeCall]:
    state = "The library opens at nine and closes at five."
    calls: list[ProbeCall] = []
    for rep in range(reps):
        for conn in ("cold", "warm"):
            calls.append(ProbeCall(
                "coldwarm", f"coldwarm:{conn}:r{rep}",
                _choice(["yes", "no"], "Is the library open at noon?", state),
                {"rep": rep, "connection": conn}))
    return calls


def build_ancestry(*, rotations: int | None = None) -> list[ProbeCall]:
    """Frames x FULL cyclic offsets so every name visits every index once.

    With all n cyclic offsets per frame, each option occupies each serial
    position exactly once, so the documented serial-position bias (order probe
    section 11) cancels in the per-option mean probability. `rotations` (if
    given) subsamples offsets; default uses the full cycle.
    """
    calls: list[ProbeCall] = []
    n = len(ANCESTRY_OPTIONS)
    offsets = range(n) if rotations is None else range(rotations)
    for fi, frame in enumerate(ANCESTRY_FRAMES):
        for rot in offsets:
            opts = ANCESTRY_OPTIONS[rot:] + ANCESTRY_OPTIONS[:rot]
            calls.append(ProbeCall(
                "ancestry", f"ancestry:f{fi}:rot{rot}",
                _choice(opts, frame, "This is a conversation with an AI assistant."),
                {"frame": fi, "rotation": rot, "first_option": opts[0]}))
    return calls


def build_concurrency(*, reps: int = 4) -> list[ProbeCall]:
    """Groups of `c` INDEPENDENT near-identical requests; the runner dispatches
    each group concurrently. meta["group"] identifies the set to fire together."""
    calls: list[ProbeCall] = []
    for rep in range(reps):
        for c in (1, 2, 4, 8, 16, 32):
            for j in range(c):
                state = f"Batch item {j} of {c}, repetition {rep}: the sky is blue."
                calls.append(ProbeCall(
                    "concurrency", f"concurrency:c{c}:r{rep}:j{j}",
                    _choice(["yes", "no"], "Is the sky blue?", state),
                    {"rep": rep, "group": f"c{c}r{rep}", "c": c, "member": j}))
    return calls


def build_tokenizer(*, reps: int = 2) -> list[ProbeCall]:
    """Fixed template; the probe string varies. Server token count = oracle."""
    calls: list[ProbeCall] = []
    for si, s in enumerate(TOKENIZER_STRINGS):
        for rep in range(reps):
            state = "Token probe [" + s + "]"
            calls.append(ProbeCall(
                "tokenizer", f"tokenizer:s{si}:r{rep}",
                _choice(["yes", "no"], "Is the bracketed text empty?", state),
                {"string_index": si, "rep": rep, "probe": s}))
    return calls


# Re-exported so BUILDERS/analyze_rows below (and callers) see them; imported
# here (after _choice/ProbeCall are defined) to avoid a circular import.
from .arch_probe_mergerate import build_mergerate, analyze_mergerate  # noqa: E402


BUILDERS = {
    "prefill": build_prefill, "headcount": build_headcount,
    "optioncount": build_optioncount, "coldwarm": build_coldwarm,
    "concurrency": build_concurrency,
    "ancestry": build_ancestry, "tokenizer": build_tokenizer,
    "mergerate": build_mergerate,
}


def build_all(**kw: Any) -> list[ProbeCall]:
    calls: list[ProbeCall] = []
    for name in FAMILIES:
        calls.extend(BUILDERS[name](**{k: v for k, v in kw.items()
                                       if k in BUILDERS[name].__code__.co_varnames}))
    return calls


# ---------------------------------------------------------------- recording
def record_from_response(call: ProbeCall, transport: Any,
                         *, cold: bool) -> dict[str, Any]:
    """POST this call through `transport`, return one flat evidence row.

    transport is the raw HttpxTransport (or a stub in tests): we time the whole
    exchange ourselves and also keep the queue-free server compute time from the
    response header when present.
    """
    import time
    payload = call.request.to_payload()
    started = time.perf_counter()
    first_byte_ms: float | None = None
    body = b""
    status = 0
    upstream_ms: float | None = None
    usage_in = usage_out = None
    error = None
    try:
        with transport.stream("POST", "/v1/systemone", json=payload) as resp:
            status = resp.status_code
            hdrs = {k.lower(): v for k, v in resp.headers.items()}
            u = hdrs.get(UPSTREAM_HEADER)
            if u is not None:
                try:
                    upstream_ms = float(u)
                except ValueError:
                    upstream_ms = None
            for chunk in resp.iter_bytes():
                if first_byte_ms is None:
                    first_byte_ms = (time.perf_counter() - started) * 1000.0
                body += chunk
    except Exception as exc:  # transport/timeout: record, never crash the probe
        error = f"{type(exc).__name__}:{exc}"
    wall_ms = (time.perf_counter() - started) * 1000.0
    try:
        parsed = json.loads(body.decode("utf-8")) if body else {}
    except Exception:
        parsed = {}
    usage = parsed.get("usage") or {}
    if isinstance(usage, dict):
        usage_in = _as_int(usage.get("input_tokens"))
        usage_out = _as_int(usage.get("output_tokens"))
    # flatten the first question's answer back onto option TEXTS (the criteria
    # map turns our o{i} keys into the real names) so ancestry can aggregate
    probs_by_name: dict[str, float] = {}
    choice_name: str | None = None
    answers = parsed.get("answers") or {}
    criteria = next(iter(call.request.questions.values())).criteria
    for _qid, ans in answers.items():
        if not isinstance(ans, dict):
            continue
        pp = ans.get("probabilities")
        if isinstance(pp, dict):
            for k, v in pp.items():
                name = criteria.get(k, k)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    probs_by_name[name] = float(v)
        ch = ans.get("choice")
        if isinstance(ch, str):
            choice_name = criteria.get(ch, ch)
        break
    row = {
        "family": call.family, "config_id": call.config_id,
        "cold": cold, "http_status": status, "error": error,
        "wall_ms": round(wall_ms, 3),
        "first_byte_ms": round(first_byte_ms, 3) if first_byte_ms is not None else None,
        "upstream_ms": upstream_ms,
        "usage_input_tokens": usage_in, "usage_output_tokens": usage_out,
        "response_bytes": len(body),
        "probabilities": probs_by_name, "choice": choice_name,
        **call.payload_metrics(),
    }
    row.update(call.meta)
    return row


def _as_int(v: Any) -> int | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return int(v)
    return None


# ----------------------------------------------------------------- analysis
def _by_config(rows: list[dict[str, Any]], metric: str,
               reducer=median) -> dict[str, float]:
    out: dict[str, list[float]] = {}
    for r in rows:
        v = r.get(metric)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out.setdefault(r["config_id"], []).append(float(v))
    return {cfg: reducer(vals) for cfg, vals in out.items() if vals}


def analyze_prefill(rows: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np
    ok = [r for r in rows if r["family"] == "prefill" and r.get("usage_input_tokens")
          and r.get("upstream_ms") is not None]
    cfg = _by_config(ok, "upstream_ms")
    tokcfg = _by_config(ok, "usage_input_tokens")
    if len(cfg) < 3:
        return {"family": "prefill", "claim_type": "exploratory",
                "status": "insufficient_data", "n_configs": len(cfg)}
    xs = np.array([tokcfg[c] for c in sorted(cfg)]) / 1000.0
    ys = np.array([cfg[c] for c in sorted(cfg)])
    design = np.vstack([np.ones_like(xs), xs]).T
    (b0, b1), *_ = np.linalg.lstsq(design, ys, rcond=None)
    resid = ys - design @ np.array([b0, b1])
    quad = np.vstack([np.ones_like(xs), xs, xs ** 2]).T
    bq, *_ = np.linalg.lstsq(quad, ys, rcond=None)
    r2_lin = 1 - (resid ** 2).sum() / ((ys - ys.mean()) ** 2).sum()
    return {
        "family": "prefill", "claim_type": "exploratory", "status": "ok",
        "n_configs": len(cfg),
        "fixed_floor_ms": round(float(b0), 2),
        "ms_per_1k_input_tokens": round(float(b1), 3),
        "r2_linear": round(float(r2_lin), 3),
        "quadratic_term": round(float(bq[2]), 4),
        "token_range": [int(xs.min() * 1000), int(xs.max() * 1000)],
        "interpretation": ("compute is dominated by a fixed floor plus a linear "
                           "per-token term; a large quadratic term would point to "
                           "attention over long prompts. Behavioral evidence only."),
    }


def _slope_at_x0(rows, metric, xmetric):
    import numpy as np
    ok = [r for r in rows if r.get(xmetric) and r.get(metric) is not None]
    cfg_m = _by_config(ok, metric)
    cfg_x = _by_config(ok, xmetric)
    cfgs = sorted(set(cfg_m) & set(cfg_x))
    if len(cfgs) < 3:
        return None, None, 0
    xs = np.array([cfg_x[c] for c in cfgs], dtype=float)
    ys = np.array([cfg_m[c] for c in cfgs], dtype=float)
    x0 = xs.min()
    near = xs <= x0 * 1.5
    if near.sum() >= 2 and np.ptp(xs[near]) > 0:
        slope = float(np.polyfit(xs[near], ys[near], 1)[0])
    else:
        slope = float(np.polyfit(xs, ys, 1)[0])
    return slope, x0, len(cfgs)


def analyze_headcount(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fam = [r for r in rows if r["family"] == "headcount"]
    slope, x0, n = _slope_at_x0(fam, "upstream_ms", "n_questions")
    outslope, _, _ = _slope_at_x0(fam, "usage_output_tokens", "n_questions")
    return {
        "family": "headcount", "claim_type": "exploratory", "n_configs": n,
        "marginal_ms_per_question": round(slope, 4) if slope is not None else None,
        "reference_nq": x0,
        "marginal_output_tokens_per_question": round(outslope, 3) if outslope is not None else None,
        "self_batch_note": ("~0 ms per extra question supports a single forward "
                            "pass with many read-outs; a large positive slope "
                            "would contradict it. Output tokens per question "
                            "reflect the per-question distribution being emitted."),
    }


def analyze_optioncount(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fam = [r for r in rows if r["family"] == "optioncount"]
    slope, x0, n = _slope_at_x0(fam, "upstream_ms", "n_options_total")
    outslope, _, _ = _slope_at_x0(fam, "usage_output_tokens", "n_options_total")
    return {
        "family": "optioncount", "claim_type": "exploratory", "n_configs": n,
        "marginal_ms_per_option": round(slope, 4) if slope is not None else None,
        "marginal_output_tokens_per_option": round(outslope, 3) if outslope is not None else None,
        "note": ("cost that grows with the option count is the output head "
                 "scoring/emitting more candidates from the same state."),
    }


def analyze_coldwarm(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fam = [r for r in rows if r["family"] == "coldwarm"]
    cold = [r["wall_ms"] for r in fam if r["cold"] and r.get("wall_ms")]
    warm = [r["wall_ms"] for r in fam if not r["cold"] and r.get("wall_ms")]
    return {
        "family": "coldwarm", "claim_type": "exploratory",
        "n_cold": len(cold), "n_warm": len(warm),
        "median_cold_ms": round(median(cold), 1) if cold else None,
        "median_warm_ms": round(median(warm), 1) if warm else None,
        "connection_overhead_ms": (round(median(cold) - median(warm), 1)
                                   if cold and warm else None),
        "note": "wall-clock difference is TCP+TLS setup; upstream_ms is unaffected "
                "by connection state and isolates compute.",
    }


FAMILY_GROUPS = {  # map each parity option to a base-family bucket
    "OpenAI": "openai", "GPT": "openai", "ChatGPT": "openai",
    "Google": "google", "Gemini": "google", "DeepMind": "google",
    "Anthropic": "anthropic", "Claude": "anthropic",
    "Meta": "meta", "Llama": "meta",
    "Mistral": "mistral", "Alibaba": "qwen", "Qwen": "qwen",
    "DeepSeek": "deepseek", "xAI": "xai", "Grok": "xai",
    "Zhipu": "glm", "GLM": "glm", "Microsoft": "microsoft", "Copilot": "microsoft",
    "Typesafe": "typesafe", "Jev": "typesafe", "Mellum": "typesafe",
    "Amazon": "amazon", "Nvidia": "nvidia",
}


def analyze_ancestry(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-option mean probability + greedy votes, position-balanced.

    Because each frame is shown under `rotations` cyclic orders, every option
    sits in each index the same number of times, so the serial-position bias
    (order probe §11) is balanced out in the aggregate.
    """
    fam = [r for r in rows if r["family"] == "ancestry"]
    by_opt: dict[str, list[float]] = {}
    votes: dict[str, int] = {}
    probs_seen = 0
    for r in fam:
        probs = r.get("probabilities")
        choice = r.get("choice")
        if isinstance(probs, dict) and probs:
            probs_seen += 1
            for opt, p in probs.items():
                if isinstance(p, (int, float)) and not isinstance(p, bool):
                    by_opt.setdefault(opt, []).append(float(p))
        if isinstance(choice, str):
            votes[choice] = votes.get(choice, 0) + 1
    mean_p = {o: fmean(v) for o, v in by_opt.items()}
    ranked = sorted(mean_p.items(), key=lambda kv: -kv[1])
    # position-balance diagnostic: where does the chosen option sit in the
    # presented order? a ~uniform spread (not piled at index 0) shows the
    # cyclic rotations cancelled serial-position bias, so the winner is content.
    choice_rank: dict[int, int] = {}
    for r in fam:
        order = list((r.get("probabilities") or {}).keys())
        ch = r.get("choice")
        if ch in order:
            idx = order.index(ch)
            choice_rank[idx] = choice_rank.get(idx, 0) + 1
    n_rank = sum(choice_rank.values())
    at_index0 = choice_rank.get(0, 0) / n_rank if n_rank else None
    grouped: dict[str, float] = {}
    for opt, p in mean_p.items():
        g = FAMILY_GROUPS.get(opt, "other")
        grouped[g] = grouped.get(g, 0.0) + p
    return {
        "family": "ancestry", "claim_type": "exploratory",
        "n_frames_with_probs": probs_seen,
        "top_options_by_mean_p": [[o, round(p, 4)] for o, p in ranked[:8]],
        "greedy_votes": dict(sorted(votes.items(), key=lambda kv: -kv[1])[:8]),
        "choice_index_share": choice_rank,
        "choice_at_index_0_frac": round(at_index0, 3) if at_index0 is not None else None,
        "position_cancellation_ok": (at_index0 is not None and at_index0 < 3.0 / max(1, len(order))),
        "mean_p_jev": round(fmean([p.get("Jev") for p in
                                   [r.get("probabilities") for r in fam]
                                   if isinstance(p, dict) and "Jev" in p] or [0.0]), 4),
        "mean_p_typesafe": round(fmean([p.get("Typesafe") for p in
                                        [r.get("probabilities") for r in fam]
                                        if isinstance(p, dict) and "Typesafe" in p] or [0.0]), 4),
        "family_mass": {g: round(m, 4) for g, m in
                        sorted(grouped.items(), key=lambda kv: -kv[1])},
        "typesafe_offer": FAMILY_GROUPS.get("Typesafe", "typesafe") in grouped,
        "note": ("position-balanced preference among a parity set that INCLUDES "
                 "Typesafe/Jev. A consistent pull to one family despite parity "
                 "is suggestive of a prior; it does not identify an actual base."),
    }


def analyze_concurrency(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-call wall as a function of in-flight group size (from config_id cN)."""
    fam = [r for r in rows if r["family"] == "concurrency" and not r.get("error")
           and r.get("http_status") == 200]
    by_c: dict[int, list[float]] = {}
    up_c: dict[int, list[float]] = {}
    groups: dict[str, list[float]] = {}
    for r in fam:
        m = re.match(r"concurrency:c(\d+):", str(r["config_id"]))
        if not m:
            continue
        c = int(m.group(1))
        by_c.setdefault(c, []).append(float(r.get("wall_ms") or 0.0))
        if r.get("upstream_ms") is not None:
            up_c.setdefault(c, []).append(float(r["upstream_ms"]))
        groups.setdefault(r.get("group", str(r["config_id"])), []).append(
            float(r.get("wall_ms") or 0.0))
    curve = {}
    for c, v in sorted(by_c.items()):
        ups = up_c.get(c, [])
        curve[str(c)] = {"median_wall_ms": round(median(v), 1),
                         "median_upstream_ms": round(median(ups), 1) if ups else None,
                         "n": len(v)}
    tput = {}
    for g, v in groups.items():
        m = re.match(r"c(\d+)", g)
        if m and v:
            tput[m.group(1)] = round(max(v) / int(m.group(1)), 2)
    return {
        "family": "concurrency", "claim_type": "exploratory",
        "per_call_wall": curve,
        "batch_completion_ms_per_call": tput,
        "note": ("compare median_wall_ms (client-observed) vs median_upstream_ms "
                 "(server compute). Where wall rises but upstream stays flat the "
                 "slowdown is our own connection pool / TLS, not a server batch "
                 "limit. Flat upstream across in-flight counts means independent "
                 "requests share compute (continuous batching)."),
    }


def analyze_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "claim_level": "exploratory_architecture_evidence",
        "counts": {"n_rows": len(rows),
                   "n_errors": sum(1 for r in rows if r.get("error"))},
        "prefill": analyze_prefill(rows),
        "headcount": analyze_headcount(rows),
        "optioncount": analyze_optioncount(rows),
        "coldwarm": analyze_coldwarm(rows),
        "concurrency": analyze_concurrency(rows),
        "ancestry": analyze_ancestry(rows),
        "mergerate": analyze_mergerate(rows),
    }
