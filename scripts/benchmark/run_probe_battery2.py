#!/usr/bin/env python3
"""Probe battery 2 (P1-P5): the discriminating follow-up probes specified in
ARCHITECTURE-ANALYSIS.md section 10.

Gate-guarded like every live runner in this repo: without BOTH
JEVO_ALLOW_LIVE=1 and TYPESAFE_API_KEY in the process environment this script
prints the plan and exits (status 1). The key is only ever read from the
environment and never written to any artifact.

  # plan only (offline, always safe):
  python scripts/benchmark/run_probe_battery2.py --plan

  # dry-run: exercise every builder AND analyzer on synthetic responses with
  # planted ground truth (repo convention: validate on the simulated endpoint
  # before live dispatch); writes <out>/probe2_dryrun.json:
  python scripts/benchmark/run_probe_battery2.py --dry-run

  # live (paid; parent authorization required, DESIGN.md section 12):
  JEVO_ALLOW_LIVE=1 TYPESAFE_API_KEY=... \
    python scripts/benchmark/run_probe_battery2.py --out runs_archprobe

Families (hypothesis each separates, per ARCHITECTURE-ANALYSIS.md section 10):
  p1_fallback  fallback granularity of the tokenizer (byte-level BPE vs
               codepoint vs UTF-16 units) + the serving normalizer's exact
               character set; rare-codepoint classes and combining sequences.
  p1_merge     merge-boundary contrasts (ab|ab vs a..a b..b), case seams,
               punctuation bridges, repeat-saturation curves: fingerprints
               merge ORDER, beyond counts.
  p2_grid      K x option-length grid: is decision cost fully accounted by
               the prefill of option tokens (single trunk) or does anything
               scale with K or K x length beyond tokens (per-option passes)?
  p3_kcal      calibration under option-count scaling on 200 gold-labeled
               synthetic items (fresh_capability_spec.json, project-authored,
               Unlicense): dilution law of p_gold(K), floor structure vs K,
               accuracy invariance, IIA odds stability.
  p4_lattice   identical-option vectors at K in {2,3,4,6,7,8,12,255}:
               deterministic rounding/apportionment rule (0.33x3 vs
               0.34/0.33/0.33), tie-break behavior, internal noise scale
               from repeat flip rates.
  p5_horizon   dated real-world events Oct-2024..Aug-2025 + pre-cutoff and
               fictional controls, forced and abstention frames: fixed
               parametric cutoff (cliff) vs rolling knowledge (no cliff).

Live outputs: <out>/probe2_rows.jsonl, <out>/probe2_analysis.json,
<out>/BILLING-probe2.json. Estimated live cost at the measured $0.042/M
input tokens: ~$0.11 for the full battery (see --plan token estimate).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from jev_observatory import arch_probe as ap
from jev_observatory.arch_probe import ProbeCall
from jev_observatory.redact import Redactor, base_url, get_api_key, live_calls_allowed
from jev_observatory.schema import ChoiceQuestion, SystemOneRequest, DEFAULT_MODEL_PIN
from jev_observatory.transport import HttpxTransport

ROOT = Path(__file__).resolve().parents[2]

# The mergerate frame from battery 1: a bracketed sample inside a fixed state,
# one inert yes/no question. Reported input_tokens minus the empty-sample
# baseline is the class's token cost (template cancels).
_EMPTY_STATE = "State: [ ] End of state."
_FRAME = "Is the text between the brackets empty? Answer yes or no."


def _u(*cps: int) -> str:
    return "".join(chr(c) for c in cps)


def _sample_call(family: str, cid: str, sample: str, meta: dict) -> ProbeCall:
    state = f"State: [{sample}] End of state."
    req = SystemOneRequest(
        state=state, model=DEFAULT_MODEL_PIN,
        questions={"q0": ChoiceQuestion(instructions=_FRAME,
                                        criteria={"o0": "yes", "o1": "no"})})
    m = {"sample_chars": len(sample), "sample_bytes": len(sample.encode("utf-8")),
         "reps": meta.get("rep", 0)}
    m.update(meta)
    return ProbeCall(family=family, config_id=cid, request=req, meta=m)


# --------------------------------------------------------------- P1 fallback
def _p1_fallback_classes() -> dict[str, str]:
    rnd = random.Random(11)
    return {
        # baselines (template-only; battery 1 measured ~316 tokens empty-state)
        "baseline_empty": "",
        "baseline_spaces": " " * 60,
        # covered-reference controls (known ~1.0 tok/char from battery 1)
        "ctrl_cyrillic": _u(0x43f, 0x440, 0x438, 0x432, 0x435, 0x442) * 10,
        "ctrl_cjk_common": _u(0x4f60, 0x597d, 0x4e16, 0x754c) * 15,
        # uncovered BMP blocks (3-byte UTF-8; 1 UTF-16 unit)
        "cjk_ext_a": "".join(chr(rnd.randint(0x3400, 0x4DB5)) for _ in range(60)),
        "cherokee": "".join(chr(rnd.randint(0x13A0, 0x13F5)) for _ in range(60)),
        "ethiopic": "".join(chr(rnd.randint(0x1200, 0x137C)) for _ in range(60)),
        "yi_syllables": "".join(chr(rnd.randint(0xA000, 0xA48C)) for _ in range(60)),
        "hangul_rare": "".join(chr(rnd.randint(0xC000, 0xD7A3)) for _ in range(60)),
        "cjk_compat_ideo": "".join(chr(rnd.randint(0xF900, 0xFAFF)) for _ in range(60)),
        # astral planes (4-byte UTF-8; 2 UTF-16 units)
        "cjk_ext_b": "".join(chr(rnd.randint(0x20000, 0x2A6DF)) for _ in range(40)),
        "math_bold": "".join(chr(rnd.randint(0x1D400, 0x1D7FF)) for _ in range(40)),
        "emoji_rare": "".join(chr(rnd.randint(0x1F900, 0x1FAFF)) for _ in range(40)),
        "regional_pairs": (_u(0x1F1E6, 0x1F1E8) + _u(0x1F1F2, 0x1F1FD)
                           + _u(0x1F1FC, 0x1F1F8) + _u(0x1F1EF, 0x1F1F5)) * 6,
        # normalization / combining behavior
        "combining_decomposed": ("e" + _u(0x0301)) * 40,        # e + U+0301
        "combining_precomposed": _u(0x00E9) * 40,               # precomposed
        "devanagari_conjunct": _u(0x0915, 0x094D, 0x0937) * 20,  # k + virama + ssa
        # the serving normalizer's character set
        "nbsp_run": _u(0x00A0) * 60,
        "tab_run": "\t" * 60,
        "newline_run": "\n" * 60,
        "ideographic_space_run": _u(0x3000) * 60,
        "zwj_run": _u(0x200D) * 60,
        "zwnj_run": _u(0x200C) * 60,
        "bom_run": _u(0xFEFF) * 60,
        "soft_hyphen_run": _u(0x00AD) * 60,
        "mixed_space_kinds": (" \t\n" + _u(0x00A0)) * 20,
        "single_spaces_between_words": " ".join(["word"] * 40),
        "double_spaces_between_words": "  ".join(["word"] * 40),
    }


def build_p1_fallback(reps: int = 4) -> list[ProbeCall]:
    calls = []
    for name, sample in _p1_fallback_classes().items():
        for r in range(reps):
            calls.append(_sample_call("p1_fallback", f"p1f:{name}:r{r}", sample,
                                      {"cls": name, "rep": r}))
    # lone-surrogate JSON escape: does the parser/normalizer work in UTF-16?
    # (pydantic str cannot hold a lone surrogate through json.dumps->utf-8, so
    # this is dispatched as a raw-body call; see _fire_raw in main.)
    calls.append(ProbeCall(
        family="p1_fallback", config_id="p1f:lone_surrogate:r0",
        request=SystemOneRequest(
            state="State: [x] End of state.", model=DEFAULT_MODEL_PIN,
            questions={"q0": ChoiceQuestion(instructions=_FRAME,
                                            criteria={"o0": "yes", "o1": "no"})}),
        meta={"cls": "lone_surrogate",
              "raw_state": "State: [" + chr(0xD800) + "x] End of state.",
              "sample_chars": 2, "sample_bytes": 0, "reps": 0}))
    return calls


# --------------------------------------------------------------- P1 merge
def build_p1_merge(reps: int = 3) -> list[ProbeCall]:
    contrasts = {
        "ab_x60": "ab" * 60,
        "a60_b60": "a" * 60 + "b" * 60,
        "a_ba59_b": "a" + "ba" * 59 + "b",
        "th_x80": "th" * 80,
        "t80_h80": "t" * 80 + "h" * 80,
        "hello_x10": "hello" * 10,
        "hello_x20": "hello" * 20,
        "hello_x40": "hello" * 40,
        "hello_x80": "hello" * 80,
        "hello_x160": "hello" * 160,
        "digit7_x10": "7" * 10,
        "digit7_x40": "7" * 40,
        "digit7_x160": "7" * 160,
        "dash_x40": "-" * 40,
        "dash_x160": "-" * 160,
        "case_aa60": "aa" * 60,
        "case_aA60": "aA" * 60,
        "case_AA60": "AA" * 60,
        "bridge_word.word": "word.word" * 15,
        "bridge_wordword": "wordword" * 15,
        "bridge_word_word": "word word" * 15,
        "bridge_word-word": "word-word" * 15,
    }
    calls = []
    for name, sample in contrasts.items():
        for r in range(reps):
            calls.append(_sample_call("p1_merge", f"p1m:{name}:r{r}", sample,
                                      {"cls": name, "rep": r}))
    return calls


# --------------------------------------------------------------- P2 grid
P2_KS = (2, 8, 32, 128, 255)
P2_LENS = (2, 8, 32, 64)


def build_p2_grid(reps: int = 3) -> list[ProbeCall]:
    calls = []
    state = "State: The warehouse log is quiet. End of state."
    instr = "Which token is the reference? Pick one."
    for k in P2_KS:
        for ln in P2_LENS:
            for r in range(reps):
                # digit-string options: ~1 token/char (battery 1), length-controlled;
                # a 3-digit index suffix keeps option texts unique
                opts = [("7" * max(ln - 3, 1)) + f"{i:03d}" for i in range(k)]
                req = SystemOneRequest(
                    state=state, model=DEFAULT_MODEL_PIN,
                    questions={"q0": ChoiceQuestion(
                        instructions=instr,
                        criteria={f"o{i}": o for i, o in enumerate(opts)})})
                calls.append(ProbeCall(
                    family="p2_grid", config_id=f"p2:K{k}:L{ln}:r{r}",
                    request=req,
                    meta={"K": k, "L": ln, "rep": r,
                          "option_chars": len(opts[0]), "sample_chars": 0,
                          "sample_bytes": 0}))
    return calls


# --------------------------------------------------------------- P3 K-calibration
P3_KS = (2, 4, 8, 16, 32, 64, 128, 255)


def _load_p3_items(n_items: int = 200, seed: int = 20260925) -> list[dict]:
    spec = json.loads((ROOT / "fresh_capability_spec.json").read_text(encoding="utf-8"))
    items = [it for it in spec["items"]
             if any(q.get("type") == "choice" for q in it["questions"].values())]
    by_group: dict[str, list] = {}
    for it in items:
        by_group.setdefault(it["group"], []).append(it)
    rng = random.Random(seed)
    picked = []
    per = max(1, n_items // len(by_group))
    for g in sorted(by_group):
        pool = by_group[g][:]
        rng.shuffle(pool)
        picked.extend(pool[:per])
    rng.shuffle(picked)
    return picked[:n_items]


def build_p3_kcal(reps: int = 2, n_items: int = 200) -> list[ProbeCall]:
    calls = []
    for it in _load_p3_items(n_items):
        qid = next(q for q, v in it["questions"].items() if v.get("type") == "choice")
        q = it["questions"][qid]
        gold = it["gold"][qid]["value"]
        base_opts = list(q["criteria"])           # original option keys (names)
        if gold not in base_opts:
            continue
        for K in P3_KS:
            if K < len(base_opts):
                continue
            pads = [f"zzq-{i:04d}-9" for i in range(K - len(base_opts))]
            opts = base_opts + pads
            for r in range(reps):
                req = SystemOneRequest(
                    state=it["state"], model=DEFAULT_MODEL_PIN,
                    questions={"q0": ChoiceQuestion(
                        instructions=q["instructions"],
                        criteria={f"o{i}": o for i, o in enumerate(opts)})})
                calls.append(ProbeCall(
                    family="p3_kcal", config_id=f"p3:{it['id']}:{qid}:K{K}:r{r}",
                    request=req,
                    meta={"item": it["id"], "qid": qid, "K": K, "rep": r,
                          "group": it["group"], "gold": gold,
                          "n_base": len(base_opts), "sample_chars": 0,
                          "sample_bytes": 0}))
    return calls


# --------------------------------------------------------------- P4 lattice
P4_IDENT_KS = (2, 3, 4, 6, 7, 8, 12, 255)


def build_p4_lattice(reps: int = 8) -> list[ProbeCall]:
    calls = []
    state = "State: The reference token is zzq. End of state."
    instr = "Which token is the reference? Pick one."
    for k in P4_IDENT_KS:
        for r in range(reps):
            req = SystemOneRequest(
                state=state, model=DEFAULT_MODEL_PIN,
                questions={"q0": ChoiceQuestion(
                    instructions=instr,
                    criteria={f"o{i}": "zzq" for i in range(k)})})
            calls.append(ProbeCall(family="p4_lattice", config_id=f"p4:ident{k}:r{r}",
                                   request=req, meta={"variant": "identical", "K": k,
                                                      "rep": r, "sample_chars": 0,
                                                      "sample_bytes": 0}))
    # near-ties: case/whitespace variants of one word (surface-form scoring)
    for r in range(20):
        req = SystemOneRequest(
            state=state, model=DEFAULT_MODEL_PIN,
            questions={"q0": ChoiceQuestion(
                instructions=instr,
                criteria={"o0": "zzq", "o1": "ZZQ", "o2": " zzq", "o3": "zzq "})})
        calls.append(ProbeCall(family="p4_lattice", config_id=f"p4:neartie:r{r}",
                               request=req, meta={"variant": "near_tie", "K": 4,
                                                  "rep": r, "sample_chars": 0,
                                                  "sample_bytes": 0}))
    return calls


# --------------------------------------------------------------- P5 horizon
# Designer-verified real-world events (verified against public record at design
# time, 2026-09). month = "YYYY-MM" of the event. kind: post (candidate
# post-cutoff), pre (control, expected known), fictional (control, expected
# refused). Caveat: if the deployment world's timeline diverges from the
# designer's record for 2025 events, "post" items measure horizon anyway -
# a model that never saw the event behaves like a model that predates it.
P5_EVENTS: list[tuple[str, str, str, str]] = [
    # (month, event, kind, shifted-month distractor)
    ("2024-10", "the Nobel Prize in Physics being awarded to John Hopfield and Geoffrey Hinton", "post", "2024-09"),
    ("2024-10", "the Nobel Peace Prize being awarded to Nihon Hidankyo", "post", "2024-11"),
    ("2024-11", "the United States presidential election", "post", "2024-10"),
    ("2024-12", "the reopening of Notre-Dame Cathedral in Paris after the fire restoration", "post", "2025-01"),
    ("2024-12", "the collapse of the Assad government in Syria", "post", "2024-11"),
    ("2025-01", "the Palisades and Eaton wildfires in Los Angeles County", "post", "2024-12"),
    ("2025-01", "the inauguration of Donald Trump for a second term", "post", "2025-02"),
    ("2025-01", "the release of DeepSeek-R1 and the market reaction to it", "post", "2024-12"),
    ("2025-02", "Super Bowl LIX, won by the Philadelphia Eagles", "post", "2025-01"),
    ("2025-02", "the German federal election returning the CDU/CSU as the largest bloc", "post", "2025-03"),
    ("2025-04", "the United States 'Liberation Day' tariff announcements", "post", "2025-03"),
    ("2025-04", "the death of Pope Francis", "post", "2025-05"),
    ("2025-05", "the papal conclave electing Pope Leo XIV", "post", "2025-06"),
    ("2025-06", "Israel's air campaign against Iran beginning", "post", "2025-05"),
    ("2025-06", "United States strikes on Iranian nuclear sites and the ceasefire that followed", "post", "2025-07"),
    ("2025-07", "the Trump-Putin summit in Anchorage, Alaska", "post", "2025-08"),
    ("2025-08", "the release of OpenAI's GPT-5", "post", "2025-07"),
    # pre-cutoff controls (expected known from FINDINGS.md section 5 horizon)
    ("2022-11", "the initial public launch of ChatGPT", "pre", "2022-10"),
    ("2023-10", "the Hamas attack on southern Israel", "pre", "2023-09"),
    ("2024-04", "the total solar eclipse crossing North America", "pre", "2024-05"),
    ("2024-07", "the opening of the Paris Summer Olympics", "pre", "2024-06"),
    ("2024-08", "the Boeing Starliner crewed flight test returning its crew", "pre", "2024-09"),
    # fictional controls (expected refused / "did not occur")
    ("2025-03", "the Eiffel Tower being permanently relocated to Berlin", "fictional", "2025-04"),
    ("2025-02", "OpenAI announcing a model named GPT-9", "fictional", "2025-01"),
    ("2024-12", "the 2024 Summer Olympics being held in Tokyo", "fictional", "2025-01"),
    ("2025-05", "the United Nations dissolving and being replaced by the World Assembly", "fictional", "2025-06"),
]

_MONTHS = ["2024-09", "2024-10", "2024-11", "2024-12", "2025-01", "2025-02",
           "2025-03", "2025-04", "2025-05", "2025-06", "2025-07", "2025-08"]


def build_p5_horizon(reps: int = 2) -> list[ProbeCall]:
    calls = []
    state = "State: A timeline of public events is being checked. End of state."
    for mi, (month, event, kind, shifted) in enumerate(P5_EVENTS):
        months = sorted({month, shifted, _MONTHS[(mi * 5) % len(_MONTHS)]})
        for frame in ("forced", "abstain"):
            opts = ([f"in {m}" for m in months] + ["it did not occur"])
            if frame == "abstain":
                opts = opts + ["cannot determine from what I know"]
            gold = "it did not occur" if kind == "fictional" else f"in {month}"
            for r in range(reps):
                req = SystemOneRequest(
                    state=state, model=DEFAULT_MODEL_PIN,
                    questions={"q0": ChoiceQuestion(
                        instructions=f"When did {event} happen?",
                        criteria={f"o{i}": o for i, o in enumerate(opts)})})
                calls.append(ProbeCall(
                    family="p5_horizon", config_id=f"p5:{kind}:{mi}:{frame}:r{r}",
                    request=req,
                    meta={"kind": kind, "month": month, "frame": frame, "rep": r,
                          "gold": gold, "event_i": mi, "sample_chars": 0,
                          "sample_bytes": 0}))
    return calls


BUILDERS = {
    "p1_fallback": build_p1_fallback,
    "p1_merge": build_p1_merge,
    "p2_grid": build_p2_grid,
    "p3_kcal": build_p3_kcal,
    "p4_lattice": build_p4_lattice,
    "p5_horizon": build_p5_horizon,
}


# --------------------------------------------------------------- analysis
def _median(xs):
    s = sorted(xs)
    n = len(s)
    return (s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2) if n else None


def analyze(rows: list[dict]) -> dict:
    out: dict = {"claim_level": "exploratory_architecture_evidence",
                 "n_rows": len(rows),
                 "n_errors": sum(1 for r in rows if r.get("error"))}
    fam: dict[str, list] = {}
    for r in rows:
        fam.setdefault(r["family"], []).append(r)

    # ---- P1 fallback: median reported tokens per class, tokens per char/byte
    if "p1_fallback" in fam:
        base = _median([r["usage_input_tokens"] for r in fam["p1_fallback"]
                        if r.get("cls") == "baseline_empty" and not r.get("error")])
        if base is None:
            base = _median([r["usage_input_tokens"] for r in fam["p1_fallback"]
                            if r.get("cls") == "baseline_spaces" and not r.get("error")]) or 316
        cls: dict[str, dict] = {}
        for r in fam["p1_fallback"]:
            if r.get("error") or not r.get("usage_input_tokens"):
                c = cls.setdefault(r.get("cls", "?"), {"tok": [], "errors": 0})
                c["errors"] = c.get("errors", 0) + 1
                continue
            c = cls.setdefault(r.get("cls", "?"),
                               {"tok": [], "chars": r.get("sample_chars"),
                                "bytes": r.get("sample_bytes"), "errors": 0})
            c["tok"].append(r["usage_input_tokens"])
        per = {}
        for name, c in cls.items():
            if not c.get("tok"):
                per[name] = {"n": 0, "errors": c.get("errors", 0)}
                continue
            med = _median(c["tok"])
            over = med - base
            per[name] = {"n": len(c["tok"]), "median_reported_in": med,
                         "tokens_over_baseline": over,
                         "chars": c["chars"], "bytes": c["bytes"],
                         "tokens_per_char": round(over / c["chars"], 3) if c["chars"] else None,
                         "tokens_per_utf8_byte": round(over / c["bytes"], 3) if c["bytes"] else None}
        out["p1_fallback"] = {
            "baseline_tokens": base,
            "classes": per,
            "reading": ("tokens_per_char ~1 => codepoint-level entry; ~1.5-2 for 3-byte "
                        "chars and ~2-3 for 4-byte chars => byte-level BPE fallback with "
                        "partial byte-n-gram merges; <=2 uniformly for astral => UTF-16 "
                        "code-unit fallback; combining vs precomposed equality => NFC "
                        "normalization; whitespace-class rows map the serving normalizer."),
        }

    # ---- P1 merge: contrast deltas
    if "p1_merge" in fam:
        med: dict[str, float] = {}
        for r in fam["p1_merge"]:
            if r.get("error") or not r.get("usage_input_tokens"):
                continue
            med.setdefault(r["cls"], []).append(r["usage_input_tokens"])
        meds = {k: _median(v) for k, v in med.items()}
        out["p1_merge"] = {
            "median_tokens": meds,
            "contrasts": {
                "ab_vs_aabb": (meds.get("ab_x60", 0) - meds.get("a60_b60", 0))
                if "ab_x60" in meds and "a60_b60" in meds else None,
                "hello_saturation": [meds.get(f"hello_x{n}") for n in (10, 20, 40, 80, 160)],
                "case_seam_aa_minus_aA": (meds.get("case_aa60", 0) - meds.get("case_aA60", 0))
                if "case_aa60" in meds and "case_aA60" in meds else None,
                "bridge_dot_minus_none": (meds.get("bridge_word.word", 0) - meds.get("bridge_wordword", 0))
                if "bridge_word.word" in meds and "bridge_wordword" in meds else None,
            },
        }

    # ---- P2 grid: does anything scale with K beyond tokens?
    if "p2_grid" in fam:
        pts = [(r["usage_input_tokens"], r.get("K") or r.get("n_options_total"),
                r["upstream_ms"]) for r in fam["p2_grid"]
               if not r.get("error") and r.get("upstream_ms") is not None]
        if len(pts) > 10:
            # OLS upstream ~ 1 + tokens/1000 + K/100
            X = [[1.0, t / 1000.0, k / 100.0] for t, k, _ in pts]
            y = [m for _, _, m in pts]
            coef = _ols(X, y)
            resid_sd = _resid_sd(X, y, coef)
            out["p2_grid"] = {
                "n": len(pts),
                "ols_intercept_ms": round(coef[0], 2),
                "ols_ms_per_1k_input_tokens": round(coef[1], 3),
                "ols_ms_per_100_options_beyond_tokens": round(coef[2], 3),
                "residual_sd_ms": round(resid_sd, 1),
                "reading": ("a K-coefficient within +-residual_sd of 0 supports "
                            "single-trunk in-context scoring; a positive K-coefficient "
                            "several x residual_sd indicates per-option work beyond prefill."),
            }

    # ---- P3 K-calibration: dilution law, floors, accuracy, IIA
    if "p3_kcal" in fam:
        byk: dict[int, dict] = {}
        iia: dict[tuple, list] = {}
        for r in fam["p3_kcal"]:
            if r.get("error") or not r.get("probabilities"):
                continue
            K = r["K"]
            p = r["probabilities"]
            gold_opt = r.get("gold")
            b = byk.setdefault(K, {"p_gold": [], "acc": [], "floor00": [], "floor01": []})
            pg = p.get(gold_opt)
            if pg is not None:
                b["p_gold"].append(pg)
            if gold_opt is not None and r.get("choice") is not None:
                b["acc"].append(1.0 if r["choice"] == gold_opt else 0.0)
            vals = list(p.values())
            b["floor00"].append(sum(1 for v in vals if v == 0.0) / len(vals))
            b["floor01"].append(sum(1 for v in vals if v == 0.01) / len(vals))
        summary = {}
        for K, b in sorted(byk.items()):
            summary[str(K)] = {
                "n": len(b["p_gold"]),
                "mean_p_gold": round(_mean(b["p_gold"]), 4) if b["p_gold"] else None,
                "accuracy": round(_mean(b["acc"]), 4) if b["acc"] else None,
                "mean_frac_at_0.00": round(_mean(b["floor00"]), 3),
                "mean_frac_at_0.01": round(_mean(b["floor01"]), 3),
            }
        out["p3_kcal"] = {
            "by_K": summary,
            "reading": ("set-normalized read-out: mean_p_gold decays with K while accuracy "
                        "stays flat and floor mass grows ~linearly in K; independent "
                        "absolute scoring would keep mean_p_gold flat."),
        }

    # ---- P4 lattice: identical-option apportionment
    if "p4_lattice" in fam:
        ident: dict[int, dict] = {}
        for r in fam["p4_lattice"]:
            if r.get("error") or not r.get("probabilities") or r.get("variant") != "identical":
                continue
            vals = sorted(r["probabilities"].values(), reverse=True)
            k = r["K"]
            b = ident.setdefault(k, {"patterns": {}, "sums": []})
            key = tuple(round(v, 2) for v in vals)
            b["patterns"][str(list(key)[:4]) + ("..." if k > 4 else "")] = \
                b["patterns"].get(str(list(key)[:4]) + ("..." if k > 4 else ""), 0) + 1
            b["sums"].append(sum(vals))
        summary = {}
        for k, b in sorted(ident.items()):
            summary[str(k)] = {"n": len(b["sums"]),
                               "distinct_patterns": len(b["patterns"]),
                               "top_patterns": dict(sorted(b["patterns"].items(),
                                                           key=lambda kv: -kv[1])[:3]),
                               "sum_min": round(min(b["sums"]), 3),
                               "sum_max": round(max(b["sums"]), 3)}
        ties = {}
        for r in fam["p4_lattice"]:
            if r.get("variant") == "near_tie" and not r.get("error"):
                p = r.get("probabilities") or {}
                if p:
                    mx = max(p.values())
                    winners = [k for k, v in p.items() if v == mx]
                    ties[r.get("choice")] = ties.get(r.get("choice"), 0) + 1
        out["p4_lattice"] = {
            "identical_options": summary,
            "near_tie_choice_counts": ties,
            "reading": ("identical options expose the apportionment rule directly "
                        "(e.g. K=3: [0.34,0.33,0.33] => largest-remainder to 1.00; "
                        "[0.33,0.33,0.33] => round-down leaving 0.99), and near-tie "
                        "flip counts bound the internal noise scale and tie-break bias."),
        }

    # ---- P5 horizon: accuracy/abstention by event month
    if "p5_horizon" in fam:
        bymonth: dict[str, dict] = {}
        for r in fam["p5_horizon"]:
            if r.get("error"):
                continue
            m = r.get("month") or "?"
            frame = r.get("frame")
            b = bymonth.setdefault(m, {"n": 0, "gold_hits": 0, "abstain": 0,
                                       "did_not_occur": 0, "kinds": {}})
            b["n"] += 1
            b["kinds"][r.get("kind")] = b["kinds"].get(r.get("kind"), 0) + 1
            ch = r.get("choice")
            if ch == r.get("gold"):
                b["gold_hits"] += 1
            if ch == "cannot determine from what I know":
                b["abstain"] += 1
            if ch == "it did not occur":
                b["did_not_occur"] += 1
        out["p5_horizon"] = {
            "by_month": {m: dict(b, gold_rate=round(b["gold_hits"] / b["n"], 3) if b["n"] else None)
                         for m, b in sorted(bymonth.items())},
            "reading": ("a fixed parametric cutoff shows pre-months near ceiling and "
                        "post-months at/below the 'did not occur'+abstain floor with a "
                        "cliff at one month; rolling or retrieved knowledge shows no cliff."),
        }
    return out


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _ols(X, y):
    n, k = len(y), len(X[0])
    A = [[sum(X[i][a] * X[i][b] for i in range(n)) for b in range(k)] for a in range(k)]
    bv = [sum(X[i][a] * y[i] for i in range(n)) for a in range(k)]
    M = [A[a][:] + [bv[a]] for a in range(k)]
    for c in range(k):
        p = max(range(c, k), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        for r in range(k):
            if r != c:
                f = M[r][c] / M[c][c]
                for j in range(c, k + 1):
                    M[r][j] -= f * M[c][j]
    return [M[a][k] / M[a][a] for a in range(k)]


def _resid_sd(X, y, coef):
    n = len(y)
    ss = sum((y[i] - sum(X[i][a] * coef[a] for a in range(len(coef)))) ** 2
             for i in range(n))
    return (ss / max(n - len(coef), 1)) ** 0.5


# --------------------------------------------------------------- dry run
def _apportion_largest_remainder(weights: list[float], total: int = 100) -> list[int]:
    """Integer hundredths summing exactly to `total` (largest-remainder rule)."""
    s = sum(weights)
    raw = [w / s * total for w in weights]
    floors = [int(x) for x in raw]
    rem = total - sum(floors)
    order = sorted(range(len(raw)), key=lambda i: raw[i] - floors[i], reverse=True)
    for i in order[:rem]:
        floors[i] += 1
    return floors


_P1_RATES = {
    "baseline_empty": 0.0, "baseline_spaces": 0.0,
    "ctrl_cyrillic": 0.98, "ctrl_cjk_common": 0.95,
    "cjk_ext_a": 1.95, "cherokee": 2.1, "ethiopic": 2.0, "yi_syllables": 2.05,
    "hangul_rare": 1.2, "cjk_compat_ideo": 1.9,
    "cjk_ext_b": 1.4, "math_bold": 1.6, "emoji_rare": 1.3, "regional_pairs": 1.55,
    "combining_decomposed": 1.4, "combining_precomposed": 0.7,
    "devanagari_conjunct": 1.0,
    "nbsp_run": 0.0, "tab_run": 0.0, "newline_run": 0.0,
    "ideographic_space_run": 0.0, "zwj_run": 0.0, "zwnj_run": 0.0,
    "bom_run": 0.0, "soft_hyphen_run": 0.0, "mixed_space_kinds": 0.0,
    "single_spaces_between_words": 0.30, "double_spaces_between_words": 0.30,
}


def _simulate_row(call: ProbeCall, rng: random.Random) -> dict:
    """Synthetic response with planted ground truth, to exercise the analyzers.

    The planted rules are DELIBERATELY one candidate world (largest-remainder
    display quantization, Luce dilution in P3, a 2024-11 cutoff in P5, byte-ish
    fallback rates in P1, zero per-option cost in P2): a dry run that recovers
    them proves the analyzer plumbing, nothing about the real server.
    """
    payload = call.request.to_payload()
    q = next(iter(payload["questions"].values()))
    criteria = q.get("criteria", {"o0": "yes", "o1": "no"})
    opts = list(criteria.values())
    k = len(opts)
    in_tok = 316 + int(len(json.dumps(payload, ensure_ascii=False)) / 4)
    fam = call.family
    meta = call.meta

    if fam == "p1_fallback":
        if meta.get("cls") == "lone_surrogate":
            return {"family": fam, "config_id": call.config_id, "cold": False,
                    "http_status": 400, "error": "simulated: lone surrogate rejected",
                    "wall_ms": 120.0, "first_byte_ms": 118.0, "upstream_ms": None,
                    "usage_input_tokens": None, "usage_output_tokens": None,
                    "response_bytes": 90, "probabilities": {}, "choice": None,
                    **{kk: vv for kk, vv in meta.items() if kk != "raw_state"}}
        rate = _P1_RATES.get(meta.get("cls", ""), 1.0)
        in_tok = 316 + round(meta.get("sample_chars", 0) * rate)
        probs = {"yes": 0.01, "no": 0.99}
        choice = "no"
    elif fam == "p1_merge":
        rate = {"ab_x60": 0.35, "a60_b60": 0.60, "a_ba59_b": 0.40,
                "th_x80": 0.25, "t80_h80": 0.60}.get(meta.get("cls"), None)
        if rate is None:
            rate = 0.30 if meta.get("cls", "").startswith("hello") else \
                   0.95 if meta.get("cls", "").startswith("digit") else \
                   0.05 if meta.get("cls", "").startswith("dash") else 0.25
        in_tok = 316 + round(meta.get("sample_chars", 0) * rate)
        probs = {"yes": 0.01, "no": 0.99}
        choice = "no"
    elif fam == "p4_lattice":
        if meta.get("variant") == "identical":
            units = _apportion_largest_remainder([1.0] * k)
        else:
            w = [4.0 if o == "zzq" else 2.0 if o == "ZZQ" else 1.0 for o in opts]
            units = _apportion_largest_remainder(w)
        # p4 rows are captured KEYED (o0..oN), because identical option texts
        # would collapse under name-flattening (record_from_response)
        keys = list(criteria.keys())
        probs = {keys[i]: units[i] / 100.0 for i in range(k)}
        choice = max(probs, key=lambda kk: (probs[kk], -list(probs).index(kk)))
    elif fam == "p3_kcal":
        w = [12.18 if o == meta.get("gold") else 1.0 for o in opts]
        units = _apportion_largest_remainder(w)
        probs = {o: units[i] / 100.0 for i, o in enumerate(opts)}
        choice = meta.get("gold") if probs.get(meta.get("gold"), 0) == max(probs.values()) \
            else max(probs, key=probs.get)
    elif fam == "p5_horizon":
        month, kind, frame = meta.get("month"), meta.get("kind"), meta.get("frame")
        known = kind == "pre" or (kind == "post" and month <= "2024-11")
        if kind == "fictional" or known:
            choice = meta.get("gold")
            w = [8.0 if o == choice else 1.0 for o in opts]
        elif frame == "abstain":
            choice = "cannot determine from what I know"
            w = [8.0 if o == choice else 1.0 for o in opts]
        else:
            choice = "it did not occur" if rng.random() < 0.7 else \
                next(o for o in opts if o.startswith("in ") and o != f"in {month}")
            w = [8.0 if o == choice else 1.0 for o in opts]
        units = _apportion_largest_remainder(w)
        probs = {o: units[i] / 100.0 for i, o in enumerate(opts)}
    else:  # p2_grid and anything else: sharp vector, pure-prefill timing
        w = [50.0] + [1.0] * (k - 1)
        units = _apportion_largest_remainder(w)
        probs = {o: units[i] / 100.0 for i, o in enumerate(opts)}
        choice = opts[0]

    upstream = 73.0 + 6.05 * in_tok / 1000.0 + rng.gauss(0, 4)
    return {
        "family": fam, "config_id": call.config_id, "cold": False,
        "http_status": 200, "error": None,
        "wall_ms": round(upstream + 210 + rng.gauss(0, 8), 3),
        "first_byte_ms": round(upstream + 205, 3),
        "upstream_ms": round(upstream, 1),
        "usage_input_tokens": in_tok,
        "usage_output_tokens": 24 + int(9.6 * (k - 1)),
        "response_bytes": 40 + 18 * k,
        "probabilities": probs, "choice": choice,
        **{kk: vv for kk, vv in meta.items() if kk != "raw_state"},
    }


def build_calls(families: list[str], reps: int = 0) -> list[ProbeCall]:
    calls: list[ProbeCall] = []
    for fam in families:
        b = BUILDERS[fam]
        if fam == "p3_kcal" and reps:
            calls.extend(b(reps=reps))
        elif reps:
            try:
                calls.extend(b(reps=reps))
            except TypeError:
                calls.extend(b())
        else:
            calls.extend(b())
    return calls


def estimate_cost(calls: list[ProbeCall]) -> dict:
    per_fam: dict[str, dict] = {}
    for c in calls:
        chars = len(json.dumps(c.request.to_payload(), ensure_ascii=False))
        tok = 316 + chars // 4
        f = per_fam.setdefault(c.family, {"calls": 0, "est_input_tokens": 0})
        f["calls"] += 1
        f["est_input_tokens"] += tok
    total_tok = sum(f["est_input_tokens"] for f in per_fam.values())
    return {"per_family": per_fam, "est_input_tokens_total": total_tok,
            "est_cost_usd_at_0p042_per_M_in": round(total_tok / 1e6 * 0.042, 4)}


def dry_run(out_dir: Path, families: list[str], seed: int) -> int:
    calls = build_calls(families)
    rng = random.Random(seed)
    rows = [_simulate_row(c, rng) for c in calls]
    analysis = analyze(rows)
    planted = {
        "p1_rates": _P1_RATES,
        "p3_model": "Luce with u_gold=ln(12.18), unit utilities elsewhere",
        "p4_rule": "largest-remainder integer hundredths (sums exactly 1.00)",
        "p5_cutoff": "2024-11 (months after are unknown; abstain-frame says cannot determine)",
        "p2_per_option_ms": 0.0,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "probe2_dryrun.json").write_text(json.dumps(
        {"mode": "dry_run", "seed": seed, "plan": estimate_cost(calls),
         "planted_truth": planted, "analysis": analysis},
        indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[dry-run] {len(rows)} simulated rows -> {out_dir/'probe2_dryrun.json'}")
    _print_analysis(analysis)
    return 0


def _print_analysis(analysis: dict) -> None:
    for fam in ("p1_fallback", "p1_merge", "p2_grid", "p3_kcal", "p4_lattice",
                "p5_horizon"):
        a = analysis.get(fam)
        if not a:
            continue
        summary = {k: v for k, v in a.items() if k != "reading"}
        txt = json.dumps(summary, ensure_ascii=False)
        print(f"--- {fam}: {txt[:900]}{'...' if len(txt) > 900 else ''}")


def _fire_raw_lone_surrogate(call: ProbeCall, client, redactor: Redactor) -> dict:
    """Dispatch the lone-surrogate probe as a raw ASCII body.

    The JSON body carries the lone-surrogate escape as literal ASCII text
    (ensure_ascii=True), so serialization cannot fail client-side; whatever
    the server's parser/normalizer does with it (accept, 400, or silent
    repair) is the measured signature.
    """
    payload = call.request.to_payload()
    payload["state"] = call.meta.get("raw_state", payload["state"])
    body_bytes = json.dumps(payload, ensure_ascii=True).encode("ascii")
    started = time.perf_counter()
    status, err, body = 0, None, b""
    try:
        resp = client.post("/v1/systemone", content=body_bytes,
                           headers={"Content-Type": "application/json"})
        status = resp.status_code
        body = resp.content
    except Exception as exc:
        err = f"{type(exc).__name__}:{exc}"
    wall = (time.perf_counter() - started) * 1000.0
    usage = {}
    try:
        usage = (json.loads(body.decode("utf-8", "replace")) or {}).get("usage") or {}
    except Exception:
        pass
    row = {"family": call.family, "config_id": call.config_id, "cold": False,
           "http_status": status, "error": err or (None if status == 200 else f"http_{status}"),
           "wall_ms": round(wall, 3), "first_byte_ms": None, "upstream_ms": None,
           "usage_input_tokens": usage.get("input_tokens"),
           "usage_output_tokens": usage.get("output_tokens"),
           "response_bytes": len(body), "probabilities": {}, "choice": None,
           "body_excerpt": redactor.text(body.decode("utf-8", "replace"))[:200],
           **{k: v for k, v in call.meta.items() if k != "raw_state"}}
    return row


def _fire_keep_keys(call: ProbeCall, client) -> dict:
    """Like ap.record_from_response but preserves the server's raw probability
    keys (o0..oN) instead of flattening to option texts - required for the
    identical-option lattice probes, whose texts would collapse."""
    payload = call.request.to_payload()
    started = time.perf_counter()
    first_byte_ms = None
    body = b""
    status = 0
    upstream_ms = None
    error = None
    try:
        with client.stream("POST", "/v1/systemone", json=payload) as resp:
            status = resp.status_code
            hdrs = {k.lower(): v for k, v in resp.headers.items()}
            u = hdrs.get(ap.UPSTREAM_HEADER)
            if u is not None:
                try:
                    upstream_ms = float(u)
                except ValueError:
                    upstream_ms = None
            for chunk in resp.iter_bytes():
                if first_byte_ms is None:
                    first_byte_ms = (time.perf_counter() - started) * 1000.0
                body += chunk
    except Exception as exc:
        error = f"{type(exc).__name__}:{exc}"
    wall_ms = (time.perf_counter() - started) * 1000.0
    try:
        parsed = json.loads(body.decode("utf-8")) if body else {}
    except Exception:
        parsed = {}
    usage = parsed.get("usage") or {}
    probs = {}
    choice_key = None
    for _qid, ans in (parsed.get("answers") or {}).items():
        if isinstance(ans, dict):
            pp = ans.get("probabilities")
            if isinstance(pp, dict):
                probs = {k: float(v) for k, v in pp.items()
                         if isinstance(v, (int, float)) and not isinstance(v, bool)}
            ch = ans.get("choice")
            if isinstance(ch, str):
                choice_key = ch
            break
    row = {"family": call.family, "config_id": call.config_id, "cold": False,
           "http_status": status,
           "error": error or (None if status == 200 else f"http_{status}"),
           "wall_ms": round(wall_ms, 3),
           "first_byte_ms": round(first_byte_ms, 3) if first_byte_ms is not None else None,
           "upstream_ms": upstream_ms,
           "usage_input_tokens": usage.get("input_tokens") if isinstance(usage, dict) else None,
           "usage_output_tokens": usage.get("output_tokens") if isinstance(usage, dict) else None,
           "response_bytes": len(body),
           "probabilities": probs, "choice": choice_key,
           **call.payload_metrics()}
    row.update(call.meta)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs_archprobe")
    parser.add_argument("--families", default=",".join(BUILDERS))
    parser.add_argument("--plan", action="store_true",
                        help="print the call plan + cost estimate and exit 0")
    parser.add_argument("--dry-run", action="store_true",
                        help="exercise builders+analyzers on synthetic rows (offline)")
    parser.add_argument("--seed", type=int, default=97)
    parser.add_argument("--reps", type=int, default=0,
                        help="override per-family reps (0 = builders' default)")
    parser.add_argument("--limit", type=int, default=0,
                        help="run only N randomized calls (cheap smoke)")
    parser.add_argument("--reset-every", type=int, default=40)
    args = parser.parse_args()

    families = [f.strip() for f in args.families.split(",") if f.strip()]
    calls = build_calls(families, args.reps)
    est = estimate_cost(calls)

    if args.dry_run:
        return dry_run(Path(args.out), families, args.seed)

    if args.plan:
        print(json.dumps({"plan": est, "families": families}, indent=2))
        return 0

    print("[plan]", json.dumps(est["per_family"], indent=1))
    print("[plan] est cost:", est["est_cost_usd_at_0p042_per_M_in"], "USD")

    if not (live_calls_allowed() and get_api_key()):
        print("[plan-only] live gate not satisfied (need JEVO_ALLOW_LIVE=1 and "
              "TYPESAFE_API_KEY in the process environment). Nothing dispatched.")
        return 1

    rng = random.Random(args.seed)
    rng.shuffle(calls)
    if args.limit:
        calls = calls[: args.limit]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "probe2_rows.jsonl"
    if not rows_path.exists():
        rows_path.write_text("", encoding="utf-8")

    redactor = Redactor.from_environment()
    http = HttpxTransport(base_url=base_url(), api_key=get_api_key() or "",
                          timeout_seconds=120.0, redactor=redactor)
    client = http._client
    rows: list[dict] = []
    since_reset = 0
    with rows_path.open("a", encoding="utf-8") as fh:
        for call in calls:
            if since_reset >= args.reset_every:
                http.reset_pool(); client = http._client; since_reset = 0
            if call.meta.get("cls") == "lone_surrogate":
                row = _fire_raw_lone_surrogate(call, client, redactor)
            elif call.family == "p4_lattice":
                row = _fire_keep_keys(call, client)
            else:
                row = ap.record_from_response(call, client, cold=False)
            row["seq"] = len(rows)
            rows.append(row)
            fh.write(redactor.text(json.dumps(row, ensure_ascii=False)) + "\n")
            fh.flush()
            since_reset += 1
            if row.get("error"):
                print(f"[err] {call.config_id}: {row['error']}", file=sys.stderr)
    http.close()

    analysis = analyze(rows)
    (out / "probe2_analysis.json").write_text(
        json.dumps(analysis, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    total_in = sum(r.get("usage_input_tokens") or 0 for r in rows)
    total_out = sum(r.get("usage_output_tokens") or 0 for r in rows)
    (out / "BILLING-probe2.json").write_text(json.dumps({
        "n_calls": len(rows),
        "n_errors": sum(1 for r in rows if r.get("error")),
        "usage_input_tokens": total_in,
        "usage_output_tokens": total_out,
        "estimated_cost_usd_at_0p042_per_M_in": round(total_in / 1e6 * 0.042, 4),
        "seed": args.seed, "families": families, "limit": args.limit,
        "authorization": "live dispatch requires JEVO_ALLOW_LIVE=1 + key in env "
                         "(DESIGN.md section 12 parent approval recorded in-session)",
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"n_calls": len(rows), "input_tokens": total_in,
                      "rows": str(rows_path),
                      "analysis": str(out / "probe2_analysis.json")}, indent=2))
    _print_analysis(analysis)
    return 0


if __name__ == "__main__":
    sys.exit(main())
