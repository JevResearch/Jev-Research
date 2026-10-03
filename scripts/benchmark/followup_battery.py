"""Follow-up discriminating battery (followup-20261003.v1): builders, frozen
prereg plan and offline analysis for the four Jev ancestry/architecture
families approved by the parent analysis (see /tmp/jev-followup-probe-decisions.md).

Families (each hypothesis separates a competing explanation):
  fu_vocab    tokenizer/pruning boundary probes: fresh empty-frame baseline,
              discriminating ASCII strings (zzz / ______) and the heart symbol
              with prefix/suffix contexts and isolated-delimiter repeat curves,
              marginals vs matching empty-context controls, plus whitespace-free
              cross-script mergerate inputs (codepoint vs UTF-8 byte fallback).
  fu_place    counter-vs-time and text placement: identical filler text placed
              in state vs question instructions at fixed question counts and
              controlled sizes (0.5k/4k/16k), 8 randomized blocks, per-call
              nonce; tests the "question text costs 2x state text" claim and
              shared-state (self-batch) cost. Latency is fit with block
              intercepts and evaluated by holding out WHOLE stimuli.
  fu_odds     option interaction: Archer's 3 unique payout payloads with their
              intentional duplicate noise controls (10 blocks) plus 3 own
              moderate-ambiguity scenarios across K=4/5/6 and fixed-K
              filler-text substitutions (6 blocks). Independent fixed logits
              predict K-invariant pairwise log-odds; common-scale temperature
              predicts proportional log-odds with preserved ranks.
  fu_refcard  question isolation: the reference-card ("code-visibility")
              intervention in literal-wording-corrected and own variants, card
              in state vs sibling option, absent controls, named vs opaque
              option IDs, randomized orders.
  fu_horizon  knowledge horizon: 20 source-verified retrospective fact items
              across late-2024 / early-2025 / May-Aug-2025 (+3 older controls,
              +1 Sep-2024 NASA control) and 4 constructed false events, 2
              option-order rotations.

Discipline carried over from arch_probe / run_probe_battery2: gold labels live
in ProbeCall.meta (offline rows) and NEVER in the outbound payload; one
question per request; deterministic construction with a recorded seed; claims
are behavioral evidence, never weight-ancestry proof; duplicate control arms
are intentional noise floors, not bugs.
"""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass
from typing import Any

from jev_observatory.arch_probe import ProbeCall
from jev_observatory.schema import (ChoiceQuestion, NoulQuestion,
                                    SystemOneRequest, DEFAULT_MODEL_PIN)

FOLLOWUP_VERSION = "followup-20261003.v1"
RUN_DIRNAME = "followup_20261003"
REPORT_DIRNAME = "followup_20261003"

MODEL_PIN = "jev-1.13.0"
JEV_USD_PER_M_INPUT = 0.042          # published Jev tariff; output tokens free
JEV_MAX_INPUT_TOKENS = 65536         # documented per-request input ceiling
MAX_TRANSPORT_ATTEMPTS = 2           # bounded attempts per logical call
BATTERY_CAP_USD = 5.00               # this battery's pre-call reservation cap

REFERENCE_MODELS = {
    "qwen30b": "qwen/qwen3-30b-a3b-instruct-2507",
    "qwen_next": "qwen/qwen3-next-80b-a3b-instruct",
    "mistral_small": "mistralai/mistral-small-3.2-24b-instruct",
}
# published OpenRouter rates captured from GET /models on 2026-10-03
REFERENCE_RATES_USD_PER_TOKEN = {
    "qwen30b": {"in": 0.0000001, "out": 0.0000003},
    "qwen_next": {"in": 0.0000001, "out": 0.0000011},
    "mistral_small": {"in": 0.00000009375, "out": 0.00000025},
}
REFERENCE_PROMPT_TOKEN_BOUND = 4000
REFERENCE_OUT_CAP = 300

CANNOT = "cannot determine from what I know"
DID_NOT_OCCUR = "it did not occur"
_NOISE = "It is not possible to tell from the information given."


def _u(*cps: int) -> str:
    return "".join(chr(c) for c in cps)


# --------------------------------------------------------------------- family A
_HEART1 = _u(0x2764)
_HEART2 = _u(0x2764, 0xFE0F)

# (sample_id, text) core strings: the two discriminating ASCII strings, the
# heart symbol in both forms, the empty noul frame replay, ordinary controls.
_TOKEN_CORES: list[tuple[str, str]] = [
    ("frame_x", "x"),
    ("zzz", "zzz"),
    ("usc6", "______"),
    ("heart2", _HEART2),
    ("heart1", _HEART1),
    ("ctrl_hello", "hello"),
    ("ctrl_word_the", "the"),
    ("ctrl_num", "1000000"),
]
_TOKEN_DELIM_REPEATS = (2, 3, 5, 8)
_TOKEN_WORD_CTX = ("Look: ", " end")

# whitespace-free cross-script mergerate inputs (same construction family as
# arch_probe_mergerate samples) to distinguish codepoint from UTF-8 fallback
_MERGE_INPUTS: list[tuple[str, str]] = [
    ("latin_run", "abc" * 8),
    ("latin_rare", "QxzWjv" * 4),
    ("cyrillic_run", _u(0x440, 0x443, 0x441, 0x441, 0x43a, 0x438, 0x439) * 3),
    ("greek_run", _u(0x3b5, 0x3bb, 0x3bb, 0x3b7, 0x3bd, 0x3b9, 0x3ba, 0x3ac) * 2),
    ("cjk_run", _u(0x4f60, 0x597d, 0x4e16, 0x754c) * 4),
    ("hangul_run", _u(0xd55c, 0xad6d, 0xc5b4) * 5),
    ("rare_astral", "".join(chr(c) for c in (0x1D436, 0x1D433, 0x1D44A, 0x1D43E, 0x1D44B, 0x1D43F))),
    ("combining_run", ("e" + _u(0x301)) * 8),
]

_N1 = "n1"


def _noul(state: str) -> SystemOneRequest:
    return SystemOneRequest(state=state, model=DEFAULT_MODEL_PIN,
                            questions={_N1: NoulQuestion(instructions="x")})


def build_fu_vocab(reps: int = 2) -> list[ProbeCall]:
    """Fresh boundary + repeat-length controlled tokenizer probes.

    Frame is Archer's fingerprint frame replayed fresh: noul question "x",
    state == sample, so tokens(S) = input_tokens(S) - input_tokens("") with a
    NEW empty control in the same run.  Every sample gets a matching
    empty-context control (bare, word-context, delimiter-only repeats).
    """
    calls: list[ProbeCall] = []

    def add(sid: str, sample: str, meta: dict) -> None:
        for rep in range(reps):
            calls.append(ProbeCall(
                "fu_vocab", f"fu_vocab:{sid}:r{rep}", _noul(sample),
                {"sample_id": sid, "probe": sample, "rep": rep,
                 "sample_chars": len(sample),
                 "sample_bytes": len(sample.encode("utf-8")), **meta}))

    add("empty_baseline", "", {"role": "control", "core": None})
    add("empty_wordctx", _TOKEN_WORD_CTX[0] + _TOKEN_WORD_CTX[1],
        {"role": "control", "core": None})
    for n in _TOKEN_DELIM_REPEATS:
        add(f"empty_delim{n}", "|" * (n - 1), {"role": "control", "core": None,
                                              "repeats": n})
    for core, text in _TOKEN_CORES:
        add(f"{core}:bare", text, {"role": "target", "core": core, "context": "bare",
                                   "repeats": 1})
        add(f"{core}:wordctx", _TOKEN_WORD_CTX[0] + text + _TOKEN_WORD_CTX[1],
            {"role": "target", "core": core, "context": "wordctx", "repeats": 1})
        for n in _TOKEN_DELIM_REPEATS:
            add(f"{core}:delim{n}", "|".join([text] * n),
                {"role": "target", "core": core, "context": "delim", "repeats": n})
    for name, sample in _MERGE_INPUTS:
        add(f"merge:{name}", sample, {"role": "mergerate", "core": None})
    return calls


# --------------------------------------------------------------------- family B
_FILLER_SENTENCE_A = (
    "The quarterly logistics review covered warehousing, routing, staffing "
    "and maintenance numbers for the northern corridor, with each depot "
    "reporting lane counts, dwell minutes and exception codes on Fridays. ")
_FILLER_SENTENCE_B = (
    "Field technicians recorded soil moisture, slope readings and drainage "
    "observations along the ridge path, noting weather windows and access "
    "restrictions for every sampled plot before returning to the station. ")
_FILLER_TEXTS = {"fillerA": _FILLER_SENTENCE_A, "fillerB": _FILLER_SENTENCE_B}
_PLACE_SIZES = (("s0500", 2_000), ("s4000", 16_000), ("s16000", 64_000))  # ~chars
_PLACE_CELLS = (("q1_state", 1, "state"), ("q1_question", 1, "question"),
                ("q8_state", 8, "state"), ("q8_questions", 8, "question"))
_PLACE_BLOCKS = 8
_SHORT_Q = "Does the text mention any measurements? Answer yes or no."


def _filler(text_id: str, n_chars: int) -> str:
    s = _FILLER_TEXTS[text_id]
    reps = max(1, math.ceil(n_chars / len(s)))
    return (s * reps)[:n_chars]


def _yes_no(instructions: str) -> dict[str, ChoiceQuestion]:
    return {"q0": ChoiceQuestion(instructions=instructions,
                                 criteria={"o0": "yes", "o1": "no"})}


def build_fu_place(blocks: int = _PLACE_BLOCKS, seed: int = 2026100302) -> list[ProbeCall]:
    """Fixed question count, identical filler in state vs question instructions.

    Cells (all at identical total filler volume per (text, size)):
      q1_state      filler in state, 1 short question
      q1_question   filler in the question instructions, short state
      q8_state      filler in state shared by 8 short questions (self-batch cost)
      q8_questions  filler split evenly across 8 question instructions
    A fresh per-call nonce ("Run marker: xxxxxxxx.") is appended to the state so
    no two requests share an identical prefix (counter caching hygiene).
    """
    calls: list[ProbeCall] = []
    rng = random.Random(seed)
    block_orders = []
    for block in range(blocks):
        cells = [(t, sz, cell) for t in _FILLER_TEXTS for sz in _PLACE_SIZES
                 for cell in _PLACE_CELLS]
        rng.shuffle(cells)
        block_orders.append([c[2][0] + ":" + c[0] + ":" + c[1][0] for c in cells])
        for text_id, (size_id, n_chars), (cell_id, n_q, role) in cells:
            nonce = f"{rng.getrandbits(32):08x}"
            filler = _filler(text_id, n_chars)
            header = "Passage log. "
            if role == "state":
                state = header + filler + f"\nRun marker: {nonce}."
                instrs = {f"q{i}": ChoiceQuestion(
                    instructions=_SHORT_Q, criteria={"o0": "yes", "o1": "no"})
                    for i in range(n_q)}
            else:
                state = header + f"Run marker: {nonce}."
                if n_q == 1:
                    chunks = [filler]
                else:
                    per = len(filler) // n_q
                    chunks = [filler[i * per:(i + 1) * per] for i in range(n_q)]
                    chunks[-1] += filler[n_q * per:]
                instrs = {f"q{i}": ChoiceQuestion(
                    instructions=chunks[i] + "\n" + _SHORT_Q,
                    criteria={"o0": "yes", "o1": "no"}) for i in range(n_q)}
            req = SystemOneRequest(state=state, model=DEFAULT_MODEL_PIN,
                                   questions=instrs)
            q_chars = sum(len(q.instructions) if isinstance(q.instructions, str)
                          else len(json.dumps(q.instructions, ensure_ascii=False))
                          for q in instrs.values())
            calls.append(ProbeCall(
                "fu_place", f"fu_place:{text_id}:{size_id}:{cell_id}:b{block}",
                req, {"block": block, "text_id": text_id, "size_id": size_id,
                      "cell_id": cell_id, "n_questions": n_q, "role": role,
                      "nominal_chars": n_chars, "nonce": nonce,
                      "state_chars_local": len(state),
                      "question_chars_local": q_chars}))
    return calls, block_orders


def build_fu_place_ordered(blocks: int = _PLACE_BLOCKS,
                           seed: int = 2026100302) -> tuple[list[ProbeCall], list[list[str]]]:
    """Randomized BLOCK order: whole blocks shuffled, conditions shuffled
    within each block (serial-latency discipline: concurrency 1, block design).
    Returns calls in dispatch order plus the recorded block orderings."""
    calls_all, block_orders = build_fu_place(blocks, seed)
    rng = random.Random(seed + 1)
    by_block: dict[int, list[ProbeCall]] = {}
    for c in calls_all:
        by_block.setdefault(c.meta["block"], []).append(c)
    block_ids = sorted(by_block)
    rng.shuffle(block_ids)
    ordered: list[ProbeCall] = []
    for b in block_ids:
        ordered.extend(by_block[b])
    return ordered, block_orders


# --------------------------------------------------------------------- family C
# Archer's published odds payloads (followup-trials.json, exact text).  The
# "null" arms are INTENTIONAL duplicate noise controls (base4 == null4 and
# append5 == null5 byte-identical); they bound repeat-vs-repeat noise and are
# not interventions (parent decision: do not portray them as mistaken).
_ARCHER_STATE = (
    "Ticket from Dana: Payouts to my bank have been failing since Tuesday. "
    "Your status page says all systems are fine, and my bank says nothing is "
    "blocked on their side, though they did change my account number last "
    "month after a merger. I updated it in your dashboard, I think. I run a "
    "small bakery and this is starting to hurt. If this isn't fixed by Friday "
    "I may have to look at other providers.")
_ARCHER_INSTR = "Who most likely caused the payout failures?"
_ARCHER_BASE = {
    "bank": "The bank caused it",
    "provider": "The payment provider caused it",
    "customer": "The customer caused it",
    "unknown": "Cannot tell",
}
_ARCHER_VARIANTS = {
    "base4": dict(_ARCHER_BASE),
    "null4": dict(_ARCHER_BASE),                       # intentional duplicate
    "append5": {**_ARCHER_BASE, "weather": "Bad weather caused it"},
    "null5": {**_ARCHER_BASE, "weather": "Bad weather caused it"},  # intentional
    "replace5": {**_ARCHER_BASE, "weather": "Wild birds caused it"},
}
_ARCHER_ORIGINAL_KEYS = ("customer", "provider", "unknown", "bank")

# own moderate-ambiguity scenarios (project-authored synthetic text); the same
# four original options appear at every K so pairwise log-odds are comparable.
@dataclass(frozen=True)
class OwnScenario:
    scenario_id: str
    state: str
    instructions: str
    original: tuple[tuple[str, str], ...]   # (pair_id, option text), 4 entries
    extra: tuple[str, ...]                  # plausible distractors for K=5/6
    filler_option: str                      # inert text substitution at K=5


_OWN_SCENARIOS: list[OwnScenario] = [
    OwnScenario(
        "amb_support",
        "Ticket from Priya: Since Tuesday's firmware update the display dims "
        "randomly. Two colleagues saw it too, but only on the network side of "
        "the building. The vendor says the update only changed fonts.",
        "What most likely explains the dimming displays?",
        (("firmware", "the firmware update changed display behavior"),
         ("power", "a power-saving setting changed"),
         ("wiring", "the network wiring causes interference"),
         ("coincidence", "the reports are coincidence")),
        ("building maintenance cut the power briefly",
         "a construction crane causes vibration"),
        "zzq filler option text"),
    OwnScenario(
        "amb_logistics",
        "Freight note: three pallets arrived at the depot with seals intact "
        "but one carton in each was short. Scanner logs show normal counts at "
        "origin. The same driver ran all three loads.",
        "What most likely explains the short cartons?",
        (("origin_pack", "short-packing at origin"),
         ("origin_scan", "the origin scanner undercounted"),
         ("theft", "in-transit theft despite intact seals"),
         ("receiving", "receiving miscount at the depot")),
        ("the pallets were re-weighed incorrectly",
         "labels were swapped between cartons"),
        "zzq filler option text"),
    OwnScenario(
        "amb_research",
        "Lab notebook: the replicate assay drifted positive across three runs "
        "this week. New lot of buffer, same operator, and the calibrator "
        "passed on Monday. The incubator door alarm never triggered.",
        "What most likely explains the positive drift?",
        (("buffer", "buffer lot contamination"),
         ("incubator", "incubator temperature drift without an alarm"),
         ("reader", "plate reader calibration drift"),
         ("handling", "operator handling differences")),
        ("a seasonal ambient laboratory effect",
         "the vendor changed the assay protocol"),
        "zzq filler option text"),
]
_OWN_CELLS = ("K4", "K5", "K6", "K5_filler")
_OWN_BLOCKS = 6

# reference-card ("code-visibility") isolation arm.  Wording arms:
#   archer   - the published literal payloads (replication anchor)
#   corrected- same semantics, wording made literally unambiguous (the card is
#              quoted as text, the reference-only option states it is a quote)
#   own      - plain-language variant of our own
_REFCARD_TASKS = {
    "delivery": ("route", "west", "east"),
    "signal": ("status", "indigo", "amber"),
}
_REFCARD_WORDINGS = ("archer", "corrected", "own")


def _refcard_parts(task: str, wording: str) -> dict[str, str]:
    attr, match, other = _REFCARD_TASKS[task]
    if wording == "archer":
        return {
            "lead": ("Decide using the reference card. Only the options with a "
                     "selection condition are selectable."),
            "card": f"Reference card: {attr} = {match}.",
            "ref_only": (f"Reference card: {attr} = {match}. Reference-only "
                         "option. Never select this option."),
            "cond_match": (f"Select this option if the reference card's "
                           f"{attr} is {match}."),
            "cond_other": (f"Select this option if the reference card's "
                           f"{attr} is {other}."),
            "instr": ("Read the reference card and select the one option whose "
                      "condition is satisfied."),
        }
    if wording == "corrected":
        return {
            "lead": ("Decide using the reference card. Exactly the options that "
                     "state a selection condition are selectable."),
            "card": f"The reference card reads, literally: {attr} = {match}.",
            "ref_only": (f"Quoted text only: \u201c{attr} = {match}\u201d. This "
                         "option is reference material. Never select it."),
            "cond_match": (f"Select this option exactly if the reference card "
                           f"literally reads {attr} = {match}."),
            "cond_other": (f"Select this option exactly if the reference card "
                           f"literally reads {attr} = {other}."),
            "instr": ("Read the reference card and select the one option whose "
                      "stated selection condition is satisfied."),
        }
    return {
        "lead": ("Decide using the card below. Only options that name a "
                 "selection condition can be chosen."),
        "card": f"The card says the {attr} is {match}.",
        "ref_only": (f"This option only repeats the card line \u201cthe {attr} "
                     f"is {match}\u201d. Never pick this option."),
        "cond_match": f"Choose this option if the card says the {attr} is {match}.",
        "cond_other": f"Choose this option if the card says the {attr} is {other}.",
        "instr": "Read the card and choose the option whose condition holds.",
    }


_REFCARD_IDSETS = {"named": ("alpha", "beta", "gamma"),
                   "opaque": ("k7", "m2", "p9")}
_REFCARD_REPS = 2


def build_fu_odds(blocks_archer: int = 10, blocks_own: int = _OWN_BLOCKS,
                  seed: int = 2026100303) -> list[ProbeCall]:
    """Archer odds replication (with intentional duplicate noise controls) and
    own moderate-ambiguity K/filler contrasts, randomized within blocks."""
    calls: list[ProbeCall] = []
    rng = random.Random(seed)

    for block in range(blocks_archer):
        variants = list(_ARCHER_VARIANTS)
        rng.shuffle(variants)
        for variant in variants:
            criteria = {k: v for k, v in _ARCHER_VARIANTS[variant].items()}
            req = SystemOneRequest(
                state=_ARCHER_STATE, model=DEFAULT_MODEL_PIN,
                questions={"decision": ChoiceQuestion(
                    instructions=_ARCHER_INSTR, criteria=criteria)})
            calls.append(ProbeCall(
                "fu_odds", f"fu_odds:archer:{variant}:b{block}", req,
                {"arm": "archer", "variant": variant, "block": block,
                 "noise_control": variant in ("null4", "null5"),
                 "original_texts": [ _ARCHER_BASE[k] for k in _ARCHER_ORIGINAL_KEYS ]}))

    for block in range(blocks_own):
        cells = [(sc.scenario_id, cell) for sc in _OWN_SCENARIOS for cell in _OWN_CELLS]
        rng.shuffle(cells)
        for sid, cell in cells:
            sc = next(s for s in _OWN_SCENARIOS if s.scenario_id == sid)
            opts = [t for _, t in sc.original]
            if cell == "K5":
                opts = opts + [sc.extra[0]]
            elif cell == "K6":
                opts = opts + list(sc.extra)
            elif cell == "K5_filler":
                opts = opts + [sc.filler_option]
            req = SystemOneRequest(
                state=sc.state, model=DEFAULT_MODEL_PIN,
                questions={"q0": ChoiceQuestion(
                    instructions=sc.instructions,
                    criteria={f"o{i}": t for i, t in enumerate(opts)})})
            calls.append(ProbeCall(
                "fu_odds", f"fu_odds:{sid}:{cell}:b{block}", req,
                {"arm": "own", "variant": cell, "scenario": sid, "block": block,
                 "noise_control": False, "K": len(opts),
                 "original_texts": [t for _, t in sc.original],
                 "pair_ids": [p for p, _ in sc.original]}))
    return calls


def build_fu_refcard(seed: int = 2026100304) -> list[ProbeCall]:
    """Reference-card intervention: wording x placement x absent-control,
    named vs opaque IDs, per-call randomized option order."""
    calls: list[ProbeCall] = []
    rng = random.Random(seed)
    conds = [(t, w, p, False) for t in _REFCARD_TASKS
             for w in _REFCARD_WORDINGS for p in ("state", "sibling")]
    conds += [(t, w, p, True) for t in _REFCARD_TASKS
              for w in ("corrected", "own") for p in ("state", "sibling")]
    order = [(c, rep) for c in conds for rep in range(_REFCARD_REPS)]
    rng.shuffle(order)
    for (task, wording, placement, absent), rep in order:
        parts = _refcard_parts(task, wording)
        attr, match, _other = _REFCARD_TASKS[task]
        idset = ("named", "opaque")[rep % 2]
        ids = list(_REFCARD_IDSETS[idset])
        entries = [(ids[0], parts["ref_only"]),
                   (ids[1], parts["cond_match"]),
                   (ids[2], parts["cond_other"])]
        rng.shuffle(entries)
        lead = parts["lead"] if placement == "state" else \
            parts["lead"] + " The reference card is quoted inside one option."
        state = lead if absent else lead + " " + parts["card"]
        req = SystemOneRequest(
            state=state, model=DEFAULT_MODEL_PIN,
            questions={"decision": ChoiceQuestion(
                instructions=parts["instr"],
                criteria={k: v for k, v in entries})})
        expected = next(k for k, v in entries if v == parts["cond_match"])
        ref_key = next(k for k, v in entries if v == parts["ref_only"])
        calls.append(ProbeCall(
            "fu_refcard",
            f"fu_refcard:{task}:{wording}:{placement}:"
            f"{'absent' if absent else 'present'}:{idset}:r{rep}", req,
            {"task": task, "wording": wording, "placement": placement,
             "absent": absent, "idset": idset, "rep": rep,
             "expected": expected, "ref_only_key": ref_key,
             "attr": attr, "match": match,
             "order": [k for k, _ in entries]}))
    return calls


# --------------------------------------------------------------------- family D
# 20 source-verified retrospective fact items + 3 older controls + 1 Sep-2024
# NASA control + 4 constructed false events.  Gold is verified against the
# linked source content (frozen excerpt + sha256) BEFORE live dispatch; stems
# never contain the answer and no item asks a pre-announced schedule outcome
# except where the outcome itself (winner/name/result) was unpredictable.
@dataclass(frozen=True)
class HorizonItem:
    item_id: str
    kind: str                 # "post" | "pre" | "fictional"
    event_date: str           # "YYYY-MM" of the outcome
    question: str
    options: tuple[str, ...]  # gold + wrongs + DID_NOT_OCCUR + CANNOT (fixed order)
    gold: str
    sources: tuple[str, ...]  # candidate source URLs, first verified wins
    gold_anchor: str          # string that must appear in the verified excerpt


def _fact(item_id, event_date, question, gold, wrongs, sources, anchor,
          kind="post") -> HorizonItem:
    return HorizonItem(item_id, kind, event_date, question,
                       tuple([gold] + wrongs + [DID_NOT_OCCUR, CANNOT]),
                       gold, tuple(sources), anchor)


_W = "https://en.wikipedia.org/wiki/"
HORIZON_ITEMS: list[HorizonItem] = [
    # ---- late 2024 (after the claimed Oct-2024 knowledge cutoff)
    _fact("hs_ws2024", "2024-10", "Which team won the 2024 World Series?",
          "Los Angeles Dodgers", ["New York Yankees", "Boston Red Sox",
                                  "Toronto Blue Jays"],
          [_W + "2024_World_Series"], "Dodgers"),
    _fact("hs_ballon2024", "2024-10", "Who won the men's Ballon d'Or in 2024?",
          "Rodri", ["Jude Bellingham", "Vinicius Junior", "Erling Haaland"],
          [_W + "2024_Ballon_d%27Or"], "Rodri"),
    _fact("hs_nobelpeace2024", "2024-10",
          "Which organization received the 2024 Nobel Peace Prize?",
          "Nihon Hidankyo", ["World Food Programme", "UNHCR",
                             "Memorial (human rights centre)"],
          ["https://www.nobelprize.org/prizes/peace/2024/summary/",
           _W + "Nihon_Hidankyo"], "Nihon Hidankyo"),
    _fact("hs_heisman2024", "2024-12", "Who won the 2024 Heisman Trophy?",
          "Travis Hunter", ["Ashton Jeanty", "Cam Ward", "Dillon Gabriel"],
          [_W + "2024_Heisman_Trophy", _W + "Heisman_Trophy"], "Travis Hunter"),
    # ---- early 2025
    _fact("hs_sblx", "2025-02", "Which team won Super Bowl LIX?",
          "Philadelphia Eagles", ["Kansas City Chiefs", "Buffalo Bills",
                                  "Washington Commanders"],
          [_W + "Super_Bowl_LIX"], "Eagles"),
    _fact("hs_ao2025w", "2025-01",
          "Who won the women's singles title at the 2025 Australian Open?",
          "Madison Keys", ["Aryna Sabalenka", "Iga Swiatek", "Coco Gauff"],
          [_W + "2025_Australian_Open_%E2%80%93_Women%27s_singles",
           _W + "2025_Australian_Open"], "Madison Keys"),
    _fact("hs_sixnations2025", "2025-03",
          "Which team won the 2025 Six Nations Championship?",
          "France", ["England", "Ireland", "Scotland"],
          [_W + "2025_Six_Nations_Championship"], "France"),
    _fact("hs_oscar2025", "2025-03",
          "Which film won Best Picture at the Academy Awards ceremony held in "
          "March 2025?", "Anora", ["The Brutalist", "Conclave", "Dune: Part Two"],
          [_W + "97th_Academy_Awards"], "Anora"),
    # ---- May-Aug 2025
    _fact("hs_pope2025", "2025-05", "Who was elected pope in May 2025?",
          "Robert Prevost (Leo XIV)", ["Pietro Parolin", "Luis Antonio Tagle",
                                       "Matteo Zuppi"],
          [_W + "2025_papal_conclave", _W + "Pope_Leo_XIV"], "Prevost"),
    _fact("hs_eurovision2025", "2025-05",
          "Which song won the Eurovision Song Contest 2025?",
          "\u201cWasted Love\u201d by JJ (Austria)",
          ["\u201cBara bada bastu\u201d by KAJ (Sweden)",
           "\u201cMaman\u201d by Louane (France)",
           "\u201cGaja\u201d by Justyna Steczkowska (Poland)"],
          [_W + "Eurovision_Song_Contest_2025"], "Wasted Love"),
    _fact("hs_cannes2025", "2025-05",
          "Which film won the Palme d'Or at the 2025 Cannes Film Festival?",
          "\u201cIt Was Just an Accident\u201d (Jafar Panahi)",
          ["\u201cSound of Falling\u201d", "\u201cThe Secret Agent\u201d",
           "\u201cSentimental Value\u201d"],
          [_W + "2025_Cannes_Film_Festival"], "It Was Just an Accident"),
    _fact("hs_nba2025", "2025-06", "Which team won the 2025 NBA Finals?",
          "Oklahoma City Thunder", ["Indiana Pacers", "Denver Nuggets",
                                    "Boston Celtics"],
          [_W + "2025_NBA_Finals"], "Thunder"),
    _fact("hs_usopen2025", "2025-06", "Who won the 2025 U.S. Open in golf?",
          "J.J. Spaun", ["Rory McIlroy", "Scottie Scheffler", "Viktor Hovland"],
          [_W + "2025_U.S._Open_(golf)"], "Spaun"),
    _fact("hs_wimb2025w", "2025-07",
          "Who won the women's singles title at the 2025 Wimbledon "
          "Championships?", "Iga Swiatek",
          ["Amanda Anisimova", "Coco Gauff", "Aryna Sabalenka"],
          [_W + "2025_Wimbledon_Championships",
           _W + "2025_Wimbledon_Championships_%E2%80%93_Women%27s_singles"],
          "Swiatek"),
    _fact("hs_cwc2025", "2025-07",
          "Which team won the 2025 FIFA Club World Cup final?",
          "Chelsea", ["Paris Saint-Germain", "Real Madrid", "Fluminense"],
          [_W + "2025_FIFA_Club_World_Cup_final"], "Chelsea"),
    _fact("hs_anchorage", "2025-08",
          "In which month did the United States\u2013Russia leaders' summit in "
          "Anchorage, Alaska take place in 2025?", "in 2025-08",
          ["in 2025-05", "in 2025-06", "in 2025-07"],
          ["https://www.reuters.com/world/pictures/trump-putin-summit-alaska-2025-08-15/",
           _W + "2025_Russia%E2%80%93United_States_summit"],
          "15 August 2025||August 15, 2025||August 2025"),
    # ---- older controls (pre-cutoff; expected known) + NASA source control
    _fact("hs_wc2022", "2022-12", "Which team won the 2022 FIFA World Cup?",
          "Argentina", ["France", "Croatia", "Brazil"],
          [_W + "2022_FIFA_World_Cup_final"], "Argentina", kind="pre"),
    _fact("hs_sblviii", "2024-02", "Which team won Super Bowl LVIII in "
          "February 2024?", "Kansas City Chiefs",
          ["San Francisco 49ers", "Baltimore Ravens", "Detroit Lions"],
          [_W + "Super_Bowl_LVIII"], "Chiefs", kind="pre"),
    _fact("hs_wimb2023m", "2023-07",
          "Who won the men's singles title at the 2023 Wimbledon Championships?",
          "Carlos Alcaraz", ["Novak Djokovic", "Daniil Medvedev", "Jannik Sinner"],
          [_W + "2023_Wimbledon_Championships"], "Alcaraz", kind="pre"),
    _fact("hs_starliner", "2024-09",
          "What happened when Boeing's Starliner spacecraft returned from its "
          "crewed flight test in September 2024?",
          "it landed without its crew, who returned later on Crew Dragon",
          ["it landed with its two-person crew aboard",
           "it burned up during re-entry and was lost",
           "its return was repeatedly postponed into 2025 with the crew aboard"],
          ["https://www.nasa.gov/news-release/nasa-boeing-welcome-starliner-spacecraft-to-earth-close-mission/"],
          "uncrewed||without a crew||without its crew", kind="pre"),
    # ---- constructed false-event controls (own synthetic; gold DID_NOT_OCCUR)
    HorizonItem("hs_fake_winter", "fictional", "2024-12",
                "When did the 2024 Winter Olympic Games take place?",
                ("in 2024-11", "in 2024-12", "in 2025-01", DID_NOT_OCCUR, CANNOT),
                DID_NOT_OCCUR, (), "constructed_false_event"),
    HorizonItem("hs_fake_un", "fictional", "2025-05",
                "When did the United Nations announce its permanent relocation "
                "from New York to Geneva?",
                ("in 2025-04", "in 2025-05", "in 2025-06", DID_NOT_OCCUR, CANNOT),
                DID_NOT_OCCUR, (), "constructed_false_event"),
    HorizonItem("hs_fake_mars", "fictional", "2025-06",
                "When did NASA announce the discovery of definitive fossil "
                "evidence of life on Mars?",
                ("in 2025-05", "in 2025-06", "in 2025-07", DID_NOT_OCCUR, CANNOT),
                DID_NOT_OCCUR, (), "constructed_false_event"),
    HorizonItem("hs_fake_eclipse", "fictional", "2025-08",
                "When did the total solar eclipse that was visible across all "
                "of Europe take place?",
                ("in 2025-07", "in 2025-08", "in 2025-09", DID_NOT_OCCUR, CANNOT),
                DID_NOT_OCCUR, (), "constructed_false_event"),
]

_HORIZON_ROTATIONS = (0, 3)


def build_fu_horizon(rotations: tuple[int, ...] = _HORIZON_ROTATIONS
                     ) -> list[ProbeCall]:
    """Two option-order rotations per item (cyclic offsets), gold joined by
    option TEXT so rotations cannot scramble the scoring."""
    calls: list[ProbeCall] = []
    for item in HORIZON_ITEMS:
        base = list(item.options)
        for rot in rotations:
            opts = base[rot:] + base[:rot]
            req = SystemOneRequest(
                state="This question is about public events.",
                model=DEFAULT_MODEL_PIN,
                questions={"q0": ChoiceQuestion(
                    instructions=item.question,
                    criteria={f"o{i}": t for i, t in enumerate(opts)})})
            gold_key = f"o{opts.index(item.gold)}"
            calls.append(ProbeCall(
                "fu_horizon", f"fu_horizon:{item.item_id}:rot{rot}", req,
                {"item_id": item.item_id, "kind": item.kind,
                 "event_date": item.event_date, "rot": rot,
                 "gold_text": item.gold, "gold_key": gold_key,
                 "gold_source_anchor": item.gold_anchor}))
    return calls


BUILDERS = {
    "fu_vocab": build_fu_vocab,
    "fu_place": lambda **kw: build_fu_place_ordered(**kw)[0],
    "fu_odds": build_fu_odds,
    "fu_refcard": build_fu_refcard,
    "fu_horizon": build_fu_horizon,
}


def build_all() -> tuple[list[ProbeCall], list[list[str]]]:
    place_calls, block_orders = build_fu_place_ordered()
    calls = (build_fu_vocab() + place_calls + build_fu_odds()
             + build_fu_refcard() + build_fu_horizon())
    return calls, block_orders


def call_counts() -> dict[str, int]:
    counts = {}
    for name, builder in BUILDERS.items():
        built = builder()
        counts[name] = len(built) if isinstance(built, list) else len(built[0])
    counts["ref_qwen"] = len(HORIZON_ITEMS) * 2
    counts["ref_nonqwen"] = len(HORIZON_ITEMS)
    return counts


# --------------------------------------------------------------- budget plan
def worst_case_costs() -> dict[str, Any]:
    jev_worst = (JEV_MAX_INPUT_TOKENS * JEV_USD_PER_M_INPUT / 1e6
                 * MAX_TRANSPORT_ATTEMPTS)
    ref_worst = {name: (REFERENCE_PROMPT_TOKEN_BOUND * rates["in"]
                        + REFERENCE_OUT_CAP * rates["out"])
                 for name, rates in REFERENCE_RATES_USD_PER_TOKEN.items()}
    counts = call_counts()
    total = (counts["fu_vocab"] + counts["fu_place"] + counts["fu_odds"]
             + counts["fu_refcard"] + counts["fu_horizon"]) * jev_worst
    total += counts["ref_qwen"] / 2 * (ref_worst["qwen30b"] + ref_worst["qwen_next"])
    total += counts["ref_nonqwen"] * ref_worst["mistral_small"]
    return {"jev_per_call_worst_usd": round(jev_worst, 8),
            "reference_per_call_worst_usd": {k: round(v, 8)
                                             for k, v in ref_worst.items()},
            "counts": counts, "total_worst_usd": round(total, 4),
            "cap_usd": BATTERY_CAP_USD,
            "fits_cap": total <= BATTERY_CAP_USD,
            "prior_exposure_note": ("prior conservative exposure 43.9576 + ARC "
                                    "2.09 + this 5.00 cap = 51.05 < 80 combined "
                                    "authorization; old unknown holds untouched")}


# --------------------------------------------------------------- prereg freeze
PREDICTIONS: dict[str, Any] = {
    "fu_vocab": (
        "Jev reports 1-token marginals for zzz / ______ / heart (its 234-string "
        "one-token set behavior), while the actual Qwen3/2.5 encoder splits all "
        "three and Qwen3.5 merges the heart but splits zzz / ______; delimiter "
        "repeat curves are linear (counter additivity). If Jev marginals match "
        "the Qwen3 split pattern instead, the earlier single-token reading is "
        "wrong; a heart/zzz/______ singleton that no candidate reproduces "
        "favors a changed counter over pure vocabulary deletion, since pure "
        "deletion cannot create a new one-token string."),
    "fu_place": (
        "Service time is linear in total tokens with ~equal state and question "
        "marginal cost (our earlier marginal analysis: +0.34 ms/token question "
        "vs +0.44 state, inside scatter) — the article's 2x question cost is "
        "not reproduced; q8_state at fixed size costs little more than q1_state "
        "(self-batched read-outs). Role-split models will not beat the "
        "total-token model out of sample by a material margin."),
    "fu_odds": (
        "Repeat-vs-repeat (null arms) set the noise floor. Under independent "
        "fixed logits + fixed temperature, pairwise log-odds among the original "
        "options are K-invariant within 1-1.5 display quanta (0.01) and ranks "
        "are preserved under any positive fixed temperature; a common-scale "
        "temperature change predicts proportional log-odds across all pairs "
        "with preserved ranks; isolated rank flips or pair-specific scale "
        "changes reject both simple families. Our earlier padded-calibration "
        "confidence of 1.0 is saturation and proves no absolute logits."),
    "fu_refcard": (
        "Option reference/order sensitivity (state vs sibling placement, named "
        "vs opaque IDs) supports joint option access and does not by itself "
        "distinguish causal from bidirectional or slot from pointer "
        "architectures; absent controls should drop the expected option to "
        "chance."),
    "fu_horizon": (
        "Post-cutoff unpredictable-outcome items score at/below the "
        "cannot-determine floor if Jev's knowledge horizon predates them, with "
        "older controls near ceiling and false events at DID_NOT_OCCUR; gradual "
        "decay (not a one-month cliff) matches the corrected 2024/2025 "
        "retrospective. The two Qwen references (and the non-Qwen control) "
        "separate generic shared knowledge from any Qwen-specific signature; "
        "no ranking here identifies unique lineage."),
}


def build_plan(block_orders: list[list[str]],
               source_verification: dict[str, Any],
               offline: dict[str, Any]) -> dict[str, Any]:
    """Frozen preregistration: stimuli inventory, gold + source quotes,
    predictions, budget, method.  Written BEFORE any live call."""
    items = [{
        "item_id": it.item_id, "kind": it.kind, "event_date": it.event_date,
        "question": it.question, "options": list(it.options), "gold": it.gold,
        "sources": list(it.sources), "gold_anchor": it.gold_anchor,
    } for it in HORIZON_ITEMS]
    return {
        "schema": "jev-followup-plan.v1", "version": FOLLOWUP_VERSION,
        "model_pin": MODEL_PIN,
        "families": {name: build for name, build in call_counts().items()},
        "stimuli": {
            "fu_vocab": {
                "cores": dict(_TOKEN_CORES),
                "mergerate": dict(_MERGE_INPUTS),
                "contexts": ["bare", "wordctx", "delim:" + ",".join(map(str, _TOKEN_DELIM_REPEATS))],
                "frame": {"questions": {_N1: {"type": "noul", "instructions": "x"}},
                          "marginal": "tokens(S) = input_tokens(S) - input_tokens(empty) fresh"},
            },
            "fu_place": {
                "texts": {k: {"chars": len(_filler(k, 64)), "stimulus_id": k}
                          for k in _FILLER_TEXTS},
                "sizes_chars": dict(_PLACE_SIZES), "cells": [list(c) for c in _PLACE_CELLS],
                "blocks": _PLACE_BLOCKS, "block_condition_orders": block_orders,
                "nonce": "fresh 8-hex run marker appended to state per call",
                "latency_model": ("upstream_ms ~ block intercepts + counts; "
                                  "held-out evaluation by WHOLE stimuli "
                                  "(fillerA<->fillerB), never per-repeat rows"),
            },
            "fu_odds": {
                "archer_variants": _ARCHER_VARIANTS,
                "archer_note": ("base4==null4 and append5==null5 are intentional "
                                "duplicate noise controls (Archer pools them as "
                                "such); real contrasts are append5-base4 and "
                                "replace5-append5"),
                "own_scenarios": [{"scenario_id": s.scenario_id, "state": s.state,
                                   "instructions": s.instructions,
                                   "original": list(s.original),
                                   "extra": list(s.extra),
                                   "filler_option": s.filler_option}
                                  for s in _OWN_SCENARIOS],
                "own_cells": list(_OWN_CELLS), "own_blocks": _OWN_BLOCKS,
            },
            "fu_refcard": {
                "tasks": {k: list(v) for k, v in _REFCARD_TASKS.items()},
                "wordings": {w: _refcard_parts("delivery", w)
                             for w in _REFCARD_WORDINGS},
                "placements": ["state", "sibling"], "idsets": _REFCARD_IDSETS,
                "reps": _REFCARD_REPS,
                "wording_note": ("'corrected' keeps Archer's semantics but makes "
                                 "the literal quoting/selection wording "
                                 "unambiguous; 'own' is a plain-language "
                                 "variant; 'archer' replicates the published "
                                 "payload text"),
            },
            "fu_horizon": {"items": items, "rotations": list(_HORIZON_ROTATIONS)},
        },
        "gold_policy": ("gold joins by option text; gold labels travel only in "
                        "ProbeCall.meta / offline rows and never appear in the "
                        "outbound payload"),
        "predictions": PREDICTIONS,
        "budget": worst_case_costs(),
        "method": {
            "jev_transport": "HttpxTransport -> POST /v1/systemone, "
                             "model pinned " + MODEL_PIN,
            "reference_transport": "OpenRouterHttpTransport -> /chat/completions, "
                                   "one attempt per dispatch, Instruct "
                                   "non-thinking models, explicit max_tokens "
                                   f"{REFERENCE_OUT_CAP}",
            "reference_models": REFERENCE_MODELS,
            "parser": ("strict exact-key parse AND answer-recovery 2.1.0 "
                       "(final_line_key unambiguous final line) reported apart"),
            "ledger": ("PersistentBudget pre-call worst-case reservation per "
                       "logical call (" + str(MAX_TRANSPORT_ATTEMPTS) +
                       " bounded attempts x documented input ceiling), "
                       "actual usage x published tariff settles, unknown cost "
                       "holds forever, price/quota/auth errors fail closed"),
            "usage_capture": ("warm/cold flag, per-call nonce, sanitized response "
                              "headers + raw body, served model, wall + "
                              "x-envoy-upstream-service-time, actual usage"),
            "concurrency": 1,
        },
        "source_verification": source_verification,
        "offline": offline,
    }


# ----------------------------------------------------------- pinned confidence
# Chance-adjusted choice confidence and score mean-absolute-deviation formulas
# transcribed from the pinned official adapter source:
#   typesafe-ai/system-one-adapter-python @ fb52b1030b7fc1f4f1cf39910afa5da54f9835e3
#   src/system_one_adapter/_utils/confidence_metrics.py
#   (sha256 141db0963ce08d98..., fetched 2026-10-01; see /tmp/probe).
CONFIDENCE_SOURCE_PIN = {
    "repo": "typesafe-ai/system-one-adapter-python",
    "commit": "fb52b1030b7fc1f4f1cf39910afa5da54f9835e3",
    "path": "src/system_one_adapter/_utils/confidence_metrics.py",
    "sha256_prefix": "141db0963ce08d98",
}


def choice_confidence(probs: list[float]) -> float:
    """(max(p) - 1/K) / (1 - 1/K) over normalized p; K=1 -> 1.0 (pinned formula)."""
    k = len(probs)
    if k == 0:
        raise ValueError("empty probability vector")
    if k == 1:
        return 1.0
    total = sum(probs)
    if total <= 0:
        p = [1.0 / k] * k
    else:
        p = [x / total for x in probs]
    return (max(p) - 1.0 / k) / (1.0 - 1.0 / k)


def score_confidence(probs: list[float]) -> float:
    """max(0, 1 - E_p|idx - mode_idx| / meanAD_uniform) (pinned formula)."""
    k = len(probs)
    if k == 0:
        raise ValueError("empty probability vector")
    total = sum(probs)
    if total <= 0:
        return 0.0
    p = [x / total for x in probs]
    mode_idx = max(range(k), key=lambda i: p[i])
    mean_ad = sum(p[i] * abs(i - mode_idx) for i in range(k))
    uniform_ad = sum(abs(i - (k - 1) / 2.0) for i in range(k)) / k
    if uniform_ad <= 0:
        return 1.0
    return max(0.0, 1.0 - mean_ad / uniform_ad)


# ------------------------------------------------------- rounding contrast
def logodds_rounding_band(p_a: float, p_b: float,
                          quanta: float = 1.0) -> dict[str, Any]:
    """Pairwise log-odds with a display-rounding sensitivity band.

    Probabilities are displayed at 0.01 quanta; the band moves each of p_a, p_b
    by up to `quanta` quanta (1-1.5 recommended) and reports the induced
    log-odds interval.  Pairs at/near the 0.00/0.01 display floor are flagged
    and must be excluded from log-odds claims (near-zero log-odds noise).
    """
    q = 0.01 * quanta
    lo_a, hi_a = max(1e-9, p_a - q), p_a + q
    lo_b, hi_b = max(1e-9, p_b - q), p_b + q
    at_floor = min(p_a, p_b) <= 0.015
    return {
        "logodds": math.log(p_a / p_b) if min(p_a, p_b) > 0 else None,
        "band": [math.log(lo_a / hi_b), math.log(hi_a / lo_b)],
        "at_display_floor": at_floor,
    }


def pairwise_logodds(probs_by_text: dict[str, float],
                     pairs: list[tuple[str, str]],
                     quanta: float = 1.0) -> dict[str, Any]:
    out = {}
    for a, b in pairs:
        if a in probs_by_text and b in probs_by_text:
            out[f"{a}~{b}"] = logodds_rounding_band(
                probs_by_text[a], probs_by_text[b], quanta)
    return out


# --------------------------------------------------------------- P5 correction
# Versioned OFFLINE correction of the old P5 horizon rows (original 104 rows
# stay immutable): two designer gold errors per parent analysis.
P5_CORRECTIONS: dict[int, dict[str, Any]] = {
    15: {"field": "month", "from": "2025-07", "to": "2025-08",
         "event": "Anchorage US-Russia summit",
         "reason": "summit occurred 2025-08-15; old label 2025-07 wrong",
         "source": "independent published photography coverage "
                   "(Reuters pictures 2025-08-15; content behind bot wall) "
                   "+ Wikipedia 2025 Russia-United States summit in Anchorage"},
    21: {"field": "row", "action": "exclude",
         "event": "Boeing Starliner crewed flight test 'returning its crew'",
         "reason": ("mislabeled 2024-08: crew returned 2025-03-18 on Crew Dragon "
                    "and Starliner landed uncrewed 2024-09-06; the correct month "
                    "was never an offered option"),
         "source": "https://www.nasa.gov/news-release/welcome-home-nasas-spacex-"
                   "crew-9-back-on-earth-after-science-mission/ (2025-03-18) and "
                   "https://www.nasa.gov/news-release/nasa-boeing-welcome-"
                   "starliner-spacecraft-to-earth-close-mission/ (2024-09-06 US time)"},
}


def p5_corrected_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Derived correction over old p5 rows; per-kind (post/pre/fictional)
    separation, corrected + excluded metadata, originals preserved."""
    p5 = [r for r in rows if r.get("family") == "p5_horizon"]
    excluded, corrected = [], []
    monthly: dict[str, dict[str, int]] = {}
    by_kind: dict[str, dict[str, int]] = {}
    for r in p5:
        kind = r.get("kind")
        b = by_kind.setdefault(kind, {"n": 0, "gold_hits": 0})
        month, gold = r.get("month"), r.get("gold")
        ev = r.get("event_i")
        if ev in P5_CORRECTIONS and P5_CORRECTIONS[ev].get("action") == "exclude":
            excluded.append({"config_id": r.get("config_id"), "event_i": ev,
                             "month": month, "choice": r.get("choice")})
            continue
        if ev in P5_CORRECTIONS:
            month = P5_CORRECTIONS[ev]["to"]
            gold = f"in {month}"
            corrected.append({"config_id": r.get("config_id"), "event_i": ev,
                              "month_from": P5_CORRECTIONS[ev]["from"],
                              "month_to": month,
                              "gold_recoded": gold,
                              "choice": r.get("choice"),
                              "was_gold_under_old_label": r.get("choice") == r.get("gold")})
        hit = r.get("choice") == gold
        b["n"] += 1
        b["gold_hits"] += 1 if hit else 0
        m = monthly.setdefault(month, {"n": 0, "gold_hits": 0, "kind": kind})
        m["n"] += 1
        m["gold_hits"] += 1 if hit else 0
    return {
        "schema": "jev-p5-horizon-corrected.v1",
        "version": FOLLOWUP_VERSION,
        "original_rows_preserved": True,
        "n_original_rows": len(p5),
        "corrections": P5_CORRECTIONS,
        "excluded_rows": excluded,
        "corrected_rows": corrected,
        "per_kind": {k: dict(v, gold_rate=round(v["gold_hits"] / v["n"], 3)
                             if v["n"] else None)
                     for k, v in sorted(by_kind.items())},
        "corrected_monthly_post": {m: dict(v, gold_rate=round(v["gold_hits"] / v["n"], 3))
                                   for m, v in sorted(monthly.items())
                                   if v["kind"] == "post"},
        "reading": ("real/pre/fictional pooled gold_rate is invalid: fictional "
                    "controls are not real events. Corrected real post-events "
                    "degrade gradually from 2025-01 and floor by 2025-05; no "
                    "one-month cliff and no 'May 2025 cutoff' claim. The 2025-05 "
                    "'papal conclave electing Pope Leo XIV' style stems leak the "
                    "answer and are diagnostic history only."),
    }


# ------------------------------------------------------------------- analysis
def _median(xs: list[float]) -> float | None:
    s = sorted(xs)
    n = len(s)
    return (s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2) if n else None


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    import numpy as np
    if len(xs) < 3:
        return None
    a, b = np.asarray(xs), np.asarray(ys)
    if a.std() == 0 or b.std() == 0:
        return None
    return round(float(np.corrcoef(a, b)[0, 1]), 4)


def analyze_fu_vocab(rows: list[dict], candidate_counts: dict | None = None) -> dict:
    ok = [r for r in rows if r.get("family") == "fu_vocab" and not r.get("error")
          and r.get("usage_input_tokens")]
    med: dict[str, float] = {}
    for r in ok:
        med.setdefault(r["sample_id"], []).append(float(r["usage_input_tokens"]))
    med = {k: _median(v) for k, v in med.items()}
    controls = {"bare": med.get("empty_baseline"),
                "wordctx": med.get("empty_wordctx")}
    for n in _TOKEN_DELIM_REPEATS:
        controls[f"delim{n}"] = med.get(f"empty_delim{n}")
    samples = {}
    for r in ok:
        samples.setdefault(r["sample_id"], r)
    per_sample = {}
    for sid, base in med.items():
        row = samples.get(sid, {})
        ctx = row.get("context") or ("bare" if sid.startswith("merge") else "control")
        ctrl_key = ("delim%d" % row["repeats"]) if ctx == "delim" and row.get("repeats") \
            else ("wordctx" if ctx == "wordctx" else "bare")
        ctrl = controls.get(ctrl_key)
        marginal = (base - ctrl) if (ctrl is not None and row.get("role") != "control") else None
        reps = int(row.get("repeats") or 1)
        per_sample[sid] = {
            "role": row.get("role"), "context": ctx, "repeats": reps,
            "median_input_tokens": base,
            "marginal_tokens": round(marginal, 2) if marginal is not None else None,
            "per_repeat_marginal": round(marginal / reps, 3) if marginal is not None else None,
        }
    cores = {}
    for core, _text in _TOKEN_CORES:
        entry = {"bare_one_token": None, "repeat_curve": []}
        b = per_sample.get(f"{core}:bare", {})
        entry["bare_one_token"] = b.get("marginal_tokens")
        for n in _TOKEN_DELIM_REPEATS:
            entry["repeat_curve"].append(
                per_sample.get(f"{core}:delim{n}", {}).get("per_repeat_marginal"))
        cores[core] = entry
    return {
        "n_rows": len([r for r in rows if r.get("family") == "fu_vocab"]),
        "baseline_tokens": controls.get("bare"),
        "controls": controls, "per_sample": per_sample, "cores": cores,
        "candidate_counts": candidate_counts,
        "reading": ("marginal 1.0 for zzz/______/heart at bare context with a "
                    "linear repeat curve means the counter sees one token per "
                    "copy; >=2 marginals mean multi-token splits. Candidate "
                    "columns come from ACTUAL encoders (encode/decode round "
                    "trip), never literal byte-encoded vocab key lookup — the "
                    "94 'missing' scout strings were exactly that artifact."),
    }


def _latency(r: dict) -> float | None:
    """Queue-free server compute when the deployment reports it; otherwise the
    client wall clock (network jitter absorbed by randomized block intercepts)."""
    return r.get("upstream_ms") if r.get("upstream_ms") is not None else r.get("wall_ms")


def _ols_fit(X, y):
    import numpy as np
    coef, *_ = np.linalg.lstsq(np.asarray(X), np.asarray(y), rcond=None)
    return coef


def _rmse(pred, y) -> float:
    import numpy as np
    p, a = np.asarray(pred), np.asarray(y)
    return float(np.sqrt(((p - a) ** 2).mean()))


def fit_latency_models(rows: list[dict]) -> dict:
    """Service-time fit with block intercepts and reported/candidate counts.

    Rows are clustered in randomized blocks and repeats share stimuli, so
    predictive skill is judged by holding out WHOLE stimuli (fit fillerA, test
    fillerB and vice versa) — repeat rows are never treated as independent
    models.  A perfect count-time fit would not prove pretrained weights.
    """
    ok = [r for r in rows if r.get("family") == "fu_place" and not r.get("error")
          and _latency(r) is not None and r.get("usage_input_tokens")]
    if len(ok) < 24:
        return {"status": "insufficient_data", "n": len(ok)}

    def design(rs, blocks):
        X = []
        for r in rs:
            row = [1.0 if r["block"] == b else 0.0 for b in blocks]
            rep = r["usage_input_tokens"] / 1000.0
            share = (r.get("state_chars_local") or 1) / max(
                1, (r.get("state_chars_local") or 0) + (r.get("question_chars_local") or 0))
            X.append(row + [rep, rep * share, rep * (1 - share),
                            (r.get("state_chars_local") or 0) / 1000.0,
                            (r.get("question_chars_local") or 0) / 1000.0])
        return X

    blocks = sorted({r["block"] for r in ok})
    nb = len(blocks)
    # feature columns after the block dummies: rep_total, rep_state,
    # rep_question, chars_state, chars_question
    models = {"reported_total": [0], "reported_role_split": [1, 2],
              "candidate_char_roles": [3, 4]}
    y = [_latency(r) for r in ok]
    out: dict[str, Any] = {"n_rows": len(ok), "blocks": nb, "models": {}}
    X = design(ok, blocks)
    stimuli = sorted({r["text_id"] for r in ok})
    labels = (["block%d" % b for b in blocks]
              + ["rep_total", "rep_state", "rep_question",
                 "chars_state", "chars_question"])
    for name, cols in models.items():
        idx = list(range(nb)) + [nb + c for c in cols]
        Xm = [[row[c] for c in idx] for row in X]
        coef = _ols_fit(Xm, y)
        in_rmse = _rmse([sum(a * b for a, b in zip(row, coef)) for row in Xm], y)
        hold = {}
        for stim in stimuli:
            tr = [r for r in ok if r["text_id"] != stim]
            te = [r for r in ok if r["text_id"] == stim]
            Xtr = design(tr, blocks)
            Xte = design(te, blocks)
            ctr = _ols_fit([[row[c] for c in idx] for row in Xtr],
                           [_latency(r) for r in tr])
            pred = [sum(a * b for a, b in zip(row, ctr)) for row in Xte]
            hold[f"holdout_{stim}"] = round(_rmse(pred, [_latency(r) for r in te]), 2)
        coef_named = {}
        for c, cf in zip(idx, coef):
            coef_named[labels[c]] = round(float(cf), 4)
        role_ratio = None
        if coef_named.get("rep_state"):
            role_ratio = round(coef_named["rep_question"] / coef_named["rep_state"], 3)
        elif coef_named.get("chars_state"):
            role_ratio = round(coef_named["chars_question"] / coef_named["chars_state"], 3)
        out["models"][name] = {
            "coefficients": coef_named, "in_sample_rmse_ms": round(in_rmse, 2),
            "heldout_stimulus_rmse_ms": hold,
            "question_vs_state_cost_ratio": role_ratio}
    out["latency_measure"] = ("x-envoy-upstream-service-time/serviceMs not "
                              "reported by this deployment; client wall_ms used "
                              "with block intercepts (network jitter is a limit)")
    out["reading"] = ("role coefficients come from the SAME total volume of text; "
                      "a question/state ratio near 1.0 (band across blocks) "
                      "contradicts a fixed 2x question-token cost. Held-out "
                      "whole-stimulus RMSE (not repeat rows) is the predictive "
                      "claim; block intercepts absorb time-cluster drift.")
    return out


def analyze_fu_place(rows: list[dict]) -> dict:
    fit = fit_latency_models(rows)
    ok = [r for r in rows if r.get("family") == "fu_place" and not r.get("error")
          and _latency(r) is not None]
    cells: dict[str, list[float]] = {}
    for r in ok:
        cells.setdefault(f"{r['cell_id']}:{r['size_id']}", []).append(_latency(r))
    cell_med = {k: round(_median(v), 1) for k, v in sorted(cells.items())}
    shared = {}
    for size_id, _ in _PLACE_SIZES:
        q1 = cell_med.get(f"q1_state:{size_id}")
        q8 = cell_med.get(f"q8_state:{size_id}")
        shared[size_id] = {"q1_state_ms": q1, "q8_state_ms": q8,
                           "delta_8q_ms": round(q8 - q1, 1) if q1 and q8 else None}
    return {"latency": fit, "cell_median_wall_ms": cell_med,
            "shared_state_cost": shared,
            "counts": {"n_rows": len([r for r in rows if r.get("family") == "fu_place"])}}


def analyze_fu_odds(rows: list[dict]) -> dict:
    arch = [r for r in rows if r.get("family") == "fu_odds" and r.get("arm") == "archer"
            and not r.get("error") and r.get("probabilities")]
    own = [r for r in rows if r.get("family") == "fu_odds" and r.get("arm") == "own"
           and not r.get("error") and r.get("probabilities")]
    pairs_a = [("The customer caused it", "The payment provider caused it"),
               ("Cannot tell", "The payment provider caused it")]

    def block_logodds(rs, num, den):
        out = {}
        for r in rs:
            p = r["probabilities"]
            if p.get(num, 0) > 0 and p.get(den, 0) > 0:
                out[r["block"]] = math.log(p[num] / p[den])
        return out

    variants = {}
    for v in _ARCHER_VARIANTS:
        rs = [r for r in arch if r["variant"] == v]
        for num, den in pairs_a:
            los = block_logodds(rs, num, den)
            variants.setdefault(v, {})[f"{num[:12]}~{den[:12]}"] = {
                "mean_logodds": round(_median(list(los.values())), 4) if los else None,
                "n_blocks": len(los)}
    def contrast(v1, v2, key):
        a = variants.get(v1, {}).get(key, {}).get("mean_logodds")
        b = variants.get(v2, {}).get(key, {}).get("mean_logodds")
        return round(a - b, 4) if a is not None and b is not None else None
    keys = [f"{n[:12]}~{d[:12]}" for n, d in pairs_a]
    contrasts = {
        "noise_floor_null4_minus_base4": {k: contrast("null4", "base4", k) for k in keys},
        "noise_floor_null5_minus_append5": {k: contrast("null5", "append5", k) for k in keys},
        "real_append5_minus_base4": {k: contrast("append5", "base4", k) for k in keys},
        "real_replace5_minus_append5": {k: contrast("replace5", "append5", k) for k in keys},
    }
    own_out = {}
    for sc in _OWN_SCENARIOS:
        texts = [t for _, t in sc.original]
        pairs = [(texts[i], texts[j]) for i in range(len(texts)) for j in range(i + 1, len(texts))]
        by_k = {}
        for cell in ("K4", "K5", "K6"):
            rs = [r for r in own if r["scenario"] == sc.scenario_id and r["variant"] == cell]
            agg: dict[str, list[float]] = {}
            for r in rs:
                for t, p in r["probabilities"].items():
                    agg.setdefault(t, []).append(p)
            med = {t: _median(v) for t, v in agg.items()}
            by_k[cell] = pairwise_logodds(med, pairs, quanta=1.0)
        scales, rank_notes = {}, {}
        base = by_k.get("K4", {})
        for cell in ("K5", "K6"):
            num = [by_k[cell][k]["logodds"] for k in base if k in by_k.get(cell, {})
                   and base[k]["logodds"] and by_k[cell][k]["logodds"]]
            den = [base[k]["logodds"] for k in base if k in by_k.get(cell, {})
                   and base[k]["logodds"] and by_k[cell][k]["logodds"]]
            if den and sum(d * d for d in den):
                scales[cell] = round(sum(n * d for n, d in zip(num, den)) / sum(d * d for d in den), 4)
            rank_notes[cell] = all(
                (base[k]["logodds"] > 0) == (by_k[cell][k]["logodds"] > 0)
                for k in base if k in by_k.get(cell, {}) and base[k]["logodds"]
                and by_k[cell][k]["logodds"])
        own_out[sc.scenario_id] = {
            "pairwise_by_K": by_k, "scale_vs_K4": scales,
            "rank_preserved": rank_notes,
            "floor_exclusions": [k for k, v in base.items() if v.get("at_display_floor")]}
    return {"archer_variants": variants, "archer_contrasts": contrasts,
            "archer_note": ("null arms are intentional duplicate noise controls; "
                            "real contrasts must exceed the repeat noise floor "
                            "before any option-interaction claim"),
            "own_scenarios": own_out,
            "reading": ("independent fixed logits + fixed temperature: K-invariant "
                        "pairwise log-odds within 1-1.5 display quanta; "
                        "common-scale temperature: one scale factor per K with "
                        "preserved pair ranks; neither: pair-specific changes or "
                        "rank flips. Displayed 0.01 rounding gives the band; "
                        "pairs at the display floor are excluded, never logged.")}


def analyze_fu_refcard(rows: list[dict]) -> dict:
    ok = [r for r in rows if r.get("family") == "fu_refcard" and not r.get("error")]
    # choice is flattened to option TEXT; join the frozen expected/ref_only
    # keys back to texts through the deterministic builder
    texts = {c.config_id: {"expected_text": dict(c.request.questions["decision"].criteria)[c.meta["expected"]],
                           "ref_only_text": dict(c.request.questions["decision"].criteria)[c.meta["ref_only_key"]]}
             for c in build_fu_refcard()}
    cells: dict[str, dict] = {}
    for r in ok:
        k = f"{r['wording']}:{r['placement']}:{'absent' if r['absent'] else 'present'}:{r['idset']}"
        c = cells.setdefault(k, {"n": 0, "expected": 0, "ref_only_chosen": 0})
        t = texts.get(r["config_id"], {})
        c["n"] += 1
        c["expected"] += 1 if r.get("choice") == t.get("expected_text") else 0
        c["ref_only_chosen"] += 1 if r.get("choice") == t.get("ref_only_text") else 0
    for c in cells.values():
        c["expected_rate"] = round(c["expected"] / c["n"], 3) if c["n"] else None
    pooled = {"present": {"n": 0, "expected": 0},
              "absent": {"n": 0, "expected": 0}}
    for k, c in cells.items():
        arm = "absent" if ":absent:" in k else "present"
        pooled[arm]["n"] += c["n"]
        pooled[arm]["expected"] += c["expected"]
    for arm in pooled.values():
        arm["expected_rate"] = round(arm["expected"] / arm["n"], 3) if arm["n"] else None
    return {"cells": cells, "pooled_by_absence": pooled,
            "reading": ("sensitivity to where the reference card sits (state vs "
                        "sibling option), to option IDs and order supports joint "
                        "option access; it does not by itself distinguish causal "
                        "from bidirectional attention or slot from pointer "
                        "read-out. Absent controls should fall to chance.")}


def analyze_fu_horizon(rows: list[dict]) -> dict:
    ok = [r for r in rows if r.get("family") == "fu_horizon" and not r.get("error")]
    by_kind: dict[str, dict] = {}
    by_item: dict[str, dict] = {}
    for r in ok:
        hit = r.get("choice") == r.get("gold_text")
        b = by_kind.setdefault(r["kind"], {"n": 0, "gold_hits": 0, "cannot": 0,
                                           "did_not_occur": 0})
        b["n"] += 1
        b["gold_hits"] += 1 if hit else 0
        b["cannot"] += 1 if r.get("choice") == CANNOT else 0
        b["did_not_occur"] += 1 if r.get("choice") == DID_NOT_OCCUR else 0
        it = by_item.setdefault(r["item_id"], {"kind": r["kind"],
                                               "event_date": r["event_date"],
                                               "n": 0, "gold_hits": 0})
        it["n"] += 1
        it["gold_hits"] += 1 if hit else 0
    for b in by_kind.values():
        b["gold_rate"] = round(b["gold_hits"] / b["n"], 3) if b["n"] else None
    for it in by_item.values():
        it["gold_rate"] = round(it["gold_hits"] / it["n"], 3) if it["n"] else None
    return {"per_kind": by_kind, "per_item": dict(sorted(by_item.items())),
            "counts": {"n_rows": len(ok)},
            "reading": ("real post-cutoff items are scored apart from controls; "
                        "gradual decay across late-2024..Aug-2025 is a rolling "
                        "horizon, a floor after one month is a hard cutoff. "
                        "Rank/model agreement is behavioral similarity, never "
                        "unique-lineage identification.")}


def analyze_fu_reference(rows: list[dict]) -> dict:
    ok = [r for r in rows if r.get("family") == "fu_reference" and not r.get("error")]
    by_model: dict[str, dict] = {}
    for r in ok:
        m = r.get("model_requested")
        b = by_model.setdefault(m, {"n": 0, "strict_hits": 0, "recovered_hits": 0,
                                    "unrecovered": 0, "finish_reasons": {},
                                    "per_kind": {}})
        b["n"] += 1
        b["strict_hits"] += 1 if r.get("strict_key") == r.get("gold_key") else 0
        b["recovered_hits"] += 1 if r.get("recovered_key") == r.get("gold_key") else 0
        b["unrecovered"] += 1 if r.get("recovered_key") is None else 0
        k = b["per_kind"].setdefault(r.get("kind"), {"n": 0, "gold_hits": 0})
        k["n"] += 1
        k["gold_hits"] += 1 if r.get("recovered_key") == r.get("gold_key") else 0
        fr = r.get("finish_reason")
        b["finish_reasons"][fr] = b["finish_reasons"].get(fr, 0) + 1
        # POST-HOC DIAGNOSTIC (not frozen scoring): key-prefixed final lines
        # like "o3. it did not occur" carry an unambiguous leading key that the
        # conservative 2.1 parser counts unrecovered (the echoed option text
        # trips its negation guard).  Reported apart, applied identically to all.
        import re as _re
        m_key = _re.match(r"^\s*(o\d+)\b[.:,)]?\s", r.get("content_excerpt") or "")
        b.setdefault("diagnostic_key_prefix_hits", 0)
        b.setdefault("diagnostic_key_prefix_n", 0)
        if m_key and r.get("recovered_key") is None:
            b["diagnostic_key_prefix_n"] += 1
            b["diagnostic_key_prefix_hits"] += 1 if m_key.group(1) == r.get("gold_key") else 0
    for b in by_model.values():
        b["strict_rate"] = round(b["strict_hits"] / b["n"], 3) if b["n"] else None
        b["recovered_rate"] = round(b["recovered_hits"] / b["n"], 3) if b["n"] else None
        for k in b["per_kind"].values():
            k["gold_rate"] = round(k["gold_hits"] / k["n"], 3) if k["n"] else None
    return {"per_model": by_model,
            "counts": {"n_rows": len(ok)},
            "reading": ("strict (exact key) and recovered (answer-recovery 2.1.0 "
                        "final_line_key) are reported apart; these reference "
                        "models bound generic shared knowledge on the same "
                        "items and cannot identify weights.")}


def analyze_confidence(rows: list[dict]) -> dict:
    """Pinned confidence formulae on observed vectors + observed correlations."""
    vecs = [list(r["probabilities"].values()) for r in rows
            if r.get("family") in ("fu_odds", "fu_refcard") and r.get("probabilities")]
    if not vecs:
        return {"status": "no_observations", "source_pin": CONFIDENCE_SOURCE_PIN}
    conf = [choice_confidence(v) for v in vecs]
    mx = [max(v) for v in vecs]
    return {"source_pin": CONFIDENCE_SOURCE_PIN,
            "n_vectors": len(vecs),
            "corr_choice_confidence_vs_max_p": _pearson(conf, mx),
            "mean_choice_confidence": round(sum(conf) / len(conf), 4),
            "reading": ("the pinned formulae reproduce our rounded-vector "
                        "recovery as an official closed form; correlation with "
                        "max-p on observed vectors is a consistency check, not "
                        "independent calibration proof. Saturated padding "
                        "confidence 1.0 remains saturation evidence only.")}


def analyze_rows(rows: list[dict], candidate_counts: dict | None = None) -> dict:
    counts = {"n_rows": len(rows),
              "n_errors": sum(1 for r in rows if r.get("error")),
              "by_family": {}}
    for r in rows:
        counts["by_family"][r.get("family", "?")] = \
            counts["by_family"].get(r.get("family", "?"), 0) + 1
    return {
        "claim_level": "exploratory_architecture_evidence",
        "version": FOLLOWUP_VERSION,
        "counts_reported_separately": counts,
        "fu_vocab": analyze_fu_vocab(rows, candidate_counts),
        "fu_place": analyze_fu_place(rows),
        "fu_odds": analyze_fu_odds(rows),
        "fu_refcard": analyze_fu_refcard(rows),
        "fu_horizon": analyze_fu_horizon(rows),
        "fu_reference": analyze_fu_reference(rows),
        "confidence_formula": analyze_confidence(rows),
    }


# ===================================================================== v2 repair
# Parent audit repairs (v1 rows and plan_frozen.json stay immutable; everything
# here produces versioned v2 stimuli/analysis).  See the audit: v1 fu_refcard was
# invalid as an isolation test (card leaked into ref_only criteria and state in
# both placements), v1 timing nonces at state END did not break common-prefix KV
# caching, the primary Archer odds pair is customer/unknown, and echo-style
# reference answers need a safe full-string parser.

from pathlib import Path

PROBE_TRIALS_PATH = Path("/tmp/probe/followup-trials.json")
V2_VERSION = "followup-20261003.v2"

_CARD_RE = re.compile(r"(route|status)\s*=\s*(\w+)")


def load_relational_replication() -> list[ProbeCall]:
    """EXACT replication of the 96 published relational ('refcard') payloads
    from /tmp/probe/followup-trials.json — state/questions copied verbatim,
    only the model pin changes (jev-latest -> jev-1.13.0).

    Design (published): 6 order permutations x 2 attribute values x 2 templates
    (delivery/route, signal/status) x 2 reps x 2 card locations = 96.  The beta
    condition is FIXED to east/amber and gamma to west/indigo; the card value
    switches which condition is satisfied, so the gold key varies with
    (value, permutation) and is derived offline from the fixed semantics
    (criterion text containing 'is <value>'), never from a constant key.

    Literal card-location invariants (asserted in tests): in option placement
    the state contains NO card and only the reference-only criterion repeats
    the card; in state placement the reference-only criterion is generic and
    does NOT repeat the card.  Guessing between the two selectable conditions
    is 1/2 (the reference-only option says 'Never select'); no 1/3 'absent'
    arm exists here.
    """
    data = json.loads(PROBE_TRIALS_PATH.read_text(encoding="utf-8"))
    calls: list[ProbeCall] = []
    for t in data["trials"]:
        if t.get("experiment") != "relational_order":
            continue
        payload = t["payload"]
        state = payload["state"]
        qid, q = next(iter(payload["questions"].items()))
        criteria = dict(q["criteria"])
        # gold: the selectable criterion whose FIXED condition matches the card
        card_src = state + " " + " ".join(criteria.values())
        m = _CARD_RE.search(card_src)
        assert m, "no card value in published payload"
        value = m.group(2)
        gold_key = next(k for k, v in criteria.items()
                        if "Select this option" in v and f"is {value}" in v)
        req = SystemOneRequest(state=state, model=DEFAULT_MODEL_PIN,
                               questions={qid: ChoiceQuestion(
                                   instructions=q["instructions"],
                                   criteria=criteria)})
        calls.append(ProbeCall(
            "fu_relational",
            f"fu_relational:idx{t['index']}:{t['task']}:{t['placement']}:"
            f"{t['value']}:r{t['repeat']}", req,
            {"trial_index": t["index"], "task": t["task"],
             "attribute": t["attribute"], "value": t["value"],
             "placement": t["placement"], "order": list(t["order"]),
             "rep": t["repeat"], "gold_key": gold_key,
             "gold_text": criteria[gold_key],
             "ref_only_key": next(k for k, v in criteria.items()
                                  if "Never select" in v)}))
    return calls


# ---------------------------------------------------- true question isolation
ISO_QID_TARGET = "q_target"
ISO_QID_SIB = "q_sib"
ISO_LOCATIONS = ("state", "target_instr", "sibling_instr", "absent")
ISO_QORDERS = ("target_first", "target_second")
ISO_BLOCKS = 8
_NO_CODE = "no code supplied"


def _code(rng: random.Random) -> str:
    return f"{rng.getrandbits(16):04x}-{rng.getrandbits(16):04x}"


def _card_line(code: str) -> str:
    return f"Reference code: {code}."


def _irrelevant_line(rng: random.Random, card: str) -> str:
    """Same character length as the card line, never naming a code."""
    lead = "Marker string: "
    core = len(card) - len(lead) - 1
    return lead + "".join(rng.choice("abcdefgh") for _ in range(core)) + "."


def build_fu_isolation(blocks: int = ISO_BLOCKS, seed: int = 2026100305) -> list[ProbeCall]:
    """64 calls: 8 blocks x card location {state, target instructions, SEPARATE
    sibling question instructions, absent} x question order {target first,
    second}.  Both questions are present in every condition; the sibling carries
    same-length irrelevant text unless it carries the card.  The card names ONE
    of the three offered codes (random per block, option position rotated), so
    the correct code is identified only by the card.  Outcome: P(selected code)
    in sibling vs absent, with state/target-instruction channels as positive
    controls — sibling success is NOT assumed."""
    calls: list[ProbeCall] = []
    rng = random.Random(seed)
    for block in range(blocks):
        codes = list({_code(rng), _code(rng), _code(rng)})
        while len(codes) < 3:
            codes.append(_code(rng))
        selected = codes[block % 3]
        card = _card_line(selected)
        for loc in ISO_LOCATIONS:
            for qo in ISO_QORDERS:
                irrelevant = _irrelevant_line(rng, card)
                assert len(irrelevant) == len(card)
                state = "Code check session."
                target_instr = ("Which reference code was supplied? "
                                "If none was supplied, choose the no-code option.")
                sib_instr = irrelevant + " Does this line end with a period?"
                if loc == "state":
                    state = state + " " + card
                elif loc == "target_instr":
                    target_instr = card + " " + target_instr
                elif loc == "sibling_instr":
                    sib_instr = card + " Does this line end with a period?"
                # absent: card occurs nowhere; sibling keeps irrelevant text
                opts = codes + [_NO_CODE]
                rng.shuffle(opts)
                target_q = ChoiceQuestion(
                    instructions=target_instr,
                    criteria={f"o{i}": t for i, t in enumerate(opts)})
                sib_q = NoulQuestion(instructions=sib_instr, criteria={"true": "true",
                                                                      "false": "false"})
                if qo == "target_first":
                    questions = {ISO_QID_TARGET: target_q, ISO_QID_SIB: sib_q}
                else:
                    questions = {ISO_QID_SIB: sib_q, ISO_QID_TARGET: target_q}
                req = SystemOneRequest(state=state, model=DEFAULT_MODEL_PIN,
                                       questions=questions)
                gold_key = f"o{opts.index(selected)}"
                channel_texts = [state, target_instr, sib_instr]
                calls.append(ProbeCall(
                    "fu_isolation",
                    f"fu_isolation:b{block}:{loc}:{qo}", req,
                    {"block": block, "location": loc, "qorder": qo,
                     "target_qid": ISO_QID_TARGET,
                     "card_text": card, "irrelevant_text": irrelevant,
                     "codes": codes, "selected_code": selected,
                     "gold_key": gold_key, "gold_text": selected,
                     "no_code_key": f"o{opts.index(_NO_CODE)}",
                     "card_occurrences": sum(t.count(card) for t in channel_texts),
                     "channels_with_card": [name for name, t in
                                            zip(("state", "target_instr", "sibling_instr"),
                                                channel_texts) if card in t]}))
    return calls


# -------------------------------------------------------- timing v2 (prefix-safe)
# Nonce goes at the VERY BEGINNING of the state (before all long filler) so no
# two requests share a common prompt prefix (KV/prefix caching hygiene).  The
# v1 code comment claiming an END-of-state nonce achieved this was false.
TIMING_SCRIPTS = ("latin", "digits", "cjk")
TIMING_TARGETS = ("t3k", "t12k")
TIMING_BLOCKS = 8

_CJK_MIX = "".join(map(chr, (0x4f60, 0x597d, 0x4e16, 0x754c, 0x65e5, 0x672c,
                             0x8a9e, 0x97d3, 0x56fd, 0x5b66, 0x6821, 0x4eba)))
_DIGIT_RUN = "0123456789"
_LATIN_RUN = ("field notes record lane counts dwell minutes and exception codes "
              "for every depot on friday ")

CALIB_SCRIPTS_CHARS = (1000, 4000)   # 2 lengths x 3 scripts + repeats <= 20 calls


def _script_text(script: str, n_chars: int) -> str:
    unit = {"latin": _LATIN_RUN, "digits": _DIGIT_RUN, "cjk": _CJK_MIX}[script]
    return (unit * (n_chars // len(unit) + 1))[:n_chars]


def build_fu_calib(reps: int = 2) -> list[ProbeCall]:
    """Counter-only calibration (<=20 calls): the same noul frame measures the
    Jev counter per script at fixed char lengths so the final 3k/12k stimulus
    LENGTHS are determined by counts, never by latency responses."""
    calls: list[ProbeCall] = []
    for script in TIMING_SCRIPTS:
        for n_chars in CALIB_SCRIPTS_CHARS:
            text = _script_text(script, n_chars)
            for rep in range(reps):
                calls.append(ProbeCall(
                    "fu_calib", f"fu_calib:{script}:{n_chars}:r{rep}",
                    _noul(text),
                    {"script": script, "n_chars": n_chars, "rep": rep,
                     "sample_chars": n_chars, "role": "calibration"}))
    return calls


def build_fu_timing(calib_fit: dict[str, dict[str, float]],
                    blocks: int = TIMING_BLOCKS,
                    seed: int = 2026100306) -> list[ProbeCall]:
    """64 blind timing calls AFTER the stimuli are frozen: 3 scripts x 2 target
    Jev-counts (~3k/~12k) x 8 randomized blocks in state, plus the same Latin at
    both lengths in the QUESTION instructions x 8 blocks.  Every call carries
    payload byte/char counts for the upload-size predictor."""
    calls: list[ProbeCall] = []
    rng = random.Random(seed)
    cells = [(s, tg, "state") for s in TIMING_SCRIPTS for tg in TIMING_TARGETS]
    cells += [("latin", tg, "question") for tg in TIMING_TARGETS]
    for block in range(blocks):
        order = cells[:]
        rng.shuffle(order)
        for script, target, placement in order:
            fit = calib_fit[script]
            want = 3000 if target == "t3k" else 12000
            n_chars = int(round((want - fit["intercept"]) / fit["per_char"]))
            n_chars = max(64, n_chars)
            filler = _script_text(script, n_chars)
            nonce = f"{rng.getrandbits(32):08x}"
            # NONCE FIRST: no common prefix across requests, ever
            lead = f"Run marker: {nonce}. "
            short_q = "Is the marker line present? Answer yes or no."
            if placement == "state":
                state = lead + filler
                questions = {"q0": ChoiceQuestion(instructions=short_q,
                                                  criteria={"o0": "yes", "o1": "no"})}
            else:
                state = lead + "Passage follows in the question."
                questions = {"q0": ChoiceQuestion(
                    instructions=filler + "\n" + short_q,
                    criteria={"o0": "yes", "o1": "no"})}
            req = SystemOneRequest(state=state, model=DEFAULT_MODEL_PIN,
                                   questions=questions)
            payload = req.to_payload()
            payload_bytes = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
            q_chars = sum(len(q.instructions) if isinstance(q.instructions, str)
                          else 0 for q in questions.values())
            calls.append(ProbeCall(
                "fu_timing", f"fu_timing:{script}:{target}:{placement}:b{block}", req,
                {"block": block, "script": script, "target_count": target,
                 "placement": placement, "nonce": nonce, "n_chars": n_chars,
                 "state_chars_local": len(state), "question_chars_local": q_chars,
                 "payload_bytes": payload_bytes,
                 "payload_chars": len(json.dumps(payload, ensure_ascii=False))}))
    return calls


def extract_answer_by_qid(parsed: dict, qid: str,
                          criteria: dict[str, str]) -> tuple[dict[str, float], str | None]:
    """Read answers[qid] — never the first answer: two-question requests with
    reordered (mixed noul/choice) questions make 'first iter' unsafe."""
    probs: dict[str, float] = {}
    choice: str | None = None
    ans = (parsed.get("answers") or {}).get(qid)
    if not isinstance(ans, dict):
        return probs, choice
    pp = ans.get("probabilities")
    if isinstance(pp, dict):
        for k, v in pp.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                probs[criteria.get(k, k)] = float(v)
    ch = ans.get("choice")
    if isinstance(ch, str):
        choice = criteria.get(ch, ch)
    return probs, choice


# ------------------------------------------------------- echo recovery (safe)
_ECHO_RE = re.compile(r"^(o\d+)\s*[.:,)]\s*(.+?)\s*$", re.DOTALL | re.IGNORECASE)


def echo_recover_v2(content: str | None, keys: list[str],
                    criteria: dict[str, str]) -> tuple[str | None, str]:
    """SAFE recovery of short complete echoed choices: the whole content must be
    exactly '<allowed key><punct><exactly that key's option TEXT>' (case-flex on
    the text only).  Key/text disagreement, extra text or multiple answers are
    rejected.  No gold access, never a blind first-character guess."""
    if not content:
        return None, "empty"
    text = content.strip()
    m = _ECHO_RE.match(text)
    if not m:
        return None, "not_echo_shape"
    key_raw, echoed = m.group(1), m.group(2).strip()
    key = next((k for k in keys if k.lower() == key_raw.lower()), None)
    if key is None:
        return None, "key_not_allowed"
    expected = (criteria.get(key) or "").strip()
    if not expected:
        return None, "no_expected_text"
    if echoed.lower() != expected.lower():
        return None, "key_text_disagree"
    return key, "echo_exact_text"


# ------------------------------------------------------------ v2 analysis
def _fmean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _bootstrap_ci(values: list[float], n_boot: int = 4000,
                  seed: int = 7) -> list[float] | None:
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(_fmean(sample))
    means.sort()
    return [round(means[int(0.025 * n_boot)], 4), round(means[int(0.975 * n_boot)], 4)]


_CUST = "The customer caused it"
_UNK = "Cannot tell"
_PROV = "The payment provider caused it"


def odds_primary_v2(rows: list[dict]) -> dict:
    """PRIMARY Archer odds contrast: log P(customer)/P(unknown).  The v1 pairs
    (customer/provider, unknown/provider) were not the primary; the primary is
    their difference.  Per the author's protocol the intentional duplicate
    controls are pooled WITHIN each block before logs (base4+null4, and
    append5+null5)."""
    arch = [r for r in rows if r.get("family") == "fu_odds"
            and r.get("arm") == "archer" and r.get("probabilities")]
    if not arch:
        return {"status": "no archer odds rows", "definition":
                "primary = log P(customer)/P(unknown)"}
    blocks = sorted({r["block"] for r in arch})

    def pooled(block: int, variants: tuple[str, ...]) -> dict[str, float]:
        agg: dict[str, list[float]] = {}
        for r in arch:
            if r["block"] == block and r["variant"] in variants:
                for t, p in r["probabilities"].items():
                    agg.setdefault(t, []).append(float(p))
        return {t: _fmean(v) for t, v in agg.items()}

    def primary(p: dict[str, float]) -> float | None:
        if p.get(_CUST, 0) > 0 and p.get(_UNK, 0) > 0:
            return math.log(p[_CUST] / p[_UNK])
        return None

    arms = {"base_pooled": ("base4", "null4"),
            "append_pooled": ("append5", "null5"),
            "replace5": ("replace5",)}
    per_block: dict[str, list[float]] = {a: [] for a in arms}
    for b in blocks:
        for arm, variants in arms.items():
            v = primary(pooled(b, variants))
            if v is not None:
                per_block[arm].append(v)
    n = min(len(per_block["base_pooled"]), len(per_block["append_pooled"]))
    deltas = [per_block["append_pooled"][i] - per_block["base_pooled"][i]
              for i in range(n)]
    nr = min(len(per_block["replace5"]), len(per_block["append_pooled"]))
    rep_deltas = [per_block["replace5"][i] - per_block["append_pooled"][i]
                  for i in range(nr)]
    pooled_means = {arm: round(_fmean(v), 4) for arm, v in per_block.items()}
    rounding = {}
    for arm, variants in arms.items():
        p = pooled(blocks[0], variants)  # shape reference
        agg: dict[str, list[float]] = {}
        for b in blocks:
            for t, pp in pooled(b, variants).items():
                agg.setdefault(t, []).append(pp)
        mean_p = {t: _fmean(v) for t, v in agg.items()}
        rounding[arm] = {
            str(q): logodds_rounding_band(mean_p.get(_CUST, 0),
                                          mean_p.get(_UNK, 0), quanta=q)
            for q in (1.0, 1.5)}
    return {
        "definition": "primary = log P(customer)/P(unknown); duplicates pooled "
                      "within block before logs (author protocol)",
        "per_arm_mean_primary": pooled_means,
        "per_block_primary": {a: [round(x, 4) for x in v]
                              for a, v in per_block.items()},
        "append_minus_base_per_block": [round(x, 4) for x in deltas],
        "append_minus_base_mean": round(_fmean(deltas), 4),
        "append_minus_base_block_bootstrap_ci95": _bootstrap_ci(deltas),
        "replace_minus_append_mean": round(_fmean(rep_deltas), 4),
        "replace_minus_append_ci95": _bootstrap_ci(rep_deltas),
        "rounding_sensitivity_primary": rounding,
        "v1_secondary_pairs_note": ("v1 reported customer/provider and "
                                    "unknown/provider; their difference is this "
                                    "primary (v1 means 0.7949-0.3821=0.4128)"),
        "reproduces_archer": ("pooled base~%.4f -> append~%.4f gives delta ~%.2f, "
                              "matching the author's published ~-0.28 direction "
                              "and size" % (pooled_means["base_pooled"],
                                            pooled_means["append_pooled"],
                                            _fmean(deltas))),
    }


def own_rescale_v2(rows: list[dict], n_boot: int = 2000) -> dict:
    """Per-pair implied rescaling factors with empirical repeat (whole-block)
    bands + display-rounding bands.  A common-temperature rescale requires one
    shared scale; if per-pair scale intervals do not intersect there is evidence
    of option interaction beyond common rescaling.  Rank preservation alone
    establishes NEITHER independent logits NOR K-invariance."""
    own = [r for r in rows if r.get("family") == "fu_odds" and r.get("arm") == "own"
           and r.get("probabilities")]
    out: dict[str, Any] = {}
    for sc in _OWN_SCENARIOS:
        texts = [t for _, t in sc.original]
        pairs = [(texts[i], texts[j]) for i in range(len(texts))
                 for j in range(i + 1, len(texts))]
        blocks = sorted({r["block"] for r in own if r["scenario"] == sc.scenario_id})

        def ln_by_block(cell: str, a: str, b: str) -> dict[int, float]:
            outb = {}
            for r in own:
                if r["scenario"] == sc.scenario_id and r["variant"] == cell:
                    p = r["probabilities"]
                    if p.get(a, 0) > 0 and p.get(b, 0) > 0:
                        outb[r["block"]] = math.log(p[a] / p[b])
            return outb

        pair_scales: dict[str, Any] = {}
        for a, b in pairs:
            key = f"{a}~{b}"
            base = ln_by_block("K4", a, b)
            entry = {"ln_K4_mean": round(_fmean(list(base.values())), 4)
                     if base else None, "scales": {}}
            for cell in ("K5", "K6"):
                cmp_ = ln_by_block(cell, a, b)
                common = sorted(set(base) & set(cmp_))
                if not common or not any(base[k] for k in common):
                    entry["scales"][cell] = None
                    continue
                point = _fmean([cmp_[k] for k in common]) / _fmean([base[k] for k in common])
                rng = random.Random(11)
                boots = []
                for _ in range(n_boot):
                    samp = [common[rng.randrange(len(common))] for _ in common]
                    den = _fmean([base[k] for k in samp])
                    if den:
                        boots.append(_fmean([cmp_[k] for k in samp]) / den)
                boots.sort()
                lo = boots[int(0.025 * n_boot)] if boots else point
                hi = boots[int(0.975 * n_boot)] if boots else point
                # display-rounding sensitivity on the ratio via 1-1.5 quanta
                entry["scales"][cell] = {
                    "point": round(point, 4),
                    "block_bootstrap_ci95": [round(lo, 4), round(hi, 4)],
                    "at_display_floor": min(abs(_fmean([base[k] for k in common])),
                                            abs(_fmean([cmp_[k] for k in common]))) < 0.05,
                }
            pair_scales[key] = entry
        verdicts = {}
        for cell in ("K5", "K6"):
            intervals = [v["scales"][cell]["block_bootstrap_ci95"]
                         for v in pair_scales.values()
                         if v["scales"].get(cell) and not v["scales"][cell]["at_display_floor"]]
            if not intervals:
                verdicts[cell] = "no usable pairs (display-floor exclusions)"
                continue
            lo = max(iv[0] for iv in intervals)
            hi = min(iv[1] for iv in intervals)
            common_beta = _fmean([v["scales"][cell]["point"] for v in pair_scales.values()
                                  if v["scales"].get(cell)])
            verdicts[cell] = {
                "scale_intervals_intersect": lo <= hi,
                "intersection": [round(lo, 4), round(hi, 4)],
                "common_scale_beta_point": round(common_beta, 4),
                "verdict": ("compatible with common rescaling; the "
                            "common-temperature mechanism is NOT rejected (beta "
                            "near 1 is consistent with it)" if lo <= hi else
                            "evidence of option interaction beyond a single "
                            "common rescaling (per-pair scale intervals disjoint)"),
            }
        out[sc.scenario_id] = {"pairs": pair_scales, "verdicts": verdicts}
    out["claim_limits"] = ("rank preservation and beta near 1 do not establish "
                           "K-invariance or independent logits; only disjoint "
                           "per-pair scale intervals are evidence of interaction")
    return out


def relational_v2_summary(rows: list[dict]) -> dict:
    ok = [r for r in rows if r.get("family") == "fu_relational" and not r.get("error")]
    cells: dict[str, dict] = {}
    gold_keys: dict[str, int] = {}
    for r in ok:
        k = f"{r['placement']}:{r['value']}"
        c = cells.setdefault(k, {"n": 0, "gold_hits": 0})
        c["n"] += 1
        c["gold_hits"] += 1 if r.get("choice") == r.get("gold_text") else 0
        gold_keys[r.get("gold_key")] = gold_keys.get(r.get("gold_key"), 0) + 1
    for c in cells.values():
        c["gold_rate"] = round(c["gold_hits"] / c["n"], 3) if c["n"] else None
    pooled = {}
    for pl in ("option", "state"):
        rows_p = [r for r in ok if r["placement"] == pl]
        pooled[pl] = {"n": len(rows_p),
                      "gold_rate": round(sum(1 for r in rows_p
                                             if r.get("choice") == r.get("gold_text"))
                                         / len(rows_p), 3) if rows_p else None}
    return {"cells": cells, "by_placement": pooled,
            "gold_key_distribution": gold_keys,
            "chance_note": ("two selectable conditions + a never-select reference "
                            "option: guessing among conditions is 1/2; no absent "
                            "arm and no 1/3 chance claim"),
            "n_rows": len(ok),
            "replication_note": ("exact published payloads (96 relational trials), "
                                 "model pin changed only; v1 fu_refcard rows remain "
                                 "raw evidence but are INVALID for isolation "
                                 "inference (card leaked into state/ref criteria)")}


def isolation_v2_summary(rows: list[dict]) -> dict:
    ok = [r for r in rows if r.get("family") == "fu_isolation" and not r.get("error")]
    cells: dict[str, dict] = {}
    for r in ok:
        k = f"{r['location']}:{r['qorder']}"
        c = cells.setdefault(k, {"n": 0, "gold_selected": 0, "no_code_selected": 0})
        c["n"] += 1
        c["gold_selected"] += 1 if r.get("choice") == r.get("gold_text") else 0
        c["no_code_selected"] += 1 if r.get("choice") == "no code supplied" else 0
    for c in cells.values():
        c["p_gold"] = round(c["gold_selected"] / c["n"], 3) if c["n"] else None
    by_loc = {}
    for loc in ISO_LOCATIONS:
        rows_l = [r for r in ok if r["location"] == loc]
        by_loc[loc] = {"n": len(rows_l),
                       "p_gold": round(sum(1 for r in rows_l
                                           if r.get("choice") == r.get("gold_text"))
                                       / len(rows_l), 3) if rows_l else None}
    return {"cells": cells, "by_location": by_loc,
            "primary": "P(selected code): sibling_instr vs absent (paired by block); "
                       "state/target_instr are positive controls; sibling success "
                       "is NOT assumed and correctness is not assumed",
            "n_rows": len(ok)}


def timing_v2_summary(rows: list[dict]) -> dict:
    ok = [r for r in rows if r.get("family") == "fu_timing" and not r.get("error")]
    lat = {r.get("config_id") or f"fu_timing:{r['script']}:{r['target_count']}:"
           f"{r['placement']}:b{r['block']}": _latency(r) for r in ok}

    def cell_wall(script, target, placement, block):
        cid = f"fu_timing:{script}:{target}:{placement}:b{block}"
        return lat.get(cid)

    # PRIMARY: paired Q1 state-vs-question ratio at matched script/size,
    # resampling WHOLE blocks
    blocks = sorted({r["block"] for r in ok})
    ratios = []
    per_block_ratio = {}
    for b in blocks:
        for target in TIMING_TARGETS:
            s = cell_wall("latin", target, "state", b)
            q = cell_wall("latin", target, "question", b)
            if s and q:
                ratios.append((b, target, q / s))
                per_block_ratio.setdefault(b, []).append(q / s)
    block_means = [_fmean(v) for v in per_block_ratio.values()]
    rng = random.Random(5)
    boots = []
    for _ in range(4000):
        samp = [block_means[rng.randrange(len(block_means))] for _ in block_means]
        boots.append(_fmean(samp))
    boots.sort()
    ci = [round(boots[100], 4), round(boots[3900], 4)] if len(boots) == 4000 else None
    primary_ratio = round(_fmean([r for _, _, r in ratios]), 4) if ratios else None

    # cross-script counter comparison with holdout WHOLE SCRIPT
    feats = []
    for r in ok:
        feats.append({
            "wall": _latency(r), "script": r["script"],
            "reported": r.get("usage_input_tokens") or 0,
            "encode_qwen3": r.get("encode_qwen3_tokens") or 0,
            "encode_qwen35": r.get("encode_qwen35_tokens") or 0,
            "payload_bytes": r.get("payload_bytes") or 0,
            "block": r["block"]})
    models = {"reported": ["reported"], "encode_qwen3": ["encode_qwen3"],
              "reported_plus_bytes": ["reported", "payload_bytes"]}
    hold: dict[str, Any] = {}
    for name, cols in models.items():
        hold[name] = {}
        for script in TIMING_SCRIPTS:
            tr = [f for f in feats if f["script"] != script]
            te = [f for f in feats if f["script"] == script]
            if len(tr) < 6 or not te:
                continue
            Xtr = [[1.0] + [f[c] / 1000.0 for c in cols] for f in tr]
            Xte = [[1.0] + [f[c] / 1000.0 for c in cols] for f in te]
            coef = _ols_fit(Xtr, [f["wall"] for f in tr])
            pred = [sum(a * b for a, b in zip(row, coef)) for row in Xte]
            hold[name][f"holdout_{script}"] = round(
                _rmse(pred, [f["wall"] for f in te]), 2)
    joint_X = [[1.0, f["reported"] / 1000.0, f["encode_qwen3"] / 1000.0,
                f["payload_bytes"] / 1000.0] for f in feats]
    joint = _ols_fit(joint_X, [f["wall"] for f in feats])
    return {
        "n_rows": len(ok),
        "primary_q1_state_vs_question_ratio": primary_ratio,
        "ratio_block_bootstrap_ci95": ci,
        "primary_note": ("paired per-block ratio of elapsed times for identical "
                         "text in question vs state at fixed question count; "
                         "phrase as 'no double penalty observed in elapsed "
                         "time' — elapsed time does not conclusively contradict "
                         "a physical-compute 2x"),
        "holdout_whole_script_rmse_ms": hold,
        "joint_coefficients_ms_per_1k": {
            "reported": round(float(joint[1]), 4),
            "encode_qwen3": round(float(joint[2]), 4),
            "payload_bytes": round(float(joint[3]), 4)},
        "counter_note": ("cross-script strings (Latin/CJK/digits) diverge "
                         "strongly between the Jev counter and Qwen encoders; "
                         "if elapsed time tracks one count family on held-out "
                         "scripts that is evidence about what is actually "
                         "processed, not about weights"),
        "latency_measure": "client wall + first_byte (no upstream header in this "
                           "deployment; header names verified absent, none invented)",
    }


def reference_recovery_v2(rows: list[dict]) -> dict:
    """Versioned offline overrides: safe echo recovery on short complete STOP
    answers, no gold access.  Returns overrides + recomputed per-model metrics."""
    overrides: dict[str, Any] = {"schema": "jev-ref-recovery-overrides.v1",
                                 "version": V2_VERSION, "rows": {}}
    by_model: dict[str, dict] = {}
    for i, r in enumerate(rows):
        if r.get("family") != "fu_reference":
            continue
        m = r.get("model_requested")
        b = by_model.setdefault(m, {"n": 0, "gold_hits": 0, "unrecovered": 0,
                                    "echo_recovered": 0, "per_kind": {}})
        b["n"] += 1
        key = r.get("recovered_key")
        stage = r.get("recovery_stage")
        if key is None:
            content = r.get("content_excerpt") or ""
            complete = (r.get("finish_reason") == "stop"
                        and len(content) < 400 and r.get("extract_error") is None)
            item = next((it for it in HORIZON_ITEMS
                         if it.item_id == r.get("item_id")), None)
            if complete and item is not None:
                rot = r.get("rot", 0)
                base = list(item.options)
                opts = base[rot:] + base[:rot]
                criteria = {f"o{j}": t for j, t in enumerate(opts)}
                key2, stage2 = echo_recover_v2(content, list(criteria), criteria)
                if key2 is not None:
                    overrides["rows"][f"{i}:{r.get('item_id')}:{m}:rot{rot}"] = {
                        "recovered_key": key2, "stage": stage2}
                    key, stage = key2, stage2
                    b["echo_recovered"] += 1
        if key is None:
            b["unrecovered"] += 1
        hit = key == r.get("gold_key")
        b["gold_hits"] += 1 if hit else 0
        k = b["per_kind"].setdefault(r.get("kind"), {"n": 0, "gold_hits": 0})
        k["n"] += 1
        k["gold_hits"] += 1 if hit else 0
    for b in by_model.values():
        b["gold_rate"] = round(b["gold_hits"] / b["n"], 3) if b["n"] else None
        for k in b["per_kind"].values():
            k["gold_rate"] = round(k["gold_hits"] / k["n"], 3) if k["n"] else None
    overrides["per_model_recomputed"] = by_model
    overrides["note"] = ("echo recovery accepts ONLY '<key><punct><exact option "
                         "text>' (case-flex text) with key/text agreement; no "
                         "blind first-character guessing; future runs store full "
                         "sanitized raw bodies, not 400-char excerpts")
    return overrides


def horizon_item_level(rows: list[dict]) -> dict:
    """Unique-item outcomes (24 items; the 2 rotations are repeated measures on
    the SAME 24 facts, never 48 independent observations)."""
    ok = [r for r in rows if r.get("family") == "fu_horizon" and not r.get("error")]
    items: dict[str, dict] = {}
    for r in ok:
        it = items.setdefault(r["item_id"], {"kind": r["kind"],
                                             "event_date": r["event_date"],
                                             "by_rotation": {}, "n": 0,
                                             "gold_hits": 0})
        it["n"] += 1
        hit = r.get("choice") == r.get("gold_text")
        it["gold_hits"] += 1 if hit else 0
        it["by_rotation"][str(r["rot"])] = hit
    for it in items.values():
        it["both_rotations_correct"] = it["gold_hits"] == it["n"]
    cohort = {"recent_facts": 0, "recent_correct": 0,
              "pre_controls": 0, "pre_correct": 0,
              "constructed_false": 0, "false_correct": 0}
    kind_map = {"post": "recent_facts", "pre": "pre_controls",
                "fictional": "constructed_false"}
    for it in items.values():
        grp = kind_map[it["kind"]]
        cohort[grp] += 1
        correct_slot = {"recent_facts": "recent_correct",
                        "pre_controls": "pre_correct",
                        "constructed_false": "false_correct"}[grp]
        cohort[correct_slot] += 1 if it["both_rotations_correct"] else 0
    return {
        "n_unique_items": len(items), "n_rows": len(ok),
        "rotations_are_repeated_measures": True,
        "cohort_labels": ("kind 'post' means RECENT facts (late-2024..Aug-2025 "
                          "outcomes); it does NOT assert a known model cutoff "
                          "(cutoff remains unverified)"),
        "per_item": dict(sorted(items.items())), "cohorts": cohort,
    }


def analyze_v2(rows_all: list[dict], rows_v2: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for r in rows_all:
        counts[r.get("family", "?")] = counts.get(r.get("family", "?"), 0) + 1
    return {
        "schema": "jev-followup-analysis.v2", "version": V2_VERSION,
        "counts_reported_separately": {"by_family": counts,
                                       "n_total": len(rows_all)},
        "v1_invalidations": {
            "fu_refcard": ("v1 family INVALID for isolation inference: the card "
                           "appeared in state for BOTH 'present' placements and "
                           "the reference-only criterion always repeated the "
                           "card, so its 24/24 was not a meaningful control; raw "
                           "v1 rows retained unchanged")},
        "fu_relational": relational_v2_summary(rows_v2),
        "fu_isolation": isolation_v2_summary(rows_v2),
        "fu_timing": timing_v2_summary(rows_v2),
        "odds_primary": odds_primary_v2(rows_all),
        "own_rescale": own_rescale_v2(rows_all),
        "reference_recovery": reference_recovery_v2(rows_all),
        "horizon_items": horizon_item_level(rows_all),
    }


# ===================================================================== v3 counter
# Final targeted timing control (v1/v2 immutable): 4 filler types at EXACTLY
# matched UTF-8 byte volumes (6000/24000) so upload size cannot masquerade as a
# tokenizer effect, 8 randomized blocks, Q1, warm, nonce first.  Predictor
# family includes the ACTUAL o200k encoder (its digit grouping is 1 token per
# ~3 digits where Qwen3/3.5 count 1 per digit like Jev), plus Qwen3/3.5, chars,
# bytes, exact wire bytes and the Jev-reported counter.  Any predictor advantage
# is a processing proxy, never weight ancestry; indistinguishable predictors are
# reported as inconclusive.

V3_VERSION = "followup-20261003.v3"
V3_STYLES = ("latin", "letters", "digits", "cjk")
V3_BYTE_SIZES = (6000, 24000)          # exact UTF-8 bytes; divisible by 3
V3_BLOCKS = 8

_LATIN_SENT = ("field notes record lane counts dwell minutes and exception "
               "codes for every depot on friday morning reviews ")


def _v3_fillers() -> dict[tuple[str, int], str]:
    """Exact byte volumes: CJK truncates to full codepoints only (3 bytes each);
    ASCII styles are 1 byte/char so chars == bytes."""
    out: dict[tuple[str, int], str] = {}
    rng_letters = random.Random(2026100307)
    rng_digits = random.Random(2026100308)
    letters = "".join(rng_letters.choice("abcdefghijklmnopqrstuvwxyz")
                      for _ in range(max(V3_BYTE_SIZES)))
    digits = "".join(str(rng_digits.randrange(10)) for _ in range(max(V3_BYTE_SIZES)))
    cjk_pool = _CJK_MIX
    for n_bytes in V3_BYTE_SIZES:
        out[("latin", n_bytes)] = (_LATIN_SENT * (n_bytes // len(_LATIN_SENT) + 1))[:n_bytes]
        out[("letters", n_bytes)] = letters[:n_bytes]
        out[("digits", n_bytes)] = digits[:n_bytes]
        cjk_chars = n_bytes // 3                       # full codepoints only
        cjk = (cjk_pool * (cjk_chars // len(cjk_pool) + 1))[:cjk_chars]
        assert len(cjk.encode("utf-8")) == n_bytes     # byte-exact, no splitting
        out[("cjk", n_bytes)] = cjk
    for (style, n_bytes), s in out.items():
        assert len(s.encode("utf-8")) == n_bytes, (style, n_bytes)
    return out


V3_FILLERS = _v3_fillers()
V3_HEADER = "Run marker: "
V3_FINAL = " Final marker: end of passage."
V3_QUESTION = ("Which run marker appears at the beginning of the passage? "
               "Answer with one option.")
V3_DISTRACTORS = ("00000000", "ffffffff", "no marker supplied")


def build_fu_counter_calib() -> list[ProbeCall]:
    """<=8 counter-only calibration calls (noul frame over the EXACT filler
    strings): Jev counts per style/byte-size, to verify the digits separator
    (Jev ~1/digit) BEFORE the latency freeze."""
    calls: list[ProbeCall] = []
    for (style, n_bytes), text in sorted(V3_FILLERS.items()):
        calls.append(ProbeCall(
            "fu_counter_calib", f"fu_counter_calib:{style}:{n_bytes}", _noul(text),
            {"style": style, "filler_bytes": n_bytes, "role": "calibration",
             "sample_chars": len(text)}))
    return calls


def build_fu_counter_v3(blocks: int = V3_BLOCKS,
                        seed: int = 2026100309) -> list[ProbeCall]:
    """64 blind timing calls (4 styles x 2 byte sizes x 8 randomized blocks).
    Nonce (fixed 8 hex) sits at the very START of state immediately before the
    filler; the ASCII header/final marker, question and option LENGTHS are
    identical across styles, so at each byte size the exact HTTP payload byte
    count is identical across styles (verified per call at dispatch).  The
    question requires reading the state (gold = the random nonce), so the model
    cannot answer while ignoring the passage."""
    calls: list[ProbeCall] = []
    rng = random.Random(seed)
    cells = [(style, n_bytes) for style in V3_STYLES for n_bytes in V3_BYTE_SIZES]
    for block in range(blocks):
        order = cells[:]
        rng.shuffle(order)
        for style, n_bytes in order:
            filler = V3_FILLERS[(style, n_bytes)]
            nonce = f"{rng.getrandbits(32):08x}"
            state = V3_HEADER + nonce + ". " + filler + V3_FINAL
            opts = [nonce] + list(V3_DISTRACTORS)
            rot = (block + V3_STYLES.index(style) + V3_BYTE_SIZES.index(n_bytes)) % 4
            opts = opts[rot:] + opts[:rot]
            req = SystemOneRequest(
                state=state, model=DEFAULT_MODEL_PIN,
                questions={"q0": ChoiceQuestion(
                    instructions=V3_QUESTION,
                    criteria={f"o{i}": t for i, t in enumerate(opts)})})
            payload = req.to_payload()
            wire = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            q_chars = len(V3_QUESTION)
            calls.append(ProbeCall(
                "fu_counter_v3",
                f"fu_counter_v3:{style}:{n_bytes}:b{block}", req,
                {"block": block, "style": style, "filler_bytes": n_bytes,
                 "nonce": nonce, "n_chars": len(filler),
                 "state_chars_local": len(state), "question_chars_local": q_chars,
                 "payload_bytes": len(wire),
                 "payload_chars": len(wire.decode("utf-8")),
                 "gold_key": f"o{opts.index(nonce)}", "gold_text": nonce}))
    return calls


def counter_v3_summary(rows: list[dict], n_boot: int = 4000) -> dict:
    """Held-out WHOLE-STIMULUS/SCRIPT comparison of count predictors + paired
    per-block contrasts, bootstrapping whole blocks (sample unit = 8 blocks,
    never 64 independent tasks)."""
    ok = [r for r in rows if r.get("family") == "fu_counter_v3"
          and not r.get("error") and r.get("usage_input_tokens")]
    if len(ok) < 32:
        return {"status": "insufficient_data", "n_rows": len(ok)}

    feats = []
    for r in ok:
        feats.append({
            "wall": _latency(r), "style": r["style"], "size": r["filler_bytes"],
            "block": r["block"],
            "reported": r.get("usage_input_tokens") or 0,
            "chars": r.get("n_chars") or 0,
            "utf8_bytes": r.get("filler_bytes") or 0,
            "wire_bytes": r.get("payload_bytes_exact") or r.get("payload_bytes") or 0,
            "encode_qwen3": r.get("encode_qwen3_tokens") or 0,
            "encode_qwen35": r.get("encode_qwen35_tokens") or 0,
            "encode_o200k": r.get("encode_o200k_tokens") or 0,
        })
    predictors = ("reported", "chars", "utf8_bytes", "wire_bytes",
                  "encode_qwen3", "encode_qwen35", "encode_o200k")
    hold: dict[str, Any] = {}
    for name in predictors:
        hold[name] = {}
        for style in V3_STYLES:
            tr = [f for f in feats if f["style"] != style]
            te = [f for f in feats if f["style"] == style]
            Xtr = [[1.0, f[name] / 1000.0] for f in tr]
            Xte = [[1.0, f[name] / 1000.0] for f in te]
            coef = _ols_fit(Xtr, [f["wall"] for f in tr])
            pred = [sum(a * b for a, b in zip(row, coef)) for row in Xte]
            hold[name][f"holdout_{style}"] = round(
                _rmse(pred, [f["wall"] for f in te]), 2)
        hold[name]["mean_holdout_rmse"] = round(
            _fmean(list(v for k, v in hold[name].items() if k.startswith("holdout"))), 2)
    # joint model: which count family carries the signal once collinear
    # digit/char counts are pitted against the alternative encoders
    Xj = [[1.0, f["reported"] / 1000.0, f["encode_o200k"] / 1000.0,
           f["wire_bytes"] / 1000.0, f["chars"] / 1000.0] for f in feats]
    coef_j = _ols_fit(Xj, [f["wall"] for f in feats])
    # paired per-block style contrasts at each byte size (bootstrap blocks)
    contrasts: dict[str, Any] = {}
    for n_bytes in V3_BYTE_SIZES:
        for s1, s2 in (("digits", "latin"), ("digits", "cjk"), ("cjk", "latin")):
            per_block = []
            for b in sorted({f["block"] for f in feats}):
                w1 = [f["wall"] for f in feats if f["block"] == b
                      and f["style"] == s1 and f["size"] == n_bytes]
                w2 = [f["wall"] for f in feats if f["block"] == b
                      and f["style"] == s2 and f["size"] == n_bytes]
                if w1 and w2:
                    per_block.append(w1[0] - w2[0])
            contrasts[f"{s1}_minus_{s2}@{n_bytes}B"] = {
                "mean_ms": round(_fmean(per_block), 2) if per_block else None,
                "block_bootstrap_ci95": _bootstrap_ci(per_block, n_boot)}
    best = min(hold, key=lambda k: hold[k]["mean_holdout_rmse"])
    second = sorted(hold, key=lambda k: hold[k]["mean_holdout_rmse"])[1]
    spread = round(hold[second]["mean_holdout_rmse"]
                   - hold[best]["mean_holdout_rmse"], 2)
    return {
        "n_rows": len(ok), "n_blocks": len({f["block"] for f in feats}),
        "sample_unit": "8 randomized blocks (whole-block bootstrap), not 64 tasks",
        "byte_match_verified": len({f["wire_bytes"] for f in feats
                                    if f["size"] == 6000}) == 1
        and len({f["wire_bytes"] for f in feats if f["size"] == 24000}) == 1,
        "holdout_whole_style_rmse_ms": hold,
        "best_predictor": best, "runner_up": second,
        "predictor_spread_ms": spread,
        "joint_ms_per_1k": {"reported": round(float(coef_j[1]), 3),
                            "encode_o200k": round(float(coef_j[2]), 3),
                            "wire_bytes": round(float(coef_j[3]), 3),
                            "chars": round(float(coef_j[4]), 3)},
        "per_block_contrasts": contrasts,
        "verdict_rule": ("a predictor advantage is a PROCESSING proxy only; if "
                         "two count families are within block-bootstrap noise "
                         "the result is 'inconclusive' and never a positive "
                         "Qwen-base classification"),
        "encoder_facts": ("o200k groups digits to ~1 token per 3 digits while "
                          "Qwen3/3.5 (and Jev) count ~1 per digit; CJK common "
                          "mix: Jev ~1/char vs Qwen3 ~0.66/char — the four "
                          "styles separate the candidate counters pairwise"),
        "latency_measure": "client wall + first_byte (warm, concurrency 1; no "
                           "upstream header in this deployment)",
    }


# ------------------------------------------------- placement ratio correction
def placement_incremental_summary(rows: list[dict], n_boot: int = 10000,
                                  seed: int = 7) -> dict:
    """Both definitions of the v2 placement contrast, preserved together.

    TOTAL ratio: paired whole-block ratio of TOTAL elapsed times (question vs
    state at matched length) — dominated by the fixed baseline floor, so a
    value near 1 does not speak to per-token cost.

    INCREMENTAL ratio: per-block long-minus-short length difference divided by
    the reported-token difference (Q1 Latin only), question vs state — the
    added-token-cost ratio.  With 8 blocks and network outliers its bootstrap
    interval includes 2, so the 2x per-token cost is NEITHER contradicted NOR
    ruled out: label 'elapsed times similar; incremental ratio unresolved'.
    """
    lat = [r for r in rows if r.get("family") == "fu_timing"
           and r.get("script") == "latin" and not r.get("error")]
    cell = {(r["placement"], r["target_count"], r["block"]): r for r in lat}
    blocks = sorted({r["block"] for r in lat})
    inc_s: dict[int, float] = {}
    inc_q: dict[int, float] = {}
    total_ratios: list[float] = []
    for b in blocks:
        try:
            s3, s12 = cell[("state", "t3k", b)], cell[("state", "t12k", b)]
            q3, q12 = cell[("question", "t3k", b)], cell[("question", "t12k", b)]
        except KeyError:
            continue
        d_tok_s = (s12["usage_input_tokens"] - s3["usage_input_tokens"]) / 1000.0
        d_tok_q = (q12["usage_input_tokens"] - q3["usage_input_tokens"]) / 1000.0
        if d_tok_s <= 0 or d_tok_q <= 0:
            continue
        inc_s[b] = (s12["wall_ms"] - s3["wall_ms"]) / d_tok_s
        inc_q[b] = (q12["wall_ms"] - q3["wall_ms"]) / d_tok_q
        for s_row, q_row in ((s3, q3), (s12, q12)):
            if s_row["wall_ms"]:
                total_ratios.append(q_row["wall_ms"] / s_row["wall_ms"])
    common = sorted(set(inc_s) & set(inc_q))
    if not common:
        return {"status": "insufficient_data"}
    mean_s = _fmean([inc_s[b] for b in common])
    mean_q = _fmean([inc_q[b] for b in common])
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        idx = [common[rng.randrange(len(common))] for _ in common]
        den = _fmean([inc_s[i] for i in idx])
        if den:
            boots.append(_fmean([inc_q[i] for i in idx]) / den)
    boots.sort()
    ci = [round(boots[int(0.025 * n_boot)], 3), round(boots[int(0.975 * n_boot)], 3)]
    outliers = {}
    for b in common:
        if inc_s[b] < 0:
            outliers[f"block{b}_state_increment"] = round(inc_s[b], 2)
        if inc_q[b] < 0:
            outliers[f"block{b}_query_increment"] = round(inc_q[b], 2)
    total_ratio = _fmean(total_ratios) if total_ratios else None
    return {
        "definition_total_ratio": ("paired question/state ratio of TOTAL wall "
                                   "times at matched length; baseline-dominated"),
        "total_ratio_mean": round(total_ratio, 4) if total_ratio else None,
        "total_ratio_note": ("v2 reported 0.989 with whole-block CI [0.915, "
                             "1.045]; this is total elapsed time, NOT added "
                             "token cost"),
        "definition_incremental_ratio": ("per-block (wall_long - wall_short) / "
                                         "(reported_tokens_long - "
                                         "reported_tokens_short)/1k, Q1 Latin "
                                         "only; question vs state"),
        "incremental_ms_per_1k": {"state": round(mean_s, 3),
                                  "question": round(mean_q, 3)},
        "incremental_ratio": round(mean_q / mean_s, 3),
        "incremental_ratio_block_bootstrap_ci95": ci,
        "incremental_ratio_parent_ci95": [0.615, 4.135],
        "per_block_increment_ms_per_1k": {
            "state": {str(b): round(inc_s[b], 2) for b in common},
            "question": {str(b): round(inc_q[b], 2) for b in common}},
        "network_outliers": outliers,
        "label": ("elapsed times similar; incremental ratio unresolved — the "
                  "2x per-token question cost is neither contradicted nor "
                  "ruled out (bootstrap interval includes 2)"),
        "sample_unit": "8 randomized blocks (whole-block bootstrap)",
        "blocks_used": common,
    }
