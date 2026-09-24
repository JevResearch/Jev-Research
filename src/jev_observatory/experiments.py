"""M2 experiment builders: sweeps, isolation, odds, and the block scheduler.

Design constraints (DESIGN.md §7, IMPLEMENTATION.md M2):
* one-factor sweeps around a baseline plus a *screening* fractional design —
  never an unconditional cartesian product;
* every condition is encoded in the item `condition` string so analyses can
  group without hidden state;
* the scheduler randomizes within blocks and interleaves a sentinel condition
  for drift/jitter detection;
* dispatch order is data: experiments set `shuffle: false` and lay items out in
  their intended block order (concurrency must be 1 for latency work anyway).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any

WORDS = (
    "audit beacon candle delta ember fjord garnet harbor inkling jigsaw kernel "
    "lantern meadow nectar orchid parchment quarry ribbon saffron tundra umbrella "
    "velvet willow xenon yonder zephyr anchor bramble cinder dapple".split()
)

PRIMITIVES = ("choice", "noul", "score")
DEFAULT_L_LEVELS = (128, 512, 2048, 8192)
DEFAULT_Q_LEVELS = (1, 4, 16, 64)
DEFAULT_K_LEVELS = (2, 8, 64, 255)
BASELINE = {"L": 512, "Q": 4, "K": 8, "primitive": "choice", "qlen": "short",
            "content": "natural", "cold": 0}

MAX_STATE_CHARS = 30_000  # headroom under the documented 32k-token state limit
MAX_QUESTIONS = 64
MAX_OPTIONS = 255


def condition_id(params: dict[str, Any]) -> str:
    """Canonical condition encoding; analyses parse this back with condition_param."""
    order = ("L", "Q", "K", "primitive", "qlen", "content", "cold", "role",
             "sibling_set", "odds_variant", "block")
    parts = [f"{key}={params[key]}" for key in order if key in params]
    return ";".join(parts)


def condition_param(condition: str, key: str) -> str | None:
    for part in condition.split(";"):
        if part.startswith(f"{key}="):
            return part.split("=", 1)[1]
    return None


# ------------------------------------------------------------------ content
def _state_text(rng: random.Random, *, length: int, content: str) -> str:
    if content == "random_ids":
        chunk = " ".join(f"REC-{rng.getrandbits(32):08x}" for _ in range(max(1, length // 14)))
        return chunk[:length]
    if content == "repetitive":
        sentence = "The quarterly report was filed on time and reviewed by the committee. "
        return (sentence * (length // len(sentence) + 1))[:length]
    words = " ".join(rng.choice(WORDS) for _ in range(max(1, length // 7)))
    return words[:length]


def _question_text(rng: random.Random, *, qlen: str, index: int, primitive: str) -> str:
    if qlen == "long":
        filler = " ".join(rng.choice(WORDS) for _ in range(90))
        return f"Question {index}: weighing the considerations above ({filler}), rate the state."
    return f"Question {index}: apply the rubric to the state."


def _make_question(rng: random.Random, *, index: int, primitive: str, k: int, qlen: str) -> tuple[str, dict[str, Any]]:
    text = _question_text(rng, qlen=qlen, index=index, primitive=primitive)
    if primitive == "noul":
        return f"n{index:03d}", {
            "type": "noul",
            "instructions": f"{text} Answer yes or no.",
        }
    if primitive == "score":
        return f"s{index:03d}", {
            "type": "score",
            "instructions": text,
            "criteria": [f"level {i}: lowest" if i == 0 else
                         (f"level {i}: highest" if i == k - 1 else f"level {i}: middle")
                         for i in range(min(k, 10))],
        }
    k = min(k, MAX_OPTIONS)
    options = {}
    for option in range(k):
        options[f"opt_{option:03d}"] = f"rubric option {option}: " + rng.choice(WORDS)
    return f"c{index:03d}", {
        "type": "choice",
        "instructions": text,
        "criteria": options,
    }


@dataclass(frozen=True)
class SweepCondition:
    L: int
    Q: int
    K: int
    primitive: str
    qlen: str
    content: str
    cold: int

    def params(self) -> dict[str, Any]:
        return {"L": self.L, "Q": self.Q, "K": self.K, "primitive": self.primitive,
                "qlen": self.qlen, "content": self.content, "cold": self.cold}

    def validate(self) -> None:
        if self.L > MAX_STATE_CHARS:
            raise ValueError(f"L={self.L} exceeds context headroom {MAX_STATE_CHARS}")
        if self.Q > MAX_QUESTIONS:
            raise ValueError(f"Q={self.Q} exceeds tested question bound {MAX_QUESTIONS}")
        if self.K > MAX_OPTIONS:
            raise ValueError(f"K={self.K} exceeds option cap {MAX_OPTIONS}")
        if self.primitive not in PRIMITIVES:
            raise ValueError(f"unknown primitive {self.primitive!r}")


def one_factor_sweep(
    *,
    l_levels: tuple[int, ...] = DEFAULT_L_LEVELS,
    q_levels: tuple[int, ...] = DEFAULT_Q_LEVELS,
    k_levels: tuple[int, ...] = DEFAULT_K_LEVELS,
    primitives: tuple[str, ...] = ("choice",),
    qlens: tuple[str, ...] = ("short",),
    contents: tuple[str, ...] = ("natural",),
    cold_levels: tuple[int, ...] = (0,),
    baseline: dict[str, Any] | None = None,
) -> list[SweepCondition]:
    """Vary one factor at a time from the baseline; plus the baseline itself."""
    base = {**BASELINE, **(baseline or {})}
    conditions: list[SweepCondition] = []
    seen: set[tuple] = set()

    def add(params: dict[str, Any]) -> None:
        cond = SweepCondition(
            L=params["L"], Q=params["Q"], K=params["K"], primitive=params["primitive"],
            qlen=params["qlen"], content=params["content"], cold=params["cold"],
        )
        key = tuple(sorted(cond.params().items()))
        if key not in seen:
            seen.add(key)
            conditions.append(cond)

    add(base)
    for level in l_levels:
        add({**base, "L": level})
    for level in q_levels:
        add({**base, "Q": level})
    for level in k_levels:
        add({**base, "K": level})
    for primitive in primitives:
        add({**base, "primitive": primitive})
    for qlen in qlens:
        add({**base, "qlen": qlen})
    for content in contents:
        add({**base, "content": content})
    for cold in cold_levels:
        add({**base, "cold": cold})
    return conditions


def screening_cross(l_levels: tuple[int, ...], q_levels: tuple[int, ...],
                    k_levels: tuple[int, ...] | None = None) -> list[SweepCondition]:
    """Small L×Q (and Q×K) interaction screen: aligned + anti-aligned diagonals.

    This is an explicitly labelled *screening* design, not a full factorial.
    """
    conditions: list[SweepCondition] = []
    base = dict(BASELINE)
    pairs = list(zip(l_levels, q_levels)) + list(zip(reversed(l_levels), q_levels))
    for L, Q in dict.fromkeys(pairs):  # dedupe, keep order
        conditions.append(SweepCondition(L=L, Q=Q, K=base["K"], primitive="choice",
                                         qlen="short", content="natural", cold=0))
    if k_levels:
        for K, Q in dict.fromkeys(zip(k_levels, q_levels)):
            conditions.append(SweepCondition(L=base["L"], Q=Q, K=K, primitive="choice",
                                             qlen="short", content="natural", cold=0))
    return conditions


def build_sweep_items(
    conditions: list[SweepCondition],
    *,
    repeats: int,
    seed: int,
    question_length_levels: tuple[str, ...] = ("short",),
) -> list[dict[str, Any]]:
    """Items laid out block-by-block: sentinel first, then every condition once.

    Conditions are *shuffled within each block* (seeded, reproducible): a fixed
    condition order would let slow drift masquerade as a condition effect.
    Each item carries its block index in `condition`, so analyses can hold out
    whole blocks and the planned→dispatched→recorded chain is checkable.
    """
    for cond in conditions:
        cond.validate()
    items: list[dict[str, Any]] = []
    rng = random.Random(seed)
    for block in range(repeats):
        items.append(_sentinel_item(block, seed=seed))
        order = list(range(len(conditions)))
        rng.shuffle(order)  # within-block randomization, reproducible from seed
        for index in order:
            cond = conditions[index]
            qlen = question_length_levels[index % len(question_length_levels)]
            state = _state_text(random.Random(f"{seed}:{index}:{cond.L}:{cond.content}"),
                                length=cond.L, content=cond.content)
            questions: dict[str, Any] = {}
            for q in range(cond.Q):
                qid, question = _make_question(random.Random(f"{seed}:{index}:{q}:{cond.primitive}:{cond.qlen}"),
                                               index=q, primitive=cond.primitive, k=cond.K, qlen=cond.qlen)
                questions[qid] = question
            items.append({
                "id": f"sweep-b{block:02d}-c{index:03d}",
                "group": "scaling_sweep",
                "cluster": condition_id(cond.params()),  # cluster = condition for block bootstrap
                "condition": condition_id({**cond.params(), "block": block}),
                "state": state,
                "gold": {},
                "questions": questions,
            })
    return items


def _sentinel_item(block: int, *, seed: int) -> dict[str, Any]:
    """Fixed reference condition, identical every block, for drift/jitter checks."""
    state = _state_text(random.Random(f"{seed}:sentinel"), length=512, content="natural")
    questions = {
        "c000": {"type": "choice", "instructions": "Sentinel: apply the rubric.",
                 "criteria": {"opt_000": "rubric option 0", "opt_001": "rubric option 1",
                              "opt_002": "rubric option 2", "opt_003": "rubric option 3"}},
        "n000": {"type": "noul", "instructions": "Sentinel: answer yes or no."},
    }
    return {
        "id": f"sentinel-b{block:02d}",
        "group": "sentinel",
        "cluster": "sentinel",
        "condition": f"role=sentinel;block={block}",
        "state": state,
        "gold": {},
        "questions": questions,
    }


def sweep_spec(
    conditions: list[SweepCondition],
    *,
    repeats: int,
    seed: int,
    model: str = "jev-1.13.0",
    question_length_levels: tuple[str, ...] = ("short",),
) -> dict[str, Any]:
    items = build_sweep_items(conditions, repeats=repeats, seed=seed,
                              question_length_levels=question_length_levels)
    return {
        "experiment": "scaling-sweep",
        "model": model,
        "test_family": "mechanism-scaling",
        "seeds": {"order": seed},
        "shuffle": False,  # dispatch order is the randomized block layout
        "claim_type": "confirmatory",  # preregistered sweeps; isolation/odds are exploratory
        "dataset": {
            "name": "scaling-sweep",
            "source_url": "generated locally",
            "license_note": "generated; no redistribution restrictions",
            "n_population": len(items),
            "sampling": {"mode": "designed_sweep", "repeats": repeats,
                         "n_conditions": len(conditions),
                         "design_note": "one-factor + screening cross; NOT a full factorial"},
        },
        "items": items,
    }


# ------------------------------------------------------------------ isolation
def isolation_spec(
    *,
    sibling_levels: tuple[int, ...] = (0, 1, 8),
    repeats: int = 5,
    seed: int = 0,
    model: str = "jev-1.13.0",
) -> dict[str, Any]:
    """Anchor-with-siblings: identical anchor question, varying sibling sets.

    The anchor's probability distribution is compared across sibling sets within
    each repeat (cluster). DESIGN.md §7: a null result is not proof of
    independence; we report equivalence margins explicitly and label the claim
    exploratory.
    """
    rng = random.Random(seed)
    state = _state_text(random.Random(f"{seed}:anchor"), length=2048, content="natural")
    anchor_question = {
        "type": "choice",
        "instructions": "Does the state describe a refund request? Choose exactly one.",
        "criteria": {"yes": "mentions a refund", "no": "does not mention a refund"},
    }
    items: list[dict[str, Any]] = []
    for repeat in range(repeats):
        for level in sibling_levels:
            questions: dict[str, Any] = {"anchor": anchor_question}
            for sib in range(level):
                if sib % 3 == 0:
                    questions[f"s{sib:03d}"] = {
                        "type": "noul",
                        "instructions": f"Does the state mention the word {rng.choice(WORDS)}?",
                    }
                elif sib % 3 == 1:
                    questions[f"s{sib:03d}"] = {
                        "type": "choice",
                        "instructions": f"Which {rng.choice(WORDS)} category fits best?",
                        "criteria": {"alpha": "first", "beta": "second", "gamma": "third"},
                    }
                else:
                    questions[f"s{sib:03d}"] = {
                        "type": "score",
                        "instructions": f"Rate the state's relevance to {rng.choice(WORDS)}.",
                        "criteria": ["irrelevant", "tangential", "central"],
                    }
            items.append({
                "id": f"anchor-r{repeat:02d}-s{level:03d}",
                "group": "isolation",
                "cluster": f"anchor-r{repeat:02d}",  # repeats cluster together
                "condition": f"role=anchor;sibling_set={level};repeat={repeat}",
                "state": state,  # byte-identical state across the whole design
                "gold": {},
                "questions": questions,
            })
    return {
        "experiment": "isolation-anchor",
        "model": model,
        "test_family": "mechanism-isolation",
        "seeds": {"order": seed},
        "shuffle": False,
        "claim_type": "exploratory",
        "dataset": {
            "name": "isolation-anchor",
            "source_url": "generated locally",
            "license_note": "generated",
            "n_population": len(items),
            "sampling": {"mode": "designed_isolation", "sibling_levels": list(sibling_levels),
                         "repeats": repeats},
        },
        "items": items,
    }


# ------------------------------------------------------------------ isolation v2
def isolation_v2_spec(
    *,
    sibling_levels: tuple[int, ...] = (0, 8),
    renamed: bool = True,
    repeats: int = 4,
    seed: int = 0,
    model: str = "jev-1.13.0",
) -> dict[str, Any]:
    """Non-saturated anchors: a design that can actually detect influence.

    Audit finding: the original isolation design used one trivially-saturated
    anchor ('no' at 1.0), which quantization would hide any change in. Here each
    anchor is a *mixed-signal* state (refund language AND billing language) so
    the displayed distribution should be split, not one-hot. Sibling sets are
    irrelevant or conflicting, and a renamed-ID condition probes the documented
    'ids never reach the model' claim (S3).
    """
    rng = random.Random(seed)
    anchors = [
        {
            "name": "mixed-refund-billing",
            "state": (
                "Ticket 88213: The customer was charged twice for order 5541 — once on the 3rd "
                "and once on the 5th. They are asking whether this is a billing error or whether "
                "they should request a refund for the duplicate charge. The card statement shows "
                "two line items of 49.90 EUR each."
            ),
        },
        {
            "name": "partial-refund",
            "state": (
                "Ticket 91402: The customer received item A of their two-item order but item B "
                "never arrived. They want to know what can be done; they mention they would "
                'accept a refund "for at least the missing part" but would prefer replacement.'
            ),
        },
        {
            "name": "ambiguous-complaint",
            "state": (
                "Ticket 90337: The customer writes: 'Your service has been unacceptable lately. "
                "I demand this be made right.' No order number, no product, no dates are given, "
                "and the message does not say what remedy they expect."
            ),
        },
    ]
    anchor_question = {
        "type": "choice",
        "instructions": "Does the state describe a refund request? Choose exactly one.",
        "criteria": {"yes": "mentions a refund", "no": "does not mention a refund"},
    }

    def siblings(count: int, conflicting: bool) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for sib in range(count):
            word = rng.choice(WORDS)
            if conflicting and sib % 2 == 0:
                # Siblings whose instructions contradict the anchor's framing.
                out[f"c{sib:03d}"] = {
                    "type": "choice",
                    "instructions": "Which single word below appears verbatim in the state?",
                    "criteria": {word: f"the word {word}", "zz_none": "none of these"},
                }
            elif sib % 3 == 0:
                out[f"s{sib:03d}"] = {
                    "type": "noul",
                    "instructions": f"Does the state mention the word {word}?",
                }
            elif sib % 3 == 1:
                out[f"s{sib:03d}"] = {
                    "type": "choice",
                    "instructions": f"Which {word} category fits best?",
                    "criteria": {"alpha": "first", "beta": "second", "gamma": "third"},
                }
            else:
                out[f"s{sib:03d}"] = {
                    "type": "score",
                    "instructions": f"Rate the state's relevance to {word}.",
                    "criteria": ["irrelevant", "tangential", "central"],
                }
        return out

    items: list[dict[str, Any]] = []
    for repeat in range(repeats):
        for anchor in anchors:
            conditions = {
                f"alone": siblings(0, False),
                "irrelevant8": siblings(8, False),
                "conflict8": siblings(8, True),
            }
            for cond_name, sibling_questions in conditions.items():
                questions = {"anchor": anchor_question, **sibling_questions}
                item = {
                    "id": f"anchor2-{anchor['name']}-{cond_name}-r{repeat:02d}",
                    "group": "isolation_v2",
                    "cluster": f"anchor2-{anchor['name']}-r{repeat:02d}",
                    "condition": f"role=anchor2;variant={cond_name};anchor={anchor['name']};repeat={repeat}",
                    "state": anchor["state"],
                    "gold": {},
                    "questions": questions,
                }
                if renamed and cond_name != "alone":
                    item = rename_question_ids(item, style="opaque")
                items.append(item)
    return {
        "experiment": "isolation-anchor-v2",
        "model": model,
        "test_family": "mechanism-isolation",
        "seeds": {"order": seed},
        "shuffle": False,
        "claim_type": "confirmatory",
        "dataset": {
            "name": "isolation-anchor-v2",
            "source_url": "generated locally",
            "license_note": "generated",
            "n_population": len(items),
            "sampling": {"mode": "designed_isolation_v2", "sibling_levels": list(sibling_levels),
                         "repeats": repeats, "n_anchors": len(anchors)},
        },
        "items": items,
    }


# ------------------------------------------------------------------ jitter
def jitter_spec(
    *,
    n_repeats: int = 30,
    seed: int = 0,
    model: str = "jev-1.13.0",
) -> dict[str, Any]:
    """N byte-identical payloads sent as distinct logical requests.

    Measures ordinary serving variation without confounding from changed
    content. The identical payload also probes response determinism.
    """
    state = _state_text(random.Random(f"{seed}:jitter"), length=1024, content="natural")
    questions = {
        "q0": {"type": "choice", "instructions": "Apply the rubric.",
               "criteria": {"opt_0": "first", "opt_1": "second", "opt_2": "third", "opt_3": "fourth"}},
        "q1": {"type": "noul", "instructions": "Answer yes or no."},
    }
    items = [{
        "id": f"jitter-{i:03d}",
        "group": "jitter",
        "cluster": "jitter",
        "condition": f"role=jitter;index={i}",
        "state": state,  # byte-identical across items
        "gold": {},
        "questions": json.loads(json.dumps(questions)),
    } for i in range(n_repeats)]
    return {
        "experiment": "serving-jitter",
        "model": model,
        "test_family": "mechanism-jitter",
        "seeds": {"order": seed},
        "shuffle": False,
        "claim_type": "confirmatory",
        "dataset": {
            "name": "serving-jitter",
            "source_url": "generated locally",
            "license_note": "generated",
            "n_population": len(items),
            "sampling": {"mode": "identical_repeats", "n_repeats": n_repeats},
        },
        "items": items,
    }


# ------------------------------------------------------------------ batch vs separate
BATCH_STATE = (
    "Ticket 77120: The customer reports that the mobile app crashes when opening the "
    "export screen, that they were billed twice this month, and that they would like "
    "to know whether their data was affected by the recent incident. They also ask "
    "about upgrading to the business plan and mention they may cancel if this is not "
    "resolved quickly."
)


def batch_vs_separate_spec(
    *,
    repeats: int = 3,
    seed: int = 0,
    model: str = "jev-1.13.0",
) -> dict[str, Any]:
    """One batched call with K questions vs K separate one-question calls.

    Directly compares the batching benefit (S2) and whether per-question answers
    change between the batched and separate conditions.
    """
    batch_questions = {
        "technical": {"type": "choice", "instructions": "Which team should handle the app crash?",
                      "criteria": {"billing": "payments, invoicing, refunds",
                                   "technical": "bugs, outages, integrations",
                                   "sales": "pricing, upgrades, accounts"}},
        "billing_issue": {"type": "noul", "instructions": "Does the state describe a billing problem?"},
        "security_concern": {"type": "noul", "instructions": "Does the state raise a data-security question?"},
        "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                        "criteria": ["calm", "annoyed", "angry"]},
    }
    items: list[dict[str, Any]] = []
    for repeat in range(repeats):
        items.append({
            "id": f"batch-r{repeat:02d}",
            "group": "batch_vs_separate",
            "cluster": f"batch-r{repeat:02d}",
            "condition": f"role=batch;mode=batched;repeat={repeat}",
            "state": BATCH_STATE,
            "gold": {},
            "questions": json.loads(json.dumps(batch_questions)),
        })
        for qid, question in batch_questions.items():
            items.append({
                "id": f"sep-r{repeat:02d}-{qid}",
                "group": "batch_vs_separate",
                "cluster": f"batch-r{repeat:02d}",
                "condition": f"role=batch;mode=separate;question={qid};repeat={repeat}",
                "state": BATCH_STATE,
                "gold": {},
                "questions": {qid: json.loads(json.dumps(question))},
            })
    return {
        "experiment": "batch-vs-separate",
        "model": model,
        "test_family": "mechanism-batching",
        "seeds": {"order": seed},
        "shuffle": False,
        "claim_type": "confirmatory",
        "dataset": {
            "name": "batch-vs-separate",
            "source_url": "generated locally",
            "license_note": "generated",
            "n_population": len(items),
            "sampling": {"mode": "designed_batching", "k_questions": len(batch_questions),
                         "repeats": repeats},
        },
        "items": items,
    }


# ------------------------------------------------------------------ ambiguous odds
ODDS_AMBIGUOUS_STATE = (
    "Ticket 60411: The customer was charged twice this month for the same subscription "
    "(order 8331, 19.99 EUR each). They ask whether this is a billing error and say they "
    'may need to "ask for the money back" if it cannot be resolved. The duplicate charge '
    "has already been flagged automatically by the billing system."
)
ODDS_AMBIGUOUS_OPTIONS = {
    "billing": "route to the billing team",
    "technical": "route to the technical team",
    "refund_flow": "start the refund flow",
    "human_review": "escalate to a human",
}


def odds_ambiguous_spec(*, repeats: int = 6, seed: int = 0, model: str = "jev-1.13.0") -> dict[str, Any]:
    """Odds-ratio probe on a task designed to split probability mass.

    Audit finding: the original odds task produced one-hot answers, making the
    A:B odds undefined. Here the state deliberately mixes two strong signals
    (duplicate charge -> billing AND refund flow), so 'billing' and 'refund_flow'
    should both carry mass in the base condition.
    """
    items: list[dict[str, Any]] = []
    for repeat in range(repeats):
        variants = {
            "base": dict(ODDS_AMBIGUOUS_OPTIONS),
            "irrelevant": {**ODDS_AMBIGUOUS_OPTIONS, **IRRELEVANT_OPTIONS},
        }
        for variant, criteria in variants.items():
            items.append({
                "id": f"oddsamb-r{repeat:02d}-{variant}",
                "group": "candidate_odds_v2",
                "cluster": f"oddsamb-r{repeat:02d}",
                "condition": (f"role=odds2;odds_variant={variant};"
                              f"tracked_pair=billing,refund_flow;repeat={repeat}"),
                "state": ODDS_AMBIGUOUS_STATE,
                "gold": {},
                "questions": {"verdict": {
                    "type": "choice",
                    "instructions": "Which action best fits this ticket? Choose exactly one.",
                    "criteria": criteria,
                }},
            })
    return {
        "experiment": "candidate-odds-ambiguous",
        "model": model,
        "test_family": "mechanism-odds",
        "seeds": {"order": seed},
        "shuffle": False,
        "claim_type": "exploratory",
        "dataset": {
            "name": "candidate-odds-ambiguous",
            "source_url": "generated locally",
            "license_note": "generated",
            "n_population": len(items),
            "sampling": {"mode": "designed_odds_v2", "repeats": repeats,
                         "tracked_pair": ["billing", "refund_flow"]},
        },
        "items": items,
    }
ODDS_BASE_OPTIONS = {
    "A": "the invoice total exceeds the threshold",
    "B": "the invoice total is within the threshold",
    "C": "the invoice is a duplicate submission",
    "D": "the invoice lacks a purchase order",
}
IRRELEVANT_OPTIONS = {
    "E": "the weather in the shipping region",
    "F": "the accountant's favourite colour",
    "G": "a poem about the sea",
    "H": "the office plant watering schedule",
}


def odds_spec(*, repeats: int = 8, seed: int = 0, model: str = "jev-1.13.0") -> dict[str, Any]:
    """Add irrelevant or duplicate distractors; track odds of the base pair A/B.

    Candidate-local scoring predicts the A:B odds ratio is unchanged by
    appending unrelated options; a duplicate of A can legitimately split A's
    mass, which is a documented expectation, not a violation.
    """
    items: list[dict[str, Any]] = []
    for repeat in range(repeats):
        variants = {
            "base": dict(ODDS_BASE_OPTIONS),
            "irrelevant": {**ODDS_BASE_OPTIONS, **IRRELEVANT_OPTIONS},
            "duplicate": {**ODDS_BASE_OPTIONS, "A2": ODDS_BASE_OPTIONS["A"]},
        }
        for variant, criteria in variants.items():
            items.append({
                "id": f"odds-r{repeat:02d}-{variant}",
                "group": "candidate_odds",
                "cluster": f"odds-r{repeat:02d}",
                "condition": f"role=odds;odds_variant={variant};tracked_pair=A,B;repeat={repeat}",
                "state": "Invoice INV-4412 totals 1,240.00 EUR and arrived without a purchase order reference.",
                "gold": {},
                "questions": {"verdict": {
                    "type": "choice",
                    "instructions": "Which description best fits the invoice?",
                    "criteria": criteria,
                }},
            })
    return {
        "experiment": "candidate-odds",
        "model": model,
        "test_family": "mechanism-odds",
        "seeds": {"order": seed},
        "shuffle": False,
        "claim_type": "exploratory",
        "dataset": {
            "name": "candidate-odds",
            "source_url": "generated locally",
            "license_note": "generated",
            "n_population": len(items),
            "sampling": {"mode": "designed_odds", "repeats": repeats,
                         "tracked_pair": ["A", "B"]},
        },
        "items": items,
    }


# ------------------------------------------------------------------ scheduler
def randomize_blocks(units: list[Any], *, block_size: int, seed: int) -> list[Any]:
    """Deterministic within-block shuffle; blocks keep their relative order."""
    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    rng = random.Random(seed)
    out: list[Any] = []
    for start in range(0, len(units), block_size):
        block = units[start:start + block_size]
        indices = list(range(len(block)))
        rng.shuffle(indices)
        out.extend(block[i] for i in indices)
    return out


def rename_question_ids(item: dict[str, Any], *, style: str = "opaque") -> dict[str, Any]:
    """Question-ID visibility probe: rename keys without touching content.

    The vendor claims question ids never reach the model (S3); if renaming
    changes answers beyond jitter, that claim is falsifiable.
    """
    import copy

    out = copy.deepcopy(item)
    questions = out["questions"]
    if style == "opaque":
        renamed = {f"q_{i:05d}": q for i, (_, q) in enumerate(sorted(questions.items()))}
    elif style == "semantic":
        renamed = {f"semantic_{name}": q for i, (name, q) in enumerate(questions.items())}
    else:
        raise ValueError(f"unknown style {style!r}")
    out["questions"] = renamed
    out["id"] = f"{item['id']}:{style}"
    out["condition"] = f"{item.get('condition', '')};renamed={style}".lstrip(";")
    return out


def reorder_questions(item: dict[str, Any], *, seed: int) -> dict[str, Any]:
    import copy

    out = copy.deepcopy(item)
    names = list(item["questions"])
    random.Random(seed).shuffle(names)
    out["questions"] = {name: item["questions"][name] for name in names}
    out["id"] = f"{item['id']}:reordered"
    out["condition"] = f"{item.get('condition', '')};reordered=1".lstrip(";")
    return out