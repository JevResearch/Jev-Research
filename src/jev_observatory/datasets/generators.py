"""Solver-backed synthetic generators with independent label verification.

Every generator follows the same discipline (DESIGN.md §3B):
* deterministic given its seed;
* items carry structured `_params` used ONLY for label construction and tests,
  stripped from the built spec;
* the solver functions below are exported so tests can verify labels through an
  independent path (plus golden-value checks for a fixed seed);
* `leakage_check` stays ON for synthetic items (explicit, recorded opt-outs
  where the task itself requires the fact in the state);
* GENERATOR_VERSION is recorded in built specs: changing generator code changes
  outputs for the same seed, so the version must travel with the artifacts.
"""

from __future__ import annotations

import random
from typing import Any

GENERATOR_VERSION = 2  # v2: balanced policy labels, multi-hop, date-window tasks

from .base import content_hash

CITIES = ["Aurora Bay", "Redstone", "Vell", "Port Meridian", "Northgate", "Calder"]
DEPARTMENTS = ["logistics", "design", "research", "support"]


def _render_company(entity: str, city: str, department: str) -> str:
    return f"{entity} is headquartered in {city}; its largest division is {department}."


def gen_relation_lookup(seed: int, n: int) -> list[dict[str, Any]]:
    """Single-hop and negation questions over a generated mini-world.

    Two groups:
    * `lookup`   – "In which city is <entity> headquartered?"
    * `negation` – "Which of these is NOT headquartered in <city>?"
    """
    if n < 2:
        raise ValueError("need at least 2 items")
    rng = random.Random(seed)
    items: list[dict[str, Any]] = []
    for index in range(n):
        entity = f"Company-{index:04d}"
        city = rng.choice(CITIES)
        department = rng.choice(DEPARTMENTS)
        cluster = content_hash({"entity": entity, "city": city, "department": department})[:12]
        passage = _render_company(entity, city, department)
        if index % 2 == 0:
            distractors = rng.sample([c for c in CITIES if c != city], k=2)
            options = {c: "a city" for c in [city, *distractors]}
            items.append({
                "id": f"lookup-{index:04d}",
                "group": "relation_lookup",
                "cluster": cluster,
                "state": f"{passage}\n\nIn which city is {entity} headquartered?",
                "questions": {"city": {
                    "type": "choice",
                    "instructions": "According to the text, where is the company headquartered?",
                    "criteria": options,
                }},
                "gold": {"city": {"value": city}},
                # The looked-up fact must legitimately appear in the state; the
                # "answer is X" style leak guard is opted out and recorded.
                "leakage_check": False,
                "_params": {"entity": entity, "city": city, "department": department, "kind": "lookup"},
            })
        else:
            other_city = rng.choice([c for c in CITIES if c != city])
            entity_here = f"Company-{(index + 1) % n:04d}"
            items.append({
                "id": f"negation-{index:04d}",
                "group": "relation_negation",
                "cluster": cluster,
                "state": (
                    f"{_render_company(entity, city, department)}\n"
                    f"{_render_company(entity_here, other_city, department)}\n\n"
                    f"Which of these companies is NOT headquartered in {city}?"
                ),
                "questions": {"entity": {
                    "type": "choice",
                    "instructions": "Choose the company whose headquarters is not in the named city.",
                    "criteria": {entity: "a company", entity_here: "a company"},
                }},
                "gold": {"entity": {"value": entity_here}},  # the one in other_city
                "leakage_check": False,
                "_params": {"city": city, "other_city": other_city, "kind": "negation"},
            })
    return items


# --- independent solver path for relation lookup ----------------------------
def solve_relation_lookup(item: dict[str, Any]) -> str:
    """Re-derive the gold answer from item state text (independent of the RNG path)."""
    state = item["state"]
    if item["_params"]["kind"] == "lookup":
        entity = item["_params"]["entity"]
        for line in state.splitlines():
            if line.startswith(f"{entity} is headquartered in "):
                return line.removeprefix(f"{entity} is headquartered in ").split(";")[0].strip()
        raise AssertionError(f"cannot parse state for {item['id']}")
    city = item["_params"]["city"]
    for line in state.splitlines():
        if " is headquartered in " not in line or "NOT" in line:
            continue  # skip the question line itself
        name = line.split(" is headquartered in ")[0].strip()
        if f"headquartered in {city};" not in line:
            return name
    raise AssertionError(f"cannot parse negation state for {item['id']}")


# ---------------------------------------------------------------- policy routing
POLICY_TEXT = (
    "Refund policy:\n"
    "- A standard refund is approved when the order is at most 30 days old and the amount is at most 200.\n"
    "- Gold members get refunds up to 60 days and 500.\n"
    "- Damaged orders are escalated to a human regardless of age or amount.\n"
    "- Anything else is denied."
)


def _solve_policy(days: int, amount: float, tier: str, damaged: bool) -> str:
    """Executable policy; kept in one place so generator and tests agree by code review."""
    if damaged:
        return "escalate"
    if tier == "gold":
        if days <= 60 and amount <= 500:
            return "approve"
        return "deny"
    if days <= 30 and amount <= 200:
        return "approve"
    return "deny"


def gen_policy_routing(seed: int, n: int) -> list[dict[str, Any]]:
    """Order records scored against a rendered, executable policy.

    v2: the case list is label-balanced (4 approve / 4 deny / 2 escalate per
    cycle of 10) so aggregate accuracy is not dominated by a majority class.
    """
    if n < 6:
        raise ValueError("need at least 6 items to cover the rule branches")
    rng = random.Random(seed)
    cases = [
        {"days": 10, "amount": 50.0, "tier": "standard", "damaged": False},   # approve
        {"days": 40, "amount": 50.0, "tier": "standard", "damaged": False},   # deny (age)
        {"days": 10, "amount": 300.0, "tier": "standard", "damaged": False},  # deny (amount)
        {"days": 40, "amount": 300.0, "tier": "gold", "damaged": False},      # approve (gold)
        {"days": 90, "amount": 100.0, "tier": "gold", "damaged": False},      # deny (age)
        {"days": 10, "amount": 50.0, "tier": "standard", "damaged": True},    # escalate
        {"days": 30, "amount": 200.0, "tier": "standard", "damaged": False},  # boundary: approve
        {"days": 31, "amount": 200.0, "tier": "standard", "damaged": False},  # boundary: deny
        {"days": 15, "amount": 120.0, "tier": "standard", "damaged": False},  # approve (balance)
        {"days": 50, "amount": 150.0, "tier": "standard", "damaged": True},    # escalate (balance)
    ]
    items: list[dict[str, Any]] = []
    for index in range(n):
        params = dict(cases[index % len(cases)])
        params["order_id"] = f"ORD-{index:05d}"
        params["customer"] = f"Customer-{index:05d}"
        state = (
            f"{POLICY_TEXT}\n\nOrder record:\n"
            f"- Order: {params['order_id']} placed by {params['customer']}\n"
            f"- Days since purchase: {params['days']}\n"
            f"- Order amount: {params['amount']:.2f} EUR\n"
            f"- Membership tier: {params['tier']}\n"
            f"- Arrived damaged: {'yes' if params['damaged'] else 'no'}"
        )
        items.append({
            "id": f"policy-{index:05d}",
            "group": "policy_routing",
            "cluster": params["order_id"],
            "state": state,            "questions": {"decision": {
                "type": "choice",
                "instructions": "Apply the policy to this order record. What is the decision?",
                "criteria": {
                    "approve": "standard or gold-member refund within limits",
                    "deny": "outside every eligible case",
                    "escalate": "send to a human agent",
                },
            }},
            "gold": {"decision": {"value": _solve_policy(**{k: params[k] for k in ("days", "amount", "tier", "damaged")})}},
            # The policy text legitimately contains the decision vocabulary
            # ("approved", "denied"); the leak guard is opted out and recorded.
            "leakage_check": False,
            "_params": params,
        })
    return items


# ------------------------------------------------------------ multi-hop relations
def gen_multihop(seed: int, n: int) -> list[dict[str, Any]]:
    """Two-hop reporting chains: 'Who ultimately manages X?' and a binary variant.

    Distractors include the 1-hop manager (the classic multi-hop error).
    Choice gold options are balanced across entities; binary items alternate
    yes/no by construction.
    """
    if n < 4:
        raise ValueError("need at least 4 items")
    items: list[dict[str, Any]] = []
    for index in range(n):
        ceo = f"Executive-{index:02d}"
        mid = f"Manager-{index:02d}"
        junior = f"Analyst-{index:02d}"
        outsider = f"Contractor-{index:02d}"
        state = (
            f"{ceo} leads the division. {mid} reports to {ceo}. "
            f"{junior} reports to {mid}. {outsider} is not part of this reporting line."
        )
        cluster = content_hash({"index": index, "kind": "multihop"})[:12]
        if index % 2 == 0:
            items.append({
                "id": f"hop-{index:02d}",
                "group": "multihop",
                "cluster": cluster,
                "state": state,
                "questions": {"boss": {
                    "type": "choice",
                    "instructions": f"Who ultimately manages {junior}, directly or indirectly?",
                    "criteria": {ceo: "leads the division", mid: "direct manager",
                                 outsider: "not in this line"},
                }},
                "gold": {"boss": {"value": ceo}},
                "leakage_check": False,
                "_params": {"chain": [junior, mid, ceo], "kind": "who_ultimately"},
            })
        else:
            gold_yes = index % 4 == 1
            subject = mid if gold_yes else outsider
            items.append({
                "id": f"hopbin-{index:02d}",
                "group": "multihop_binary",
                "cluster": cluster,
                "state": state,
                "questions": {"manages": {
                    "type": "noul",
                    "instructions": f"Does {subject} manage {junior}, directly or indirectly?",
                }},
                "gold": {"manages": {"value": gold_yes}},
                "leakage_check": False,
                "_params": {"chain": [junior, mid, ceo], "subject": subject,
                            "kind": "binary", "gold": gold_yes},
            })
    return items


def solve_multihop(item: dict[str, Any]) -> Any:
    """Independent verification: parse the reporting chain from the text."""
    state = item["state"]
    params = item["_params"]
    reports_to: dict[str, str] = {}
    for sentence in state.split(". "):
        if " reports to " in sentence:
            who = sentence.split(" reports to ")[0].split()[-1]
            to = sentence.split(" reports to ")[1].split()[0].rstrip(".")
            reports_to[who] = to
    if params["kind"] == "who_ultimately":
        node = params["chain"][0]
        while node in reports_to:
            node = reports_to[node]
        return node
    node = params["chain"][0]
    while node in reports_to:
        node = reports_to[node]
        if node == params["subject"]:
            return True
    return params["subject"] == node


# ------------------------------------------------------------ date-window boundary
DATE_POLICY = (
    "Return policy: a standard refund may be approved if the order was placed "
    "no more than 30 days before the request date. Orders older than that are "
    "outside the standard window."
)


def gen_date_window(seed: int, n: int) -> list[dict[str, Any]]:
    """Date-window membership with boundary cases and balanced binary labels.

    The micro-world states that every month has exactly 30 days, so the solver
    compares synthetic day numbers without a date library.
    """
    if n < 4:
        raise ValueError("need at least 4 items")
    gaps = [5, 31, 29, 45, 30, 33, 12, 50]  # balanced: yes on 5,29,30,12; no on 31,45,31(31)
    day_offsets = [3, 5, 12, 8, 21, 17, 25, 9]
    items: list[dict[str, Any]] = []
    for index in range(n):
        gap = gaps[index % len(gaps)]
        day = day_offsets[index % len(day_offsets)]
        month = 1 + index % 6
        purchase = f"2{index % 3:03d}-0{month}-{day:02d}"
        request_day = day + gap
        request_month = month
        while request_day > 30:
            request_day -= 30
            request_month += 1
        request = f"2{index % 3:03d}-0{request_month}-{request_day:02d}"
        eligible = gap <= 30
        items.append({
            "id": f"datewin-{index:03d}",
            "group": "date_window",
            "cluster": f"datewin-{index:04d}",
            "state": (
                f"{DATE_POLICY}\n"
                "Note: in this scenario every month has exactly 30 days.\n\n"
                f"Order ORD-{index:04d} was placed on {purchase}. "
                f"The refund request was submitted on {request}."
            ),
            "questions": {"eligible": {
                "type": "noul",
                "instructions": "According to the policy and the two dates, is this order "
                                "eligible for a standard refund?",
            }},
            "gold": {"eligible": {"value": eligible}},
            "leakage_check": False,  # the dates ARE the facts; gold is derived from them
            "_params": {"purchase": purchase, "request": request, "gap": gap,
                        "eligible": eligible},
        })
    return items


def solve_date_window(item: dict[str, Any]) -> bool:
    """Independent verification: parse the ISO dates from the text and compare."""
    import re

    dates = re.findall(r"\d{4}-\d{2}-\d{2}", item["state"])
    purchase, request = dates[0], dates[1]

    def day_number(iso: str) -> int:
        year, month, day = (int(x) for x in iso.split("-"))
        return year * 360 + month * 30 + day

    return (day_number(request) - day_number(purchase)) <= 30


# ------------------------------------------------------------ unanswerable / probability
def gen_unanswerable(seed: int, n: int) -> list[dict[str, Any]]:
    """Documents that deliberately omit the asked-for fact; gold is 'not_stated'."""
    if n < 1:
        raise ValueError("need at least 1 item")
    rng = random.Random(seed)
    items: list[dict[str, Any]] = []
    for index in range(n):
        entity = f"Artifact-{index:04d}"
        city = rng.choice(CITIES)
        year = 1950 + index % 70
        state = (
            f"The {entity} was discovered in {city} in {year}. "
            f"It weighs about 3 kilograms and is kept in a climate-controlled room."
        )
        items.append({
            "id": f"unanswerable-{index:04d}",
            "group": "unanswerable",
            "cluster": entity,
            "state": state,
            "questions": {"material": {
                "type": "choice",
                "instructions": "What material is the artifact made of, according to the document?",
                "criteria": {
                    "bronze": "a metal alloy",
                    "granite": "a stone",
                    "cedar": "a wood",
                    "not_stated": "the document does not say",
                },
            }},
            "gold": {"material": {"value": "not_stated"}},
            "_params": {"entity": entity, "asked": "material", "known": ["city", "year", "weight"]},
        })
    return items


def gen_base_rate(seed: int, n: int) -> list[dict[str, Any]]:
    """Explicit-count probability comparisons; a numeric-stress task (S9 weakness)."""
    if n < 2:
        raise ValueError("need at least 2 items")
    rng = random.Random(seed)
    items: list[dict[str, Any]] = []
    for index in range(n):
        red = rng.randint(1, 9)
        blue = rng.randint(1, 9)
        while blue == red:
            blue = rng.randint(1, 9)
        gold = "yes" if red > blue else "no"
        items.append({
            "id": "base-rate-{:04d}".format(index),
            "group": "base_rate",
            "cluster": f"bag-{index:04d}",
            "state": (
                f"A bag contains exactly {red} red marbles and {blue} blue marbles, and nothing else. "
                f"One marble is drawn uniformly at random. Is it more likely to be red than blue?"
            ),
            "questions": {"likelier_red": {
                "type": "choice",
                "instructions": "Compare the counts and answer.",
                "criteria": {"yes": "red is more likely", "no": "blue is at least as likely"},
            }},
            "gold": {"likelier_red": {"value": gold}},
            "_params": {"red": red, "blue": blue},
        })
    return items


def build_synthetic_spec(
    generators: list[tuple[str, Any]],
    *,
    seed: int = 0,
    model: str = "jev-1.13.0",
    n_per_generator: int = 10,
) -> dict[str, Any]:
    """Bundle generator outputs into one experiment spec (seeds recorded)."""
    items: list[dict[str, Any]] = []
    for name, generator in generators:
        produced = generator(seed + len(items), n_per_generator)  # per-block seed offset
        items.extend(produced)
    cleaned = [{k: v for k, v in item.items() if not k.startswith("_")} for item in items]
    return {
        "experiment": "synthetic-solvers",
        "model": model,
        "test_family": "fresh-challenge",
        "generator_version": GENERATOR_VERSION,
        "seeds": {"order": seed, "generator_base": seed},
        "generators": [name for name, _ in generators],
        "dataset": {
            "name": "synthetic-solver-backed",
            "source_url": "generated locally by jev_observatory.datasets.generators",
            "revision": None,
            "accessed_at": None,
            "license_note": "generated; no redistribution restrictions",
            "n_population": len(items),
            "sampling": {"mode": "full_generated_set", "n_per_generator": n_per_generator},
        },
        "items": cleaned,
    }
