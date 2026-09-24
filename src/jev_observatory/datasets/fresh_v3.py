"""Fresh-task generator families v3 (LEAD-GATE 2026-09-18, requirement B).

Three solver-verified families designed to fix the audited weaknesses of the
earlier synthetic generators:

* ``graph_two_hop`` — anonymous named entities and random edges; the question
  asks precisely for the node exactly TWO edges from a selected node. Every
  item contains a true two-hop positive, a one-hop distractor and a
  disconnected (> two edges) negative. No Executive/Manager or other role
  cues; option descriptions are neutral. Labels are verified by a separately
  implemented traversal (``fresh_v3_verify``), never by trusting generator
  gold.
* ``missing_info`` — random entity/value associations with globally disjoint
  values; balanced answerable and absent-fact items with an explicit
  ``not_stated`` option that is NOT always correct. Option mappings are
  shuffled and the gold option key position is balanced *after* serialisation.
* ``synthetic_dates`` — an explicitly specified 30-day-month calendar; start
  day and gap are varied independently (gaps 5/10/20/29/30/31/45/50 plus far
  controls), covering same-month and crossing-month cells wherever they are
  mathematically possible. Labels are verified two independent ways: integer
  day arithmetic versus incremental calendar stepping. Impossible factorial
  cells (e.g. a 50-day gap cannot land in the same month of a 30-day month)
  are recorded in the coverage table, never asserted.

Every family genuinely varies with its seed: items carry a *substantive
signature* (structural content with volatile name/value tokens masked), and
the spec validator refuses a family whose signatures collapse to rename-only
variation. Development and test items must have disjoint substantive
signatures. Generator version and actual per-family seeds are recorded in the
spec. Cross-language / knowledge-cutoff work is deferred.
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any

GENERATOR_VERSION = "fresh_v3-1.0.0"

# Neutral person names (no titles, no roles, no gendered hints).
NAME_POOL = (
    "Alina Boris Cemir Dalia Eino Farid Gunhild Hideko Ivo Jasleen Kelso Luminita "
    "Mirek Nadia Odalys Pavel Qira Rosalind Sefa Toma Umeko Vasile Wren Ximena "
    "Yusuf Zora Adria Basim Cato Delphine Elmas Fenn Gavril Halia Iker Jorunn "
    "Kalinda Levan Mahsa Nils Oksana Pirro Quenna Rurik Sanne Teodora Ulmo "
).split()

GRAPH_ROLE_TOKENS = (
    "two", "hop", "edge", "adjacent", "neighbo", "distance", "direct",
    "first", "second", "third", "connected", "path", "executive", "manager",
)

DATE_GAPS = (5, 10, 20, 29, 30, 31, 45, 50)
DAYS_IN_MONTH = 30


# ------------------------------------------------------------- substantive signatures
def substantive_signature(item: dict[str, Any], *, family: str) -> str:
    """Deterministic structural hash derived from OUTBOUND content.

    The signature is computed from structures PARSED out of the state text,
    instructions and criteria (same parsers as the independent verifier):

    * graph: a rename-invariant structural template — the sorted multiset of
      (BFS-distance-from-target, BFS-distance-from-target) edge pairs with
      disconnected nodes coded -1, the per-shell node counts, and the option
      distances;
    * missing-info: the parsed attribute, the target's registry content, the
      sorted other registry contents, and the sorted offered option contents;
    * dates: the parsed (start_day, gap) cell.

    Consequently the signature is invariant to pure name substitution, option
    ordering, registry-line ordering and date phrasing, and it is deterministic
    across PYTHONHASHSEED (every ordering comes from sorted lists, never from
    set/dict iteration). Two items share a signature iff they ask the same
    substantive question with the same structure and quantities — rename-only
    or presentation-only generators collapse to few signatures and are caught
    by the validator and adversarial tests.
    """
    from . import fresh_v3_verify as _verify  # lazy: avoid import cycle

    if family == "graph_two_hop":
        # Rename-invariant structural template: every node is abstracted to its
        # BFS distance from the parsed target (-1 = not connected to the
        # target's component), and the edge set becomes the sorted multiset of
        # (distance, distance) pairs. Names, name ordering and option order
        # never enter the signature, so pure rename-only or reorder-only
        # variation cannot create novelty; a genuinely different structure
        # (different shells / edge profile) does.
        target = _verify.parse_graph_target(item["questions"]["answer"]["instructions"])
        edges = _verify.parse_edges(item["state"])
        distances = _verify.bfs_distances(edges, target)
        shell = {node: distances.get(node, -1) for node in
                 {a for a, _b in edges} | {b for _a, b in edges}}
        profile = _wl_structure_profile(edges, shell)
        option_distances = sorted(
            shell.get(key, distances.get(key, -1))
            for key in item["questions"]["answer"]["criteria"]
        )
        options = sorted(item["questions"]["answer"]["criteria"].values())
        payload = ["graph", str(profile["node_colors"]), str(profile["edge_colors"]),
                   str(option_distances), str(options)]
    elif family == "missing_info":
        attribute, target = _verify.parse_missing_info_question(
            item["questions"]["answer"]["instructions"]
        )
        registry = _verify.parse_registry(item["state"])
        target_content = registry[target]
        other_contents = sorted(
            rest for entity, rest in registry.items() if entity != target
        )
        offered = sorted(item["questions"]["answer"]["criteria"].values())
        payload = ["missing", attribute, target_content, str(other_contents), str(offered)]
    elif family == "synthetic_dates":
        start, gap = _verify.parse_dates_state(item["state"])
        payload = ["dates", f"start={start}", f"gap={gap}"]
    else:
        raise ValueError(f"no signature rule for family {family!r}")
    return hashlib.sha256("|".join([family, *payload]).encode("utf-8")).hexdigest()


def _wl_structure_profile(edges: list[tuple[str, str]],
                          initial: dict[str, int], *, rounds: int = 3) -> dict[str, list[int]]:
    """Rename-invariant structural template of a graph (1-WL colour
    refinement seeded with name-independent initial colours, here the BFS
    distance from the parsed target, -1 = disconnected).

    Every ordering comes from sorted lists and every colour merge from a
    sha256 of a sorted tuple, so the profile is fully deterministic across
    PYTHONHASHSEED and invariant to node renaming and edge listing order.
    This is an explicit structural template (LEAD-GATE: "an explicit
    structural template representation with verified payload correspondence is
    acceptable"); it makes NO universal graph-isomorphism guarantee.
    """
    adjacency: dict[str, set[str]] = {}
    for a, b in edges:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    colors = {node: initial.get(node, -1) for node in adjacency}
    for _ in range(rounds):
        signatures = {
            node: hashlib.sha256(
                "|".join([
                    str(colors[node]),
                    *sorted(str(colors[neighbour]) for neighbour in adjacency[node]),
                ]).encode("utf-8")).hexdigest()
            for node in adjacency
        }
        ranks = {signature: rank for rank, signature in enumerate(sorted(set(signatures.values())))}
        colors = {node: ranks[signatures[node]] for node in adjacency}
    edge_colors = sorted(
        tuple(sorted((colors[a], colors[b]))) for a, b in edges
    )
    return {"node_colors": sorted(colors.values()), "edge_colors": edge_colors}


# ------------------------------------------------------------------ graph family
def _graph_state(rng: random.Random, names: list[str], edges: list[tuple[str, str]]) -> str:
    edge_text = ", ".join(f"{a}–{b}" for a, b in edges)
    return f"Directory of working relationships: {edge_text}."


def make_graph_item(*, seed: int, item_id: str, gold_position: int = 0) -> dict[str, Any]:
    """One anonymous two-hop graph item: positive + one-hop distractor + negative."""
    rng = random.Random(f"graph:{seed}:{item_id}")
    names = rng.sample(NAME_POOL, 10)
    query, positive, distractor, negative = names[0], names[1], names[2], names[3]
    others = names[4:]

    # Fix the structural roles first, then add random extra edges that never
    # shorten the roles: positive stays at distance exactly 2, distractor at
    # distance 1, negative at distance > 2.
    edges = [(query, distractor), (distractor, positive)]
    for name in others:
        # attach every remaining node inside the far side or to the distractor's
        # component *beyond* distance 2 from query: attach to positive or negative
        candidates = [n for n in (positive, negative, *others) if n != name]
        edges.append((rng.choice(candidates), name))
    # keep the negative disconnected from the query's side: negative only links
    # into the far component via positive/other far nodes
    edges = [(a, b) for a, b in edges if {a, b} != {query, positive} and {a, b} != {query, negative}]
    rng.shuffle(edges)
    state = _graph_state(rng, names, edges)

    option_keys = [positive, distractor, negative]
    rng.shuffle(option_keys)
    gold_key = positive
    # balance the gold position AFTER serialisation: rotate the shuffled keys so
    # the gold key lands exactly on the requested position (round-robin per item)
    target = gold_position % len(option_keys)
    j = option_keys.index(gold_key)
    k = (j - target) % len(option_keys)
    option_keys = option_keys[k:] + option_keys[:k]
    criteria = {key: "a person named in the relationship list" for key in option_keys}
    role_of = {"positive": positive, "distractor": distractor, "negative": negative}
    item = {
        "id": item_id,
        "group": "fresh_v3_graph",
        "cluster": f"fresh_v3_graph:{seed}",
        "condition": f"role=fresh_v3;family=graph_two_hop;seed={seed}",
        "state": state,
        "leakage_check": False,  # the gold node name legitimately appears in the graph state
        "gold": {
            "answer": {"value": gold_key},
            "target_node": query,
            "role_of": role_of,
        },
        "questions": {
            "answer": {
                "type": "choice",
                "instructions": (
                    f"In the directory described, which person is exactly two working "
                    f"relationships away from {query}? Choose exactly one."
                ),
                "criteria": criteria,
            },
        },
    }
    return item


# ------------------------------------------------------------ missing-info family
ATTRIBUTES = {
    "desk location": ("floor 1", "floor 2", "floor 3", "floor 4"),
    "equipment model": ("Alphaview 7", "Betaview 9", "Gammaview 2", "Deltaview 5"),
    "start quarter": ("Q1", "Q2", "Q3", "Q4"),
}
NOT_STATED_KEY = "not_stated"
NOT_STATED_DESCRIPTION = "the registry does not state this"


def make_missing_info_item(*, seed: int, item_id: str, answerable: bool,
                           gold_position: int = 0) -> dict[str, Any]:
    """Balanced registry item: the asked fact is stated or genuinely absent."""
    rng = random.Random(f"missing:{seed}:{item_id}")
    names = rng.sample(NAME_POOL, 4)
    target, *others = names
    # globally disjoint values: each (attribute, value) pair used once per item
    attribute = rng.choice(sorted(ATTRIBUTES))
    values = list(ATTRIBUTES[attribute])
    rng.shuffle(values)
    stated: list[str] = []
    for name in names:
        stated.append(values[len(stated) % len(values)])
    registry_lines = []
    for name, value in zip(names, stated):
        registry_lines.append(f"{name}: {attribute} {value}")
    if answerable:
        correct_value = stated[0]
        registry_lines[0] = f"{target}: {attribute} {correct_value}"
    else:
        # the target's fact is absent from the registry entirely
        registry_lines[0] = f"{target}: on leave this quarter"
    rng.shuffle(registry_lines)
    state = "Office registry:\n" + "\n".join(f"- {line}" for line in registry_lines)

    if answerable:
        wrong_values = [v for v in values if v != correct_value][:2]
        contents = [correct_value, *wrong_values, NOT_STATED_DESCRIPTION]
    else:
        contents = [*values[:3], NOT_STATED_DESCRIPTION]
    option_keys = [f"opt_{i}" for i in range(len(contents))]
    rng.shuffle(option_keys)
    gold_key = option_keys[contents.index(correct_value if answerable else NOT_STATED_DESCRIPTION)]
    # Balance the gold position AFTER serialisation: criteria pairs keys with
    # contents by position and the gold content starts at index 0, so rotate
    # the CONTENT list so the gold content lands exactly on the requested
    # round-robin position.
    n_options = len(contents)
    gold_content = correct_value if answerable else NOT_STATED_DESCRIPTION
    r = (contents.index(gold_content) - gold_position) % n_options
    contents = contents[r:] + contents[:r]
    criteria = dict(zip(option_keys, contents))
    gold_key = next(k for k, v in criteria.items()
                    if v == (correct_value if answerable else NOT_STATED_DESCRIPTION))
    item = {
        "id": item_id,
        "group": "fresh_v3_missing_info",  # leakage_check disabled below; see comment
        "cluster": f"fresh_v3_missing:{seed}",
        "condition": f"role=fresh_v3;family=missing_info;answerable={int(answerable)};seed={seed}",
        "leakage_check": False,  # stated values legitimately appear in the state
        "state": state,
        "gold": {
            "answer": {"value": gold_key},
            "option_values": {key: value for key, value in zip(option_keys, contents)},
            "target_entity": target,
            "attribute": attribute,
            "answerable": answerable,
        },
        "questions": {
            "answer": {
                "type": "choice",
                "instructions": (
                    f"According only to the registry, what is the {attribute} of {target}? "
                    "Choose exactly one."
                ),
                "criteria": criteria,
            },
        },
    }
    return item


# ------------------------------------------------------------------ dates family
def make_dates_item(*, seed: int, item_id: str, start_day: int, gap: int) -> dict[str, Any]:
    """30-day-month calendar item; label verified by two independent methods."""
    if not 1 <= start_day <= DAYS_IN_MONTH:
        raise ValueError(f"start_day {start_day} outside 1..{DAYS_IN_MONTH}")
    same_month = (start_day + gap) <= DAYS_IN_MONTH  # integer-arithmetic label
    rng = random.Random(f"dates:{seed}:{item_id}")
    opening = rng.choice((
        f"Planning calendar rules: every month in this calendar has exactly {DAYS_IN_MONTH} days, "
        f"numbered 1 to {DAYS_IN_MONTH}.",
        f"Calendar convention: each month lasts exactly {DAYS_IN_MONTH} days (day 1 through day "
        f"{DAYS_IN_MONTH}); the next day after day {DAYS_IN_MONTH} is day 1 of the following month.",
        f"This planner uses uniform months of {DAYS_IN_MONTH} days; there are no shorter or longer months.",
    ))
    meeting = rng.choice((
        f"A review meeting is scheduled for day {start_day} of Month 1.",
        f"The review meeting takes place on day {start_day} of Month 1.",
        f"Month 1, day {start_day}: review meeting.",
    ))
    audit = rng.choice((
        f"The follow-up audit is scheduled exactly {gap} days after the review meeting.",
        f"Exactly {gap} days after the review meeting, the follow-up audit happens.",
        f"The follow-up audit follows the review meeting by exactly {gap} days.",
    ))
    state = f"{opening} {meeting} {audit}"
    item = {
        "id": item_id,
        "group": "fresh_v3_dates",
        "cluster": f"fresh_v3_dates:{seed}",
        "condition": f"role=fresh_v3;family=synthetic_dates;start={start_day};gap={gap};seed={seed}",
        "state": state,
        "leakage_check": False,  # synthetic items: calendar text is the only content
        "gold": {
            "answer": {"value": same_month},
            "start_day": start_day,
            "gap_days": gap,
            "calendar": f"uniform {DAYS_IN_MONTH}-day months",
        },
        "questions": {
            "answer": {
                "type": "noul",
                "instructions": (
                    "Based only on the calendar rules stated, answer yes or no: does the "
                    "follow-up audit fall in the same month as the review meeting?"
                ),
            },
        },
    }
    return item



# ------------------------------------------------------------------ coverage
def dates_coverage_table(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Conditional coverage: same-month / crossing cells that actually exist.

    Impossible cells (e.g. gap 50 with a same-month outcome in a 30-day month)
    are recorded as impossible, never asserted as covered.
    """
    cells: dict[tuple[int, bool], int] = {}
    for item in items:
        gold = item["gold"]
        cells[(gold["gap_days"], bool(gold["answer"]["value"]))] = (
            cells.get((gold["gap_days"], bool(gold["answer"]["value"])), 0) + 1
        )
    table: dict[str, Any] = {"cells": {}, "impossible_cells": []}
    for gap in DATE_GAPS:
        same = cells.get((gap, True), 0)
        cross = cells.get((gap, False), 0)
        possible = gap < DAYS_IN_MONTH  # same-month needs start_day <= 30 - gap >= 1
        table["cells"][f"gap={gap}"] = {
            "same_month": same, "crossing_month": cross,
            "same_month_mathematically_possible": possible,
        }
        if not possible:
            table["impossible_cells"].append(
                f"gap={gap}: a {gap}-day gap from any valid start day (1..{DAYS_IN_MONTH}) "
                f"always crosses the month boundary in a {DAYS_IN_MONTH}-day month"
            )
    return table


# ------------------------------------------------------------------ spec builder
def fresh_v3_items(*, family: str, n: int, seed: int, split: str,
                   avoid_signatures: set[str] | None = None) -> list[dict[str, Any]]:
    """Generate `n` items for one family/split with per-item reproducible seeds.

    Items whose substantive signature already occurs in `avoid_signatures`
    (e.g. the development split) are re-drawn with a deterministic retry salt,
    so the development/test split stays structurally disjoint by construction.
    """
    avoid = set(avoid_signatures or ())
    items: list[dict[str, Any]] = []
    if family == "graph_two_hop":
        for i in range(n):
            items.append(_retry_novel(
                lambda item_id, pos=i: make_graph_item(seed=seed, item_id=item_id, gold_position=pos),
                base_id=f"freshv3-{split}-graph-{i:03d}", family=family, avoid=avoid))
    elif family == "missing_info":
        n_answerable = n // 2
        for i in range(n):
            answerable = i < n_answerable
            items.append(_retry_novel(
                lambda item_id, a=answerable, pos=i: make_missing_info_item(
                    seed=seed, item_id=item_id, answerable=a, gold_position=pos),
                base_id=f"freshv3-{split}-missing-{i:03d}", family=family, avoid=avoid))
    elif family == "synthetic_dates":
        cells = _dates_cell_plan(n, seed=seed)
        for i, (start_day, gap) in enumerate(cells):
            items.append(make_dates_item(
                seed=seed, item_id=f"freshv3-{split}-dates-{i:03d}",
                start_day=start_day, gap=gap,
            ))
    else:
        raise ValueError(f"unknown family {family!r}")
    return items


def _retry_novel(make: Any, *, base_id: str, family: str, avoid: set[str],
                 max_attempts: int = 200) -> dict[str, Any]:
    """Draw items until the substantive signature is novel within `avoid`."""
    for attempt in range(max_attempts):
        item_id = base_id if attempt == 0 else f"{base_id}#r{attempt}"
        item = make(item_id)
        item["id"] = base_id
        signature = substantive_signature(item, family=family)
        if signature not in avoid:
            avoid.add(signature)
            return item
    raise ValueError(f"could not draw a structurally novel item for {base_id}")


def _dates_cell_plan(n: int, *, avoid: set[tuple[int, int]] | None = None,
                     seed: int = 0) -> list[tuple[int, int]]:
    """Deterministic, seed-dependent (start_day, gap) plan covering both outcomes.

    The seed drives the quantitative cell sampling, so different seeds draw
    different (start, gap) cells — date items genuinely vary with their seed
    beyond surface phrasing. Same-month and crossing cells stay balanced and
    `avoid` keeps the development/test cell sets disjoint.
    """
    rng = random.Random(f"dates-cells:{seed}")
    same_cells = [(start, gap) for gap in DATE_GAPS if gap < DAYS_IN_MONTH
                  for start in range(1, DAYS_IN_MONTH - gap + 1)]
    cross_cells = [(start, gap) for gap in DATE_GAPS
                   for start in range(max(1, DAYS_IN_MONTH - gap + 1), DAYS_IN_MONTH + 1)]
    pool_same = [c for c in same_cells if c not in (avoid or set())]
    pool_cross = [c for c in cross_cells if c not in (avoid or set())]
    n_same = n // 2
    n_cross = n - n_same
    same = rng.sample(pool_same, min(n_same, len(pool_same)))
    cross = rng.sample(pool_cross, min(n_cross, len(pool_cross)))
    plan = same + cross
    rng.shuffle(plan)
    while len(plan) < n:  # top up if the pools were smaller than requested
        plan.append(rng.choice(same_cells + cross_cells))
    return plan[:n]


FAMILIES = ("graph_two_hop", "missing_info", "synthetic_dates")


def fresh_v3_spec(
    *,
    dev_n: int = 24,
    test_n: int = 120,
    dev_seed: int = 101,
    test_seed: int = 4242,
    model: str = "jev-1.13.0",
) -> dict[str, Any]:
    """24 DEVELOPMENT + 120 TEST items across the three families.

    DEV items may be used for prompt tuning; TEST items are held out and are
    never used for tuning. Substantive signatures of the two splits are
    asserted disjoint at build time.
    """
    per_family_dev = dev_n // len(FAMILIES)
    per_family_test = test_n // len(FAMILIES)
    items: list[dict[str, Any]] = []
    per_family_seeds: dict[str, dict[str, int]] = {}
    # dates: the test plan avoids the development cells entirely; each split's
    # cells are sampled from its own seed
    dev_date_cells = set(_dates_cell_plan(per_family_dev, seed=dev_seed))
    for family in FAMILIES:
        per_family_seeds[family] = {"dev": dev_seed, "test": test_seed}
        dev = fresh_v3_items(family=family, n=per_family_dev, seed=dev_seed, split="dev")
        if family == "synthetic_dates":
            test = _dates_items_avoiding_cells(
                n=per_family_test, seed=test_seed, split="test", avoid_cells=dev_date_cells)
        else:
            dev_sigs = {substantive_signature(item, family=family) for item in dev}
            test = fresh_v3_items(family=family, n=per_family_test, seed=test_seed,
                                  split="test", avoid_signatures=dev_sigs)
        items.extend(dev)
        items.extend(test)

    dev_sigs = {substantive_signature(item, family=_family_of(item)) for item in items if "-dev-" in item["id"]}
    test_sigs = {substantive_signature(item, family=_family_of(item)) for item in items if "-test-" in item["id"]}
    overlap = dev_sigs & test_sigs
    if overlap:
        raise ValueError(f"development/test substantive signatures overlap: {len(overlap)} shared")

    spec = {
        "experiment": "fresh-capability-v3",
        "model": model,
        "test_family": "capability-fresh-v3",
        "seeds": {
            "order": test_seed,
            "dev_seed": dev_seed,
            "test_seed": test_seed,
            "per_family": per_family_seeds,
        },
        "shuffle": True,
        "claim_type": "exploratory",
        "generator_version": GENERATOR_VERSION,
        "dataset": {
            "name": "fresh-capability-v3",
            "source_url": "generated locally",
            "license_note": "generated; no redistribution restrictions",
            "n_population": len(items),
            "sampling": {
                "mode": "fresh_v3_dev_test",
                "dev_n": per_family_dev * len(FAMILIES),
                "test_n": per_family_test * len(FAMILIES),
                "note": "DEV items are for prompt tuning only; TEST items are held out from tuning",
                "generator_version": GENERATOR_VERSION,
            },
        },
        "items": items,
    }
    return spec


def _dates_items_avoiding_cells(*, n: int, seed: int, split: str,
                                avoid_cells: set[tuple[int, int]]) -> list[dict[str, Any]]:
    cells = _dates_cell_plan(n, avoid=avoid_cells, seed=seed)
    overlap = avoid_cells & set(cells)
    if overlap:
        raise ValueError(f"dates dev/test cells overlap: {sorted(overlap)}")
    return [make_dates_item(seed=seed, item_id=f"freshv3-{split}-dates-{i:03d}",
                            start_day=start, gap=gap)
            for i, (start, gap) in enumerate(cells)]


def _family_of(item: dict[str, Any]) -> str:
    group = item["group"]
    return {
        "fresh_v3_graph": "graph_two_hop",
        "fresh_v3_missing_info": "missing_info",
        "fresh_v3_dates": "synthetic_dates",
    }[group]


# ------------------------------------------------------------------ validation
class FreshSpecError(ValueError):
    """A fresh-v3 spec failed a design-quality check (fault caught, not silent)."""


def _serialized_gold_position(item: dict[str, Any]) -> int:
    """Position of the gold key in the criteria dict AFTER serialisation."""
    gold_key = item["gold"]["answer"]["value"]
    criteria = item["questions"]["answer"]["criteria"]
    return list(criteria).index(gold_key)


def validate_fresh_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Run every design-quality check; raise FreshSpecError on any violation.

    Checks (LEAD-GATE 'tests must FAIL on' list, requirement B):
    * seed sensitivity — a family whose substantive signatures collapse when
      the seed changes is refused (rename-only variation);
    * missing-info balance — not every answer is absent, not every answer is
      stated, and the explicit not-stated option is neither always nor never
      correct;
    * gold position balance — gold positions are balanced across the
      serialised option order, never fixed;
    * graph answer independence — option text carries no structural role
      tokens, and all labels are re-verified by the independent verifier;
    * development/test substantive-signature disjointness.
    """
    from . import fresh_v3_verify as verifier

    findings: dict[str, Any] = {}
    items = spec["items"]
    by_family: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        by_family.setdefault(_family_of(item), []).append(item)

    # 1. missing-info balance
    missing_items = by_family.get("missing_info", [])
    n_answerable = sum(1 for i in missing_items if i["gold"]["answerable"])
    if missing_items:
        share = n_answerable / len(missing_items)
        if not 0.4 <= share <= 0.6:
            raise FreshSpecError(
                f"missing-info answerable share {share:.2f} outside the required balance window"
            )
        not_stated_gold = sum(
            1 for i in missing_items if not i["gold"]["answerable"]
        )
        if not_stated_gold == 0 or not_stated_gold == len(missing_items):
            raise FreshSpecError("not_stated must not be always correct nor never correct")
    findings["missing_info_answerable_share"] = (
        n_answerable / len(missing_items) if missing_items else None
    )

    # 2. gold position balance over all choice items
    choice_items = [i for i in items if i["questions"]["answer"]["type"] == "choice"]
    by_option_count: dict[int, list[dict[str, Any]]] = {}
    for item in choice_items:
        by_option_count.setdefault(len(item["questions"]["answer"]["criteria"]), []).append(item)
    positions_by_group: dict[str, dict[int, int]] = {}
    for n_options, group_items in sorted(by_option_count.items()):
        positions: dict[int, int] = {}
        for item in group_items:
            pos = _serialized_gold_position(item)
            positions[pos] = positions.get(pos, 0) + 1
        positions_by_group[f"{n_options}_options"] = positions
        if sorted(positions) != list(range(n_options)):
            raise FreshSpecError(
                f"gold position fixed or partially unused: {positions} over {n_options} options"
            )
        worst_share = max(positions.values()) / len(group_items)
        if worst_share > 0.5:
            raise FreshSpecError(f"gold position concentrated: {positions}")
    findings["gold_position_counts"] = positions_by_group

    # 3. graph role leak + full independent label verification
    verification = verifier.verify_all(items)
    findings["verification"] = {
        "n_items": verification["n_items"],
        "n_failures": verification["n_failures"],
        "verifier": verification["verifier"],
    }
    if not verification["all_labels_verified"]:
        raise FreshSpecError(f"independent verification failed: {verification['failures'][:3]}")

    # 4. seed sensitivity: signatures must not collapse to rename-only variation
    per_family_distinct = {}
    for family, family_items in by_family.items():
        sigs = {substantive_signature(i, family=family) for i in family_items}
        per_family_distinct[family] = len(sigs)
        if family_items and len(sigs) < 0.8 * len(family_items):
            raise FreshSpecError(
                f"family {family} shows rename-only variation: {len(sigs)} distinct "
                f"substantive signatures for {len(family_items)} items"
            )
    findings["distinct_signatures_per_family"] = per_family_distinct

    # 5. development/test substantive signature disjointness
    dev_sigs = {substantive_signature(i, family=_family_of(i)) for i in items if "-dev-" in i["id"]}
    test_sigs = {substantive_signature(i, family=_family_of(i)) for i in items if "-test-" in i["id"]}
    overlap = dev_sigs & test_sigs
    if overlap:
        raise FreshSpecError(f"development/test substantive signatures overlap: {len(overlap)}")
    findings["dev_signatures"] = len(dev_sigs)
    findings["test_signatures"] = len(test_sigs)
    findings["dev_test_signature_overlap"] = 0
    return findings


def seed_sensitivity_report(family: str, *, seeds: list[int], n: int = 12) -> dict[str, Any]:
    """Distinct-signature fraction for one family across several seeds.

    Used by tests with fault injection: a generator that ignores its seed (or
    only renames entities) collapses to ~1 signature and is reported here.
    """
    sigs: set[str] = set()
    per_seed: dict[int, int] = {}
    for seed in seeds:
        items = fresh_v3_items(family=family, n=n, seed=seed, split="probe")
        seed_sigs = {substantive_signature(i, family=family) for i in items}
        per_seed[seed] = len(seed_sigs)
        sigs |= seed_sigs
    return {
        "family": family,
        "seeds": seeds,
        "n_per_seed": n,
        "distinct_signatures_per_seed": per_seed,
        "distinct_signatures_total": len(sigs),
        "rename_only_suspected": len(sigs) <= max(1, len(seeds) // 2),
    }
