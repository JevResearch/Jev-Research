"""Independent verification of fresh-v3 labels (LEAD-GATE requirement B).

Deliberately implemented WITHOUT importing the generator's gold-producing
logic: labels are re-derived from the serialized item content (state text and
question criteria) by separate algorithms and then compared with the recorded
gold. Any mismatch raises ``VerificationError``.

Three verifiers:

* ``verify_graph_item`` — parses the state's edge list back into a graph and
  runs a breadth-first traversal to find the node(s) at distance exactly two
  from the target. The gold key must BE that unique node, the distractor role
  must sit at distance one, and the negative role must be at distance > 2
  (disconnected from the target).
* ``verify_missing_info_item`` — reconstructs the registry from the state text
  and re-derives whether the target's fact is stated; the gold key must map to
  the stated value or to the explicit not-stated option exactly when the fact
  is absent.
* ``verify_dates_item`` — re-derives the same-month label two independent
  ways: closed-form integer day arithmetic and an incremental day-stepping
  simulation with month rollover at 30. Both must agree with the gold.

Role-leak guard: option descriptions must not contain structural role tokens
("two", "adjacent", "hop", ...), so the graph answer can never be read off the
option text.
"""

from __future__ import annotations

import re
from typing import Any

from .fresh_v3 import DAYS_IN_MONTH, GRAPH_ROLE_TOKENS, NOT_STATED_DESCRIPTION


class VerificationError(AssertionError):
    """An independently re-derived label disagrees with the recorded gold."""


# ------------------------------------------------------------------ graph
def parse_edges(state: str) -> list[tuple[str, str]]:
    """Parse 'a–b' pairs out of the directory state text (own regex, own logic)."""
    edges: list[tuple[str, str]] = []
    for a, b in re.findall(r"([A-ZÀ-Ža-z]+)–([A-ZÀ-Ža-z]+)", state):
        edges.append((a, b))
    if not edges:
        raise VerificationError("graph state contained no parseable edges")
    return edges


def parse_graph_target(instructions: str) -> str:
    """Parse the queried node from the OUTBOUND instructions text."""
    match = re.search(r"away from ([A-Za-z]+)", instructions)
    if not match:
        raise VerificationError("could not parse the target node from the instructions")
    return match.group(1)


def bfs_distances(edges: list[tuple[str, str]], source: str) -> dict[str, int]:
    """Plain BFS over an undirected adjacency list (independent implementation)."""
    adjacency: dict[str, set[str]] = {}
    for a, b in edges:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    distance = {source: 0}
    frontier = [source]
    level = 0
    while frontier:
        level += 1
        nxt: list[str] = []
        for node in frontier:
            for neighbour in sorted(adjacency.get(node, ())):
                if neighbour not in distance:
                    distance[neighbour] = level
                    nxt.append(neighbour)
        frontier = nxt
    return distance


def verify_graph_item(item: dict[str, Any]) -> dict[str, Any]:
    gold = item["gold"]
    # OUTBOUND CONTENT IS THE AUTHORITY: the target node is parsed from the
    # instructions, not read from gold metadata.
    target = parse_graph_target(item["questions"]["answer"]["instructions"])
    if gold.get("target_node") != target:
        raise VerificationError(
            f"gold target_node {gold.get('target_node')!r} disagrees with the target parsed "
            f"from the instructions ({target!r})"
        )
    gold_key = gold["answer"]["value"]
    distances = bfs_distances(parse_edges(item["state"]), target)
    roles = gold["role_of"]

    for key, description in item["questions"]["answer"]["criteria"].items():
        lowered = description.lower()
        for token in GRAPH_ROLE_TOKENS:
            if token in lowered:
                raise VerificationError(
                    f"option description leaks a structural role token ({token!r}): {description!r}"
                )

    exactly_two = sorted(node for node, d in distances.items() if d == 2)
    if exactly_two != [gold_key]:
        raise VerificationError(
            f"BFS says nodes at distance exactly two are {exactly_two}, gold claims {gold_key!r}"
        )
    if distances.get(roles.get("distractor")) != 1:
        raise VerificationError("distractor role is not at distance one from the target")
    if distances.get(roles.get("negative"), 10 ** 9) <= 2:
        raise VerificationError("negative role is within two edges of the target")
    return {"ok": True, "distance_two_nodes": exactly_two, "n_components_reachable": len(distances)}


# ------------------------------------------------------------ missing info
LINE_PATTERN = re.compile(r"^-\s*(?P<entity>[^:]+):\s*(?P<rest>.+)$")
INSTRUCTIONS_PATTERN = re.compile(
    r"what is the (?P<attribute>.+?) of (?P<target>[A-Za-z]+)\?", re.DOTALL
)


def parse_registry(state: str) -> dict[str, str]:
    """Parse 'entity: rest' entries out of the registry state text."""
    registry: dict[str, str] = {}
    for line in state.splitlines():
        match = LINE_PATTERN.match(line.strip())
        if match:
            registry[match.group("entity").strip()] = match.group("rest").strip()
    if not registry:
        raise VerificationError("registry state contained no parseable entries")
    return registry


def parse_missing_info_question(instructions: str) -> tuple[str, str]:
    """Parse (attribute, target entity) from the OUTBOUND instructions text."""
    match = INSTRUCTIONS_PATTERN.search(instructions)
    if not match:
        raise VerificationError(
            "could not parse attribute and target from the instructions"
        )
    return match.group("attribute").strip(), match.group("target").strip()


def verify_missing_info_item(item: dict[str, Any]) -> dict[str, Any]:
    gold = item["gold"]
    registry = parse_registry(item["state"])
    # OUTBOUND CONTENT IS THE AUTHORITY: attribute and target come from the
    # instructions; offered option contents come from the criteria.
    attribute, target = parse_missing_info_question(item["questions"]["answer"]["instructions"])
    criteria = item["questions"]["answer"]["criteria"]
    gold_key = gold["answer"]["value"]

    # metadata is evidence to cross-check, never authority
    if gold.get("target_entity") != target:
        raise VerificationError(
            f"gold target_entity {gold.get('target_entity')!r} disagrees with the target parsed "
            f"from the instructions ({target!r})"
        )
    if gold.get("attribute") != attribute:
        raise VerificationError(
            f"gold attribute {gold.get('attribute')!r} disagrees with the attribute parsed "
            f"from the instructions ({attribute!r})"
        )
    for key, value in gold.get("option_values", {}).items():
        if key in criteria and criteria[key] != value:
            raise VerificationError(
                f"gold option_values[{key!r}] disagrees with the serialized criteria"
            )

    entries = [rest for entity, rest in registry.items() if entity == target]
    if len(entries) != 1:
        raise VerificationError(f"target entity {target!r} appears {len(entries)} times")
    entry = entries[0]
    stated_value = entry[len(attribute):].strip() if entry.startswith(attribute) else None

    # derive the correct option independently from state + criteria
    if stated_value is None:
        derived_content = NOT_STATED_DESCRIPTION
    else:
        derived_content = stated_value
    matches = [key for key, content in criteria.items() if content == derived_content]
    if len(matches) != 1:
        raise VerificationError(
            f"derived correct content {derived_content!r} matches {len(matches)} options; "
            "the offered option set is ambiguous or the correct value is absent"
        )
    derived_key = matches[0]
    if derived_key != gold_key:
        raise VerificationError(
            f"derived correct option {derived_key!r} ({derived_content!r}) != gold {gold_key!r} "
            f"({criteria.get(gold_key)!r})"
        )
    # values must be globally disjoint across options
    contents = [criteria[key] for key in criteria]
    factual = [c for c in contents if c != NOT_STATED_DESCRIPTION]
    if len(set(factual)) != len(factual):
        raise VerificationError("duplicate factual option values")
    return {"ok": True, "stated_value": stated_value, "gold_key": gold_key,
            "derived_key": derived_key}


# ------------------------------------------------------------------ dates
def same_month_by_arithmetic(start_day: int, gap: int) -> bool:
    """Closed-form: label from integer day numbers."""
    return start_day + gap <= DAYS_IN_MONTH


def same_month_by_stepping(start_day: int, gap: int) -> bool:
    """Independent simulation: step one day at a time with month rollover."""
    day, month = start_day, 1
    for _ in range(gap):
        if day == DAYS_IN_MONTH:
            day, month = 1, month + 1
        else:
            day += 1
    return month == 1


def parse_dates_state(state: str) -> tuple[int, int]:
    """Parse (start_day, gap) from the OUTBOUND state text.

    All three calendar phrasings are covered; the day-of-month is the number
    tied to Month 1's review meeting and the gap is the one tied to the
    follow-up audit (the 30-day calendar length never matches these patterns).
    """
    start_match = re.search(r"day (\d+) of Month 1", state) or re.search(
        r"Month 1, day (\d+)", state
    )
    gap_match = re.search(r"exactly (\d+) days after the review meeting", state, re.IGNORECASE) \
        or re.search(r"by exactly (\d+) days", state)
    if not start_match or not gap_match:
        raise VerificationError(
            "could not parse start day and gap from the calendar state text"
        )
    return int(start_match.group(1)), int(gap_match.group(1))


def verify_dates_item(item: dict[str, Any]) -> dict[str, Any]:
    gold = item["gold"]
    # OUTBOUND CONTENT IS THE AUTHORITY: start day and gap are parsed from the
    # state text, never read from gold metadata.
    start_day, gap = parse_dates_state(item["state"])
    if gold.get("start_day") != start_day or gold.get("gap_days") != gap:
        raise VerificationError(
            f"gold metadata (start={gold.get('start_day')}, gap={gold.get('gap_days')}) "
            f"disagrees with the values parsed from the state text "
            f"(start={start_day}, gap={gap})"
        )
    by_arithmetic = same_month_by_arithmetic(start_day, gap)
    by_stepping = same_month_by_stepping(start_day, gap)
    if by_arithmetic != by_stepping:
        raise VerificationError(
            f"independent date methods disagree for start={start_day} gap={gap}: "
            f"arithmetic={by_arithmetic} stepping={by_stepping}"
        )
    if bool(gold["answer"]["value"]) != by_arithmetic:
        raise VerificationError(
            f"gold label {gold['answer']['value']!r} != independently derived {by_arithmetic!r} "
            f"(start={start_day}, gap={gap})"
        )
    return {"ok": True, "same_month": by_arithmetic, "method_agreement": True,
            "parsed_start": start_day, "parsed_gap": gap}


def verify_item(item: dict[str, Any]) -> dict[str, Any]:
    group = item["group"]
    if group == "fresh_v3_graph":
        return verify_graph_item(item)
    if group == "fresh_v3_missing_info":
        return verify_missing_info_item(item)
    if group == "fresh_v3_dates":
        return verify_dates_item(item)
    raise VerificationError(f"no verifier for group {group!r}")


def verify_all(items: list[dict[str, Any]]) -> dict[str, Any]:
    results = []
    failures = []
    for item in items:
        try:
            detail = verify_item(item)
            results.append({"id": item["id"], **detail})
        except VerificationError as exc:
            failures.append({"id": item["id"], "error": str(exc)})
    return {
        "n_items": len(items),
        "n_verified": len(results),
        "n_failures": len(failures),
        "failures": failures,
        "results": results,
        "verifier": "fresh_v3_verify-1.0.0 (independent re-derivation, not generator gold)",
        "all_labels_verified": not failures and len(results) == len(items),
    }
