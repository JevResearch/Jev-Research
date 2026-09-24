"""Option-permutation variants: equivariance checks with recorded remappings.

A permuted item is a *separate condition* (DESIGN.md §3B: label-order
transformations reduce superficial memorization but never replace the base
item).  Permutation maps are recorded at spec level for provenance.
"""

from __future__ import annotations

import random
from typing import Any


def permute_choice_item(item: dict[str, Any], *, seed: int, variant: int = 1) -> dict[str, Any]:
    """Return a copy of a choice item with option descriptions reassigned to keys.

    Option keys keep their meaning only through their descriptions, so the
    permutation moves descriptions between keys and remaps the gold key.
    Non-choice items are returned unchanged (callers filter).
    """
    questions = item.get("questions", {})
    if len(questions) != 1:
        raise ValueError("permutation expects exactly one question per item")
    qid, question = next(iter(questions.items()))
    if question.get("type") != "choice":
        raise ValueError("permutation applies to choice questions only")

    criteria: dict[str, Any] = question["criteria"]
    keys = list(criteria)
    descriptions = [criteria[key] for key in keys]
    rng = random.Random(f"{seed}:{item['id']}:{variant}")
    shuffled = descriptions[:]
    # Reject identity permutations so the variant is a real test.
    while len(set(descriptions)) > 1 and shuffled == descriptions:
        rng.shuffle(shuffled)

    new_criteria = {key: desc for key, desc in zip(keys, shuffled)}

    # gold must follow its description to the new key
    gold_key = _gold_key(item, qid)
    new_gold_key = next(key for key, desc in new_criteria.items() if desc == criteria[gold_key])

    remapping = {key: next(k for k, d in new_criteria.items() if d == criteria[key]) for key in keys}
    permuted = {
        **{k: v for k, v in item.items() if k not in {"id", "condition", "questions", "gold"}},
        "id": f"{item['id']}:perm{variant}",
        "condition": f"{item.get('condition', 'native')}:perm{variant}",
        "questions": {qid: {**question, "criteria": new_criteria}},
        "gold": {qid2: value for qid2, value in (item.get("gold") or {}).items()},
    }
    permuted["gold"] = {qid: {"value": new_gold_key}}
    return {"item": permuted, "remapping": remapping}


def _gold_key(item: dict[str, Any], qid: str) -> str:
    gold = item.get("gold", {}).get(qid)
    if isinstance(gold, dict):
        return gold.get("value") or gold.get("key")
    return gold


def attach_permutations(spec: dict[str, Any], *, seed: int, variants: int = 1) -> dict[str, Any]:
    """Extend a spec with permuted variants of every single-question choice item.

    The spec gains a top-level `permutations` record; variant items share the
    base item's cluster so paired analyses group them correctly.
    """
    import copy

    out = copy.deepcopy(spec)
    records: dict[str, Any] = {}
    extra_items: list[dict[str, Any]] = []
    for item in out.get("items", []):
        try:
            for variant in range(1, variants + 1):
                result = permute_choice_item(item, seed=seed, variant=variant)
                extra_items.append({k: v for k, v in result["item"].items() if not k.startswith("_")})
                records[result["item"]["id"]] = {
                    "base_item": item["id"],
                    "remapping": result["remapping"],
                }
        except ValueError:
            continue  # non-choice or multi-question items get no variants
    out["items"] = list(out.get("items", [])) + extra_items
    out["permutations"] = records
    return out