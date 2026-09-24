"""Controlled mechanism dry-run design (LEAD-GATE 2026-09-18, requirement C).

12 randomized blocks × 2 already-screened anchor states × 4 arms = 96
requests. The two anchors are the same non-saturated states screened during
mechanism v3; two anchors are limited coverage, not general proof.

Arms per (anchor, block), relative to the S payload:

1. ``alone``     (A) — anchor question alone; repeated across blocks supplies
   within-anchor repeat variability;
2. ``siblings``  (S) — anchor + 8 fixed irrelevant siblings, canonical order;
3. ``renamed``   (N) — EXACTLY S with ONLY the question IDs renamed (ALL ids,
   including the anchor's): same state, same question content, same insertion
   positions. Position-preserving explicit mapping. This isolates the anchor-
   ID effect, which the sibling-only v3 design confounded;
4. ``reordered`` (O) — EXACTLY S with ONLY the question insertion order
   changed (ALL questions, anchor included, moves): same keys, same content,
   same state. This isolates the anchor-order effect.

The audited mechanism-v3 confounds are repaired by construction: in v3 the
rename arm also moved the anchor's key/position and the reorder arm renamed
the anchor. Here the arm contrasts are defined on the outbound payload and
checked structurally (see ``payload_diff``).

Within each block the 8 (anchor, arm) requests are shuffled with a recorded
seed; the exact dispatch order and all question-ID mappings are stored in an
explicit sidecar mapping document — never inferred from condition strings.

Analysis units are the paired per-block signed P(yes) shifts (one shift per
block per contrast; n = number of blocks, NOT pairwise distances). Diagnostics
are exploratory sign-flip randomisations with a Holm correction across the
2×3 contrast family. Any equivalence claim would additionally require the
ENTIRE uncertainty interval of the shift to lie inside the prespecified ±0.03
practical-tolerance window; a p-value above 0.05 is never sufficient.
"""

from __future__ import annotations

import json
import random
import statistics
from typing import Any

from .experiments import WORDS

MECH_DRYRUN_VERSION = "mech_dryrun-1.0.0"
ARMS = ("alone", "siblings", "renamed", "reordered")
PRIMARY_CONTRASTS = (("siblings", "alone"), ("renamed", "siblings"), ("reordered", "siblings"))
PRACTICAL_TOLERANCE = 0.03  # probability units; prespecified

ANCHOR_STATES = {
    # Both anchors are the EXACT already-screened states from the live screening
    # run spec (source recorded in `ANCHOR_STATE_SOURCE`; hash re-verified at
    # build time). dup-charge came from the v3 spec; subscription-proration is
    # the screen-subscription-proration item (Ticket 220 downgrade/credit).
    "dup-charge": (
        "Ticket 201: The customer was charged twice for order 7712 (29.99 EUR each) "
        "and asks us to fix it."
    ),
    "subscription-proration": (
        "Ticket 220: After downgrading mid-cycle, the customer asks whether the "
        "unused portion will be credited."
    ),
}
ANCHOR_STATE_SOURCE_FILE = "runs_live_spec_screen.json"
ANCHOR_STATE_SOURCE_ITEM = "screen-subscription-proration"
ANCHOR_QUESTION = {
    "type": "choice",
    "instructions": "Does the state describe a refund request? Choose exactly one.",
    "criteria": {"yes": "mentions a refund", "no": "does not mention a refund"},
}


def _sibling_questions(rng: random.Random, *, n: int = 8) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for sib in range(n):
        word = rng.choice(WORDS)
        if sib % 3 == 0:
            out[f"s{sib:03d}"] = {"type": "noul", "instructions": f"Does the state mention the word {word}?"}
        elif sib % 3 == 1:
            out[f"s{sib:03d}"] = {
                "type": "choice",
                "instructions": f"Which {word} category fits best?",
                "criteria": {"alpha": "first", "beta": "second", "gamma": "third"},
            }
        else:
            out[f"s{sib:03d}"] = {
                "type": "score",
                "instructions": f"Rate relevance to {word}.",
                "criteria": ["irrelevant", "tangential", "central"],
            }
    return out


MAPPING_REQUIRED_FIELDS = ("item_id", "block", "anchor", "arm", "dispatch_index", "question_order")


def validate_item_mapping(entry: dict[str, Any]) -> bool:
    """Explicit mapping integrity guard: block metadata must be present data,
    never inferred from condition strings."""
    for field in MAPPING_REQUIRED_FIELDS:
        if field not in entry:
            raise ValueError(f"mech dry-run item mapping missing required field: {field}")
    if entry["arm"] not in ARMS:
        raise ValueError(f"unknown arm in mapping: {entry['arm']!r}")
    if entry["anchor"] not in ANCHOR_STATES:
        raise ValueError(f"unknown anchor in mapping: {entry['anchor']!r}")
    return True


def measured_anchor_id(mappings: dict[str, Any], item_id: str) -> str:
    """The question id carrying the anchor prediction for one item, from the
    explicit sidecar mapping — never inferred from condition strings."""
    for entry in mappings["items"]:
        if entry["item_id"] == item_id:
            return entry["measured_anchor_id"]
    raise ValueError(f"no mapping entry for {item_id!r}")


def rename_sibling_ids(questions: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
    """Rename keys via an explicit recorded mapping, preserving insertion order."""
    out: dict[str, Any] = {}
    for qid, question in questions.items():
        out[mapping.get(qid, qid)] = question  # same question object, new key, same position
    return out


def reorder_questions(questions: dict[str, Any], order: list[str]) -> dict[str, Any]:
    """Rebuild the questions dict in an explicit recorded order (same keys)."""
    if sorted(order) != sorted(questions):
        raise ValueError("reorder arm must use exactly the original question ids")
    return {qid: questions[qid] for qid in order}


def build_mech_dryrun_spec(
    *,
    blocks: int = 12,
    anchors: tuple[str, ...] = ("dup-charge", "subscription-proration"),
    seed: int = 4200,
    model: str = "jev-1.13.0",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (spec, sidecar_mappings) for the 12×2×4 controlled design.

    Arm contracts (parent decision, LEAD-GATE re-review):
    * ``renamed`` (N): ALL question IDs are renamed, INCLUDING the anchor's,
      via an explicit position-preserving mapping — same insertion positions,
      same question content, same state. This isolates the anchor-ID effect,
      which the sibling-only v3 design confounded.
    * ``reordered`` (O): ALL questions are reordered, INCLUDING moving the
      anchor — all IDs and content retained. This isolates the anchor-order
      effect.
    The sidecar records ``measured_anchor_id`` per item so predictions can be
    extracted from the correct key in every arm (never by string parsing).
    """
    import hashlib
    from pathlib import Path

    rng = random.Random(f"mech-dryrun:{seed}")
    per_anchor_siblings = {
        anchor: _sibling_questions(random.Random(f"mech-dryrun-sibs:{seed}:{anchor}"))
        for anchor in anchors
    }
    # position-preserving rename map: ALL ids renamed, anchor included, and the
    # rebuilt dict iterates the ORIGINAL insertion order so positions never move
    per_anchor_rename = {
        anchor: {"anchor": "renamed_anchor",
                 **{qid: f"q_{i:05d}" for i, qid in enumerate(sibs)}}
        for anchor, sibs in per_anchor_siblings.items()
    }
    # verify the anchor-state source file hash (evidence, not authority)
    source_hash = None
    source_path = Path(ANCHOR_STATE_SOURCE_FILE)
    if source_path.exists():
        source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    items: list[dict[str, Any]] = []
    mappings: dict[str, Any] = {
        "design": "mech_dryrun_12x2x4",
        "version": MECH_DRYRUN_VERSION,
        "blocks": blocks,
        "anchors": list(anchors),
        "arms": list(ARMS),
        "order_seed": seed,
        "arm_contracts": {
            "renamed": "ALL question ids renamed (anchor included); positions and content preserved",
            "reordered": "ALL questions reordered (anchor moved); all ids and content retained",
        },
        "anchor_state_source": {
            "file": ANCHOR_STATE_SOURCE_FILE,
            "item_id": ANCHOR_STATE_SOURCE_ITEM,
            "sha256": source_hash,
            "note": "states are exact copies of the screened items; hash recorded for traceability",
        },
        "sibling_rename_maps": per_anchor_rename,
        "items": [],
    }
    for block in range(blocks):
        units = [(anchor, arm) for anchor in anchors for arm in ARMS]
        rng.shuffle(units)  # within-block dispatch randomization, recorded below
        for dispatch_index, (anchor, arm) in enumerate(units):
            state = ANCHOR_STATES[anchor]
            siblings = per_anchor_siblings[anchor]
            if arm == "alone":
                questions = {"anchor": dict(ANCHOR_QUESTION)}
                measured = "anchor"
            elif arm == "siblings":
                questions = {"anchor": dict(ANCHOR_QUESTION), **siblings}
                measured = "anchor"
            elif arm == "renamed":
                # ALL ids renamed (anchor included) via the explicit map;
                # rebuild in ORIGINAL insertion order so positions are preserved
                questions = rename_sibling_ids(
                    {"anchor": dict(ANCHOR_QUESTION), **siblings},
                    per_anchor_rename[anchor],
                )
                measured = per_anchor_rename[anchor]["anchor"]
            elif arm == "reordered":
                all_ids = ["anchor", *[f"s{i:03d}" for i in range(len(siblings))]]
                shuffled = list(all_ids)
                random.Random(f"mech-dryrun-reorder:{seed}:{anchor}:{block}").shuffle(shuffled)
                # ALL questions reordered (anchor moves); ids/content retained
                questions = reorder_questions(
                    {"anchor": dict(ANCHOR_QUESTION), **siblings}, shuffled
                )
                measured = "anchor"
            else:
                raise ValueError(f"unknown arm {arm!r}")
            item_id = f"m4-{anchor}-b{block:02d}-{arm}"
            item = {
                "id": item_id,
                "group": "mech_dryrun",
                "cluster": f"m4-{anchor}-b{block:02d}",
                "condition": f"role=mech4;anchor={anchor};arm={arm};block={block}",
                "state": state,
                "gold": {},
                "questions": questions,
            }
            items.append(item)
            mappings["items"].append({
                "item_id": item_id,
                "block": block,
                "anchor": anchor,
                "arm": arm,
                "dispatch_index": dispatch_index,
                "question_order": list(questions),
                "measured_anchor_id": measured,
                "renamed_ids": dict(per_anchor_rename[anchor]) if arm == "renamed" else {},
            })
    spec = {
        "experiment": "mechanism-dryrun-12x2x4",
        "model": model,
        "test_family": "mechanism-controlled-v4",
        "seeds": {"order": seed},
        "shuffle": False,  # dispatch order is the recorded block layout
        "claim_type": "exploratory",
        "generator_version": MECH_DRYRUN_VERSION,
        "dataset": {
            "name": "mechanism-dryrun-12x2x4",
            "source_url": "generated locally",
            "license_note": "generated",
            "n_population": len(items),
            "sampling": {
                "mode": "controlled_anchors_v4",
                "blocks": blocks,
                "anchors": list(anchors),
                "arms": list(ARMS),
                "note": "two already-screened anchors; limited coverage, not general proof",
                "mappings_sidecar": "mech_dryrun_mappings.json",
            },
        },
        "items": items,
    }
    return spec, mappings


# ---------------------------------------------------- payload order inspection
def ordered_payload_json(payload: dict[str, Any]) -> str:
    """The outbound wire representation: insertion order preserved, keys NOT sorted.

    Canonical content hashes may sort keys; the outbound payload representation
    must not. Any code that serialises the payload with ``sort_keys=True``
    destroys the question insertion order and is detected by
    :func:`serialization_preserves_order`.
    """
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def order_signature(obj: Any) -> tuple:
    """Recursive key/list order signature (content values are abstracted away)."""
    if isinstance(obj, dict):
        return ("dict", tuple(str(k) for k in obj),
                tuple(order_signature(v) for v in obj.values()))
    if isinstance(obj, (list, tuple)):
        return ("list", tuple(order_signature(v) for v in obj))
    return ("scalar",)


def serialization_preserves_order(payload: dict[str, Any], serialized: str) -> bool:
    """True iff the serialized text preserves the payload's insertion order."""
    reparsed = json.loads(serialized)
    return order_signature(reparsed) == order_signature(payload)


def payload_diff(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Structural diff of two outbound payloads (state, ids, order, content)."""
    diff: dict[str, Any] = {
        "state_identical": a.get("state") == b.get("state"),
        "model_identical": a.get("model") == b.get("model"),
        "question_keys_a": list(a.get("questions", {})),
        "question_keys_b": list(b.get("questions", {})),
    }
    keys_a = list(a.get("questions", {}))
    keys_b = list(b.get("questions", {}))
    diff["key_multiset_identical"] = sorted(keys_a) == sorted(keys_b)
    diff["key_order_identical"] = keys_a == keys_b
    qa, qb = a.get("questions", {}), b.get("questions", {})
    shared = set(keys_a) & set(keys_b)
    diff["shared_content_identical"] = all(qa[q] == qb[q] for q in shared)
    diff["content_of_missing_in_b"] = [q for q in keys_a if q not in keys_b]
    diff["content_of_missing_in_a"] = [q for q in keys_b if q not in qa]
    diff["order_signature_identical"] = order_signature(a) == order_signature(b)
    return diff


def assert_renamed_only(a: dict[str, Any], b: dict[str, Any]) -> None:
    """Raise unless b is exactly a with ONLY question IDs renamed.

    Renaming must preserve insertion positions, question content, and state.
    The key multiset necessarily changes (that is what renaming is); what must
    NOT change is which content sits at which position.
    """
    diff = payload_diff(a, b)
    problems = []
    if not diff["state_identical"]:
        problems.append("state changed")
    keys_a = diff["question_keys_a"]
    keys_b = diff["question_keys_b"]
    if len(keys_a) != len(keys_b) or len(keys_a) == 0:
        problems.append("question count changed")
    else:
        qa, qb = a["questions"], b["questions"]
        if list(qa.values()) != list(qb.values()):
            problems.append("question content or its order changed")
        renamed = [old for old, new in zip(keys_a, keys_b) if old != new]
        if not renamed:
            problems.append("no question id was renamed at all")
    if problems:
        raise AssertionError(f"renamed arm is not rename-only: {'; '.join(problems)} ({diff})")


def assert_reordered_only(a: dict[str, Any], b: dict[str, Any]) -> None:
    """Raise unless b is exactly a with ONLY the question order changed."""
    diff = payload_diff(a, b)
    problems = []
    if not diff["state_identical"]:
        problems.append("state changed")
    if not diff["key_multiset_identical"]:
        problems.append("question ids changed while reordering")
    if diff["key_order_identical"]:
        problems.append("question order did not change at all")
    if not diff["shared_content_identical"]:
        problems.append("question content changed")
    if problems:
        raise AssertionError(f"reordered arm is not reorder-only: {'; '.join(problems)} ({diff})")


# ------------------------------------------------------------------- analysis
def paired_signed_shifts(
    p_yes_by_block: dict[int, float],
    reference_by_block: dict[int, float],
    *,
    n_blocks: int,
) -> list[dict[str, Any]]:
    """One signed shift per block — never pairwise distances between repeats.

    Guards against pseudoreplication: refuses more than one observation per
    block, refuses a sample larger than the design's block count, and FAILS
    CLOSED on out-of-range or non-finite probabilities (they are never
    silently included in a shift).
    """
    if len(p_yes_by_block) > n_blocks or len(reference_by_block) > n_blocks:
        raise ValueError(
            f"more observations ({len(p_yes_by_block)}/{len(reference_by_block)}) than "
            f"blocks ({n_blocks}): pairwise-distance pseudoreplication is not permitted"
        )
    import math

    for label, values in (("treatment", p_yes_by_block), ("reference", reference_by_block)):
        for block, p in values.items():
            if not isinstance(p, (int, float)) or isinstance(p, bool) \
                    or not math.isfinite(p) or not 0.0 <= p <= 1.0:
                raise ValueError(
                    f"fail-closed: {label} P(yes) for block {block} is out of range or "
                    f"non-finite: {p!r}"
                )
    shifts = []
    for block in sorted(set(p_yes_by_block) & set(reference_by_block)):
        shifts.append({
            "block": block,
            "shift_signed": p_yes_by_block[block] - reference_by_block[block],
        })
    return shifts


def sign_flip_diagnostic(shifts: list[dict[str, Any]]) -> dict[str, Any]:
    """Exploratory exact sign-flip randomisation over paired block shifts."""
    diffs = [entry["shift_signed"] for entry in shifts]
    if not diffs:
        return {"n_pairs": 0, "mean_signed_shift": None, "p_sign_flip": None}
    observed = statistics.mean(diffs)
    count = 0
    total = 2 ** len(diffs)
    if total > 2 ** 20:
        raise ValueError("too many blocks for an exact sign-flip enumeration")
    for mask in range(total):
        sample = statistics.mean(d if (mask >> i) & 1 == 0 else -d for i, d in enumerate(diffs))
        if abs(sample) >= abs(observed) - 1e-12:
            count += 1
    return {
        "n_pairs": len(diffs),
        "mean_signed_shift": observed,
        "p_sign_flip": count / total,
    }


def holm_stepdown(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni step-down adjusted p-values (fixed multiplicity policy)."""
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, (name, p) in enumerate(ordered):
        adj = min(1.0, (m - index) * p)
        running = max(running, adj)
        adjusted[name] = running
    return adjusted


def equivalence_within_tolerance(low: float, high: float, tol: float = PRACTICAL_TOLERANCE) -> bool:
    """True only if the ENTIRE interval lies inside ±tol (never p>0.05 alone)."""
    return -tol <= low and high <= tol


def contrast_summary(
    shifts: list[dict[str, Any]],
    *,
    n_blocks: int,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Mean shift, paired percentile bootstrap (blocks resampled, statistic
    recomputed), sign-flip p, and the tolerance verdict.

    Empty paired samples are handled explicitly: the summary reports zero
    contributing blocks with null statistics instead of crashing or inventing
    a value.
    """
    diag = sign_flip_diagnostic(shifts)
    diffs = [entry["shift_signed"] for entry in shifts]
    if not diffs:
        return {
            "n_blocks_contributing": 0,
            "n_blocks_design": n_blocks,
            "n_missing_blocks": n_blocks,
            "mean_signed_shift": None,
            "p_sign_flip_exploratory": None,
            "ci95_paired_bootstrap": None,
            "entire_interval_within_tolerance_0.03": None,
            "note": ("EMPTY paired sample: no statistic computed; this is not evidence "
                     "of equivalence"),
        }
    rng = random.Random(seed)
    boot: list[float] = []
    for _ in range(n_bootstrap):
        sample = rng.choices(diffs, k=len(diffs))
        boot.append(statistics.mean(sample))
    boot.sort()
    lo = boot[int(0.025 * n_bootstrap)]
    hi = boot[min(n_bootstrap - 1, int(0.975 * n_bootstrap))]
    return {
        "n_blocks_contributing": len(diffs),
        "n_blocks_design": n_blocks,
        "n_missing_blocks": n_blocks - len(diffs),
        "mean_signed_shift": diag["mean_signed_shift"],
        "p_sign_flip_exploratory": diag["p_sign_flip"],
        "ci95_paired_bootstrap": {"low": lo, "high": hi},
        "entire_interval_within_tolerance_0.03": equivalence_within_tolerance(lo, hi),
        "note": ("paired block resampling; a non-significant p is NOT an equivalence claim; "
                 "equivalence requires the whole interval inside ±0.03"),
    }
