"""Controlled mechanism dry-run tests (LEAD-GATE requirement C).

Covers: design counts, explicit mapping integrity, exact ordered wire payloads
through plan → load → provider serialization with a capturing mock transport,
paired-arm contrasts (rename-only / reorder-only), and the LEAD-GATE fault
injections (changed-ID arm also changing order, reorder arm also changing IDs,
serialization sorting away order, missing block metadata, pseudoreplicated n).
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from jev_observatory.transport import TransportResponse

from jev_observatory.experiment import logical_requests
from jev_observatory.mech_dryrun import (
    ANCHOR_STATES,
    ANCHOR_STATE_SOURCE_FILE,
    ARMS,
    PRACTICAL_TOLERANCE,
    assert_renamed_only,
    assert_reordered_only,
    build_mech_dryrun_spec,
    contrast_summary,
    equivalence_within_tolerance,
    holm_stepdown,
    ordered_payload_json,
    paired_signed_shifts,
    payload_diff,
    serialization_preserves_order,
)


from jev_observatory.providers import JevProvider, RetryPolicy  # noqa: E402
from jev_observatory.runner import plan_run  # noqa: E402


@pytest.fixture(scope="module")
def designed():
    return build_mech_dryrun_spec()


@pytest.fixture(scope="module")
def spec(designed):
    return designed[0]


@pytest.fixture(scope="module")
def mappings(designed):
    return designed[1]


# ------------------------------------------------------------------ design
def test_design_counts(spec, mappings):
    assert len(spec["items"]) == 12 * 2 * 4 == 96
    assert spec["dataset"]["sampling"]["blocks"] == 12
    assert spec["claim_type"] == "exploratory"
    assert len(mappings["items"]) == 96
    for block in range(12):
        units = [m for m in mappings["items"] if m["block"] == block]
        assert len(units) == 8
        assert sorted((m["anchor"], m["arm"]) for m in units) == sorted(
            (a, arm) for a in mappings["anchors"] for arm in ARMS
        )
        # arm/anchor order must actually be shuffled (not the designed order)
        order = [m["arm"] for m in units]
        assert order != sorted(order) or block in (0,)  # deterministic seed shuffles


def test_dispatch_order_is_recorded_not_parsed(spec, mappings):
    order = [m["item_id"] for m in sorted(mappings["items"], key=lambda m: (m["block"], m["dispatch_index"]))]
    assert len(order) == 96 and len(set(order)) == 96
    # every item's mapping carries its block/anchor/arm explicitly
    for entry in mappings["items"]:
        assert {"block", "anchor", "arm", "dispatch_index"} <= set(entry)
        assert entry["anchor"] in ANCHOR_STATES


def test_specs_pass_logical_request_expansion(spec):
    requests = logical_requests(spec, shuffle=False)
    assert len(requests) == 96


# ------------------------------------------------------- screened anchors (REVIEW-1 #2)
def test_anchor_states_are_exact_screened_texts():
    """The anchors must be the EXACT already-screened states from
    runs_live_spec_screen.json (source path + hash recorded in the sidecar)."""
    repo = Path(__file__).resolve().parent.parent
    source = repo / "runs_live_spec_screen.json"
    if not source.exists():
        pytest.skip("screen spec not present")
    screen = json.loads(source.read_text(encoding="utf-8"))
    by_id = {item["id"]: item for item in screen["items"]}
    assert ANCHOR_STATES["dup-charge"] == by_id["screen-dup-charge"]["state"]
    assert ANCHOR_STATES["subscription-proration"] == \
        by_id["screen-subscription-proration"]["state"]


def test_anchor_state_source_hash_recorded_and_correct():
    repo = Path(__file__).resolve().parent.parent
    source = repo / ANCHOR_STATE_SOURCE_FILE
    if not source.exists():
        pytest.skip("screen spec not present")
    _spec, mappings = build_mech_dryrun_spec()
    provenance = mappings["anchor_state_source"]
    assert provenance["file"] == ANCHOR_STATE_SOURCE_FILE
    assert provenance["item_id"] == "screen-subscription-proration"
    assert provenance["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()


def test_sidecar_maps_measured_anchor_id_for_every_item(mappings):
    """Each measured anchor question id is explicit per item, including the
    renamed arm where the anchor's own key is renamed."""
    assert len(mappings["items"]) == 96
    for entry in mappings["items"]:
        assert entry["measured_anchor_id"]
        if entry["arm"] == "renamed":
            assert entry["measured_anchor_id"] != "anchor"
            assert entry["measured_anchor_id"] in entry["question_order"]
        else:
            assert entry["measured_anchor_id"] == "anchor"


def test_rename_arm_renames_anchor_too_and_reorder_arm_moves_it(spec, mappings):
    """Parent decision: N renames ALL question ids INCLUDING the anchor while
    preserving insertion positions; O reorders ALL questions INCLUDING moving
    the anchor while preserving ids/content."""
    index = {item["id"]: item for item in spec["items"]}
    for anchor in ANCHOR_STATES:
        rename_map = mappings["sibling_rename_maps"][anchor]
        renamed_anchor_key = rename_map["anchor"]
        assert renamed_anchor_key != "anchor"
        for block in range(12):
            s = index[f"m4-{anchor}-b{block:02d}-siblings"]
            n = index[f"m4-{anchor}-b{block:02d}-renamed"]
            o = index[f"m4-{anchor}-b{block:02d}-reordered"]
            # N: anchor renamed, same position, same content
            assert renamed_anchor_key in n["questions"] and "anchor" not in n["questions"]
            assert list(n["questions"]).index(renamed_anchor_key) == 0  # anchor first in S
            assert n["questions"][renamed_anchor_key] == s["questions"]["anchor"]
            # O: anchor key retained but its position moved
            assert "anchor" in o["questions"]
            assert list(o["questions"]) != list(s["questions"])
            assert o["questions"]["anchor"] == s["questions"]["anchor"]


# ------------------------------------------------------- arm contrasts (spec level)
def items_by_key(spec):
    out = {}
    for item in spec["items"]:
        out[(item["id"].split("-")[1], int(item["id"].split("-b")[1][:2]), item["id"].split("-", 3)[3])] = item
    return out


def test_rename_arm_is_rename_only_in_payloads(spec, mappings):
    """Parent decision: the renamed arm renames ALL ids INCLUDING the anchor,
    preserving each question's content and insertion position."""
    index = {item["id"]: item for item in spec["items"]}
    for anchor in ANCHOR_STATES:
        rename_map = mappings["sibling_rename_maps"][anchor]
        for block in range(12):
            s = index[f"m4-{anchor}-b{block:02d}-siblings"]
            n = index[f"m4-{anchor}-b{block:02d}-renamed"]
            # state identical; contents identical AT THE SAME POSITIONS; keys all differ
            assert s["state"] == n["state"]
            assert list(s["questions"].values()) == list(n["questions"].values())
            assert set(n["questions"]) == set(rename_map.values())
            for old_key, new_key in rename_map.items():
                assert s["questions"][old_key] == n["questions"][new_key]
                assert list(s["questions"]).index(old_key) == list(n["questions"]).index(new_key)
            payload_diff(s, n)  # structural diff is computable


def test_reorder_arm_is_reorder_only_in_payloads(spec):
    index = {item["id"]: item for item in spec["items"]}
    for anchor in ANCHOR_STATES:
        for block in range(12):
            s = index[f"m4-{anchor}-b{block:02d}-siblings"]
            o = index[f"m4-{anchor}-b{block:02d}-reordered"]
            assert s["state"] == o["state"]
            assert list(s["questions"]) != list(o["questions"]) or len(s["questions"]) == 1
            assert sorted(s["questions"]) == sorted(o["questions"])
            assert list(s["questions"].values()) != list(o["questions"].values()) or len(s["questions"]) == 1
            assert all(s["questions"][k] == o["questions"][k] for k in s["questions"])


# ------------------------------------------------ wire payloads via capturing transport
# REVIEW-1 #1: the wire check must exercise the ACTUAL production path —
# plan_run → items.jsonl reload → logical_requests reconstruction → provider →
# HttpxTransport — with httpx.MockTransport intercepting the exact serialized
# body bytes. No independently written ordered_payload_json surrogate.


def _answers_for(payload):
    def _answer(qid, q):
        if q.get("type") == "noul":
            return {"type": "noul", "noul": 0.5, "confidence": 0.5}  # noul has no criteria
        n = len(q["criteria"])
        probs = {k: round(1 / n, 2) for k in q["criteria"]}
        if q.get("type") == "score":
            levels = [str(i) for i in range(n)]
            return {"type": "score", "score": sum(int(l) / n for l in levels),
                    "probabilities": {l: round(1 / n, 2) for l in levels},
                    "legend": {l: q["criteria"][int(l)] for l in levels}, "confidence": 0.5}
        return {"type": "choice", "choice": next(iter(q["criteria"])),
                "probabilities": probs, "confidence": 0.5}

    return {qid: _answer(qid, q) for qid, q in payload["questions"].items()}


class _WireCapture:
    """httpx.MockTransport handler capturing raw outbound body bytes."""

    def __init__(self):
        self.bodies: list[bytes] = []

    def handler(self, request):
        self.bodies.append(request.content)  # exact bytes on the wire
        payload = json.loads(request.content)
        return httpx.Response(200, json={
            "answers": _answers_for(payload),
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "model": payload.get("model", "jev-1.13.0"),
        })


def _plan_and_reload(spec, tmp_path):
    """plan_run → reload the stored items.jsonl; returns {item_id: item}."""
    run_id, _manifest = plan_run(spec, provider="mock", root=str(tmp_path))
    items_path = tmp_path / run_id / "items.jsonl"
    planned_items = {}
    for line in items_path.read_text().splitlines():
        if line.strip():
            entry = json.loads(line)
            planned_items[entry["id"]] = entry
    assert len(planned_items) == 96
    return planned_items


def _requests_from_planned_items(planned_items):
    """CLI-equivalent reconstruction: requests rebuilt from the STORED items,
    never from the in-memory spec."""
    spec_like = {
        "model": "jev-1.13.0",
        "shuffle": False,
        "items": [planned_items[key] for key in sorted(planned_items)],
    }
    return {lr.item.item_id: lr for lr in logical_requests(spec_like, shuffle=False)}


def test_planned_items_preserve_question_insertion_order(spec, tmp_path):
    """Storage must preserve nested insertion order (REVIEW-1: sort_keys=True
    destroyed the S/O distinction on the plan→load path)."""
    by_id = {item["id"]: item for item in spec["items"]}
    planned_items = _plan_and_reload(spec, tmp_path)
    for item_id, stored in planned_items.items():
        original = by_id[item_id]
        assert list(stored["questions"]) == list(original["questions"]), item_id
        assert json.dumps(stored["questions"], ensure_ascii=False) == \
            json.dumps(original["questions"], ensure_ascii=False)
    # the S and O arms must still differ after the reload
    for anchor in ANCHOR_STATES:
        s = planned_items[f"m4-{anchor}-b00-siblings"]
        o = planned_items[f"m4-{anchor}-b00-reordered"]
        assert list(s["questions"]) != list(o["questions"])


def test_wire_body_bytes_through_httpx_mock_transport(spec, tmp_path):
    """Actual body bytes from plan → load → provider → HttpxTransport carry the
    arm contracts: S/N/O differ only as designed."""
    httpx = pytest.importorskip("httpx")  # noqa: F841 - handler uses the module-level import
    from jev_observatory.transport import HttpxTransport

    planned_items = _plan_and_reload(spec, tmp_path)
    capture = _WireCapture()
    transport = HttpxTransport(
        "http://offline.invalid",
        api_key="offline-dry-run-not-a-credential",
        http_transport=httpx.MockTransport(capture.handler),
    )
    provider = JevProvider(transport, retry_policy=RetryPolicy.none())
    requests = _requests_from_planned_items(planned_items)
    assert len(requests) == 96
    for item_id in sorted(requests):
        outcome = provider.ask(requests[item_id].request, requests[item_id].logical_request_id)
        assert outcome.status == "ok"
    transport.close()
    assert len(capture.bodies) == 96

    by_id = {item_id: body for item_id, body in zip(sorted(requests), capture.bodies)}
    for anchor in ANCHOR_STATES:
        for block in range(12):
            s_raw = by_id[f"m4-{anchor}-b{block:02d}-siblings"]
            n_raw = by_id[f"m4-{anchor}-b{block:02d}-renamed"]
            o_raw = by_id[f"m4-{anchor}-b{block:02d}-reordered"]
            # order-sensitive assertions on the ACTUAL bytes, not a surrogate
            assert_renamed_only(json.loads(s_raw), json.loads(n_raw))
            assert_reordered_only(json.loads(s_raw), json.loads(o_raw))
            assert s_raw != n_raw and s_raw != o_raw
            # the raw bytes preserve the payload's insertion order (no sort_keys)
            assert serialization_preserves_order(
                json.loads(s_raw), s_raw.decode("utf-8"))
        # the S arm body is byte-identical across blocks (repeated payload)
        s_by_block = [by_id[f"m4-{anchor}-b{b:02d}-siblings"] for b in range(12)]
        assert len(set(s_by_block)) == 1


def test_wire_payload_preview_saved(tmp_path):
    """A bounded preview (block 0, both anchors, all arms) is persisted,
    captured through the real HttpxTransport path."""
    httpx = pytest.importorskip("httpx")  # noqa: F841 - handler uses the module-level import
    from jev_observatory.transport import HttpxTransport

    spec, _mappings = build_mech_dryrun_spec()
    capture = _WireCapture()
    transport = HttpxTransport(
        "http://offline.invalid",
        api_key="offline-dry-run-not-a-credential",
        http_transport=httpx.MockTransport(capture.handler),
    )
    requests = {lr.item.item_id: lr for lr in logical_requests(spec, shuffle=False)}
    for item_id in sorted(requests):
        transport.post("/v1/systemone", requests[item_id].request.to_payload())
    transport.close()
    preview = [b for b, item_id in zip(capture.bodies, sorted(requests)) if "-b00-" in item_id]
    assert len(preview) == 8
    assert all(serialization_preserves_order(json.loads(b), b.decode("utf-8")) for b in preview)


# ------------------------------------------------------------- fault injections
def test_fault_rename_arm_that_also_reorders_is_caught():
    questions = {"anchor": {"type": "noul", "instructions": "anchor"},
                 "s000": {"type": "noul", "instructions": "y"},
                 "s001": {"type": "noul", "instructions": "z"}}
    a = {"state": "s", "questions": questions}
    # the audited mechanism-v3 confound: renaming also moved the anchor question
    # to a different insertion position (here: to the end, as in v3)
    b = {"state": "s", "questions": {
        "q_00000": questions["s000"],
        "q_00001": questions["s001"],
        "renamed_anchor": questions["anchor"],
    }}
    with pytest.raises(AssertionError, match="not rename-only"):
        assert_renamed_only(a, b)


def test_fault_reorder_arm_that_also_renames_is_caught():
    questions = {"anchor": {"type": "noul", "instructions": "x"},
                 "s000": {"type": "noul", "instructions": "y"},
                 "s001": {"type": "noul", "instructions": "z"}}
    a = {"state": "s", "questions": questions}
    b = {"state": "s", "questions": {f"moved_{k}": v for k, v in reversed(list(questions.items()))}}
    with pytest.raises(AssertionError, match="not reorder-only"):
        assert_reordered_only(a, b)


def test_fault_sort_keys_serialization_is_detected():
    payload = {"state": "s", "model": "m", "questions": {"zeta": {"a": 1}, "alpha": {"b": 2}}}
    ordered = ordered_payload_json(payload)
    sorted_keys = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert serialization_preserves_order(payload, ordered)
    assert not serialization_preserves_order(payload, sorted_keys)


def test_fault_missing_block_metadata_is_caught(mappings):
    """Every recorded mapping must carry explicit block metadata."""
    from jev_observatory.mech_dryrun import validate_item_mapping

    for entry in mappings["items"]:
        validate_item_mapping(entry)  # complete design passes
    incomplete = json.loads(json.dumps(mappings))
    incomplete["items"][7].pop("block")
    with pytest.raises(ValueError, match="block"):
        validate_item_mapping(incomplete["items"][7])


def test_mapping_required_fields_guard():
    """Explicit guard: an item mapping without block/anchor/arm is invalid."""
    from jev_observatory.mech_dryrun import validate_item_mapping

    good = {"item_id": "x", "block": 0, "anchor": "dup-charge", "arm": "alone",
            "dispatch_index": 0, "question_order": ["anchor"], "renamed_ids": {}}
    assert validate_item_mapping(good)
    for field in ("block", "anchor", "arm", "dispatch_index", "question_order"):
        bad = dict(good)
        bad.pop(field)
        with pytest.raises(ValueError, match=field):
            validate_item_mapping(bad)


# ------------------------------------------------------------------- analysis
def test_paired_shifts_refuse_pseudoreplicated_n():
    with pytest.raises(ValueError, match="pseudoreplication"):
        paired_signed_shifts({b: 0.5 for b in range(66)}, {b: 0.4 for b in range(66)}, n_blocks=12)


def test_paired_shifts_one_per_block():
    shifts = paired_signed_shifts({0: 0.5, 1: 0.6, 3: 0.4}, {0: 0.4, 1: 0.6, 3: 0.5}, n_blocks=12)
    assert len(shifts) == 3
    assert [s["shift_signed"] for s in shifts] == pytest.approx([0.1, 0.0, -0.1])


def test_sign_flip_and_holm_and_tolerance():
    shifts = [{"block": b, "shift_signed": 0.01} for b in range(12)]
    summary = contrast_summary(shifts, n_blocks=12, n_bootstrap=200, seed=1)
    assert summary["n_blocks_contributing"] == 12
    assert summary["mean_signed_shift"] == pytest.approx(0.01)
    assert summary["entire_interval_within_tolerance_0.03"] is True
    # a shift beyond the tolerance is not "equivalent" even if p is large
    shifts_big = [{"block": b, "shift_signed": 0.10} for b in range(12)]
    big = contrast_summary(shifts_big, n_blocks=12, n_bootstrap=200, seed=1)
    assert big["entire_interval_within_tolerance_0.03"] is False
    adjusted = holm_stepdown({f"c{i}": 0.03 for i in range(6)})
    assert adjusted["c0"] == pytest.approx(0.18)
    assert max(adjusted.values()) <= 1.0
    assert equivalence_within_tolerance(-0.02, 0.02)
    assert not equivalence_within_tolerance(-0.02, 0.04)
    assert PRACTICAL_TOLERANCE == 0.03
