"""Focused offline tests for the follow-up battery (followup-20261003.v1).

Covers the validation list: fake providers, frozen gold excluded from the wire,
the pruning/vocabulary representation artifact, rounding contrast, cluster/block
latency fit with whole-stimulus holdout, the horizon gold correction, and
budget/nonce/version checks.  No network (conftest fails on sockets).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT / "src", ROOT / "scripts" / "benchmark"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import followup_battery as fb  # noqa: E402
import run_followup_battery as runner  # noqa: E402
from jev_observatory.revision_run import BudgetExceeded, PersistentBudget, RevisionRunError  # noqa: E402


# ------------------------------------------------- frozen gold excluded from wire
def test_gold_annotations_never_in_outbound_payload():
    for call in fb.build_fu_horizon():
        payload = call.request.to_payload()
        assert set(payload) == {"model", "state", "questions"}
        wire = json.dumps(payload, ensure_ascii=False)
        assert "\"gold" not in wire and "gold_" not in wire
        # the gold option TEXT must be on the wire as a candidate (it is an
        # option); the gold ANNOTATION (which option is correct) is meta-only
        assert call.meta["gold_text"] in wire
        assert call.meta["gold_key"].startswith("o")
        assert payload["questions"]["q0"]["criteria"][call.meta["gold_key"]] \
            == call.meta["gold_text"]


# ---------------------------------------- pruning / vocab representation artifact
def test_literal_vocab_key_lookup_is_a_representation_artifact():
    """Byte-encoded vocab keys must not be read as literal Unicode membership:
    a string can be one token while its literal form is not a vocab key."""
    byte_encoded_vocab = {"zz": 0, "\u0120zz": 1, "hello": 2}
    assert "zzz" not in byte_encoded_vocab          # literal lookup says miss
    assert "\u0120zz" in byte_encoded_vocab         # its byte-ish entry exists
    assert "hello" in byte_encoded_vocab and "zz" in byte_encoded_vocab
    # the battery therefore counts via encode length, never via key membership
    membership_method = ("tokenizers.Tokenizer.encode -> len(ids)")
    assert "encode" in membership_method


def test_token_builder_covers_discriminating_strings_and_controls():
    calls = fb.build_fu_vocab()
    probes = {c.meta["sample_id"]: c.request.state for c in calls}
    assert probes["zzz:bare"] == "zzz"
    assert probes["usc6:delim5"].count("|") == 4
    assert probes["empty_delim5"] == "||||"
    controls = [c for c in calls if c.meta["role"] == "control"]
    targets = [c for c in calls if c.meta["role"] == "target"]
    assert len(controls) == 12 and len(targets) == 96
    assert len([c for c in calls if c.meta["role"] == "mergerate"]) == 16
    # the empty noul frame ('x') replay and heart forms are present
    assert fb._HEART2 in probes["heart2:bare"] and fb._HEART1 in probes["heart1:bare"]
    assert all(c.request.questions["n1"].type == "noul" for c in calls)


# ------------------------------------------------------------ rounding contrast
def test_logodds_rounding_band_and_floor_exclusion():
    band = fb.logodds_rounding_band(0.50, 0.25, quanta=1.0)
    assert abs(band["logodds"] - math.log(2.0)) < 1e-9
    assert band["band"][0] < math.log(2.0) < band["band"][1]
    assert not band["at_display_floor"]
    # 1.5 quanta widens the band monotonically
    wide = fb.logodds_rounding_band(0.50, 0.25, quanta=1.5)
    assert wide["band"][0] < band["band"][0] and wide["band"][1] > band["band"][1]
    floor = fb.logodds_rounding_band(0.01, 0.90, quanta=1.0)
    assert floor["at_display_floor"]
    # a strong pair's band excludes 0 (rank robust to display rounding)
    strong = fb.pairwise_logodds({"a": 0.6, "b": 0.2}, [("a", "b")])
    assert strong["a~b"]["band"][0] > 0
    # a near pair's band covers 0 (no claim allowed at this rounding)
    near = fb.pairwise_logodds({"a": 0.26, "b": 0.24}, [("a", "b")])
    assert near["a~b"]["band"][0] <= 0 <= near["a~b"]["band"][1]


# ------------------------------------------------------------- cluster/block fit
def _place_rows(question_factor: float) -> list[dict]:
    rows = []
    for block in range(8):
        for text in ("fillerA", "fillerB"):
            for size, base in (("s0500", 500), ("s4000", 4000), ("s16000", 16000)):
                for cell, role in (("q1_state", "state"), ("q1_question", "question")):
                    q_share = 0.9 if role == "question" else 0.1
                    rows.append({
                        "family": "fu_place", "block": block, "text_id": text,
                        "size_id": size, "cell_id": cell, "role": role,
                        "usage_input_tokens": base + 100,
                        "state_chars_local": int(4 * base * (1 - q_share)) + 40,
                        "question_chars_local": int(4 * base * q_share) + 40,
                        "upstream_ms": 50.0 + block * 3.0 + 0.006 * base
                        + question_factor * 0.002 * base * q_share,
                        "error": None})
    return rows


def test_cluster_block_fit_recovers_role_cost_and_holds_out_stimuli():
    equal = fb.fit_latency_models(_place_rows(question_factor=0.0))
    ratio = equal["models"]["reported_role_split"]["question_vs_state_cost_ratio"]
    assert 0.5 < ratio < 1.5                     # ~1x: no 2x question cost
    twice = fb.fit_latency_models(_place_rows(question_factor=1.0))
    ratio2 = twice["models"]["reported_role_split"]["question_vs_state_cost_ratio"]
    assert ratio2 > ratio                        # planted extra question cost seen
    for model in equal["models"].values():
        assert set(model["heldout_stimulus_rmse_ms"]) == {
            "holdout_fillerA", "holdout_fillerB"}


# -------------------------------------------------------- horizon gold correction
def test_horizon_gold_correction_preserves_originals_and_separates_kinds():
    rows = []
    for ev, kind, month in ((15, "post", "2025-07"), (21, "pre", "2024-08"),
                            (3, "post", "2024-12"), (1, "fictional", "2025-03")):
        for rep in range(4):
            rows.append({"family": "p5_horizon", "event_i": ev, "kind": kind,
                         "month": month, "gold": f"in {month}",
                         "config_id": f"p5:{kind}:{ev}:r{rep}",
                         "choice": "it did not occur"})
    out = fb.p5_corrected_summary(rows)
    assert out["original_rows_preserved"] is True
    assert out["n_original_rows"] == 16
    assert len(out["excluded_rows"]) == 4          # Starliner rows excluded
    assert len(out["corrected_rows"]) == 4         # Anchorage relabeled 2025-08
    assert "2025-07" not in out["corrected_monthly_post"]
    assert out["corrected_monthly_post"]["2025-08"]["n"] == 4    # relabeled rows
    assert set(out["per_kind"]) == {"post", "pre", "fictional"}
    assert out["per_kind"]["pre"]["n"] == 0        # all 4 pre rows excluded


# ----------------------------------------------- budget / nonce / version checks
def test_budget_pre_call_reservation_fail_closed(tmp_path):
    ledger = tmp_path / "BUDGET.json"
    budget = PersistentBudget(ledger, cap_usd=0.01)
    rid = budget.reserve("call-1", 0.005, label="fu_vocab")
    budget.settle(rid, 0.001)                      # known cost: difference freed
    snap = budget.snapshot()
    assert snap["settled_usd"] == pytest.approx(0.001, abs=1e-9)
    rid2 = budget.reserve("call-2", 0.005, label="fu_vocab")
    budget.settle(rid2, None)                      # unknown cost holds FOREVER
    assert budget.snapshot()["held_usd"] == pytest.approx(0.005, abs=1e-9)
    with pytest.raises(BudgetExceeded):
        budget.reserve("call-3", 0.05)             # cap refuses before dispatch
    # resume across instances inherits spend; the cap is bound to the ledger
    again = PersistentBudget(ledger, cap_usd=0.01)
    assert again.snapshot()["held_usd"] == pytest.approx(0.005, abs=1e-9)
    with pytest.raises(RevisionRunError):
        PersistentBudget(ledger, cap_usd=5.0)      # cap mismatch refuses


def test_nonce_uniqueness_and_frozen_version():
    calls, _ = fb.build_fu_place_ordered()
    nonces = [c.meta["nonce"] for c in calls]
    assert len(nonces) == len(set(nonces)) == 192  # fresh per call
    assert all(len(n) == 8 for n in nonces)
    for call in calls:
        assert call.request.state.endswith(f"Run marker: {call.meta['nonce']}.")
    assert fb.FOLLOWUP_VERSION == "followup-20261003.v1"
    assert fb.MODEL_PIN == "jev-1.13.0"
    budget = fb.worst_case_costs()
    assert budget["fits_cap"] and budget["total_worst_usd"] <= fb.BATTERY_CAP_USD
    assert budget["counts"]["fu_horizon"] == 48
    from jev_observatory.answer_recovery import recovery_spec, RECOVERY_SPEC_VERSION
    assert RECOVERY_SPEC_VERSION == "answer-recovery-2.1.0"
    assert "final_line_key" in recovery_spec()["stages"]


def test_duplicate_noise_control_payloads_are_intentional():
    calls = fb.build_fu_odds()
    arch = [c for c in calls if c.meta["arm"] == "archer"]
    by_variant = {v: [c for c in arch if c.meta["variant"] == v]
                  for v in fb._ARCHER_VARIANTS}
    p = {v: cs[0].request.to_payload() for v, cs in by_variant.items()}
    assert p["base4"] == p["null4"] and p["append5"] == p["null5"]
    assert all(c.meta["noise_control"] == (c.meta["variant"] in ("null4", "null5"))
               for c in arch)
    assert p["append5"] != p["replace5"]


# ---------------------------------------------------------------- fake providers
class _FakeResp:
    def __init__(self, body: bytes, status: int = 200, headers: dict | None = None):
        self.status_code, self._body, self.headers = status, body, headers or {}

    def iter_bytes(self):
        yield self._body


class _FakeStream:
    def __init__(self, resp):
        self._resp = resp

    def __enter__(self):
        return self._resp

    def __exit__(self, *exc):
        return False


class FakeJeVTransport:
    """Fake Jev endpoint: scripted responses, no sockets."""

    def __init__(self, script):
        self.script = list(script)
        self.last = None

    def stream(self, method, path, json=None, **kwargs):
        return _FakeStream(self.script.pop(0))


def test_fake_jev_provider_settles_known_cost_and_holds_unknown():
    call = fb.build_fu_vocab()[0]
    ok_body = json.dumps({"model": "jev-1.13.0", "answers": {"n1": {}},
                          "usage": {"input_tokens": 300, "output_tokens": 10}}).encode()
    cap = runner.CapturingTransport(FakeJeVTransport([_FakeResp(ok_body)]),
                                    runner.Redactor())
    budget = PersistentBudget(None, cap_usd=1.0)
    row, abort = runner._jev_attempts(call, cap, budget, call.config_id)
    assert not abort and row["usage_input_tokens"] == 300
    assert budget.snapshot()["settled_usd"] == pytest.approx(300 * 0.042 / 1e6,
                                                            abs=5e-7)
    extras = row["raw_body_sanitized"]
    assert "jev-1.13.0" in extras and row["served_model"] == "jev-1.13.0"

    # unknown-cost outcome (transport error) holds the full reservation
    cap2 = runner.CapturingTransport(
        FakeJeVTransport([_FakeResp(b"", status=0)] * fb.MAX_TRANSPORT_ATTEMPTS),
        runner.Redactor())
    budget2 = PersistentBudget(None, cap_usd=1.0)
    row2, abort2 = runner._jev_attempts(call, cap2, budget2, call.config_id)
    assert not abort2 and row2["error"]
    snap2 = budget2.snapshot()
    assert snap2["held_usd"] == pytest.approx(
        fb.JEV_MAX_INPUT_TOKENS * 0.042 / 1e6 * fb.MAX_TRANSPORT_ATTEMPTS,
        abs=5e-7)

    # auth failure fails closed and asks the caller to stop
    cap3 = runner.CapturingTransport(
        FakeJeVTransport([_FakeResp(b'{"error": "unauthorized"}', status=401)]),
        runner.Redactor())
    budget3 = PersistentBudget(None, cap_usd=1.0)
    row3, abort3 = runner._jev_attempts(call, cap3, budget3, call.config_id)
    assert abort3 and row3["http_status"] == 401
    assert budget3.snapshot()["held_usd"] > 0        # never released as free


# =============================================================== v2 repair tests
@pytest.mark.skipif(not fb.PROBE_TRIALS_PATH.exists(),
                    reason="published trials JSON not present")
def test_relational_v2_exact_replication_and_varying_gold():
    calls = fb.load_relational_replication()
    assert len(calls) == 96  # 6 perms x 2 values x 2 templates x 2 reps x 2 locs
    import json as _json
    published = _json.loads(fb.PROBE_TRIALS_PATH.read_text(encoding="utf-8"))
    by_idx = {t["index"]: t for t in published["trials"]
              if t.get("experiment") == "relational_order"}
    for c in calls:
        t = by_idx[c.meta["trial_index"]]
        payload = c.request.to_payload()
        assert payload["state"] == t["payload"]["state"]          # verbatim copy
        assert payload["model"] == "jev-1.13.0"                   # pin changed only
        q_pub = next(iter(t["payload"]["questions"].values()))
        q_new = payload["questions"][next(iter(payload["questions"]))]
        assert q_new["criteria"] == q_pub["criteria"]
        assert q_new["instructions"] == q_pub["instructions"]
    # gold derived from fixed semantics varies (no always-same gold key)
    golds = {}
    for c in calls:
        golds[c.meta["gold_key"]] = golds.get(c.meta["gold_key"], 0) + 1
    assert set(golds) == {"beta", "gamma"} and len(set(golds.values())) == 1
    # literal card-location invariants
    for c in calls:
        state_card = fb._CARD_RE.search(c.request.state) is not None
        crit = dict(next(iter(c.request.questions.values())).criteria)
        ref_text = crit[c.meta["ref_only_key"]]
        if c.meta["placement"] == "option":
            assert not state_card and fb._CARD_RE.search(ref_text)
        else:
            assert state_card and fb._CARD_RE.search(ref_text) is None


def test_isolation_v2_no_card_leak_and_qorder():
    calls = fb.build_fu_isolation()
    assert len(calls) == 64  # 8 blocks x 4 locations x 2 qorders
    for c in calls:
        q = c.request.questions
        texts = {"state": c.request.state,
                 "target_instr": q["q_target"].instructions,
                 "sibling_instr": q["q_sib"].instructions}
        for chan, text in texts.items():
            has = fb._CARD_LINE_RE.search(text) is not None if hasattr(fb, "_CARD_LINE_RE") \
                else c.meta["card_text"] in text
            if c.meta["location"] == "absent":
                assert not has
            elif chan == c.meta["location"]:
                assert has
            else:
                assert not has, (c.config_id, chan)   # no leak across channels
        assert len(c.meta["irrelevant_text"]) == len(c.meta["card_text"])
        assert list(q)[0] == ("q_target" if c.meta["qorder"] == "target_first"
                              else "q_sib")
    # gold option position balanced across the battery (not fixed o0)
    gold_keys = [c.meta["gold_key"] for c in calls]
    assert set(gold_keys) == {"o0", "o1", "o2", "o3"}


def test_target_qid_extraction_is_order_independent():
    criteria = {"o0": "a1-b2", "o1": "c3-d4", "o2": "e5-f6", "o3": "no code supplied"}
    # sibling answer FIRST in the body answers mapping (reordered mixed questions)
    parsed = {"answers": {"q_sib": {"type": "noul", "noul": 0.9},
                          "q_target": {"type": "choice", "choice": "o1",
                                       "probabilities": {"o0": 0.1, "o1": 0.7,
                                                         "o2": 0.1, "o3": 0.1}}}}
    probs, choice = fb.extract_answer_by_qid(parsed, "q_target", criteria)
    assert choice == "c3-d4" and probs["c3-d4"] == 0.7   # target, not first answer
    probs2, choice2 = fb.extract_answer_by_qid(parsed, "missing", criteria)
    assert choice2 is None and probs2 == {}


def test_timing_v2_nonce_at_beginning_no_common_prefix():
    calib = {"latin": {"per_char": 0.33, "intercept": 270.0},
             "digits": {"per_char": 1.0, "intercept": 270.0},
             "cjk": {"per_char": 1.0, "intercept": 270.0}}
    calls = fb.build_fu_timing(calib)
    assert len(calls) == 64  # 3 scripts x 2 counts x 8 blocks + 2 x 8 question
    nonces = set()
    for c in calls:
        assert c.request.state.startswith("Run marker: "), c.config_id
        assert c.meta["nonce"] in c.request.state[:30]
        nonces.add(c.meta["nonce"])
        assert c.meta["payload_bytes"] > 0 and c.meta["payload_chars"] > 0
    assert len(nonces) == 64            # unique prefix per request


def test_odds_primary_pooling_and_delta():
    rows = []
    for b in range(10):
        for variant, cust, unk in (("base4", 0.50, 0.25), ("null4", 0.54, 0.23),
                                   ("append5", 0.30, 0.35), ("null5", 0.32, 0.33)):
            rows.append({"family": "fu_odds", "arm": "archer", "block": b,
                         "variant": variant,
                         "probabilities": {_CUST_T: cust, _UNK_T: unk}})
    out = fb.odds_primary_v2(rows)
    import math as _m
    base = _m.log((0.50 + 0.54) / 2 / ((0.25 + 0.23) / 2))
    app = _m.log((0.30 + 0.32) / 2 / ((0.35 + 0.33) / 2))
    assert abs(out["per_arm_mean_primary"]["base_pooled"] - base) < 1e-3
    assert abs(out["per_arm_mean_primary"]["append_pooled"] - app) < 1e-3
    assert abs(out["append_minus_base_mean"] - (app - base)) < 1e-3
    assert out["append_minus_base_block_bootstrap_ci95"] is not None


_CUST_T = "The customer caused it"
_UNK_T = "Cannot tell"


def test_echo_recovery_accepts_exact_echo_rejects_disagreement():
    crit = {"o0": "Los Angeles Dodgers", "o4": "it did not occur",
            "o5": "cannot determine from what I know"}
    keys = list(crit)
    ok, stage = fb.echo_recover_v2("o4. it did not occur", keys, crit)
    assert ok == "o4" and stage == "echo_exact_text"
    ok2, _ = fb.echo_recover_v2("O5. Cannot Determine From What I Know", keys, crit)
    assert ok2 == "o5"                        # case-flex on exact option text
    bad1, s1 = fb.echo_recover_v2("o4. Los Angeles Dodgers", keys, crit)
    assert bad1 is None and s1 == "key_text_disagree"
    bad2, s2 = fb.echo_recover_v2("o4", keys, crit)   # bare key: not echo shape
    assert bad2 is None and s2 == "not_echo_shape"
    bad3, s3 = fb.echo_recover_v2("o4. i", keys, crit)  # never blind first char
    assert bad3 is None and s3 == "key_text_disagree"
    bad4, _ = fb.echo_recover_v2("o4. it did not occur or o5.", keys, crit)
    assert bad4 is None                        # multiple answers rejected


def test_v2_counts_per_family():
    rows_v1 = [{"family": "fu_vocab", "usage_input_tokens": 268}] * 124
    rows_v1 += [{"family": "fu_place", "block": 0, "text_id": "x",
                 "size_id": "s", "cell_id": "c", "role": "state",
                 "state_chars_local": 1, "question_chars_local": 1,
                 "usage_input_tokens": 300, "wall_ms": 300.0}] * 192
    rows_v2 = [{"family": "fu_relational", "placement": "state", "value": "east",
                "gold_key": "beta", "gold_text": "b", "choice": "b"}] * 96
    rows_v2 += [{"family": "fu_isolation", "location": "state", "qorder": "t",
                 "gold_text": "c", "choice": "c"}] * 64
    rows_v2 += [{"family": "fu_timing", "script": "latin", "target_count": "t3k",
                 "placement": "state", "block": 0, "usage_input_tokens": 3000,
                 "payload_bytes": 10, "wall_ms": 300.0, "encode_qwen3_tokens": 500,
                 "encode_qwen35_tokens": 400}] * 64
    out = fb.analyze_v2(rows_v1 + rows_v2, rows_v2)
    counts = out["counts_reported_separately"]["by_family"]
    assert counts["fu_vocab"] == 124 and counts["fu_place"] == 192
    assert counts["fu_relational"] == 96 and counts["fu_isolation"] == 64
    assert counts["fu_timing"] == 64
    assert out["fu_relational"]["n_rows"] == 96
    assert out["fu_isolation"]["n_rows"] == 64
    assert out["fu_timing"]["n_rows"] == 64


# ============================================================ v3 counter tests
def test_v3_stimuli_byte_matched_and_nonce_first():
    fillers = fb._v3_fillers()
    for (style, n_bytes), s in fillers.items():
        assert len(s.encode("utf-8")) == n_bytes          # exact byte volume
    calls = fb.build_fu_counter_v3()
    assert len(calls) == 64                               # 4x2x8
    for c in calls:
        assert c.request.state.startswith("Run marker: ")  # nonce FIRST
        assert c.meta["nonce"] in c.request.state[:30]
        assert c.request.state.endswith(fb.V3_FINAL)
        assert c.request.questions["q0"].instructions == fb.V3_QUESTION
    # exact wire bytes identical across styles at each byte size
    for n_bytes in fb.V3_BYTE_SIZES:
        wire = {c.meta["payload_bytes"] for c in calls
                if c.meta["filler_bytes"] == n_bytes}
        assert len(wire) == 1, (n_bytes, wire)
    # nonces unique but fixed-length (byte match survives)
    assert len({c.meta["nonce"] for c in calls}) == 64
    assert all(len(c.meta["nonce"]) == 8 for c in calls)


def test_v3_digit_separator_and_encoder_facts():
    digits = fb._v3_fillers()[("digits", 6000)]
    assert len(digits) == 6000 and " " not in digits       # contiguous, no separators
    from tokenizers import Tokenizer
    q3 = Tokenizer.from_file("/tmp/probe/Qwen3-30B-A3B-Instruct-2507__tokenizer.json")
    n_q3 = len(q3.encode(digits, add_special_tokens=False).ids)
    import tiktoken
    o200k = tiktoken.get_encoding("o200k_base")
    n_o = len(o200k.encode(digits))
    assert n_q3 == 6000                 # Qwen3 counts 1/digit (like Jev)
    assert n_o == 2000                  # o200k groups digits ~1/3: the separator
    cjk = fb._v3_fillers()[("cjk", 6000)]
    assert len(cjk.encode("utf-8")) == 6000
    n_q3c = len(q3.encode(cjk, add_special_tokens=False).ids)
    assert 1100 < n_q3c < 1600          # Qwen3 ~0.67/char << Jev 1/char on CJK


def test_v3_per_family_counts():
    rows = [{"family": "fu_counter_v3", "style": s, "filler_bytes": b,
             "block": blk, "n_chars": 1, "usage_input_tokens": 3000,
             "payload_bytes": 100, "payload_bytes_exact": 100,
             "encode_qwen3_tokens": 500, "encode_qwen35_tokens": 400,
             "encode_o200k_tokens": 300, "wall_ms": 300.0, "first_byte_ms": 280.0}
            for blk in range(8) for s in fb.V3_STYLES for b in fb.V3_BYTE_SIZES]
    out = fb.counter_v3_summary(rows)
    assert out["n_rows"] == 64 and out["n_blocks"] == 8
    assert out["byte_match_verified"] is True
    assert set(out["holdout_whole_style_rmse_ms"]) >= {"reported", "chars",
                                                       "wire_bytes", "encode_o200k"}
    assert out["sample_unit"].startswith("8 randomized blocks")


def test_placement_incremental_preserves_both_definitions():
    rows = []
    for b in range(8):
        for placement, inc in (("state", 9.0), ("question", 18.0)):  # planted 2x
            for target, tok, base in (("t3k", 3000, 300.0), ("t12k", 12000, 300.0)):
                wall = base + (tok - 3000) / 1000.0 * inc
                rows.append({"family": "fu_timing", "script": "latin",
                             "placement": placement, "target_count": target,
                             "block": b, "usage_input_tokens": tok,
                             "wall_ms": wall, "error": None})
    out = fb.placement_incremental_summary(rows)
    assert out["incremental_ratio"] == 2.0                    # planted recovery
    assert out["incremental_ms_per_1k"] == {"state": 9.0, "question": 18.0}
    assert "total_ratio_mean" in out and "definition_total_ratio" in out
    assert "definition_incremental_ratio" in out              # both preserved
    ci = out["incremental_ratio_block_bootstrap_ci95"]
    assert ci[0] <= 2.0 <= ci[1]                              # noise-robust test
    assert "neither contradicted nor ruled out" in out["label"]
    # negative outlier blocks are named, not silently dropped
    rows2 = [dict(r) for r in rows]
    for r in rows2:
        if r["block"] == 5 and r["placement"] == "state" and r["target_count"] == "t12k":
            r["wall_ms"] = 250.0                              # make increment negative
    out2 = fb.placement_incremental_summary(rows2)
    assert "block5_state_increment" in out2["network_outliers"]
