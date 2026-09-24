"""Offline tests for the architecture-probe module and its analyzers.

No network, no credentials, no live calls (conftest bans sockets): a stub
transport stands in for httpx and exercises the FULL record->analyze path with
synthetic responses whose token/latency values follow known linear laws, so the
analyzers must recover those laws.
"""

from __future__ import annotations

import json

import pytest

from jev_observatory import arch_probe as ap


# --------------------------------------------------------------- builders
def test_builders_produce_valid_requests():
    for name in ap.FAMILIES:
        calls = ap.BUILDERS[name]()
        assert calls, name
        for c in calls:
            assert c.family == name
            json.dumps(c.request.to_payload(), ensure_ascii=False)
            m = c.payload_metrics()
            assert m["n_questions"] >= 1 and m["n_options_total"] >= 2


def test_ancestry_always_offers_typesafe_and_jev():
    for c in ap.build_ancestry():
        names = set(c.request.questions["q0"].criteria.values())
        assert "Typesafe" in names and "Jev" in names
        assert "OpenAI" in names and "Mellum" in names


def test_ancestry_cyclic_rotations_balanced():
    """Across a frame the option set rotates, so position bias can cancel."""
    rots = ap.build_ancestry(rotations=6)
    per_frame = {}
    for r in rots:
        per_frame.setdefault(r.meta["frame"], []).append(r.meta["rotation"])
    assert all(sorted(v) == list(range(6)) for v in per_frame.values())


# ------------------------------------------------------- stub transport path
class _FakeStream:
    def __init__(self, resp): self._r = resp
    def __enter__(self): return self._r
    def __exit__(self, *a): return False


class _FakeResp:
    def __init__(self, status, headers, body):
        self.status_code = status
        self.headers = headers
        self._body = body
    def iter_bytes(self):
        yield self._body


class StubClient:
    """httpx-Client-like: stream() returns a canned choice/noul answer.

    Latency model: upstream_ms = 40 + 3.0 * (n_questions) + 0.02 * (n_options)
    + 0.05 * (input_tokens/1000).  Token model: 60 + 1.0 * payload_chars/4.
    The stub reads the request payload to drive these.
    """
    def __init__(self, kind="choice"): self.kind = kind; self.resets = 0
    def reset(self): self.resets += 1

    def stream(self, method, path, json=None):
        payload = json or {}
        qs = payload.get("questions") or {}
        first = next(iter(qs.values()), {})
        crit = first.get("criteria") or {}
        nq = len(qs); nopt = len(crit)
        state_chars = len(str(payload.get("state", "")))
        in_tok = 60 + (state_chars + 40 * nq * max(1, nopt)) // 4
        up = 40.0 + 3.0 * nq + 0.02 * nopt + 0.05 * (in_tok / 1000.0)
        out_tok = 20 * nq * max(1, nopt) // 2
        probs = {k: round(1.0 / max(1, nopt), 3) for k in crit} or {"true": 0.6, "false": 0.4}
        choice = (max(probs, key=probs.get) if probs else None)
        body = json_mod_dumps({"model": "stub", "answers": {
            "q0": {"type": "choice", "choice": choice, "probabilities": probs}},
            "usage": {"input_tokens": in_tok, "output_tokens": out_tok}})
        return _FakeStream(_FakeResp(200, {ap.UPSTREAM_HEADER: f"{up:.1f}"},
                                     body.encode("utf-8")))


def json_mod_dumps(obj):
    import json as _j
    return _j.dumps(obj)


def test_record_from_response_captures_metrics():
    client = StubClient()
    call = ap.build_optioncount(reps=1)[2]  # k=10 optioncount
    row = ap.record_from_response(call, client, cold=False)
    assert row["http_status"] == 200
    assert row["usage_input_tokens"] > 0
    assert row["upstream_ms"] is not None
    assert row["n_options_total"] == call.payload_metrics()["n_options_total"]
    assert isinstance(row["probabilities"], dict) and row["probabilities"]


def test_record_handles_transport_error():
    class Boom:
        def stream(self, *a, **k): raise RuntimeError("network down")
    call = ap.build_coldwarm(reps=1)[0]
    row = ap.record_from_response(call, Boom(), cold=True)
    assert row["error"] and row["upstream_ms"] is None


# ------------------------------------------------------------- analyzers
def _synth_rows():
    """Build rows whose upstream_ms follows known laws; analyzers must recover."""
    rows = []
    for target, tok in zip((1000, 2000, 4000, 8000), (250, 500, 1000, 2000)):
        # prefill: up = 50 + 2.0 ms per 1k tokens
        for rep in range(3):
            rows.append({"family": "prefill", "config_id": f"p{target}r{rep}",
                         "usage_input_tokens": tok,
                         "upstream_ms": 50.0 + 2.0 * (tok / 1000.0)})
    # headcount: up flat at 60ms (self-batch), output grows with nq
    for nq in (1, 8, 64, 192):
        for rep in range(3):
            rows.append({"family": "headcount", "config_id": f"h{nq}r{rep}",
                         "n_questions": nq, "upstream_ms": 60.0,
                         "usage_output_tokens": 4.0 * nq})
    # optioncount: up = 55 + 0.05ms per option
    for k in (2, 32, 128, 255):
        for rep in range(3):
            rows.append({"family": "optioncount", "config_id": f"k{k}r{rep}",
                         "n_options_total": k, "upstream_ms": 55.0 + 0.05 * k,
                         "usage_output_tokens": 0.5 * k})
    # coldwarm
    for rep in range(3):
        rows.append({"family": "coldwarm", "config_id": f"coldr{rep}",
                     "cold": True, "wall_ms": 210.0})
        rows.append({"family": "coldwarm", "config_id": f"warrr{rep}",
                     "cold": False, "wall_ms": 90.0})
    # ancestry: strong pull to OpenAI despite parity, uniform otherwise
    names = ap.ANCESTRY_OPTIONS
    for fi in range(len(ap.ANCESTRY_FRAMES)):
        for rot in range(6):
            probs = {n: (0.30 if n == "OpenAI" else 0.70 / (len(names) - 1))
                     for n in names}
            rows.append({"family": "ancestry", "config_id": f"a{fi}r{rot}",
                         "probabilities": probs, "choice": "OpenAI"})
    return rows


def test_analyze_recovers_prefill_slope_and_floor():
    out = ap.analyze_prefill(_synth_rows())
    assert out["status"] != "insufficient_data"
    assert abs(out["fixed_floor_ms"] - 50.0) < 1.0
    assert abs(out["ms_per_1k_input_tokens"] - 2.0) < 0.3
    assert out["r2_linear"] > 0.99


def test_analyze_headcount_marginal_is_flat():
    out = ap.analyze_headcount(_synth_rows())
    assert abs(out["marginal_ms_per_question"]) < 0.01  # self-batch: ~0
    assert abs(out["marginal_output_tokens_per_question"] - 4.0) < 0.5


def test_analyze_optioncount_marginal():
    out = ap.analyze_optioncount(_synth_rows())
    assert abs(out["marginal_ms_per_option"] - 0.05) < 0.01


def test_analyze_coldwarm_overhead():
    out = ap.analyze_coldwarm(_synth_rows())
    assert out["connection_overhead_ms"] == 120.0


def test_analyze_ancestry_detects_openai_prior_and_typesafe_offered():
    out = ap.analyze_ancestry(_synth_rows())
    assert out["typesafe_offer"] is True
    assert out["top_options_by_mean_p"][0][0] == "OpenAI"
    assert out["family_mass"].get("openai", 0) > out["family_mass"].get("typesafe", 0)
    assert out["greedy_votes"]["OpenAI"] == len(ap.ANCESTRY_FRAMES) * 6


def test_analyze_rows_end_to_end_has_every_family():
    out = ap.analyze_rows(_synth_rows())
    for fam in ("prefill", "headcount", "optioncount", "coldwarm", "ancestry"):
        assert fam in out and out[fam]


def test_build_concurrency_groups_complete():
    calls = ap.build_concurrency(reps=1)
    assert len(calls) == 1 + 2 + 4 + 8 + 16 + 32
    by_group: dict[str, int] = {}
    for c in calls:
        by_group[c.meta["group"]] = by_group.get(c.meta["group"], 0) + 1
    assert all(by_group[g] == next(c.meta["c"] for c in calls
                                   if c.meta["group"] == g) for g in by_group)


def test_analyze_concurrency_curve_and_filtering():
    rows = []
    for c, ms in [(1, 90.0), (2, 95.0), (4, 100.0), (8, 110.0)]:
        for j in range(c):
            rows.append({"family": "concurrency", "config_id": f"concurrency:c{c}:r0:j{j}",
                         "group": f"c{c}r0", "c": c, "http_status": 200,
                         "wall_ms": ms, "error": None})
    # a rate-limited row must be excluded from the curve
    rows.append({"family": "concurrency", "config_id": "concurrency:c8:r0:j9",
                 "group": "c8r0", "c": 8, "http_status": 429, "wall_ms": 20.0,
                 "error": None})
    out = ap.analyze_concurrency(rows)
    assert out["per_call_wall"]["1"]["median_wall_ms"] == 90.0
    assert out["per_call_wall"]["8"]["n"] == 8  # the 429 excluded
    assert out["batch_completion_ms_per_call"]["1"] == 90.0


def test_mergerate_samples_are_space_free():
    from jev_observatory.arch_probe_mergerate import SAMPLES
    for name, script, text in SAMPLES:
        assert text, name
        # interior single spaces are allowed (code tokens); no runs / no edges
        import re
        assert not re.search(r"\s\s|^\s|\s$", text), name
        assert not any("\n" in text or "\t" in text for _ in [0]), name


def test_mergerate_builder_and_analyzer():
    from jev_observatory import arch_probe as ap
    from jev_observatory.arch_probe_mergerate import analyze_mergerate
    calls = ap.build_mergerate(reps=2)
    assert {c.family for c in calls} == {"mergerate"}
    rows = [{"family": "mergerate", "sample": c.meta["sample"], "script": c.meta["script"],
             "http_status": 200, "usage_input_tokens": 500 + c.meta["n_chars"],
             "state_chars": c.meta["n_chars"], "usage_output_tokens": 20,
             "n_chars": c.meta["n_chars"]}
            for c in calls]
    out = analyze_mergerate(rows)
    assert out["family"] == "mergerate" and len(out["samples"]) >= 20
    one = next(iter(out["samples"].values()))
    assert one["median_reported_in"] and one["n_chars"] > 0
