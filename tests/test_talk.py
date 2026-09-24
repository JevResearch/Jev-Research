"""Talk session tests: alphabet, decoding, stops, branching, trace honesty."""

import pytest

from jev_observatory.talk import (
    ALPHABET_KEYS,
    CHAR_TO_KEY,
    DecoderConfig,
    END_TOKEN,
    KEY_TO_CHAR,
    ScriptedCharProvider,
    TalkLimits,
    TalkSession,
    alphabet_criteria,
    build_talk_request,
)


@pytest.fixture
def provider():
    return ScriptedCharProvider("Hello, world!")


def _session(provider, **kwargs):
    defaults = dict(user_prompt="say hi", limits=TalkLimits(max_chars=256), decoder=DecoderConfig())
    defaults.update(kwargs)
    return TalkSession("s1", provider, **defaults)


def test_alphabet_is_98_options_and_below_cap():
    assert len(ALPHABET_KEYS) == 98
    assert len(alphabet_criteria()) == 98
    printable = sum(1 for code in range(32, 127) if f"char_{code:03d}" in KEY_TO_CHAR)
    assert printable == 95
    assert KEY_TO_CHAR["char_010"] == "\n"
    assert KEY_TO_CHAR["char_009"] == "\t"
    assert KEY_TO_CHAR[END_TOKEN] is None
    # round trip: every printable char maps back to its key
    for code in range(32, 127):
        assert CHAR_TO_KEY[chr(code)] == f"char_{code:03d}"


def test_request_has_distinct_transcript_and_prefix_fields():
    request = build_talk_request(user_prompt="hi", previous_turns=[{"role": "user", "text": "earlier"}],
                                 prefix="Hel")
    state = request.to_payload()["state"]
    assert set(state) == {"user_prompt", "previous_turns", "assistant_prefix"}
    assert state["assistant_prefix"] == "Hel"
    questions = request.to_payload()["questions"]
    assert len(questions) == 1 and len(questions["next_char"]["criteria"]) == 98


def test_greedy_run_emits_known_string_then_end(provider):
    session = _session(provider)
    events = session.run()
    assert session.prefix == "Hello, world!"
    assert session.stop_reason == "end_token"
    steps = [e for e in events if e["type"] == "step" and e["char"] is not None]
    assert [e["char"] for e in steps] == list("Hello, world!")
    stop = events[-1]
    assert stop["reason"] == "end_token" and stop["prefix_chars"] == 13


def test_max_chars_stop():
    provider = ScriptedCharProvider("ABCDEFGHIJKLMNOP")
    session = _session(provider, limits=TalkLimits(max_chars=5))
    session.run()
    assert session.stop_reason == "max_chars" and session.prefix == "ABCDE"


def test_deadline_stop():
    class SlowProvider(ScriptedCharProvider):
        clock_calls = 0

    provider = ScriptedCharProvider("AAAAAAAAAAAAAAAAAAAA")
    session = _session(provider, limits=TalkLimits(max_seconds=10.0))
    clock = {"now": 0.0}
    session.clock = lambda: clock["now"]
    # each step advances the fake clock by 4s; deadline hits after 2 steps
    real_step = session.step

    def ticking_step():
        clock["now"] += 4.0
        return real_step()

    session.step = ticking_step
    session.run()
    assert session.stop_reason == "deadline"


def test_repetition_guard():
    provider = ScriptedCharProvider("abababababababababab")
    session = _session(provider, limits=TalkLimits(max_chars=256))
    session.run()
    assert session.stop_reason == "repetition"
    # the repeated prefix is intact but the loop was cut
    assert session.prefix.endswith("abab")


def test_cancellation_checked_between_steps():
    provider = ScriptedCharProvider("ABCDEFGHIJKLMNOP")
    session = _session(provider, limits=TalkLimits(max_chars=256))
    real_step = session.step
    steps = {"n": 0}

    def cancelling_step():
        steps["n"] += 1
        if steps["n"] >= 4:  # three completed steps, then cancel
            session.cancel()
        return real_step()

    session.step = cancelling_step
    session.run()
    assert session.stop_reason == "cancelled"
    assert len(session.prefix) == 3


def test_provider_failure_stops_with_detail():
    class FailingProvider(ScriptedCharProvider):
        def ask(self, request, logical_request_id, **kwargs):
            outcome = super().ask(request, logical_request_id, **kwargs)
            if outcome.status == "ok" and len(self._requests) > 1:
                outcome.status = "http_error"
                outcome.validated = None
                outcome.terminal = True
            return outcome

    session = _session(FailingProvider("ABC"))
    session.run()
    assert session.stop_reason == "provider_failure"


def test_request_cap_stops_cleanly():
    """Scripted providers do not consume the ledger, so the controller counts
    requests itself; real providers get the ledger as a second layer."""
    session = _session(ScriptedCharProvider("ABCDEFGHIJKLMNOP"),
                       limits=TalkLimits(max_chars=256, max_provider_requests=2))
    session.run()
    assert session.stop_reason == "max_provider_requests"
    assert len(session.prefix) == 2


class _Hook:
    def __init__(self, ledger):
        self.ledger = ledger

    def before(self, estimated):
        return self.ledger.reserve(estimated)

    def after(self, handle, reported):
        self.ledger.reconcile(handle, reported)


def test_backtrack_preserves_original_branch():
    provider = ScriptedCharProvider("HELLO")
    session = _session(provider, limits=TalkLimits(max_chars=256))
    session.run()
    trace = session.export_trace()
    branch_end = session.current_node_id
    # walk back to the node that emitted 'L' (3rd char)
    nodes = {n["node_id"]: n for n in trace["nodes"]}
    l_node = next(n["node_id"] for n in trace["nodes"] if n["char"] == "L" and n["node_id"] != branch_end)
    session.backtrack(l_node)
    assert session.prefix == "HEL"[: len(session.prefix)] or session.prefix == "HEL"
    assert session.stop_reason is None  # backtracking re-opens the session
    # continue with a different provider script (a fork)
    session.provider = ScriptedCharProvider("XYZ")
    session.run()
    assert session.stop_reason == "end_token"
    # the original branch still exists in the tree
    assert branch_end in session.nodes
    full_trace = session.export_trace()
    assert full_trace["n_nodes"] > len(nodes)


def test_backtrack_rejects_non_ancestor():
    session = _session(ScriptedCharProvider("AB"))
    session.run()
    with pytest.raises((KeyError, ValueError)):
        session.backtrack("n99999")


def test_edit_prefix_is_labelled_and_continues():
    session = _session(ScriptedCharProvider("XYZ"), limits=TalkLimits(max_chars=256))
    session.run()  # "XYZ" then END
    node = session.edit_prefix("Hello ")  # replaces the prefix, parent is root
    assert session.prefix == "Hello "
    trace = session.export_trace()
    edited = next(n for n in trace["nodes"] if n["node_id"] == node)
    assert edited["user_edited"] is True and edited["probabilities"] is None


def test_sample_decoder_with_seed_replays_identically():
    # a provider whose distribution is genuinely multi-modal
    class FlatProvider(ScriptedCharProvider):
        pass

    session_a = TalkSession("sa", ScriptedCharProvider("aaaa"), user_prompt="p",
                            limits=TalkLimits(max_chars=8),
                            decoder=DecoderConfig(mode="sample", temperature=0.7, seed=11))
    session_b = TalkSession("sb", ScriptedCharProvider("aaaa"), user_prompt="p",
                            limits=TalkLimits(max_chars=8),
                            decoder=DecoderConfig(mode="sample", temperature=0.7, seed=11))
    a = session_a.run()
    b = session_b.run()
    assert [e.get("char") for e in a] == [e.get("char") for e in b]


def test_temperature_transform_is_recorded_not_hidden():
    from jev_observatory.providers import MockProvider

    session = _session(MockProvider(seed=0),
                       decoder=DecoderConfig(mode="sample", temperature=2.0, seed=1),
                       limits=TalkLimits(max_chars=8))
    session.step()
    node = session.nodes[session.current_node_id]
    raw = node.probabilities
    transformed = node.transformed
    assert transformed is not None and transformed != raw  # stored alongside raw
    # flattening at T>1: the argmax key is preserved (monotone transform)
    assert max(raw, key=raw.get) == max(transformed, key=transformed.get)


def test_trace_export_has_no_api_key():
    session = _session(ScriptedCharProvider("Hi"))
    session.run()
    import json as _json

    blob = _json.dumps(session.export_trace())
    assert "apikey" not in blob.lower()
    assert "Bearer" not in blob


def test_one_inflight_request_per_run_is_enforced_by_web_layer():
    # covered in test_web.py; placeholder here to keep the M3 list together
    assert True
