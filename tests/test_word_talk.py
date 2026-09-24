"""Word-completion Talk tests: menu structure, emission, backspace, stops."""

import pytest

from jev_observatory.char_talk import END_TOKEN
from jev_observatory.token_talk import DecoderConfig, TokenTalkLimits
from jev_observatory.vocab import build_vocab
from jev_observatory.word_talk import (
    BACKSPACE,
    WordTalkSession,
    build_options,
)


@pytest.fixture(scope="module")
def vocab():
    return build_vocab()


class TextPicker:
    """Stateless provider: picks the option whose *display text* matches a
    script of texts (or END); asserts when the script's option isn't offered."""

    name = "text-picker"

    def __init__(self, texts: list[str]) -> None:
        self.texts = list(texts)

    def ask(self, request, logical_request_id, **kwargs):
        from jev_observatory.providers import AttemptRecord, CallOutcome
        from jev_observatory.validation import validate_response

        want = self.texts.pop(0)
        criteria = request.questions["next_word"].criteria
        if want == END_TOKEN:
            choice = END_TOKEN
        else:
            matches = []
            for k, d in criteria.items():
                if d == f"write {want!r}" or d == f"punctuation: {want}":
                    matches.append(k)
                elif d.startswith("append the letter, making "):
                    # display may include capitalized earlier letters: 'Hi' for 'i'
                    core = d[len("append the letter, making "):-1]
                    if core == want or core.endswith(want):
                        matches.append(k)
            assert matches, f"option {want!r} not offered"
            choice = matches[0]
        probs = {k: (1.0 if k == choice else 0.0)
                 for k in list(criteria) + [END_TOKEN]}
        body = {"model": request.model, "answers": {"next_word": {
            "type": "choice", "choice": choice, "probabilities": probs,
            "confidence": 1.0}},
            "usage": {"input_tokens": 10, "output_tokens": 2}}
        validated = validate_response(body, request, http_status=200)
        record = AttemptRecord(
            attempt_id=f"{logical_request_id}.0",
            logical_request_id=logical_request_id, attempt_index=0,
            provider=self.name, model_requested=request.model,
            started_at="", finished_at="", latency_ms=1.0, first_byte_ms=None,
            cold_connection=False, http_status=200, outcome="ok", uncertain=False,
            error=None, usage_input_tokens=10, usage_output_tokens=2,
            estimated_input_tokens=10, response_bytes=10,
            request_sha256=request.request_sha256())
        return CallOutcome(logical_request_id=logical_request_id, status="ok",
                           terminal=True, attempts=[record], validated=validated,
                           usage_input_tokens=10, usage_output_tokens=2,
                           total_latency_ms=1.0, n_retries=0,
                           request_sha256=request.request_sha256())

    def close(self):
        pass


def _session(provider, vocab, **kwargs):
    defaults = dict(user_prompt="hi", limits=TokenTalkLimits(max_tokens=200),
                    decoder=DecoderConfig(mode="jev_greedy"))
    defaults.update(kwargs)
    return WordTalkSession("w1", provider, vocab, **defaults)


# ------------------------------------------------------------------ menu
def test_menu_helicop_case(vocab):
    opts = build_options("i saw a ", "helicop", vocab)
    texts = {o.text for o in opts}
    assert len(opts) <= 255
    assert "helicopa" in texts and "helicopz" in texts      # merged letters
    assert "helicopter" in texts and "helicopters" in texts   # whole words
    assert "helicopt" in texts                                # 2-letter combo
    assert "helicopter a" in texts and "helicopters b" in texts
    assert any(o.text.startswith("helicopter ") and len(o.text.split()[-1]) == 2
               for o in opts if o.kind == "nextword_combo")
    assert "[space (end-of-word)]" in texts
    assert "[backspace (remove last letter)]" in texts
    assert "[end (finish the response)]" in texts


def test_menu_start_offers_common_whole_words(vocab):
    opts = build_options("", "", vocab)
    texts = [o.text for o in opts if o.kind == "word"]
    assert "the" in texts and "and" in texts
    assert len([o for o in opts if o.kind == "letter"]) == 26


def test_menu_stuck_escape(vocab):
    opts = build_options("my name is ", "jzqg", vocab)
    texts = {o.text for o in opts}
    assert "jzqg a" in texts and "jzqg z" in texts   # endorse-and-move-on
    assert "jzqga" in texts                          # may insist on garbage
    assert "[space (end-of-word)]" in texts


def test_menu_forced_end_caps_partial(vocab):
    partial = "x" * 15
    opts = build_options("", partial, vocab)
    assert [o for o in opts if o.kind in ("letter", "combo")] == []
    assert any(o.kind == "escape" for o in opts)


def test_run_ban_no_triple_letters(vocab):
    opts = build_options("", "ll", vocab)
    assert "lll" not in {o.text for o in opts}
    opts2 = build_options("", "lett", vocab)
    assert any(o.text == "letter" for o in opts2)   # doubles stay writable


# ------------------------------------------------------------------ emission
def test_session_writes_and_capitalizes(vocab):
    provider = TextPicker(["h", "i", "[space (end-of-word)]", "t", "h", "e",
                           "r", "e", "[period . (end-of-sentence)]", END_TOKEN])
    session = _session(provider, vocab)
    session.run()
    assert session.prefix == "Hi there. "
    nodes = list(session.nodes.values())
    assert nodes[0]["cased"] is True          # sentence start capitalized
    assert nodes[3]["cased"] is False         # 'there' correctly lowercase
    assert session.stop_reason == "end_token"


def test_session_backspace_edits(vocab):
    provider = TextPicker(["h", "e", "j", "[backspace (remove last letter)]",
                           "l", "l", "o", "[space (end-of-word)]", END_TOKEN])
    session = _session(provider, vocab)
    session.run()
    assert session.prefix == "Hello "
    assert any(n["deletes"] == 1 for n in session.nodes.values())


def test_stuck_escape_emits_space_plus_letter(vocab):
    provider = TextPicker(["j", "z", "q", "g", "Jzqg a", END_TOKEN])
    session = _session(provider, vocab)
    session.run()
    assert session.prefix == "Jzqg a"


def test_identical_words_capped_at_four_by_anti_repeat(vocab):
    # the structural adjacent-doubling ban removes the letter option whose
    # emission would create a fifth identical word; Jev cannot loop words
    from jev_observatory.providers import AttemptRecord, CallOutcome
    from jev_observatory.validation import validate_response

    class ZOrEnd:
        name = "z-or-end"

        def __init__(self) -> None:
            self.want_space = False

        def ask(self, request, logical_request_id, **kwargs):
            criteria = request.questions["next_word"].criteria
            if self.want_space:
                choice = next(k for k, d in criteria.items()
                              if d == "punctuation: [space (end-of-word)]")
                self.want_space = False
            else:
                z_keys = [k for k, d in criteria.items()
                          if d.startswith("append the letter, making ")
                          and d[len("append the letter, making "):-1].endswith("z")]
                if z_keys:
                    choice = z_keys[0]
                    self.want_space = True
                else:
                    choice = END_TOKEN
            probs = {k: (1.0 if k == choice else 0.0)
                     for k in list(criteria) + [END_TOKEN]}
            body = {"model": request.model, "answers": {"next_word": {
                "type": "choice", "choice": choice, "probabilities": probs,
                "confidence": 1.0}},
                "usage": {"input_tokens": 10, "output_tokens": 2}}
            validated = validate_response(body, request, http_status=200)
            record = AttemptRecord(
                attempt_id=f"{logical_request_id}.0",
                logical_request_id=logical_request_id, attempt_index=0,
                provider=self.name, model_requested=request.model,
                started_at="", finished_at="", latency_ms=1.0,
                first_byte_ms=None, cold_connection=False, http_status=200,
                outcome="ok", uncertain=False, error=None,
                usage_input_tokens=10, usage_output_tokens=2,
                estimated_input_tokens=10, response_bytes=10,
                request_sha256=request.request_sha256())
            return CallOutcome(logical_request_id=logical_request_id,
                               status="ok", terminal=True, attempts=[record],
                               validated=validated, usage_input_tokens=10,
                               usage_output_tokens=2, total_latency_ms=1.0,
                               n_retries=0, request_sha256=request.request_sha256())

        def close(self):
            pass

    session = _session(ZOrEnd(), vocab)
    session.run()
    assert session.stop_reason == "end_token"
    # three identical words writable; the fourth 'z' would complete the
    # period-4 doubling "z z z z " and is structurally banned, so the
    # provider falls through to END
    assert session.prefix == "Z z z "
    assert len(session.prefix.split()) == 3


def test_decoder_modes_limited(vocab):
    with pytest.raises(ValueError):
        _session(TextPicker([END_TOKEN]), vocab,
                 decoder=DecoderConfig(mode="product"))


def test_zero_emit_word_option_never_offered(vocab):
    # the partial is a complete word: a 'commit it again' option would be a
    # no-op loop; only extensions and next-word starts may appear
    opts = build_options("my name is ", "an", vocab)
    for o in opts:
        assert o.emit != ""


def test_recent_word_ban_tracks_text_not_deltas(vocab):
    # 'an' completed via extension 'n': the word 'an' must be banned next step
    opts = build_options("The ", "an", vocab, banned_words=frozenset({"an", "a", "the"}))
    assert not any(o.text == "an" for o in opts)
    assert any(o.text == "and" for o in opts)
