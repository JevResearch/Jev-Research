"""Letter-level Talk v2 tests: alphabet, run-banning, decoders, stops."""

from jev_observatory.char_talk import (
    END_TOKEN,
    AlphabetLM,
    CharTalkSession,
    LETTERS,
    alphabet_criteria_v2,
)
from jev_observatory.token_talk import (
    DecoderConfig,
    ScriptedTokenProvider,
    TokenTalkLimits,
)


def _provider(answers):
    return ScriptedTokenProvider(answers)


def _one(key):
    return {"choice": key, "probabilities": {key: 1.0}, "confidence": 1.0}


def test_alphabet_is_98_uniform_candidates():
    lm = AlphabetLM()
    cands = lm.score()
    assert len(cands) == 97
    assert [c.text for c in cands] == list(LETTERS)
    assert abs(cands[0].p_local - 1 / 97) < 1e-12
    assert len(alphabet_criteria_v2()) == 98  # + END = the attempt-1 alphabet size


class PickLetterProvider(ScriptedTokenProvider):
    """Always picks the offered key for `letter` if present, else END."""

    def __init__(self, letter: str, budget: int = 10) -> None:
        from jev_observatory.validation import validate_response
        from jev_observatory.providers import AttemptRecord, CallOutcome
        import json as _json

        self.letter = letter
        self.budget = budget
        self.calls = 0

    def ask(self, request, logical_request_id, **kwargs):
        import json as _json
        from jev_observatory.providers import AttemptRecord, CallOutcome
        from jev_observatory.validation import validate_response

        self.calls += 1
        criteria = list(request.questions["next_token"].criteria)
        # which key holds our letter, considering repr quoting in the text
        target = None
        for key, desc in request.questions["next_token"].criteria.items():
            if desc == f"the token {self.letter!r}":
                target = key
        if target is None or self.calls > self.budget:
            choice, probs = END_TOKEN, {END_TOKEN: 1.0}
        else:
            choice, probs = target, {k: (1.0 if k == target else 0.0) for k in criteria}
        # fill the full distribution over all offered options (server contract)
        probs = dict(probs)
        for key in criteria:
            probs.setdefault(key, 0.0)
        body = {"model": request.model, "answers": {"next_token":
                {"type": "choice", "choice": choice, "probabilities": probs,
                 "confidence": 1.0} if False else
                {"type": "choice", "choice": choice, "probabilities": probs,
                 "confidence": 1.0}},
                "usage": {"input_tokens": 10, "output_tokens": 2}}
        v = validate_response(body, request, http_status=200)
        rec = AttemptRecord(
            attempt_id=f"{logical_request_id}.0", logical_request_id=logical_request_id,
            attempt_index=0, provider="pick-letter", model_requested=request.model,
            started_at="", finished_at="", latency_ms=1.0, first_byte_ms=None,
            cold_connection=False, http_status=200, outcome="ok", uncertain=False,
            error=None, usage_input_tokens=10, usage_output_tokens=2,
            estimated_input_tokens=10, response_bytes=10,
            request_sha256=request.request_sha256())
        return CallOutcome(logical_request_id=logical_request_id, status="ok",
                           terminal=True, attempts=[rec], validated=v,
                           usage_input_tokens=10, usage_output_tokens=2,
                           total_latency_ms=1.0, n_retries=0,
                           request_sha256=request.request_sha256())


def test_triple_letter_is_impossible_but_double_is_fine():
    lm = AlphabetLM()
    session = CharTalkSession(
        "s1", PickLetterProvider("l"), lm, user_prompt="hi",
        limits=TokenTalkLimits(max_tokens=10),
        decoder=DecoderConfig(mode="jev_greedy"),
    )
    session.run()
    # Jev always wants 'l'; the ban makes the third offering impossible
    assert session.prefix == "ll"
    assert session.stop_reason in ("end_token", "no_candidates")


def test_word_with_double_letter_writable():
    lm = AlphabetLM()
    text = "letter"
    answers = []
    for ch in text:
        idx = LETTERS.index(ch)
        answers.append({"choice": f"tok_{idx:03d}",
                        "probabilities": {f"tok_{idx:03d}": 1.0}})
    answers.append({"choice": END_TOKEN, "probabilities": {END_TOKEN: 1.0}})
    session = CharTalkSession(
        "s2", _provider(answers), lm, user_prompt="hi",
        limits=TokenTalkLimits(max_tokens=10),
    )
    session.run()
    assert session.prefix == "letter"
    assert session.stop_reason == "end_token"


def test_space_runs_capped():
    lm = AlphabetLM()
    sp = LETTERS.index(" ")
    answers = [{"choice": f"tok_{sp:03d}", "probabilities": {f"tok_{sp:03d}": 1.0}}] * 5
    session = CharTalkSession(
        "s3", _provider(answers), lm, user_prompt="hi",
        limits=TokenTalkLimits(max_tokens=10),
    )
    session.run()
    assert "   " not in session.prefix  # never three spaces


def test_floor_and_nucleus_reach_char_decoding():
    lm = AlphabetLM()
    a_idx, b_idx = LETTERS.index("a"), LETTERS.index("b")
    provider = _provider([
        {"choice": f"tok_{a_idx:03d}",
         "probabilities": {f"tok_{a_idx:03d}": 0.60, f"tok_{b_idx:03d}": 0.20, "END": 0.20}},
        {"choice": END_TOKEN, "probabilities": {END_TOKEN: 1.0}},
    ])
    session = CharTalkSession(
        "s4", provider, lm, user_prompt="hi",
        limits=TokenTalkLimits(max_tokens=5),
        decoder=DecoderConfig(mode="jev_sample", temperature=0.8, top_p=0.9,
                              floor=0.1, seed=3),
    )
    session.run()
    # floor strips END's 0.20 raw mass to zero before temperature; nucleus may
    # still retain a renormalized END to reach 0.9 cumulative mass, but the
    # transformed value must be far below the raw 0.20
    node = session.nodes["n0000"]
    assert node.transformed is not None
    assert node.transformed.get("END", 0) < 0.2
