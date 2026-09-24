"""Schema contract tests: option/level limits, payload shape, option maps."""

import json
import math

import pytest

from jev_observatory.schema import (
    MAX_CHOICE_OPTIONS,
    ChoiceQuestion,
    NoulQuestion,
    ScoreQuestion,
    SystemOneRequest,
    estimate_input_tokens,
)


def _choice(n: int) -> ChoiceQuestion:
    return ChoiceQuestion(instructions="pick", criteria={f"opt_{i:03d}": f"desc {i}" for i in range(n)})


def test_choice_2_options_valid():
    question = _choice(2)
    assert len(question.options) == 2


def test_choice_255_options_valid_256_rejected():
    question = _choice(MAX_CHOICE_OPTIONS)
    assert len(question.options) == 255
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="255"):
        _choice(256)


def test_choice_rejects_duplicate_keys():
    # Duplicate keys are impossible in a dict literal (last wins at parse time),
    # so this is a no-op by construction; documented here to avoid a false test.
    question = ChoiceQuestion(instructions="x", criteria={"a": "1", "b": "2", "a": "3"})
    assert list(question.criteria) == ["a", "b"]


def test_choice_needs_two_options():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ChoiceQuestion(instructions="x", criteria={"only": "one"})


def test_score_levels_2_to_10():
    import pytest
    from pydantic import ValidationError

    assert ScoreQuestion(instructions="s", criteria=["low", "high"]).levels == ["0", "1"]
    assert len(ScoreQuestion(instructions="s", criteria=[str(i) for i in range(10)]).levels) == 10
    with pytest.raises(ValidationError):
        ScoreQuestion(instructions="s", criteria=["only"])
    with pytest.raises(ValidationError):
        ScoreQuestion(instructions="s", criteria=[str(i) for i in range(11)])


def test_noul_optional_criteria_keys():
    import pytest
    from pydantic import ValidationError

    NoulQuestion(instructions="is it so?")
    NoulQuestion(instructions="is it so?", criteria={"true": "yes means yes", "false": "no"})
    with pytest.raises(ValidationError):
        NoulQuestion(instructions="x", criteria={"true": "t", "maybe": "?"})


def test_payload_keeps_null_option_descriptions():
    """Vendor docs explicitly use null descriptions; payload must not drop them."""
    request = SystemOneRequest(
        state="state",
        questions={"tone": ChoiceQuestion(instructions="tone?", criteria={"happy": None, "sad": "feels sad"})},
    )
    criteria = request.to_payload()["questions"]["tone"]["criteria"]
    assert criteria == {"happy": None, "sad": "feels sad"}
    assert "null" in request.canonical_json()


def test_payload_top_level_keys_only():
    request = SystemOneRequest(
        state="s",
        model="jev-1.13.0",
        questions={"n": NoulQuestion(instructions="q")},
    )
    payload = request.to_payload()
    assert set(payload) == {"state", "model", "questions"}
    assert set(payload["questions"]["n"]) <= {"type", "instructions", "criteria"}


def test_request_sha256_deterministic_and_order_insensitive():
    a = SystemOneRequest(state="s", questions={"x": NoulQuestion(instructions="q"), "y": NoulQuestion(instructions="r")})
    b = SystemOneRequest(state="s", questions={"y": NoulQuestion(instructions="r"), "x": NoulQuestion(instructions="q")})
    assert a.request_sha256() == b.request_sha256()
    assert len(a.request_sha256()) == 64


def test_option_maps_round_trip():
    request = SystemOneRequest(
        state="s",
        questions={
            "c": ChoiceQuestion(instructions="c", criteria={"alpha": "a desc", "beta": "b desc", "gamma": None}),
            "s": ScoreQuestion(instructions="s", criteria=["low", "mid", "high"]),
            "n": NoulQuestion(instructions="n"),
        },
    )
    maps = request.option_maps()
    assert maps["c"] == {"type": "choice", "options": ["alpha", "beta", "gamma"]}
    assert maps["s"] == {"type": "score", "levels": ["0", "1", "2"]}
    assert maps["n"] == {"type": "noul", "keys": ["true", "false"]}


def test_measures_track_question_and_candidate_counts():
    request = SystemOneRequest(
        state="s",
        questions={
            "c": _choice(4),
            "n": NoulQuestion(instructions="q"),
            "s": ScoreQuestion(instructions="s", criteria=["a", "b", "c"]),
        },
    )
    measures = request.measures()
    assert measures["q"] == 3
    assert measures["k_per_question"]["c"] == 4
    assert measures["c_total_candidates"] == 7
    assert measures["primitive_mix"] == {"noul": 1, "choice": 1, "score": 1}


def test_estimate_input_tokens_is_conservative():
    # One token per character: an over-estimate that stops the budget early.
    assert estimate_input_tokens(1000) == 1001
    assert estimate_input_tokens(0) == 1


def test_empty_questions_rejected():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        SystemOneRequest(state="s", questions={})