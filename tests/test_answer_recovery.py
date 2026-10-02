"""Deterministic answer-recovery parser tests (matched cheap baselines).

Offline only.  Pins the documented stage order and the honesty rules:
actual allowed keys (MATH ``o0-o3``, MMLU/HLE keys beyond ``J``), token
boundaries only, unambiguous explicit final-answer preference, no gold
access, no indiscriminate first-character guessing — unrecovered is returned
rather than guessing, so recovery never inflates accuracy.
"""

from __future__ import annotations

import pytest

from jev_observatory.answer_recovery import (
    RECOVERY_SPEC_VERSION,
    recover_choice,
    recovery_spec,
)

MATH_KEYS = ["o0", "o1", "o2", "o3"]
WIDE_KEYS = list("ABCDEFGHIJKL")   # MMLU/HLE keys extend beyond J


def test_recovery_spec_is_machine_readable_and_versioned():
    spec = recovery_spec()
    assert spec["version"] == RECOVERY_SPEC_VERSION
    assert spec["stages"][0] == "exact"
    assert "first_character_guessing" in spec["excluded_heuristics"]


# ---------------------------------------------------------------- exact keys
def test_exact_key_o1():
    assert recover_choice("o1", MATH_KEYS) == ("o1", "exact")
    assert recover_choice("  o1\n", MATH_KEYS) == ("o1", "exact")


def test_stripped_wrapped_key():
    assert recover_choice("(o1)", MATH_KEYS) == ("o1", "stripped")
    assert recover_choice("**B**", list("ABCD")) == ("B", "stripped")


# ------------------------------------------------------- explicit statements
def test_answer_pattern_math_keys():
    assert recover_choice("The answer is o1.", MATH_KEYS) == ("o1", "answer_pattern")
    assert recover_choice("Choose option o1.", MATH_KEYS) == ("o1", "answer_pattern")
    assert recover_choice("The answer is: o2", MATH_KEYS) == ("o2", "answer_pattern")


def test_answer_pattern_keys_beyond_j():
    assert recover_choice("The answer is K.", WIDE_KEYS) == ("K", "answer_pattern")
    assert recover_choice("The answer is L.", WIDE_KEYS) == ("L", "answer_pattern")
    assert recover_choice("I choose option K.", WIDE_KEYS) == ("K", "answer_pattern")


def test_final_explicit_statement_wins():
    assert recover_choice("The answer is A. Wait, the answer is B.",
                          list("ABCD")) == ("B", "answer_pattern")


def test_explicit_statement_beats_earlier_mentions():
    text = "I first considered A, but the answer is C."
    assert recover_choice(text, list("ABCD")) == ("C", "answer_pattern")


# ------------------------------------------------------------ never guessing
def test_no_indiscriminate_first_character_guessing():
    # the OLD first_char stage recovered the first letter of any prose
    assert recover_choice("Absolutely not sure.", list("ABCD")) == (None, "unrecovered")
    assert recover_choice("Because of symmetry.", list("ABCD")) == (None, "unrecovered")


def test_key_substrings_never_match():
    assert recover_choice("o10", MATH_KEYS) == (None, "unrecovered")
    assert recover_choice("The answer is o10.", MATH_KEYS) == (None, "unrecovered")
    assert recover_choice("co1l", MATH_KEYS) == (None, "unrecovered")
    assert recover_choice("cat", list("ABCD")) == (None, "unrecovered")
    assert recover_choice("TAB", list("ABCD")) == (None, "unrecovered")
    # lowercase article 'a' must never recover key 'A'
    assert recover_choice("a simple sentence", list("ABCD")) == (None, "unrecovered")


def test_ambiguous_explanatory_mentions_recover_nothing():
    # multiple distinct key mentions with no explicit answer: ambiguous
    assert recover_choice("A is plausible, but B is often chosen.", list("ABCD")) \
        == (None, "unrecovered")
    assert recover_choice("Some say A; others say C.", list("ABCD")) \
        == (None, "unrecovered")
    assert recover_choice("The answer is either A or B.", list("ABCD")) \
        == (None, "unrecovered")
    assert recover_choice("The answer is A or B.", list("ABCD")) == (None, "unrecovered")


def test_negated_option_statement_never_recovers():
    assert recover_choice("Option C is wrong.", list("ABCD")) == (None, "unrecovered")
    assert recover_choice("The answer is not D.", list("ABCD")) == (None, "unrecovered")


def test_negation_of_another_key_does_not_block_recovery():
    assert recover_choice("The answer is B, not C.", list("ABCD")) == ("B", "answer_pattern")


def test_single_distinct_mention_recovers():
    assert recover_choice("Given the constraints, B.", list("ABCD")) == ("B", "isolated_key")
    assert recover_choice("I would say B here, nothing else fits.", list("ABCD")) \
        == ("B", "isolated_key")


def test_repeated_single_key_recovers():
    assert recover_choice("B, definitely B.", list("ABCD")) == ("B", "isolated_key")


# --------------------------------------------------------- reasoning channel
def test_reasoning_channel_only_when_content_empty():
    assert recover_choice("", MATH_KEYS, reasoning="... The answer is o3.") == \
        ("o3", "reasoning_channel")
    assert recover_choice("The answer is o1.", MATH_KEYS, reasoning="The answer is o2.") == \
        ("o1", "answer_pattern")   # content wins; reasoning never overrides
    assert recover_choice("", list("ABCD"), reasoning="hmm A or B, unclear") == \
        (None, "unrecovered")


def test_case_handling_is_explicit():
    # single-char keys: exact case in free text, case-insensitive in explicit
    # answer statements only
    assert recover_choice("the answer is b.", list("ABCD")) == ("B", "answer_pattern")
    assert recover_choice("b is what I would say", list("ABCD")) == (None, "unrecovered")
    # multi-char keys fold case
    assert recover_choice("The answer is O1.", MATH_KEYS) == ("o1", "answer_pattern")


def test_empty_keys_refused():
    with pytest.raises(ValueError):
        recover_choice("A", [])

# --------------------- final_line_key (unambiguous terminal answer lines) ---
KEYS_ABCD = ["A", "B", "C", "D"]


def test_final_standalone_line_beats_earlier_letter_mentions():
    # the 51-row defect shape: explanation full of uppercase letters/equations,
    # final paragraph is a bare key
    text = ("Since A and B are eliminated and equation F reduces to G, "
            "we compare C against A again.\n\nD")
    assert recover_choice(text, KEYS_ABCD) == ("D", "final_line_key")


def test_final_line_key_outranks_earlier_explicit_statement():
    text = "The answer is C, because A and B fail.\n\nD"
    assert recover_choice(text, KEYS_ABCD) == ("D", "final_line_key")


def test_final_line_key_strips_markdown_and_punctuation():
    for wrapper in ("**D**", "(D)", "D.", "D:", "- D", "`D`", "**D**."):
        assert recover_choice(f"Because A and B differ.\n\n{wrapper}",
                              KEYS_ABCD) == ("D", "final_line_key"), wrapper


def test_final_line_non_key_rejected():
    # not a key on the final line -> falls through, never last-letter guess
    pred, stage = recover_choice("A and B are wrong.\n\ncat", KEYS_ABCD)
    assert pred != "D" and stage in ("unrecovered", "isolated_key")
    pred, stage = recover_choice("Equation F holds.\n\nD1", KEYS_ABCD)
    assert pred is None and stage == "unrecovered"   # substring "D" never


def test_final_line_disjunction_rejected():
    pred, stage = recover_choice("We saw A and B earlier.\n\nA or B", KEYS_ABCD)
    assert pred is None and stage == "unrecovered"


def test_final_line_key_resolves_ambiguity_but_not_plain_ambiguity():
    # final bare key resolves otherwise-ambiguous content ...
    assert recover_choice("Maybe A, maybe B.\n\nC", KEYS_ABCD) == ("C", "final_line_key")
    # ... while ambiguity without a terminal line stays unrecovered
    pred, stage = recover_choice("Maybe A, maybe B.", KEYS_ABCD)
    assert pred is None and stage == "unrecovered"


def test_final_line_key_math_style_keys():
    keys = ["o0", "o1", "o2", "o3"]
    assert recover_choice("o1 and o2 both appear in the derivation.\n\no3",
                          keys) == ("o3", "final_line_key")
    assert recover_choice("o1 and o2 appear.\n\no10", keys)[0] is None
