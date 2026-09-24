"""Letter-by-letter Talk, attempt 2: the token protocol over a char alphabet.

Re-applies the lessons from the token-level experiments to DESIGN.md §9's
original 98-option character alphabet (95 printable ASCII + newline + tab +
END), without touching ``talk.py`` (attempt 1):

* **Repeated letters are removed from the menu**: a letter that has already
  occurred twice in a row is not offered, so ``lll`` is impossible while
  ``letter`` (one ``tt``) stays writable.  This replaces the attempt-1
  post-hoc repetition guard with structural prevention, as done for words in
  the token session.
* **Sampling controls**: the quantization floor is subtracted and the
  remainder renormalized; temperature and nucleus/top-k are available, all
  recorded alongside the raw distribution.
* Same state discipline (distinct user_prompt / previous_turns /
  assistant_prefix), same trace honesty, same explicit stops.
"""

from __future__ import annotations

from typing import Any

from .token_talk import (
    TokenCandidate,
    TokenTalkSession,
)

END_TOKEN = "END"
LETTERS: tuple[str, ...] = tuple(chr(code) for code in range(32, 127)) + ("\n", "\t")


class AlphabetLM:
    """A degenerate LocalLM: scores the fixed character alphabet uniformly.

    There is no local language model here — every character is offered at
    equal weight, in native order.  Stopping is Jev's END choice only.
    """

    name = "alphabet-98"
    special_token_ids = frozenset()

    def __init__(self) -> None:
        self._candidates = [
            TokenCandidate(token_id=ord(char), text=char, p_local=1.0 / len(LETTERS))
            for char in LETTERS
        ]

    def prepare(self, *, user_prompt: str, previous_turns: list[dict[str, str]]) -> None:
        return None

    def advance(self, token_id: int) -> None:
        return None

    def score(self, limit: int | None = None) -> list[TokenCandidate]:
        return list(self._candidates)

    def save_state(self) -> bytes:
        return b""

    def load_state(self, state: bytes) -> None:
        return None

    def detokenize_text(self, token_id: int) -> str | None:
        return chr(token_id) if 0 <= token_id < 0x110000 else None


class CharTalkSession(TokenTalkSession):
    """TokenTalkSession over the character alphabet, with letter-run banning.

    A character that has occurred twice in a row at the tail of the prefix is
    not offered again (``lll`` impossible, ``letter`` fine).  The inherited
    word-level bans, repetition guards, decoders, and trace export carry
    over unchanged.
    """

    def _recent_words(self, depth: int = 4) -> frozenset[str]:
        prefix = self.prefix
        if len(prefix) >= 2 and prefix[-1] == prefix[-2]:
            return frozenset({prefix[-1]})
        return frozenset()


def alphabet_criteria_v2() -> dict[str, str]:
    """The 98-option criteria as Jev sees them (repr-quoted characters)."""
    from .token_talk import candidate_criteria

    return candidate_criteria(
        [TokenCandidate(token_id=ord(c), text=c, p_local=0.0) for c in LETTERS]
    )
