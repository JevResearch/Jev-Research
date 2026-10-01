"""Deterministic answer recovery for text baselines (matched cheap-model runs).

A documented, deterministic recovery parser applied IDENTICALLY to every
model's raw content.  It never sees the gold key and it never guesses.

Stages, in order (the returned stage records which one fired):

1. ``exact``           — the trimmed content is exactly one allowed key.
2. ``stripped``        — the content, stripped of wrapping punctuation/
                         quotes/backticks, is exactly one allowed key.
3. ``answer_pattern``  — an EXPLICIT answer statement ("The answer is X",
                         "final answer: X", "Choose option X.", "option X")
                         with the key matched as a full token against the
                         actual allowed keys.  The LAST valid explicit
                         statement wins (documented final-answer preference:
                         a later self-correction supersedes an earlier claim).
                         A statement is NOT an unambiguous explicit answer —
                         and never recovers — when it is negated ("option C is
                         wrong") or disjunctive ("A or B").
4. ``isolated_key``    — a full-token key mention, only when exactly ONE
                         distinct key is mentioned anywhere in the content.
                         Two or more distinct mentions are ambiguous and
                         return unrecovered — never "last mention wins".
5. ``reasoning_channel`` — stages 3 then 4 over the reasoning text, only when
                         the content is empty (thinking models that spent the
                         whole output budget in the reasoning channel); the
                         isolated-key scan in this stage is limited to the
                         trailing REASONING_TAIL_CHARS characters.
6. ``unrecovered``     — counted as wrong.

Explicitly excluded heuristics (they inflate recovered accuracy by guessing
and are GONE): first-character guessing, substring matching ("o10" is not
"o1"; "cat" contains neither "A" nor "a" as a token), case-folded
single-letter matching ("a" the article must never recover key "A"), and any
access to the gold key.

Key matching rules (token boundaries always):
* multi-character keys match case-insensitively ("O1" resolves to ``o1``);
* single-character keys match exactly as listed — in the explicit
  ``answer_pattern`` stage they additionally match case-insensitively (the
  answer statement makes the key unambiguous), in ``isolated_key`` they never
  do (so prose articles and ordinary words never recover a letter key).
"""

from __future__ import annotations

import re

RECOVERY_SPEC_VERSION = "answer-recovery-2.0.0"

STAGES = ("exact", "stripped", "answer_pattern", "isolated_key",
          "reasoning_channel", "unrecovered")

REASONING_TAIL_CHARS = 40

STRIP_CHARS = "*_`\"' \t\n()[]{}<>.,:;!?/|\\-+=~^%"
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_CUE_TOKENS = frozenset({"answer", "option", "choice", "key", "ans"})
# tokens skipped between a cue and its key ("the answer: A", "answer is B")
_CONNECTORS = frozenset({"is", "are", "was", "were", "the", "single", "best",
                         "correct", "final", "to", "be", "of", "my", "our",
                         "one", "answer", "option", "choice", "key", "given",
                         "therefore", "thus", "so", "then", "must", "would",
                         "will", ":", "=", "-", ">", "->", "→"})
_NEGATION_TOKENS = frozenset({"not", "wrong", "incorrect", "false", "never",
                              "unlikely", "impossible", "mistake", "except",
                              "cannot", "cant", "isnt", "wasnt", "no"})
_NEGATION_LEAD_TOKENS = frozenset({"not", "never", "no"})
_DISJUNCTION_TOKENS = frozenset({"or", "alternatively", "maybe", "perhaps"})
_EXPLICIT_WINDOW_TOKENS = 8
_QUALIFIER_SPAN = 3


def recovery_spec() -> dict:
    """Machine-readable recovery contract (recorded in future run freezes)."""
    return {
        "version": RECOVERY_SPEC_VERSION,
        "stages": list(STAGES),
        "key_matching": ("token boundaries only; multi-character keys "
                         "case-insensitive, single-character keys exact "
                         "(case-insensitive only inside explicit answer "
                         "statements)"),
        "explicit_preference": ("the LAST unambiguous explicit answer "
                                "statement wins; negated and disjunctive "
                                "statements never recover"),
        "ambiguity": ("two or more distinct isolated key mentions recover "
                      "nothing"),
        "excluded_heuristics": ["first_character_guessing", "substring_matching",
                                "case_folded_single_letter", "gold_access"],
        "reasoning_channel": (f"only when content is empty; explicit "
                              f"statements over full reasoning, else unique "
                              f"isolated key in the trailing "
                              f"{REASONING_TAIL_CHARS} chars"),
    }


def _resolve_token(token: str, keyset: set[str], *,
                   casefold: bool) -> str | None:
    """Map a token to an allowed key; None when it is not exactly one key."""
    if token in keyset:
        return token
    if casefold or len(token) > 1:
        folded = token.lower()
        hits = {k for k in keyset
                if len(k) == len(token) and k.lower() == folded}
        if len(hits) == 1:
            return next(iter(hits))
    return None


def _tokens(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]


def _explicit_answer(text: str, keyset: set[str]) -> str | None:
    """Last unambiguous explicit answer statement, or None.

    A cue token (answer/option/choice/key) followed within a short window by
    exactly one key token yields that key, unless the statement is negated
    ("option C is wrong") or disjunctive ("A or B") — those are explanatory
    mentions, never explicit answers.
    """
    toks = _tokens(text)
    found: str | None = None
    for i, (word, _s, _e) in enumerate(toks):
        if word.lower() not in _CUE_TOKENS:
            continue
        window = toks[i + 1:i + 1 + _EXPLICIT_WINDOW_TOKENS]
        for j, (w2, _s2, _e2) in enumerate(window):
            low = w2.lower()
            if low in _CONNECTORS:
                continue
            key = _resolve_token(w2, keyset, casefold=True)
            if key is None:
                break  # a non-connector, non-key word closes the window
            rest = [w.lower() for w, _s3, _e3 in window[j + 1:j + 5]]
            ambiguous = negated = False
            for k, token in enumerate(rest):
                later = rest[k + 1:k + 1 + _QUALIFIER_SPAN]
                targets_other_key = any(
                    _resolve_token(w, keyset, casefold=True) is not None
                    for w in later)
                if token in _DISJUNCTION_TOKENS and targets_other_key:
                    ambiguous = True   # "A or B" is not an explicit answer
                    break
                if token in _NEGATION_TOKENS:
                    if targets_other_key:
                        continue       # "answer is B, not C": C is negated
                    negated = True     # "option C is wrong" never recovers C
                    break
            if ambiguous or negated:
                break
            found = key
            break
    return found


def _mention_negated(toks: list[tuple[str, int, int]], idx: int,
                     keyset: set[str]) -> bool:
    """A key mention is negated when the text says it is wrong/not the answer.

    "option C is wrong" and "the answer is not D" negate their key; "the
    answer is B, not C" negates C — never B.
    """
    prev = [w.lower() for w, _s, _e in toks[max(0, idx - 2):idx]]
    if any(t in _NEGATION_LEAD_TOKENS for t in prev):
        return True
    for k, (w, _s, _e) in enumerate(toks[idx + 1:idx + 4]):
        if w.lower() in _NEGATION_TOKENS:
            later = [t for t, _s2, _e2 in toks[idx + 2 + k:idx + 5 + k]]
            if any(_resolve_token(t, keyset, casefold=True) is not None
                   for t in later):
                continue  # the negation targets another key
            return True
    return False


def _isolated_key(text: str, keyset: set[str]) -> str | None:
    """The single distinct non-negated full-token key mention, or None.

    Two or more distinct mentions are ambiguous and recover nothing.
    """
    toks = _tokens(text)
    seen: set[str] = set()
    for idx, (word, _s, _e) in enumerate(toks):
        key = _resolve_token(word, keyset, casefold=False)
        if key is None or _mention_negated(toks, idx, keyset):
            continue
        seen.add(key)
        if len(seen) > 1:
            return None  # ambiguous: never guess which mention wins
    return next(iter(seen)) if len(seen) == 1 else None


def recover_choice(text: str | None, keys: list[str],
                   reasoning: str | None = None) -> tuple[str | None, str]:
    """Deterministic, documented recovery. Returns (key|None, stage).

    ``keys`` are the ACTUAL allowed keys of the question (MATH keys are
    ``o0..o3``, some MMLU/HLE keys extend beyond ``J``) — never hardcoded
    letters.  Unrecovered answers count as wrong at the scoring stage.
    """
    if not keys:
        raise ValueError("recover_choice: keys must be a non-empty list")
    keyset = {str(k) for k in keys}
    content = (text or "").strip()
    if not content:
        if reasoning and reasoning.strip():
            hit = _explicit_answer(reasoning, keyset)
            if hit is not None:
                return hit, "reasoning_channel"
            hit = _isolated_key(reasoning.strip()[-REASONING_TAIL_CHARS:], keyset)
            if hit is not None:
                return hit, "reasoning_channel"
        return None, "unrecovered"
    if content in keyset:
        return content, "exact"
    direct = _resolve_token(content, keyset, casefold=False)
    if direct is not None and " " not in content:
        return direct, "exact"
    stripped = content.strip(STRIP_CHARS)
    if stripped in keyset:
        return stripped, "stripped"
    stripped_hit = _resolve_token(stripped, keyset, casefold=False)
    if stripped_hit is not None and " " not in stripped:
        return stripped_hit, "stripped"
    hit = _explicit_answer(content, keyset)
    if hit is not None:
        return hit, "answer_pattern"
    hit = _isolated_key(content, keyset)
    if hit is not None:
        return hit, "isolated_key"
    return None, "unrecovered"
