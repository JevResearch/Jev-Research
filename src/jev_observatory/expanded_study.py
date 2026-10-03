"""Offline planning helpers for the owner-authorized expanded tokenizer study.

This module deliberately stops at auditable candidate generation, counting and
freezing.  It does not dispatch requests or contain political prompts.
"""
from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import asdict, dataclass
from typing import Callable, Iterable, Sequence


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    family: str
    text: str
    variant: str = "raw"


@dataclass(frozen=True)
class CountedCandidate:
    candidate: Candidate
    reported: int
    offline: int


@dataclass(frozen=True)
class MatchedPair:
    pair_id: str
    relation: str
    left: str
    right: str
    reported_gap: int
    offline_gap: int


def candidate_pool() -> list[Candidate]:
    """Return a deterministic, synthetic pool with within-script contrasts."""
    groups = {
        "latin": ("ordinary words", "antidisestablishmentarianism", "camelCaseHTTPParser", "rare qzvj xkpt"),
        "cyrillic": ("обычные слова", "вжцфхъщ", "тестирование модели"),
        "greek": ("κανονικές λέξεις", "ξψφθχ", "δοκιμή μοντέλου"),
        "cjk": ("自然语言处理", "自然 語言 處理", "漢字複合語"),
        "arabic": ("لغة طبيعية", "ضظثخ",),
        "hebrew": ("שפה טבעית", "אבגדהוזח",),
        "code": ("def parse_http_header(value): return value.strip()", "HTTPRequest2JSON_v17", "0xDEADBEEF + 1234567890"),
        "numeric": ("1 2 3 4 5", "12345678901234567890", "3.141592653589793"),
    }
    out: list[Candidate] = []
    # Deterministic compounds create independent families rather than merely
    # repeating a small set of strings.  IDs are opaque to any model payload.
    for family, values in groups.items():
        for i, text in enumerate(values):
            variants = tuple(
                [text, text + " " + text, text.replace(" ", ""),
                 text + "-v" + str(i), text + "  " + str(i)]
                + [text + (" " * (j % 4 + 1)) + str(j) + "_token"
                   for j in range(15)]
            )
            for j, value in enumerate(variants):
                base = Candidate(f"{family}-{i}-{j}", family, value)
                out.append(base)
                nfc = unicodedata.normalize("NFC", value)
                if nfc != value:
                    out.append(Candidate(f"{family}-{i}-{j}-nfc", family, nfc, "nfc"))
    return out


def count_candidates(candidates: Iterable[Candidate], reported_counter: Callable[[str], int],
                     offline_counter: Callable[[str], int]) -> list[CountedCandidate]:
    return [CountedCandidate(c, int(reported_counter(c.text)), int(offline_counter(c.text)))
            for c in candidates]


def select_pairs(rows: Sequence[CountedCandidate], *, minimum: int = 24,
                 reported_tolerance: float = .02, offline_ratio: float = 2.0) -> list[MatchedPair]:
    """Select count-disagreement pairs without using latency or probabilities.

    Candidates are paired only within a script family.  The two directions are
    represented by ``reported-matched`` and ``offline-matched`` relations.
    """
    result: list[MatchedPair] = []
    used: set[str] = set()
    for relation in ("reported-matched", "offline-matched"):
        for i, left in enumerate(rows):
            if left.candidate.candidate_id in used:
                continue
            for right in rows[i + 1:]:
                if right.candidate.candidate_id in used:
                    continue
                if left.candidate.family != right.candidate.family:
                    continue
                rg = abs(left.reported - right.reported)
                og = abs(left.offline - right.offline)
                if relation == "reported-matched":
                    denom = max(1, max(left.reported, right.reported))
                    ok = rg / denom <= reported_tolerance and (max(left.offline, right.offline) >= offline_ratio * max(1, min(left.offline, right.offline)) or og >= 8)
                else:
                    denom = max(1, max(left.offline, right.offline))
                    ok = og / denom <= reported_tolerance and (max(left.reported, right.reported) >= offline_ratio * max(1, min(left.reported, right.reported)) or rg >= 8)
                if ok:
                    result.append(MatchedPair(f"p{len(result):04d}", relation,
                        left.candidate.candidate_id, right.candidate.candidate_id, rg, og))
                    used.update((left.candidate.candidate_id, right.candidate.candidate_id))
                    break
    return result


def freeze_plan(pairs: Sequence[MatchedPair], seed: int = 20261003) -> dict:
    """Serialize an immutable pre-timing plan and its hash."""
    body = {"schema": 1, "seed": seed, "pairs": [asdict(p) for p in pairs]}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    body["plan_sha256"] = hashlib.sha256(encoded).hexdigest()
    return body
