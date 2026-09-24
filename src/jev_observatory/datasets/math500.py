"""MATH-500 (HuggingFaceH4/MATH-500) — Jev-native adapter.

MATH-500 is a 500-problem representative subset of the Hendrycks MATH TEST
split, introduced as the evaluation slice in DeepSeek-R1 (arXiv:2501.12948).
It is public, byte-stable (single pinned JSONL), and late-2026 reference
scores are abundant — but the field treats it as saturated for frontier
reasoning systems (top ~97-99%), so its role here is a LONGITUDINAL anchor:
Jev's native protocol is direct-answer (no chain-of-thought, no reasoning),
which matches the PRE-reasoning era of the benchmark, not the saturated
reasoning-era scores.  Reference rows are labelled with their protocol.

Jev cannot emit free-form text, so the exact-match answers are expressed
natively:

* ``choice``  — when the gold answer is an integer, at least three
  deterministic numeric perturbations fit in the option budget, AND the
  problem statement does not already contain the gold value as a standalone
  number (answer-disclosure exclusion — MATH answers like ``3`` would
  otherwise be given away by symbols such as ``x^3`` in the problem), the
  item is a 4-option MCQ: gold + 3 numeric distractors (deterministic
  derivation from the gold, seeded per item), keys shuffled by a fixed
  per-item seed.  The problem text is amended with an exact serialization of
  the answer format, so the question remains well-posed ("give your answer
  as ...").
* ``score``   — when the gold answer is a non-negative value below 1000, the
  item is a fixed-width decimal readout through the 10-level rubric: level i
  means digit i, asked per digit position (integer part most-significant
  first, then exactly two decimals), 3-5 questions per item.  A preamble in
  the instructions defines the digit readout; nothing outside the API
  primitives is used.

Exactness contract: gold strings are taken verbatim from the pinned JSONL;
choice-mode gold is the exact gold string under its option key; score-mode
gold is the exact digit string, reconstructable without loss.  Items that fit
neither encoding (LaTeX/text answers, out-of-range values, or fewer than 3
derivable distractors) are counted and excluded AT BUILD TIME — never
silently dropped after dispatch: ``excluded_math500_ids`` is recorded in the
spec and the build refuses if exclusions exceed ``max_exclusions``.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

from .base import DatasetProvenance, LoadedDataset, read_jsonl

SOURCE_URL = "https://huggingface.co/datasets/HuggingFaceH4/MATH-500"
EXPECTED_N_TEST = 500
SCORE_MAX_QUESTIONS = 5          # 1-3 integer digits + exactly 2 decimals
SCORE_MAX_INTEGER_DIGITS = 3     # values < 1000
CHOICE_OPTIONS = 4
SEED = 902100                    # frozen; distractor derivation + key shuffle
SCORE_INSTRUCTIONS_TEMPLATE = (
    "Numeric readout of the answer to the state's problem, one digit per "
    "question, in reading order (integer part most-significant first, then "
    "tenths, then hundredths). Answer {n_questions} digit questions for the "
    "fixed-width value {format_description}. Level i means digit i.")
# Position semantics MUST live in the question CONTENT: the server never sees
# question ids (isolation probes), so identical instructions would make every
# digit question indistinguishable (observed live 2026-09-20: position-blind
# answers, 7/273 exact).  Each digit question therefore states WHICH digit it
# asks for.
SCORE_POSITION_TEMPLATES = {
    "leading": (
        "This question asks for integer digit {ordinal} of {n_integer} in the "
        "answer's decimal readout, counting from the most significant "
        "(leftmost) integer digit. Level i means the digit i (0-9)."),
    "tenths": (
        "This question asks for the TENTHS digit (first decimal) of the "
        "answer. Level i means the digit i (0-9)."),
    "hundredths": (
        "This question asks for the HUNDREDTHS digit (second decimal) of the "
        "answer. Level i means the digit i (0-9)."),
}
CHOICE_INSTRUCTIONS = (
    "Answer the mathematics problem. Choose the single best option; exactly "
    "one option is the correct answer in the requested answer format."
)
_DIGIT = re.compile(r"^-?\d+$")


class Math500Error(RuntimeError):
    """Fail-closed condition for the MATH-500 adapter."""


def load_math500(
    path: Path | str,
    *,
    revision: str | None = None,
    accessed_at: str | None = None,
    expected_n: int | None = None,
) -> LoadedDataset:
    """Load the pinned MATH-500 JSONL. ``expected_n`` relaxed ONLY for fixtures."""
    required_n = EXPECTED_N_TEST if expected_n is None else expected_n
    records = read_jsonl(Path(path))
    if len(records) != required_n:
        raise Math500Error(
            f"{path}: expected the MATH-500 test file ({required_n} items), got "
            f"{len(records)}; refusing"
        )
    fields = {"problem", "answer", "subject", "level", "unique_id"}
    items: list[dict[str, Any]] = []
    for record in records:
        missing = fields - set(record)
        if missing:
            raise Math500Error(f"MATH-500 record missing fields {sorted(missing)}")
        unique_id = str(record["unique_id"])
        # Item ids become logical-request ids and raw-blob filename components;
        # the source ids contain ``/`` (e.g. ``test/algebra/0.json``), which is
        # unsafe downstream — flatten them, keep the original id recorded.
        item_id = unique_id.replace("/", "_")
        items.append({
            "id": item_id,
            "group": str(record["subject"]),
            "cluster": item_id,
            "state": str(record["problem"]),
            "leakage_check": False,  # the gold answer is not free-form in the state
            "gold": {"math500": {"value": str(record["answer"])}},
            "questions": {},   # installed per-mode by the spec builders
            "_source": {k: record.get(k) for k in ("level", "unique_id")},
        })
    provenance = DatasetProvenance(
        name="math-500",
        source_url=SOURCE_URL,
        revision=revision,
        accessed_at=accessed_at,
        license_note="MIT (HuggingFaceH4/MATH-500 redistribution); source MATH "
                     "(hendrycks/math) test items; subset introduced by DeepSeek-R1",
    )
    return LoadedDataset(name="math-500", items=items, provenance=provenance,
                         category_field="group")


# --------------------------------------------------------------- encoding ---
def _decimal_parts(value: float, *, integer_digits: int) -> list[str]:
    """Fixed-width decimal digit string: integer part + exactly 2 decimals."""
    quantized = round(value + 1e-9, 2)
    scaled = int(round(quantized * 100))
    text = f"{scaled:0{integer_digits + 2}d}"
    return list(text)

def _parse_number(answer: str) -> float | None:
    text = answer.strip()
    if not _DIGIT.match(text):
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    if value != int(value):
        return None
    return value


def _format_description(n_questions: int) -> str:
    integer_digits = n_questions - 2
    return f"an integer with up to {integer_digits} digit(s) plus exactly 2 decimals"


def _distractors(answer: str, item_seed: int) -> list[str] | None:
    """Three deterministic numeric distractors for a gold answer, or None.

    Integers: neighbours and scaled variants, deduplicated against the gold.
    Anything else (LaTeX, text, tuples) is not MCQ-convertible without
    inventing content: excluded, never guessed.
    """
    value = _parse_number(answer)
    if value is None:
        return None
    def fmt(x: float) -> str:
        return str(int(x))
    candidates: list[str] = []
    for alt in (value + 1, value - 1, value + 2, value - 2, 2 * value, -value,
                value + 10, value - 10):
        if alt == value:
            continue
        text = fmt(alt)
        if text != answer and text not in candidates and len(text) <= 40:
            candidates.append(text)
    if len(candidates) < 3:
        return None
    rng = random.Random(f"math500-distractors:{item_seed}")
    return rng.sample(candidates, 3)


def build_items(
    dataset: LoadedDataset, *, mode: str, model: str,
    max_exclusions: int = 250, seed: int = SEED,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build per-mode items + build-time accounting (exclusions are explicit).

    ``mode`` is ``choice`` (4-option MCQ for numeric answers with derivable
    distractors) or ``score`` (per-digit 10-level rubric for values < 100).
    """
    if mode not in ("choice", "score"):
        raise Math500Error(f"unknown mode {mode!r}")
    items: list[dict[str, Any]] = []
    excluded_ids: list[str] = []
    exclusion_reasons: dict[str, int] = {}
    for raw in dataset.items:
        record = {k: v for k, v in raw.items() if not k.startswith("_")}
        gold = record["gold"]["math500"]["value"]
        if mode == "choice":
            distractors = _distractors(gold, record["id"])
            if distractors is None or re.search(
                    r"(?<![\d.])" + re.escape(gold) + r"(?![\d.])",
                    str(record["state"])):
                excluded_ids.append(record["id"])
                reason = ("no_deterministic_distractors" if distractors is None
                          else "answer_disclosed_in_problem")
                exclusion_reasons[reason] = exclusion_reasons.get(reason, 0) + 1
                continue
            options = [gold] + distractors
            rng = random.Random(f"math500-keyorder:{record['id']}:{seed}")
            rng.shuffle(options)
            key_by_option = {option: f"o{i}" for i, option in enumerate(options)}
            criteria = {key_by_option[opt]: opt for opt in options}
            gold_key = key_by_option[gold]
            state = (record["state"] +
                     "\n\nGive your answer as an exact integer (no units, no "
                     "punctuation). Exactly one option matches that answer.")
            items.append({
                "id": record["id"],
                "group": record["group"],
                "cluster": record["cluster"],
                "condition": "math500_choice",
                "state": state,
                "questions": {"math500": {
                    "type": "choice",
                    "instructions": CHOICE_INSTRUCTIONS,
                    "criteria": criteria,
                }},
                "gold": {"math500": {"value": gold_key}},
                "math500_meta": {
                    "source_id": raw.get("_source", {}).get("unique_id", record["id"]),
                    "gold_answer_text": gold,
                    "encoding": "numeric-integers 4-option MCQ",
                    "level": raw.get("_source", {}).get("level"),
                },
                "leakage_check": False,
            })
        else:
            value = _parse_number(gold)
            if value is None or not 0 <= value < 1000:
                excluded_ids.append(record["id"])
                reason = ("answer_not_number_in_range" if _parse_number(gold) is None
                          else "answer_out_of_score_range")
                exclusion_reasons[reason] = exclusion_reasons.get(reason, 0) + 1
                continue
            digits = _decimal_parts(value, integer_digits=len(str(int(value))))
            n_integer = len(digits) - 2
            questions: dict[str, Any] = {}
            for position, digit in enumerate(digits):
                if position < n_integer:
                    position_text = SCORE_POSITION_TEMPLATES["leading"].format(
                        ordinal=position + 1, n_integer=n_integer)
                elif position == n_integer:
                    position_text = SCORE_POSITION_TEMPLATES["tenths"]
                else:
                    position_text = SCORE_POSITION_TEMPLATES["hundredths"]
                questions[f"d{position}"] = {
                    "type": "score",
                    "instructions": position_text,
                    "criteria": [f"digit {i}" for i in range(10)],
                }
            items.append({
                "id": record["id"],
                "group": record["group"],
                "cluster": record["cluster"],
                "condition": "math500_score",
                "state": record["state"] + (
                    "\n\nThe answer is a non-negative value below 1000. Report it "
                    "with exactly two decimal places as digit questions (one "
                    "question per digit). Each question states WHICH digit it "
                    "asks for. Example: if the answer were 12.34, the readout "
                    "has four digit questions: integer digits '1','2' (most "
                    "significant first), then tenths '3', then hundredths '4'."),
                "questions": questions,
                "gold": {f"d{p}": {"math500_digit": {"value": d}}
                         for p, d in enumerate(digits)},
                "math500_meta": {
                    "source_id": raw.get("_source", {}).get("unique_id", record["id"]),
                    "gold_answer_text": gold,
                    "encoding": "2-decimal digit readout via 10-level score",
                    "level": raw.get("_source", {}).get("level"),
                },
                "leakage_check": False,
            })
    if len(excluded_ids) > max_exclusions:
        raise Math500Error(
            f"{len(excluded_ids)} items exceed the exclusion budget "
            f"({max_exclusions}); refusing to build a silently shrunken stage"
        )
    accounting = {
        "n_population": len(dataset.items),
        "n_built": len(items),
        "n_excluded": len(excluded_ids),
        "excluded_ids": sorted(excluded_ids),
        "exclusion_reasons": exclusion_reasons,
        "seed": seed,
    }
    return items, accounting


def build_spec(
    dataset: LoadedDataset,
    *,
    mode: str,
    model: str = "jev-1.13.0",
    dataset_file: str | None = None,
    source_sha256: str | None = None,
) -> dict[str, Any]:
    items, accounting = build_items(dataset, mode=mode, model=model)
    experiments = {
        "choice": "math500-choice-native-mcq",
        "score": "math500-score-native-digits",
    }
    titles = {
        "choice": "MATH-500 numeric-answer 4-option MCQ (gold + 3 deterministic "
                  "perturbations; non-convertible items excluded at build time)",
        "score": "MATH-500 per-digit decimal readout via the 10-level score rubric "
                 "(non-negative values < 1000, exactly 2 decimals; exact digit gold)",
    }
    return {
        "experiment": experiments[mode],
        "model": model,
        "test_family": "public-benchmark",
        "seeds": {"order": 0, "distractor_seed": accounting["seed"]},
        "shuffle": True,
        "protocol": {
            "instructions": ("Jev-native MATH-500 adaptation: exact-match answers "
                             "expressed through the API's own primitives. No "
                             "chain-of-thought, fixed instructions, 0 retries. "
                             "This is a DIRECT-ANSWER protocol, NOT the saturated "
                             "reasoning-era protocol of late-2026 reference scores."),
            "prompt_variants": 1,
            "fixed_instructions": True,
            "no_chain_of_thought": True,
            "correct_answer_retries": 0,
            "encoding": mode,
            "build_exclusions": accounting,
        },
        "dataset": {
            "name": dataset.name,
            "source_url": dataset.provenance.source_url,
            "revision": dataset.provenance.revision,
            "accessed_at": dataset.provenance.accessed_at,
            "file": dataset_file,
            "file_sha256": source_sha256,
            "license_note": dataset.provenance.license_note,
            "n_population": accounting["n_population"],
            "sampling": {"mode": "full_set", "split": "test (MATH-500 subset of "
                                                      "the MATH TEST split)"},
        },
        "items": items,
    }
