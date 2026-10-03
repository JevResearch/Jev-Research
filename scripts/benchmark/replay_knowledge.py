#!/usr/bin/env python3
"""Portable replay of the public knowledge probe (followup-20261003).

Replays the 48 public prompts from
``data_report/followup_20261003/knowledge-questions.json`` (24 opaque items x
2 option rotations) as fresh, stateless, one-request-per-variant calls.  No
history, browsing, tools, sampler or reasoning parameters; every requested
call is recorded in full and an existing output file is NEVER overwritten or
resumed (the original study is not re-runnable from partial rows).

Providers (official endpoints only; credentials only from the environment):
  --provider jev         native POST /v1/systemone, TYPESAFE_API_KEY
  --provider openrouter  chat completions, OPENROUTER_API_KEY

Nothing calls anything unless JEVO_ALLOW_LIVE=1 is set AND the provider key is
present; calls are billable (use --limit 4 for a smoke run).  The key value is
never printed, logged or written anywhere (Redactor wraps every persisted row).

Exit codes: 0 all calls recorded with a known contract; 1 finished but some
rows carry explicit contract/transport safety errors; 2 refused to run (gates,
existing output, missing key) or stopped early on an auth/quota error.

Answer key (--answer-key) is joined offline AFTER each reply, per item and
rotation; gold/source/kind/date metadata never enters the model payload.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from jev_observatory.answer_recovery import recover_choice  # noqa: E402
from jev_observatory.openrouter import (  # noqa: E402
    CHAT_PATH, OpenRouterHttpTransport, WireConfig,
    build_chat_payload, extract_choice_content, parse_choice_answer)
from jev_observatory.redact import Redactor, base_url, get_api_key  # noqa: E402
from jev_observatory.schema import ChoiceQuestion, SystemOneRequest  # noqa: E402
from jev_observatory.transport import HttpxTransport  # noqa: E402

DEFAULT_QUESTIONS = (_ROOT / "data_report" / "followup_20261003"
                     / "knowledge-questions.json")
STATE_TEXT = "This question is about public events."
MODELS = {"jev": "jev-1.13.0",
          "openrouter": "qwen/qwen3-30b-a3b-instruct-2507"}
KEY_ENVS = {"jev": "TYPESAFE_API_KEY", "openrouter": "OPENROUTER_API_KEY"}
SYSTEMONE_PATH = "/v1/systemone"
TIMEOUT_SECONDS = 120.0
OUT_CAP_DEFAULT = 512      # bounded direct answers; non-thinking Instruct IDs
STOP_STATUSES = {401, 402, 403, 429}

# SAFE full-string echo (same contract as the follow-up v2 override): the whole
# reply must be '<allowed key><punct><exactly that key's option text>'.
_ECHO_RE = re.compile(r"^(o\d+)\s*[.:,)]\s*(.+?)\s*$", re.DOTALL | re.IGNORECASE)


def echo_recover(content: str | None, keys: list[str],
                 criteria: dict[str, str]) -> tuple[str | None, str]:
    if not content:
        return None, "empty"
    m = _ECHO_RE.match(content.strip())
    if not m:
        return None, "not_echo_shape"
    key_raw, echoed = m.group(1), m.group(2).strip()
    key = next((k for k in keys if k.lower() == key_raw.lower()), None)
    if key is None:
        return None, "key_not_allowed"
    expected = (criteria.get(key) or "").strip()
    if not expected:
        return None, "no_expected_text"
    if echoed.lower() != expected.lower():
        return None, "key_text_disagree"
    return key, "echo_exact_text"


def recover_reply(content: str | None, keys: list[str],
                  criteria: dict[str, str]) -> tuple[str | None, str]:
    """answer-recovery 2.1 first, then the SAFE full-string echo.  Never gold
    access, never a first-character guess."""
    key, stage = recover_choice(content, keys)
    if key is not None:
        return key, stage
    return echo_recover(content, keys, criteria)


def load_calls(questions_path: Path, model: str) -> list[dict[str, Any]]:
    """48 stateless calls: items in file order, each item's variants in order."""
    doc = json.loads(questions_path.read_text(encoding="utf-8"))
    calls: list[dict[str, Any]] = []
    for item in doc["items"]:
        for variant in item["variants"]:
            criteria = {f"o{i}": t for i, t in enumerate(variant["options"])}
            request = SystemOneRequest(
                state=STATE_TEXT, model=model,
                questions={"q0": ChoiceQuestion(instructions=item["question"],
                                                criteria=criteria)})
            calls.append({"item_id": item["id"], "rotation": variant["rotation"],
                          "request": request, "keys": list(criteria),
                          "criteria": criteria})
    return calls


def load_answer_key(path: Path) -> dict[tuple[str, int], str]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    table: dict[tuple[str, int], str] = {}
    for item in doc["items"]:
        for variant in item["variants"]:
            table[(item["id"], int(variant["rotation"]))] = str(variant["choice"])
    return table


def _parse_body(raw: bytes | None) -> tuple[dict[str, Any] | None, str | None]:
    if not raw:
        return None, "empty_body"
    try:
        parsed = json.loads(raw.decode("utf-8", "replace"))
    except Exception:
        return None, "unparseable_body"
    if not isinstance(parsed, dict):
        return None, "body_not_object"
    return parsed, None


def _native_answer(body: dict[str, Any], keys: list[str]
                   ) -> tuple[str | None, dict[str, Any] | None, str | None]:
    """Native contract: answers.q0.{choice,probabilities}.  The choice must be
    an allowed key; probabilities NEVER override it (no argmax, no rounding)."""
    answers = body.get("answers")
    if not isinstance(answers, dict) or "q0" not in answers:
        return None, None, "missing_answers_q0"
    ans = answers.get("q0")
    if not isinstance(ans, dict):
        return None, None, "answers_q0_not_object"
    probs = ans.get("probabilities")
    probs = probs if isinstance(probs, dict) else None
    choice = ans.get("choice")
    if not isinstance(choice, str):
        return None, probs, "missing_choice"
    if choice not in keys:
        return None, probs, "choice_not_allowed"
    return choice, probs, None


def run(argv: list[str] | None = None, *,
        http_transport: Any = None, or_transport: Any = None) -> int:
    p = argparse.ArgumentParser(
        prog="replay_knowledge.py",
        description=__doc__.split("\n\n")[1],
        epilog=("exit codes: 0 = all calls recorded with a known contract; "
                "1 = finished with explicit contract/transport safety errors; "
                "2 = refused (gates/key/existing output) or stopped on "
                "auth/quota. Calls are billable: --limit 4 is the smoke run."))
    p.add_argument("--provider", choices=tuple(MODELS), default="jev",
                   help="jev = native /v1/systemone; openrouter = chat completions")
    p.add_argument("--model", default=None,
                   help=f"default: {MODELS['jev']} (jev) / "
                        f"{MODELS['openrouter']} (openrouter)")
    p.add_argument("--questions", default=str(DEFAULT_QUESTIONS),
                   help="public questions export (model-facing, no gold)")
    p.add_argument("--answer-key", default=None,
                   help="optional answer key joined offline after each reply")
    p.add_argument("--limit", type=int, default=0,
                   help="max calls to dispatch (0 = all 48; 4 = smoke)")
    p.add_argument("--output", default=None,
                   help="jsonl path; default is a fresh timestamped file; "
                        "existing files are refused (no overwrite, no resume)")
    p.add_argument("--max-output-tokens", type=int, default=OUT_CAP_DEFAULT,
                   help=f"output budget per call (default {OUT_CAP_DEFAULT})")
    args = p.parse_args(argv)

    if args.limit < 0:
        p.error("--limit must be >= 0")
    if args.max_output_tokens < 1:
        p.error("--max-output-tokens must be >= 1")
    provider = args.provider
    model = args.model or MODELS[provider]
    calls = load_calls(Path(args.questions), model)
    if args.limit:
        calls = calls[:args.limit]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    out_path = Path(args.output) if args.output else Path(
        f"knowledge-replay-{provider}-{stamp}.jsonl")
    if out_path.exists():
        print(f"refusing existing output {out_path}: replays are always fresh "
              "(no overwrite, no resume)", file=sys.stderr)
        return 2
    if os.environ.get("JEVO_ALLOW_LIVE") != "1":
        print(f"refusing: {len(calls)} billable calls requested; set "
              "JEVO_ALLOW_LIVE=1 to allow live dispatch (use --limit 4 to smoke)",
              file=sys.stderr)
        return 2
    key_env = KEY_ENVS[provider]
    api_key = get_api_key() if provider == "jev" else os.environ.get(key_env)
    if not api_key:
        print(f"refusing: no {key_env} in the environment (export it first; "
              "the value is never printed)", file=sys.stderr)
        return 2
    key_table = (load_answer_key(Path(args.answer_key))
                 if args.answer_key else None)

    red = Redactor.from_environment()
    if provider == "jev":
        transport = HttpxTransport(base_url=base_url(), api_key=api_key,
                                   timeout_seconds=TIMEOUT_SECONDS,
                                   redactor=red, http_transport=http_transport)
        post_path = SYSTEMONE_PATH
    else:
        transport = or_transport or OpenRouterHttpTransport(
            api_key, timeout_seconds=TIMEOUT_SECONDS, redactor=red)
        post_path = CHAT_PATH

    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts = {"requested": len(calls), "attempted": 0, "rows": 0,
              "no_reply": 0, "truncated": 0, "safety_errors": 0}
    graded = {"strict_hits": 0, "recovered_hits": 0, "wrong": 0}
    stop = 0
    try:
        with out_path.open("x", encoding="utf-8") as fh:
            for seq, call in enumerate(calls, 1):
                keys, criteria = call["keys"], call["criteria"]
                if provider == "jev":
                    payload = call["request"].to_payload()
                else:
                    payload = build_chat_payload(
                        call["request"],
                        WireConfig(model=model,
                                   max_output_tokens=args.max_output_tokens))
                resp = transport.post(post_path, payload)
                counts["attempted"] += 1
                body, body_error = _parse_body(resp.body)
                row: dict[str, Any] = {
                    "schema": "jev-knowledge-replay.v1", "seq": seq,
                    "item_id": call["item_id"], "rotation": call["rotation"],
                    "provider": provider, "request_model": model,
                    "request_sha256": call["request"].request_sha256(),
                    "http_status": resp.status_code,
                    "error": resp.error or (None if resp.status_code == 200
                                            else f"http_{resp.status_code}"),
                    "total_ms": round(resp.total_ms, 3),
                    "model_returned": None, "finish_reason": None,
                    "truncated": False, "probabilities": None, "usage": None,
                    "choice": None, "choice_text": None,
                    "strict_key": None, "strict_error": None,
                    "recovered_key": None, "recovery_stage": None,
                    "grade_expected_key": None, "grade_strict_hit": None,
                    "grade_recovered_hit": None, "grade_error": None,
                    "contract_error": (f"transport:{resp.error}"
                                       if resp.error else body_error),
                    "response_sanitized": red.text(
                        resp.body.decode("utf-8", "replace") if resp.body else ""),
                }
                if body is not None:
                    row["model_returned"] = body.get("model")
                    usage = body.get("usage")
                    row["usage"] = red.obj(usage) if isinstance(usage, dict) else None
                    if provider == "jev":
                        choice, probs, contract_error = _native_answer(body, keys)
                        row["probabilities"] = probs
                        row["contract_error"] = row["contract_error"] or contract_error
                        if choice is not None:
                            row["strict_key"] = row["recovered_key"] = choice
                            row["strict_error"] = None
                            row["recovery_stage"] = "native_contract"
                    else:
                        choices = body.get("choices")
                        if isinstance(choices, list) and choices:
                            row["finish_reason"] = choices[0].get("finish_reason")
                        row["truncated"] = row["finish_reason"] == "length"
                        if row["truncated"]:
                            counts["truncated"] += 1
                        content, extract_error = extract_choice_content(body)
                        strict_key, strict_error = parse_choice_answer(content, keys)
                        row["strict_key"], row["strict_error"] = strict_key, strict_error
                        key, stage = recover_reply(content, keys, criteria)
                        row["recovered_key"], row["recovery_stage"] = key, stage
                        if extract_error and content is None:
                            row["contract_error"] = (row["contract_error"]
                                                     or extract_error)
                replied = row["strict_key"] is not None or row["recovered_key"] is not None
                if not replied and row["contract_error"] is None \
                        and row["error"] is not None:
                    row["contract_error"] = f"transport:{row['error']}"
                if row["contract_error"] is not None:
                    counts["safety_errors"] += 1
                if not replied:
                    counts["no_reply"] += 1
                if key_table is not None:
                    expected = key_table.get((call["item_id"], call["rotation"]))
                    if expected is None:
                        row["grade_error"] = "no_answer_key_join"
                        counts["safety_errors"] += 1
                    else:
                        row["grade_expected_key"] = expected
                        if replied:
                            row["grade_strict_hit"] = row["strict_key"] == expected
                            row["grade_recovered_hit"] = (row["recovered_key"]
                                                          == expected)
                            graded["strict_hits"] += int(row["grade_strict_hit"])
                            graded["recovered_hits"] += int(row["grade_recovered_hit"])
                            graded["wrong"] += int(not row["grade_recovered_hit"])
                row["choice"] = row["recovered_key"] or row["strict_key"]
                if row["choice"] is not None:
                    row["choice_text"] = criteria.get(row["choice"])
                fh.write(red.text(json.dumps(row, ensure_ascii=False)) + "\n")
                fh.flush()
                counts["rows"] += 1
                if resp.status_code in STOP_STATUSES:
                    print(f"stopping on http_{resp.status_code}: auth/quota "
                          "error (fail closed)", file=sys.stderr)
                    stop = 2
                    break
    finally:
        close = getattr(transport, "close", None)
        if close is not None:
            close()

    summary = dict(counts)
    if key_table is not None:
        summary["graded"] = dict(graded)
        summary["grading_note"] = ("counts cover all requested calls; transport "
                                   "failures are NOT knowledge misses")
    if counts["truncated"]:
        summary["truncation_note"] = ("finish_reason=length outputs are flagged, "
                                      "not claimed as clean successes")
    summary["output"] = str(out_path)
    print(json.dumps(summary, indent=1))
    return stop or (1 if counts["safety_errors"] else 0)


if __name__ == "__main__":
    sys.exit(run())
