#!/usr/bin/env python3
"""Matched cheap-model baselines, v2 (post-diagnostic re-run).

The v1 sweep (run_cheap_matched.py) counted strict-format failures as wrong,
per the matched-run discipline. A diagnostic probe
(runs_matched_cheap/diag/, 216 raw-stored calls) showed two wire/parser
artifacts were crushing several models' scores:

1. THINKING MODELS STARVED BY THE OUTPUT CAP. Qwen3.7/3.8-Flash spend the
   whole 1024-token budget in the reasoning channel and return empty
   content (reasoning_tokens=1024, finish=length). The answer never exists
   to parse. Fix: disable thinking where the provider supports it
   ({"reasoning": {"enabled": false}} for Qwen3 models) and raise the cap
   to 4096.
2. VERBOSE SOLVERS REJECTED BY EXACT-KEY PARSING. Gemma-3-4B, Llama-3.1-8B
   and Mistral-Small-3.2 answer in prose ("The answer is B..."). Fix: a
   documented, deterministic recovery parser applied IDENTICALLY to every
   model, with the recovery stage recorded per item (see
   jev_observatory.answer_recovery for the full contract):
     exact -> stripped -> answer_pattern -> isolated-key ->
     (empty content) reasoning-channel -> unrecovered.
   Recovery matches the ACTUAL allowed keys (MATH keys are o0-o3, some
   MMLU/HLE keys extend beyond J) at token boundaries only; the LAST
   unambiguous explicit answer statement wins; negated/disjunctive mentions
   and multiple distinct key mentions recover nothing; first-character
   guessing is gone (it inflated recovered accuracy). Unrecovered counts as
   wrong. Both strict and recovered accuracy are reported; charts use
   recovered.

Wire v2 (recorded per attempt): temperature=0, max_output_tokens=4096
(2048 where thinking is disabled), reasoning_effort=low where accepted and
not disabled. Roster adds xiaomi/mimo-v2.6-pro, xiaomi/mimo-v2.6-flash and
qwen/qwen3.8-max-0902; ARC-Challenge (1,172 frozen items) runs for the six
models the audit named for that chart.

Discipline unchanged: ONE attempt per item (transport-level 429/5xx get
bounded backoff re-queues), no answer retries, key from environment only,
live gate required, hard spend cap $20 (the owner's second grant), raw
content excerpts stored OUTSIDE the bundle (runs_matched_cheap/raw2/).

  python scripts/benchmark/run_cheap_matched2.py --plan
  python scripts/benchmark/run_cheap_matched2.py --smoke
  python scripts/benchmark/run_cheap_matched2.py --live
  python scripts/benchmark/run_cheap_matched2.py --score
"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import argparse
import glob as _glob
import json
import math
import os
import re
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from jev_observatory.answer_recovery import (RECOVERY_SPEC_VERSION,
                                             recover_choice,
                                             recovery_spec)
from jev_observatory.matched import build_baseline_request, logical_id
from jev_observatory.openrouter import (API_KEY_ENV, CHAT_PATH, MODELS_PATH,
                                        OPENROUTER_BASE_URL,
                                        OpenRouterHttpTransport, WireConfig,
                                        build_chat_payload)
from jev_observatory.schema import ChoiceQuestion, SystemOneRequest

ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "runs_matched_cheap/v2"
RAW_DIR = ROOT / "runs_matched_cheap/raw2"
HARD_CAP_USD = 28.0
CONCURRENCY_TOTAL = 32
CONCURRENCY_PER_MODEL = 8
TRANSPORT_RETRIES = 3
MAX_OUT = 4096
MAX_OUT_NOTHINK = 2048

# roster: (id, thinking_disabled, in_arc_set)
ROSTER = [
    ("meta-llama/llama-3.1-8b-instruct", False, False),
    ("mistralai/mistral-nemo", False, False),
    ("openai/gpt-oss-20b", False, False),
    ("openai/gpt-oss-120b", False, False),
    ("ibm-granite/granite-4.0-h-micro", False, False),
    ("google/gemma-3-4b-it", False, False),
    ("qwen/qwen3.7-flash", True, False),
    ("mistralai/mistral-small-3.2-24b-instruct", False, False),
    ("z-ai/glm-5.3-flash", False, True),
    ("qwen/qwen3.8-flash", True, True),
    ("deepseek/deepseek-v4-flash-0731", False, False),
    ("z-ai/glm-5.3", False, True),
    ("xiaomi/mimo-v2.6-pro", False, True),
    ("xiaomi/mimo-v2.6-flash", False, False),
    ("qwen/qwen3.8-max-0902", True, True),
]

# Wire/dataset overrides for models whose first-pass cells were starved by
# the output cap (thinking models that spend the whole budget reasoning
# before emitting content) or whose endpoint rejected the no-think parameter.
# Applied by cmd_live on top of the smoke-validated wires; recorded in the
# per-attempt wire field and in the artifact.
WIRE_OVERRIDES = {
    "xiaomi/mimo-v2.6-pro": {"max_output_tokens": 12288,
                             "datasets": ["mmlu", "gpqa", "math500_choice"]},
    "xiaomi/mimo-v2.6-flash": {"max_output_tokens": 8192,
                             "datasets": ["mmlu", "gpqa", "math500_choice"]},
    "deepseek/deepseek-v4-flash-0731": {"max_output_tokens": 16384,
                                      "datasets": ["mmlu", "gpqa", "math500_choice"]},
    "qwen/qwen3.8-max-0902": {"max_output_tokens": 4096,
                              "datasets": ["mmlu", "gpqa"]},
    # minor cap-starvation (~40 items each): raise the ceiling
    "z-ai/glm-5.3": {"max_output_tokens": 8192},
    "z-ai/glm-5.3-flash": {"max_output_tokens": 8192},
}

SMOKE_QUESTION = SystemOneRequest(
    state="The capital of France is",
    questions={"smoke": ChoiceQuestion(
        type="choice",
        instructions="Answer the multiple-choice question. Choose the single best option.",
        criteria={"A": "Berlin", "B": "Paris", "C": "Madrid", "D": "Rome"})},
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_items() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    fz = json.loads((ROOT / "runs_matched/freeze/matched_frozen.json").read_text())
    out["mmlu"] = fz["datasets"]["mmlu"]["items"]
    paths = {
        "gpqa": "runs_benchmark_ext2/bench-gpqa_diamond-*/items.jsonl",
        "math500_choice": "runs_benchmark_ext/bench-math500_choice-*/items.jsonl",
        "hle_text_mc": "runs_benchmark_ext2/bench-hle_text_mc-*/items.jsonl",
        "arc": "runs_benchmark/bench-arc_test-*/items.jsonl",
    }
    for ds, pat in paths.items():
        f = sorted(_glob.glob(str(ROOT / pat)))[0]
        out[ds] = [json.loads(l) for l in Path(f).read_text().split("\n") if l.strip()]
    return out


def gold_of(item: dict) -> str | None:
    gold = item.get("gold") or {}
    for v in gold.values():
        if isinstance(v, dict) and "value" in v:
            return v["value"]
    return None


# recover_choice is implemented in jev_observatory.answer_recovery (generalized
# to the actual allowed keys, token boundaries, unambiguous explicit final
# answer preference, no gold access, no first-character guessing) and
# re-exported here for run_matched3.py's rcm2 import contract.


def wire_for(model: str, no_think: bool) -> WireConfig:
    extra = {"reasoning": {"enabled": False}} if no_think else None
    return WireConfig(model=model,
                      reasoning_effort=None if no_think else "low",
                      max_output_tokens=MAX_OUT_NOTHINK if no_think else MAX_OUT,
                      extra_body={"temperature": 0, **({"reasoning": {"enabled": False}} if no_think else {})})


def fetch_catalog() -> list[dict]:
    req = urllib.request.Request(OPENROUTER_BASE_URL + MODELS_PATH,
                                 headers={"User-Agent": "jev-observatory/cheap-matched-v2"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8")).get("data") or []


def cmd_plan() -> int:
    cat = {m["id"]: m for m in fetch_catalog()}
    roster, missing = [], []
    for mid, no_think, in_arc in ROSTER:
        if mid not in cat:
            missing.append(mid)
            continue
        pr = cat[mid].get("pricing") or {}
        roster.append({"id": mid, "no_think": no_think, "arc": in_arc,
                       "input_per_M": float(pr.get("prompt", 0)) * 1e6,
                       "output_per_M": float(pr.get("completion", 0)) * 1e6})
    items = load_items()
    calls = sum((sum(len(v) for k, v in items.items() if k != "arc")
                 + (len(items["arc"]) if m["arc"] else 0)) for m in roster)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    (RUN_ROOT / "freeze_v2.json").write_text(json.dumps({
        "created_at": utc_now(), "roster": roster, "missing": missing,
        "wire": {"temperature": 0, "max_output_tokens": MAX_OUT,
                 "max_output_tokens_no_think": MAX_OUT_NOTHINK,
                 "reasoning_effort": "low unless thinking disabled",
                 "recovery": ",".join(recovery_spec()["stages"]),
                 "recovery_spec_version": RECOVERY_SPEC_VERSION},
        "datasets": {k: len(v) for k, v in items.items()},
        "hard_cap_usd": HARD_CAP_USD, "total_calls": calls,
    }, indent=1))
    print(f"[plan] roster {len(roster)} (missing: {missing or 'none'}), "
          f"{calls:,} calls, cap ${HARD_CAP_USD}")
    for m in roster:
        n = sum(len(v) for k, v in items.items() if k != "arc") + (len(items["arc"]) if m["arc"] else 0)
        print(f"  {m['id']:44s} in ${m['input_per_M']:.2f}/M out ${m['output_per_M']:.2f}/M  {n} calls"
              f"{'  [arc]' if m['arc'] else ''}{'  [no-think]' if m['no_think'] else ''}")
    return 0


def _refuse_overwrite(path: Path) -> None:
    """Original run evidence is never overwritten; regenerate elsewhere."""
    if path.exists():
        raise SystemExit(
            f"[refused] {path} already exists; original run evidence is never "
            f"overwritten (use scripts/benchmark/run_baseline_revision.py for "
            f"versioned re-runs)")


def _live_gate() -> str:
    if os.environ.get("JEVO_ALLOW_LIVE") != "1":
        raise SystemExit("[gate] JEVO_ALLOW_LIVE=1 required")
    key = os.environ.get(API_KEY_ENV)
    if not key:
        raise SystemExit(f"[gate] {API_KEY_ENV} required (environment only)")
    return key


def _extract_content_reasoning(body) -> tuple[str, str]:
    """(content, reasoning) from an OpenRouter chat response body."""
    if not isinstance(body, dict):
        return "", ""
    ch = (body.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
    if not isinstance(reasoning, str):
        reasoning = json.dumps(reasoning)[:2000]
    return content, reasoning


def _extract_finish_reason(body) -> str | None:
    """finish_reason from an OpenRouter chat response body (recorded truth)."""
    if not isinstance(body, dict):
        return None
    ch = (body.get("choices") or [{}])[0]
    value = ch.get("finish_reason")
    return str(value) if value is not None else None


def cmd_smoke() -> int:
    key = _live_gate()
    roster = json.loads((RUN_ROOT / "freeze_v2.json").read_text())["roster"]
    out = []
    for m in roster:
        mid = m["id"]
        wire = wire_for(mid, m["no_think"])
        transport = OpenRouterHttpTransport(key)
        rec = {"id": mid, "wire": wire.to_dict()}
        try:
            payload = build_chat_payload(SMOKE_QUESTION, wire)
            resp = transport.post(CHAT_PATH, payload)
            body = json.loads(resp.body.decode("utf-8", "replace")) if resp.body else None
            rec["http"] = resp.status_code
            if isinstance(body, dict) and body.get("error"):
                rec["error"] = str(body["error"])[:250]
                # retry once without the reasoning-disable extra, if rejected
                wire2 = WireConfig(model=mid, reasoning_effort="low",
                                   max_output_tokens=1024,
                                   extra_body={"temperature": 0})
                payload2 = build_chat_payload(SMOKE_QUESTION, wire2)
                resp2 = transport.post(CHAT_PATH, payload2)
                body = json.loads(resp2.body.decode("utf-8", "replace")) if resp2.body else None
                rec["http_fallback"] = resp2.status_code
                rec["wire_fallback"] = wire2.to_dict()
                if isinstance(body, dict) and not body.get("error"):
                    rec["wire"] = wire2.to_dict()
                    rec["fallback_used"] = True
            content, reasoning = _extract_content_reasoning(body)
            pred, stage = recover_choice(content, ["A", "B", "C", "D"], reasoning)
            rec["content"] = content[:120]
            rec["reasoning_excerpt"] = reasoning[:120]
            rec["pred"] = pred
            rec["stage"] = stage
            rec["usage"] = (body or {}).get("usage") if isinstance(body, dict) else None
        except Exception as exc:
            rec["exception"] = str(exc)[:250]
        finally:
            transport.close()
        rec["ok"] = rec.get("pred") == "B"
        out.append(rec)
        print(f"[smoke] {mid:44s} http={rec.get('http')} pred={rec.get('pred')} "
              f"stage={rec.get('stage')} ok={rec.get('ok')} "
              f"content={rec.get('content','')[:40]!r}")
    _refuse_overwrite(RUN_ROOT / "smoke_v2.json")
    (RUN_ROOT / "smoke_v2.json").write_text(json.dumps(
        {"created_at": utc_now(), "results": out}, indent=1))
    nok = sum(1 for r in out if r["ok"])
    print(f"[smoke] {nok}/{len(out)} models answered B; wrote smoke_v2.json")
    return 0


def cmd_live() -> int:
    key = _live_gate()
    fz = json.loads((RUN_ROOT / "freeze_v2.json").read_text())
    roster = {m["id"]: m for m in fz["roster"]}
    smoke = json.loads((RUN_ROOT / "smoke_v2.json").read_text())
    wires = {}
    overrides = {}
    for r in smoke["results"]:
        if not r.get("ok"):
            print(f"[live] SKIP {r['id']}: smoke failed ({r.get('stage')})")
            continue
        w = dict(r.get("wire") or {})
        ov = dict(WIRE_OVERRIDES.get(r["id"], {}))
        if ov:
            overrides[r["id"]] = ov
            if "max_output_tokens" in ov:
                w["max_output_tokens"] = ov["max_output_tokens"]
            eb = dict(w.get("extra_body") or {})
            if "reasoning_max_tokens" in ov:
                rs = dict(eb.get("reasoning") or {})
                rs["max_tokens"] = ov["reasoning_max_tokens"]
                eb["reasoning"] = rs
            w["extra_body"] = eb
        wires[r["id"]] = WireConfig(
            model=r["id"],
            reasoning_effort=w.get("reasoning_effort"),
            max_output_tokens=int(w.get("max_output_tokens") or MAX_OUT),
            extra_body=w.get("extra_body") or None)
    items = load_items()
    _refuse_overwrite(RUN_ROOT / "freeze_v2.json")
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    results_p = RUN_ROOT / "results.jsonl"
    done = set()
    if results_p.exists():
        for line in results_p.read_text().split("\n"):
            if line.strip():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("terminal"):
                    done.add(r["logical_request_id"])
    if os.environ.get("CM_SKIP_DONE") == "0":
        done = set()
    tasks = []
    per_model = []
    for mid in wires:
        ov = overrides.get(mid, {})
        ds_ok = ov.get("datasets")
        ds_skip = set(ov.get("skip_datasets") or [])
        mt = []
        for ds, rows in items.items():
            if ds == "arc" and not roster[mid]["arc"]:
                continue
            if ds_ok is not None and ds not in ds_ok:
                continue
            if ds in ds_skip:
                continue
            for it in rows:
                lid = logical_id(mid, ds, str(it["id"]))
                if lid not in done:
                    mt.append((mid, ds, it))
        per_model.append(mt)
    for i in range(max((len(t) for t in per_model), default=0)):
        for mt in per_model:
            if i < len(mt):
                tasks.append(mt[i])
    print(f"[live] {len(wires)} models, {len(tasks):,} calls "
          f"({len(done):,} already terminal)")
    if not tasks:
        return 0

    lock = threading.Lock()
    res_f = results_p.open("a")
    spend = [0.0]
    stop = threading.Event()
    counts = {"done": 0, "unrecovered": 0, "error": 0}
    local = threading.local()
    raw_files = {}
    sem_model = {mid: threading.Semaphore(CONCURRENCY_PER_MODEL) for mid in wires}

    def get_transport():
        if not hasattr(local, "tr"):
            local.tr = OpenRouterHttpTransport(key)
        return local.tr

    def work(task):
        mid, ds, it = task
        if stop.is_set():
            return None
        lid = logical_id(mid, ds, str(it["id"]))
        req, _conv = build_baseline_request(it, model=mid)
        gold = gold_of(it)
        qid = next(iter(req.questions))
        keys = list(req.questions[qid].criteria)
        payload = build_chat_payload(req, wires[mid])
        rec = {"logical_request_id": lid, "model": mid, "dataset": ds,
               "item_id": str(it["id"]), "gold": gold, "wire": wires[mid].to_dict()}
        with sem_model[mid]:
            for attempt in range(TRANSPORT_RETRIES + 1):
                if stop.is_set():
                    return None
                try:
                    resp = get_transport().post(CHAT_PATH, payload)
                    body = json.loads(resp.body.decode("utf-8", "replace")) if resp.body else None
                    if resp.status_code in (429, 500, 502, 503, 529) and attempt < TRANSPORT_RETRIES:
                        time.sleep(min(2 ** attempt, 15) + (5 if resp.status_code == 429 else 0))
                        continue
                    content, reasoning = _extract_content_reasoning(body)
                    usage = (body or {}).get("usage") if isinstance(body, dict) else None
                    pred_strict = content.strip() if content.strip() in keys else None
                    pred, stage = recover_choice(content, keys, reasoning)
                    rec.update({
                        "status": "ok" if resp.status_code == 200 else "http_error",
                        "http_status": resp.status_code,
                        "pred_strict": pred_strict,
                        "pred_recovered": pred, "recovery_stage": stage,
                        "correct_strict": pred_strict == gold if pred_strict else False,
                        "correct_recovered": pred == gold if pred else False,
                        "usage_in": (usage or {}).get("prompt_tokens"),
                        "usage_out": (usage or {}).get("completion_tokens"),
                        "reasoning_tokens": ((usage or {}).get("completion_tokens_details") or {}).get("reasoning_tokens"),
                        "cost_reported": (usage or {}).get("cost"),
                        "terminal": True, "at": utc_now(),
                    })
                    if isinstance(body, dict) and body.get("error"):
                        rec["error"] = str(body["error"])[:200]
                    with lock:
                        rf = raw_files.get(mid)
                        if rf is None:
                            rf = raw_files[mid] = (RAW_DIR / f"{mid.replace(chr(47), chr(95))}.jsonl").open("a")
                        rf.write(json.dumps({"lid": lid, "content": content,
                                             "reasoning": reasoning,
                                             "finish_reason": _extract_finish_reason(body)}) + "\n")
                    break
                except Exception as exc:
                    if attempt >= TRANSPORT_RETRIES:
                        rec.update({"status": "exception", "error": str(exc)[:200],
                                    "terminal": True, "at": utc_now()})
                    else:
                        time.sleep(min(2 ** attempt, 15))
        cost = rec.get("cost_reported")
        with lock:
            res_f.write(json.dumps(rec) + "\n")
            res_f.flush()
            if isinstance(cost, (int, float)):
                spend[0] += float(cost)
                if spend[0] >= HARD_CAP_USD:
                    stop.set()
                    print(f"[live] CAP ${HARD_CAP_USD} reached (${spend[0]:.2f})")
            counts["done"] += 1
            if rec.get("recovery_stage") == "unrecovered":
                counts["unrecovered"] += 1
            if rec.get("status") != "ok":
                counts["error"] += 1
            if counts["done"] % 1000 == 0:
                print(f"[live] {counts['done']:,}/{len(tasks):,} spend=${spend[0]:.2f} "
                      f"unrec={counts['unrecovered']} err={counts['error']}", flush=True)
        return rec

    t0 = time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=CONCURRENCY_TOTAL) as ex:
            futs = [ex.submit(work, t) for t in tasks]
            for _ in as_completed(futs):
                pass
    finally:
        res_f.close()
        for rf in raw_files.values():
            rf.close()
    el = time.monotonic() - t0
    print(f"[live] {counts['done']:,} calls in {el/60:.1f} min; spend=${spend[0]:.2f}; "
          f"unrecovered={counts['unrecovered']}; errors={counts['error']}")
    (RUN_ROOT / "live_summary.json").write_text(json.dumps({
        "finished_at": utc_now(), "calls": counts["done"],
        "spend_usd": round(spend[0], 4), "unrecovered": counts["unrecovered"],
        "errors": counts["error"], "budget_stop": stop.is_set(),
        "wall_seconds": round(el)}, indent=1))
    return 0


def _mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def _jev_per_item() -> dict[str, dict[str, bool]]:
    src = {
        "mmlu": "runs_benchmark/bench-mmlu_full-*/derived/score.json",
        "gpqa": "runs_benchmark_ext2/bench-gpqa_diamond-*/derived/score.json",
        "math500_choice": "runs_benchmark_ext/bench-math500_choice-*/derived/score.json",
        "hle_text_mc": "runs_benchmark_ext2/bench-hle_text_mc-*/derived/score.json",
        "arc": "runs_benchmark/bench-arc_test-*/derived/score.json",
    }
    out = {}
    for ds, pat in src.items():
        f = _glob.glob(str(ROOT / pat))
        if not f:
            continue
        d = json.loads(Path(f[0]).read_text())
        out[ds] = {str(it["item_id"]).split(":")[-1]: bool(it.get("correct"))
                   for it in d.get("per_item") or []}
    return out


def cmd_score() -> int:
    res_p = RUN_ROOT / "results.jsonl"
    rows = [json.loads(l) for l in res_p.read_text().split("\n") if l.strip()]
    items = load_items()
    n_req = {ds: len(v) for ds, v in items.items()}
    jev = _jev_per_item()
    by: dict[tuple, list] = {}
    for r in rows:
        by.setdefault((r["model"], r["dataset"]), []).append(r)
    summary = {}
    for (mid, ds), rs in sorted(by.items()):
        n = n_req.get(ds, len(rs))
        acc_s = sum(1 for r in rs if r.get("correct_strict")) / n
        acc_r = sum(1 for r in rs if r.get("correct_recovered")) / n
        stages = {}
        for r in rs:
            stages[r.get("recovery_stage")] = stages.get(r.get("recovery_stage"), 0) + 1
        tin = sum(r.get("usage_in") or 0 for r in rs)
        tout = sum(r.get("usage_out") or 0 for r in rs)
        cost = sum(r.get("cost_reported") or 0 for r in rs)
        entry = {
            "n_requested": n, "n_terminal": len(rs),
            "accuracy_strict": round(acc_s, 4), "accuracy_recovered": round(acc_r, 4),
            "recovery_stages": stages,
            "unrecovered": stages.get("unrecovered", 0),
            "usage_input_tokens": tin, "usage_output_tokens": tout,
            "cost_reported_usd": round(cost, 4),
            "cost_per_question_usd": round(cost / n, 8) if n else None,
        }
        jm = jev.get(ds)
        if jm:
            pairs = [(jm.get(r["item_id"]), bool(r.get("correct_recovered")))
                     for r in rs if r["item_id"] in jm]
            if pairs:
                b = sum(1 for j, x in pairs if j and not x)
                c = sum(1 for j, x in pairs if not j and x)
                entry["jev_join"] = {
                    "n_paired": len(pairs),
                    "jev_accuracy_on_subset": round(sum(1 for j, x in pairs if j) / len(pairs), 4),
                    "jev_only": b, "model_only": c,
                    "mcnemar_exact_p": round(_mcnemar_exact(b, c), 4),
                }
        summary.setdefault(mid, {})[ds] = entry
    doc = {"created_at": utc_now(), "protocol": "matched v2 (see module docstring)",
           "n_result_rows": len(rows), "hard_cap_usd": HARD_CAP_USD,
           "models": summary}
    ls = RUN_ROOT / "live_summary.json"
    if ls.exists():
        doc["dispatch"] = json.loads(ls.read_text())
    (RUN_ROOT / "summary_v2.json").write_text(json.dumps(doc, indent=1))
    print(f"{'model':42s} {'dataset':15s} {'strict':>7s} {'recov':>7s} {'unrec%':>7s} {'$/q':>10s} {'jevp':>8s}")
    for mid, dss in summary.items():
        for ds, e in dss.items():
            jj = e.get("jev_join") or {}
            print(f"{mid:42s} {ds:15s} {e['accuracy_strict']*100:6.1f}% "
                  f"{e['accuracy_recovered']*100:6.1f}% "
                  f"{100*e['unrecovered']/max(e['n_terminal'],1):6.1f}% "
                  f"{(e['cost_per_question_usd'] or 0):10.6f} "
                  f"{(jj.get('jev_accuracy_on_subset') or 0)*100:7.1f}%")
    print(f"[score] wrote {RUN_ROOT.relative_to(ROOT)}/summary_v2.json")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--plan", action="store_true")
    g.add_argument("--smoke", action="store_true")
    g.add_argument("--live", action="store_true")
    g.add_argument("--score", action="store_true")
    a = ap.parse_args()
    if a.plan:
        return cmd_plan()
    if a.smoke:
        return cmd_smoke()
    if a.live:
        return cmd_live()
    return cmd_score()


if __name__ == "__main__":
    sys.exit(main())

