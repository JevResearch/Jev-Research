#!/usr/bin/env python3
"""Matched cheap-model baselines on OpenRouter (audit pass 3).

Purpose: the Pareto/comparison charts were built on vals.ai rows, which
contain no Gemma/Llama/Granite-class commodity models - the cheap end of the
market where Jev actually competes on price. This runner dispatches the SAME
frozen benchmark items Jev answered, byte-identical state/options text, to a
roster of cheap OpenRouter models, and records measured accuracy + cost.

Discipline (inherited from the matched-run gate and the Jev provider):
* ONE attempt per item - a strict-format failure is COUNTED AS WRONG
  (all-requested accuracy), never repaired, never retried;
* transport-level failures only (429/5xx/network) get bounded backoff
  re-queues, recorded as such - never answer retries;
* every attempt persists raw usage/cost/latency before scoring;
* the OpenRouter key comes ONLY from the process environment
  (OPENROUTER_API_KEY); live dispatch requires JEVO_ALLOW_LIVE=1;
* hard aggregate budget cap (default $30) stops dispatch when reached;
* wire config is explicit per model: the frozen matched condition
  (reasoning_effort=low, max_output_tokens=1024) where the model supports
  it, reasoning_effort omitted where the API rejects it - both recorded.

Datasets (identical items to the Jev runs):
* mmlu   - the frozen matched 1,000-item stratified subset (seed 20260920)
* gpqa   - the frozen 196-item Diamond run items
* math500_choice - the frozen 261-item MCQ conversion
* hle_text_mc    - the frozen 494-item multiple-choice subset

Commands:
  python scripts/benchmark/run_cheap_matched.py --plan    # catalog + freeze (no key)
  python scripts/benchmark/run_cheap_matched.py --smoke   # 1 call/model (key)
  python scripts/benchmark/run_cheap_matched.py --live    # full dispatch (key)
  python scripts/benchmark/run_cheap_matched.py --score   # summaries + Jev join
"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import argparse
import json
import math
import os
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from jev_observatory.matched import (build_baseline_request, logical_id,
                                     model_slug, split_logical_id)
from jev_observatory.openrouter import (API_KEY_ENV, CHAT_PATH, MODELS_PATH,
                                        OPENROUTER_BASE_URL,
                                        OpenRouterHttpTransport,
                                        OpenRouterProvider, WireConfig,
                                        build_chat_payload, parse_choice_answer)
from jev_observatory.schema import ChoiceQuestion, SystemOneRequest

ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "runs_matched_cheap"
HARD_CAP_USD = 30.0
CONCURRENCY_TOTAL = 8
CONCURRENCY_PER_MODEL = 3
TRANSPORT_RETRIES = 3
WIRE_EFFORT = "low"
WIRE_MAX_OUT = 1024

# roster: (catalog search string, user/catalog price hint $/M input, note)
ROSTER_WISH = [
    ("meta-llama/llama-3.1-8b-instruct", 0.02, "user list"),
    ("mistralai/mistral-nemo", 0.018, "user list"),
    ("openai/gpt-oss-20b", 0.018, "user list"),
    ("openai/gpt-oss-120b", 0.03, "user list"),
    ("ibm-granite/granite-4.0-h-micro", 0.017, "user list"),
    ("google/gemma-3-4b-it", 0.05, "user list"),
    ("qwen/qwen3.7-flash", 0.03, "user list"),
    ("mistralai/mistral-small-3", 0.05, "user list: 'Mistral 3 small'"),
    ("z-ai/glm-5.3-flash", 0.09, "frozen matched nine"),
    ("qwen/qwen3.8-flash", 0.15, "frozen matched nine"),
    ("deepseek/deepseek-v4-flash-0731", 0.04, "frozen matched nine"),
    ("z-ai/glm-5.3", 0.91, "mid-tier anchor; ~$2 of the budget"),
]

DATASETS = {
    "mmlu": {"kind": "matched_freeze", "n": 1000},
    "gpqa": {"kind": "items_jsonl",
             "path": "runs_benchmark_ext2/bench-gpqa_diamond-*/items.jsonl"},
    "math500_choice": {"kind": "items_jsonl",
                       "path": "runs_benchmark_ext/bench-math500_choice-*/items.jsonl"},
    "hle_text_mc": {"kind": "items_jsonl",
                    "path": "runs_benchmark_ext2/bench-hle_text_mc-*/items.jsonl"},
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
    import glob as _glob
    out: dict[str, list[dict]] = {}
    for ds, cfg in DATASETS.items():
        if cfg["kind"] == "matched_freeze":
            fz = json.loads((ROOT / "runs_matched/freeze/matched_frozen.json").read_text())
            out[ds] = fz["datasets"]["mmlu"]["items"]
        else:
            hits = sorted(_glob.glob(str(ROOT / cfg["path"])))
            assert hits, f"no items.jsonl for {ds}"
            rows = [json.loads(l) for l in Path(hits[0]).read_text().split("\n") if l.strip()]
            out[ds] = rows
    return out


def gold_of(item: dict, ds: str) -> str | None:
    gold = item.get("gold") or {}
    if ds in gold and isinstance(gold[ds], dict):
        return gold[ds].get("value")
    for v in gold.values():
        if isinstance(v, dict) and "value" in v:
            return v["value"]
    return None


def fetch_catalog() -> list[dict]:
    req = urllib.request.Request(OPENROUTER_BASE_URL + MODELS_PATH,
                                 headers={"User-Agent": "jev-observatory/cheap-matched"})
    with urllib.request.urlopen(req, timeout=60) as r:
        body = json.loads(r.read().decode("utf-8"))
    return body.get("data") or []


def resolve_roster(catalog: list[dict]) -> list[dict]:
    by_id = {m["id"]: m for m in catalog}
    roster = []
    for wish, hint, note in ROSTER_WISH:
        if wish in by_id:
            pick = by_id[wish]
        else:
            cands = [m for m in catalog if wish.lower() in m["id"].lower()]
            if not cands:
                print(f"[roster] SKIP {wish}: no catalog match")
                continue
            def price_dist(m):
                try:
                    return abs(float(m.get("pricing", {}).get("prompt", 0)) * 1e6 - hint)
                except (TypeError, ValueError):
                    return 1e9
            pick = min(cands, key=price_dist)
            if len(cands) > 1:
                print(f"[roster] {wish}: {len(cands)} candidates -> {pick['id']} "
                      f"(closest to ${hint}/M hint)")
        pr = pick.get("pricing") or {}
        roster.append({
            "id": pick["id"], "name": pick.get("name") or pick["id"],
            "context_length": pick.get("context_length"),
            "input_per_M": float(pr.get("prompt", 0)) * 1e6,
            "output_per_M": float(pr.get("completion", 0)) * 1e6,
            "wish": wish, "note": note,
        })
    return roster


def cmd_plan() -> int:
    catalog = fetch_catalog()
    print(f"[plan] catalog: {len(catalog)} models")
    roster = resolve_roster(catalog)
    items = load_items()
    per_model_calls = sum(len(v) for v in items.values())
    # conservative reservation: 1 token per 3 chars of payload + full output cap
    est = []
    for m in roster:
        payload_chars = 0
        for ds, rows in items.items():
            for it in rows[:50]:
                req, _ = build_baseline_request(it, model=m["id"])
                payload_chars += len(json.dumps(req.to_payload(), ensure_ascii=False)) / 50
        est_in = payload_chars / 3.0
        res = per_model_calls * (est_in * m["input_per_M"] / 1e6
                                 + WIRE_MAX_OUT * m["output_per_M"] / 1e6)
        est.append((m["id"], per_model_calls, round(res, 2)))
    total = sum(e[2] for e in est)
    freeze = {
        "created_at": utc_now(),
        "purpose": "cheap-model matched baselines on identical frozen items",
        "hard_cap_usd": HARD_CAP_USD,
        "wire": {"reasoning_effort": WIRE_EFFORT, "max_output_tokens": WIRE_MAX_OUT,
                 "fallback": "reasoning_effort omitted if the API rejects it (recorded)"},
        "roster": roster,
        "datasets": {ds: {"n": len(rows),
                          "item_ids_sha256": __import__("hashlib").sha256(
                              json.dumps(sorted(str(r["id"]) for r in rows)).encode()).hexdigest()}
                     for ds, rows in items.items()},
        "reservation_estimates_usd": {mid: res for mid, n, res in est},
        "reservation_total_usd": round(total, 2),
        "calls_per_model": per_model_calls,
        "total_calls": per_model_calls * len(roster),
    }
    RUN_ROOT.mkdir(exist_ok=True)
    (RUN_ROOT / "freeze_cheap.json").write_text(json.dumps(freeze, indent=1))
    for mid, n, res in est:
        print(f"  {mid:44s} {n:5d} calls  reserved<=${res:6.2f}")
    print(f"[plan] roster {len(roster)} models, {freeze['total_calls']:,} total calls, "
          f"reservation worst-case ${total:.2f} (hard cap ${HARD_CAP_USD})")
    print(f"[plan] wrote {RUN_ROOT.relative_to(ROOT)}/freeze_cheap.json")
    return 0


def _live_gate() -> str:
    if os.environ.get("JEVO_ALLOW_LIVE") != "1":
        raise SystemExit("[gate] JEVO_ALLOW_LIVE=1 required for live calls")
    key = os.environ.get(API_KEY_ENV)
    if not key:
        raise SystemExit(f"[gate] {API_KEY_ENV} required (environment only)")
    return key


def _wire_for(rec: dict | None, model: str) -> WireConfig:
    effort = WIRE_EFFORT
    if rec and rec.get("wire", {}).get("reasoning_effort") is None:
        effort = None
    return WireConfig(model=model, reasoning_effort=effort,
                      max_output_tokens=WIRE_MAX_OUT)


def _load_roster() -> list[dict]:
    fz = json.loads((RUN_ROOT / "freeze_cheap.json").read_text())
    return fz["roster"]


def cmd_smoke() -> int:
    """One bounded call per model; discovers whether reasoning_effort is accepted.

    A wire-config discovery retry is not an answer retry: nothing here is a
    benchmark item and no result is scored.
    """
    key = _live_gate()
    roster = _load_roster()
    out = []
    for m in roster:
        rec = {"id": m["id"], "attempts": []}
        wire_ok = "UNUSABLE"
        for effort in (WIRE_EFFORT, None):
            entry = {"effort": effort}
            transport = OpenRouterHttpTransport(key)
            provider = OpenRouterProvider(
                transport, wire=WireConfig(model=m["id"], reasoning_effort=effort,
                                           max_output_tokens=256))
            t0 = time.monotonic()
            try:
                outcome = provider.ask(SMOKE_QUESTION, logical_id(m["id"], "smoke", "0"))
            except Exception as exc:
                outcome = None
                entry["error"] = str(exc)[:300]
            finally:
                provider.close()
            entry["ms"] = round((time.monotonic() - t0) * 1000)
            if outcome is not None:
                entry["status"] = outcome.status
                entry["usage_in"] = outcome.usage_input_tokens
                entry["usage_out"] = outcome.usage_output_tokens
                att = outcome.attempts[0] if outcome.attempts else None
                entry["http_status"] = getattr(att, "http_status", None)
                entry["error"] = (getattr(att, "error", None) or "")[:300]
                va = outcome.validated
                if va is not None and va.answers:
                    q = list(va.answers.values())[0]
                    entry["choice"] = q.values.get("choice")
            else:
                entry["status"] = "exception"
            rec["attempts"].append(entry)
            rejected = (entry["status"] == "error"
                        and entry.get("http_status") in (400, 404, 422))
            if not rejected and entry["status"] != "exception":
                wire_ok = effort if effort is not None else "omitted"
                break
            if effort is None:
                wire_ok = "UNUSABLE"
        rec["wire"] = {"reasoning_effort": (WIRE_EFFORT if wire_ok == WIRE_EFFORT else None),
                       "usable": wire_ok != "UNUSABLE", "resolved": wire_ok}
        out.append(rec)
        last = rec["attempts"][-1]
        print(f"[smoke] {m['id']:42s} wire={wire_ok:8s} "
              f"status={last.get('status')} choice={last.get('choice')} "
              f"{last.get('ms')}ms")
    (RUN_ROOT / "smoke_cheap.json").write_text(json.dumps(
        {"created_at": utc_now(), "results": out}, indent=1))
    usable = [r["id"] for r in out if r["wire"]["usable"]]
    print(f"[smoke] usable {len(usable)}/{len(out)}; wrote "
          f"{RUN_ROOT.relative_to(ROOT)}/smoke_cheap.json")
    return 0


def _looks_like_param_reject(entry: dict) -> bool:
    txt = json.dumps(entry).lower()
    return "reasoning" in txt and ("unknown" in txt or "unsupported" in txt
                                    or "invalid" in txt or "400" in txt)


def load_wire_map() -> dict[str, dict]:
    p = RUN_ROOT / "smoke_cheap.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    return {r["id"]: r["wire"] for r in d["results"] if r.get("wire", {}).get("usable")}

def _dispatch_one(provider, item, ds, model_id, wires):
    req, _conv = build_baseline_request(item, model=model_id)
    lid = logical_id(model_id, ds, str(item["id"]))
    gold = gold_of(item, ds)
    qid = next(iter(req.questions))
    keys = list(req.questions[qid].criteria)
    outcome = provider.ask(req, lid)
    pred = None
    va = outcome.validated
    if va is not None and va.answers:
        pred = list(va.answers.values())[0].values.get("choice")
    att = outcome.attempts[0] if outcome.attempts else None
    usage_raw = (getattr(att, "extra", {}) or {}).get("usage_raw") or {}
    reported = usage_raw.get("cost")
    reported = float(reported) if isinstance(reported, (int, float)) else None
    return {
        "logical_request_id": lid, "model": model_id, "dataset": ds,
        "item_id": str(item["id"]), "gold": gold, "predicted": pred,
        "correct": (pred == gold) if pred is not None and gold is not None else False,
        "status": outcome.status, "terminal": outcome.terminal,
        "usage_in": outcome.usage_input_tokens, "usage_out": outcome.usage_output_tokens,
        "cost_reported": reported,
        "latency_ms": outcome.total_latency_ms,
        "http_status": getattr(att, "http_status", None),
        "error": (getattr(att, "error", None) or "")[:200] or None,
        "finish_reason": (getattr(att, "extra", {}) or {}).get("finish_reason"),
        "at": utc_now(),
    }


def cmd_live() -> int:
    key = _live_gate()
    roster = _load_roster()
    wires = load_wire_map()
    if not wires:
        raise SystemExit("[live] run --smoke first (wire map required)")
    roster = [m for m in roster if m["id"] in wires]
    items = load_items()
    RUN_ROOT.mkdir(exist_ok=True)
    results_p = RUN_ROOT / "results.jsonl"
    attempts_p = RUN_ROOT / "attempts.jsonl"
    done = set()
    if results_p.exists():
        for line in results_p.read_text().split("\n"):
            if line.strip():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue  # tolerate a torn tail line from an interrupted run
                if r.get("terminal"):
                    done.add(r["logical_request_id"])
    per_model = []
    for m in roster:
        mt = []
        for ds, rows in items.items():
            for it in rows:
                lid = logical_id(m["id"], ds, str(it["id"]))
                if lid not in done:
                    mt.append((m["id"], ds, it))
        per_model.append(mt)
    tasks = []
    for i in range(max((len(t) for t in per_model), default=0)):
        for mt in per_model:
            if i < len(mt):
                tasks.append(mt[i])
    print(f"[live] {len(roster)} models, {len(tasks):,} calls to dispatch "
          f"({len(done):,} already terminal)")
    if not tasks:
        return 0

    lock = threading.Lock()
    att_f = attempts_p.open("a")
    res_f = results_p.open("a")
    spend = [0.0]
    stop = threading.Event()
    counts = {"done": 0, "format_fail": 0, "error": 0}
    local = threading.local()
    sem_model = {m["id"]: threading.Semaphore(CONCURRENCY_PER_MODEL) for m in roster}
    prices = {m["id"]: (m["input_per_M"] / 1e6, m["output_per_M"] / 1e6) for m in roster}

    def get_provider(model_id):
        cache = getattr(local, "prov", None)
        if cache is None:
            cache = local.prov = {}
        if model_id not in cache:
            w = wires[model_id]
            transport = OpenRouterHttpTransport(key)
            cache[model_id] = OpenRouterProvider(
                transport, wire=WireConfig(model=model_id,
                                           reasoning_effort=w.get("reasoning_effort"),
                                           max_output_tokens=WIRE_MAX_OUT))
        return cache[model_id]

    def work(task):
        model_id, ds, it = task
        if stop.is_set():
            return None
        lid = logical_id(model_id, ds, str(it["id"]))
        sem = sem_model[model_id]
        with sem:
            for attempt in range(TRANSPORT_RETRIES + 1):
                if stop.is_set():
                    return None
                try:
                    rec = _dispatch_one(get_provider(model_id), it, ds, model_id, wires)
                except Exception as exc:
                    rec = {"logical_request_id": lid, "model": model_id, "dataset": ds,
                           "item_id": str(it["id"]), "status": "exception",
                           "terminal": attempt >= TRANSPORT_RETRIES,
                           "error": str(exc)[:200], "at": utc_now()}
                transport_fail = rec["status"] in {"error", "exception", "timeout"} or (
                    rec.get("http_status") in (429, 500, 502, 503, 529))
                if transport_fail and attempt < TRANSPORT_RETRIES:
                    time.sleep(min(2 ** attempt * 1.0, 20.0)
                               + (5.0 if rec.get("http_status") == 429 else 0.0))
                    rec["transport_retry"] = attempt + 1
                    with lock:
                        att_f.write(json.dumps(rec) + "\n")
                        att_f.flush()
                    continue
                break
        cost = rec.get("cost_reported")
        if cost is None and rec.get("usage_in") is not None:
            pin, pout = prices[model_id]
            cost = rec["usage_in"] * pin + (rec.get("usage_out") or 0) * pout
        with lock:
            att_f.write(json.dumps(rec) + "\n")
            att_f.flush()
            if rec.get("terminal", True):
                res_f.write(json.dumps(rec) + "\n")
                res_f.flush()
            if cost:
                spend[0] += cost
                if spend[0] >= HARD_CAP_USD:
                    stop.set()
                    print(f"[live] BUDGET CAP ${HARD_CAP_USD} reached (${spend[0]:.2f}); stopping")
            counts["done"] += 1
            if rec.get("status") == "ok" and rec.get("predicted") is None:
                counts["format_fail"] += 1
            if rec.get("status") in {"error", "exception", "timeout"}:
                counts["error"] += 1
            if counts["done"] % 500 == 0:
                print(f"[live] {counts['done']:,}/{len(tasks):,} "
                      f"spend=${spend[0]:.2f} fmt_fail={counts['format_fail']} "
                      f"err={counts['error']}", flush=True)
        return rec

    t0 = time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=CONCURRENCY_TOTAL) as ex:
            futs = [ex.submit(work, t) for t in tasks]
            for _ in as_completed(futs):
                pass
    finally:
        att_f.close()
        res_f.close()
    el = time.monotonic() - t0
    print(f"[live] finished {counts['done']:,} calls in {el/60:.1f} min; "
          f"spend=${spend[0]:.2f}; format_fail={counts['format_fail']}; "
          f"errors={counts['error']}; stopped={stop.is_set()}")
    (RUN_ROOT / "live_summary.json").write_text(json.dumps({
        "finished_at": utc_now(), "calls": counts["done"], "spend_usd": round(spend[0], 4),
        "format_fail": counts["format_fail"], "errors": counts["error"],
        "budget_stop": stop.is_set(), "wall_seconds": round(el)}, indent=1))
    return 0

def _mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def _jev_per_item() -> dict[str, dict[str, bool]]:
    """Jev correctness per dataset per item id, from working-tree per_item lists."""
    import glob as _glob
    src = {
        "mmlu": "runs_benchmark/bench-mmlu_full-*/derived/score.json",
        "gpqa": "runs_benchmark_ext2/bench-gpqa_diamond-*/derived/score.json",
        "math500_choice": "runs_benchmark_ext/bench-math500_choice-*/derived/score.json",
        "hle_text_mc": "runs_benchmark_ext2/bench-hle_text_mc-*/derived/score.json",
    }
    out = {}
    for ds, pat in src.items():
        f = _glob.glob(str(ROOT / pat))
        if not f:
            continue
        d = json.loads(Path(f[0]).read_text())
        m = {}
        for it in d.get("per_item") or []:
            iid = str(it["item_id"]).split(":")[-1]
            m[iid] = bool(it.get("correct"))
        out[ds] = m
    return out


def cmd_score() -> int:
    res_p = RUN_ROOT / "results.jsonl"
    if not res_p.exists():
        raise SystemExit("[score] no results.jsonl - run --live first")
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
        terminal = [r for r in rs if r.get("terminal", True)]
        valid = [r for r in terminal if r.get("status") == "ok" and r.get("predicted")]
        fmt_fail = len(terminal) - len(valid)
        errors = sum(1 for r in rs if r.get("status") in {"error", "exception", "timeout"})
        acc_all = sum(1 for r in rs if r.get("correct")) / n if n else None
        acc_valid = (sum(1 for r in valid if r.get("correct")) / len(valid)) if valid else None
        tin = sum(r.get("usage_in") or 0 for r in rs)
        tout = sum(r.get("usage_out") or 0 for r in rs)
        cost_rep = sum(r.get("cost_reported") or 0 for r in rs)
        lats = sorted(r.get("latency_ms") or 0 for r in rs if r.get("latency_ms"))
        med_lat = lats[len(lats) // 2] if lats else None
        entry = {
            "n_requested": n, "n_terminal": len(terminal), "n_valid": len(valid),
            "format_failed": fmt_fail, "transport_errors": errors,
            "accuracy_all_requested": acc_all, "accuracy_valid_only": acc_valid,
            "usage_input_tokens": tin, "usage_output_tokens": tout,
            "cost_reported_usd": round(cost_rep, 4),
            "cost_per_question_usd": round(cost_rep / n, 8) if n and cost_rep else None,
            "median_latency_ms": med_lat,
        }
        # Jev join on the same items + McNemar
        jm = jev.get(ds)
        if jm:
            pairs = [(jm.get(r["item_id"]), bool(r.get("correct")))
                     for r in rs if r["item_id"] in jm]
            if pairs:
                both = sum(1 for j, x in pairs if j and x)
                jev_only = sum(1 for j, x in pairs if j and not x)
                mod_only = sum(1 for j, x in pairs if not j and x)
                neither = sum(1 for j, x in pairs if not j and not x)
                entry["jev_join"] = {
                    "n_paired": len(pairs),
                    "jev_accuracy_on_subset": round((both + jev_only) / len(pairs), 4),
                    "both_correct": both, "jev_only": jev_only,
                    "model_only": mod_only, "neither": neither,
                    "mcnemar_exact_p": round(_mcnemar_exact(jev_only, mod_only), 4),
                }
        summary.setdefault(mid, {})[ds] = entry
    doc = {"created_at": utc_now(), "n_result_rows": len(rows),
           "hard_cap_usd": HARD_CAP_USD, "models": summary}
    live_p = RUN_ROOT / "live_summary.json"
    if live_p.exists():
        doc["dispatch"] = json.loads(live_p.read_text())
    (RUN_ROOT / "summary.json").write_text(json.dumps(doc, indent=1))
    # console table
    print(f"{'model':40s} {'dataset':16s} {'acc(all)':>8s} {'fmt%':>5s} "
          f"{'$/q':>10s} {'jev-sub':>8s} {'mcnemar':>8s}")
    for mid, dss in summary.items():
        for ds, e in dss.items():
            jj = e.get("jev_join") or {}
            print(f"{mid:40s} {ds:16s} "
                  f"{(e['accuracy_all_requested'] or 0)*100:7.1f}% "
                  f"{100*e['format_failed']/max(e['n_terminal'],1):4.1f}% "
                  f"{(e['cost_per_question_usd'] or 0):10.6f} "
                  f"{(jj.get('jev_accuracy_on_subset') or 0)*100:7.1f}% "
                  f"{jj.get('mcnemar_exact_p', ''):>8}")
    print(f"[score] wrote {RUN_ROOT.relative_to(ROOT)}/summary.json")
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
