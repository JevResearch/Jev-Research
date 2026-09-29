#!/usr/bin/env python3
"""Matched cheap-model baselines, v3 - SPEC-COMPLIANT wires (audit note 5).

The v2 sweep pinned temperature=0 and reasoning_effort=low on every model.
That is out of spec for reasoning models (GLM-5.3 documents default effort
max for leaderboard reproduction; OpenAI's gpt-oss documents default
sampling; DeepSeek/Qwen thinking models have their own defaults) and it
showed: GLM-5.3 scored BELOW GLM-5.3-Flash on MMLU and 74.5 vs vals' 88.1 on
GPQA at ~1k thinking tokens against vals' ~50k. v3 re-dispatches every model
with NO sampling parameters and NO reasoning parameters - provider defaults,
i.e. each model's documented spec - and a generous max_tokens so thinking is
never truncated (caps per model below; a cap is a bill guard, not a thinking
budget, except where noted).

Datasets per model are budget-scoped (owner grant: $80 total; ~$50 remained
when v3 was planned; hard cap here $40):
  - the four priciest reasoners skip HLE (long prompts x big thinking);
  - qwen3.8-max runs GPQA only (its $6/M thinking makes MMLU ~$18; its v2
    MMLU cell stays in the summary, labeled effort-low - flagged in report);
  - glm-5.3 skips HLE/ARC; ARC re-runs spec for glm-5.3-flash,
    qwen3.8-flash, mimo-v2.6-flash.
Discipline otherwise identical to v2: one attempt, recovery parser applied
identically to all, transport-only retries, key from environment, live gate.
Artifacts: runs_matched_cheap/v3/. Scoring merges v2 cells for models v3
does not re-run (see --score).

  python scripts/benchmark/run_matched3.py --plan
  JEVO_ALLOW_LIVE=1 OPENROUTER_API_KEY=... python scripts/benchmark/run_matched3.py --live
  python scripts/benchmark/run_matched3.py --score
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("rcm2", _HERE / "run_cheap_matched2.py")
rcm2 = importlib.util.module_from_spec(_spec)
sys.modules["rcm2"] = rcm2
_spec.loader.exec_module(rcm2)

from jev_observatory.matched import build_baseline_request, logical_id
from jev_observatory.openrouter import (CHAT_PATH, OpenRouterHttpTransport,
                                        WireConfig, build_chat_payload)

ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "runs_matched_cheap/v3"
HARD_CAP_USD = 32.0
TOTAL_WORKERS = 64
PER_MODEL = 16
RETRIES = 6

FULL = ["mmlu", "gpqa", "math500_choice", "hle_text_mc"]
# (model, datasets, max_tokens, note)
PLAN = [
    ("z-ai/glm-5.3", ["mmlu", "gpqa", "math500_choice"], 32768,
     "documented default effort max; finish first (partially complete)"),
    ("meta-llama/llama-3.1-8b-instruct", FULL, 1024, "non-thinking; defaults"),
    ("mistralai/mistral-nemo", FULL, 1024, "non-thinking; defaults"),
    ("google/gemma-3-4b-it", FULL, 4096, "non-thinking; defaults"),
    ("mistralai/mistral-small-3.2-24b-instruct", FULL, 4096,
     "non-thinking; defaults"),
    ("z-ai/glm-5.3-flash", ["mmlu", "gpqa", "math500_choice"], 32768,
     "provider defaults; hle/arc keep v2 cells (budget/time scope)"),
    ("openai/gpt-oss-20b", ["mmlu", "gpqa", "math500_choice"], 16384,
     "default sampling/effort; hle keeps v2 cell"),
    ("qwen/qwen3.7-flash", ["mmlu", "gpqa", "math500_choice"], 16384,
     "provider defaults; hle keeps v2 cell"),
    ("xiaomi/mimo-v2.6-flash", ["mmlu", "gpqa", "math500_choice"], 32768,
     "provider defaults; hle/arc keep v2 cells"),
    # gpt-oss-120b: v2 cells kept (labeled) - scope cut on wall-time
    ("qwen/qwen3.8-flash", ["mmlu", "gpqa", "math500_choice"], 16384,
     "provider defaults; hle/arc keep v2 cells"),
    ("deepseek/deepseek-v4-flash-0731", ["mmlu", "gpqa", "math500_choice"], 32768,
     "provider defaults; hle keeps v2 cell"),
    ("xiaomi/mimo-v2.6-pro", ["mmlu", "gpqa", "math500_choice"], 32768,
     "provider defaults"),
    ("qwen/qwen3.8-max-0902", ["gpqa"], 16384,
     "reasoning mandatory; GPQA only on budget (v2 MMLU cell kept, labeled)"),
]


def cmd_plan() -> int:
    items = rcm2.load_items()
    fz = json.loads((ROOT / "runs_matched_cheap/v2/freeze_v2.json").read_text())
    prices = {m["id"]: m for m in fz["roster"]}
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for mid, dss, cap, note in PLAN:
        n = sum(len(items[d]) for d in dss)
        pr = prices.get(mid, {})
        rows.append({"id": mid, "datasets": dss, "max_tokens": cap, "note": note,
                     "n_calls": n,
                     "input_per_M": pr.get("input_per_M"),
                     "output_per_M": pr.get("output_per_M")})
        print(f"  {mid:44s} {n:5d} calls  cap={cap:6d}  {note}")
    (RUN_ROOT / "freeze_v3.json").write_text(json.dumps({
        "created_at": rcm2.utc_now(),
        "wire_policy": "NO temperature/top_p/seed, NO reasoning params - provider "
                       "defaults (documented spec); max_tokens is a bill guard only",
        "hard_cap_usd": HARD_CAP_USD, "models": rows,
        "total_calls": sum(r["n_calls"] for r in rows)}, indent=1))
    print(f"[plan] {sum(r['n_calls'] for r in rows):,} calls total; "
          f"hard cap ${HARD_CAP_USD}")
    return 0


def cmd_probe() -> int:
    """10 items/model at provider defaults: measure real thinking burn and
    extrapolate per-cell cost before committing the budget."""
    key = rcm2._live_gate()
    plan = json.loads((RUN_ROOT / "freeze_v3.json").read_text())["models"]
    items = rcm2.load_items()
    out = []
    for m in plan:
        mid, cap = m["id"], m["max_tokens"]
        wire = WireConfig(model=mid, reasoning_effort=None,
                          max_output_tokens=cap, extra_body=None)
        transport = OpenRouterHttpTransport(key, timeout_seconds=300.0)
        rec = {"id": mid, "calls": []}
        for ds, n in (("mmlu", 5), ("gpqa", 5)):
            if ds not in m["datasets"]:
                continue
            for it in items[ds][:n]:
                req, _ = build_baseline_request(it, model=mid)
                keys = list(req.questions[next(iter(req.questions))].criteria)
                try:
                    resp = transport.post(CHAT_PATH, build_chat_payload(req, wire))
                    body = json.loads(resp.body.decode("utf-8", "replace")) if resp.body else None
                    content, reasoning = rcm2._extract_content_reasoning(body)
                    usage = (body or {}).get("usage") if isinstance(body, dict) else None
                    pred, stage = rcm2.recover_choice(content, keys, reasoning)
                    rec["calls"].append({
                        "ds": ds, "http": resp.status_code,
                        "out": (usage or {}).get("completion_tokens"),
                        "rtok": ((usage or {}).get("completion_tokens_details") or {}).get("reasoning_tokens"),
                        "cost": (usage or {}).get("cost"), "stage": stage})
                except Exception as exc:
                    rec["calls"].append({"ds": ds, "error": str(exc)[:150]})
                time.sleep(0.3)
        transport.close()
        ok = [c for c in rec["calls"] if c.get("cost") is not None]
        if ok:
            avg_cost = sum(c["cost"] for c in ok) / len(ok)
            avg_out = sum(c.get("out") or 0 for c in ok) / len(ok)
            per_ds = {}
            for m2 in plan:
                if m2["id"] == mid:
                    per_ds = {d: items[d].__len__() for d in m2["datasets"]}
            est = {d: round(avg_cost * n, 2) for d, n in per_ds.items()}
            rec["avg_cost_per_call"] = round(avg_cost, 6)
            rec["avg_out_tokens"] = round(avg_out)
            rec["extrapolated_cell_costs_usd"] = est
            rec["extrapolated_total_usd"] = round(sum(est.values()), 2)
        out.append(rec)
        print(f"[probe] {mid:44s} avg_out={rec.get('avg_out_tokens')} "
              f"avg$={rec.get('avg_cost_per_call')} est_total=${rec.get('extrapolated_total_usd')}")
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    (RUN_ROOT / "probe_spec.json").write_text(json.dumps(
        {"created_at": rcm2.utc_now(), "results": out}, indent=1))
    tot = sum(r.get("extrapolated_total_usd") or 0 for r in out)
    print(f"[probe] extrapolated FULL sweep: ${tot:.2f}; wrote probe_spec.json")
    return 0


def cmd_live() -> int:
    key = rcm2._live_gate()
    plan = json.loads((RUN_ROOT / "freeze_v3.json").read_text())["models"]
    items = rcm2.load_items()
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
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
    per_model = []
    for m in plan:
        mt = []
        for ds in m["datasets"]:
            for it in items[ds]:
                lid = logical_id(m["id"], ds, str(it["id"]))
                if lid not in done:
                    mt.append((m["id"], ds, it, m["max_tokens"]))
        per_model.append(mt)
    # interleaved across models: one throttled provider (gemma 429s) cannot
    # block the others; the daily-budget concern that motivated sequential
    # order was resolved by the owner
    tasks = []
    for i in range(max((len(t) for t in per_model), default=0)):
        for mt in per_model:
            if i < len(mt):
                tasks.append(mt[i])
    print(f"[live] {len(plan)} models, {len(tasks):,} calls "
          f"({len(done):,} already terminal)", flush=True)
    if not tasks:
        return 0

    lock = threading.Lock()
    res_f = results_p.open("a")
    spend = [0.0]
    stop = threading.Event()
    counts = {"done": 0, "unrec": 0, "err": 0}
    local = threading.local()
    sems = {m["id"]: threading.Semaphore(PER_MODEL) for m in plan}

    def transport():
        if not hasattr(local, "tr"):
            local.tr = OpenRouterHttpTransport(key, timeout_seconds=900.0)
        return local.tr

    def work(task):
        mid, ds, it, cap = task
        if stop.is_set():
            return
        lid = logical_id(mid, ds, str(it["id"]))
        req, _ = build_baseline_request(it, model=mid)
        gold = rcm2.gold_of(it)
        keys = list(req.questions[next(iter(req.questions))].criteria)
        wire = WireConfig(model=mid, reasoning_effort=None,
                          max_output_tokens=cap, extra_body=None)
        payload = build_chat_payload(req, wire)
        rec = {"logical_request_id": lid, "model": mid, "dataset": ds,
               "item_id": str(it["id"]), "gold": gold, "wire": wire.to_dict(),
               "protocol": "provider-defaults"}
        with sems[mid]:
            for attempt in range(RETRIES + 1):
                if stop.is_set():
                    return
                t0 = time.monotonic()
                try:
                    resp = transport().post(CHAT_PATH, payload)
                    ms = round((time.monotonic() - t0) * 1000)
                    body = json.loads(resp.body.decode("utf-8", "replace")) if resp.body else None
                    if resp.status_code in (429, 500, 502, 503, 529) and attempt < RETRIES:
                        time.sleep(min(2 ** attempt, 45) + (15 if resp.status_code == 429 else 0))
                        continue
                    content, reasoning = rcm2._extract_content_reasoning(body)
                    usage = (body or {}).get("usage") if isinstance(body, dict) else None
                    pred_strict = content.strip() if content.strip() in keys else None
                    pred, stage = rcm2.recover_choice(content, keys, reasoning)
                    rec.update({
                        "status": "ok" if resp.status_code == 200 else "http_error",
                        "http_status": resp.status_code, "latency_ms": ms,
                        "pred_strict": pred_strict, "pred_recovered": pred,
                        "recovery_stage": stage,
                        "correct_strict": pred_strict == gold if pred_strict else False,
                        "correct_recovered": pred == gold if pred else False,
                        "usage_in": (usage or {}).get("prompt_tokens"),
                        "usage_out": (usage or {}).get("completion_tokens"),
                        "reasoning_tokens": ((usage or {}).get("completion_tokens_details") or {}).get("reasoning_tokens"),
                        "cost_reported": (usage or {}).get("cost"),
                        "terminal": True, "at": rcm2.utc_now(),
                    })
                    if isinstance(body, dict) and body.get("error"):
                        rec["error"] = str(body["error"])[:200]
                    break
                except Exception as exc:
                    if attempt >= RETRIES:
                        rec.update({"status": "exception", "error": str(exc)[:200],
                                    "terminal": True, "at": rcm2.utc_now()})
                    else:
                        time.sleep(min(2 ** attempt, 45))
        c = rec.get("cost_reported")
        with lock:
            res_f.write(json.dumps(rec) + "\n")
            res_f.flush()
            if isinstance(c, (int, float)):
                spend[0] += float(c)
                if spend[0] >= HARD_CAP_USD:
                    stop.set()
                    print(f"[live] CAP ${HARD_CAP_USD} hit (${spend[0]:.2f})", flush=True)
            counts["done"] += 1
            if rec.get("recovery_stage") == "unrecovered":
                counts["unrec"] += 1
            if rec.get("status") != "ok":
                counts["err"] += 1
            if counts["done"] % 250 == 0:
                print(f"[live] {counts['done']:,}/{len(tasks):,} "
                      f"spend=${spend[0]:.2f} unrec={counts['unrec']} "
                      f"err={counts['err']}", flush=True)

    t0 = time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=TOTAL_WORKERS) as ex:
            futs = [ex.submit(work, t) for t in tasks]
            for _ in as_completed(futs):
                pass
    finally:
        res_f.close()
    el = time.monotonic() - t0
    print(f"[live] {counts['done']:,} calls in {el/60:.1f} min; "
          f"spend=${spend[0]:.2f}; unrec={counts['unrec']}; err={counts['err']}")
    (RUN_ROOT / "live_summary.json").write_text(json.dumps({
        "finished_at": rcm2.utc_now(), "calls": counts["done"],
        "spend_usd": round(spend[0], 4), "unrecovered": counts["unrec"],
        "errors": counts["err"], "budget_stop": stop.is_set(),
        "wall_seconds": round(el)}, indent=1))
    return 0


def _read(path):
    out = []
    if Path(path).exists():
        for l in Path(path).read_text().split("\n"):
            if l.strip():
                try:
                    out.append(json.loads(l))
                except json.JSONDecodeError:
                    pass
    return out


def cmd_score() -> int:
    plan = json.loads((RUN_ROOT / "freeze_v3.json").read_text())["models"]
    v3_models = {m["id"] for m in plan}
    rows_v3 = _read(RUN_ROOT / "results.jsonl")
    rows_v2 = _read(ROOT / "runs_matched_cheap/v2/results.jsonl")
    items = rcm2.load_items()
    n_req = {ds: len(v) for ds, v in items.items()}
    # cell-level merge: a COMPLETE v3 cell supersedes v2; an incomplete v3
    # cell (budget truncation) falls back to the complete v2 cell, labeled.
    def _cells(rows):
        cells = {}
        for r in rows:
            cells.setdefault((r["model"], r["dataset"]), []).append(r)
        return cells
    c3, c2 = _cells(rows_v3), _cells(rows_v2)
    rows = []
    for key, rs in c3.items():
        need = n_req.get(key[1], len(rs))
        ok3 = sum(1 for r in rs if r.get("status") == "ok")
        ok2 = sum(1 for r in c2.get(key, []) if r.get("status") == "ok")
        if ok3 >= need or key not in c2 or ok2 < need:
            rows += rs
        else:
            rows += c2[key]
    for key, rs in c2.items():
        if key not in c3 and (key[0] not in v3_models
                              or key == ("qwen/qwen3.8-max-0902", "mmlu")):
            rows += rs
    jev = None
    try:
        jev = _jev()
    except Exception:
        jev = {}
    by = {}
    for r in rows:
        by.setdefault((r["model"], r["dataset"]), []).append(r)
    summary = {}
    for (mid, ds), rs in sorted(by.items()):
        n = n_req.get(ds, len(rs))
        acc_r = sum(1 for r in rs if r.get("correct_recovered")) / n
        acc_s = sum(1 for r in rs if r.get("correct_strict")) / n
        stages = {}
        for r in rs:
            stages[r.get("recovery_stage")] = stages.get(r.get("recovery_stage"), 0) + 1
        cost = sum(r.get("cost_reported") or 0 for r in rs)
        tin = sum(r.get("usage_in") or 0 for r in rs)
        tout = sum(r.get("usage_out") or 0 for r in rs)
        entry = {
            "n_requested": n, "n_terminal": len(rs),
            "accuracy_strict": round(acc_s, 4), "accuracy_recovered": round(acc_r, 4),
            "recovery_stages": stages, "unrecovered": stages.get("unrecovered", 0),
            "usage_input_tokens": tin, "usage_output_tokens": tout,
            "cost_reported_usd": round(cost, 4),
            "cost_per_question_usd": round(cost / n, 8) if n else None,
            "protocol": ("provider-defaults (v3)"
                         if (mid, ds) in {(r["model"], r["dataset"]) for r in rows_v3}
                         and len([r for r in rows_v3
                                  if r["model"] == mid and r["dataset"] == ds]) >= n
                         else "v2 (temperature 0, effort low where accepted)"),
        }
        if mid in v3_models and ds != "mmlu" or mid not in v3_models:
            pass
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
                    "mcnemar_exact_p": round(rcm2 and __import__("math") and _mn(b, c), 4),
                }
        summary.setdefault(mid, {})[ds] = entry
    doc = {"created_at": rcm2.utc_now(),
           "protocol": "v3 spec-compliant merges (provider defaults); models not "
                       "re-run in v3 keep their v2 cells, labeled",
           "n_result_rows": len(rows), "hard_cap_usd": HARD_CAP_USD,
           "models": summary}
    ls = RUN_ROOT / "live_summary.json"
    if ls.exists():
        doc["dispatch"] = json.loads(ls.read_text())
    (RUN_ROOT / "summary_v3.json").write_text(json.dumps(doc, indent=1))
    print(f"{'model':44s} {'dataset':15s} {'strict':>7s} {'recov':>7s} {'$/q':>10s} {'prot':>6s}")
    for mid, dss in summary.items():
        for ds, e in dss.items():
            tag = "v3" if e["protocol"].startswith("provider") else "v2"
            print(f"{mid:44s} {ds:15s} {e['accuracy_strict']*100:6.1f}% "
                  f"{e['accuracy_recovered']*100:6.1f}% "
                  f"{(e['cost_per_question_usd'] or 0):10.6f} {tag:>6s}")
    print(f"[score] wrote {RUN_ROOT.relative_to(ROOT)}/summary_v3.json")
    return 0


def _mn(b, c):
    import math
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def _jev():
    import glob
    src = {
        "mmlu": "runs_benchmark/bench-mmlu_full-*/derived/score.json",
        "gpqa": "runs_benchmark_ext2/bench-gpqa_diamond-*/derived/score.json",
        "math500_choice": "runs_benchmark_ext/bench-math500_choice-*/derived/score.json",
        "hle_text_mc": "runs_benchmark_ext2/bench-hle_text_mc-*/derived/score.json",
        "arc": "runs_benchmark/bench-arc_test-*/derived/score.json",
    }
    out = {}
    for ds, pat in src.items():
        f = glob.glob(str(ROOT / pat))
        if not f:
            continue
        d = json.loads(Path(f[0]).read_text())
        out[ds] = {str(it["item_id"]).split(":")[-1]: bool(it.get("correct"))
                   for it in d.get("per_item") or []}
    return out


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--plan", action="store_true")
    g.add_argument("--probe", action="store_true")
    g.add_argument("--live", action="store_true")
    g.add_argument("--score", action="store_true")
    a = ap.parse_args()
    if a.plan:
        return cmd_plan()
    if a.probe:
        return cmd_probe()
    if a.live:
        return cmd_live()
    return cmd_score()


if __name__ == "__main__":
    sys.exit(main())

