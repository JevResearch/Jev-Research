#!/usr/bin/env python3
"""Template-free per-script tokenizer match (decisive exact-match test).

Consumes the MERGERATE family rows written by run_arch_probe.py (whitespace-free
long samples per writing system, median over reps) and, for each candidate
tokenizer, computes the reference token count of the EXACT serialized payload the
probe sent. reported = a + b * ref_count is fit by least squares (a absorbs the
fixed template; b is the relative efficiency). A correct tokenizer fits ALL 31
samples near-perfectly (residual ~0); a wrong vocabulary leaves large,
script-systematic residuals.

We report, per candidate: fit slope, RMSE/max residual, and the residual split
by script family (this split is the real discriminator).

  python scripts/benchmark/tokenizer_perscript.py  (HF_TOKEN in env for gated)
"""

from __future__ import annotations
import argparse, json, math, os, sys, urllib.request
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

_SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SRC / "src"))
from jev_observatory.arch_probe import _choice
from jev_observatory.arch_probe_mergerate import SAMPLES

HF = "https://huggingface.co"
TOKEN = os.environ.get("HF_TOKEN") or ""
TEXT = {name: text for name, _s, text in SAMPLES}
SCRIPT = {name: sc for name, sc, _t in SAMPLES}


def fetch(u, timeout=90):
    try:
        h = {"User-Agent": "jev/1.0"}
        if TOKEN:
            h["Authorization"] = "Bearer " + TOKEN
        with urllib.request.urlopen(urllib.request.Request(u, headers=h), timeout=timeout) as r:
            return r.read()
    except Exception:
        return None


def payload(state):
    return json.dumps(_choice(["yes", "no"], "Is the sequence empty?", state).to_payload(),
                      ensure_ascii=False)


def load_jev(rows_path):
    rep = defaultdict(list)
    for line in Path(rows_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("family") == "mergerate" and isinstance(r.get("usage_input_tokens"), int) \
           and r.get("http_status") == 200:
            rep[r["sample"]].append(float(r["usage_input_tokens"]))
    return {s: median(v) for s, v in rep.items()}


def fit_and_resid(ref, jev):
    xs = [ref[s] for s in jev]; ys = [jev[s] for s in jev]
    n = len(xs); mx = sum(xs)/n; my = sum(ys)/n
    sxx = sum((x-mx)**2 for x in xs)
    if sxx == 0: return None
    b = sum((x-mx)*(y-my) for x, y in zip(xs, ys))/sxx; a = my - b*mx
    per = {s: (ref[s], ys[i], ys[i]-(a+b*xs[i])) for i, s in enumerate(jev)}
    errs = [t[2] for t in per.values()]
    rmse = math.sqrt(mean(e*e for e in errs))
    return {"slope": round(b, 4), "intercept": round(a, 1), "rmse": round(rmse, 3),
            "max_abs": round(max(abs(e) for e in errs), 2),
            "worst": sorted(((SCRIPT[s], s, round(t[2], 1)) for s, t in per.items()),
                            key=lambda x: -abs(x[2]))[:6],
            "per_script": {sc: round(mean([t[2] for s, t in per.items() if SCRIPT[s] == sc]), 2)
                           for sc in set(SCRIPT.values()) if any(SCRIPT[s] == sc for s in per)}}


def candidates():
    ids = ["tencent/Hy-MT2-7B", "tencent/Youtu-LLM-2B", "Qwen/Qwen2.5-7B", "Qwen/Qwen3-8B",
           "zai-org/GLM-4.5", "mistralai/Mistral-7B-v0.1", "google/gemma-3-4b-it",
           "google/gemma-2-2b", "NousResearch/Meta-Llama-3-8B", "meta-llama/Llama-3.1-8B",
           "deepseek-ai/DeepSeek-V3", "microsoft/Phi-3-medium-4k-instruct", "microsoft/Phi-4",
           "microsoft/phi-2", "openai/gpt-oss-20b", "01-ai/Yi-6B", "XiaomiMiMo/MiMo-7B-Base",
           "moonshotai/Kimi-K2-Instruct", "upstage/solar-pro-280B-Instruct", "Qwen/Qwen3.8-27B",
           "Qwen/Qwen3.5-9B"]
    bs = Path("runs_archprobe/tokenizer_broadscan.json")
    if bs.exists():
        for r in json.loads(bs.read_text()).get("ranking", []):
            if r.get("represent"):
                ids.append(r["represent"])
    return list(dict.fromkeys(ids))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", default="runs_archprobe/rows.jsonl")
    ap.add_argument("--out", default="runs_archprobe/tokenizer_perscript.json")
    a = ap.parse_args()
    jev = load_jev(a.rows)
    if len(jev) < 8:
        print("need >=8 mergerate samples in rows.jsonl; run the probe first", file=sys.stderr)
        return 2
    from tokenizers import Tokenizer
    results = []
    for mid in candidates():
        raw = fetch(f"{HF}/{mid}/resolve/main/tokenizer.json")
        if not raw:
            continue
        try:
            tok = Tokenizer.from_str(raw.decode("utf-8"))
        except Exception:
            continue
        empty = payload("SEQ::END")
        ref = {}
        ok = True
        for s in jev:
            try:
                ref[s] = len(tok.encode(payload("SEQ:" + TEXT[s] + ":END"), add_special_tokens=False).ids)
            except Exception:
                ok = False; break
        if not ok or len(set(ref.values())) < 4:
            continue
        r = fit_and_resid(ref, jev)
        if r:
            r["id"] = mid; r["n_samples"] = len(ref)
            results.append(r)
    results.sort(key=lambda r: r["rmse"])
    doc = {"claim_type": "exploratory",
           "method": "affine fit reported~slope*ref_count over 31 whitespace-free mergerate samples; per-script residuals are the discriminator",
           "n_candidates": len(results), "n_samples": len(jev),
           "jev": {s: jev[s] for s in jev}, "ranking": results}
    Path(a.out).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{'rmse':>7}{'maxabs':>7}{'slope':>8}  id")
    for r in results[:20]:
        print(f"{r['rmse']:>7}{r['max_abs']:>7}{r['slope']:>8}  {r['id']}")
        print(f"        worst: {r['worst'][:4]}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
