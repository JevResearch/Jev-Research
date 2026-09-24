#!/usr/bin/env python3
"""Broad open-source tokenizer scan: find the EXACT tokenizer Jev's endpoint uses.

Cheap clustering first, expensive download second:
  1. enumerate candidate models (curated orgs by popularity + an explicit
     revision-representative list),
  2. build a cheap signature from config.json + tokenizer_config.json ONLY
     (vocab_size + the set of added/special tokens) -- tiny files,
  3. cluster by that signature -> candidate distinct tokenizers,
  4. download tokenizer.json ONLY for each cluster representative (deduped),
  5. score each against Jev with the DELTA method:
        delta(p) = tokens(payload_with_p) - tokens(payload_empty)
     The fixed server template cancels in the difference, so the residual
     reflects only per-string vocabulary + merge behaviour -- this is what makes
     the test sensitive enough to separate tokenizer *revisions*, not just families.
  6. rank by mean absolute residual (tokens).

Requires HF_TOKEN in the environment for gated repos (Llama/Gemma); public repos
work without it. Gated repos that fail to fetch are skipped and counted.
"""

from __future__ import annotations

import argparse, hashlib, json, math, os, sys, urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SRC / "src"))
from jev_observatory.arch_probe import _choice

HF = "https://huggingface.co"
TOKEN = os.environ.get("HF_TOKEN") or ""

AUTHORS = ["Qwen", "microsoft", "google", "meta-llama", "mistralai",
           "deepseek-ai", "zai-org", "THUDM", "moonshotai", "XiaomiMiMo",
           "NVIDIA", "tencent", "01-ai", "MiniMaxAI", "internlm", "CohereLabs",
           "HuggingFaceTB", "EleutherAI", "allenai", "nvidia", "upstage",
           "aisingapore", "KT", "lgai-ai-research", "kakaocorp", "TII",
           "BAAI", "SeaAI", "databricks", "Snowflake", "gryphe", "NousResearch"]

# key revision representatives, incl. models that reveal tokenizer GENERATIONS
EXPLICIT = [
    "Qwen/Qwen-7B", "Qwen/Qwen1.5-7B", "Qwen/Qwen2-7B", "Qwen/Qwen2.5-7B",
    "Qwen/Qwen3-8B", "Qwen/Qwen3.5-9B", "Qwen/Qwen3.8-27B", "Qwen/Qwen3-4B",
    "Qwen/Qwen2.5-Coder-7B", "Qwen/Qwen3-Coder-30B-A3B-Instruct", "Qwen/QwQ-32B",
    "deepseek-ai/deepseek-llm-7b-base", "deepseek-ai/DeepSeek-V2",
    "deepseek-ai/DeepSeek-V3", "deepseek-ai/DeepSeek-R1",
    "microsoft/phi-2", "microsoft/Phi-3-medium-4k-instruct",
    "microsoft/Phi-4", "microsoft/Phi-4-mini-instruct",
    "google/gemma-2b", "google/gemma-2-2b", "google/gemma-3-4b-it",
    "meta-llama/Llama-2-7b-hf", "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-4-Scout-17B-16E", "mistralai/Mistral-7B-v0.1",
    "mistralai/Mistral-Nemo-Instruct-2407", "mistralai/Magistral-2506",
    "THUDM/chatglm3-6b", "THUDM/glm-4-9b-chat", "zai-org/GLM-4.5",
    "moonshotai/Kimi-K2-Instruct", "01-ai/Yi-6B", "01-ai/Yi-1.5-9B-Chat-16K",
    "MiniMaxAI/MiniMax-Text-01", "internlm/internlm2-chat-7b",
    "internlm/internlm3-8b-instruct", "CohereLabs/c4ai-command-r-v01",
    "HuggingFaceTB/SmolLM3-3B", "upstage/solar-pro-280B-Instruct",
    "aisingapore/Gemma-SEA-LAYERN-v2-27B-IT", "snowflake/snowflake-arctic-instruct",
    "nvidia/NVIDIA-Nemotron-70B-Instruct-Hybrid", "databricks/dbrx-instruct",
    "openai/gpt-oss-20b", "XiaomiMiMo/MiMo-7B-Base", "tencent/Hunyuan-A13B-Instruct",
]

def fetch(url, timeout=25):
    hdr = {"User-Agent": "jev/1.0"}
    if TOKEN: hdr["Authorization"] = f"Bearer {TOKEN}"
    try:
        req = urllib.request.Request(url, headers=hdr)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except Exception:
        return None

def jget(url, timeout=25):
    b = fetch(url, timeout)
    if not b: return None
    try: return json.loads(b)
    except Exception: return None

def list_author(author, cap=60):
    ids = []
    for skip in range(0, cap, 50):
        d = jget(f"{HF}/api/models?author={author}&limit=50&skip={skip}"
                 f"&sort=downloads&direction=-1&full=false") or []
        ids += [m["id"] for m in d]
        if len(d) < 50: break
    return ids[:cap]

def sig(mid):
    cfg = jget(f"{HF}/{mid}/resolve/main/config.json") or {}
    tc = jget(f"{HF}/{mid}/resolve/main/tokenizer_config.json")
    if not tc: return None
    atd = tc.get("added_tokens_decoder") or {}
    sp = sorted({t.get("content") for t in atd.values() if isinstance(t, dict)
                 and t.get("content")} | set(tc.get("special_tokens_list") or []))
    if not sp: return None
    vocab = (cfg.get("vocab_size")
             or (cfg.get("text_config") or {}).get("vocab_size")
             or (cfg.get("language_config") or {}).get("vocab_size")
             or (cfg.get("text_config") or {}).get("vocab_size")
             or (cfg.get("llm_config") or {}).get("vocab_size"))
    spk = hashlib.sha256("|".join(sp).encode()).hexdigest()[:10]
    return {"id": mid, "vocab": vocab, "spk": spk, "n_sp": len(sp),
            "key": (vocab, spk)}

def payload_str(probe):
    return json.dumps(_choice(["yes", "no"], "Is the bracketed text empty?",
                               "Token probe [" + probe + "]").to_payload(),
                      ensure_ascii=False)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", default="runs_archprobe/rows.jsonl")
    p.add_argument("--out", default="runs_archprobe/tokenizer_broadscan.json")
    p.add_argument("--cap", type=int, default=60)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--cache", default="/tmp/tokcache")
    a = p.parse_args()
    cache = Path(a.cache); cache.mkdir(parents=True, exist_ok=True)

    ids = list(EXPLICIT)
    for auth in AUTHORS:
        for mid in list_author(auth, a.cap):
            if mid not in ids: ids.append(mid)
    print(f"[catalog] {len(ids)} candidate repos")

    sigs = {}
    with ThreadPoolExecutor(a.workers) as ex:
        for s in ex.map(sig, ids):
            if s: sigs[s["id"]] = s
    print(f"[catalog] {len(sigs)} with a parseable tokenizer_config")

    groups = defaultdict(list)
    for s in sigs.values(): groups[s["key"]].append(s["id"])
    print(f"[cluster] {len(groups)} distinct (vocab+specials) signatures")

    # download one representative tokenizer.json per signature; dedup by content hash
    tok_sha_to_reps = {}
    dl_failed = 0
    for key, members in groups.items():   # members = list of repo-id strings
        rep = members[0]
        raw = fetch(f"{HF}/{rep}/resolve/main/tokenizer.json")
        if not raw:
            dl_failed += 1; continue
        sha = hashlib.sha256(raw).hexdigest()[:16]
        entry = tok_sha_to_reps.setdefault(
            sha, {"bytes": raw, "members": [], "vocab": key[0]})
        entry["members"] += members
    print(f"[download] {len(tok_sha_to_reps)} unique tokenizer.json files "
          f"({dl_failed} reps unfetchable/gated)")

    by = {}
    for line in Path(a.rows).read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        r = json.loads(line)
        if r.get("family")=="tokenizer" and isinstance(r.get("usage_input_tokens"),int) \
           and r.get("probe") is not None:
            by.setdefault(r["probe"], []).append(r["usage_input_tokens"])
    med = {p: sorted(v)[len(v)//2] for p, v in by.items()}
    probes = sorted(med)
    # reported absolute counts (not deltas): LOOCV affine fit absorbs the fixed
    # template offset (intercept) and any uniform scaling (slope) so the metric
    # measures per-string *structure* — what actually separates tokenizers.
    def loocv(counts):
        n = len(probes)
        xs = [counts.get(p) for p in probes]
        ys = [med[p] for p in probes]
        errs = []
        for i in range(n):
            xi = [xs[j] for j in range(n) if j != i and xs[j] is not None]
            yi = [ys[j] for j in range(n) if j != i and xs[j] is not None]
            if len(xi) < 6: return None
            mx = sum(xi)/len(xi); my = sum(yi)/len(yi)
            sxx = sum((x-mx)**2 for x in xi)
            if sxx == 0: continue
            b = sum((x-mx)*(y-my) for x, y in zip(xi, yi))/sxx
            a = my - b*mx
            if xs[i] is not None:
                errs.append(abs(ys[i] - (a + b*xs[i])))
        return errs

    from tokenizers import Tokenizer
    results = []
    for sha, info in tok_sha_to_reps.items():
        try: tok = Tokenizer.from_str(info["bytes"].decode("utf-8"))
        except Exception: continue
        counts = {}
        try:
            for p in probes:
                counts[p] = len(tok.encode(payload_str(p), add_special_tokens=False).ids)
        except Exception: continue
        errs = loocv(counts)
        if not errs or len(errs) < int(len(probes)*0.7): continue
        mean = sum(errs)/len(errs); mx_err = max(errs)
        rmse = math.sqrt(sum(e*e for e in errs)/len(errs))
        # full-set slope (diagnostic: a correct tokenizer fits ~1.0)
        xs=[counts[p] for p in probes]; ys=[med[p] for p in probes]
        Mx=sum(xs)/len(xs); My=sum(ys)/len(ys)
        sxx=sum((x-Mx)**2 for x in xs)
        slope=sum((x-Mx)*(y-My) for x,y in zip(xs,ys))/sxx if sxx else 0
        results.append({"sha": sha, "loocv_mean_abs": round(mean,3),
                        "loocv_rmse": round(rmse,3), "loocv_max_abs": round(mx_err,3),
                        "fit_slope": round(slope,3), "n_probes_used": len(errs),
                        "vocab": info["vocab"], "n_models": len(info["members"]),
                        "represent": info["members"][0],
                        "models_sample": sorted(info["members"])[:6]})
    results.sort(key=lambda r: (r["loocv_mean_abs"], r["loocv_max_abs"]))
    doc = {"claim_type": "exploratory",
           "method": "leave-one-out affine residual in tokens (absorbs template offset + scale)",
           "n_probes": len(probes), "n_tokenizers_scored": len(results),
           "n_repos_scanned": len(ids),
           "ranking": results[:80]}
    Path(a.out).write_text(json.dumps(doc, indent=1, ensure_ascii=False)+"\n", encoding="utf-8")
    print(f"\n== {len(results)} tokenizers scored; best (lowest mean |residual| tokens) ==")
    print(f"{'loocv_mean':>10}{'loocv_rmse':>11}{'loocv_max':>10}{'slope':>7}{'vocab':>8}{'nmodel':>7}  representative")
    for r in results[:30]:
        print(f"{r['loocv_mean_abs']:>10}{r['loocv_rmse']:>11}{r['loocv_max_abs']:>10}"
              f"{r['fit_slope']:>7}{str(r['vocab']):>8}{r['n_models']:>7}  {r['represent']}")
    print(f"\nwrote {a.out}")

if __name__ == "__main__":
    main()
