#!/usr/bin/env python3
"""Exact tokenizer matching from the space-free MERGERATE battery.

Compares Jev's reported token counts for long, whitespace-free samples against
candidate reference tokenizers. Because the samples are long (200-600 chars) and
have NO whitespace, per-sample template/merge behaviour dominates the small
boundary effects at SEQ:/END, giving a far sharper exact-match test than the
short tokenizer battery.

For each candidate tokenizer we fit reported ~ a + b * ref_tokens(whole_payload)
by least squares across the 31 distinct samples (one per family), then report
the residual RMSE. A correct tokenizer fits with slope ~1 and LOW residuals;
a wrong vocabulary leaves large per-script residuals (e.g. Cyrillic/CJK).

Also produces a per-script tokens-per-character table so the reader can see
WHERE a candidate fails.

  python scripts/benchmark/tokenizer_merge_score.py
"""

from __future__ import annotations
import argparse, json, math, os, sys, urllib.request
from pathlib import Path
_SRC = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(_SRC / "src"))
from jev_observatory.arch_probe_mergerate import SAMPLES, _u
from jev_observatory.arch_probe import _choice

HF="https://huggingface.co"; TOKEN=os.environ.get("HF_TOKEN") or ""

def fetch(url, timeout=60):
    hdr={"User-Agent":"jev/1.0"}
    if TOKEN: hdr["Authorization"]=f"Bearer {TOKEN}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url,headers=hdr),timeout=timeout) as r:
            return r.read()
    except Exception: return None

def payload(probe_state):
    req=_choice(["yes","no"],"Is the sequence empty?",probe_state)
    return json.dumps(req.to_payload(),ensure_ascii=False)

def load_jev(rows_path):
    by={}
    for line in Path(rows_path).read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        r=json.loads(line)
        if r.get("family")=="mergerate" and isinstance(r.get("usage_input_tokens"),int):
            by.setdefault(r["sample"],[]).append(r["usage_input_tokens"])
    med={s:float(sorted(v)[len(v)//2]) for s,v in by.items()}
    return med

def states():
    return {name: "SEQ:"+text+":END" for name,_script,text in SAMPLES}

def candidates():
    # curated list of tokenizer-generation representatives to test
    ids = [
      # Qwen generations
      "Qwen/Qwen1.5-7B","Qwen/Qwen2-7B","Qwen/Qwen2.5-7B","Qwen/Qwen3-8B",
      "Qwen/Qwen3.5-9B","Qwen/Qwen3.8-27B","Qwen/QwQ-32B",
      # Llama
      "NousResearch/Meta-Llama-3-8B","meta-llama/Llama-3.1-8B","meta-llama/Llama-2-7b-hf",
      "unsloth/Llama-3.2-1B","meta-llama/Llama-4-Scout-17B-16E-Instruct",
      # Mistral
      "mistralai/Mistral-7B-v0.1","mistralai/Mistral-Nemo-Instruct-2407",
      # Gemma
      "google/gemma-2-2b","google/gemma-3-4b-it","ggml-org/gemma-3-4b-it-GGUF",
      # DeepSeek
      "deepseek-ai/DeepSeek-V3","deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
      # Phi
      "microsoft/Phi-3-medium-4k-instruct","microsoft/Phi-4","microsoft/phi-2",
      # GLM
      "zai-org/GLM-4.5","THUDM/glm-4-9b-chat",
      # Kimi
      "moonshotai/Kimi-K2-Instruct",
      # Tencent / Hy
      "tencent/Hy3-preview","tencent/Hy-MT2-7B","tencent/Youtu-LLM-2B",
      # MiMo
      "XiaomiMiMo/MiMo-7B-Base",
      # Nemotron / others
      "nvidia/NVIDIA-Nemotron-70B-Instruct-Hybrid","upstage/solar-pro-280B-Instruct",
      "01-ai/Yi-6B","MiniMaxAI/MiniMax-Text-01","openai/gpt-oss-20b",
    ]
    # also top reps from the earlier broadscan json if present
    bs=Path("runs_archprobe/tokenizer_broadscan.json")
    if bs.exists():
        for r in json.loads(bs.read_text()).get("ranking",[])[:25]:
            rep=r.get("represent")
            if rep and rep not in ids: ids.append(rep)
    return ids

def score(tokenizer_raw, jev, sts):
    from tokenizers import Tokenizer
    tok=Tokenizer.from_str(tokenizer_raw.decode("utf-8"))
    names=list(sts); xs=[]; ys=[]
    try:
        for n in names:
            t=tok.encode(payload(sts[n]),add_special_tokens=False).ids
            xs.append(len(t)); ys.append(jev[n])
    except Exception:
        return None
    n=len(xs); mx=sum(xs)/n; my=sum(ys)/n
    sxx=sum((x-mx)**2 for x in xs)
    if sxx==0: return None
    b=sum((x-mx)*(y-my) for x,y in zip(xs,ys))/sxx; a=my-b*mx
    errs=[ys[i]-(a+b*xs[i]) for i in range(n)]
    rmse=math.sqrt(sum(e*e for e in errs)/n)
    # worst scripts
    worst=sorted(zip(names,errs),key=lambda t:-abs(t[1]))[:4]
    return {"n":n,"slope":round(b,3),"intercept":round(a,1),"rmse":round(rmse,3),
            "max_abs":round(max(abs(e) for e in errs),2),
            "exact_count":sum(1 for e in errs if abs(e)<=0.5),
            "worst":[(s,round(e,1)) for s,e in worst]}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--rows",default="runs_archprobe/rows.jsonl")
    ap.add_argument("--out",default="runs_archprobe/tokenizer_merge_score.json")
    a=ap.parse_args()
    jev=load_jev(a.rows); sts=states()
    from collections import Counter
    print("scripts:", dict(Counter(s for _n,s,_t in SAMPLES)))
    results=[]
    for mid in candidates():
        raw=fetch(f"{HF}/{mid}/resolve/main/tokenizer.json")
        if not raw: continue
        try: res=score(raw,jev,sts)
        except Exception: res=None
        if res:
            res["id"]=mid; results.append(res)
        print(f"  {mid:52s} rmse={res['rmse'] if res else 'fail'} "
              f"slope={res['slope'] if res else '-'} exact={res['exact_count'] if res else '-'}/{res['n'] if res else '-'}")
    results.sort(key=lambda r:(r["rmse"],-r["exact_count"]))
    Path(a.out).write_text(json.dumps({"claim_type":"exploratory",
        "method":"OLS on 31 long space-free samples; exact=|residual|<=0.5","ranking":results},
        indent=1)+"\n",encoding="utf-8")
    print("\n== BEST (lowest RMSE) ==")
    for r in results[:12]:
        print(f"  rmse={r['rmse']:5.2f} max={r['max_abs']:5.2f} exact={r['exact_count']}/31 "
              f"slope={r['slope']} int={r['intercept']}  {r['id']}")
        if r["worst"]: print(f"       worst: {r['worst']}")
    print(f"\nwrote {a.out}")

if __name__=="__main__":
    import sys
    if "--clean" in sys.argv: sys.argv.remove("--clean"); main_clean()
    else: main()

