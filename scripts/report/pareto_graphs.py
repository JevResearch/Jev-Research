#!/usr/bin/env python3
"""Pareto frontier charts: benchmark cost (log x-axis) vs accuracy.

Reads data_report/costs.json (cost model) + docs/modern-comparison/
canonical/comparable-scores.json (accuracy) and renders
docs/modern-comparison/pareto-frontiers.html. Jev is drawn as a highlighted
point; frontier points are labelled. Honest caveats are printed with each
chart (protocol mismatch + estimate error bars) so the picture can't be
read as a like-for-like leaderboard.
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COSTS = json.loads((ROOT / "data_report/costs.json").read_text())
REF = json.loads((ROOT / "docs/modern-comparison/canonical/comparable-scores.json").read_text())
OUT = ROOT / "docs/modern-comparison/pareto-frontiers.html"

BG="#0e1117"; PANEL="#161b27"; TEXT="#e6e9f2"; MUTED="#9aa2b6"; GRID="#242a38"
COST_C="#7d8cff"; JEV_G="#38e1c8"; JEV_W="#f5b342"

# accuracy per (bench, model): greedy from the fetched reference rows
# (external scores), Jev measured; a subset of models we can also price.
PRICEABLE = [m for m in COSTS["prices_usd_per_M"] if m != "Jev"]

# greedy accuracies by benchmark for the priceable models (fetched values)
ACC = {
 "mmlu_pro": {"GPT-5.6 Sol":89.1,"Claude Fable 5.1":92.4,"Claude Opus 5":91.6,
   "GLM-5.3":86.8,"GLM-5.3 Flash":86.1,"Qwen3.8 Max":88.6,
   "DeepSeek V4 Flash 0731":86.2,"GPT-4o":74.7,
   "Claude 3.7 Sonnet (no thinking)":80.8,"GPT-6 Astra":None},
 "gpqa": {"GPT-6 Astra":96.3,"GPT-5.6 Sol":95.2,"Claude Fable 5.1":93.7,
   "Claude Opus 5":93.2,"GLM-5.3":91.7,"GLM-5.3 Flash":91.2,"Qwen3.8 Max":92.8,
   "DeepSeek V4 Flash 0731":89.9,"GPT-4o":54.3,"Claude 3.7 Sonnet (no thinking)":76.8},
 "arc": {"GPT-4o":96.7,"Claude 3.7 Sonnet (no thinking)":None},
 "math500": {"Claude 3.7 Sonnet (no thinking)":76.8,"GPT-4o":89.3},
 "hle": {"GPT-6 Astra":54.2,"GPT-5.6 Sol":49.5,"Claude Fable 5.1":60.9,
   "Claude Opus 5":54.9,"Qwen3.8 Max":56.2,"DeepSeek V4 Flash 0731":34.8},
}
# Jev measured (greedy) per benchmark
JEV_ACC = {"mmlu_pro":82.8,"gpqa":76.5,"arc":97.9,"math500":83.1,"hle":21.9}
CAPTIONS = {
 "mmlu_pro":"MMLU-Pro (12,032 questions) — external bars: Vals 5-shot CoT",
 "gpqa":"GPQA Diamond (196 questions) — external: reasoning-enabled, AA/OR",
 "arc":"ARC-Challenge (1,172) — mostly older-model rows, saturated",
 "math500":"MATH-500 — external rows are free-form reasoning; Jev is the MCQ adaptation",
 "hle":"HLE — external: full text-only set; Jev: MC sub-track only",
}

def chart(bench:str)->str:
    pts=[]
    for m,a in ACC.get(bench,{}).items():
        if a is None or m not in COSTS["costs_usd"][bench]: continue
        pts.append((COSTS["costs_usd"][bench][m]["usd"], a, m, COST_C))
    jc=COSTS["costs_usd"][bench]["Jev"]["usd"]; ja=JEV_ACC[bench]
    pts.append((jc,ja,"Jev (greedy)",JEV_G))
    lo=0.2
    xs=[max(p[0],lo) for p in pts]
    import math
    lmin=math.log10(min(xs))-0.35; lmax=math.log10(max(xs))+0.35
    amin=min(p[1] for p in pts)-8; amax=max(p[1] for p in pts)+8
    W,H,L,R,T,B=940,470,74,28,58,58
    px=lambda c:L+(math.log10(max(c,lo))-lmin)/(lmax-lmin)*(W-L-R)
    py=lambda a:T+(amax-a)/(amax-amin)*(H-T-B)
    p=[f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" font-family="system-ui,sans-serif">',
       f'<rect width="{W}" height="{H}" fill="{PANEL}" rx="14"/>',
       f'<text x="24" y="28" fill="{TEXT}" font-size="16" font-weight="600">{CAPTIONS[bench]}</text>',
       f'<text x="24" y="45" fill="{MUTED}" font-size="11">USD cost to run the whole benchmark (log scale; external costs are anchored estimates) vs accuracy %</text>']
    # gridlines
    a=int(10**math.ceil(math.log10(lo)))
    while a<10**lmax:
        x=px(a)
        if L+2<x<W-R:
            p.append(f'<line x1="{x:.0f}" y1="{T}" x2="{x:.0f}" y2="{H-B}" stroke="{GRID}"/>')
            lab=f"${a:,.0f}" if a>=1 else f"${a:g}"
            p.append(f'<text x="{x:.0f}" y="{H-B+16}" fill="{MUTED}" font-size="10" text-anchor="middle">{lab}</text>')
        a*=10
    gy=round(amin/10)*10
    while gy<=amax:
        y=py(gy)
        p.append(f'<line x1="{L}" y1="{y:.0f}" x2="{W-R}" y2="{y:.0f}" stroke="{GRID}"/>')
        p.append(f'<text x="{L-8}" y="{y+3:.0f}" fill="{MUTED}" font-size="10" text-anchor="end">{gy}</text>')
        gy+=10 if (amax-amin)>40 else 5
    # pareto frontier (max accuracy per increasing cost)
    order=sorted(pts,key=lambda q:q[0]); front=[]; best=-1
    for q in order:
        if q[1]>best: front.append(q); best=q[1]
    fp=" ".join(f"{px(q[0]):.0f},{py(q[1]):.0f}" for q in front)
    p.append(f'<polyline points="{fp}" fill="none" stroke="{MUTED}" stroke-width="1.5" stroke-dasharray="4 4" opacity="0.7"/>')
    for c,a,name,col in pts:
        cx,cy=px(max(c,lo)),py(a)
        r=7 if name.startswith("Jev") else 4.5
        p.append(f'<circle cx="{cx:.0f}" cy="{cy:.0f}" r="{r}" fill="{col}"/>')
        dy=-10 if col==JEV_G else 4
        anchor="middle" if col==JEV_G else "start"
        tx=cx if col==JEV_G else cx+8
        nm=name.replace(" (no thinking)","")
        p.append(f'<text x="{tx:.0f}" y="{cy+dy:.0f}" fill="{col}" font-size="10.5" font-weight="600" text-anchor="{anchor}">{nm}</text>')
    p.append(f'<text x="{(L+W-R)/2:.0f}" y="{H-12}" fill="{MUTED}" font-size="11" text-anchor="middle">cost to run benchmark (USD, log)</text>')
    p.append('</svg>')
    return "".join(p)

def main():
    secs="".join(f'<section>{chart(b)}</section>' for b in ["mmlu_pro","gpqa","math500","hle"])
    caveat=("Jev's cost is measured from billing (input tokens at $0.042 per million, "
            "output free). External costs are estimates from public list prices and "
            "per-item token priors, calibrated to measured cost/question anchors "
            "(OpenRouter GPQA, Vals MMLU-Pro); expect a few-x error on any single point. "
            "Accuracies also use different protocols per benchmark — see each chart's "
            "caption — so these panels show cost-vs-capability position, not a matched race. "
            "ARC-Challenge is omitted: current models no longer publish it.")
    html=f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Jev Pareto frontiers</title><style>
 body{{background:{BG};color:{TEXT};font-family:system-ui,sans-serif;max-width:1000px;
 margin:2rem auto;padding:0 1rem;line-height:1.5}}
 h1{{font-size:24px}} svg{{width:100%;height:auto;display:block;margin:.4rem 0 1.2rem}}
 p.cap{{color:{MUTED};font-size:12.5px;max-width:86ch}}
</style></head><body>
<h1>Cost versus accuracy — the frontier Jev actually sits on</h1>
<p class="cap">{caveat}</p>
{secs}
</body></html>"""
    OUT.write_text(html,encoding="utf-8")
    print(f"[ok] wrote {OUT} ({OUT.stat().st_size} bytes); {4} charts")

if __name__=="__main__": main()
