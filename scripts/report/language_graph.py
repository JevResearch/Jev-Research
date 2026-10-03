"""Deterministic grouped-bar chart for the multilingual held-out aggregate."""
from pathlib import Path
import json
try:
    from .chartkit import BG, JEV_G
except ImportError:
    from chartkit import BG, JEV_G
QWEN="#9b8cff"; ORDER={"en":"English","ar":"Arabic","zh":"Chinese","de":"German","ru":"Russian","es":"Spanish"}
def render(root: Path)->str:
    d=json.loads((root/'data_report/expanded_20261003/multilingual_heldout_aggregates.json').read_text())
    rows={r['language']:r for r in d['results']}; W,H=1280,720; L,R,T,B=100,70,70,115; pw,ph=W-L-R,H-T-B
    def y(v): return T+ph*(1-v/100)
    out=[f'<svg id="language-comparison-chart" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" aria-labelledby="language-title language-desc"><title id="language-title">Held-out multilingual accuracy by language</title><desc id="language-desc">Grouped Jev and Qwen3.5 9B accuracy bars for six languages; each bar is one accuracy score.</desc>',f'<rect width="{W}" height="{H}" fill="{BG}"/><style>#language-comparison-chart text{{font-family:system-ui,sans-serif;fill:#e6e9f2;font-size:20px}}#language-comparison-chart .grid{{stroke:#384052}}#language-comparison-chart .axis{{fill:#9aa2b6;font-size:17px}}#language-comparison-chart .bar{{stroke:none}}</style>']
    for t in range(0,101,20): out += [f'<line class="grid" x1="{L}" y1="{y(t):.1f}" x2="{W-R}" y2="{y(t):.1f}"/><text class="axis" x="{L-14}" y="{y(t)+6:.1f}" text-anchor="end">{t}%</text>']
    group=pw/6; bw=60; gap=18
    for i,code in enumerate(("en","ar","zh","de","ru","es")):
        r=rows[code]; cx=L+group*(i+.5)
        for j,(key,color,name) in enumerate((("jev_accuracy",JEV_G,"Jev"),("qwen_accuracy",QWEN,"Qwen3.5 9B"))):
            val=r[key]*100; xx=cx-(2*bw+gap)/2+j*(bw+gap); yy=y(val); hh=T+ph-yy
            out += [f'<rect class="bar" data-language="{ORDER[code]}" data-model="{name}" x="{xx:.1f}" y="{yy:.1f}" width="{bw:.1f}" height="{hh:.1f}" fill="{color}" role="img" aria-label="{ORDER[code]} {name} accuracy {val:.2f} percent"/>',f'<text x="{xx+bw/2:.1f}" y="{yy-10:.1f}" text-anchor="middle">{val:.2f}%</text>']
        out.append(f'<text x="{cx:.1f}" y="{H-B+38}" text-anchor="middle">{ORDER[code]}</text>')
    out += [f'<text class="axis" x="20" y="{T+ph/2}" transform="rotate(-90 20 {T+ph/2})" text-anchor="middle">Accuracy (%)</text>',f'<rect x="{W/2-180}" y="{H-52}" width="14" height="14" fill="{JEV_G}"/><text x="{W/2-158}" y="{H-39}">Jev</text><rect x="{W/2+10}" y="{H-52}" width="14" height="14" fill="{QWEN}"/><text x="{W/2+32}" y="{H-39}">Qwen3.5 9B (reasoning off)</text></svg>']
    return ''.join(out)
def write(root:Path)->Path:
    p=root/'assets/language_comparison.svg'; p.parent.mkdir(exist_ok=True); p.write_text(render(root),encoding='utf-8'); return p
