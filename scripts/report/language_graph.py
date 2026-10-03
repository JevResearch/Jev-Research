"""Deterministic paired-dot chart for the multilingual held-out aggregate."""
from pathlib import Path
import json
try:
    from .chartkit import BG, JEV_G
except ImportError:
    from chartkit import BG, JEV_G

QWEN = "#9b8cff"
ORDER = {"en":"English", "ar":"Arabic", "zh":"Chinese", "de":"German", "ru":"Russian", "es":"Spanish"}

def render(root: Path) -> str:
    data=json.loads((root/'data_report/expanded_20261003/multilingual_heldout_aggregates.json').read_text())
    rows={r['language']:r for r in data['results']}
    W,H=1280,650; left,right,top,bottom=190,90,70,90
    x0,x1=left,W-right; y0=top
    def x(v): return x0+(x1-x0)*v/100
    out=[f'<svg id="language-comparison-chart" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" aria-labelledby="language-title language-desc"><title id="language-title">Held-out multilingual accuracy</title><desc id="language-desc">Paired Jev and Qwen3.5 9B accuracy dots connected within each language. Languages are not connected to one another.</desc>', f'<rect width="{W}" height="{H}" fill="{BG}"/>', '<style>#language-comparison-chart text{font-family:system-ui,sans-serif;font-size:20px;fill:#e6e9f2}#language-comparison-chart .grid{stroke:#384052}#language-comparison-chart .line{stroke:#687086;stroke-width:3}#language-comparison-chart .dot{stroke:none}#language-comparison-chart .axis{fill:#9aa2b6;font-size:16px}#language-comparison-chart .score{font-size:17px;font-weight:600}</style>']
    for t in range(0,101,20): out.append(f'<line class="grid" x1="{x(t):.1f}" y1="{y0-15}" x2="{x(t):.1f}" y2="{H-bottom}"/><text class="axis" x="{x(t):.1f}" y="{H-bottom+30}" text-anchor="middle">{t}%</text>')
    for i,code in enumerate(("en","ar","zh","de","ru","es")):
        r=rows[code]; y=y0+i*78; j=r['jev_accuracy']*100; q=r['qwen_accuracy']*100
        out += [f'<text x="{left-25}" y="{y+6}" text-anchor="end">{ORDER[code]}</text>',f'<line class="line" x1="{x(q):.1f}" y1="{y}" x2="{x(j):.1f}" y2="{y}"/>',f'<g tabindex="0" aria-label="{ORDER[code]}: Jev {j:.2f} percent, Qwen {q:.2f} percent"><circle class="dot" fill="{QWEN}" cx="{x(q):.1f}" cy="{y}" r="8"/><circle class="dot" fill="{JEV_G}" cx="{x(j):.1f}" cy="{y}" r="8"/><text class="score" x="{x(j)+14:.1f}" y="{y+6}">{j:.2f}%</text><text class="score" x="{x(q)-14:.1f}" y="{y+6}" text-anchor="end">{q:.2f}%</text></g>']
    out += [f'<circle fill="{JEV_G}" cx="{left+20}" cy="{H-32}" r="8"/><text x="{left+38}" y="{H-26}">Jev</text><circle fill="{QWEN}" cx="{left+120}" cy="{H-32}" r="8"/><text x="{left+138}" y="{H-26}">Qwen3.5 9B (reasoning off)</text></svg>']
    return ''.join(out)

def write(root: Path) -> Path:
    p=root/'assets/language_comparison.svg'; p.parent.mkdir(exist_ok=True); p.write_text(render(root),encoding='utf-8'); return p
