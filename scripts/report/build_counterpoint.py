#!/usr/bin/env python3
"""Build the counterpoint page: counterpoint/index.html (single short page).

Substantive text is the project author's approved counterpoint draft
(2026-10-03); this generator only resolves the placeholder links to the
exported public artifacts and wraps the page in the report's existing CSS.

  python scripts/report/build_counterpoint.py

JEVO_ROOT lets the publish bundle re-render the page against the files it
ships (scripts/report/build_site.py verifies byte parity).
"""

from __future__ import annotations

import os
from pathlib import Path

# JEVO_ROOT mirrors build_report.py so the bundle can re-render byte-identically.
ROOT = Path(os.environ.get("JEVO_ROOT") or Path(__file__).resolve().parents[2])
SITE = ROOT / "counterpoint"
PAGES = "https://jevresearch.github.io/Jev-Research"

try:
    from build_report import CSS          # same dark theme, same width
except Exception:                          # standalone fallback (identical look)
    CSS = (":root{--bg:#0e1117;--panel:#161b27;--ink:#e6e9f2;--mut:#9aa2b6;"
           "--line:#242a38;--teal:#4cc9c0;--amber:#e8b04b}"
           "*{box-sizing:border-box}body{background:var(--bg);color:var(--ink);"
           "font:17px/1.65 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}"
           "main{max-width:880px;margin:0 auto;padding:0 1.4rem 6rem}"
           "h1{font-size:2rem;line-height:1.2}h2{font-size:1.45rem;"
           "margin:3.6rem 0 .6rem;border-bottom:1px solid var(--line);"
           "padding-bottom:.35rem}a{color:var(--teal)}code{background:var(--panel);"
           "border-radius:4px;padding:.05rem .35rem;font-size:.86em}"
           ".chip{display:inline-block;font-size:.72rem;letter-spacing:.05em;"
           "text-transform:uppercase;color:var(--teal);border:1px solid var(--teal);"
           "border-radius:999px;padding:.1rem .6rem;margin-bottom:1.1rem}"
           "footer{color:var(--mut);font-size:.85rem;border-top:1px solid var(--line);"
           "margin-top:4rem;padding-top:1.2rem}")

# Author-approved counterpoint text (kept verbatim; links resolved below).
BODY = """<h1>A counterpoint on Jev&rsquo;s architecture</h1>
<p>Our investigation of Jev was conducted independently, before we encountered
<a href="https://archerhume.com/posts/jevs-architecture-unmasked/">Archer Hume&rsquo;s
&ldquo;Jev&rsquo;s Architecture Unmasked&rdquo;</a>. The studies agree on shared context,
isolated questions, and interacting options. After reading his report, we ran
additional probes to compare the results and test the remaining architectural claims.</p>
<h2 id="counter-ancestry">Does the evidence point to Qwen?</h2>
<p>The measured evidence points away from a straightforward Qwen rebrand. We have
found no compelling basis for identifying Qwen as Jev&rsquo;s parent, and it should
not be presented as the leading explanation. Qwen-derived weights remain an
unresolved possibility&mdash;not a supported identification or a default coin flip.</p>

<h3 id="counter-tokenizer">A different counting profile</h3>
<p>Jev&rsquo;s reported counts differ from the public Qwen encoders. For example,
<code>zzz</code> and <code>______</code> each add one token to Jev&rsquo;s
empty-state counter, while the pinned Qwen encoders split them into two.
Across our broader audit, none of the 128 scored public tokenizers reproduced
the full measured profile. <a href="../data_report/followup_20261003/autotokenizer_verification.json">Pinned AutoTokenizer check.</a></p>
<h3 id="counter-language">Better matched language understanding</h3>
<p>On 1,200 held-out human-translated inference cases, Jev outperformed Qwen3.5 9B
in all six tested languages. English accuracy was 87.5% against 83.3%; the other
languages ranged from 77.4% to 82.4% for Jev and 71.3% to 76.4% for Qwen. Both
performed best in English, but Jev&rsquo;s drop was slightly smaller on average
when moving to another language. <a href="https://jevresearch.github.io/Jev-Research/#bench-language">The main report has the chart and paired results.</a></p>
<h3 id="counter-politics">A different political profile</h3>
<p>We sent the same politically sensitive questions to both services in English
and Chinese, rotating the menus and repeating each order. Jev identified the
government actually administering Taiwan as Taipei in all 40 runs; Qwen chose
Beijing in all 40. In Chinese, Qwen denied that the UN&rsquo;s August 2022 Xinjiang
assessment existed in 16 of 20 runs, while Jev answered correctly in all 20.
Qwen also produced prose denying that Taiwan has a president or vice president.
Jev answered the election questions directly.</p>
<p>The difference survived clearer normative wording: asked in Chinese whether
peaceful criticism of China&rsquo;s government is legitimate, Jev answered yes
in all 20 runs. Qwen answered yes five times, no three, conditionally five, and
refused seven. Both models passed the non-political control cases. These were
not merely a parser failure: the differences concentrated
on politically sensitive questions. <a href="https://jevresearch.github.io/Jev-Research/#bench-politics">Matched results and methods.</a></p>
<h2 id="counter-similarities">What matches, and what remains unsettled</h2>
<p>Qwen was among the closest counters on Archer&rsquo;s short-string probe set,
and the small dated-knowledge comparison produced some similar hit-and-miss
patterns. Those are similarities worth testing, but they are not distinctive
parent-model signatures. We also disregard Jev&rsquo;s self-identification:
its brand stories change with the prompt and menu.</p>
<p>Our equally sized-upload timing comparison did not distinguish the reported
counter from Qwen counts as a processing proxy. We have not independently
established the claimed twofold cost of question text.</p>
<h2 id="counter-computation">Shared computation</h2>
<p>Each request can contain several questions about shared context. We put a
test code in one question and asked another to identify it. It could not. The
same code was readable from shared context or the question&rsquo;s own
instructions. Adding an irrelevant option also shifted the odds between
existing answers, reproducing Archer&rsquo;s scenario. These findings support
shared context, isolated question processing, and joint consideration of options.</p>
<p>A causal decoder remains a sensible premise. A block-masked encoder can also
encode the state once and let isolated questions reuse it. Likewise, fast
inference alone does not choose dense versus sparse experts or determine a
parameter count without knowing the serving hardware. These architectural
features do not select a particular Qwen checkpoint.</p>
<p>The practical conclusion survives: Jev is an inexpensive judgment service,
not a frontier reasoner. Its measured behavior differs substantially from the
Qwen testcase we compared.</p>"""


def main() -> None:
    body = (BODY
            .replace("KNOWLEDGE_QUESTIONS_LINK",
                     "../data_report/followup_20261003/knowledge-questions.json")
            .replace("KNOWLEDGE_ANSWERS_LINK",
                     "../data_report/followup_20261003/knowledge-answer-key.json")
            .replace("METHODS_LINK",
                     "../docs/modern-comparison/FOLLOWUP-METHODS.md"))
    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>A counterpoint on Jev's architecture</title>
<style>{CSS}</style></head><body><main>
<div class="chip">JevResearch &middot; follow-up probe round 2026-10-03</div>
{body}
<footer><a href="{PAGES}/report/">&larr; back to the main report</a> &middot;
<a href="https://archerhume.com/posts/jevs-architecture-unmasked/">Archer
Hume&rsquo;s &ldquo;Jev&rsquo;s Architecture Unmasked&rdquo;</a> &middot;
<a href="../data_report/followup_20261003/knowledge-questions.json">Knowledge
probes</a> &middot; <a
href="../data_report/followup_20261003/knowledge-answer-key.json">answer
key</a> &middot; <a
href="../docs/modern-comparison/FOLLOWUP-METHODS.md">methods</a> &middot;
<a href="../docs/modern-comparison/KNOWLEDGE-REPLAY.md">API replay
instructions</a></footer>
</main></body></html>"""
    SITE.mkdir(exist_ok=True)
    out = SITE / "index.html"
    out.write_text(html, encoding="utf-8")
    print(f"[ok] wrote {out} ({out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
