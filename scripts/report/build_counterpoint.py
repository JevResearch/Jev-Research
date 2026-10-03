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
<p><a href="https://archerhume.com/posts/jevs-architecture-unmasked/">Archer
Hume&rsquo;s &ldquo;Jev&rsquo;s Architecture Unmasked&rdquo;</a> is worth reading. Our follow-up reproduces its
strongest behavioral findings: Jev shares state, isolates questions, and lets
options influence one another. The disagreement is about how much of the
machinery behind those behaviors we can identify.</p>
<h2>The tokenizer is not a pedigree</h2>
<p>Jev&rsquo;s reported counts really differ from the proposed Qwen tokenizers.
For example, <code>zzz</code> and <code>______</code> each add one token to
Jev&rsquo;s empty-state counter, whereas the pinned Qwen encoders split them
into two. The proposed Qwen3-30B-A3B and Qwen3-Next candidates also share
their ordinary-text vocabulary with Qwen2.5; a missing Qwen3 vocabulary
expansion does not explain the mismatch.</p>
<p>We also tested timing with equally sized uploads but different token
counts. The reported counts and Qwen counts predicted elapsed time similarly.
That experiment did not settle which tokenizer the neural network uses. A
count&ndash;latency relationship is not a weight fingerprint.</p>
<h2>Shared computation does not identify the backbone</h2>
<p>Each request can contain several questions about shared context. We put a
test code in one question and asked another to identify it. It could not. The
same code was readable from the shared context or the question&rsquo;s own
instructions. Adding an irrelevant option also shifted the odds between
existing answers, reproducing Archer&rsquo;s result. These findings support
shared context, isolated question processing, and joint consideration of
options.</p>
<p>A causal decoder remains a sensible premise. However, shared-state caching
is not exclusive to it: a block-masked encoder can encode the state once and
let isolated questions reuse it. Likewise, fast inference does not identify
sparse experts or a parameter count without knowing the serving hardware.
Those are architecture hypotheses, not recovered internals.</p>
<p>Our tests have not independently established the claimed twofold cost of
question text.</p>
<p>The practical conclusion survives: Jev is an inexpensive judgment service,
not a frontier reasoner. We now have a clearer account of its behavior than of
its family tree.</p>"""


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
