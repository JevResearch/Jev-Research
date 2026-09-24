#!/usr/bin/env python3
"""Build the publishable, sanitized research bundle: jev-report/

The layout mirrors the working tree's evidence paths so that
`JEVO_ROOT=jev-report python scripts/report/build_report.py` re-renders the
published page byte-for-byte from published aggregates alone - the strongest
available form of "every number traces to disk".

Contents
  report/index.html        the page (also copied to bundle root for Pages)
  assets/                  standalone chart pages
  src/ scripts/ tests/     the harness that ran everything (MIT)
  runs_benchmark*/         derived aggregates ONLY (per-item lists stripped)
                           + freeze records (frozen.json, preparation.json)
                           + per-stage usage_summary.json roll-ups
  runs_archprobe/          probe JSONs, per-call rows, BILLING
  runs_live/               top-level Talk traces + BILLING + FINDINGS (the
                           dated run dirs are never shipped)
  data_report/             costs.json, billing_totals.json, aborted-stage usage
  docs/                    research write-ups (gate/handoff/executor docs in)
  README.md LICENSE NOTICE.md .gitignore .env.example pyproject + lock

Hard exclusions anywhere in the tree:
  data/ (gated sources + vocab), raw/, items.jsonl, *spec*.json,
  attempts.jsonl, results.jsonl, logs/*.log, *.exit, .venv, __pycache__,
  models/, runs_reviewed/, runs_matched/, the aborted stage dir,
  docs/review-gate/, internal gate/handoff/executor docs, MANIFEST.sha256,
  wire_preview.json, plan.json, executor_state.json, runs_live dated dirs.

Gates (fail-closed; run over the finished bundle):
  G1 structure : no banned path component/file name survives
  G2 secrets   : vendor/OpenAI/GitHub/HF/Slack/private-key patterns;
                 obviously-fake values (deadbeef, zeros, placeholders) pass
  G3 licensed  : fixed windows cut from every gated item prompt
                 (data/*.parquet|csv string columns, every items.jsonl
                 'state', every spec 'state') must appear in NO bundle file;
                 template boilerplate is removed by a frequency filter
  G4 re-render : build_report.py run with JEVO_ROOT=bundle must reproduce
                 report/index.html byte-identically

Usage: python scripts/report/build_site.py
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "jev-report"

# ------------------------------------------------------------------- includes
INCLUDE_FILES = [
    "report/index.html",
    "research.md",
    "DESIGN.md",
    "SOURCES.md",
    "IMPLEMENTATION.md",
    "pyproject.toml",
    "requirements.lock",
    ".env.example",
    "data_report/costs.json",
    "data_report/billing_totals.json",
    "data_report/aborted_stage_usage.json",
    "runs_live/BILLING.json",
    "runs_live/FINDINGS.md",
    "docs/token-talk-findings.md",
    "docs/jev-talk-program-report.md",
    "docs/token-talk-plan.md",
    "tests/conftest.py",
]
INCLUDE_DIRS = [
    "src/jev_observatory",
    "scripts/report",
    "scripts/benchmark",
    "tests",
    "docs/modern-comparison",
    "runs_archprobe",
]
DOCS_ALLOW = {  # only these survive under docs/modern-comparison/
    "ARCHITECTURE-PROBES.md", "COMPARABLE-SCORES.md", "REFERENCE-MATRIX.md",
    "comparison-graphs.html", "pareto-frontiers.html", "architecture-evidence.html",
    "canonical",
}
# Internal synthetic specs the published harness loads at runtime (generator
# output / public catalog snapshot - NOT gate or handoff prose). The G3
# licensed-content gate still scans them like everything else.
INTERNAL_ALLOW = {
    "docs/review-gate/audit/fresh_v3_test_spec.json",
    "docs/review-gate/audit/mech_dryrun_spec.json",
    "docs/review-gate/audit/mech_dryrun_mappings.json",
    "docs/modern-comparison/openrouter-catalog-nine-20260919.json",
}
BENCH_DIRS = ("runs_benchmark", "runs_benchmark_ext", "runs_benchmark_ext2")

BANNED_COMPONENTS = {"raw", "data", "models", "__pycache__", ".venv",
                     ".pytest_cache", "runs_reviewed", "runs_matched",
                     "review-gate", "examples"}
BANNED_FILES = {"items.jsonl", "spec.json", "attempts.jsonl", "results.jsonl",
                "wire_preview.json", "plan.json", "executor_state.json",
                "MANIFEST.sha256"}
BANNED_NAME_RE = re.compile(
    r"(^runs_live_spec_.*\.json$|_spec\.json$|aborted-|aborted1|\.log$|\.exit$"
    r"|GATE\.md$|handoff|parent-|continuation|screen|TALK-AUDIT|\.egg-info$)", re.I)

STRIP_LISTS = ("per_item", "model_returned_values", "missing_logical_ids")

TEXT_SUFFIX = {".html", ".json", ".py", ".md", ".sh", ".toml", ".lock",
               ".example", ".txt", ".js", ".css"}


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def banned_path(r: str) -> bool:
    parts = r.split("/")
    if parts[0].startswith("runs_live/") or (parts[0] == "runs_live" and len(parts) > 1):
        return True                       # dated run dirs never ship; top-level
                                          # files are allow-listed via INCLUDE
    if parts[-1] in BANNED_FILES:
        return True
    if set(parts[:-1]) & BANNED_COMPONENTS or parts[-1] in BANNED_COMPONENTS:
        return True
    if BANNED_NAME_RE.search(parts[-1]):
        return True
    if r.startswith("docs/modern-comparison/"):
        sub = r.split("/", 2)[2] if r.count("/") >= 2 else ""
        return sub.split("/")[0] not in DOCS_ALLOW
    return False


def wanted(r: str) -> bool:
    """Whitelist before blacklist: only evidence trees that belong in the page."""
    tops = ("src/", "scripts/", "tests/", "docs/modern-comparison/", "runs_archprobe/")
    return any(r.startswith(t) for t in tops)


# ------------------------------------------------------------------ assembly
def build() -> list[str]:
    notes: list[str] = []
    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    BUNDLE.mkdir()

    for f in INCLUDE_FILES:
        src = ROOT / f
        if not src.exists():
            raise SystemExit(f"[bundle] missing required file: {f}")
        (BUNDLE / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, BUNDLE / f)

    for d in INCLUDE_DIRS:
        for p in sorted((ROOT / d).rglob("*")):
            if not p.is_file():
                continue
            r = rel(p)
            if not wanted(r) or banned_path(r):
                continue
            dest = BUNDLE / r
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)

    # internal synthetic specs needed by the published test suite at runtime
    for r in sorted(INTERNAL_ALLOW):
        src = ROOT / r
        if not src.exists():
            raise SystemExit(f"[bundle] missing internal allow-list source: {r}")
        dest = BUNDLE / r
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)

    # benchmark runs: derived aggregates + freeze records, per-stage usage
    for bench in BENCH_DIRS:
        for run in sorted((ROOT / bench).glob("bench-*")):
            for name in ("score.json", "weighted_score.json", "usage_summary.json"):
                f = run / "derived" / name
                if not f.exists():
                    continue
                doc = json.loads(f.read_text())
                if name != "usage_summary.json":
                    removed = [k for k in STRIP_LISTS if k in doc]
                    for k in removed:
                        doc.pop(k)
                    doc["publish_note"] = (
                        "per-item lists stripped at publish (item ids/answers "
                        "reference licensed third-party content); aggregates kept")
                out = BUNDLE / rel(f)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(doc, indent=2) + "\n")
        fz = ROOT / bench / "freeze"
        for name in ("frozen.json", "preparation.json"):
            if (fz / name).exists():
                dest = BUNDLE / rel(fz / name)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(fz / name, dest)
        notes.append(bench)

    # runs_live: top-level json traces (talk program) except infra state
    skip_live = {"BILLING.json", "next_run_ids.json"}
    for p in sorted((ROOT / "runs_live").glob("*.json")):
        if p.name in skip_live:
            continue
        shutil.copy2(p, BUNDLE / rel(p))

    return notes


# ------------------------------------------------------------------ gates G1
def structural_scan(bundle: Path) -> list[str]:
    problems = []
    for p in bundle.rglob("*"):
        r = p.relative_to(bundle).as_posix()
        parts = p.relative_to(bundle).parts
        if not p.is_file():
            continue
        if set(parts[:-1]) & BANNED_COMPONENTS and r not in INTERNAL_ALLOW:
            problems.append(f"G1 banned dir: {r}")
        if parts and parts[-1] in BANNED_FILES:
            problems.append(f"G1 banned file: {r}")
        if re.search(r"\.(log|exit)$", r) or "raw" == (parts[-2] if len(parts) > 1 else ""):
            problems.append(f"G1 banned tail: {r}")
        if re.match(r"runs_live/\d{4}-", r):
            problems.append(f"G1 dated run dir: {r}")
    return problems


# ------------------------------------------------------------------ gates G2
FAKE_OK = re.compile(r"(deadbeef|f{16,}|0{20,}|example|placeholder|"
                     r"changeme|your[_-]|<[a-z_]+>|\{ *\w+ *\}|"
                     r"something|dummy|testkey|fake)", re.I)
SECRET_PATTERNS = [
    ("vendor key", re.compile(r"apikey_[0-9a-f]{24,}")),
    ("openai key", re.compile(r"sk-[A-Za-z0-9_-]{18,}")),
    ("github token", re.compile(r"(?:ghp|gho|ghs|github_pat_)[A-Za-z0-9_]{16,}")),
    ("hf token", re.compile(r"hf_[A-Za-z0-9]{24,}")),
    ("slack token", re.compile(r"xox[abprs]-[A-Za-z0-9-]{8,}")),
    ("private key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("bearer literal", re.compile(r"Bearer\s+(?!f\"|\\|\{)[A-Za-z0-9._+/=-]{24,}")),
    ("env assignment", re.compile(
        r"(?:API_KEY|TOKEN|SECRET|PASSWORD)[A-Z_]*\s*[=:]\s*"
        r"['\"](?=[A-Za-z0-9_\-]{14,})[^'\"]*[0-9][^'\"]*['\"]")),
]


def bundle_text(bundle: Path):
    for p in sorted(bundle.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix not in TEXT_SUFFIX and p.name not in {".gitignore", ".env.example"}:
            continue
        yield p, p.read_text(encoding="utf-8", errors="ignore")


def secret_scan(bundle: Path) -> list[str]:
    hits = []
    for p, text in bundle_text(bundle):
        for label, rx in SECRET_PATTERNS:
            for m in rx.finditer(text):
                val = m.group(0)
                if FAKE_OK.search(val):
                    continue
                hits.append(f"G2 {label} in {p.relative_to(bundle)}: {val[:28]}...")
    return hits


# ------------------------------------------------------------------ gate G3
SHINGLE_WORDS = 10      # exact phrase length checked against the bundle
BOILER_FRAC = 0.02      # a shingle in >2% of gated items is our own template
BUNDLE_FILE_SPREAD = 3  # a shingle in >=3 bundle files is our own repeated text

# gated sources: real dataset files + the benchmark runs' outbound prompts.
# Our synthetic probe items/specs (runs_live dated dirs, probes, fresh_capability
# generators) are OUR authored text and ship verbatim - not gated.
GATED_GLOBS = ("runs_benchmark*/bench-*/items.jsonl",
               "runs_benchmark*/bench-*/spec.json",
               "boolq_spec.json", "boolq_paired_spec.json", "mmlu_pilot_spec.json")
# Whitelist: our own authored prose (synthetic probe corpora, research docs).
# A phrase only counts as a leak if it is NOT already in our public documents -
# probe questions were authored for this project, not copied from datasets.
KNOWN_GLOBS = ("runs_live/*.json", "runs_live/*.md", "runs_live_spec_*.json",
               "runs_live/*/items.jsonl", "docs/**/*.md", "README.md",
               "research.md", "SOURCES.md", "IMPLEMENTATION.md", "DESIGN.md",
               "fresh_capability_spec.json", "runs_reviewed/*/items.jsonl",
               "runs_matched/**/items.jsonl")


def _shingles(text: str, out: dict) -> None:
    toks = re.sub(r"\s+", " ", text).strip().split(" ")
    if len(toks) < SHINGLE_WORDS:
        return
    for i in range(len(toks) - SHINGLE_WORDS + 1):
        sh = " ".join(toks[i:i + SHINGLE_WORDS])
        out[sh] = out.get(sh, 0) + 1


def _add_strings(obj, out: dict) -> None:
    """Walk a decoded json object; feed every prose string to _shingles."""
    stack = [obj]
    while stack:
        x = stack.pop()
        if isinstance(x, str):
            if len(x) > 40:
                _shingles(x, out)
        elif isinstance(x, dict):
            stack.extend(x.values())
        elif isinstance(x, list):
            stack.extend(x[:4000])


def _read_parquet_strings(f: Path, out: dict, cap_rows: int = 3000) -> None:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(f)
    done = 0
    for batch in pf.iter_batches(batch_size=250):
        for col in batch.columns:
            if col.type.__class__.__name__ not in ("string", "large_string"):
                continue
            for v in col.to_pylist():
                if isinstance(v, str):
                    _shingles(v, out)
        done += len(batch)
        if done >= cap_rows:
            return


def _add_outbound_strings(spec: dict, out: dict) -> None:
    """Feed only the outbound prompt payload of a spec (item states + question
    criteria) - spec metadata is our own prose and must not gate our code."""
    for item in spec.get("items", []):
        if isinstance(item, dict):
            st = item.get("state")
            if isinstance(st, str):
                _shingles(st, out)
            for q in (item.get("questions") or {}).values():
                if isinstance(q, dict):
                    for v in q.values():
                        if isinstance(v, str):
                            _shingles(v, out)


def known_shingles() -> set[str]:
    out: dict[str, int] = {}
    for pattern in KNOWN_GLOBS:
        for f in sorted(ROOT.glob(pattern)):
            try:
                if f.suffix == ".json":
                    _add_strings(json.loads(f.read_text(encoding="utf-8")), out)
                else:
                    _shingles(f.read_text(encoding="utf-8", errors="ignore"), out)
            except Exception:
                continue
    print(f"[G3] whitelist shingles from our authored corpus: {len(out):,}")
    return set(out)


def gated_shingles() -> set[str]:
    """Boilerplate-filtered shingle set of every licensed item we ingested.

    Filtering is PER SOURCE FILE: our fixed instruction/rubric text repeats
    across every item of one source and is dropped there; real item content
    appears in exactly one item and survives.
    """
    keep: set[str] = set()

    def flush(local: dict, label: str) -> None:
        n = max(1, sum(local.values()) // 1)  # not item count; use entries:
        keep_local = {sh for sh, c in local.items() if c <= thresh}
        keep.update(keep_local)

    def thresh_for(items: int) -> int:
        return max(2, int(BOILER_FRAC * max(1, items)))

    total = 0
    for pattern in GATED_GLOBS:
        for f in sorted(ROOT.glob(pattern)):
            local: dict[str, int] = {}
            units = 0
            if f.name.endswith("items.jsonl") or (f.suffix == ".jsonl"):
                for line in f.open(encoding="utf-8", errors="ignore"):
                    try:
                        _add_outbound_strings(json.loads(line), local)
                    except json.JSONDecodeError:
                        continue
                    units += 1
            else:
                try:
                    _add_outbound_strings(json.loads(f.read_text(encoding="utf-8")), local)
                    units = len(json.loads(f.read_text(encoding="utf-8")).get("items", [])) or 1
                except Exception as e:
                    print(f"  [G3] skip {f.name}: {e}")
            th = thresh_for(units)
            kept = {sh for sh, c in local.items() if c <= th}
            keep |= kept
            total += units

    for f in sorted((ROOT / "data").iterdir()):
        name = f.name.lower()
        local = {}
        try:
            if name.endswith(".parquet"):
                _read_parquet_strings(f, local)
            elif name.endswith((".csv", ".jsonl", ".txt")):
                _add_strings(f.read_text(encoding="utf-8", errors="ignore"), local)
            else:
                continue
            units = max(1, len(local) // 10)
            keep |= {sh for sh, c in local.items() if c <= max(2, len(local) // 12)}
            total += units
        except Exception as e:
            print(f"  [G3] data skip {f.name}: {e}")

    print(f"[G3] {total:,} gated items -> {len(keep):,} content shingles "
          f"(per-source boilerplate filter)")
    return keep


def bundle_shingles(bundle: Path) -> tuple[dict, dict]:
    """(shingle -> set(files), file count) over the bundle's text corpus."""
    spread: dict[str, set] = {}
    for p, text in bundle_text(bundle):
        name = str(p.relative_to(bundle))
        toks = re.sub(r"\s+", " ", text).strip().split(" ")
        for i in range(len(toks) - SHINGLE_WORDS + 1):
            sh = " ".join(toks[i:i + SHINGLE_WORDS])
            spread.setdefault(sh, set()).add(name)
    return spread, {}


def licensed_scan(bundle: Path) -> list[str]:
    gated = gated_shingles() - known_shingles()
    spread, _ = bundle_shingles(bundle)
    hits = []
    for sh, files in spread.items():
        if sh in gated and len(files) < BUNDLE_FILE_SPREAD:
            hits.append(f"G3 licensed shingle in {sorted(files)[0]}: {sh[:70]}...")
    return sorted(hits)[:200]


# ------------------------------------------------------------------ gate G4
def verify_render(bundle: Path) -> str | None:
    target = bundle / "report" / "index.html"
    original = target.read_bytes()
    env = dict(os.environ, JEVO_ROOT=str(bundle))
    out = subprocess.run(
        [sys.executable, str(bundle / "scripts/report/build_report.py")],
        capture_output=True, text=True, env=env, cwd=str(bundle))
    if out.returncode != 0:
        return f"builder failed in bundle:\n{out.stdout}\n{out.stderr}"
    if target.read_bytes() != original:
        return "re-render from published aggregates differs from shipped page"
    return None


# ------------------------------------------------------------- scaffolding
def write_scaffolding() -> None:
    (BUNDLE / "README.md").write_text(README_TEXT, encoding="utf-8")
    (BUNDLE / "LICENSE").write_text(LICENSE_TEXT, encoding="utf-8")
    (BUNDLE / "NOTICE.md").write_text(NOTICE_TEXT, encoding="utf-8")
    (BUNDLE / ".gitignore").write_text(
        "__pycache__/\n*.pyc\n.env\ndata/\nmodels/\n.venv/\n.pytest_cache/\n",
        encoding="utf-8")
    (BUNDLE / ".nojekyll").write_text("", encoding="utf-8")
    (BUNDLE / "assets").mkdir(exist_ok=True)
    for name in ("comparison-graphs.html", "pareto-frontiers.html",
                 "architecture-evidence.html"):
        src = BUNDLE / "docs/modern-comparison" / name
        if src.exists():
            shutil.copy2(src, BUNDLE / "assets" / name)
    shutil.copy2(BUNDLE / "report" / "index.html", BUNDLE / "index.html")


def manifest() -> int:
    rows = []
    for p in sorted(BUNDLE.rglob("*")):
        if p.is_file() and p.name != "BUNDLE.sha256":
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            rows.append(f"{h}  {p.relative_to(BUNDLE).as_posix()}")
    (BUNDLE / "BUNDLE.sha256").write_text("\n".join(rows) + "\n")
    return len(rows)


# --------------------------------------------------------------- bundle prose
README_TEXT = """# Jev: Not Frontier, But Still Worth Your Attention

An independent, hands-on evaluation of TypeSafe AI's `jev-1.13.0` API:
16,379 live benchmark requests across three frozen suites, a 987-call
architecture probe, a multi-protocol Talk-to-Jev program, and a
token-counting study - 23,459 recorded calls costing $3.58 in total, every
figure re-derivable from the published aggregates in this repository.

**The report page: [`index.html`](index.html)** (also at `report/index.html`).
Headline: MMLU-Pro 82.8%, GPQA Diamond 76.5%, ~73 ms fixed + ~6 ms/1k-token
server compute, ~$0.28 per 12k-question MMLU-Pro run - a small, new,
English-centric model with a probability read-out in place of a generation
head, not a frontier system.

## What's here

| path | contents |
|---|---|
| `index.html`, `report/` | the report (single page, figures embedded) |
| `assets/` | standalone chart pages (same figures, un-embedded) |
| `src/`, `scripts/`, `tests/` | the harness that ran everything (MIT) |
| `runs_benchmark*/` | per-stage derived aggregates (scores, weighted scores, calibration, usage) + freeze manifests with SHA-256 of every input |
| `runs_archprobe/` | architecture-probe analysis, per-call rows, tokenizer studies, billing |
| `runs_live/` | Talk-to-Jev traces (character / vocabulary-menu / token programs), probes, billing, findings |
| `data_report/` | cost model, billing roll-up, aborted-stage usage |
| `docs/` | research write-ups: architecture probes, comparable scores, the Talk program |

## Reproducing the page from the data

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python scripts/report/aggregate_billing.py   # -> data_report/billing_totals.json
.venv/bin/python scripts/report/build_report.py        # -> report/index.html (byte-identical to shipped)
.venv/bin/pytest                                       # saved-run regressions skip in this checkout
```

The page embeds no hardcoded results: `build_report.py` reads every number
from the JSON aggregates above at render time, and the CI gate in
`scripts/report/build_site.py` verified this bundle re-renders byte-for-byte
from published data alone. Re-running the *live* experiments requires your own
`TYPESAFE_API_KEY` (see `.env.example`) and hits the paid API; the freeze
manifests pin exactly what we ran.

## Data availability and licensing

Benchmark **item text is third-party licensed content** (MMLU-Pro is
CC-BY-NC-4.0; GPQA, HLE and ARC-AGI-2 are under their own terms, some gated)
and is **not republished here**. This repository publishes instead:

* derived aggregates per stage (accuracy, Wilson CIs, Brier / log-loss,
  weighted scores, usage roll-ups) with per-item lists stripped;
* freeze records (`frozen.json`, `preparation.json`) carrying SHA-256 hashes
  of every source file and request plan - holders of the datasets can verify
  item identity against those hashes;
* our own synthetic prompts verbatim (the Talk traces, the architecture-probe
  rows, the tokenizer samples - none is third-party licensed);
* the complete cost model and billing roll-up.

External comparison scores were fetched from public publisher pages; sources
and access dates are recorded in `docs/modern-comparison/canonical/` and
`docs/modern-comparison/COMPARABLE-SCORES.md`.

## License

Code (`src/`, `scripts/`, `tests/`) is MIT-licensed; report prose and derived
data tables are CC-BY-4.0 (see `LICENSE`). Jev is a trademark of TypeSafe AI;
this project is independent and unaffiliated - see `NOTICE.md`.
"""

NOTICE_TEXT = """# Notice

This is an **independent, unaffiliated** evaluation. We have no relationship
with TypeSafe AI; "Jev" and "TypeSafe" are trademarks of their owner; the
model string measured is the service-reported `jev-1.13.0`.

## Protocol caveats that travel with every number here

* **Direct answering only.** Jev answered one-shot: no chain of thought, no
  tools, no retries, failures counted. Published frontier comparison numbers
  generally use reasoning-enabled, few-shot protocols. Comparisons are
  positioning, not matched races.
* **Pinned version, no changelog.** All calls used the service-reported
  `jev-1.13.0` during September 2026. The vendor publishes no per-version
  behavioral notes; another day's `jev-1.13.0` could differ.
* **0.01 quantization.** Every returned probability landed on a 0.01 grid;
  sub-quantum differences are noise and are never interpreted.
* **Behavioral evidence only.** No weights, gradients, or serving internals
  were accessed or reconstructed. Architecture statements are inferences from
  API-visible signals: answers, probability vectors, token counts, timing
  headers, billing usage.
* **Latency slopes are contaminated by multi-tenant batching** and are not
  convertible to parameter counts.
* **Self-reports are learned text.** The model's identity answers are a brand
  prior, not provenance; the tokenizer evidence contradicts the self-report.
* **MATH-500 / ARC-AGI-2 results are protocol conversions** (multiple-choice,
  per-cell re-encodings) measuring knowledge, not generation.
* **Talk-to-Jev outputs are collaborative**: token-level generation is steered
  by a local scorer model, and option position demonstrably moves Jev's
  choices; character- and vocabulary-menu results are the model's own and
  degenerate. Neither is "the model writing".

Raw per-request logs containing third-party licensed prompt text are held
privately; the freeze hashes in `runs_benchmark*/freeze/` pin them exactly.
"""

LICENSE_TEXT = """MIT License (code)

Copyright (c) 2026 Jev Observatory contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

Report text, figures, and derived data tables (everything outside src/,
scripts/, and tests/) are licensed under Creative Commons Attribution 4.0
International (CC-BY-4.0):
https://creativecommons.org/licenses/by/4.0/legalcode

Third-party benchmark datasets referenced by these measurements remain under
their own licenses; no item text is redistributed here.
"""


# ------------------------------------------------------------------ pipeline
def main() -> None:
    for need in ("data_report/costs.json", "data_report/billing_totals.json",
                 "data_report/aborted_stage_usage.json",
                 "runs_benchmark/freeze/frozen.json"):
        if not (ROOT / need).exists():
            raise SystemExit(f"[bundle] run prerequisite first: {need} missing")
    build()
    write_scaffolding()

    problems: list[str] = []
    problems += structural_scan(BUNDLE)
    problems += secret_scan(BUNDLE)
    problems += licensed_scan(BUNDLE)
    err = verify_render(BUNDLE)
    if err:
        problems.append("G4 " + err)
    n = manifest()
    print(f"[bundle] {n} files")
    if problems:
        print(f"[bundle] FAILED: {len(problems)} problem(s)")
        for x in problems[:60]:
            print("   ", x)
        sys.exit(1)
    print("[bundle] all gates green: structure, secrets, licensed content, re-render")


if __name__ == "__main__":
    main()
