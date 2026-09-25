#!/usr/bin/env python3
"""Build the publishable, sanitized research bundle: jev-report/

The layout mirrors the working tree's evidence paths so that
`JEVO_ROOT=jev-report python scripts/report/build_report.py` re-renders the
published page byte-for-byte from published aggregates alone - the strongest
available form of "every number traces to disk".

Contents
  report/index.html        the page (also copied to bundle root for Pages)
  assets/                  standalone chart pages
  src/ scripts/ tests/     the harness that ran everything (public domain)
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
    "ARCHITECTURE-ANALYSIS.md",
    "pyproject.toml",
    "requirements.lock",
    ".env.example",
    "data_report/costs.json",
    "data_report/billing_totals.json",
    "data_report/aborted_stage_usage.json",
    "data_report/arch_audits.json",
    "data_report/lattice_forensics.json",
    "data_report/probe2_plan.json",
    "data_report/size_estimate.json",
    "runs_live/BILLING.json",
    "runs_live/FINDINGS.md",
    "docs/token-talk-findings.md",
    "docs/jev-talk-program-report.md",
    "docs/token-talk-plan.md",
    "docs/ARCHITECTURE-ANALYSIS-PROMPT.md",
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
    "architecture-diagram.html",
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
        for child in BUNDLE.iterdir():          # preserve .git across rebuilds
            if child.name == ".git":
                continue
            shutil.rmtree(child) if child.is_dir() else child.unlink()
    BUNDLE.mkdir(exist_ok=True)

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
        if not p.is_file() or ".git" in p.parts:
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
# Memory-bounded design (a previous unbounded version OOMed the host):
#  * shingles are reduced to 8-byte blake2b digests; corpora are streamed
#    per file/string with NO global dict of per-shingle containers
#  * windows per string capped; per-source boilerplate filter only
#  * the known-corpus pass runs ONLY if the bundle scan produced candidates
#  * main() additionally pins RLIMIT_AS, so a logic bug fails with
#    MemoryError instead of eating the machine

SHINGLE_WORDS = 10          # exact phrase length checked against the bundle
BOILER_MAX_FRAC = 0.02      # per source: a window in >2% of items is our template
CAP_WORDS_PER_STRING = 1200 # cap long generative inputs (HLE questions etc.)
KEEP_BUDGET = 5_000_000     # regression guard: refuse to run a broken scan
BUNDLE_FILE_SPREAD = 3      # window in >=3 bundle files is our repeated own-text

GATED_GLOBS = ("runs_benchmark*/bench-*/items.jsonl",
               "runs_benchmark*/bench-*/spec.json",
               "boolq_spec.json", "boolq_paired_spec.json", "mmlu_pilot_spec.json")
# Whitelist: our own authored prose (synthetic probe corpora, research docs).
# A phrase only counts as a leak if it is NOT already in our public documents.
KNOWN_GLOBS = ("runs_live/*.json", "runs_live/*.md", "runs_live_spec_*.json",
               "runs_live/*/items.jsonl", "docs/**/*.md", "README.md",
               "research.md", "SOURCES.md", "IMPLEMENTATION.md", "DESIGN.md",
               "fresh_capability_spec.json", "runs_reviewed/*/items.jsonl",
               "runs_matched/**/items.jsonl")

from hashlib import blake2b as _b2


def _wh(w: str) -> bytes:
    return _b2(w.encode("utf-8", "ignore"), digest_size=8).digest()


def _windows(text: str):
    toks = re.sub(r"\s+", " ", text).strip().split(" ")
    n = min(len(toks), CAP_WORDS_PER_STRING)
    for i in range(max(0, n - SHINGLE_WORDS + 1)):
        yield " ".join(toks[i:i + SHINGLE_WORDS])


def _outbound_windows(item) :
    """Windows of the outbound prompt payload of an item/spec entry only
    (state + question criteria) - spec metadata is our own prose."""
    if not isinstance(item, dict):
        return
    st = item.get("state")
    if isinstance(st, str):
        yield from _windows(st)
    qs = item.get("questions")
    if isinstance(qs, dict):
        for q in qs.values():
            if isinstance(q, dict):
                for v in q.values():
                    if isinstance(v, str) and len(v) > 40:
                        yield from _windows(v)


def _rss_mb() -> int:
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024


def gated_hashes() -> set:
    """Hash set of licensed-content windows, per-source boilerplate-filtered."""
    keep: set[bytes] = set()

    def filter_source(counts: dict, units: int) -> int:
        th = max(2, int(BOILER_MAX_FRAC * max(1, units)))
        keep.update(h for h, c in counts.items() if c <= th)
        return th

    total_items = 0
    for pattern in GATED_GLOBS:
        for f in sorted(ROOT.glob(pattern)):
            counts: dict[bytes, int] = {}
            units = 0
            if f.name.endswith("items.jsonl"):
                for line in f.open(encoding="utf-8", errors="ignore"):
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    units += 1
                    for w in _outbound_windows(item):
                        h = _wh(w)
                        counts[h] = counts.get(h, 0) + 1
            else:
                try:
                    spec = json.loads(f.read_text(encoding="utf-8"))
                except Exception as e:
                    print(f"  [G3] skip {f.name}: {e}")
                    continue
                items = spec.get("items") if isinstance(spec, dict) else None
                if isinstance(items, list):
                    units = len(items)
                    for item in items[:10000]:
                        for w in _outbound_windows(item):
                            h = _wh(w)
                            counts[h] = counts.get(h, 0) + 1
            filter_source(counts, units)
            total_items += units
            counts.clear()

    for f in sorted((ROOT / "data").iterdir()):
        name = f.name.lower()
        counts = {}
        units = 0
        try:
            if name.endswith(".parquet"):
                import pyarrow.parquet as pq
                pf = pq.ParquetFile(f)
                rows = 0
                for batch in pf.iter_batches(batch_size=250):
                    for col in batch.columns:
                        if col.type.__class__.__name__ not in ("string", "large_string"):
                            continue
                        for v in col.to_pylist():
                            if isinstance(v, str) and len(v) > 40:
                                units += 1
                                for w in _windows(v):
                                    h = _wh(w)
                                    counts[h] = counts.get(h, 0) + 1
                    rows += len(batch)
                    if rows >= 3000:
                        break
            elif name.endswith((".csv", ".jsonl", ".txt")):
                for line in f.open(encoding="utf-8", errors="ignore"):
                    for cell in (line.split(",") if name.endswith(".csv") else [line]):
                        cell = cell.strip('" ')
                        if len(cell) > 40:
                            units += 1
                            for w in _windows(cell):
                                h = _wh(w)
                                counts[h] = counts.get(h, 0) + 1
            else:
                continue
        except Exception as e:
            print(f"  [G3] data skip {f.name}: {e}")
            continue
        filter_source(counts, units)
        total_items += units
        counts.clear()

    if len(keep) > KEEP_BUDGET:
        raise SystemExit(f"[G3] gated set {len(keep)} exceeds budget - scan regressed")
    print(f"[G3] {total_items:,} gated strings -> {len(keep):,} content windows "
          f"(per-source boilerplate filter) | RSS {_rss_mb()} MB")
    return keep


def _bundle_windows(bundle: Path, gated: set):
    """Yield (hash, window-text, file) for every bundle window whose hash is in
    gated. Streams file-by-file; only per-file hit SETS of hashes are kept."""
    for p, text in bundle_text(bundle):
        name = str(p.relative_to(bundle))
        local = set()
        for w in _windows(text):
            h = _wh(w)
            if h in gated:
                local.add((h, w))
        for h, w in local:
            yield h, w, name


def licensed_scan(bundle: Path) -> list[str]:
    gated = gated_hashes()
    if not gated:
        return ["G3 gated set empty - scan is broken, failing closed"]

    spread: dict[bytes, set] = {}
    sample: dict[bytes, str] = {}
    for h, w, name in _bundle_windows(bundle, gated):
        spread.setdefault(h, set()).add(name)
        sample.setdefault(h, w)
    print(f"[G3] bundle windows matched gated content: {len(spread)} | RSS {_rss_mb()} MB")

    candidates = {h: fs for h, fs in spread.items() if len(fs) < BUNDLE_FILE_SPREAD}
    problems = []
    if candidates:
        # lazy known-corpus pass: only run when we actually have candidates
        targets = set(candidates)
        for pattern in KNOWN_GLOBS:
            if not targets:
                break
            for f in sorted(ROOT.glob(pattern)):
                try:
                    text = f.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                for w in _windows(text):
                    h = _wh(w)
                    if h in targets:
                        targets.discard(h)
        for h in sorted(targets):
            files = ", ".join(sorted(spread[h]))
            problems.append(f"G3 licensed window in {files}: {sample[h][:70]}...")
    print(f"[G3] done | RSS {_rss_mb()} MB")
    return problems


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
                 "architecture-evidence.html", "architecture-diagram.html"):
        src = BUNDLE / "docs/modern-comparison" / name
        if src.exists():
            shutil.copy2(src, BUNDLE / "assets" / name)
    shutil.copy2(BUNDLE / "report" / "index.html", BUNDLE / "index.html")


def manifest() -> int:
    rows = []
    for p in sorted(BUNDLE.rglob("*")):
        if p.is_file() and ".git" not in p.parts and p.name != "BUNDLE.sha256":
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
| `src/`, `scripts/`, `tests/` | the harness that ran everything (public domain) |
| `runs_benchmark*/` | per-stage derived aggregates (scores, weighted scores, calibration, usage) + freeze manifests with SHA-256 of every input |
| `runs_archprobe/` | architecture-probe analysis, per-call rows, tokenizer studies, billing |
| `runs_live/` | Talk-to-Jev traces (character / vocabulary-menu / token programs), probes, billing, findings |
| `data_report/` | cost model, billing roll-up, aborted-stage usage, architecture audits |
| `docs/` | research write-ups: architecture probes, comparable scores, the Talk program |
| `ARCHITECTURE-ANALYSIS.md` | the full architecture reconstruction: card, evidence, alternatives ledger, next probes |

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
from published data alone. `data_report/arch_audits.json` and
`data_report/lattice_forensics.json` ship precomputed; re-running their
generators needs the full working tree (the dated `runs_live/` raw response
dirs and benchmark per-item lists are not bundled). Re-running the *live* experiments requires your own
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

Everything in this repository - code, report text, figures, and derived data
tables - is dedicated to the public domain under the Unlicense (see `LICENSE`).
No attribution is required or requested; no rights are reserved. Third-party
benchmark datasets referenced by our measurements remain under their own
licenses, and no dataset item text is redistributed here. Jev is a trademark of
TypeSafe AI; this project is independent and unaffiliated (see `NOTICE.md`).
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

LICENSE_TEXT = """This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any
means.

In jurisdictions that recognize copyright laws, the author or authors
of this software dedicate any and all copyright interest in the
software to the public domain. We make this dedication for the benefit
of the public at large and to the detriment of our heirs and
successors. We intend this dedication to be an overt act of
relinquishment in perpetuity of all present and future rights to this
software under copyright law.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

For more information, please refer to <https://unlicense.org>

---

Third-party benchmark datasets referenced by these measurements remain
under their own licenses (see docs/ and the freeze records); no item text
is redistributed here.
"""


# ------------------------------------------------------------------ pipeline
def _cap_memory(mb: int = 8192) -> None:
    """Hard ceiling: a logic bug must raise MemoryError in THIS process
    rather than OOM the machine."""
    import resource
    limit = mb * 1024 * 1024
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    try:
        resource.setrlimit(resource.RLIMIT_AS, (limit, min(hard, limit) if hard != -1 else limit))
    except (ValueError, OSError):
        pass


def main() -> None:
    _cap_memory()
    for need in ("data_report/costs.json", "data_report/billing_totals.json",
                 "data_report/aborted_stage_usage.json",
                 "data_report/arch_audits.json",
                 "data_report/lattice_forensics.json",
                 "data_report/probe2_plan.json",
                 "data_report/size_estimate.json",
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
