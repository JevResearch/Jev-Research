#!/usr/bin/env python3
"""Follow-up battery CLI (followup-20261003.v1): preregistration, live
dispatch and analysis for the Jev ancestry/architecture follow-up.

Stages (offline stages need no credentials and touch only public metadata):

  plan          call counts + worst-case budget fit check (offline)
  fetch-sources verify every horizon gold against linked source content and
                freeze excerpt + sha256 per item (runs_archprobe/.../sources.json)
  offline       pin HF tokenizer commits via api metadata and confirm the
                already-downloaded file SHAs; decode/encode membership for the
                fresh boundary strings (actual tokenizers, never literal vocab
                key lookup); pinned confidence-formula vectors; versioned P5
                gold correction over the old probe2 rows
  freeze        write the frozen prereg plan + provenance BEFORE any live call;
                refuses to freeze if any real-event gold is unverified
  live-jev      dispatch the four Jev families through the pinned
                jev-1.13.0 endpoint (parent-approved live battery)
  live-ref      dispatch the frozen horizon items to the two Qwen references
                and the non-Qwen control (one attempt per dispatch)
  analyze       analysis.json with contrasts, held-out latency fits and counts

Live credential handling: the owner-0600 credentials file is read inside this
process and injected into os.environ for the transports; no key value is ever
printed, logged or written to any artifact (redact.Redactor wraps captures).
Every logical call pre-reserves its worst-case cost in a persistent ledger
(PersistentBudget) before dispatch; unknown-cost outcomes hold their
reservation forever; price/quota/auth failures stop the run (fail closed).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
import urllib.request
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT / "src", _ROOT / "scripts" / "benchmark"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import followup_battery as fb  # noqa: E402
from jev_observatory import arch_probe as ap  # noqa: E402
from jev_observatory.answer_recovery import recover_choice, recovery_spec  # noqa: E402
from jev_observatory.openrouter import (CHAT_PATH, WireConfig,  # noqa: E402
                                        build_chat_payload, extract_choice_content,
                                        parse_choice_answer)
from jev_observatory.redact import Redactor, base_url, get_api_key  # noqa: E402
from jev_observatory.revision_run import PersistentBudget  # noqa: E402
from jev_observatory.transport import HttpxTransport  # noqa: E402

RUN_DIR = _ROOT / "runs_archprobe" / fb.RUN_DIRNAME
REPORT_DIR = _ROOT / "data_report" / fb.REPORT_DIRNAME
CRED_PATH = Path("/tmp/jev-private-followup-probes-br56dzfs/credentials.json")
PROBE_DIR = Path("/tmp/probe")
ROWS_PATH = RUN_DIR / "rows.jsonl"
PLAN_PATH = RUN_DIR / "plan_frozen.json"
SOURCES_PATH = RUN_DIR / "sources.json"
LEDGER_PATH = RUN_DIR / "BUDGET-followup.json"

HF_REPOS = ["Qwen/Qwen3-30B-A3B-Instruct-2507", "Qwen/Qwen3-Next-80B-A3B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen3.5-9B"]
HF_FILES = ["tokenizer.json", "tokenizer_config.json", "config.json"]
_PROBE_FILE = {"Qwen/Qwen3-30B-A3B-Instruct-2507": "Qwen3-30B-A3B-Instruct-2507__",
               "Qwen/Qwen3-Next-80B-A3B-Instruct": "Qwen3-Next-80B-A3B-Instruct__",
               "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5-7B-Instruct__",
               "Qwen/Qwen3.5-9B": "Qwen3.5-9B__"}

JevFailed = RuntimeError


# ------------------------------------------------------------------ credentials
def load_credentials() -> None:
    """Read the owner-0600 credentials file in-process and inject env vars."""
    if not CRED_PATH.exists():
        return
    mode = CRED_PATH.stat().st_mode & 0o777
    if mode & 0o077:
        raise SystemExit(f"{CRED_PATH}: refusing credentials file wider than 0600")
    creds = json.loads(CRED_PATH.read_text(encoding="utf-8"))
    for key in ("TYPESAFE_API_KEY", "OPENROUTER_API_KEY", "HF_TOKEN"):
        if creds.get(key):
            os.environ[key] = creds[key]


# ------------------------------------------------------------------ captures
class _RespProxy:
    def __init__(self, resp, owner):
        self._r, self._o = resp, owner
        self._body = b""

    @property
    def status_code(self):
        return self._r.status_code

    @property
    def headers(self):
        return self._r.headers

    def iter_bytes(self):
        for chunk in self._r.iter_bytes():
            self._body += chunk
            yield chunk
        self._o.last = {"status": self._r.status_code,
                        "headers": dict(self._r.headers), "body": self._body}


class _CapturedStream:
    def __init__(self, cap, method, path, payload):
        self._cap, self._method, self._path, self._payload = cap, method, path, payload

    def __enter__(self):
        # exact wire bytes: serialize with ensure_ascii=False so CJK stays raw
        # UTF-8 (httpx json= would escape it and break byte-matched stimuli)
        body = (jsonlib.dumps(self._payload, ensure_ascii=False)
                .encode("utf-8")) if self._payload is not None else None
        self._cap.last_payload_bytes = len(body or b"")
        self._cm = self._cap._client.stream(
            self._method, self._path, content=body,
            headers={"Content-Type": "application/json"})
        return _RespProxy(self._cm.__enter__(), self._cap)

    def __exit__(self, *exc):
        return self._cm.__exit__(*exc)


class CapturingTransport:
    """Wraps the raw httpx client so record_from_response rows can carry the
    sanitized raw body, full response headers and the SERVED model string."""

    def __init__(self, client, redactor: Redactor):
        self._client, self.redactor = client, redactor
        self.last: dict | None = None

    def stream(self, method, path, json=None):
        return _CapturedStream(self, method, path, json)

    def capture_extras(self) -> dict[str, Any]:
        last = self.last or {}
        body = last.get("body") or b""
        served = None
        service_ms = None
        try:
            parsed = jsonlib.loads(body.decode("utf-8", "replace"))
            served = parsed.get("model")
            service_ms = parsed.get("serviceMs")
        except Exception:
            parsed = None
        return {
            "response_headers_sanitized": self.redactor.headers(last.get("headers") or {}),
            "raw_body_sanitized": self.redactor.text(body.decode("utf-8", "replace"))[:20000],
            "served_model": served,
            "service_ms_body": service_ms,
            "capture_status": last.get("status"),
        }


jsonlib = json


# ------------------------------------------------------------ source verification
def _norm(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold()


def _plaintext(html: bytes) -> str:
    text = html.decode("utf-8", "replace")
    text = re.sub(r"(?is)<(script|style).*?</\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text)


def _excerpt_for(text: str, anchor: str) -> str | None:
    nt = _norm(text)
    for alt in anchor.split("||"):
        na = _norm(alt)
        i = nt.find(na)
        if i >= 0:
            return text[max(0, i - 200):i + len(alt) + 200].strip()
    return None


def fetch_sources(timeout: float = 25.0) -> dict[str, Any]:
    out: dict[str, Any] = {"schema": "jev-followup-sources.v1",
                           "version": fb.FOLLOWUP_VERSION, "items": {}}
    req_headers = {"User-Agent": "jev-observatory-research/1.0 (source verification)"}
    for item in fb.HORIZON_ITEMS:
        entry: dict[str, Any] = {"item_id": item.item_id, "kind": item.kind,
                                 "event_date": item.event_date,
                                 "gold": item.gold, "gold_anchor": item.gold_anchor,
                                 "attempts": [], "verified": False}
        for url in item.sources:
            try:
                req = urllib.request.Request(url, headers=req_headers)
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    raw = resp.read(5_000_000)
                text = _plaintext(raw)
                excerpt = _excerpt_for(text, item.gold_anchor)
                entry["attempts"].append({
                    "url": url, "status": 200, "bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "excerpt": excerpt, "anchor_found": excerpt is not None})
                if excerpt is not None:
                    entry.update({"source_url": url,
                                  "source_sha256": hashlib.sha256(raw).hexdigest(),
                                  "source_excerpt": excerpt, "verified": True})
                    break
            except Exception as exc:
                entry["attempts"].append({"url": url,
                                          "error": f"{type(exc).__name__}: {exc}"})
        if not item.sources:
            entry.update({"source_url": "constructed control (own synthetic)",
                          "source_excerpt": fb._NOISE, "verified": True,
                          "note": "false-event control; no real source exists"})
        out["items"][item.item_id] = entry
    out["unverified"] = [i for i, e in out["items"].items() if not e["verified"]]
    out["all_real_golds_verified"] = not out["unverified"]
    return out


# ---------------------------------------------------------------- offline stage
def offline_stage() -> dict[str, Any]:
    summary: dict[str, Any] = {}
    pins: dict[str, Any] = {"schema": "jev-followup-tokenizer-pins.v1",
                            "hf_api": "https://huggingface.co/api/models/",
                            "repos": {}}
    for repo in HF_REPOS:
        try:
            with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}",
                                        timeout=20) as resp:
                meta = jsonlib.loads(resp.read())
            sha = meta.get("sha")
            files = {}
            for fname in HF_FILES:
                url = f"https://huggingface.co/{repo}/resolve/{sha}/{fname}"
                try:
                    req = urllib.request.Request(
                        url, headers={"User-Agent": "jev-observatory-research/1.0"})
                    with urllib.request.urlopen(req, timeout=60) as resp:
                        raw = resp.read(25_000_000)
                except Exception as exc:
                    files[fname] = {"error": f"{type(exc).__name__}: {exc}"}
                    continue
                digest = hashlib.sha256(raw).hexdigest()
                probe_name = _PROBE_FILE[repo] + fname.replace(".json", ".json")
                probe_path = PROBE_DIR / ({"tokenizer.json": _PROBE_FILE[repo] + "tokenizer.json",
                                           "tokenizer_config.json": "Qwen__" + repo.split("/")[1] + "__tokenizer_config.json",
                                           "config.json": "Qwen__" + repo.split("/")[1] + "__config.json"}[fname])
                prior = None
                if probe_path.exists():
                    prior = hashlib.sha256(probe_path.read_bytes()).hexdigest()
                files[fname] = {"pinned_url": url, "sha256": digest, "bytes": len(raw),
                                "prior_download_sha256": prior,
                                "matches_prior_download": prior == digest if prior else None}
            pins["repos"][repo] = {"hf_commit_sha": sha, "files": files}
        except Exception as exc:
            pins["repos"][repo] = {"error": f"{type(exc).__name__}: {exc}"}
    (REPORT_DIR / "tokenizer_pins.json").write_text(
        jsonlib.dumps(pins, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    summary["tokenizer_pins"] = {
        repo: d.get("hf_commit_sha", "error") for repo, d in pins["repos"].items()}

    # ---- decode/encode membership (ACTUAL tokenizers; never literal vocab keys)
    membership: dict[str, Any] = {"schema": "jev-followup-decode-membership.v1",
                                  "method": ("tokenizers.Tokenizer.encode -> len(ids); "
                                             "literal vocab-key lookup reported ONLY "
                                             "as the artifact contrast"),
                                  "candidates": {}}
    try:
        from tokenizers import Tokenizer  # type: ignore
        have_tok = True
    except Exception:
        have_tok = False
        membership["note"] = ("python 'tokenizers' wheel unavailable in this "
                              "environment; candidate columns not recomputed "
                              "here (parent verified: Qwen3 match 344/415, "
                              "3.5 match 337; zzz/______/heart split by Qwen3, "
                              "heart merged by Qwen3.5)")
    probe_strings = [("", "empty")] + [(t, cid) for cid, t in fb._TOKEN_CORES] \
        + [(t, n) for n, t in fb._MERGE_INPUTS]
    if have_tok:
        for repo in HF_REPOS:
            try:
                path = PROBE_DIR / ({"Qwen/Qwen3-30B-A3B-Instruct-2507":
                                     "Qwen3-30B-A3B-Instruct-2507__tokenizer.json",
                                     "Qwen/Qwen3-Next-80B-A3B-Instruct":
                                     "Qwen3-Next-80B-A3B-Instruct__tokenizer.json",
                                     "Qwen/Qwen2.5-7B-Instruct":
                                     "Qwen2.5-7B-Instruct__tokenizer.json",
                                     "Qwen/Qwen3.5-9B":
                                     "Qwen3.5-9B__tokenizer.json"}[repo])
                tok = Tokenizer.from_file(str(path))
                vocab = jsonlib.loads(path.read_bytes()).get("vocab", {})
                table = {}
                for text, sid in probe_strings:
                    n_tok = len(tok.encode(text, add_special_tokens=False).ids)
                    literal = text in vocab  # the known-wrong representation
                    table[sid or "empty"] = {"encode_tokens": n_tok,
                                             "literal_key_hit": literal,
                                             "agrees": (n_tok == 1) == literal}
                membership["candidates"][repo] = table
            except Exception as exc:
                membership["candidates"][repo] = {"error": f"{type(exc).__name__}: {exc}"}
        # re-derive the Jev 234-string one-token set and compare with candidates
        fp = PROBE_DIR / "tokenizer-fingerprint.json"
        if fp.exists():
            trials = jsonlib.loads(fp.read_bytes())["trials"]
            one = [t for t in trials if t["family"] != "control" and t["in_tok"] == 268]
            membership["jev_one_token_rederived"] = len(one)
            overlap = {}
            for repo in HF_REPOS:
                table = membership["candidates"].get(repo)
                if isinstance(table, dict) and "encode_tokens" in next(iter(table.values() or {"x": {}}), {}):
                    pass
            membership["jev_one_token_note"] = ("re-derived from Archer's own "
                                               "fingerprint rows via fresh-empty "
                                               "marginals (in_tok == 267 + 1); "
                                               "count parity with the parent's "
                                               "234 result is the cross-check")
    (REPORT_DIR / "token_decode_membership.json").write_text(
        jsonlib.dumps(membership, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    summary["decode_membership"] = {
        "tokenizers_available": have_tok,
        "jev_one_token_rederived": membership.get("jev_one_token_rederived")}

    # ---- pinned confidence formulae on synthetic vectors
    synthetic = {
        "uniform_K4": fb.choice_confidence([0.25, 0.25, 0.25, 0.25]),
        "onehot_K4": fb.choice_confidence([1.0, 0.0, 0.0, 0.0]),
        "two_thirds_K3": fb.score_confidence([0.6, 0.3, 0.1]),
        "uniform_score_K3": fb.score_confidence([1 / 3, 1 / 3, 1 / 3]),
        "K1": fb.choice_confidence([1.0]),
    }
    confidence = {"source_pin": fb.CONFIDENCE_SOURCE_PIN,
                  "recovery_spec": recovery_spec(),
                  "synthetic_vectors": {k: round(v, 6) for k, v in synthetic.items()},
                  "expected": {"uniform_K4": 0.0, "onehot_K4": 1.0, "K1": 1.0,
                               "uniform_score_K3": 0.0}}
    (REPORT_DIR / "confidence_formula_check.json").write_text(
        jsonlib.dumps(confidence, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    summary["confidence_formula"] = {"synthetic_ok": all(
        abs(synthetic[k] - v) < 1e-9 for k, v in confidence["expected"].items())}

    # ---- versioned P5 gold correction (originals preserved)
    probe2_rows = _ROOT / "runs_archprobe" / "probe2" / "probe2_rows.jsonl"
    if probe2_rows.exists():
        rows = [jsonlib.loads(l) for l in probe2_rows.read_text(encoding="utf-8").splitlines()
                if l.strip()]
        corrected = fb.p5_corrected_summary(rows)
        (REPORT_DIR / "p5_horizon_corrected.json").write_text(
            jsonlib.dumps(corrected, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        summary["p5_correction"] = {
            "n_original_rows": corrected["n_original_rows"],
            "n_excluded": len(corrected["excluded_rows"]),
            "n_corrected": len(corrected["corrected_rows"])}
    else:
        summary["p5_correction"] = {"status": "probe2 rows missing"}
    return summary


# ------------------------------------------------------------------ live stage
def _load_rows() -> list[dict]:
    if not ROWS_PATH.exists():
        return []
    return [jsonlib.loads(l) for l in ROWS_PATH.read_text(encoding="utf-8").splitlines()
            if l.strip()]


def _append_row(row: dict) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    with ROWS_PATH.open("a", encoding="utf-8") as fh:
        fh.write(Redactor().text(jsonlib.dumps(row, ensure_ascii=False)) + "\n")
        fh.flush()


def _jev_attempts(call, cap, budget, config_id) -> tuple[dict, bool]:
    """Dispatch one logical call with bounded attempts. Returns (row, abort_run)."""
    worst = fb.JEV_MAX_INPUT_TOKENS * fb.JEV_USD_PER_M_INPUT / 1e6 * fb.MAX_TRANSPORT_ATTEMPTS
    rid = budget.reserve(config_id, worst, label=call.family)
    row: dict = {}
    for attempt in range(1, fb.MAX_TRANSPORT_ATTEMPTS + 1):
        cold = attempt == 1 and cap.last is None
        row = ap.record_from_response(call, cap, cold=cold)
        row.update(cap.capture_extras())
        if row.get("upstream_ms") is None and row.get("service_ms_body") is not None:
            try:
                row["upstream_ms"] = float(row["service_ms_body"])
            except (TypeError, ValueError):
                pass
        row["attempt"] = attempt
        status = row.get("http_status") or 0
        if status in (401, 402, 403) or "401" in str(row.get("error")) \
                or "402" in str(row.get("error")):
            budget.settle(rid, None)          # fail closed: unknown/quota/auth
            return row, True
        transient = bool(row.get("error")) or status in (0, 429, 500, 502, 503, 504)
        if not transient:
            tokens = row.get("usage_input_tokens")
            cost = (tokens * fb.JEV_USD_PER_M_INPUT / 1e6
                    if isinstance(tokens, int) else None)
            budget.settle(rid, cost)          # known cost settles; None holds
            return row, False
    budget.settle(rid, None)                  # exhausted retries: hold forever
    if row is not None and not row.get("error"):
        row["error"] = f"attempts_exhausted:http_{row.get('http_status')}"
    return row, False


def live_jev(limit: int = 0) -> int:
    load_credentials()
    os.environ.setdefault("JEVO_ALLOW_LIVE", "1")
    if not get_api_key():
        raise SystemExit("no TYPESAFE_API_KEY available; refusing live dispatch")
    if not PLAN_PATH.exists():
        raise SystemExit("plan_frozen.json missing; run 'freeze' first (prereg)")
    budget = PersistentBudget(LEDGER_PATH, fb.BATTERY_CAP_USD)
    done = {r.get("config_id") for r in _load_rows() if r.get("family") != "fu_reference"}
    calls, _orders = fb.build_all()
    red = Redactor.from_environment()
    http = HttpxTransport(base_url=base_url(), api_key=get_api_key() or "",
                          timeout_seconds=180.0, redactor=red)
    cap = CapturingTransport(http._client, red)
    n = 0
    try:
        for call in calls:
            if call.config_id in done:
                continue
            if limit and n >= limit:
                break
            row, abort = _jev_attempts(call, cap, budget, call.config_id)
            row["seq_note"] = "followup-20261003 live-jev"
            _append_row(row)
            n += 1
            if abort:
                print("[fail-closed] auth/quota/price error; run stopped",
                      file=sys.stderr)
                return 2
    finally:
        http.close()
    print(jsonlib.dumps({"dispatched": n, "ledger": budget.snapshot()}, indent=1))
    return 0


def live_ref(smoke: int = 0) -> int:
    load_credentials()
    os.environ.setdefault("JEVO_ALLOW_LIVE", "1")
    if not PLAN_PATH.exists():
        raise SystemExit("plan_frozen.json missing; run 'freeze' first (prereg)")
    from jev_observatory.openrouter import OpenRouterHttpTransport
    budget = PersistentBudget(LEDGER_PATH, fb.BATTERY_CAP_USD)
    red = Redactor.from_environment()
    done = {(r.get("item_id"), r.get("model_requested"), r.get("rot", 0))
            for r in _load_rows() if r.get("family") == "fu_reference"}
    plan_models = list(fb.REFERENCE_MODELS.items())
    n = 0
    transports = {}
    try:
        for item in fb.HORIZON_ITEMS:
            for rot in fb._HORIZON_ROTATIONS:
                base = list(item.options)
                opts = base[rot:] + base[:rot]
                req = ap.SystemOneRequest(
                    state="This question is about public events.",
                    model=fb.MODEL_PIN,
                    questions={"q0": ap.ChoiceQuestion(
                        instructions=item.question,
                        criteria={f"o{i}": t for i, t in enumerate(opts)})})
                gold_key = f"o{opts.index(item.gold)}"
                keys = list(req.questions["q0"].criteria)
                for slot, model_id in plan_models:
                    if (item.item_id, model_id, rot) in done:
                        continue
                    if smoke and n >= smoke:
                        return 0
                    rates = fb.REFERENCE_RATES_USD_PER_TOKEN[slot]
                    worst = (fb.REFERENCE_PROMPT_TOKEN_BOUND * rates["in"]
                             + fb.REFERENCE_OUT_CAP * rates["out"])
                    rid = budget.reserve(f"ref:{item.item_id}:{slot}:rot{rot}", worst,
                                         label="fu_reference")
                    transport = transports.get(slot)
                    if transport is None:
                        transport = OpenRouterHttpTransport(
                            os.environ.get("OPENROUTER_API_KEY") or "", redactor=red)
                        transports[slot] = transport
                    wire = WireConfig(model=model_id,
                                      max_output_tokens=fb.REFERENCE_OUT_CAP)
                    payload = build_chat_payload(req, wire)
                    resp = transport.post(CHAT_PATH, payload)
                    body = jsonlib.loads(resp.body.decode("utf-8", "replace")) \
                        if resp.body else {}
                    content, extract_error = extract_choice_content(body)
                    strict_key, strict_error = parse_choice_answer(content, keys)
                    recovered_key, stage = recover_choice(content, keys)
                    usage = body.get("usage") or {}
                    row = {
                        "family": "fu_reference", "item_id": item.item_id,
                        "kind": item.kind, "event_date": item.event_date,
                        "rot": rot,
                        "model_requested": model_id,
                        "model_returned": body.get("model"),
                        "finish_reason": (body["choices"][0].get("finish_reason")
                                          if isinstance(body.get("choices"), list)
                                          and body["choices"] else None),
                        "http_status": resp.status_code,
                        "error": resp.error or (None if resp.status_code == 200
                                                else f"http_{resp.status_code}"),
                        "content_excerpt": red.text(content or "")[:400],
                        "extract_error": extract_error,
                        "strict_key": strict_key, "strict_error": strict_error,
                        "recovered_key": recovered_key, "recovery_stage": stage,
                        "gold_key": gold_key,
                        "usage_prompt_tokens": usage.get("prompt_tokens"),
                        "usage_completion_tokens": usage.get("completion_tokens"),
                        "usage_cost_reported": usage.get("cost"),
                        "total_ms": resp.total_ms,
                    }
                    if resp.status_code in (401, 402, 403):
                        budget.settle(rid, None)
                        _append_row(row)
                        print("[fail-closed] reference auth/quota/price error",
                              file=sys.stderr)
                        return 2
                    known = (isinstance(usage.get("prompt_tokens"), int)
                             and isinstance(usage.get("completion_tokens"), int))
                    cost = (usage["prompt_tokens"] * rates["in"]
                            + usage["completion_tokens"] * rates["out"]) if known else None
                    budget.settle(rid, cost)     # None (unknown) holds forever
                    _append_row(row)
                    n += 1
    finally:
        for t in transports.values():
            t.close()
    print(jsonlib.dumps({"dispatched": n, "ledger": budget.snapshot()}, indent=1))
    return 0


# ------------------------------------------------------------------ analyze
def analyze() -> int:
    rows = _load_rows()
    membership_path = REPORT_DIR / "token_decode_membership.json"
    candidate_counts = None
    if membership_path.exists():
        candidate_counts = jsonlib.loads(membership_path.read_text(encoding="utf-8"))
    analysis = fb.analyze_rows(rows, candidate_counts)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "analysis.json").write_text(
        jsonlib.dumps(analysis, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    counts = analysis["counts_reported_separately"]
    print(jsonlib.dumps({"counts": counts,
                         "fu_vocab_cores": analysis["fu_vocab"]["cores"],
                         "place_ratios": {k: v.get("question_vs_state_cost_ratio")
                                          for k, v in
                                          analysis["fu_place"]["latency"].get("models", {}).items()},
                         "odds_contrasts": analysis["fu_odds"]["archer_contrasts"],
                         "horizon": analysis["fu_horizon"]["per_kind"],
                         "reference": analysis["fu_reference"]["per_model"]},
                        indent=1, ensure_ascii=False)[:4000])
    return 0


# ------------------------------------------------------------------ main
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["plan", "fetch-sources", "offline",
                                          "freeze", "live-jev", "live-ref",
                                          "analyze", "sources-v2", "calibrate",
                                          "freeze-v2", "live-v2", "analyze-v2", "calibrate-v3", "freeze-v3",
                                          "live-v3", "analyze-v3", "placement-derived"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--smoke", type=int, default=0)
    args = parser.parse_args()

    if args.stage == "plan":
        budget = fb.worst_case_costs()
        print(jsonlib.dumps(budget, indent=1))
        return 0 if budget["fits_cap"] else 3

    if args.stage == "fetch-sources":
        sources = fetch_sources()
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        SOURCES_PATH.write_text(jsonlib.dumps(sources, indent=1, ensure_ascii=False) + "\n",
                                encoding="utf-8")
        print(jsonlib.dumps({"verified": sources["all_real_golds_verified"],
                             "unverified": sources["unverified"]}, indent=1))
        return 0 if sources["all_real_golds_verified"] else 4

    if args.stage == "offline":
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        summary = offline_stage()
        (REPORT_DIR / "offline_summary.json").write_text(
            jsonlib.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(jsonlib.dumps(summary, indent=1, ensure_ascii=False)[:2000])
        return 0

    if args.stage == "freeze":
        if not SOURCES_PATH.exists():
            raise SystemExit("sources.json missing; run fetch-sources first")
        sources = jsonlib.loads(SOURCES_PATH.read_text(encoding="utf-8"))
        if not sources.get("all_real_golds_verified"):
            raise SystemExit("real-event golds unverified; stop and escalate "
                             "before live (contact_supervisor)")
        offline_path = REPORT_DIR / "offline_summary.json"
        offline = (jsonlib.loads(offline_path.read_text(encoding="utf-8"))
                   if offline_path.exists() else {})
        _calls, block_orders = fb.build_all()
        place_orders = block_orders
        plan = fb.build_plan(place_orders, sources, offline)
        if not plan["budget"]["fits_cap"]:
            raise SystemExit("worst-case plan exceeds the $5 cap; escalate before live")
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        PLAN_PATH.write_text(jsonlib.dumps(plan, indent=1, ensure_ascii=False) + "\n",
                             encoding="utf-8")
        (RUN_DIR / "provenance.json").write_text(jsonlib.dumps({
            "schema": "jev-followup-provenance.v1", "version": fb.FOLLOWUP_VERSION,
            "model_pin": fb.MODEL_PIN, "reference_models": fb.REFERENCE_MODELS,
            "reference_rates_usd_per_token": fb.REFERENCE_RATES_USD_PER_TOKEN,
            "recovery_spec": recovery_spec(),
            "confidence_source_pin": fb.CONFIDENCE_SOURCE_PIN,
            "hf_pins": offline.get("tokenizer_pins"),
            "ledger": str(LEDGER_PATH), "cap_usd": fb.BATTERY_CAP_USD,
            "max_transport_attempts": fb.MAX_TRANSPORT_ATTEMPTS,
        }, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(jsonlib.dumps({"frozen": str(PLAN_PATH),
                             "budget": plan["budget"]["total_worst_usd"]}, indent=1))
        return 0

    if args.stage == "sources-v2":
        return sources_v2()
    if args.stage == "calibrate":
        return calibrate()
    if args.stage == "freeze-v2":
        return freeze_v2()
    if args.stage == "live-v2":
        return live_v2(args.limit)
    if args.stage == "analyze-v2":
        return analyze_v2()
    if args.stage == "calibrate-v3":
        return calibrate_v3()
    if args.stage == "freeze-v3":
        return freeze_v3()
    if args.stage == "live-v3":
        return live_v3(args.limit)
    if args.stage == "analyze-v3":
        return analyze_v3()
    if args.stage == "placement-derived":
        derived = fb.placement_incremental_summary(_load_rows() + _load_rows_v2())
        out = REPORT_DIR / "v2" / "placement_ratio_derived.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(jsonlib.dumps(derived, indent=1, ensure_ascii=False) + "\n",
                       encoding="utf-8")
        print(jsonlib.dumps({k: derived[k] for k in
                             ("total_ratio_mean", "incremental_ms_per_1k",
                              "incremental_ratio",
                              "incremental_ratio_block_bootstrap_ci95", "label")},
                            indent=1, ensure_ascii=False))
        return 0
    if args.stage == "live-jev":
        return live_jev(args.limit)
    if args.stage == "live-ref":
        return live_ref(args.smoke)
    return analyze()




# ===================================================================== v2 repair
RUN_DIR_V2 = RUN_DIR / "v2"
ROWS_V2 = RUN_DIR_V2 / "rows_v2.jsonl"
PLAN_V2 = RUN_DIR_V2 / "plan_v2_frozen.json"
CALIB_PATH = RUN_DIR_V2 / "calib_fit.json"
REPORT_V2 = REPORT_DIR / "v2"
SOURCES_V2 = RUN_DIR_V2 / "sources_v2.json"

_TOK_CACHE: dict[str, Any] = {}


def _encoders():
    if _TOK_CACHE:
        return _TOK_CACHE
    try:
        from tokenizers import Tokenizer  # type: ignore
        _TOK_CACHE["qwen3"] = Tokenizer.from_file(str(
            PROBE_DIR / "Qwen3-30B-A3B-Instruct-2507__tokenizer.json"))
        _TOK_CACHE["qwen35"] = Tokenizer.from_file(str(
            PROBE_DIR / "Qwen3.5-9B__tokenizer.json"))
    except Exception:
        pass
    return _TOK_CACHE


def annotate_encode(call) -> None:
    """Attach ACTUAL candidate-encoder counts to timing/calibration rows."""
    enc = _encoders()
    if not enc:
        return
    req = call.request
    text = req.state + " " + " ".join(
        (q.instructions if isinstance(q.instructions, str) else "")
        for q in req.questions.values())
    call.meta["encode_qwen3_tokens"] = len(
        enc["qwen3"].encode(text, add_special_tokens=False).ids)
    call.meta["encode_qwen35_tokens"] = len(
        enc["qwen35"].encode(text, add_special_tokens=False).ids)


def _append_row_v2(row: dict) -> None:
    RUN_DIR_V2.mkdir(parents=True, exist_ok=True)
    with ROWS_V2.open("a", encoding="utf-8") as fh:
        # FULL sanitized raw (never truncate bodies again)
        fh.write(Redactor().text(jsonlib.dumps(row, ensure_ascii=False)) + "\n")
        fh.flush()


def _load_rows_v2() -> list[dict]:
    if not ROWS_V2.exists():
        return []
    return [jsonlib.loads(l) for l in ROWS_V2.read_text(encoding="utf-8").splitlines()
            if l.strip()]


def _apply_target_qid(row: dict, call, cap) -> None:
    """Two-question rows: re-read answers[target_qid] from the full body —
    record_from_response takes the FIRST answer, unsafe under question reordering
    and mixed noul/choice."""
    target_qid = call.meta.get("target_qid")
    if not target_qid:
        return
    try:
        parsed = jsonlib.loads((cap.last or {}).get("body", b"{}").decode("utf-8"))
    except Exception:
        return
    criteria = dict(call.request.questions[target_qid].criteria)
    probs, choice = fb.extract_answer_by_qid(parsed, target_qid, criteria)
    row["probabilities"] = probs
    row["choice"] = choice
    row["target_qid_extracted"] = True


def live_v2(limit: int = 0) -> int:
    load_credentials()
    os.environ.setdefault("JEVO_ALLOW_LIVE", "1")
    if not get_api_key():
        raise SystemExit("no TYPESAFE_API_KEY available; refusing live dispatch")
    if not PLAN_V2.exists():
        raise SystemExit("plan_v2_frozen.json missing; run freeze-v2 first")
    budget = PersistentBudget(LEDGER_PATH, fb.BATTERY_CAP_USD)   # SAME ledger/cap
    done = {r.get("config_id") for r in _load_rows_v2()}
    calls = (fb.load_relational_replication() + fb.build_fu_isolation()
             + fb.build_fu_timing(jsonlib.loads(CALIB_PATH.read_text(encoding="utf-8"))))
    red = Redactor.from_environment()
    http = HttpxTransport(base_url=base_url(), api_key=get_api_key() or "",
                          timeout_seconds=180.0, redactor=red)
    cap = CapturingTransport(http._client, red)
    n = 0
    try:
        for call in calls:
            if call.config_id in done:
                continue
            if limit and n >= limit:
                break
            annotate_encode(call)
            row, abort = _jev_attempts(call, cap, budget, call.config_id)
            _apply_target_qid(row, call, cap)
            _append_row_v2(row)
            n += 1
            if abort:
                print("[fail-closed] auth/quota/price error; run stopped",
                      file=sys.stderr)
                return 2
    finally:
        http.close()
    print(jsonlib.dumps({"dispatched": n, "ledger": budget.snapshot()}, indent=1))
    return 0


def calibrate() -> int:
    """Counter-only calibration (<=20 calls): Jev counts per script at fixed
    char lengths; the final 3k/12k stimulus lengths are set from these counts,
    never from latency responses."""
    load_credentials()
    os.environ.setdefault("JEVO_ALLOW_LIVE", "1")
    budget = PersistentBudget(LEDGER_PATH, fb.BATTERY_CAP_USD)
    calls = fb.build_fu_calib()
    if len(calls) > 20:
        raise SystemExit("calibration exceeds 20 calls")
    red = Redactor.from_environment()
    http = HttpxTransport(base_url=base_url(), api_key=get_api_key() or "",
                          timeout_seconds=120.0, redactor=red)
    cap = CapturingTransport(http._client, red)
    fit: dict[str, dict[str, float]] = {}
    try:
        per: dict[str, dict[int, list[int]]] = {}
        for call in calls:
            annotate_encode(call)
            row, abort = _jev_attempts(call, cap, budget, call.config_id)
            _append_row_v2(row)
            if abort:
                return 2
            if row.get("usage_input_tokens"):
                per.setdefault(call.meta["script"], {}).setdefault(
                    call.meta["n_chars"], []).append(row["usage_input_tokens"])
        base = min(min(v) for s in per.values() for v in s.values())
        for script, pts in per.items():
            (x1, y1), (x2, y2) = sorted((x, _fmed(v)) for x, v in pts.items())[:2]
            per_char = (y2 - y1) / (x2 - x1)
            intercept = y1 - per_char * x1
            fit[script] = {"per_char": round(per_char, 5),
                           "intercept": round(intercept, 2),
                           "counts": {str(x): _fmed(v) for x, v in pts.items()}}
    finally:
        http.close()
    RUN_DIR_V2.mkdir(parents=True, exist_ok=True)
    CALIB_PATH.write_text(jsonlib.dumps(fit, indent=1) + "\n", encoding="utf-8")
    print(jsonlib.dumps({"calib_fit": fit, "ledger": budget.snapshot()}, indent=1))
    return 0


def _fmed(xs):
    s = sorted(xs)
    return s[len(s) // 2] if s else 0.0


def freeze_v2() -> int:
    if not CALIB_PATH.exists():
        raise SystemExit("calibrate first (counts determine stimulus lengths)")
    calib_fit = jsonlib.loads(CALIB_PATH.read_text(encoding="utf-8"))
    rel = fb.load_relational_replication()
    iso = fb.build_fu_isolation()
    tim = fb.build_fu_timing(calib_fit)
    # literal card-location invariants asserted at freeze time
    for c in rel:
        state_card = fb._CARD_RE.search(c.request.state) is not None
        criteria = dict(next(iter(c.request.questions.values())).criteria)
        ref_text = criteria[c.meta["ref_only_key"]]
        if c.meta["placement"] == "option":
            assert not state_card and fb._CARD_RE.search(ref_text), c.config_id
        else:
            assert state_card and fb._CARD_RE.search(ref_text) is None, c.config_id
    plan = {
        "schema": "jev-followup-plan.v2", "version": fb.V2_VERSION,
        "model_pin": fb.MODEL_PIN,
        "v1_immutable": ["runs_archprobe/followup_20261003/rows.jsonl",
                         "runs_archprobe/followup_20261003/plan_frozen.json"],
        "counts": {"fu_relational": len(rel), "fu_isolation": len(iso),
                   "fu_timing": len(tim), "fu_calib": 12,
                   "global_extra_cap": 374,
                   "total_requests_after_v2": 526 + len(rel) + len(iso) + len(tim) + 12},
        "calib_fit": calib_fit,
        "timing_stimuli_chars": {c.config_id: c.meta["n_chars"] for c in tim},
        "timing_encode_counts": {c.config_id: {"encode_qwen3_tokens": None}
                                 for c in tim},
        "hypotheses": {
            "fu_relational": ("exact published payloads: state placement near "
                              "ceiling, option placement lower (card reachable "
                              "only through a sibling option); condition-guess "
                              "baseline 1/2, gold key varies with value x order"),
            "fu_isolation": ("P(selected code) in sibling vs absent is the "
                             "question-isolation test with both questions always "
                             "present; state/target-instruction channels are "
                             "positive controls; sibling success and correctness "
                             "are NOT assumed"),
            "fu_timing": ("PRIMARY: paired whole-block bootstrap of the Q1 "
                          "state-vs-question elapsed-time ratio (expect ~1: 'no "
                          "double penalty observed in elapsed time', NOT a "
                          "conclusive refutation of physical 2x compute). Cross-"
                          "script: digit runs (Jev ~1/char vs Qwen ~1/3) separate "
                          "reported-count from Qwen-count timing models on "
                          "whole-script holdout; payload bytes included so "
                          "Unicode upload size is not ignored"),
        },
        "budget": fb.worst_case_costs(),
        "method": {"nonce": "fresh 8-hex marker at the VERY BEGINNING of state "
                            "(before all filler): no common prompt prefix across "
                            "requests (v1 end-of-state nonce did NOT break "
                            "prefix caching — v1 code comment was wrong)",
                   "target_qid": "two-question rows parse answers[target_qid] "
                                 "from the full body (never first answer)",
                   "raw_capture": "full sanitized raw bodies stored (no truncation)",
                   "ledger": "SAME persistent $5 ledger (cap unchanged, "
                             "spent 0.0473 carried)"}}
    RUN_DIR_V2.mkdir(parents=True, exist_ok=True)
    PLAN_V2.write_text(jsonlib.dumps(plan, indent=1, ensure_ascii=False) + "\n",
                       encoding="utf-8")
    print(jsonlib.dumps({"frozen": str(PLAN_V2), "counts": plan["counts"]}, indent=1))
    return 0


def sources_v2() -> int:
    """Support excerpts that STATE the outcome (winner/result sentence or
    deterministic infobox row), not TOC name hits.  Public GET only; false
    events stay marked constructed (no absence is claimed to be 'verified')."""
    import urllib.request
    import unicodedata

    def norm(s):
        d = unicodedata.normalize("NFKD", s)
        return "".join(c for c in d if not unicodedata.combining(c)).casefold()

    cues = ("won", "winner", "champion", "defeated", "awarded", "elected",
            "landed", "took place", "summit", "beat", "final", "victory",
            "earth", "returned", "welcome", "uncrewed", "crew")
    out = {"schema": "jev-followup-sources.v2", "version": fb.V2_VERSION,
           "items": {}}
    for item in fb.HORIZON_ITEMS:
        entry = {"item_id": item.item_id, "kind": item.kind,
                 "gold": item.gold, "verified": False}
        if not item.sources:
            entry.update({"evidence_type": "constructed_false_event",
                          "verified": True,
                          "note": "constructed control; no real event exists and "
                                  "no absence claim is made"})
            out["items"][item.item_id] = entry
            continue
        for url in item.sources:
            try:
                req = urllib.request.Request(
                    url, headers={"User-Agent": "jev-observatory-research/1.0"})
                with urllib.request.urlopen(req, timeout=25) as resp:
                    raw = resp.read(5_000_000)
            except Exception:
                continue
            text = _plaintext(raw)
            nt = norm(text)
            best = None
            for sent in re.split(r"(?<=[.!?])\s+", text):
                ns = norm(sent)
                if any(norm(alt) in ns for alt in item.gold_anchor.split("||")) \
                        and any(c in ns for c in cues):
                    best = ("outcome_sentence", sent.strip()[:400])
                    break
            if best is None:
                for row_line in re.split(r"[\n|]", text):
                    nr = norm(row_line)
                    if any(norm(alt) in nr for alt in item.gold_anchor.split("||")) \
                            and any(w in nr for w in ("champion", "winner", "score",
                                                      "awarded", "result")):
                        best = ("infobox_row", row_line.strip()[:400])
                        break
            if best:
                entry.update({"evidence_type": best[0], "support_excerpt": best[1],
                              "source_url": url,
                              "source_sha256": hashlib.sha256(raw).hexdigest(),
                              "verified": True})
                break
        out["items"][item.item_id] = entry
    out["unverified"] = [i for i, e in out["items"].items() if not e["verified"]]
    SOURCES_V2.parent.mkdir(parents=True, exist_ok=True)
    SOURCES_V2.write_text(jsonlib.dumps(out, indent=1, ensure_ascii=False) + "\n",
                          encoding="utf-8")
    print(jsonlib.dumps({"unverified": out["unverified"]}, indent=1))
    return 0


def analyze_v2() -> int:
    rows = _load_rows() + _load_rows_v2()
    analysis = fb.analyze_v2(rows, _load_rows_v2())
    REPORT_V2.mkdir(parents=True, exist_ok=True)
    (REPORT_V2 / "analysis_v2.json").write_text(
        jsonlib.dumps(analysis, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    overrides = analysis["reference_recovery"]
    (REPORT_V2 / "reference_recovery_overrides.json").write_text(
        jsonlib.dumps(overrides, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(jsonlib.dumps({
        "counts": analysis["counts_reported_separately"],
        "relational": analysis["fu_relational"]["by_placement"],
        "isolation": analysis["fu_isolation"]["by_location"],
        "timing_primary": {k: analysis["fu_timing"][k] for k in
                           ("primary_q1_state_vs_question_ratio",
                            "ratio_block_bootstrap_ci95")},
        "odds_primary": {k: analysis["odds_primary"][k] for k in
                         ("per_arm_mean_primary", "append_minus_base_mean",
                          "append_minus_base_block_bootstrap_ci95")},
        "reference": {m: b["gold_rate"] for m, b in
                      overrides["per_model_recomputed"].items()},
    }, indent=1, ensure_ascii=False)[:3500])
    return 0




# ===================================================================== v3 counter
RUN_DIR_V3 = RUN_DIR / "v3"
ROWS_V3 = RUN_DIR_V3 / "rows_v3.jsonl"
PLAN_V3 = RUN_DIR_V3 / "plan_v3_frozen.json"
REPORT_V3 = REPORT_DIR / "v3"


def annotate_encode_v3(call) -> None:
    """Candidate-encoder counts incl. the ACTUAL o200k encoder."""
    annotate_encode(call)
    enc = _encoders()
    req = call.request
    text = req.state + " " + " ".join(
        (q.instructions if isinstance(q.instructions, str) else "")
        for q in req.questions.values())
    try:
        import tiktoken  # type: ignore
        if "o200k" not in enc:
            enc["o200k"] = tiktoken.get_encoding("o200k_base")
        call.meta["encode_o200k_tokens"] = len(enc["o200k"].encode(text))
    except Exception:
        pass


def _append_row_v3(row: dict) -> None:
    RUN_DIR_V3.mkdir(parents=True, exist_ok=True)
    with ROWS_V3.open("a", encoding="utf-8") as fh:
        fh.write(Redactor().text(jsonlib.dumps(row, ensure_ascii=False)) + "\n")
        fh.flush()


def _load_rows_v3() -> list[dict]:
    if not ROWS_V3.exists():
        return []
    return [jsonlib.loads(l) for l in ROWS_V3.read_text(encoding="utf-8").splitlines()
            if l.strip()]


def _dispatch_v3(call, cap, budget) -> tuple[dict, bool]:
    row, abort = _jev_attempts(call, cap, budget, call.config_id)
    row["payload_bytes_exact"] = getattr(cap, "last_payload_bytes", None)
    row["served_model"] = row.get("served_model")
    return row, abort


def calibrate_v3() -> int:
    load_credentials()
    os.environ.setdefault("JEVO_ALLOW_LIVE", "1")
    budget = PersistentBudget(LEDGER_PATH, fb.BATTERY_CAP_USD)
    calls = fb.build_fu_counter_calib()
    if len(calls) > 8:
        raise SystemExit("v3 calibration exceeds 8 calls")
    red = Redactor.from_environment()
    http = HttpxTransport(base_url=base_url(), api_key=get_api_key() or "",
                          timeout_seconds=120.0, redactor=red)
    cap = CapturingTransport(http._client, red)
    out = {}
    try:
        for call in calls:
            annotate_encode_v3(call)
            row, abort = _dispatch_v3(call, cap, budget)
            _append_row_v3(row)
            if abort:
                return 2
            out[call.config_id] = {
                "reported": row.get("usage_input_tokens"),
                "encode_qwen3": call.meta.get("encode_qwen3_tokens"),
                "encode_o200k": call.meta.get("encode_o200k_tokens"),
                "filler_bytes": call.meta["filler_bytes"],
                "style": call.meta["style"]}
    finally:
        http.close()
    RUN_DIR_V3.mkdir(parents=True, exist_ok=True)
    (RUN_DIR_V3 / "calib_v3.json").write_text(
        jsonlib.dumps(out, indent=1) + "\n", encoding="utf-8")
    # digits separator check (fail-fast BEFORE any latency freeze)
    dig = {k: v for k, v in out.items() if v["style"] == "digits"}
    sep = {}
    for k, v in dig.items():
        n = v["filler_bytes"]  # 1 byte per digit char
        sep[k] = {"jev_per_digit": round((v["reported"] or 0) / n, 3),
                  "qwen3_per_digit": round((v["encode_qwen3"] or 0) / n, 3),
                  "o200k_per_digit": round((v["encode_o200k"] or 0) / n, 3)}
    print(jsonlib.dumps({"calib": out, "digits_separator": sep,
                         "ledger": budget.snapshot()}, indent=1)[:2500])
    return 0


def freeze_v3() -> int:
    calib_path = RUN_DIR_V3 / "calib_v3.json"
    if not calib_path.exists():
        raise SystemExit("run calibrate-v3 first")
    calib = jsonlib.loads(calib_path.read_text(encoding="utf-8"))
    calls = fb.build_fu_counter_v3()
    # pre-freeze checks: byte matching at each size and digits separation
    for n_bytes in fb.V3_BYTE_SIZES:
        wire = {c.meta["payload_bytes"] for c in calls if c.meta["filler_bytes"] == n_bytes}
        if len(wire) != 1:
            raise SystemExit(f"byte match failed at {n_bytes}: {wire}")
    plan = {
        "schema": "jev-followup-plan.v3", "version": fb.V3_VERSION,
        "model_pin": fb.MODEL_PIN,
        "v1_v2_immutable": True,
        "counts": {"fu_counter_calib": len(calib), "fu_counter_v3": len(calls),
                   "jev_total_after_v3": 774 + len(calib) + len(calls)},
        "byte_volumes": list(fb.V3_BYTE_SIZES),
        "styles": list(fb.V3_STYLES),
        "stimuli_chars": {f"{s}:{b}": len(fb.V3_FILLERS[(s, b)])
                          for s in fb.V3_STYLES for b in fb.V3_BYTE_SIZES},
        "calib_v3": calib,
        "hypotheses": {
            "primary": ("runtime vs count-family: if processing scales with an "
                        "o200k-like tokenizer, digit runs (o200k 1/3, Jev 1/1) "
                        "finish FASTER per Jev-reported token than byte-matched "
                        "Latin/CJK; if it scales with the reported counter, "
                        "byte-matched styles track reported counts instead"),
            "rules": ("whole-style holdout RMSE + paired whole-block contrasts "
                      "with block bootstrap (8 blocks = sample unit); predictor "
                      "advantage is a processing proxy, never weight ancestry; "
                      "indistinguishable predictors => 'inconclusive', never a "
                      "positive Qwen-base classification"),
        },
        "wire": ("exact HTTP content bytes serialized ensure_ascii=False and "
                 "recorded per call; byte-match across styles verified at freeze"),
    }
    RUN_DIR_V3.mkdir(parents=True, exist_ok=True)
    PLAN_V3.write_text(jsonlib.dumps(plan, indent=1, ensure_ascii=False) + "\n",
                       encoding="utf-8")
    print(jsonlib.dumps({"frozen": str(PLAN_V3), "counts": plan["counts"]}, indent=1))
    return 0


def live_v3(limit: int = 0) -> int:
    load_credentials()
    os.environ.setdefault("JEVO_ALLOW_LIVE", "1")
    if not get_api_key():
        raise SystemExit("no TYPESAFE_API_KEY available; refusing live dispatch")
    if not PLAN_V3.exists():
        raise SystemExit("plan_v3_frozen.json missing; run freeze-v3 first")
    budget = PersistentBudget(LEDGER_PATH, fb.BATTERY_CAP_USD)
    done = {r.get("config_id") for r in _load_rows_v3()
            if r.get("family") == "fu_counter_v3"}
    calls = fb.build_fu_counter_v3()
    red = Redactor.from_environment()
    http = HttpxTransport(base_url=base_url(), api_key=get_api_key() or "",
                          timeout_seconds=180.0, redactor=red)
    cap = CapturingTransport(http._client, red)
    n = 0
    try:
        for call in calls:
            if call.config_id in done:
                continue
            if limit and n >= limit:
                break
            annotate_encode_v3(call)
            row, abort = _dispatch_v3(call, cap, budget)
            _append_row_v3(row)
            n += 1
            if abort:
                print("[fail-closed] auth/quota/price error; stopped", file=sys.stderr)
                return 2
    finally:
        http.close()
    print(jsonlib.dumps({"dispatched": n, "ledger": budget.snapshot()}, indent=1))
    return 0


def analyze_v3() -> int:
    rows = _load_rows_v3()
    summary = fb.counter_v3_summary(rows)
    REPORT_V3.mkdir(parents=True, exist_ok=True)
    (REPORT_V3 / "analysis_v3.json").write_text(
        jsonlib.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    # reference cost reconciliation (previous runs; report-only, no rescoring)
    recon = []
    rates = fb.REFERENCE_RATES_USD_PER_TOKEN
    for r in _load_rows():
        if r.get("family") != "fu_reference":
            continue
        slot = next((k for k, v in fb.REFERENCE_MODELS.items()
                     if v == r.get("model_requested")), None)
        if not slot or not isinstance(r.get("usage_prompt_tokens"), int):
            continue
        computed = (r["usage_prompt_tokens"] * rates[slot]["in"]
                    + (r.get("usage_completion_tokens") or 0) * rates[slot]["out"])
        reported = r.get("usage_cost_reported")
        if isinstance(reported, (int, float)) and abs(reported - computed) > max(1e-6, 0.2 * computed):
            recon.append({"item_id": r.get("item_id"),
                          "model": r.get("model_requested"),
                          "reported": reported, "computed": computed})
    recon_doc = {"n_reference_rows": sum(1 for r in _load_rows()
                                         if r.get("family") == "fu_reference"),
                 "mismatches_gt_20pct": recon,
                 "note": "report-only; old scores unchanged"}
    (REPORT_V3 / "reference_cost_reconciliation.json").write_text(
        jsonlib.dumps(recon_doc, indent=1) + "\n", encoding="utf-8")
    print(jsonlib.dumps({"v3": {k: summary[k] for k in
                                ("n_rows", "byte_match_verified", "best_predictor",
                                 "runner_up", "predictor_spread_ms",
                                 "joint_ms_per_1k")},
                         "reference_recon_mismatches": len(recon)},
                        indent=1, ensure_ascii=False)[:2500])
    return 0


if __name__ == "__main__":
    sys.exit(main())
