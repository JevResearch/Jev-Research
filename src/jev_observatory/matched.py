"""Matched modern-baseline comparison (parent MATCHED-RUN-GATE).

Three pieces in one focused module — freeze, dispatch, cost guard:

1. ``build_matched_freeze`` freezes the shared matched subset offline:
   1000 MMLU-Pro TEST items (proportional subject stratification, known
   inclusion probabilities, fixed seed 20260920), 300 ARC-Challenge TEST
   items (simple random sample, same seed family), and all 120 already-
   reviewed fresh TEST items (three families). Every matched item is
   verified against the corresponding Jev run artifacts (id + gold match)
   BEFORE the freeze is written, so the paired join is structural, not a
   best-effort glob. Write-once, integrity-checked on reload.

2. ``baseline_request`` serialises an item for the standard chat adapter.
   Choice items use the ACTUAL source option keys in source order
   (``openrouter.build_messages`` fix retained; wire order never sorted).
   Noul (bool) items are converted ONCE, explicitly labelled, to a
   two-option Yes/No choice with the SAME state and SAME instruction text;
   this is recorded as a distinct API encoding, never presented as the
   native Noul interface. The Jev 0.5 threshold on native Noul output is a
   separate recorded encoding (see matched_report).

3. ``MatchedCostGuard`` is the shared aggregate spend guard (default
   $75 hard cap including outstanding reservations at the configured
   output cap) across all nine models, priced from the pinned public
   OpenRouter catalog snapshot. A model whose catalog entry does not list
   ``reasoning_effort`` among supported_parameters is FLAGGED in the
   freeze and per attempt — the effort control may not bind there, and
   that is recorded truth, never silently retried differently.

Dispatch discipline (MatchedExecutor): ONE attempt per item, never a
retry on a wrong or malformed answer; at most 4 concurrent requests TOTAL
across providers with at most 2 per model; stop on 401/402/403, on
repeated transport/format failures, on budget denial, on wall-time expiry
or cancellation; resume never re-dispatches a finished logical request
(so the 10-item pilot is counted exactly once) and refuses to resume past
uncertain (timeout) attempts; budget accounting is restored from the
stored attempt ledger and never silently reset. Credentials come ONLY
from the process environment; live calls require JEVO_ALLOW_LIVE=1.
"""

from __future__ import annotations

import json
import random
import threading
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable

from .experiment import Item, LogicalRequest
from .guards import Cancellation, RateLimiter
from .manifest import utc_now
from .mech_dryrun import ordered_payload_json
from .openrouter import OpenRouterProvider, WireConfig, build_chat_payload
from .redact import Redactor
from .runner import Runner
from .schema import SystemOneRequest, estimate_input_tokens

# ------------------------------------------------------------------ constants
MATCHED_SEED = 20260920
N_MMLU_MATCHED = 1000
N_ARC_MATCHED = 300
PILOT_N = 10                      # 10 items/model x 9 models = 90 pilot calls
MAX_OUTPUT_TOKENS = 1024          # LOW-EFFORT bounded condition
REASONING_EFFORT = "low"
HARD_COST_CAP_USD = 75.0
GLOBAL_CONCURRENCY_MAX = 4        # TOTAL across providers
PER_MODEL_CONCURRENCY = 2
DEFAULT_RPM = 600
CONSECUTIVE_FAILURE_LIMIT = 3
FATAL_HTTP_STATUSES = frozenset({401, 402, 403})
FAILURE_STATUSES = frozenset({"http_error", "transport_error", "malformed_json",
                              "contract_invalid", "timeout"})

MATCHED_ROOT_DEFAULT = "runs_matched"
CATALOG_SNAPSHOT_PATH = "docs/modern-comparison/openrouter-catalog-nine-20260919.json"
SMOKE_ARTIFACT_PATH = "runs_benchmark/smoke/openrouter_smoke_20260919T230327Z.json"
FRESH_ITEMS_PATH = "runs_reviewed/rev-fresh_test-576d9a002eaa/items.jsonl"
FRESH_RESULTS_PATH = "runs_reviewed/rev-fresh_test-576d9a002eaa/results.jsonl"

# Nine requested modern models: the EXACT ids that passed the parent's live
# access smoke (scripts/benchmark/smoke_openrouter.py NINE_MODELS). A test
# asserts the two lists stay identical; requested ids are never replaced.
NINE_MODELS = [
    "openai/gpt-6-astra",
    "openai/gpt-5.6-sol",
    "anthropic/claude-fable-5.1",
    "anthropic/claude-opus-5",
    "z-ai/glm-5.3",
    "z-ai/glm-5.3-flash",
    "qwen/qwen3.8-max-0902",     # plain qwen3.8-max absent from catalog; snapshot pinned
    "qwen/qwen3.8-flash",
    "deepseek/deepseek-v4-flash-0731",
]

DATASETS = ("mmlu", "arc", "fresh")
YES = "Yes"
NO = "No"
NOUL_CONVERSION_LABEL = "noul_bool_to_yes_no_choice"
NOUL_CONVERSION_NOTE = (
    "native-Noul items are dispatched to chat baselines as an explicit Yes/No "
    "two-option choice over the SAME state and instruction text; the gold bool "
    "maps True->Yes, False->No. This is a recorded, distinct API encoding — it "
    "is NOT the native Noul interface and no probabilities are claimed for the "
    "text baseline. Jev-side native Noul scoring uses its own 0.5 threshold, "
    "recorded separately."
)
YES_NO_CRITERIA = {
    YES: "the statement asked by the question is correct",
    NO: "the statement asked by the question is incorrect",
}
JEV_NOUL_THRESHOLD = 0.5

FREEZE_VERSION = "matched-freeze-1.0.0"
EXECUTOR_VERSION = "matched-executor-1.0.0"


class MatchedError(RuntimeError):
    """Fail-closed condition for matched preparation / dispatch."""


def model_slug(model: str) -> str:
    return model.replace("/", "__")


def run_id_for(freeze: dict[str, Any], model: str) -> str:
    return f"matched-{freeze['deterministic_sha256'][:12]}-{model_slug(model)}"


def wire_payload_json(payload: dict[str, Any]) -> str:
    """Outbound wire representation: insertion order preserved, keys NOT sorted."""
    return ordered_payload_json(payload)


def _file_sha256(path: Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def _deterministic_sha256(doc: dict[str, Any]) -> str:
    trimmed = {k: v for k, v in doc.items() if k not in ("created_at", "deterministic_sha256")}
    return sha256(json.dumps(trimmed, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":")).encode("utf-8")).hexdigest()


# ------------------------------------------------------------------- sampling
def sample_mmlu_matched(items: list[dict[str, Any]], *, n: int = N_MMLU_MATCHED,
                        seed: int = MATCHED_SEED) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Proportional subject-stratified sample with known inclusion probabilities.

    Allocation is proportional to subject population size (NOT equal), with a
    deterministic per-stratum simple random sample. Selection never observes
    any success/failure data. Per-subject counts, population sizes,
    inclusion probabilities n_h/N_h and weights N_h/n_h are recorded.
    """
    from .datasets.mmlu_pro import stratified_pilot, stratum_sizes

    if n > len(items):
        raise MatchedError(f"mmlu sample {n} exceeds population {len(items)}")
    population_sizes = stratum_sizes(items)
    sample = stratified_pilot(items, n=n, seed=seed)
    sampled: dict[str, int] = {}
    for item in sample:
        sampled[str(item["group"])] = sampled.get(str(item["group"]), 0) + 1
    per_subject = {}
    for name in sorted(population_sizes):
        n_h = sampled.get(name, 0)
        N_h = population_sizes[name]
        per_subject[name] = {
            "population": N_h,
            "sampled": n_h,
            "inclusion_probability": (n_h / N_h) if N_h else None,
            "weight": (N_h / n_h) if n_h else None,
        }
    meta = {
        "mode": "proportional_stratified_srs",
        "seed": seed,
        "n": n,
        "population_sizes": population_sizes,
        "per_subject": per_subject,
        "note": ("allocation proportional to subject size with deterministic "
                 "per-stratum SRS; headline estimation reweights to the full "
                 "TEST population via recorded weights when allocation differs"),
    }
    return sample, meta


def sample_arc_matched(items: list[dict[str, Any]], *, n: int = N_ARC_MATCHED,
                       seed: int = MATCHED_SEED) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Simple random sample of the ARC-Challenge TEST split (same seed family)."""
    if n > len(items):
        raise MatchedError(f"arc sample {n} exceeds population {len(items)}")
    ordered = sorted(items, key=lambda it: it["id"])
    rng = random.Random(f"matched:{seed}")
    picked = rng.sample(range(len(ordered)), n)
    sample = [ordered[i] for i in sorted(picked)]
    meta = {
        "mode": "simple_random_sample",
        "seed_family": f"matched:{seed}",
        "n": n,
        "population": len(ordered),
        "inclusion_probability": n / len(ordered),
    }
    return sample, meta


def load_fresh_matched(path: str | Path = FRESH_ITEMS_PATH) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The 120 already-reviewed fresh TEST items, all three families, verbatim."""
    items = []
    for line in Path(path).read_text(encoding="utf-8").split("\n"):
        if line.strip():
            items.append(json.loads(line))
    families: dict[str, int] = {}
    for item in items:
        families[str(item.get("group"))] = families.get(str(item.get("group")), 0) + 1
    if set(families) != {"fresh_v3_graph", "fresh_v3_missing_info", "fresh_v3_dates"}:
        raise MatchedError(f"fresh items lack the three reviewed families: {families}")
    meta = {
        "mode": "full_reviewed_set",
        "source_file": str(path),
        "file_sha256": _file_sha256(Path(path)),
        "families": families,
        "n": len(items),
    }
    return items, meta


# ------------------------------------------------------- baseline serialisation
def baseline_questions(item: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """Questions for the chat baseline, with explicit Noul->Yes/No conversion.

    Choice questions pass through VERBATIM (actual source option keys, source
    order). Noul questions are converted once to the recorded Yes/No choice
    encoding; the conversion label is returned so every consumer can label it.
    """
    raw = item["questions"]
    converted: dict[str, Any] = {}
    conversion: str | None = None
    for qid, question in raw.items():
        qtype = question.get("type")
        if qtype == "choice":
            converted[qid] = dict(question)
        elif qtype == "noul":
            converted[qid] = {
                "type": "choice",
                "instructions": question.get("instructions"),
                "criteria": dict(YES_NO_CRITERIA),
            }
            conversion = NOUL_CONVERSION_LABEL
        else:
            raise MatchedError(f"item {item.get('id')}: unsupported question type {qtype!r}")
    return converted, conversion


def baseline_item(item: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """Item dict with questions converted for the chat baseline (gold kept)."""
    converted, conversion = baseline_questions(item)
    new_item = {k: v for k, v in item.items() if k != "questions"}
    new_item["questions"] = converted
    return new_item, conversion


def build_baseline_request(item: dict[str, Any], *, model: str) -> tuple[SystemOneRequest, str | None]:
    """SystemOneRequest for the standard chat adapter (gold structurally excluded).

    Only {state, model, questions} can enter the request type, so gold can
    never reach the outbound payload. The request model field is the actual
    OpenRouter model id (distinct from the Jev pin, recorded as such).
    """
    converted, conversion = baseline_item(item)
    cleaned = {k: v for k, v in converted.items() if not k.startswith("_")}
    experiment_item = Item(
        item_id=str(cleaned["id"]),
        group=str(cleaned.get("group", "default")),
        condition=str(cleaned.get("condition", "native")),
        cluster=str(cleaned.get("cluster", cleaned["id"])),
        state=cleaned["state"],
        questions=cleaned["questions"],
        gold=cleaned.get("gold") or {},
        leakage_check=bool(cleaned.get("leakage_check", True)),
    )
    request = experiment_item.to_request(model)
    return request, conversion


def logical_id(model: str, dataset: str, item_id: str) -> str:
    """``{model}:{dataset}:{item_id}`` — first two components are colon-free."""
    return f"{model}:{dataset}:{item_id}"


def split_logical_id(lid: str) -> tuple[str, str, str]:
    parts = lid.split(":", 2)
    if len(parts) != 3 or not all(parts):
        raise MatchedError(f"malformed matched logical_request_id {lid!r}")
    return parts[0], parts[1], parts[2]


# ------------------------------------------------------------- catalog/pricing
def load_catalog_prices(catalog_path: str | Path = CATALOG_SNAPSHOT_PATH) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    """Per-model USD-per-token pricing from the pinned public catalog snapshot.

    OpenRouter publishes USD per TOKEN (e.g. ``0.00001`` = $10/Mtok). The
    returned dicts are per-token; unsupported-parameter flags are separate.
    """
    doc = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
    by_id = {entry["id"]: entry for entry in doc.get("selected", [])}
    prices: dict[str, dict[str, float]] = {}
    effort_supported: dict[str, bool] = {}
    for model in NINE_MODELS:
        entry = by_id.get(model)
        if entry is None:
            raise MatchedError(f"model {model} missing from pinned catalog snapshot")
        pricing = entry.get("pricing_per_token") or {}
        try:
            prompt = float(pricing["prompt"])
            completion = float(pricing["completion"])
        except (KeyError, TypeError, ValueError) as exc:
            raise MatchedError(f"model {model}: unusable catalog pricing ({exc})") from exc
        prices[model] = {"input_per_token": prompt, "output_per_token": completion}
        effort_supported[model] = "reasoning_effort" in entry.get("supported_parameters", [])
    flags = {
        "catalog_file": str(catalog_path),
        "catalog_sha256": _file_sha256(Path(catalog_path)),
        "fetched_at_utc": doc.get("fetched_at_utc"),
        "reasoning_effort_supported": effort_supported,
        "effort_flag_note": (
            "models without 'reasoning_effort' in supported_parameters still receive "
            "the recorded reasoning_effort key, but the control may not bind there; "
            "flagged per attempt, never silently re-dispatched differently"),
    }
    return prices, flags


def call_cost_usd(prices: dict[str, dict[str, float]], model: str, *,
                  input_tokens: int | None, output_tokens: int | None,
                  usage_raw: dict[str, Any] | None) -> tuple[float | None, float | None]:
    """(provider_reported, catalog_estimated) cost for one completed call.

    Provider-reported cost wins when OpenRouter supplies a numeric ``cost``.
    The estimate uses actual reported tokens; both are None when usage is
    missing (unknown usage is never counted as free).
    """
    reported = None
    if isinstance(usage_raw, dict):
        raw_cost = usage_raw.get("cost")
        if isinstance(raw_cost, (int, float)) and not isinstance(raw_cost, bool):
            reported = float(raw_cost)
    estimated = None
    rate = prices.get(model)
    if rate is not None and input_tokens is not None and output_tokens is not None:
        estimated = (input_tokens * rate["input_per_token"]
                     + output_tokens * rate["output_per_token"])
    return reported, estimated


def reservation_cost_usd(prices: dict[str, dict[str, float]], model: str, *,
                         est_input_tokens: int, max_output_tokens: int) -> float:
    """Pre-dispatch reservation: estimated input at catalog input price plus
    the FULL configured output cap at the output price (conservative)."""
    rate = prices[model]
    return (est_input_tokens * rate["input_per_token"]
            + max_output_tokens * rate["output_per_token"])


def smoke_calibration(smoke_path: str | Path = SMOKE_ARTIFACT_PATH,
                      smoke_payload_chars: int | None = None) -> dict[str, Any]:
    """Saved live smoke usage, replayed as text (never a model call).

    ``smoke_payload_chars`` is the byte length of the smoke chat payload the
    parent sent; chars/token ≈ payload_chars / smoke input tokens gives a
    modest calibration point for expected-spend estimates (recorded, never
    treated as a guarantee).
    """
    doc = json.loads(Path(smoke_path).read_text(encoding="utf-8"))
    if doc.get("mode") != "live":
        raise MatchedError(f"{smoke_path} is not a live smoke artifact")
    usage: dict[str, dict[str, int | None]] = {}
    for entry in doc.get("models", []):
        call = entry.get("chat_call") or {}
        usage[entry["model_requested"]] = {
            "input_tokens": call.get("usage", {}).get("input_tokens"),
            "output_tokens": call.get("usage", {}).get("output_tokens"),
            "status": call.get("status"),
        }
    missing = [m for m in NINE_MODELS if m not in usage]
    if missing:
        raise MatchedError(f"smoke artifact lacks models {missing}")
    failed = [m for m, u in usage.items() if u["status"] != "ok"]
    if failed:
        raise MatchedError(f"smoke artifact records non-ok status for {failed}")
    return {
        "smoke_file": str(smoke_path),
        "smoke_sha256": _file_sha256(Path(smoke_path)),
        "recorded_at": doc.get("recorded_at"),
        "usage": usage,
        "smoke_payload_chars": smoke_payload_chars,
    }


# -------------------------------------------------------------------- freeze
def build_matched_freeze(
    *,
    root: str | Path = MATCHED_ROOT_DEFAULT,
    mmlu_path: str | Path = "data/test-00000-of-00001.parquet",
    arc_path: str | Path = "data/arc_challenge_test.parquet",
    benchmark_root: str | Path = "runs_benchmark",
    fresh_items_path: str | Path = FRESH_ITEMS_PATH,
    fresh_results_path: str | Path = FRESH_RESULTS_PATH,
    catalog_path: str | Path = CATALOG_SNAPSHOT_PATH,
    smoke_path: str | Path = SMOKE_ARTIFACT_PATH,
    smoke_payload_chars: int | None = None,
    seed: int = MATCHED_SEED,
    n_mmlu: int = N_MMLU_MATCHED,
    n_arc: int = N_ARC_MATCHED,
    pilot_n: int = PILOT_N,
    models: list[str] | None = None,
    reasoning_effort: str = REASONING_EFFORT,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
    hard_cap_usd: float = HARD_COST_CAP_USD,
) -> dict[str, Any]:
    """Verify sources + Jev joins, sample, hash wire maps, write-once freeze.

    Offline only: no credentials, no network, no model calls.
    """
    models = list(models or NINE_MODELS)
    from .benchmark_exec import load_freeze, stage_run_id
    from .benchmark_spec import verify_data_sidecar

    mmlu_source = verify_data_sidecar(mmlu_path)
    arc_source = verify_data_sidecar(arc_path)
    from .datasets.mmlu_pro import load_mmlu_pro
    from .datasets.arc_challenge import load_arc_challenge

    mmlu_dataset = load_mmlu_pro(mmlu_path)
    arc_dataset = load_arc_challenge(arc_path)
    mmlu_items, mmlu_meta = sample_mmlu_matched(mmlu_dataset.items, n=n_mmlu, seed=seed)
    arc_items, arc_meta = sample_arc_matched(arc_dataset.items, n=n_arc, seed=seed)
    fresh_items, fresh_meta = load_fresh_matched(fresh_items_path)

    # --- structural item/label checks BEFORE any freeze is written ---
    for dataset, items in (("mmlu", mmlu_items), ("arc", arc_items), ("fresh", fresh_items)):
        _verify_items_scorable(dataset, items)

    # --- Jev join verification (matched items must exist in the Jev runs) ---
    jev_freeze = load_freeze(benchmark_root)
    jev_join: dict[str, Any] = {}
    for dataset, stage in (("mmlu", "mmlu_full"), ("arc", "arc_test")):
        jev_run_id = stage_run_id(jev_freeze, stage)
        jev_items = _read_jsonl(Path(benchmark_root) / jev_run_id / "items.jsonl")
        by_id = {str(rec["id"]): rec for rec in jev_items}
        ids = [str(it["id"]) for it in (mmlu_items if dataset == "mmlu" else arc_items)]
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise MatchedError(f"{dataset}: matched ids absent from Jev run {jev_run_id}: {missing[:5]}...")
        mismatched = [it["id"] for it in (mmlu_items if dataset == "mmlu" else arc_items)
                      if by_id[str(it["id"])].get("gold") != it.get("gold")]
        if mismatched:
            raise MatchedError(f"{dataset}: gold differs from Jev run for {mismatched[:5]}...")
        jev_join[dataset] = {
            "stage": stage, "run_id": jev_run_id,
            "items_sha256": _file_sha256(Path(benchmark_root) / jev_run_id / "items.jsonl"),
        }
    fresh_results = _read_jsonl(fresh_results_path)
    fresh_result_ids = {str(r["item_id"]) for r in fresh_results if r.get("terminal") is True}
    fresh_ids = {str(it["id"]) for it in fresh_items}
    if fresh_ids - fresh_result_ids:
        raise MatchedError(f"fresh: {len(fresh_ids - fresh_result_ids)} items lack terminal Jev results")
    jev_join["fresh"] = {
        "items_sha256": _file_sha256(Path(fresh_items_path)),
        "results_sha256": _file_sha256(Path(fresh_results_path)),
    }

    prices, catalog_flags = load_catalog_prices(catalog_path)
    calibration = smoke_calibration(smoke_path, smoke_payload_chars=smoke_payload_chars)

    # --- wire maps: payload hash + input estimate per (model, dataset, item) ---
    wire_sha256: dict[str, str] = {}
    est_input_tokens: dict[str, int] = {}
    request_order: dict[str, list[str]] = {}
    pilot_request_ids: dict[str, list[str]] = {}
    wire_conversions: dict[str, str] = {}
    wires = {m: WireConfig(model=m, reasoning_effort=reasoning_effort,
                           max_output_tokens=max_output_tokens) for m in models}
    for model in models:
        order: list[str] = []
        pilot: list[str] = []
        for dataset, items in (("mmlu", mmlu_items), ("arc", arc_items), ("fresh", fresh_items)):
            converted_here = 0
            for item in items:
                request, conversion = build_baseline_request(item, model=model)
                payload = build_chat_payload(request, wires[model])
                lid = logical_id(model, dataset, str(item["id"]))
                wire_sha256[lid] = sha256(wire_payload_json(payload).encode("utf-8")).hexdigest()
                est_input_tokens[lid] = estimate_input_tokens(
                    len(json.dumps(payload, ensure_ascii=False)))
                order.append(lid)
                if conversion is not None:
                    wire_conversions[lid] = conversion
                    converted_here += 1
            if dataset == "fresh" and converted_here == 0:
                raise MatchedError("fresh Noul conversion produced no converted requests")
        take = {"mmlu": 6, "arc": 2, "fresh": 2} if pilot_n == PILOT_N else None
        if take is not None:
            counts = {"mmlu": 0, "arc": 0, "fresh": 0}
            for lid in order:
                dataset = split_logical_id(lid)[1]
                if counts[dataset] < take[dataset]:
                    pilot.append(lid)
                    counts[dataset] += 1
        else:
            pilot = order[:pilot_n]
        if len(pilot) != pilot_n:
            raise MatchedError(f"pilot selection for {model}: {len(pilot)} != {pilot_n}")
        request_order[model] = order
        pilot_request_ids[model] = pilot

    n_calls_per_model = sum(len(request_order[m]) for m in models) // len(models)
    estimates = _estimate_expected_spend(prices, calibration, max_output_tokens,
                                         n_calls_per_model, est_input_tokens, models)

    doc: dict[str, Any] = {
        "freeze_version": FREEZE_VERSION,
        "executor_version": EXECUTOR_VERSION,
        "created_at": utc_now(),
        "seed": seed,
        "condition": {
            "name": "low-effort bounded direct answer",
            "reasoning_effort": reasoning_effort,
            "max_output_tokens": max_output_tokens,
            "protocol": "standard chat API, one attempt, strict exact-key parsing, no retries",
            "note": ("NOT reasoning-disabled and NOT best-capability; the low "
                     "reasoning effort and the 1024-token output cap are explicit "
                     "wire settings recorded per attempt"),
        },
        "models": models,
        "wire": {"reasoning_effort": reasoning_effort, "max_output_tokens": max_output_tokens},
        "datasets": {
            "mmlu": {"n": len(mmlu_items), "sampling": mmlu_meta,
                     "source": mmlu_source, "items": mmlu_items},
            "arc": {"n": len(arc_items), "sampling": arc_meta,
                    "source": arc_source, "items": arc_items},
            "fresh": {"n": len(fresh_items), "sampling": fresh_meta, "items": fresh_items},
        },
        "noul_conversion": {"label": NOUL_CONVERSION_LABEL, "note": NOUL_CONVERSION_NOTE,
                            "yes_key": YES, "no_key": NO},
        "jev_join": jev_join,
        "jev_context": {
            "native_full_scores": "runs_benchmark/bench-*/derived/score.json (DISTINCT records)",
            "published_5shot_cot": "historical_contextual only; NOT protocol-compatible with the direct native protocol",
        },
        "catalog": catalog_flags,
        "smoke_calibration": calibration,
        "cost_guard": {
            "hard_cap_usd": hard_cap_usd,
            "cap_scope": "aggregate across all nine models, billed + outstanding reservations",
            "reservation_basis": "estimated input tokens + FULL configured output cap at catalog output price",
            "estimates": estimates,
        },
        "pilot_n": pilot_n,
        "pilot_request_ids": pilot_request_ids,
        "request_order": request_order,
        "wire_sha256": wire_sha256,
        "est_input_tokens": est_input_tokens,
        "wire_conversions": wire_conversions,
        "execution_safety": {
            "concurrency_total_max": GLOBAL_CONCURRENCY_MAX,
            "concurrency_per_model_max": PER_MODEL_CONCURRENCY,
            "retries": "none",
            "resume": "finished logical requests never re-dispatched; uncertain (timeout) attempts refuse resume",
            "stop_on": ["401/402/403", "repeated transport/format failures",
                        "budget denial", "wall-time expiry", "cancellation"],
            "credential_source": "process environment only (OPENROUTER_API_KEY); JEVO_ALLOW_LIVE=1 opt-in",
        },
    }
    doc["deterministic_sha256"] = _deterministic_sha256(doc)

    root_path = Path(root)
    freeze_path = root_path / "freeze" / "matched_frozen.json"
    if freeze_path.exists():
        existing = json.loads(freeze_path.read_text(encoding="utf-8"))
        if existing.get("deterministic_sha256") == doc["deterministic_sha256"]:
            return existing
        raise MatchedError(
            "an existing matched freeze differs from the current build; refusing to "
            "overwrite or relabel frozen items (use a new --root)"
        )
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    freeze_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    return doc


def _verify_items_scorable(dataset: str, items: list[dict[str, Any]]) -> None:
    ids = [str(it["id"]) for it in items]
    if len(ids) != len(set(ids)):
        raise MatchedError(f"{dataset}: duplicate item ids")
    for item in items:
        gold_block = item.get("gold") or {}
        value = None
        if isinstance(gold_block, dict):
            for payload in gold_block.values():
                if isinstance(payload, dict) and payload.get("value") is not None:
                    value = payload["value"]
                    break
        if value is None:
            raise MatchedError(f"{dataset} item {item['id']}: missing gold value")
        questions, conversion = baseline_questions(item)
        qid = next(iter(questions))
        question = questions[qid]
        if question["type"] != "choice":  # pragma: no cover - guarded above
            raise MatchedError(f"{dataset} item {item['id']}: non-choice baseline question")
        keys = list(question["criteria"])
        if conversion is None:
            if str(value) not in keys:
                raise MatchedError(
                    f"{dataset} item {item['id']}: gold {value!r} is not one of the "
                    f"source option keys {keys}")
        else:
            expected = YES if value is True else NO if value is False else None
            if expected is None:
                raise MatchedError(f"{dataset} item {item['id']}: noul gold is not bool")
        state = item.get("state", "")
        if item.get("leakage_check", True) and isinstance(state, str):
            text = str(value)
            if len(text) >= 4 and text in state:
                raise MatchedError(
                    f"{dataset} item {item['id']}: gold text leaks into state")


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").split("\n")
            if line.strip()]


def _estimate_expected_spend(
    prices: dict[str, dict[str, float]],
    calibration: dict[str, Any],
    max_output_tokens: int,
    n_calls_per_model: int,
    est_input_tokens: dict[str, int],
    models: list[str],
) -> dict[str, Any]:
    """Expected + worst-case spend from catalog prices and saved smoke usage.

    Input is the CONSERVATIVE pre-dispatch estimate (1 token/char, which
    over-counts by roughly 4x); expected output per call is calibrated from
    the SAVED SMOKE usage (a tiny direct-answer probe, so it UNDER-counts
    reasoning-heavy responses); the worst case charges the full 1024-token
    output cap to every call. All three are recorded; the $75 hard cap and
    its outstanding reservations are the binding constraint.
    """
    chars = calibration.get("smoke_payload_chars")
    usage = calibration["usage"]
    per_model: dict[str, Any] = {}
    total_expected = total_worst = 0.0
    for model in models:
        rate = prices[model]
        smoke_in = usage[model]["input_tokens"]
        smoke_out = usage[model]["output_tokens"]
        if chars and smoke_in:
            per_model_estimates = [est_input_tokens[lid] for lid in est_input_tokens
                                   if lid.startswith(f"{model}:")]
            est_in = (sum(per_model_estimates) / len(per_model_estimates)
                      if per_model_estimates else None)
            per_call_in = (est_in or 0) * rate["input_per_token"]
        else:
            est_in = None
            per_call_in = 0.0
        per_call_out_expected = (smoke_out or 0) * rate["output_per_token"]
        per_call_out_worst = max_output_tokens * rate["output_per_token"]
        expected = n_calls_per_model * (per_call_in + per_call_out_expected)
        worst = n_calls_per_model * (per_call_in + per_call_out_worst)
        per_model[model] = {
            "price_input_per_mtok": round(rate["input_per_token"] * 1e6, 6),
            "price_output_per_mtok": round(rate["output_per_token"] * 1e6, 6),
            "smoke_input_tokens": smoke_in,
            "smoke_output_tokens": smoke_out,
            "estimated_input_tokens_per_call": round(est_in, 1) if est_in else None,
            "expected_usd": round(expected, 2),
            "worst_case_output_capped_usd": round(worst, 2),
        }
        total_expected += expected
        total_worst += worst
    return {
        "basis": ("catalog snapshot prices; input from the conservative 1-token-per-char "
                  "pre-dispatch estimate (over-counts ~4x); expected output calibrated "
                  "from saved smoke usage (under-counts reasoning); worst case "
                  "charges the full output cap to every call"),
        "n_calls_per_model": n_calls_per_model,
        "smoke_payload_chars": chars,
        "per_model": per_model,
        "total_expected_usd": round(total_expected, 2),
        "total_worst_case_usd": round(total_worst, 2),
        "hard_cap_usd": HARD_COST_CAP_USD,
        "note": ("estimates are not guarantees; provider-reported cost in usage_raw "
                 "is the billing authority when present; usage without cost falls "
                 "back to catalog token pricing"),
    }


def load_matched_freeze(root: str | Path = MATCHED_ROOT_DEFAULT) -> dict[str, Any]:
    path = Path(root) / "freeze" / "matched_frozen.json"
    if not path.exists():
        raise MatchedError(f"no matched freeze at {path}; run the offline preparation first")
    doc = json.loads(path.read_text(encoding="utf-8"))
    if _deterministic_sha256(doc) != doc.get("deterministic_sha256"):
        raise MatchedError(f"matched freeze {path} failed its integrity check")
    return doc


# ----------------------------------------------------------------- cost guard
@dataclass
class _Reservation:
    model: str
    cost_usd: float


class MatchedCostGuard:
    """Thread-safe shared aggregate spend guard across all nine models.

    Ceiling = billed (provider-reported or catalog-estimated) + still-
    reserved reservations. Attempts whose usage never arrives keep their
    reservation held (unknown usage is never counted as free).
    """

    def __init__(self, prices: dict[str, dict[str, float]], *,
                 hard_cap_usd: float = HARD_COST_CAP_USD,
                 max_output_tokens: int = MAX_OUTPUT_TOKENS,
                 max_requests: int | None = None) -> None:
        self.prices = prices
        self.hard_cap_usd = hard_cap_usd
        self.max_output_tokens = max_output_tokens
        self.max_requests = max_requests
        self._lock = threading.Lock()
        self.reserved_usd = 0.0
        self.billed_reported_usd = 0.0
        self.billed_estimated_usd = 0.0
        self.unknown_reserved_usd = 0.0
        self.n_dispatched = 0
        self.denials: list[str] = []

    def reserve(self, model: str, est_input_tokens: int) -> _Reservation:
        cost = reservation_cost_usd(self.prices, model, est_input_tokens=est_input_tokens,
                                    max_output_tokens=self.max_output_tokens)
        with self._lock:
            ceiling = (self.billed_reported_usd + self.billed_estimated_usd
                       + self.unknown_reserved_usd + self.reserved_usd + cost)
            if self.max_requests is not None and self.n_dispatched + 1 > self.max_requests:
                self.denials.append("max_requests")
                raise MatchedError(
                    f"budget limit max_requests: {self.n_dispatched + 1} would exceed "
                    f"{self.max_requests}")
            if ceiling > self.hard_cap_usd:
                self.denials.append("hard_cost_cap_usd")
                raise MatchedError(
                    f"hard cost cap ${self.hard_cap_usd:.2f} reached: ceiling with this "
                    f"reservation would be ${ceiling:.4f}; nothing more is dispatched")
            self.n_dispatched += 1
            self.reserved_usd += cost
            return _Reservation(model=model, cost_usd=cost)

    def release(self, reservation: _Reservation, *, keep_reserved: bool = False) -> None:
        with self._lock:
            self.reserved_usd = max(0.0, self.reserved_usd - reservation.cost_usd)
            if keep_reserved:
                self.unknown_reserved_usd += reservation.cost_usd

    def commit(self, reservation: _Reservation, *, input_tokens: int | None,
               output_tokens: int | None, usage_raw: dict[str, Any] | None) -> dict[str, float | None]:
        reported, estimated = call_cost_usd(
            self.prices, reservation.model, input_tokens=input_tokens,
            output_tokens=output_tokens, usage_raw=usage_raw)
        with self._lock:
            self.reserved_usd = max(0.0, self.reserved_usd - reservation.cost_usd)
            if reported is not None:
                self.billed_reported_usd += reported
            elif estimated is not None:
                self.billed_estimated_usd += estimated
            else:
                self.unknown_reserved_usd += reservation.cost_usd
        return {"reported": reported, "estimated": estimated}

    def ceiling(self) -> float:
        with self._lock:
            return (self.billed_reported_usd + self.billed_estimated_usd
                    + self.unknown_reserved_usd + self.reserved_usd)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "hard_cap_usd": self.hard_cap_usd,
                "ceiling_usd": round(self.billed_reported_usd + self.billed_estimated_usd
                                     + self.unknown_reserved_usd + self.reserved_usd, 4),
                "reserved_usd": round(self.reserved_usd, 4),
                "billed_reported_usd": round(self.billed_reported_usd, 4),
                "billed_estimated_usd": round(self.billed_estimated_usd, 4),
                "unknown_reserved_usd": round(self.unknown_reserved_usd, 4),
                "n_dispatched": self.n_dispatched,
                "denials": list(self.denials),
            }

    def restore(self, attempts: list[dict[str, Any]]) -> None:
        """Re-add committed costs from the stored attempt ledger (resume)."""
        for attempt in attempts:
            usage_raw = (attempt.get("extra") or {}).get("usage_raw") \
                if isinstance(attempt.get("extra"), dict) else None
            reported, estimated = call_cost_usd(
                self.prices, str(attempt.get("model_requested")),
                input_tokens=attempt.get("usage_input_tokens"),
                output_tokens=attempt.get("usage_output_tokens"),
                usage_raw=usage_raw)
            if reported is not None:
                self.billed_reported_usd += reported
            elif estimated is not None:
                self.billed_estimated_usd += estimated
            else:
                # unknown usage keeps a conservative reservation: full output cap
                est = int(attempt.get("estimated_input_tokens") or 1)
                self.unknown_reserved_usd += reservation_cost_usd(
                    self.prices, str(attempt.get("model_requested")),
                    est_input_tokens=est, max_output_tokens=self.max_output_tokens)


# ------------------------------------------------------------------- executor
class MatchedExecutor:
    """Freeze-verify -> dispatch the matched chat baseline (or plan offline).

    ``live=False`` (default) verifies freeze, request maps, label maps and
    wire hashes WITHOUT constructing any transport or provider.
    """

    def __init__(
        self,
        root: str | Path = MATCHED_ROOT_DEFAULT,
        *,
        freeze: dict[str, Any] | None = None,
        clock: Callable[[], float] | None = None,
        sleeper: Callable[[float], None] | None = None,
        models: list[str] | None = None,
        pilot_only: bool = False,
        hard_cap_usd: float | None = None,
        max_requests: int | None = None,
        max_wall_seconds: float | None = None,
        rpm: int = DEFAULT_RPM,
        global_concurrency: int = GLOBAL_CONCURRENCY_MAX,
        per_model_concurrency: int = PER_MODEL_CONCURRENCY,
        consecutive_failure_limit: int = CONSECUTIVE_FAILURE_LIMIT,
    ) -> None:
        self.root = Path(root)
        self.freeze = freeze
        self.clock = clock or time.monotonic
        self.sleeper = sleeper or time.sleep
        self.pilot_only = pilot_only
        self.hard_cap_usd = hard_cap_usd if hard_cap_usd is not None else HARD_COST_CAP_USD
        self.max_wall_seconds = max_wall_seconds
        self.rpm = rpm
        self.global_concurrency = max(1, int(global_concurrency))
        self.per_model_concurrency = max(1, int(per_model_concurrency))
        self.consecutive_failure_limit = consecutive_failure_limit
        self.cancellation = Cancellation()
        self.stop_reason: str | None = None
        self.guard: MatchedCostGuard | None = None
        self._consecutive_failures = 0
        self._state_lock = threading.Lock()
        if self.global_concurrency > GLOBAL_CONCURRENCY_MAX:
            raise MatchedError(
                f"global concurrency {self.global_concurrency} exceeds the "
                f"{GLOBAL_CONCURRENCY_MAX}-request TOTAL cap across providers")
        if self.per_model_concurrency > PER_MODEL_CONCURRENCY:
            raise MatchedError(
                f"per-model concurrency {self.per_model_concurrency} exceeds "
                f"{PER_MODEL_CONCURRENCY}")
        self.models = list(models) if models is not None else None
        if self.models:
            known = set(self.freeze["models"]) if self.freeze else set(NINE_MODELS)
            unknown = [m for m in self.models if m not in known]
            if unknown:
                raise MatchedError(f"models not in the freeze: {unknown}")

    # ------------------------------------------------------------- top level
    def run(self, *, live: bool) -> dict[str, Any]:
        from concurrent.futures import ThreadPoolExecutor

        freeze = self.freeze or load_matched_freeze(self.root)
        self.freeze = freeze
        if self.models is None:
            self.models = list(freeze["models"])
        hard_cap = freeze["cost_guard"]["hard_cap_usd"]
        if self.hard_cap_usd != hard_cap:
            raise MatchedError(
                f"hard cap ${self.hard_cap_usd} differs from the frozen "
                f"${hard_cap}; the frozen guard is authoritative")
        prices, _ = load_catalog_prices()
        self.guard = MatchedCostGuard(
            prices, hard_cap_usd=self.hard_cap_usd,
            max_output_tokens=int(freeze["wire"]["max_output_tokens"]),
            max_requests=self.max_requests_cap(freeze),
        )
        state = self._state(freeze, live=live)
        self._verify_freeze_against_code(freeze)
        per_model_targets = {
            model: (freeze["pilot_request_ids"][model] if self.pilot_only
                    else freeze["request_order"][model])
            for model in self.models
        }
        tasks = {model: self._build_tasks(freeze, model, per_model_targets[model])
                 for model in self.models}
        if live:
            import os

            if os.environ.get("JEVO_ALLOW_LIVE") != "1":
                raise MatchedError("live dispatch requires environment JEVO_ALLOW_LIVE=1")
            if not os.environ.get("OPENROUTER_API_KEY"):
                raise MatchedError(
                    "live dispatch requires OPENROUTER_API_KEY in the process "
                    "environment (never echoed)")
            # fail-closed resume gates BEFORE budget restoration
            for model in self.models:
                store = self._store(run_id_for(freeze, model))
                uncertain = store.uncertain_ids()
                if uncertain:
                    raise MatchedError(
                        f"fail-closed resume refusal for {model}: {len(uncertain)} "
                        "uncertain (timeout) attempt(s); the provider may have billed "
                        "unobserved work. This executor does not resume past ambiguous "
                        "timeouts: start a NEW freeze. Budgets were not reset.")
                restored_attempts = store.attempts()
                before = self.guard.snapshot()
                self.guard.restore(restored_attempts)
                if self.guard.ceiling() > self.hard_cap_usd:
                    raise MatchedError(
                        f"fail-closed resume refusal for {model}: restored accounting "
                        f"from stored attempts exceeds the ${self.hard_cap_usd:.2f} cap "
                        f"({before} -> {self.guard.snapshot()}); budgets were not reset")
            stores = {m: self._store(run_id_for(freeze, m)) for m in self.models}
            state["budget_restored_from_attempts"] = {
                "attempts": sum(len(stores[m].attempts()) for m in self.models)}
            deadline = self.clock() + self.max_wall_seconds if self.max_wall_seconds else None
            providers = {m: self._new_provider(freeze, m) for m in self.models}
            rate = RateLimiter(requests_per_minute=self.rpm,
                               clock=self.clock, sleeper=self.sleeper)
            model_semaphores = {m: threading.Semaphore(self.per_model_concurrency)
                                for m in self.models}
            pending: list[tuple[str, Any]] = []
            for model in self.models:
                completed = stores[model].completed_ids()
                state["stages"][model]["skipped_resume"] = 0
                for task in tasks[model]:
                    if task[0] in completed:
                        state["stages"][model]["skipped_resume"] += 1
                        continue
                    pending.append((model, task))
            state["n_pending"] = len(pending)
            tally_lock = threading.Lock()

            def worker(model: str, task: Any) -> None:
                if self.stop_reason is not None or self.cancellation.cancelled:
                    return
                if deadline is not None and self.clock() > deadline:
                    self._stop("wall_time_budget_exceeded")
                    return
                with model_semaphores[model]:
                    if self.stop_reason is not None or self.cancellation.cancelled:
                        return
                    counts = self._dispatch(freeze, model, task, providers[model],
                                            rate, stores[model])
                if counts is None:
                    return
                with tally_lock:
                    stage = state["stages"][model]
                    stage["dispatched"] += 1
                    stage["ok"] += counts["ok"]
                    stage["error"] += counts["error"]

            try:
                with ThreadPoolExecutor(max_workers=self.global_concurrency) as pool:
                    futures = [pool.submit(worker, m, t) for m, t in pending]
                    for future in futures:
                        future.result()
            except MatchedError as exc:
                self._stop(f"budget_cap_reached:{exc}")
                state["executor_error"] = str(exc)
            finally:
                for provider in providers.values():
                    provider.close()
        for model in self.models:
            state["stages"][model]["guard_snapshot"] = self.guard.snapshot()
            state["stages"][model]["stop_reason"] = self.stop_reason
        state["guard_final"] = self.guard.snapshot()
        state["global_stop_reason"] = self.stop_reason
        state["consecutive_failures_at_stop"] = self._consecutive_failures
        if not live:
            state["note"] = ("dry run: freeze verified + all item/label/wire maps "
                             "checked; nothing dispatched, no provider constructed")
        self._write_state(state)
        return state

    def max_requests_cap(self, freeze: dict[str, Any]) -> int | None:
        n_per_model = len(freeze["request_order"][freeze["models"][0]])
        if self.pilot_only:
            return len(self.models) * freeze["pilot_n"]
        return len(self.models) * n_per_model

    def _new_provider(self, freeze: dict[str, Any], model: str) -> OpenRouterProvider:
        from .openrouter import OpenRouterHttpTransport

        wire = WireConfig(model=model, reasoning_effort=freeze["wire"]["reasoning_effort"],
                          max_output_tokens=int(freeze["wire"]["max_output_tokens"]))
        transport = OpenRouterHttpTransport(get_env_api_key() or "")
        return OpenRouterProvider(transport, wire=wire, redactor=Redactor.from_environment())

    def _state(self, freeze: dict[str, Any], *, live: bool) -> dict[str, Any]:
        return {
            "executor_version": EXECUTOR_VERSION,
            "live": bool(live),
            "pilot_only": bool(self.pilot_only),
            "freeze_sha256": freeze["deterministic_sha256"],
            "models_run": list(self.models) if self.models else list(freeze["models"]),
            "caps": {
                "hard_cost_cap_usd": self.hard_cap_usd,
                "concurrency_total": self.global_concurrency,
                "concurrency_per_model": self.per_model_concurrency,
                "rpm": self.rpm,
                "max_wall_seconds": self.max_wall_seconds,
                "retries": "none",
            },
            "stages": {model: {"dispatched": 0, "ok": 0, "error": 0, "skipped_resume": 0,
                               "run_id": run_id_for(freeze, model)}
                       for model in (self.models or freeze["models"])},
            "global_stop_reason": None,
            "recorded_at": utc_now(),
        }

    # ------------------------------------------------------------- verification
    def _verify_freeze_against_code(self, freeze: dict[str, Any]) -> None:
        """Offline check: rebuilt requests/hashes must equal the stored maps.

        Proves the frozen wire (payload bytes, actual source option keys, order)
        reloads identically after the freeze round-trip.
        """
        wires = {m: WireConfig(model=m, reasoning_effort=freeze["wire"]["reasoning_effort"],
                               max_output_tokens=int(freeze["wire"]["max_output_tokens"]))
                 for m in freeze["models"]}
        for model in (self.models or freeze["models"]):
            order = freeze["request_order"][model]
            if len(order) != len(set(order)):
                raise MatchedError(f"{model}: duplicate logical ids in request order")
            expected_total = sum(freeze["datasets"][d]["n"] for d in DATASETS)
            if len(order) != expected_total:
                raise MatchedError(f"{model}: request order covers {len(order)} of "
                                   f"{expected_total} frozen items")
            for dataset in DATASETS:
                for item in freeze["datasets"][dataset]["items"]:
                    _verify_items_scorable(dataset, [item])
                    request, conversion = build_baseline_request(item, model=model)
                    lid = logical_id(model, dataset, str(item["id"]))
                    payload = build_chat_payload(request, wires[model])
                    got = sha256(wire_payload_json(payload).encode("utf-8")).hexdigest()
                    if got != freeze["wire_sha256"].get(lid):
                        raise MatchedError(
                            f"{lid}: rebuilt wire hash differs from the frozen map; "
                            "refusing to run against a drifted serializer")
                    est = estimate_input_tokens(len(json.dumps(payload, ensure_ascii=False)))
                    if est != freeze["est_input_tokens"].get(lid):
                        raise MatchedError(f"{lid}: input estimate drifted from the freeze")
                    if (conversion is not None) != (freeze["wire_conversions"].get(lid) is not None):
                        raise MatchedError(f"{lid}: conversion label drifted from the freeze")
            pilot_ids = freeze["pilot_request_ids"][model]
            if self.pilot_only and len(pilot_ids) != freeze["pilot_n"]:
                raise MatchedError(f"{model}: pilot size drifted from the freeze")
            if any(lid not in order for lid in pilot_ids):
                raise MatchedError(f"{model}: pilot ids not all in the request order")

    def _build_tasks(self, freeze: dict[str, Any], model: str,
                     lids: list[str]) -> list[tuple[str, str, LogicalRequest]]:
        by_dataset: dict[str, dict[str, dict[str, Any]]] = {
            d: {str(it["id"]): it for it in freeze["datasets"][d]["items"]}
            for d in DATASETS
        }
        tasks: list[tuple[str, str, LogicalRequest]] = []
        for lid in lids:
            _model, dataset, item_id = split_logical_id(lid)
            item = by_dataset[dataset][item_id]
            request, conversion = build_baseline_request(item, model=model)
            if conversion is not None and freeze["wire_conversions"].get(lid) != conversion:
                raise MatchedError(f"{lid}: conversion label drifted from the freeze")
            lr = LogicalRequest(
                logical_request_id=lid,
                item=Item(item_id=item_id, group=str(item.get("group", "default")),
                          condition=dataset, cluster=str(item.get("cluster", item_id)),
                          state=item["state"],
                          questions=baseline_questions(item)[0],
                          gold=item.get("gold") or {},
                          leakage_check=bool(item.get("leakage_check", True))),
                request=request,
                condition=dataset,
            )
            tasks.append((lid, dataset, lr))
        return tasks

    def _store(self, run_id: str):
        from .ledger import RunStore

        return RunStore(self.root, run_id, redactor=Redactor.from_environment())

    # ---------------------------------------------------------------- dispatch
    def _dispatch(self, freeze: dict[str, Any], model: str, task: tuple[str, str, LogicalRequest],
                  provider: OpenRouterProvider, rate: RateLimiter, store) -> dict[str, int] | None:
        lid, dataset, lr = task
        if self.cancellation.cancelled or self.stop_reason is not None:
            return None
        est = int(freeze["est_input_tokens"][lid])
        reservation = self.guard.reserve(model, est)
        try:
            rate.acquire(est, self.cancellation)
        except BaseException:
            self.guard.release(reservation)
            raise
        try:
            outcome = provider.ask(lr.request, lid, cancellation=self.cancellation, store=store)
        except Exception as exc:  # provider explosions are recorded, never raised past here
            self.guard.release(reservation, keep_reserved=True)
            self._observe_failure_exception()
            return None
        attempt = outcome.attempts[-1] if outcome.attempts else None
        usage_raw = (attempt.extra or {}).get("usage_raw") if attempt else None
        if outcome.status == "http_error":
            self.guard.release(reservation)
        elif outcome.status == "timeout":
            self.guard.release(reservation, keep_reserved=True)
        else:
            self.guard.commit(reservation, input_tokens=outcome.usage_input_tokens,
                              output_tokens=outcome.usage_output_tokens,
                              usage_raw=usage_raw)
        for record in outcome.attempts:
            store.append_attempt(record.to_dict())
        if outcome.terminal and outcome.status not in {"cancelled", "timeout"}:
            store.append_result(Runner._result_record(None, outcome, lr))
        self._observe_outcome(outcome)
        return {"ok": 1 if outcome.status == "ok" else 0, "error": 0 if outcome.status == "ok" else 1}

    def _stop(self, reason: str) -> None:
        with self._state_lock:
            if self.stop_reason is None:
                self.stop_reason = reason
        self.cancellation.cancel()

    def _observe_outcome(self, outcome: Any) -> None:
        stop: str | None = None
        with self._state_lock:
            http_status = outcome.attempts[-1].http_status if outcome.attempts else None
            if http_status in FATAL_HTTP_STATUSES:
                stop = "insufficient_credit_or_auth"
            elif outcome.status == "ok":
                self._consecutive_failures = 0
            elif outcome.status in FAILURE_STATUSES:
                self._consecutive_failures += 1
                if self._consecutive_failures >= self.consecutive_failure_limit:
                    stop = "repeated_transport_schema_failures"
        if stop is not None:
            self._stop(stop)

    def _observe_failure_exception(self) -> None:
        with self._state_lock:
            self._consecutive_failures += 1
            stop = self._consecutive_failures >= self.consecutive_failure_limit
        if stop:
            self._stop("provider_exception_failures")

    def _write_state(self, state: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "executor_state.json").write_text(
            json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def get_env_api_key() -> str | None:
    """Process-environment key only; never copied, logged or stored."""
    import os

    return os.environ.get("OPENROUTER_API_KEY") or None
