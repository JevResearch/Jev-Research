"""Full-benchmark suite builder (parent BENCHMARK-EXECUTION-GATE, 2026-09-19).

Builds the FULL benchmark suite for the Jev executor — not a pilot:

1. ``mmlu_full``        — ALL MMLU-Pro TEST items (12,032), full-set sampling,
                          native option order preserved (never sorted);
2. ``option_rotations`` — a frozen 140-item subject-stratified subset of that
                          SAME TEST split (10 per subject) × 3 deterministic
                          option rotations (420 additional items). Robustness
                          data only: reported separately, never best-of and
                          never silently averaged into the headline accuracy;
3. ``arc_test``         — the ARC-Challenge TEST split (1,172 items), public
                          parquet export; train/validation are never used.

Hard guarantees (gate items 1–3):

* gold stays ONLY in the spec/offline join layer: outbound payloads can contain
  just {state, model, questions} (enforced again by experiment.logical_requests);
* original answer/option mappings are kept; rotations move descriptions
  between canonical keys and remap the gold, and the remapping is recorded;
* the full split counts are enforced at build time — a short or long source
  file is a hard error, never a silent subset;
* the request ceiling 15,000 is checked (12,032 + 420 + 1,172 = 13,624).

All sources are verified against the sha256 sidecars next to the pinned files
before any spec is built. GPQA Diamond is NOT built here: canonical access is
gated and no token is supplied — it stays an explicit access dependency.
"""

from __future__ import annotations

import random
from hashlib import sha256
from pathlib import Path
from typing import Any

from .datasets.arc_agi2_grid import (
    ArcAgi2Error,
    build_spec as build_arc_agi2_spec,
    load_arc_agi2_public_eval,
)
from .datasets.arc_challenge import build_spec as build_arc_spec
from .datasets.arc_challenge import load_arc_challenge
from .datasets.base import DatasetProvenance, LoadedDataset, read_jsonl
from .datasets.gpqa_diamond import (
    GpqaError,
    build_spec as build_gpqa_spec,
    load_gpqa_diamond,
)
from .datasets.hle_text_mc import (
    HleError,
    build_spec as build_hle_spec,
    load_hle,
)
from .datasets.math500 import (
    Math500Error,
    build_spec as build_math500_spec,
    load_math500,
)
from .datasets.mmlu_pro import build_spec as build_mmlu_spec
from .datasets.mmlu_pro import load_mmlu_pro
from .datasets.permutation import permute_choice_item

# ------------------------------------------------------------------ constants
N_MMLU_PRO_TEST = 12032
N_ARC_CHALLENGE_TEST = 1172
ROTATION_SUBJECTS = 10          # items per subject in the frozen subset
ROTATION_VARIANTS = 3           # deterministic rotations, in addition to native
N_ROTATION_ITEMS = 140 * ROTATION_VARIANTS  # 420
REQUEST_CEILING = 15000         # MMLU 12032 + rotations 420 + ARC 1172 = 13624
ROTATION_SEED = 7701            # frozen; recorded in every derived artifact

STAGES = ("mmlu_full", "option_rotations", "arc_test")
STAGE_TITLES = {
    "mmlu_full": "MMLU-Pro TEST, full 12032 (native option order)",
    "option_rotations": "option-order audit 140x3 (10/subject x 3 rotations)",
    "arc_test": "ARC-Challenge TEST, full 1172 (dataset labels preserved)",
}

# --- late-2026 extension stages (2026-09-20) --------------------------------
#
# Rationale (see docs/modern-comparison/EXTENDED-BENCHMARK-GATE.md): MMLU-Pro
# and ARC-Challenge are 2024-era anchors with saturated, dated reference
# protocols.  The extension adds two MODERN (2025-2026) benchmarks in
# Jev-native encodings:
#
# * arc_agi2_*    — ARC-AGI-2 PUBLIC evaluation split (the flagship 2025-2026
#   fluid-intelligence benchmark).  Verified frontier references are the
#   SEMI-PRIVATE set (120 tasks); the public split is the openly published
#   counterpart (ARC Prize agreement policy: within +/-3pp).  Three encodings:
#   per-cell choice, per-cell 10-level score, whole-task requests.
# * math500_*     — MATH-500 (public 500-item subset of MATH TEST), in two
#   Jev-native encodings of the exact-match answer.  Saturated for reasoning
#   models; used as a DIRECT-ANSWER longitudinal anchor (reference protocols
#   labelled; Jev runs no CoT and no reasoning).
#
# Request math (kept under the 15,000 ceiling):
#   arc_agi2_choice  167 grids (1 request per grid)
#   arc_agi2_score   167 grids (1 request per grid)
#   arc_agi2_task    120 tasks (1 request per task)
#   math500_choice   311 items (of 500; rest excluded at build time)
#   math500_score    273 items
#   extension total  1,038 requests -> suite total 14,662 <= 15,000.

ARC_AGI2_TARBALL_DEFAULT = "data/arc_agi2_public_eval_f3283f7.tar.gz"
MATH500_JSONL_DEFAULT = "data/math500_test.jsonl"
ARC_AGI2_COMMIT = "f3283f727488ad98fe575ea6a5ac981e4a188e49"
N_ARC_AGI2_EVAL_TASKS = 120
N_MATH500_TEST = 500
EXTENSION_STAGES = ("arc_agi2_choice", "arc_agi2_score", "arc_agi2_task",
                    "math500_choice", "math500_score")

# --- gated-access stages (2026-09-20, tokens supplied + gates accepted) -----
# GPQA Diamond (Idavidrein/gpqa) and HLE text-only MC (cais/hle): downloaded
# with the user's HF token, sha256-sidecarred, gates accepted; items never
# redistributed.  Same native Choice protocol, direct answer, no CoT.
GPQA_CSV_DEFAULT = "data/gpqa_diamond.csv"
HLE_PARQUET_DEFAULT = "data/hle_test.parquet"
N_GPQA_DIAMOND = 198
N_HLE_TEST_ROWS = 2500
ACCESS_STAGES = ("gpqa_diamond", "hle_text_mc")

EXTENDED_STAGES = STAGES + EXTENSION_STAGES + ACCESS_STAGES
EXTENDED_STAGE_TITLES = {
    **STAGE_TITLES,
    "arc_agi2_choice": "ARC-AGI-2 public eval, per-cell choice (167 grids)",
    "arc_agi2_score": "ARC-AGI-2 public eval, per-cell 10-level score (167 grids)",
    "arc_agi2_task": "ARC-AGI-2 public eval, whole-task requests (120 tasks)",
    "math500_choice": "MATH-500 numeric 4-option MCQ (build-time encodable subset)",
    "math500_score": "MATH-500 per-digit score rubric (values < 1000)",
    "gpqa_diamond": "GPQA Diamond full split (seeded option shuffle; direct answer)",
    "hle_text_mc": "HLE text-only multiple-choice sub-track (native option order)",
}

# Overridable ONLY for small offline fixtures; production runs use the defaults.
DEFAULT_EXPECTED = {
    "mmlu_full": N_MMLU_PRO_TEST,
    "arc_test": N_ARC_CHALLENGE_TEST,
    "arc_agi2": N_ARC_AGI2_EVAL_TASKS,
    "math500": N_MATH500_TEST,
    "gpqa_diamond": N_GPQA_DIAMOND,
    "hle_rows": N_HLE_TEST_ROWS,
}


class BenchmarkSpecError(RuntimeError):
    """Fail-closed suite-build condition (counts, hashes, ids, ceilings)."""


# ------------------------------------------------------------------ sidecars
def verify_data_sidecar(path: Path | str) -> dict[str, Any]:
    """Verify a pinned data file against its ``<file>.sha256.json`` sidecar.

    Returns the sidecar record (file, sha256, url). A mismatch or a missing
    sidecar is a hard error — nothing downstream may run on unverified bytes.
    """
    file_path = Path(path)
    sidecar_path = Path(str(file_path) + ".sha256.json")
    if not file_path.exists():
        raise BenchmarkSpecError(f"missing data file {file_path}")
    if not sidecar_path.exists():
        raise BenchmarkSpecError(f"missing sha256 sidecar {sidecar_path}")
    record = _read_json(sidecar_path)
    for key in ("sha256", "url"):
        if not record.get(key):
            raise BenchmarkSpecError(f"sidecar {sidecar_path} missing {key!r}")
    actual = sha256(file_path.read_bytes()).hexdigest()
    if actual != record["sha256"]:
        raise BenchmarkSpecError(
            f"{file_path}: sha256 {actual} does not match sidecar {record['sha256']}; "
            "refusing to build specs over unverified data"
        )
    return {"file": str(file_path), "sha256": record["sha256"], "url": record["url"]}


def _read_json(path: Path) -> dict[str, Any]:
    import json

    return json.loads(Path(path).read_text(encoding="utf-8"))


# -------------------------------------------------------------------- subset
def subject_stratified_subset(
    items: list[dict[str, Any]], *, per_subject: int = ROTATION_SUBJECTS,
    seed: int = ROTATION_SEED, category_field: str = "group",
) -> list[dict[str, Any]]:
    """Deterministic subject-stratified subset (default 10 per subject).

    Members are ordered by id within each subject before sampling, so the
    subset is reproducible from the full TEST split + seed alone. Output is
    sorted by (subject, id).
    """
    strata: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        strata.setdefault(str(item.get(category_field, "unknown")), []).append(item)
    rng = random.Random(f"rotation-subset:{seed}")
    subset: list[dict[str, Any]] = []
    for name in sorted(strata):
        members = sorted(strata[name], key=lambda it: it["id"])
        if len(members) < per_subject:
            raise BenchmarkSpecError(
                f"subject {name!r} has {len(members)} items, fewer than {per_subject}"
            )
        picked = rng.sample(range(len(members)), per_subject)
        subset.extend(members[i] for i in picked)
    if len(subset) != per_subject * len(strata):
        raise BenchmarkSpecError("subset size drifted; refusing")
    subset.sort(key=lambda it: (str(it.get(category_field, "unknown")), it["id"]))
    return subset


def build_rotation_items(
    subset_items: list[dict[str, Any]], *, seed: int = ROTATION_SEED,
    variants: int = ROTATION_VARIANTS,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Deterministic option rotations of the frozen subset.

    Returns (items, rotations) where ``rotations[id]`` records the base item,
    the variant, and the full key→key remapping. The gold follows its
    description to the new key (permutation.py), so a rotated item's gold is
    already the canonical answer under the new labelling — and the scorer
    additionally restores canonical labels through the remapping as a check.
    """
    items: list[dict[str, Any]] = []
    rotations: dict[str, dict[str, Any]] = {}
    for base in subset_items:
        for variant in range(1, variants + 1):
            result = permute_choice_item(base, seed=seed, variant=variant)
            item = result["item"]
            # Colon-free condition (perm1..permN): logical_request_id is
            # "{condition}:{item_id}" and rotation item ids themselves contain
            # a colon, so the condition must be unambiguous to split back out.
            item["condition"] = f"perm{variant}"
            item["group"] = base.get("group", "unknown")  # keep subject breakdown
            remapping = {k: v for k, v in result["remapping"].items()}
            if sorted(remapping.values()) != sorted(remapping.keys()):
                raise BenchmarkSpecError(
                    f"rotation for {base['id']} variant {variant} is not a permutation"
                )
            rotations[item["id"]] = {
                "base_item": base["id"],
                "variant": variant,
                "remapping": remapping,
            }
            items.append({k: v for k, v in item.items() if not k.startswith("_")})
    return items, rotations


# --------------------------------------------------------------------- suite
def build_extension_stages(
    *,
    arc_agi2_tar: Path | str = ARC_AGI2_TARBALL_DEFAULT,
    math500_path: Path | str = MATH500_JSONL_DEFAULT,
    model: str = "jev-1.13.0",
    skip_arc_agi2: bool = False,
    skip_math500: bool = False,
    arc_agi2_expected: int | None = None,
    math500_expected: int | None = None,
) -> dict[str, Any]:
    """Build the late-2026 extension stages (ARC-AGI-2 + MATH-500).

    Sidecar-verified sources, full-split counts enforced at build time, gold
    structurally excluded from outbound payloads (same guarantees as the core
    suite).  Extension requests are within the same 15,000 ceiling; the caller
    re-checks the ceiling after merging with the core suite.
    """
    suite: dict[str, Any] = {}
    if not skip_arc_agi2:
        agi2_source = verify_data_sidecar(arc_agi2_tar)
        agi2 = load_arc_agi2_public_eval(
            arc_agi2_tar,
            revision=f"pin:{ARC_AGI2_COMMIT}",
            accessed_at=None,
            expected_tasks=(DEFAULT_EXPECTED["arc_agi2"] if arc_agi2_expected is None
                            else arc_agi2_expected),
        )
        for mode, stage in (("cell_choice", "arc_agi2_choice"),
                            ("cell_score", "arc_agi2_score"),
                            ("whole_task", "arc_agi2_task")):
            spec = build_arc_agi2_spec(
                agi2, mode=mode, model=model,
                dataset_file=agi2_source["file"],
                source_sha256=agi2_source["sha256"],
            )
            suite[stage] = {"spec": spec, "source": agi2_source}
    if not skip_math500:
        m5_source = verify_data_sidecar(math500_path)
        m5 = load_math500(
            math500_path,
            revision=f"pin:{m5_source['sha256']}",
            accessed_at=None,
            expected_n=(DEFAULT_EXPECTED["math500"] if math500_expected is None
                        else math500_expected),
        )
        for mode, stage in (("choice", "math500_choice"),
                            ("score", "math500_score")):
            spec = build_math500_spec(
                m5, mode=mode, model=model,
                dataset_file=m5_source["file"],
                source_sha256=m5_source["sha256"],
            )
            suite[stage] = {"spec": spec, "source": m5_source}
    # Item ids must be unique WITHIN each stage (each stage is its own run dir
    # with its own gold join); the same task id intentionally recurs across the
    # different encoding stages (separate runs, distinct condition prefixes).
    for stage in suite:
        ids = [it["id"] for it in suite[stage]["spec"]["items"]]
        if len(ids) != len(set(ids)):
            raise BenchmarkSpecError(f"duplicate item id within stage {stage}; refusing")
    suite["counts"] = {stage: len(suite[stage]["spec"]["items"]) for stage in suite}
    suite["total_requests"] = sum(suite["counts"].values())
    return suite


def build_access_stages(
    *,
    gpqa_csv: Path | str = GPQA_CSV_DEFAULT,
    hle_parquet: Path | str = HLE_PARQUET_DEFAULT,
    model: str = "jev-1.13.0",
) -> dict[str, Any]:
    """Build the gated-access stages (GPQA Diamond + HLE text-only MC).

    Requires the sidecar-pinned files downloaded with the user's HF token
    (gates accepted); offline otherwise.  Counts are enforced at build time.
    """
    suite: dict[str, Any] = {}
    gpqa_source = verify_data_sidecar(gpqa_csv)
    gpqa, gpqa_accounting = load_gpqa_diamond(
        gpqa_csv, revision=f"pin:{gpqa_source['sha256'][:16]}", accessed_at=None,
        expected_n=DEFAULT_EXPECTED["gpqa_diamond"])
    if gpqa_accounting["exclusion_reasons"]:
        # source data defects are recorded in the spec, never silently dropped
        gpqa_defects = gpqa_accounting
    else:
        gpqa_defects = gpqa_accounting
    gpqa_spec = build_gpqa_spec(
        gpqa, model=model, dataset_file=gpqa_source["file"],
        source_sha256=gpqa_source["sha256"])
    gpqa_spec["build_accounting"] = gpqa_defects
    suite["gpqa_diamond"] = {"spec": gpqa_spec, "source": gpqa_source}

    hle_source = verify_data_sidecar(hle_parquet)
    hle, hle_accounting = load_hle(
        hle_parquet, revision=f"pin:{hle_source['sha256'][:16]}", accessed_at=None,
        expected_n=DEFAULT_EXPECTED["hle_rows"])
    hle_spec = build_hle_spec(
        hle, hle_accounting, model=model, dataset_file=hle_source["file"],
        source_sha256=hle_source["sha256"])
    suite["hle_text_mc"] = {"spec": hle_spec, "source": hle_source}

    for stage in suite:
        ids = [it["id"] for it in suite[stage]["spec"]["items"]]
        if len(ids) != len(set(ids)):
            raise BenchmarkSpecError(f"duplicate item id within stage {stage}; refusing")
    suite["counts"] = {stage: len(suite[stage]["spec"]["items"]) for stage in suite}
    suite["total_requests"] = sum(suite["counts"].values())
    return suite


def build_benchmark_suite(
    *,
    mmlu_path: Path | str = "data/test-00000-of-00001.parquet",
    arc_path: Path | str = "data/arc_challenge_test.parquet",
    model: str = "jev-1.13.0",
    expected: dict[str, int] | None = None,
    skip_arc: bool = False,
    extended: bool = False,
    arc_agi2_tar: Path | str = ARC_AGI2_TARBALL_DEFAULT,
    math500_path: Path | str = MATH500_JSONL_DEFAULT,
    skip_arc_agi2: bool = False,
    skip_math500: bool = False,
) -> dict[str, Any]:
    """Verify sources, then build the full specs. Offline only.

    ``expected`` overrides the hard split counts (fixtures only); production
    uses the official full-split counts and refuses anything else.
    ``extended=True`` adds the late-2026 stages (ARC-AGI-2 public eval in
    three Jev-native encodings + MATH-500 in two) under the same ceiling.
    """
    counts = dict(DEFAULT_EXPECTED if expected is None else expected)
    agi2_expected = counts.get("arc_agi2", DEFAULT_EXPECTED["arc_agi2"])
    m5_expected = counts.get("math500", DEFAULT_EXPECTED["math500"])
    mmlu_source = verify_data_sidecar(mmlu_path)
    mmlu_dataset = load_mmlu_pro(
        mmlu_path, revision=_source_revision(mmlu_source["url"]),
        accessed_at=None,
    )
    if len(mmlu_dataset.items) != counts["mmlu_full"]:
        raise BenchmarkSpecError(
            f"MMLU-Pro TEST split: expected {counts['mmlu_full']} items, got "
            f"{len(mmlu_dataset.items)}; refusing (full split only, never a pilot)"
        )

    mmlu_full_spec = build_mmlu_spec(
        mmlu_dataset, pilot=None, model=model,
        dataset_file=mmlu_source["file"],
    )
    mmlu_full_spec["experiment"] = "mmlu-pro-test-full"
    mmlu_full_spec["protocol"] = {
        "instructions": "Answer the multiple-choice question. Choose the single best option.",
        "prompt_variants": 1,
        "fixed_instructions": True,
        "no_chain_of_thought": True,
        "correct_answer_retries": 0,
        "option_order": "native (source order preserved; never sorted)",
        "gold_location": "spec/offline-join only; never in outbound payloads",
    }

    subset = subject_stratified_subset(mmlu_dataset.items)
    rotation_items, rotations = build_rotation_items(subset)
    if len(rotation_items) != N_ROTATION_ITEMS and expected is None:
        raise BenchmarkSpecError(
            f"rotation stage: expected {N_ROTATION_ITEMS} items, got {len(rotation_items)}"
        )
    # Rotations share the same dataset provenance as the full split (same file,
    # same hash) and are a SEPARATE spec so robustness never mixes into the
    # headline stage.
    rotation_spec = {
        "experiment": "mmlu-pro-test-rotations-140x3",
        "model": model,
        "test_family": "public-benchmark",
        "seeds": {"order": 0, "rotation_seed": ROTATION_SEED},
        "shuffle": False,  # dispatch follows the frozen stage order
        "protocol": {
            "instructions": "Answer the multiple-choice question. Choose the single best option.",
            "prompt_variants": 3,
            "fixed_instructions": True,
            "no_chain_of_thought": True,
            "correct_answer_retries": 0,
            "option_order": ("3 deterministic rotations of the frozen 140-item "
                             "subject-stratified subset; reported separately"),
            "base_item_ids": [it["id"] for it in subset],
        },
        "rotations": rotations,
        "dataset": dict(mmlu_full_spec["dataset"]),
        "items": rotation_items,
    }

    suite = {
        "mmlu_full": {"spec": mmlu_full_spec, "source": mmlu_source},
        "option_rotations": {"spec": rotation_spec, "source": mmlu_source},
    }
    if not skip_arc:
        arc_source = verify_data_sidecar(arc_path)
        arc_dataset = load_arc_challenge(
            arc_path, revision=_source_revision(arc_source["url"]), accessed_at=None,
            expected_n=counts.get("arc_test"),
        )
        arc_spec = build_arc_spec(arc_dataset, model=model, dataset_file=arc_source["file"])
        suite["arc_test"] = {"spec": arc_spec, "source": arc_source}

    if extended:
        extension = build_extension_stages(
            arc_agi2_tar=arc_agi2_tar, math500_path=math500_path, model=model,
            skip_arc_agi2=skip_arc_agi2, skip_math500=skip_math500,
            arc_agi2_expected=agi2_expected, math500_expected=m5_expected,
        )
        for stage in EXTENSION_STAGES:
            if stage in extension:
                suite[stage] = extension[stage]
        suite.pop("counts", None)
        suite.pop("total_requests", None)

    total = sum(len(suite[stage]["spec"]["items"]) for stage in suite)
    if total > REQUEST_CEILING:
        raise BenchmarkSpecError(
            f"request ceiling {REQUEST_CEILING} exceeded: suite dispatches {total} requests"
        )
    all_pairs = [(stage, it["id"]) for stage in suite for it in suite[stage]["spec"]["items"]]
    if len(all_pairs) != len(set(all_pairs)):
        raise BenchmarkSpecError("duplicate (stage, item id) pair; refusing")
    for stage in suite:
        ids = [it["id"] for it in suite[stage]["spec"]["items"]]
        if len(ids) != len(set(ids)):
            raise BenchmarkSpecError(f"duplicate item id within stage {stage}; refusing")
    suite["counts"] = {
        stage: len(suite[stage]["spec"]["items"]) for stage in EXTENDED_STAGES if stage in suite
    }
    suite["total_requests"] = total
    suite["request_ceiling"] = REQUEST_CEILING
    return suite


def _source_revision(url: str) -> str:
    """Record the pinned source URL as the revision pointer (no network)."""
    return url
