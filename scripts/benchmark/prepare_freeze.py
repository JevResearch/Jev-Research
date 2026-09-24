#!/usr/bin/env python3
"""Offline benchmark preparation: verify sources, build the full suite, freeze.

Parent-invokable, offline, no credentials, no network (unless --verify-source,
which re-downloads ONLY the public pinned parquet files to a temp dir and
compares their sha256 against the sidecars — never a model call).

Outputs under --root (default runs_benchmark/):
  freeze/frozen.json          write-once freeze (specs + ordered wire hashes)
  <run_id>/spec.json          per-stage source spec (mmlu_full, option_rotations, arc_test)
  <run_id>/manifest.json      run manifest (plan_run; immutable)
  <run_id>/items.jsonl        frozen items incl. gold for the offline join
  <run_id>/plan.json          request/cost estimates
  <run_id>/rotations.json     rotation remappings (rotation stage only)
  freeze/preparation.json     counts, hashes, estimator, this run's record
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import argparse
import hashlib
import json
import sys
import tempfile
import urllib.request
from pathlib import Path

from jev_observatory.benchmark_exec import (
    FREEZE_ROOT_DEFAULT,
    build_freeze,
    load_freeze,
    stage_run_id,
)
from jev_observatory.benchmark_spec import (
    ARC_AGI2_COMMIT,
    ACCESS_STAGES,
    EXTENSION_STAGES,
    ARC_AGI2_TARBALL_DEFAULT,
    EXTENDED_STAGES,
    MATH500_JSONL_DEFAULT,
    N_ARC_AGI2_EVAL_TASKS,
    N_ARC_CHALLENGE_TEST,
    N_GPQA_DIAMOND,
    N_HLE_TEST_ROWS,
    N_MATH500_TEST,
    N_MMLU_PRO_TEST,
    N_ROTATION_ITEMS,
    GPQA_CSV_DEFAULT,
    HLE_PARQUET_DEFAULT,
    REQUEST_CEILING,
    build_benchmark_suite,
)
from jev_observatory.runner import plan_run
from jev_observatory.manifest import verify_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=FREEZE_ROOT_DEFAULT)
    parser.add_argument("--mmlu", default="data/test-00000-of-00001.parquet")
    parser.add_argument("--arc", default="data/arc_challenge_test.parquet")
    parser.add_argument("--arc-agi2", default=ARC_AGI2_TARBALL_DEFAULT,
                        help="pinned ARC-AGI-2 public-eval repo tarball")
    parser.add_argument("--math500", default=MATH500_JSONL_DEFAULT,
                        help="pinned MATH-500 test JSONL")
    parser.add_argument("--model", default="jev-1.13.0")
    parser.add_argument("--extended", action="store_true",
                        help="build the late-2026 extension stages (ARC-AGI-2 "
                             "public eval x3 encodings + MATH-500 x2 encodings)")
    parser.add_argument("--extension-only", action="store_true",
                        help="build ONLY the extension stages into a separate "
                             "freeze root (core runs_benchmark freeze untouched)")
    parser.add_argument("--access-only", action="store_true",
                        help="build ONLY the gated-access stages (GPQA Diamond + "
                             "HLE text-only MC) into a separate freeze root")
    parser.add_argument("--gpqa", default=GPQA_CSV_DEFAULT)
    parser.add_argument("--hle", default=HLE_PARQUET_DEFAULT)
    parser.add_argument("--verify-source", action="store_true",
                        help="re-download the public pinned files to a temp dir and "
                             "compare sha256 against the sidecars (public download; "
                             "no credentials, no model calls)")
    args = parser.parse_args()

    if args.extension_only:
        args.extended = True

    if args.extension_only and args.root == FREEZE_ROOT_DEFAULT:
        # never touch the completed core freeze root
        args.root = "runs_benchmark_ext"

    if args.verify_source:
        paths = [args.mmlu, args.arc]
        if args.extended:
            paths += [args.arc_agi2, args.math500]
        if args.access_only:
            paths += [args.gpqa, args.hle]
        for path in paths:
            sidecar = json.loads(Path(str(path) + ".sha256.json").read_text())
            with tempfile.TemporaryDirectory() as tmp:
                target = Path(tmp) / Path(path).name
                print(f"[verify-source] downloading {sidecar['url']}")
                urllib.request.urlretrieve(sidecar["url"], target)  # noqa: S310 - pinned https URL
                digest = hashlib.sha256(target.read_bytes()).hexdigest()
            if digest != sidecar["sha256"]:
                print(f"[verify-source] MISMATCH for {path}: {digest} != {sidecar['sha256']}")
                return 2
            print(f"[verify-source] OK {path} sha256={digest}")

    if args.access_only:
        if args.root == FREEZE_ROOT_DEFAULT:
            args.root = "runs_benchmark_ext2"
        from jev_observatory.benchmark_spec import build_access_stages

        suite = build_access_stages(gpqa_csv=args.gpqa, hle_parquet=args.hle,
                                    model=args.model)
        suite["request_ceiling"] = REQUEST_CEILING
    else:
        suite = build_benchmark_suite(
            mmlu_path=args.mmlu, arc_path=args.arc, model=args.model,
            extended=args.extended, arc_agi2_tar=args.arc_agi2,
            math500_path=args.math500,
            skip_arc=args.extension_only,
            skip_arc_agi2=not args.extended,
            skip_math500=not args.extended,
        )
    if args.extension_only:
        # extension-only freeze: drop the core stages, recompute the totals
        suite = {k: v for k, v in suite.items()
                 if k not in ("mmlu_full", "option_rotations", "arc_test")}
        suite["counts"] = {stage: len(suite[stage]["spec"]["items"])
                           for stage in suite if isinstance(suite[stage], dict)
                           and "spec" in suite[stage]}
        suite["total_requests"] = sum(suite["counts"].values())
    freeze = build_freeze(suite, root=args.root)

    root = Path(args.root)
    records: dict[str, dict] = {}
    all_stage_names = tuple(dict.fromkeys(EXTENDED_STAGES + ACCESS_STAGES))
    stage_order = [s for s in all_stage_names if s in suite]
    for stage in stage_order:
        spec = suite[stage]["spec"]
        run_id = stage_run_id(freeze, stage)
        directory = root / run_id
        spec_path = directory / "spec.json"
        if not spec_path.exists():
            directory.mkdir(parents=True, exist_ok=True)
            spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=1) + "\n",
                                 encoding="utf-8")
        plan_run(spec, provider="jev", root=str(root), run_id=run_id,
                 budgets=freeze["budget_caps"], retry_policy={"max_attempts": 1})
        if stage == "option_rotations":
            (directory / "rotations.json").write_text(
                json.dumps(spec["rotations"], indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8")
        manifest = verify_manifest(directory)
        records[stage] = {
            "run_id": run_id,
            "n_items": len(spec["items"]),
            "manifest_sha256": manifest.sha256(),
            "source_sha256": suite[stage]["source"]["sha256"],
        }

    preparation = {
        "created_at": freeze["created_at"],
        "freeze_sha256": freeze["deterministic_sha256"],
        "freeze_path": str(root / "freeze" / "frozen.json"),
        "extended": bool(args.extended),
        "counts": {
            "mmlu_pro_test_full": N_MMLU_PRO_TEST,
            "option_rotation_items": N_ROTATION_ITEMS,
            "arc_challenge_test_full": N_ARC_CHALLENGE_TEST,
            "arc_agi2_public_eval_tasks": N_ARC_AGI2_EVAL_TASKS,
            "arc_agi2_commit": ARC_AGI2_COMMIT,
            "math500_test": N_MATH500_TEST,
            "gpqa_diamond": N_GPQA_DIAMOND,
            "hle_test_rows": N_HLE_TEST_ROWS,
            "total_requests": suite["total_requests"],
            "request_ceiling": REQUEST_CEILING,
        },
        "expected_counts": suite["counts"],
        "sources": {stage: suite[stage]["source"] for stage in stage_order},
        "cost_estimator": _cost_estimator(),
        "stages": records,
    }
    out_path = root / "freeze" / "preparation.json"
    out_path.write_text(json.dumps(preparation, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    # sanity: the freeze must reload with a passing integrity check
    load_freeze(args.root)
    print(json.dumps(preparation["counts"], indent=2))
    print(f"[ok] freeze at {preparation['freeze_path']} "
          f"sha256={preparation['freeze_sha256'][:12]}")
    print(f"[ok] preparation record at {out_path}")
    return 0


def _cost_estimator() -> dict:
    """Bounded budget estimate for the FULL Jev benchmark (formula + defaults).

    Jev-side: plan.json estimates assume 1 token/char (conservative). The
    modern-baseline estimator lives in EXECUTION-PACKET.md and uses the
    recorded OpenRouter catalog prices; reasoning budgets dominate it.
    """
    return {
        "formula": "estimated_cost_usd = estimated_input_tokens / 1e6 * price_per_mtok",
        "jev_price_per_mtok_input": 0.042,
        "note": "server tokenisation is the only authority; estimates are conservative",
    }


if __name__ == "__main__":
    sys.exit(main())
# (prepare_freeze.py main already ends with sys.exit(main()))
