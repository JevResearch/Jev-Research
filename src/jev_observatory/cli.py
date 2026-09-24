"""CLI: plan, validate, run, analyze, report.

Safety defaults:
* `--provider jev` requires BOTH the explicit `--live` flag and the environment
  opt-in `JEVO_ALLOW_LIVE=1`; otherwise it refuses to dispatch anything.
* mock/replay runs work fully offline and are labelled synthetic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from . import analysis
from .experiment import SpecError, load_experiment, logical_requests, spec_sha256
from .guards import BudgetCaps, BudgetLedger, RateLimiter
from .manifest import utc_now
from .providers import JevProvider, MockProvider, RetryPolicy
from .schema import DEFAULT_MODEL_PIN
from .redact import API_KEY_ENV, LIVE_OPT_IN_ENV, base_url, get_api_key, live_calls_allowed
from .report import build_report
from .runner import Runner, load_manifest, plan_run, store_root

app = typer.Typer(help="Offline-first experimental harness for studying the Jev endpoint.", no_args_is_help=True)


def _fail(message: str) -> "typer.Exit":
    typer.secho(message, fg=typer.colors.RED, err=True)
    raise typer.Exit(code=2)


def _redactor():
    from .redact import Redactor

    return Redactor.from_environment()


def _store(run_id: str, root: str):
    from .ledger import RunStore

    return RunStore(store_root(root), run_id, redactor=_redactor())


@app.command("review-freeze")
def review_freeze(
    root: str = typer.Option("runs_reviewed", help="Private reviewed-run root (gitignored)"),
) -> None:
    """Freeze spec + ordered outbound payload hashes BEFORE any dispatch. No calls."""
    from .reviewed_executor import ExecutorError, build_freeze

    try:
        doc = build_freeze(root)
    except (ValueError, FileNotFoundError, ExecutorError) as exc:
        _fail(f"freeze failed: {exc}")
        return
    typer.secho(f"frozen (deterministic sha256 {doc['deterministic_sha256'][:12]})", fg=typer.colors.GREEN)
    for stage, entry in doc["stages"].items():
        typer.echo(f"  {stage}: {entry['n_requests']} request(s)")


@app.command("review-executor")
def review_executor(
    root: str = typer.Option("runs_reviewed", help="Private reviewed-run root (gitignored)"),
    live: bool = typer.Option(False, help="Dispatch the approved live calls (also requires JEVO_ALLOW_LIVE=1 and TYPESAFE_API_KEY in the environment); default is DRY-RUN"),
) -> None:
    """Bounded reviewed executor (REVIEW-2). Default: dry-run (plan + hash verify, no dispatch)."""
    from .reviewed_executor import ExecutorError, ReviewedExecutor

    executor = ReviewedExecutor(root)
    try:
        state = executor.run(dry_run=not live)
    except ExecutorError as exc:
        _fail(f"executor refused: {exc}")
        return
    typer.secho(
        "dry-run packet written (nothing dispatched)" if state["dry_run"]
        else f"executor finished; global stop reason: {state['global_stop_reason']}",
        fg=typer.colors.GREEN if state["dry_run"] or not state["global_stop_reason"] else typer.colors.YELLOW,
    )
    for stage, entry in state["stages"].items():
        line = f"  {stage}: run={entry['run_id']} verify={entry['freeze_verification']}"
        for key in ("dispatched", "ok", "error", "skipped_resume", "stop_reason", "skipped_reason"):
            if key in entry:
                line += f" {key}={entry[key]}"
        typer.echo(line)
    typer.echo(f"state: {Path(root) / 'executor_state.json'}")


@app.command("review-report")
def review_report(
    root: str = typer.Option("runs_reviewed"),
    out: Path = typer.Option(None, help="Output directory (default <root>/derived)"),
) -> None:
    """Regenerate numeric artifacts + markdown report FROM results/items/attempts. No provider access."""
    from .reviewed_report import build_reviewed_report

    try:
        _doc, json_path, md_path = build_reviewed_report(root, out)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        _fail(f"report failed: {exc}")
        return
    typer.secho(f"wrote {json_path} and {md_path}", fg=typer.colors.GREEN)


@app.command()
def validate(spec_path: Path = typer.Argument(..., help="Experiment spec JSON")) -> None:
    """Validate an experiment spec (schema, option maps, leakage guard). No calls."""
    try:
        spec = load_experiment(spec_path)
        requests = logical_requests(spec)
    except (SpecError, FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        _fail(f"spec invalid: {exc}")
        return
    typer.secho(f"OK: {len(requests)} logical request(s) pass schema, option-map and leakage checks",
                fg=typer.colors.GREEN)
    for request in requests[:5]:
        typer.echo(f"  {request.logical_request_id}  sha256={request.request.request_sha256()[:12]}")


@app.command()
def plan(
    spec_path: Path = typer.Argument(..., help="Experiment spec JSON"),
    provider: str = typer.Option("mock", help="mock | jev | replay"),
    root: str = typer.Option("runs", help="Directory for run artifacts"),
) -> None:
    """Build the manifest and cost estimate; dispatches nothing."""
    try:
        spec = load_experiment(spec_path)
        run_id, _manifest = plan_run(spec, provider=provider, root=root)
    except (SpecError, ValueError, json.JSONDecodeError) as exc:
        _fail(f"plan failed: {exc}")
        return
    plan_doc = json.loads((store_root(root) / run_id / "plan.json").read_text())
    typer.secho(f"planned run {run_id}", fg=typer.colors.GREEN)
    typer.echo(json.dumps(plan_doc, indent=2))


@app.command()
def run(
    run_id: str = typer.Argument(...),
    provider: str = typer.Option("mock", help="mock | jev | replay | simulated"),
    root: str = typer.Option("runs"),
    concurrency: int = typer.Option(1, min=1),
    live: bool = typer.Option(False, help="Allow live API calls (also requires JEVO_ALLOW_LIVE=1)"),
    on_uncertain: str = typer.Option("error", help="retry | skip | error (resume policy for timeouts)"),
    max_requests: Optional[int] = typer.Option(None, help="Hard cap on dispatched requests"),
    max_cost_usd: Optional[float] = typer.Option(None, help="Hard cap on reserved+reported cost"),
    max_estimated_tokens: Optional[int] = typer.Option(None, help="Hard cap on conservative estimate"),
    retry_mode: str = typer.Option("timing", help="timing (zero retries) | operational"),
    max_attempts: int = typer.Option(3, min=1, help="Used only with --retry-mode operational"),
    replay_dir: Optional[Path] = typer.Option(None, help="Recording directory for --provider replay"),
) -> None:
    """Execute a planned run. Mock/replay runs stay fully offline."""
    manifest = load_manifest(run_id, root)
    store = _store(run_id, root)
    if manifest.provider != provider:
        _fail(
            f"manifest records provider {manifest.provider!r} but --provider {provider!r} was requested; "
            "re-plan with the intended provider (manifests are immutable by design)"
        )
        return
    items_path = store.directory / "items.jsonl"
    if not items_path.exists():
        _fail(f"run {run_id} has no items.jsonl; run plan first")
        return

    try:
        spec = _spec_from_run(store, items_path, manifest)
    except ValueError as exc:
        _fail(str(exc))
        return

    try:
        provider_obj = _make_provider(provider, live=live, retry_mode=retry_mode,
                                      max_attempts_value=max_attempts, replay_dir=replay_dir)
    except typer.BadParameter:
        raise
    except _Fail as exc:
        _fail(str(exc))
        return

    budget_ledger = BudgetLedger(BudgetCaps(
        max_requests=max_requests,
        max_cost_usd=max_cost_usd,
        max_estimated_input_tokens=max_estimated_tokens,
    ))
    rate = RateLimiter()
    requests = logical_requests(spec)
    reset = getattr(getattr(provider_obj, "transport", None), "reset_pool", None)
    runner = Runner(provider_obj, store, concurrency=concurrency, budget=budget_ledger, rate=rate,
                    transport_reset=reset)
    runner.install_signal_handlers()
    try:
        result = runner.run(requests, on_uncertain=on_uncertain)
    finally:
        provider_obj.close()
        runner.write_summary()
    typer.secho(
        f"run {run_id}: dispatched={result.n_dispatched} ok={result.n_ok} "
        f"contract_invalid={result.n_contract_invalid} error={result.n_error} "
        f"uncertain={result.n_uncertain} cancelled={result.n_cancelled} "
        f"skipped_resume={result.n_skipped_resume} halted={result.halted}",
        fg=typer.colors.GREEN if not result.halted else typer.colors.YELLOW,
    )
    if result.n_contract_invalid:
        typer.echo("contract violations present; run analyze/report for codes")


class _Fail(Exception):
    pass


@app.command()
def analyze(
    run_id: str = typer.Argument(...),
    root: str = typer.Option("runs"),
) -> None:
    """Compute metrics offline from results.jsonl + items.jsonl. No provider calls."""
    try:
        metrics = analysis.analyze_run(run_id, root)
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as exc:
        _fail(f"analyze failed: {exc}")
        return
    synthetic = metrics.get("synthetic")
    typer.secho(
        f"analyzed {run_id}: {metrics['n_logical_results']} result(s), "
        f"{metrics['n_uncertain_open']} uncertain open"
        + (" [SYNTHETIC MOCK DATA]" if synthetic else ""),
        fg=typer.colors.YELLOW if synthetic else typer.colors.GREEN,
    )


@app.command()
def report(
    run_id: str = typer.Argument(...),
    root: str = typer.Option("runs"),
) -> None:
    """Write report.md for an analyzed run."""
    try:
        text = build_report(run_id, root)
    except FileNotFoundError as exc:
        _fail(str(exc))
        return
    out = store_root(root) / run_id / "report.md"
    out.write_text(text, encoding="utf-8")
    typer.secho(f"wrote {out}", fg=typer.colors.GREEN)
    typer.echo(text)


# ------------------------------------------------------------------ helpers
def _make_provider(name: str, *, live: bool, retry_mode: str, max_attempts_value: int,
                   replay_dir: Path | None = None):
    redactor = _redactor()
    if name == "mock":
        return MockProvider(seed=0)
    if name == "replay":
        if replay_dir is None:
            raise _Fail("--provider replay requires --replay-dir")
        from .transport import ReplayTransport

        try:
            transport = ReplayTransport(replay_dir)
        except FileNotFoundError as exc:
            raise _Fail(str(exc))
        policy = RetryPolicy.none() if retry_mode == "timing" else RetryPolicy.operational(max_attempts_value)
        return JevProvider(transport, retry_policy=policy, redactor=redactor)
    if name == "simulated":
        # Known-cost simulated endpoint: for validating the analysis pipeline.
        # Artifacts remain labelled by provider name 'jev'? No — provider='simulated'.
        from .simulated import SimConfig, SimulatedTransport

        policy = RetryPolicy.none() if retry_mode == "timing" else RetryPolicy.operational(max_attempts_value)
        return JevProvider(SimulatedTransport(SimConfig()), retry_policy=policy, redactor=redactor)
    if name == "jev":
        if not live:
            raise _Fail("refusing live calls without --live (default deny; Milestone 4 gate)")
        if not live_calls_allowed():
            raise _Fail(f"live calls also require environment {LIVE_OPT_IN_ENV}=1")
        api_key = get_api_key()
        if not api_key:
            raise _Fail(f"no API key in environment ({API_KEY_ENV})")
        from .transport import HttpxTransport

        policy = RetryPolicy.none() if retry_mode == "timing" else RetryPolicy.operational(max_attempts_value)
        return JevProvider(
            HttpxTransport(base_url(), api_key, redactor=redactor),
            retry_policy=policy,
            redactor=redactor,
        )
    raise typer.BadParameter(f"unknown provider {name!r}")


def _spec_from_run(store, items_path, manifest) -> dict:
    """Rebuild the logical-request spec from the run's immutable artifacts."""
    items = [json.loads(line) for line in items_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    spec = {
        "experiment": manifest.experiment,
        "model": manifest.model_requested,
        "items": items,
        "seeds": manifest.seeds,
        "shuffle": manifest.shuffle,  # designed experiments dispatch in block order
    }
    if spec_sha256(spec) != manifest.items_sha256:
        raise ValueError(
            "items.jsonl does not match manifest.items_sha256; refusing to run against mutated items"
        )
    return spec


@app.command()
def make_spec(
    dataset: str = typer.Option(..., help="mmlu-pro | boolq | synthetic"),
    data_file: Optional[Path] = typer.Option(None, help="Local data file (required for public datasets)"),
    out: Path = typer.Option(..., help="Output spec JSON path"),
    pilot: Optional[int] = typer.Option(None, help="MMLU-Pro stratified pilot size"),
    n: Optional[int] = typer.Option(None, help="BoolQ balanced sample size (even)"),
    synthetic_n: int = typer.Option(10, help="Items per synthetic generator"),
    seed: int = typer.Option(0),
    model: str = typer.Option("jev-1.13.0"),
    revisions: str = typer.Option("", help="Dataset revision label, e.g. git sha or version"),
    with_permutations: int = typer.Option(0, help="Add N permuted variants per choice item"),
) -> None:
    """Build an experiment spec from a pinned dataset or synthetic generators."""
    from .datasets.boolq import build_spec as boolq_spec, load_boolq
    from .datasets.generators import (
        build_synthetic_spec,
        gen_base_rate,
        gen_policy_routing,
        gen_relation_lookup,
        gen_unanswerable,
    )
    from .datasets.mmlu_pro import build_spec as mmlu_spec, load_mmlu_pro
    from .datasets.permutation import attach_permutations
    from .manifest import utc_now

    accessed = utc_now()
    try:
        if dataset == "mmlu-pro":
            if data_file is None:
                _fail("mmlu-pro requires --data-file (see jevo fetch-data)")
            loaded = load_mmlu_pro(data_file, revision=revisions or None, accessed_at=accessed)
            spec = mmlu_spec(loaded, pilot=pilot, seed=seed, model=model, dataset_file=str(data_file))
        elif dataset == "boolq":
            if data_file is None:
                _fail("boolq requires --data-file (see jevo fetch-data)")
            loaded = load_boolq(data_file, revision=revisions or None, accessed_at=accessed)
            spec = boolq_spec(loaded, n=n, seed=seed, model=model, dataset_file=str(data_file))
        elif dataset == "synthetic":
            generators = [
                ("relation_lookup", gen_relation_lookup),
                ("policy_routing", gen_policy_routing),
                ("unanswerable", gen_unanswerable),
                ("base_rate", gen_base_rate),
            ]
            spec = build_synthetic_spec(generators, seed=seed, model=model, n_per_generator=synthetic_n)
        else:
            _fail(f"unknown dataset {dataset!r}")
            return
    except (ValueError, FileNotFoundError) as exc:
        _fail(f"make-spec failed: {exc}")
        return
    if with_permutations > 0:
        spec = attach_permutations(spec, seed=seed, variants=with_permutations)
    out.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    typer.secho(f"wrote {out} with {len(spec['items'])} item(s)", fg=typer.colors.GREEN)


@app.command()
def fetch_data(
    dataset: str = typer.Option(..., help="mmlu-pro | boolq"),
    dest: Path = typer.Option(Path("data"), help="Download directory"),
    url: Optional[str] = typer.Option(None, help="Override the pinned source URL"),
) -> None:
    """Download a dataset (network op, run manually; never part of tests).

    The downloaded file's sha256 is recorded in a sidecar; no unverified claims
    are made about expected hashes.  Respect each dataset's own terms.
    """
    import hashlib

    import httpx

    pinned = {
        "mmlu-pro": "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/main/test-00000-of-00001.parquet",
        "boolq": "https://storage.googleapis.com/boolq/train.jsonl",
    }
    if dataset not in pinned:
        _fail(f"unknown dataset {dataset!r}; known: {sorted(pinned)}")
        return
    target_url = url or pinned[dataset]
    dest.mkdir(parents=True, exist_ok=True)
    filename = target_url.rsplit("/", 1)[-1].split("?")[0]
    target = dest / filename
    typer.echo(f"downloading {target_url} -> {target}")
    try:
        with httpx.Client(follow_redirects=True, timeout=120.0) as client:
            with client.stream("GET", target_url) as response:
                response.raise_for_status()
                with target.open("wb") as handle:
                    for chunk in response.iter_bytes():
                        handle.write(chunk)
    except httpx.HTTPError as exc:
        _fail(f"download failed: {exc}")
        return
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    sidecar = target.with_suffix(target.suffix + ".sha256.json")
    sidecar.write_text(json.dumps({
        "file": str(target), "sha256": digest, "url": target_url,
        "downloaded_at": utc_now(), "dataset": dataset,
    }, indent=2) + "\n")
    typer.secho(f"sha256 {digest} recorded in {sidecar}", fg=typer.colors.GREEN)
    typer.echo("remember to pass --revision and check the dataset's own terms")


@app.command()
def import_scores(
    scores_path: Path = typer.Argument(..., help="External-scores JSON document"),
    out: Optional[Path] = typer.Option(None, help="Write heatmap JSON here"),
    chance_adjust: bool = typer.Option(False, help="Adjust accuracy cells; needs chance= in protocol notes"),
) -> None:
    """Validate external reference scores and build heatmap data."""
    from .external import ScoreImportError, build_heatmap, heatmap_markdown, import_scores

    try:
        scores = import_scores(scores_path)
    except (ScoreImportError, FileNotFoundError, json.JSONDecodeError) as exc:
        _fail(f"import failed: {exc}")
        return
    grid = build_heatmap(scores, chance_adjust=chance_adjust)
    if out is not None:
        out.write_text(json.dumps(grid, indent=2, ensure_ascii=False) + "\n")
        typer.secho(f"wrote {out}", fg=typer.colors.GREEN)
    typer.echo(heatmap_markdown(grid))


@app.command()
def make_spec_m2(
    design: str = typer.Option(..., help="sweep | isolation | odds"),
    out: Path = typer.Option(..., help="Output spec JSON path"),
    seed: int = typer.Option(0),
    model: str = typer.Option("jev-1.13.0"),
    repeats: int = typer.Option(3, min=1),
    l_levels: str = typer.Option("128,512,2048,8192", help="Comma-separated state sizes (chars)"),
    q_levels: str = typer.Option("1,4,16,64"),
    k_levels: str = typer.Option("2,8,64,255"),
    sibling_levels: str = typer.Option("0,1,8", help="Isolation design only"),
    primitives: str = typer.Option("choice", help="Comma-separated: choice,noul,score"),
) -> None:
    """Build M2 mechanism-experiment specs (sweep / isolation / odds)."""
    from .experiments import (
        SweepCondition,
        isolation_spec,
        odds_spec,
        one_factor_sweep,
        screening_cross,
        sweep_spec,
    )

    def _levels(text: str) -> tuple[int, ...]:
        try:
            levels = tuple(int(v) for v in text.split(",") if v.strip())
        except ValueError:
            raise typer.BadParameter(f"could not parse levels {text!r}")
        if not levels:
            raise typer.BadParameter("at least one level required")
        return levels

    if design == "sweep":
        conditions = one_factor_sweep(
            l_levels=_levels(l_levels), q_levels=_levels(q_levels), k_levels=_levels(k_levels),
            primitives=tuple(p.strip() for p in primitives.split(",") if p.strip()),
        )
        conditions += screening_cross(_levels(l_levels), _levels(q_levels), _levels(k_levels))
        spec = sweep_spec(conditions, repeats=repeats, seed=seed, model=model)
    elif design == "isolation":
        spec = isolation_spec(sibling_levels=_levels(sibling_levels), repeats=repeats,
                              seed=seed, model=model)
    elif design == "odds":
        spec = odds_spec(repeats=repeats, seed=seed, model=model)
    else:
        _fail(f"unknown design {design!r}")
        return
    out.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    typer.secho(f"wrote {out} with {len(spec['items'])} item(s) "
                f"(design={design}, claim_type={spec['claim_type']})", fg=typer.colors.GREEN)


@app.command()
def analyze_isolation(
    run_id: str = typer.Argument(...),
    root: str = typer.Option("runs"),
    margin: float = typer.Option(0.05, help="Preregistered TVD equivalence margin"),
) -> None:
    """Anchor-stability analysis for an isolation run (offline)."""
    from pathlib import Path as _Path

    from .mechanism import isolation_summary

    results = list(json.loads(l) for l in ((_Path(root) / run_id / "results.jsonl").read_text().splitlines()) if l.strip())
    summary = isolation_summary(results, margin=margin)
    out_dir = _Path(root) / run_id / "derived"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "isolation.json").write_text(json.dumps(summary, indent=2) + "\n")
    typer.secho(f"verdict: {summary['verdict']}", fg=typer.colors.YELLOW if not summary["all_within_margin"] else typer.colors.GREEN)
    typer.echo(f"claim_type={summary['claim_type']}; written to {out_dir / 'isolation.json'}")


@app.command()
def analyze_odds(
    run_id: str = typer.Argument(...),
    root: str = typer.Option("runs"),
) -> None:
    """Candidate-odds analysis for an odds run (offline, exploratory)."""
    from pathlib import Path as _Path

    from .mechanism import odds_summary

    results = list(json.loads(l) for l in ((_Path(root) / run_id / "results.jsonl").read_text().splitlines()) if l.strip())
    summary = odds_summary(results)
    out_dir = _Path(root) / run_id / "derived"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "odds.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not summary.get("available", True):
        _fail(f"odds analysis unavailable: {summary.get('reason')}")
        return
    typer.echo(json.dumps(summary, indent=2))


@app.command()
def fit_latency(
    run_id: str = typer.Argument(...),
    root: str = typer.Option("runs"),
) -> None:
    """Fit competing latency models with held-out validation (offline)."""
    from pathlib import Path as _Path

    from .simulated import fit_latency_model

    results = list(json.loads(l) for l in ((_Path(root) / run_id / "results.jsonl").read_text().splitlines()) if l.strip())
    fit = fit_latency_model(results)
    out_dir = _Path(root) / run_id / "derived"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "latency_model.json").write_text(json.dumps(fit, indent=2) + "\n")
    if not fit.get("fit_available"):
        _fail(f"latency fit unavailable: {fit.get('reason')}")
        return
    typer.echo(json.dumps(fit, indent=2))


@app.command()
def talk(
    provider: str = typer.Option("mock", help="mock | jev (simulated/replay are not useful here)"),
    live: bool = typer.Option(False, help="Allow live API calls (also requires JEVO_ALLOW_LIVE=1)"),
    host: str = typer.Option("127.0.0.1", help="Must be loopback"),
    port: int = typer.Option(8765),
    max_chars: int = typer.Option(256, min=1),
    max_seconds: float = typer.Option(120.0),
    decode: str = typer.Option("greedy", help="greedy | sample"),
    temperature: Optional[float] = typer.Option(None, help="Local transform for --decode sample"),
    seed: Optional[int] = typer.Option(None, help="Seed for --decode sample"),
    max_requests: Optional[int] = typer.Option(None),
    max_cost_usd: Optional[float] = typer.Option(None),
) -> None:
    """Start the local Talk UI (loopback only). Character-by-character decoding."""
    from .talk import DecoderConfig, TalkLimits
    from .web import assert_loopback, create_app

    try:
        assert_loopback(host, port)
    except ValueError as exc:
        _fail(str(exc))
        return

    provider_obj = _make_provider(provider, live=live, retry_mode="timing", max_attempts_value=1)

    caps = BudgetCaps(max_requests=max_requests, max_cost_usd=max_cost_usd)
    budget = BudgetLedger(caps)
    limits = TalkLimits(max_chars=max_chars, max_seconds=max_seconds)
    if temperature is not None and decode != "sample":
        _fail("--temperature requires --decode sample")
        return
    decoder = DecoderConfig(mode=decode, temperature=temperature, seed=seed)

    def provider_factory():
        # A fresh provider per session keeps connection state isolated; the
        # credential stays inside the server process either way.
        return _make_provider(provider, live=live, retry_mode="timing", max_attempts_value=1)

    try:
        application = create_app(provider_factory, limits=limits, decoder=decoder,
                                 budget=budget, model=DEFAULT_MODEL_PIN)
    except Exception as exc:  # pragma: no cover - defensive
        _fail(f"failed to build app: {exc}")
        return
    typer.secho(f"Talk UI on http://{host}:{port} (loopback only; provider={provider}; "
                f"decode={decode}; caps={{'max_requests': max_requests, 'max_cost_usd': max_cost_usd}})",
                fg=typer.colors.GREEN)
    try:
        import uvicorn
    except ImportError:
        _fail("uvicorn is required to serve the Talk UI: pip install 'jev-observatory[web]'")
        return
    uvicorn.run(application, host=host, port=port, log_level="warning")


# talk uses the same pinned default as every other command


@app.command("audit-offline")
def audit_offline(
    paired_run: str = typer.Option(
        "runs_live/2026-09-18-e1a2252d3ec8",
        help="Run directory with the paired BoolQ artifacts"),
    mech_run: str = typer.Option(
        "runs_live/2026-09-18-a5200d84836c",
        help="Run directory with the mechanism-v3 artifacts"),
    out: Path = typer.Option("docs/review-gate/audit", help="Output directory for fresh derived artifacts"),
) -> None:
    """Reproducible offline audit reconstruction (LEAD-GATE requirement A).

    Loads immutable run artifacts and independently rebuilds the paired BoolQ
    two-encoding table and the mechanism-v3 paired signed P(yes) shifts. No API
    calls. Writes fresh derived artifacts under `out`; never modifies the runs.
    Strict JSON: non-finite values are explicit markers, never Infinity.
    """
    from .audit_offline import build_audit_bundle

    outputs = build_audit_bundle(paired_run_dir=paired_run, mech_run_dir=mech_run, out_dir=out)
    typer.secho(f"wrote {len(outputs)} strict-JSON audit artifact(s):", fg=typer.colors.GREEN)
    for path in outputs:
        typer.echo(f"  {path}")


@app.command()
def summarize(
    root: str = typer.Option("runs_live", help="Directory of run artifacts"),
    out: Path = typer.Option(None, help="Output markdown path (default <root>/SUMMARY-regen.md)"),
) -> None:
    """Regenerate the corrected results summary from existing artifacts.

    No API calls. Every number in the output is recomputed from saved run
    artifacts; narrative labels describe what each number is, nothing more.
    This is the audit acceptance criterion: one command, offline, reproducible.
    """
    import math as _math

    from .paired import paired_encoding_comparison

    root_path = Path(root)
    lines: list[str] = ["# Regenerated summary (offline, from artifacts)\n",
                        f"Generated: {utc_now()}\n"]
    for run_dir in sorted(root_path.iterdir()):
        if not (run_dir / "manifest.json").exists() or not (run_dir / "results.jsonl").exists():
            continue
        try:
            manifest = load_manifest(run_dir.name, str(root_path))
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
            lines.append(f"## {run_dir.name}\n\n- manifest verification FAILED: {exc}\n")
            continue
        results = [json.loads(l) for l in (run_dir / "results.jsonl").read_text().splitlines() if l.strip()]
        summary = json.loads((run_dir / "run_summary.json").read_text()) if (run_dir / "run_summary.json").exists() else {}
        budget = summary.get("budget") or {}
        statuses = collections_counter(r["status"] for r in results)
        lines.append(f"## {run_dir.name} — {manifest.experiment} (provider: {manifest.provider})\n")
        lines.append(f"- results: {len(results)}; statuses: {dict(statuses)}")
        if budget:
            lines.append(f"- reported input tokens: {budget.get('reported_input_tokens')}; "
                         f"cost ceiling USD: {budget.get('cost_ceiling_usd')}")
        if summary.get("uncertain_open"):
            lines.append(f"- uncertain attempts open: {len(summary['uncertain_open'])}")
        # paired comparison where the design is paired
        sampling = (manifest.dataset or {}).get("sampling", {})
        if sampling.get("mode") == "paired_conditions":
            items = {entry["id"]: entry for entry in _iter_items(run_dir / "items.jsonl")}
            comparison = paired_encoding_comparison(results, items)
            if comparison.get("available"):
                first, second = comparison["conditions"]
                pc = comparison["per_condition"]
                lines.append("- paired encoding comparison (computed):")
                for cond in (first, second):
                    lines.append(f"    - {cond}: accuracy {pc[cond]['accuracy']:.3f}, "
                                 f"binary Brier(P(yes)) {pc[cond]['binary_brier_p_yes']:.4f}, "
                                 f"ECE-10 {pc[cond]['p_yes_ece_10']:.4f}, "
                                 f"zero-true-probability {pc[cond]['n_zero_true_probability']}")
                lines.append(f"    - accuracy difference {comparison['accuracy_difference']:+.3f} "
                             f"(CI95 {comparison['accuracy_difference_ci95']}, "
                             f"McNemar p={comparison['mcnemar_exact_p']:.3f})")
                lines.append(f"    - Brier difference (first minus second) "
                             f"{comparison['brier_difference_first_minus_second']:+.4f} "
                             f"(CI95 {comparison['brier_difference_ci95']})")
        # violation summary
        violations = collections_counter([])
        for record in results:
            for code in record.get("violation_codes", []):
                violations[code] += 1
        if violations:
            lines.append(f"- contract violations: {dict(sorted(violations.items()))}")
        # metrics.json aggregates if analyzed
        metrics_path = run_dir / "derived" / "metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
            for qtype in ("choice", "noul"):
                agg = metrics.get("aggregates", {}).get(qtype)
                if not agg:
                    continue
                acc = agg.get("accuracy", {})
                lines.append(f"- {qtype}: accuracy {acc.get('accuracy')} (n={acc.get('n')}, "
                             f"Wilson {acc.get('wilson_95')})")
                exact = agg.get("log_loss_exact_mean")
                if exact is not None:
                    lines.append(f"- {qtype}: exact log loss {exact} "
                                 f"(infinite entries: {agg.get('n_infinite_log_loss')}); "
                                 f"clipped mean {agg.get('log_loss_clipped_1e-12_mean')}")
        lines.append("")
    out_path = out or (root_path / "SUMMARY-regen.md")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    typer.secho(f"wrote {out_path}", fg=typer.colors.GREEN)


def collections_counter(values):
    from collections import Counter

    return Counter(values)


def _iter_items(path: Path):
    import json as _json

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield _json.loads(line)


if __name__ == "__main__":  # pragma: no cover
    app()