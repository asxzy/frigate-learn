"""Automated pipeline runner (Phase 13).

``frigate-learn run`` executes the configured stages in order:

    collect → review-sync → verify → build → train → benchmark → gate → deploy

Each stage registers a skip condition (missing data, missing ML deps, no new
dataset, etc.) and reports ``executed | skipped | failed`` so a nightly cron can
be launched unconditionally while humans only watch the summary. Stage results
flow through a small context dict (e.g. the dataset version built by ``build``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

from .config import AppConfig
from .db import Database
from .logutil import debug, error, info

PIPELINE = [
    "collect",
    "review-sync",
    "verify",
    "build",
    "train",
    "benchmark",
    "gate",
    "deploy",
]


@dataclass
class StepReport:
    name: str
    status: str  # executed | skipped | failed
    message: str = ""
    data: dict = field(default_factory=dict)


def next_build_version(config: AppConfig) -> str:
    """Next auto dataset version, e.g. v001 -> v002 (max existing + 1)."""
    datasets = config.datasets_dir()
    highest = 0
    if datasets.is_dir():
        for entry in datasets.iterdir():
            if not (entry.is_dir() and entry.name.startswith("v")):
                continue
            try:
                highest = max(highest, int(entry.name[1:]))
            except ValueError:
                continue
    return f"v{highest + 1:03d}"


def _default_steps(config: AppConfig) -> list[str]:
    enabled = [s for s in config.automation.enable if s in PIPELINE]
    till = config.automation.till
    if till:
        if till not in enabled:
            # allow "till=deploy" to extend past the explicit enable list
            ordered = PIPELINE
            if till in ordered:
                idx = ordered.index(till)
                base = ordered[: idx + 1]
                enabled = base if not enabled else enabled
        else:
            idx = enabled.index(till)
            enabled = enabled[: idx + 1]
    return enabled


def run_pipeline(
    config: AppConfig,
    db: Database,
    *,
    steps: list[str] | None = None,
    until: str | None = None,
    dry_run: bool = False,
    keep_going: bool = False,
    days: int | None = None,
) -> list[StepReport]:
    db.migrate()
    ordered = steps or _default_steps(config)
    ordered = [s for s in ordered if s in PIPELINE]
    if until:
        idx = PIPELINE.index(until)
        ordered = [s for s in ordered if PIPELINE.index(s) <= idx] or ordered[: idx + 1]
    ordered = list(dict.fromkeys(ordered))  # dedupe, keep order

    ctx: dict = {"dry_run": dry_run, "version": None, "train": None,
                 "benchmark": None, "gate": None}
    reports: list[StepReport] = []

    handlers = {
        "collect": lambda: _step_collect(config, db, days, ctx),
        "review-sync": lambda: _step_review_sync(config, db, ctx),
        "verify": lambda: _step_verify(config, db, ctx),
        "build": lambda: _step_build(config, db, ctx),
        "train": lambda: _step_train(config, db, ctx),
        "benchmark": lambda: _step_benchmark(config, db, ctx),
        "gate": lambda: _step_gate(config, db, ctx),
        "deploy": lambda: _step_deploy(config, db, ctx),
    }

    for name in ordered:
        try:
            report = handlers[name]()
        except Exception as exc:
            error("step failed", step=name, error=str(exc))
            report = StepReport(name=name, status="failed", message=str(exc))
        reports.append(report)
        if report.status == "failed" and not keep_going:
            debug("pipeline halted", step=name)
            break
    for report in reports:
        info("pipeline step", step=report.name, status=report.status, msg=report.message)
    return reports


# --- stage implementations -------------------------------------------------


def _step_collect(config: AppConfig, db: Database, days: int | None, ctx: dict) -> StepReport:
    from .collection.collector import Collector

    now = datetime.now(timezone.utc)
    from_ts = now - timedelta(days=days or config.collection.default_days)
    collector = Collector(config, db)
    summary = collector.collect(from_ts=from_ts.timestamp(), to_ts=now.timestamp())
    ctx["collect"] = summary
    return StepReport(
        name="collect",
        status="failed" if summary.error else "executed",
        message=f"new={summary.new_samples} dup={summary.duplicate_samples} "
        f"merged={summary.merged_events} fail={summary.failures} "
        f"in {summary.duration_seconds:.1f}s",
    )


def _step_review_sync(config: AppConfig, db: Database, ctx: dict) -> StepReport:
    from .collection.review_sync import ReviewSyncer

    now = datetime.now(UTC)
    from_ts = now - timedelta(days=config.collection.default_days)
    syncer = ReviewSyncer(config, db)
    summary = syncer.sync(from_ts=from_ts.timestamp(), to_ts=now.timestamp())
    ctx["review_sync"] = summary
    return StepReport(
        name="review-sync",
        status="failed" if summary.error else "executed",
        message=f"seen={summary.reviews_seen} matched={summary.samples_matched} "
        f"changed={summary.flags_changed} auto_useful={summary.auto_useful} "
        f"in {summary.duration_seconds:.1f}s",
    )


def _step_verify(config: AppConfig, db: Database, ctx: dict) -> StepReport:
    if not config.vlm.enabled:
        return StepReport(name="verify", status="skipped", message="vlm.enabled=false")
    from .annotation.verifier import Verifier

    verifier = Verifier(config, db)
    summary = verifier.verify()
    ctx["verify"] = summary
    message = (
        f"annotated={summary.annotated} failed={summary.failed} "
        f"objects={summary.objects_written}"
    )
    if summary.error:
        return StepReport(name="verify", status="failed", message=summary.error)
    if summary.failed and not summary.annotated:
        return StepReport(name="verify", status="failed", message=message)
    return StepReport(name="verify", status="executed", message=message)


def _step_build(config: AppConfig, db: Database, ctx: dict) -> StepReport:
    from .dataset.builder import DatasetBuilder

    version = next_build_version(config)
    builder = DatasetBuilder(config, db)
    summary = builder.build(version)
    ctx["version"] = version
    return StepReport(
        name="build",
        status="executed",
        message=f"version={version} total={summary.total} written={summary.images_written}",
        data={"version": version, "written": summary.images_written},
    )


def _step_train(config: AppConfig, db: Database, ctx: dict) -> StepReport:
    version = ctx.get("version")
    if version is None:
        return StepReport(name="train", status="skipped", message="no dataset built this run")
    from .training.trainer import Trainer

    trainer = Trainer(config, db)
    result = trainer.train(
        config.training.model, version, dry_run=ctx["dry_run"], tag="nightly"
    )
    ctx["train"] = result
    return StepReport(
        name="train",
        status="executed",
        message=f"model={result.model_name} best={result.weights_path or 'dry-run'}",
    )


def _step_benchmark(config: AppConfig, db: Database, ctx: dict) -> StepReport:
    try:
        from ultralytics import YOLO  # noqa: F401 (probe availability)
    except ImportError:
        return StepReport(name="benchmark", status="skipped", message="ml extra not installed")
    from .evaluation.backends import UltralyticsBackend
    from .evaluation.benchmark import (
        BenchmarkConfig,
        benchmark_candidate,
        golden_to_examples,
        latest_trained_run,
        results_to_json,
    )
    from .evaluation.golden import GoldenDataset

    golden_path = config.golden_dir() / config.evaluation.golden_dataset
    if not golden_path.is_dir():
        return StepReport(name="benchmark", status="skipped", message=f"golden missing: {golden_path}")
    golden = GoldenDataset.load(golden_path)
    examples = golden_to_examples(golden, config.classes)
    if not examples:
        return StepReport(name="benchmark", status="skipped", message="golden has no examples")

    results = []
    for cand_name in config.evaluation.candidates:
        from .training.candidates import resolve as resolve_candidate

        spec = resolve_candidate(cand_name)
        backend = UltralyticsBackend(
            spec.weights,
            imgsz=config.training.image_size,
            device=ctx.get("device"),
        )
        results.append(
            benchmark_candidate(
                backend,
                examples,
                version=config.evaluation.golden_dataset,
                kind="golden",
                config=BenchmarkConfig(classes=list(config.classes)),
            )
        )
    trained = latest_trained_run(config)
    if trained is not None and not any(r.name == trained[0] for r in results):
        run_name, weights = trained
        backend = UltralyticsBackend(
            str(weights),
            imgsz=config.training.image_size,
            device=ctx.get("device"),
            name=run_name,
        )
        results.append(
            benchmark_candidate(
                backend,
                examples,
                version=config.evaluation.golden_dataset,
                kind="golden",
                config=BenchmarkConfig(classes=list(config.classes)),
            )
        )
    out_path = config.resolve(config.data.root, "benchmark-results.json")
    results_to_json(results, out_path)
    ctx["benchmark"] = results
    summary = "; ".join(
        f"{r.name} mAP50={r.map50:.3f} lat={r.latency_ms:.0f}ms" for r in results
    )
    return StepReport(name="benchmark", status="executed", message=summary,
                      data={"results_path": str(out_path)})


def _step_gate(config: AppConfig, db: Database, ctx: dict) -> StepReport:
    results = ctx.get("benchmark")
    if not results:
        file = config.resolve(config.data.root, "benchmark-results.json")
        if not file.is_file():
            return StepReport(name="gate", status="skipped", message="no benchmark results")
        from .evaluation.benchmark import results_from_json

        results = results_from_json(file)
    from .evaluation.gate import evaluate_gate, record_deployment

    baseline = None
    for r in results:
        if r.name == config.deployment.baseline:
            baseline = r
            break
    verdicts = []
    for result in results:
        ref = baseline if result.name != config.deployment.baseline else None
        gate = evaluate_gate(result, config.deployment, ref)
        record_deployment(config, db, result, gate, version=result.version)
        verdicts.append(f"{result.name}={'PASS' if gate.passed else 'FAIL'}")
    ctx["gate"] = "; ".join(verdicts)
    return StepReport(
        name="gate", status="executed",
        message=" | ".join(verdicts),
        data={"verdicts": verdicts},
    )


def _step_deploy(config: AppConfig, db: Database, ctx: dict) -> StepReport:
    from .deploy.hailo import deploy as deploy_model

    verdicts = [v for v in (ctx.get("gate") or "").split(" | ") if v]
    if verdicts:
        return StepReport(name="deploy", status="skipped", message="gate summary only (no auto-CI deploy)")
    trained = ctx.get("train")
    if trained is None or trained.weights_path is None:
        return StepReport(name="deploy", status="skipped", message="nothing trained this run")
    outcome = deploy_model(
        config,
        db,
        model_name=trained.model_name,
        weights=Path(trained.weights_path),
        version=ctx.get("version") or "",
        imgsz=config.training.image_size,
        dry_run=ctx["dry_run"],
    )
    ctx["deploy"] = outcome
    return StepReport(
        name="deploy",
        status="executed",
        message=f"hef={outcome.hef_path} dry_run={outcome.dry_run}",
    )


__all__ = ["run_pipeline", "next_build_version", "StepReport", "PIPELINE"]