"""``frigate-learn`` command line interface.

Implemented commands (P1):

    frigate-learn db init|migrate|stats|migrations   manage the SQLite DB
    frigate-learn collect ...                         collect from Frigate
    frigate-learn status                              pipeline overview
    frigate-learn web [--host --port]                 run the local webapp dashboard
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import click

from . import __version__
from .collection.collector import CollectSummary, Collector
from .config import AppConfig, load_config
from .db import Database
from .logutil import configure_logging, get_logger
from .times import parse_time_arg

logger = get_logger("frigate_learn.cli")


def _load_config(ctx: click.Context) -> AppConfig:
    config_path = ctx.obj.get("config") if ctx.obj else None
    return load_config(config_path)


def _setup(verbose: bool) -> None:
    configure_logging(verbose=verbose)


def _db(config: AppConfig) -> Database:
    return Database(config.database_path(), migrations_dir=config.migrations_dir())


@click.group()
@click.version_option(__version__, prog_name="frigate-learn")
@click.option("--config", "config_path", type=click.Path(dir_okay=False),
              default=None, help="Path to YAML config (default: config.yaml or FRIGATE_LEARN_CONFIG).")
@click.option("-v", "--verbose", is_flag=True, help="Verbose (debug) logging.")
@click.pass_context
def cli(ctx: click.Context, config_path: str | None, verbose: bool) -> None:
    """Active-learning / auto-training pipeline for Frigate NVR."""
    ctx.ensure_object(dict)
    ctx.obj["config"] = config_path
    _setup(verbose)


# --- database --------------------------------------------------------------


@cli.group()
def db() -> None:
    """Database management."""


@db.command("init")
@click.pass_context
def db_init(ctx: click.Context) -> None:
    """Create the database and apply all migrations."""
    config = _load_config(ctx)
    database = _db(config)
    applied = database.init()
    if applied is not None:  # init() applies pending
        pass
    click.echo(f"Database ready: {database.path}")
    click.echo(f"Schema version: {database.schema_version()}")
    remaining = database.pending_migrations()
    if remaining:
        click.echo(f"Pending migrations: {[f.name for f in remaining]}")


@db.command("migrate")
@click.pass_context
def db_migrate(ctx: click.Context) -> None:
    """Apply pending migrations."""
    config = _load_config(ctx)
    database = _db(config)
    applied = database.migrate()
    if applied:
        click.echo("Applied:")
        for name in applied:
            click.echo(f"  {name}")
    else:
        click.echo("No pending migrations.")
    click.echo(f"Schema version: {database.schema_version()}")


@db.command("migrations")
@click.pass_context
def db_migrations(ctx: click.Context) -> None:
    """List applied and pending migrations."""
    config = _load_config(ctx)
    database = _db(config)
    click.echo("Migrations:")
    for name in database.applied_migrations():
        click.echo(f"  [applied]  {name}")
    for path in database.pending_migrations():
        click.echo(f"  [pending]  {path.name}")


@db.command("stats")
@click.pass_context
def db_stats(ctx: click.Context) -> None:
    """Show database statistics."""
    config = _load_config(ctx)
    database = _db(config)
    database.migrate()
    from sqlalchemy import func, text

    with database.session() as session:
        total = session.execute(text("SELECT COUNT(*) FROM samples")).scalar()
        annotated = session.execute(
            text("SELECT COUNT(*) FROM annotations WHERE verified = 0")
        ).scalar()
        failures = session.execute(text("SELECT COUNT(*) FROM collection_failures")).scalar()
        jobs = session.execute(text("SELECT COUNT(*) FROM jobs")).scalar()
    click.echo(f"Database:      {database.path}")
    click.echo(f"Schema:        v{database.schema_version()}")
    click.echo(f"Samples:       {total}")
    click.echo(f"Annotations:   {annotated}")
    click.echo(f"Failures:      {failures}")
    click.echo(f"Jobs:          {jobs}")
    click.echo()
    click.echo("Samples by camera:")
    with database.session() as session:
        rows = session.execute(
            text("SELECT camera, COUNT(*) FROM samples GROUP BY camera ORDER BY 2 DESC LIMIT 20")
        )
        for camera, count in rows:
            click.echo(f"  {camera or '?'}: {count}")


# --- collection ------------------------------------------------------------


def _summary_lines(summary: CollectSummary) -> list[str]:
    def fmt_ts(ts: float | None) -> str:
        if ts is None:
            return "?"
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")

    lines = [
        "Collecting Frigate reviews",
        f"Range: {fmt_ts(summary.range_from)} -> {fmt_ts(summary.range_to)}",
        "",
        f"Reviews found: {summary.reviews_found:,}",
        f"Reviews selected: {summary.reviews_selected:,}",
        f"Events found: {summary.events_found:,}",
        f"New samples: {summary.new_samples:,}",
        f"Duplicates: {summary.duplicate_samples:,}",
        f"Failures: {summary.failures:,}",
    ]
    if summary.skipped_no_box:
        lines.append(f"No-box skipped: {summary.skipped_no_box:,}")
    if summary.error:
        lines.append(f"Error: {summary.error}")
    lines.append("")
    lines.append(f"Done in {summary.duration_seconds:.1f}s.")
    return lines


@cli.command()
@click.option("--from", "from_arg", default=None, help="Start time: '7 days ago', '2026-09-01', ISO, ...")
@click.option("--to", default=None, help="End time (default: now).")
@click.option("--days", type=int, default=None, help="Collect the last N days (alternative to --from).")
@click.option("--camera", "cameras", multiple=True, help="Camera name (repeatable).")
@click.option("--label", "labels", multiple=True, help="Label (repeatable).")
@click.option("--severity", "severities", multiple=True,
              type=click.Choice(["alert", "detection"]), help="Severity (repeatable).")
@click.option("--limit", type=int, default=None, help="Max reviews to process.")
@click.option("--concurrency", type=int, default=None, help="Parallel downloads (default: config).")
@click.option("--no-region-crop", "no_region_crop", is_flag=True, default=False,
              help="Collect full-frame clean snapshots instead of region crops.")
@click.pass_context
def collect(
    ctx: click.Context,
    from_arg: str | None,
    to: str | None,
    days: int | None,
    cameras: tuple[str, ...],
    labels: tuple[str, ...],
    severities: tuple[str, ...],
    limit: int | None,
    concurrency: int | None,
    no_region_crop: bool,
) -> None:
    """Collect review/event data from Frigate into the local dataset."""
    config = _load_config(ctx)
    if no_region_crop:
        config.collection.region_crop = False

    now = datetime.now(timezone.utc).timestamp()
    if from_arg is not None:
        from_ts = parse_time_arg(from_arg)
    elif days is not None:
        from_ts = (datetime.now(timezone.utc) - timedelta(days=days)).timestamp()
    else:
        from_ts = now - timedelta(days=config.collection.default_days).total_seconds()
    to_ts = parse_time_arg(to) if to is not None else None
    if to_ts is not None and to_ts <= from_ts:
        raise click.ClickException("--to must be after --from")

    database = _db(config)
    collector = Collector(config, database)
    severity = list(severities) if severities else None

    def progress(message: str) -> None:
        click.echo(message)

    try:
        summary = collector.collect(
            from_ts=from_ts,
            to_ts=to_ts,
            cameras=list(cameras) or None,
            labels=list(labels) or None,
            severity=severity,
            limit=limit if limit is not None else config.collection.max_reviews,
            concurrency=concurrency,
            progress=progress,
        )
    except Exception as exc:  # fatal (e.g. bad token / connection refused)
        raise click.ClickException(f"collection failed: {exc}")

    for line in _summary_lines(summary):
        click.echo(line)
    if summary.error:
        sys.exit(1)


# --- status ----------------------------------------------------------------


@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Overview of collected data and recent jobs."""
    config = _load_config(ctx)
    database = _db(config)
    database.migrate()
    from sqlalchemy import text

    click.echo(f"Database: {database.path}")
    click.echo(f"Schema:   v{database.schema_version()}")
    with database.session() as session:
        total = session.execute(text("SELECT COUNT(*) FROM samples")).scalar()
        by_status = session.execute(
            text("SELECT status, COUNT(*) FROM samples GROUP BY status ORDER BY 2 DESC")
        ).all()
        failures = session.execute(text("SELECT COUNT(*) FROM collection_failures")).scalar()
        jobs = session.execute(
            text(
                "SELECT id, type, status, started_at, finished_at, error FROM jobs "
                "ORDER BY started_at DESC LIMIT 10"
            )
        ).all()
    click.echo(f"Samples:  {total} (failures recorded: {failures})")
    for status_, count in by_status:
        click.echo(f"  {status_ or '?'}: {count}")
    click.echo()
    click.echo("Recent jobs:")
    for job in jobs:
        job_id, jtype, jstatus, started, finished, err = job
        marker = "ok " if jstatus == "finished" else ("ERR" if jstatus == "failed" else "RUN")
        click.echo(f"  [{marker}] {jtype} {job_id[:8]} {started or ''}"
                   + (f" error={err[:60]}" if err else ""))
    if not jobs:
        click.echo("  (none yet — run `frigate-learn collect`)")


# --- inspection / triage ---------------------------------------------------


@cli.command()
@click.option("--html", "html_path", type=click.Path(dir_okay=True, file_okay=False),
              default=None, help="Write a static HTML report into this directory.")
@click.option("--days", type=int, default=None, help="Only samples newer than N days.")
@click.option("--camera", "cameras", multiple=True, help="Camera (repeatable).")
@click.option("--label", "labels", multiple=True, help="Label (repeatable).")
@click.option("--quality", type=str, default=None, help="Only samples with this quality verdict.")
@click.pass_context
def inspect(
    ctx: click.Context,
    html_path: str | None,
    days: int | None,
    cameras: tuple[str, ...],
    labels: tuple[str, ...],
    quality: str | None,
) -> None:
    """Inspect the collected dataset (terminal or static HTML report)."""
    from .inspect.report import collect_records, render_html_report

    config = _load_config(ctx)
    database = _db(config)
    database.migrate()

    if quality is not None:
        from .inspect.triage import validate_quality

        validate_quality(quality)

    records = collect_records(
        config, database,
        cameras=list(cameras) or None,
        days=days,
        labels=list(labels) or None,
        quality=quality,
    )
    if html_path:
        from .inspect.report import ReportSummary

        summary: ReportSummary = render_html_report(records, html_path)
        click.echo(f"Report written: {summary.out_dir / 'index.html'}")
        click.echo(f"Frames: {summary.total} | cameras: {', '.join(summary.cameras)}")
        click.echo("Review in a browser, then mark frames with `triage` commands.")
        return

    if not records:
        click.echo("No matching samples. Run `frigate-learn collect` first.")
        return
    for rec in records[:50]:
        q = rec.quality or "unset"
        ts = datetime.fromtimestamp(rec.timestamp, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        score = rec.score if rec.score is not None else 0.0
        click.echo(
            f"{rec.sample_id[:8]}  {rec.camera:<14} {rec.label or '?':<12} "
            f"({score:.2f})  {ts}  q={q}"
        )
    if len(records) > 50:
        click.echo(f"... and {len(records) - 50} more (use --html for the full gallery).")


@cli.group()
def triage() -> None:
    """Set/clear human quality verdicts on samples (useful|bad|duplicate|ignore)."""


@triage.command("set")
@click.argument("sample_id")
@click.argument("quality", type=click.Choice(sorted({"useful", "bad", "duplicate", "ignore"})))
@click.pass_context
def triage_set(ctx: click.Context, sample_id: str, quality: str) -> None:
    """Set the quality verdict for one sample."""
    from .inspect.triage import set_sample_quality

    config = _load_config(ctx)
    if not set_sample_quality(config, _db(config), sample_id, quality):
        raise click.ClickException(f"unknown sample id: {sample_id}")
    click.echo(f"{sample_id[:8]} -> {quality}")


@triage.command("unset")
@click.argument("sample_id")
@click.pass_context
def triage_unset(ctx: click.Context, sample_id: str) -> None:
    """Clear the quality verdict for one sample."""
    from .inspect.triage import unset_quality

    config = _load_config(ctx)
    if not unset_quality(config, _db(config), sample_id):
        raise click.ClickException(f"unknown sample id: {sample_id}")
    click.echo(f"{sample_id[:8]} -> unset")


@triage.command("batch")
@click.argument("quality", type=click.Choice(sorted({"useful", "bad", "duplicate", "ignore"})))
@click.option("--clear", is_flag=True, help="Clear quality instead of setting it.")
@click.option("--camera", "cameras", multiple=True)
@click.option("--days", type=int, default=None)
@click.option("--label", "labels", multiple=True)
@click.option("--limit", type=int, default=None)
@click.pass_context
def triage_batch(
    ctx: click.Context,
    quality: str,
    clear: bool,
    cameras: tuple[str, ...],
    days: int | None,
    labels: tuple[str, ...],
    limit: int | None,
) -> None:
    """Set (or --clear) the quality verdict across a filtered batch."""
    from .inspect.triage import set_batch_quality

    config = _load_config(ctx)
    updated = set_batch_quality(
        config, _db(config), quality,
        cameras=list(cameras) or None, days=days, labels=list(labels) or None,
        limit=limit, clear=clear,
    )
    click.echo(f"Updated {updated} sample(s).")


# --- verification ----------------------------------------------------------


@cli.command()
@click.option("--force", is_flag=True, help="Re-verify samples already annotated.")
@click.option("--limit", type=int, default=None)
@click.option("--camera", "cameras", multiple=True)
@click.option("--label", "labels", multiple=True)
@click.option("--days", type=int, default=None)
@click.pass_context
def verify(
    ctx: click.Context,
    force: bool,
    limit: int | None,
    cameras: tuple[str, ...],
    labels: tuple[str, ...],
    days: int | None,
) -> None:
    """Verify unverified samples with the VLM; writes verified annotations.

    VLM failures are dropped (sample quality -> ``bad``) so they never block a
    build or get re-submitted; rescue them later with ``triage set``.
    """
    from .annotation.verifier import Verifier

    config = _load_config(ctx)
    if not config.vlm.enabled or not config.vlm.base_url:
        raise click.ClickException(
            "vlm.enabled=false or vlm.base_url missing in config; set both to verify."
        )
    database = _db(config)
    database.migrate()
    summary = Verifier(config, database).verify(
        force=force, limit=limit,
        cameras=list(cameras) or None, labels=list(labels) or None, days=days,
    )
    click.echo(
        f"processed={summary.processed} annotated={summary.annotated} "
        f"skipped={summary.skipped} failed={summary.failed} "
        f"dropped={summary.dropped} objects={summary.objects_written}"
    )
    if summary.error:
        raise click.ClickException(f"verify failed: {summary.error}")


# --- dataset management ----------------------------------------------------


@cli.group()
def dataset() -> None:
    """Build and list frozen training dataset versions."""


@dataset.command("build")
@click.argument("version")
@click.option("--camera", "cameras", multiple=True)
@click.option("--label", "labels", multiple=True)
@click.option("--quality", default=None, help="Only samples with this verdict (default: useful/unset).")
@click.option("--verified-only", is_flag=True, help="Only samples with a verified annotation.")
@click.option("--max-per-class", type=int, default=None)
@click.option("--seed", default=None)
@click.option("--overwrite", is_flag=True)
@click.pass_context
def dataset_build(
    ctx: click.Context,
    version: str,
    cameras: tuple[str, ...],
    labels: tuple[str, ...],
    quality: str | None,
    verified_only: bool,
    max_per_class: int | None,
    seed: str | None,
    overwrite: bool,
) -> None:
    """Build dataset version VERSION (e.g. v001) from the sample pool."""
    from .dataset.builder import DatasetBuilder

    config = _load_config(ctx)
    database = _db(config)
    database.migrate()
    summary = DatasetBuilder(config, database).build(
        version,
        cameras=list(cameras) or None,
        labels=list(labels) or None,
        quality=quality,
        verified_only=verified_only,
        max_per_class=max_per_class,
        seed=seed,
        overwrite=overwrite,
    )
    split = ", ".join(f"{s}={c}" for s, c in summary.split_counts.items())
    click.echo(
        f"Built {version}: total={summary.total} written={summary.images_written} "
        f"verified={summary.verified} skipped(no_box={summary.skipped_no_box} "
        f"class={summary.skipped_class} cap={summary.skipped_cap})"
    )
    click.echo(f"Splits: {split or 'none'}")
    click.echo(f"Root: {summary.root}")


@dataset.command("list")
@click.pass_context
def dataset_list(ctx: click.Context) -> None:
    """List built dataset versions and their build summaries."""
    import json

    config = _load_config(ctx)
    root = config.datasets_dir()
    if not root.is_dir():
        click.echo("No datasets yet.")
        return
    for entry in sorted(root.iterdir(), reverse=True):
        if not entry.is_dir():
            continue
        build_json = entry / "build.json"
        if build_json.is_file():
            payload = json.loads(build_json.read_text(encoding="utf-8"))
            s = payload.get("summary", {})
            click.echo(
                f"{entry.name}: written={s.get('images_written', 0)} "
                f"epoch={payload.get('built_at', '?')[:19]}"
            )
        else:
            click.echo(f"{entry.name}: <no build.json>")


# --- discovery / external import -------------------------------------------


@cli.command()
@click.option("--days", type=int, default=1, help="Look back N days (default 1).")
@click.option("--camera", "cameras", multiple=True)
@click.option("--motion-threshold", type=float, default=0.3)
@click.option("--closing-gap", type=float, default=60.0,
              help="Seconds between motion buckets that still count as one window.")
@click.option("--store", is_flag=True, help="Persist windows to SQLite.")
@click.pass_context
def discover(
    ctx: click.Context,
    days: int,
    cameras: tuple[str, ...],
    motion_threshold: float,
    closing_gap: float,
    store: bool,
) -> None:
    """Find motion buckets never seen by detection (potential missed objects)."""
    from .collection.discover import find_motion_windows, store_windows
    from .frigate.client import FrigateClient

    config = _load_config(ctx)
    database = _db(config)
    database.migrate()
    client = FrigateClient(
        base_url=config.frigate.base_url,
        token=config.frigate.token,
        timeout_seconds=config.frigate.timeout_seconds,
    )
    now = datetime.now(timezone.utc).timestamp()
    windows = find_motion_windows(
        client, config, database,
        after=now - days * 86400.0,
        before=now,
        cameras=list(cameras) or None,
        motion_threshold=motion_threshold,
        closing_gap=closing_gap,
    )
    missed = [w for w in windows if not w.covered]
    click.echo(f"Motion windows: {len(windows)} (uncovered potential misses: {len(missed)})")
    for w in missed[:50]:
        ts = datetime.fromtimestamp(w.start, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        click.echo(f"  {w.camera:<14} {ts} peak={w.peak_motion:.2f} buckets={w.buckets}")
    if store and missed:
        stored = store_windows(database, missed)
        click.echo(f"Stored {stored} discovery window(s).")
    elif store:
        click.echo("Nothing to store.")


@cli.command("import-coco")
@click.argument("annotations_json", type=click.Path(exists=True, dir_okay=False))
@click.argument("images_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--camera", required=True, help="Store images under this camera key.")
@click.option("--class-map", "class_map", multiple=True, default=None,
              help="COCO category=configclass (repeatable).")
@click.pass_context
def import_coco_cmd(
    ctx: click.Context,
    annotations_json: str,
    images_dir: str,
    camera: str,
    class_map: tuple[str, ...],
) -> None:
    """Import a COCO annotations JSON + image folder as external samples."""
    from .collection.external import import_coco

    config = _load_config(ctx)
    database = _db(config)
    database.migrate()
    mapping: dict[str, str] = {}
    for item in class_map:
        if "=" in item:
            k, _, v = item.partition("=")
            mapping[k.strip()] = v.strip()
    summary = import_coco(
        config, database, annotations_json, images_dir,
        camera=camera, class_map=mapping or None,
    )
    click.echo(
        f"images={summary.images} objects={summary.objects} "
        f"written={summary.samples_written} "
        f"skipped(missing={summary.skipped_missing} unmapped={summary.skipped_unmapped})"
    )


# --- training / evaluation --------------------------------------------------


def _require_ml() -> None:
    try:
        import ultralytics  # noqa: F401 (probe)
    except ImportError:
        raise click.ClickException(
            "ultralytics/torch not installed here. On the training machine run: "
            "pip install -e '.[ml]'"
        )


@cli.command()
@click.option("--candidate", "candidates", multiple=True,
              help="Candidate to benchmark (default: config evaluation.candidates).")
@click.option("--golden-version", default=None, help="Golden dataset version.")
@click.option("--limit", type=int, default=None, help="Cap the number of golden examples.")
@click.option("--dry-run", is_flag=True, help="Dry-run: write results without loading models.")
@click.pass_context
def benchmark(
    ctx: click.Context,
    candidates: tuple[str, ...],
    golden_version: str | None,
    limit: int | None,
    dry_run: bool,
) -> None:
    """Benchmark candidates on the golden dataset; write benchmark-results.json."""
    from .evaluation.backends import UltralyticsBackend
    from .evaluation.benchmark import (
        BenchmarkConfig,
        benchmark_candidate,
        golden_to_examples,
        results_to_json,
    )
    from .evaluation.golden import GoldenDataset
    from .training.candidates import resolve as resolve_candidate

    config = _load_config(ctx)
    database = _db(config)
    golden = GoldenDataset.load(
        config.golden_dir() / (golden_version or config.evaluation.golden_dataset)
    )
    examples = golden_to_examples(golden, config.classes, limit=limit)
    if not examples:
        raise click.ClickException(f"golden {golden.root} has no usable examples")
    names = list(candidates) or config.evaluation.candidates

    results = []
    for name in names:
        spec = resolve_candidate(name)
        if dry_run:
            result = benchmark_candidate(
                _DryBackend(spec.name),
                examples, version=golden.root.name, kind="golden",
                config=BenchmarkConfig(classes=list(config.classes)),
            )
            result.metrics.latency_ms = 0.0
            results.append(result)
            continue
        _require_ml()
        backend = UltralyticsBackend(
            spec.weights, imgsz=config.training.image_size, device=config.training.device
        )
        results.append(
            benchmark_candidate(
                backend, examples, version=golden.root.name, kind="golden",
                config=BenchmarkConfig(classes=list(config.classes)),
            )
        )
    out_path = config.resolve(config.data.root, "benchmark-results.json")
    results_to_json(results, out_path)
    for r in results:
        marker = "dry-run" if dry_run else "eval"
        latency = r.latency_ms if r.latency_ms is not None else 0.0
        click.echo(
            f"[{marker}] {r.name:<10} mAP50={r.map50:.3f} recall={r.recall:.3f} "
            f"latency={latency:.1f}ms"
        )
    click.echo(f"Wrote {out_path}")


class _DryBackend:
    """Returns no predictions (dry-run benchmark placeholder)."""

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def predict(self, image_path, confidence=0.25):  # noqa: ARG001
        return []


@cli.command()
@click.argument("model")
@click.argument("dataset_version")
@click.option("--epochs", type=int, default=None)
@click.option("--imgsz", type=int, default=None)
@click.option("--batch", type=int, default=None)
@click.option("--device", default=None)
@click.option("--freeze", type=int, default=None)
@click.option("--dry-run", "dry_run", flag_value=True, default=True, help="Dry-run (default).")
@click.option("--real", "dry_run", flag_value=False, help="Actually train (needs ultralytics).")
@click.pass_context
def train(
    ctx: click.Context,
    model: str,
    dataset_version: str,
    epochs: int | None,
    imgsz: int | None,
    batch: int | None,
    device: str | None,
    freeze: int | None,
    dry_run: bool,
) -> None:
    """Train MODEL on a built DATASET_VERSION (e.g. yolov8n v001)."""
    from .training.trainer import Trainer

    config = _load_config(ctx)
    if not dry_run:
        _require_ml()
    database = _db(config)
    database.migrate()
    result = Trainer(config, database).train(
        model, dataset_version,
        epochs=epochs, imgsz=imgsz, batch=batch, device=device, freeze=freeze,
        dry_run=dry_run,
    )
    click.echo(f"model={result.model_name} version={result.dataset_version}")
    click.echo(f"out={result.output_dir}")
    click.echo(
        f"weights={result.weights_path}"
        if result.weights_path
        else f"dry-run (no weights; install '.[ml]' to train)"
    )


@cli.command()
@click.option("--results", "results_path", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--baseline", default="yolov8l", help="Baseline candidate name for deltas.")
@click.pass_context
def gate(ctx: click.Context, results_path: str | None, baseline: str) -> None:
    """Evaluate benchmark results against the deployment envelope; record verdicts."""
    from .evaluation.benchmark import results_from_json
    from .evaluation.gate import evaluate_gate, record_deployment

    config = _load_config(ctx)
    database = _db(config)
    database.migrate()
    path = results_path or str(config.resolve(config.data.root, "benchmark-results.json"))
    results = results_from_json(Path(path))
    if not results:
        raise click.ClickException(f"no results in {path}")
    baseline_result = next((r for r in results if r.name == baseline), None)
    for r in results:
        ref = baseline_result if r.name != baseline else None
        res = evaluate_gate(r, config.deployment, ref)
        record_deployment(config, database, r, res, version=r.version)
        click.echo(f"[{'PASS' if res.passed else 'FAIL'}] {r.name}" + ("  (baseline)" if r.name == baseline else ""))
        for reason in res.reasons:
            click.echo(f"    - {reason}")


# --- deploy -----------------------------------------------------------------


@cli.command()
@click.argument("model")
@click.argument("weights", type=click.Path(exists=True))
@click.option("--version", default="")
@click.option("--imgsz", type=int, default=None)
@click.option("--out-dir", type=click.Path(file_okay=False), default=None)
@click.option("--dry-run", "dry_run", flag_value=True, default=True, help="Placeholder artifacts (default).")
@click.option("--real", "dry_run", flag_value=False, help="Run the Hailo/ONNX export on this host.")
@click.pass_context
def deploy(
    ctx: click.Context,
    model: str,
    weights: str,
    version: str,
    imgsz: int | None,
    out_dir: str | None,
    dry_run: bool,
) -> None:
    """Export a trained model to a deployable Hailo/Frigate artifact."""
    from .deploy.hailo import deploy as deploy_model

    config = _load_config(ctx)
    database = _db(config)
    database.migrate()
    outcome = deploy_model(
        config, database,
        model_name=model,
        weights=Path(weights),
        version=version,
        imgsz=imgsz or config.training.image_size,
        dry_run=dry_run,
        out_dir=Path(out_dir) if out_dir else None,
    )
    click.echo(f"onnx={outcome.onnx_path}")
    click.echo(f"hef={outcome.hef_path}")
    click.echo(f"manifest={outcome.manifest_path}")
    click.echo()
    click.echo(outcome.config_snippet)


# --- run --------------------------------------------------------------------


@cli.command()
@click.option("--steps", "steps", multiple=True, help="Pipeline stage(s) to run (repeatable).")
@click.option("--until", default=None, help="Run stages up to and including this one.")
@click.option("--dry-run", is_flag=True, help="Dry-run: placeholders wherever real ML/hardware is needed.")
@click.option("--real", "no_dry_run", is_flag=True, help="Turn off dry-run default.")
@click.option("--keep-going", is_flag=True, help="Continue past a failed stage.")
@click.option("--days", type=int, default=None, help="Look back N days for collection.")
@click.pass_context
def run(
    ctx: click.Context,
    steps: tuple[str, ...],
    until: str | None,
    dry_run: bool,
    no_dry_run: bool,
    keep_going: bool,
    days: int | None,
) -> None:
    """Run the automated learning pipeline (collect → verify → build → train → benchmark → gate → deploy)."""
    from .run import PIPELINE, run_pipeline

    config = _load_config(ctx)
    database = _db(config)
    selected: list[str] | None = list(steps) or None
    if selected:
        unknown = [s for s in selected if s not in PIPELINE]
        if unknown:
            raise click.ClickException(f"unknown stage(s): {', '.join(unknown)}; choose from {', '.join(PIPELINE)}")
    dry = dry_run and not no_dry_run
    reports = run_pipeline(
        config, database, steps=selected, until=until, dry_run=dry,
        keep_going=keep_going, days=days,
    )
    for report in reports:
        marker = {"executed": "ok ", "skipped": "-- ", "failed": "ERR"}[report.status]
        click.echo(f"[{marker}] {report.name:<10} {report.message}")
    if any(r.status == "failed" for r in reports):
        sys.exit(1)


# --- webapp -----------------------------------------------------------------


@cli.command()
@click.option("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1).")
@click.option("--port", type=int, default=8080, help="Port to listen on (default: 8080).")
def web(host: str, port: int) -> None:
    """Run the local webapp dashboard."""
    import uvicorn

    from .webapp.app import create_app

    uvicorn.run(create_app(), host=host, port=port)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()