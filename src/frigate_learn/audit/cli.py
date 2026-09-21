"""Click subcommands for the audit pipeline.

Registers as a ``click.Group`` named ``audit`` in the main CLI, plus
a top-level alias ``audit-dataset`` that maps to ``audit run``.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import click

from ..config import load_config
from ..logutil import JobLogTailHandler, job_metadata_log_tail


@click.group(name="audit", help="Dataset audit and pseudo-labeling pipeline.")
def audit_group() -> None:
    """Dataset audit and pseudo-labeling pipeline."""


@audit_group.command(name="run", help="Run the audit pipeline on a dataset.")
@click.option("--input", "input_path", type=click.Path(exists=True, path_type=Path),
              help="Root of the input dataset (Frigate DB or manifest).")
@click.option("--output", "output_path", type=click.Path(path_type=Path),
              help="Audit output directory (caches, decisions, provenance).")
@click.option("--training-output", "training_path", type=click.Path(path_type=Path),
              help="Training export root (positive/ + hard_negative/).")
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path),
              help="Config YAML file.")
@click.option("--resume", is_flag=True, default=False,
              help="Skip samples with a valid cached KEEP/DROP decision.")
@click.option("--limit", "sample_limit", type=int, default=None,
              help="Process at most N new samples (after class filter).")
@click.option("--class", "class_filter", multiple=True,
              help="Only process this Frigate class (repeatable).")
@click.option("--sam-only", is_flag=True, default=False,
              help="Run SAM only; do not call the VLM.")
@click.option("--stats-json", is_flag=True, default=False,
              help="Print final stats as JSON instead of text.")
@click.option("--no-hard-negatives", is_flag=True, default=False,
              help="Suppress hard-negative export even when enabled in config.")
@click.option("--negative-samples", "neg_samples", type=int, default=None,
              help="Override random background samples per crop (0 disables).")
def run_cmd(
    input_path: Path | None,
    output_path: Path | None,
    training_path: Path | None,
    config_path: Path | None,
    resume: bool,
    sample_limit: int | None,
    class_filter: tuple[str, ...],
    sam_only: bool,
    stats_json: bool,
    no_hard_negatives: bool,
    neg_samples: int | None,
) -> None:
    """Execute the full audit pipeline."""
    config = load_config(config_path)
    audit = config.audit

    in_path = input_path or config.resolve(audit.input)
    out_path = output_path or config.resolve(audit.output)
    trn_path = training_path or config.resolve(audit.training)

    in_path = Path(in_path)
    out_path = Path(out_path)
    trn_path = Path(trn_path)

    if not in_path.exists():
        raise click.ClickException(f"Input path does not exist: {in_path}")

    from .runner import (
        AuditRunError,
        build_adapter,
        build_gates,
        build_reconciler,
        build_sam_teacher,
    )

    try:
        adapter = build_adapter(in_path)
        sam_teacher = build_sam_teacher(audit)
        reconciler = build_reconciler(audit)
        gates = build_gates(audit)
    except AuditRunError as exc:
        raise click.ClickException(str(exc)) from exc

    hn_enabled = audit.hard_negatives_enabled if not no_hard_negatives else False
    neg_enabled = audit.negative_sampling_enabled
    neg_per_crop = neg_samples if neg_samples is not None else audit.samples_per_crop

    from .pipeline import AuditPipeline, PipelineAbort, PipelineOptions

    pipeline = AuditPipeline(
        adapter=adapter,
        sam_teacher=sam_teacher,
        reconciler=reconciler,
        audit_root=out_path,
        training_root=trn_path,
        pipeline_version=audit.pipeline_version,
        gates=gates,
        hard_negatives_enabled=hn_enabled,
        negative_sampling_enabled=neg_enabled,
        samples_per_crop=neg_per_crop,
    )

    options = PipelineOptions(
        limit=sample_limit,
        classes=list(class_filter),
        sam_only=sam_only,
        resume=resume,
        max_consecutive_vlm_errors=audit.max_consecutive_vlm_errors,
    )

    ledger = _ledger_db_path(config)
    job_id = _start_audit_job(ledger, {
        "input": str(in_path),
        "output": str(out_path),
        "limit": sample_limit,
        "sam_only": sam_only,
        "resume": resume,
    })
    run_logger = logging.getLogger("frigate_learn")
    tail_handler = JobLogTailHandler(ledger, job_id) if (ledger and job_id) else None
    if tail_handler is not None:
        run_logger.addHandler(tail_handler)
    try:
        stats = pipeline.run(options)
    except PipelineAbort as exc:
        _finish_audit_job(ledger, job_id, "failed", str(exc), {})
        raise click.ClickException(str(exc))
    except Exception as exc:
        _finish_audit_job(ledger, job_id, "failed", str(exc), {})
        raise
    else:
        _finish_audit_job(ledger, job_id, "finished", None, stats.as_dict())
    finally:
        if tail_handler is not None:
            run_logger.removeHandler(tail_handler)

    if stats_json:
        click.echo(json.dumps(stats.as_dict(), indent=2, sort_keys=True))
    else:
        from .stats import render_stats

        click.echo(render_stats(stats))


@audit_group.command(name="stats", help="Show stats from a previous audit run.")
@click.argument("audit_dir", type=click.Path(exists=True, path_type=Path))
@click.option("--json", "as_json", is_flag=True, default=False,
              help="Output as JSON instead of text.")
def stats_cmd(audit_dir: Path, as_json: bool) -> None:
    """Print aggregate statistics from decision.json files."""
    from .stats import aggregate_to_dict, render_stats, scan_audit_dir

    if as_json:
        click.echo(json.dumps(aggregate_to_dict(audit_dir), indent=2, sort_keys=True))
    else:
        stats = scan_audit_dir(audit_dir)
        click.echo(render_stats(stats))


def _ledger_db_path(config) -> str | None:
    path = Path(config.database_path())
    if not path.is_file():
        return None
    try:
        con = sqlite3.connect(path, timeout=5)
        row = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='jobs'"
        ).fetchone()
        con.close()
    except sqlite3.Error:
        return None
    return str(path) if row else None


def _start_audit_job(db_path: str | None, meta: dict) -> str | None:
    if db_path is None:
        return None
    job_id = uuid4().hex[:12]
    con = sqlite3.connect(db_path, timeout=5)
    try:
        con.execute(
            "INSERT INTO jobs (id, type, status, started_at, metadata_json) "
            "VALUES (?, 'audit', 'running', ?, ?)",
            (job_id, datetime.now(timezone.utc).isoformat(), json.dumps(meta)),
        )
        con.commit()
    finally:
        con.close()
    return job_id


def _finish_audit_job(
    db_path: str | None,
    job_id: str | None,
    status: str,
    error: str | None,
    stats: dict,
) -> None:
    if db_path is None or job_id is None:
        return
    final = dict(stats)
    tail: list[str] = []
    con = sqlite3.connect(db_path, timeout=5)
    try:
        row = con.execute(
            "SELECT metadata_json FROM jobs WHERE id=?", (job_id,)
        ).fetchone()
        if row is not None and row[0]:
            try:
                meta = json.loads(row[0])
            except (ValueError, TypeError):
                meta = {}
            if isinstance(meta, dict):
                tail = job_metadata_log_tail(meta)
        if tail:
            final["log_tail"] = tail
        con.execute(
            "UPDATE jobs SET status=?, finished_at=?, error=?, metadata_json=? "
            "WHERE id=?",
            (
                status,
                datetime.now(timezone.utc).isoformat(),
                error,
                json.dumps(final),
                job_id,
            ),
        )
        con.commit()
    finally:
        con.close()


__all__ = ["audit_group", "run_cmd", "stats_cmd"]
