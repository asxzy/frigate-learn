"""Programmatic audit-pipeline entry point shared by the CLI and webapp.

Builds the dataset adapter, SAM teacher, VLM reconciler and geometry gates
from config, then executes the reconciliation pipeline. The CLI adds click
plumbing and a jobs-ledger row around it; the webapp runs it inside a
JobManager thread so the dashboard can record and stream the run like any
other job.
"""

from __future__ import annotations

from pathlib import Path

from ..config import AppConfig


class AuditRunError(Exception):
    """Raised when the audit run cannot start (missing input, bad config)."""


def build_adapter(input_path: Path):
    """Build a DatasetAdapter from the input path.

    Recognizes, in order: a SQLite database file (``data/frigate.db`` or
    ``data/frigate_learn.db``), a dataset root containing either database
    (images resolved against ``<root>/images``), and a manifest root
    (``annotations/manifest.json`` or ``manifest.json`` at the root).
    """
    from .adapter import FrigateDatabaseAdapter, ManifestDatasetAdapter

    if input_path.is_file() and input_path.suffix == ".db":
        return FrigateDatabaseAdapter(input_path)
    for db_name in ("frigate_learn.db", "frigate.db"):
        db_file = input_path / db_name
        if db_file.is_file():
            return FrigateDatabaseAdapter(db_file, images_root=input_path / "images")
    if ((input_path / "annotations" / "manifest.json").is_file()
            or (input_path / "manifest.json").is_file()):
        return ManifestDatasetAdapter(input_path)
    raise AuditRunError(
        f"No database (frigate_learn.db|frigate.db) or manifest.json found in {input_path}"
    )


def build_sam_teacher(audit):
    """Build the SAM teacher from config (local MLX or remote endpoint)."""
    if audit.sam.backend == "http":
        from .sam import HttpSamTeacher

        if not audit.sam.base_url:
            raise AuditRunError(
                "audit.models.sam.backend=http requires audit.models.sam.base_url"
                " (URL of a frigate-learn sam-server endpoint)"
            )
        return HttpSamTeacher(
            base_url=audit.sam.base_url,
            api_key=audit.sam.api_key,
            model=audit.sam.model,
            timeout_seconds=audit.sam.timeout_seconds,
            max_retries=audit.sam.max_retries,
        )
    from .sam import MlxSam3Teacher

    return MlxSam3Teacher(
        checkpoint_path=audit.sam.checkpoint or None,
        load_from_hf=audit.sam.load_from_hf,
        hf_repo=audit.sam.hf_repo,
        quantize_bits=audit.sam.quantize_bits,
        resolution=audit.sam.resolution,
        confidence_threshold=audit.sam.confidence_threshold,
        candidate_classes=list(audit.sam.candidate_classes),
    )


def build_reconciler(audit):
    """Build the VLM reconciler from config."""
    from .vlm import build_reconciler as _build_reconciler

    return _build_reconciler(
        base_url=audit.vlm.base_url,
        model=audit.vlm.model,
        api_key=audit.vlm.api_key,
        temperature=audit.vlm.temperature,
        timeout_seconds=audit.vlm.timeout_seconds,
        max_retries=audit.vlm.max_retries,
        json_mode=audit.vlm.json_mode,
    )


def build_gates(audit):
    """Build GeometryThresholds from config."""
    from .decision import GeometryThresholds

    return GeometryThresholds(
        min_mask_area=audit.geometry.min_mask_area,
        min_bbox_iou=audit.geometry.min_bbox_iou,
        mask_bbox_ratio_min=audit.geometry.mask_bbox_ratio_min,
        mask_bbox_ratio_max=audit.geometry.mask_bbox_ratio_max,
        min_mask_frigate_containment=audit.geometry.min_mask_frigate_containment,
    )


def run_audit(
    config: AppConfig,
    *,
    resume: bool = False,
    limit: int | None = None,
    class_filter: tuple[str, ...] = (),
    sam_only: bool = False,
    no_hard_negatives: bool = False,
    neg_samples: int | None = None,
) -> dict:
    """Execute the audit pipeline; returns ``AuditStats.as_dict()``."""
    audit = config.audit

    in_path = config.resolve(audit.input)
    out_path = config.resolve(audit.output)
    trn_path = config.resolve(audit.training)

    if not in_path.exists():
        raise AuditRunError(f"input path does not exist: {in_path}")

    adapter = build_adapter(in_path)
    sam_teacher = build_sam_teacher(audit)
    reconciler = build_reconciler(audit)
    gates = build_gates(audit)

    hn_enabled = audit.hard_negatives_enabled if not no_hard_negatives else False
    neg_enabled = audit.negative_sampling_enabled
    neg_per_crop = neg_samples if neg_samples is not None else audit.samples_per_crop

    from .pipeline import AuditPipeline, PipelineOptions

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
        limit=limit,
        classes=list(class_filter),
        sam_only=sam_only,
        resume=resume,
        max_consecutive_vlm_errors=audit.max_consecutive_vlm_errors,
    )

    stats = pipeline.run(options)
    return stats.as_dict()


__all__ = [
    "AuditRunError",
    "build_adapter",
    "build_gates",
    "build_reconciler",
    "build_sam_teacher",
    "run_audit",
]