"""Frigate dataset audit & pseudo-labeling pipeline.

``frigate_learn.audit`` turns an existing Frigate crop dataset into a
SAM 3.1 + VLM reconciled training set without any human labeling:

1. a *dataset adapter* yields one Frigate crop/object at a time;
2. SAM 3.1 (MLX, ``sam3_mlx``) produces a mask + refined geometry + an
   independent class hypothesis;
3. deterministic geometry metrics are computed from boxes/masks;
4. a single-object reconciliation image is rendered;
5. an oMLX-hosted VLM visually verifies object presence, class agreement,
   box and mask validity;
6. only SAM/VLM agreement produces a positive training sample (original
   Frigate class + SAM bbox + SAM mask). Anything else is dropped.

The Frigate label vocabulary is immutable: the pipeline never renames,
reorders, or introduces classification labels.
"""

from __future__ import annotations

from ..config import AuditSettings
from .cache import content_hash, sample_hash
from .decision import GeometryThresholds, decide_sample, is_valid_training_sample
from .export import write_classes, write_positive_sample
from .pipeline import AuditPipeline, AuditStats
from .types import (
    BoundingBox,
    Decision,
    FrigateObject,
    GeometryMetrics,
    Mask,
    SamCandidate,
    SamResult,
    VlmResult,
)
from .vlm import build_reconciler, parse_vlm_result

__all__ = [
    "AuditPipeline",
    "AuditSettings",
    "AuditStats",
    "BoundingBox",
    "Decision",
    "FrigateObject",
    "GeometryMetrics",
    "GeometryThresholds",
    "Mask",
    "SamCandidate",
    "SamResult",
    "VlmResult",
    "build_reconciler",
    "content_hash",
    "decide_sample",
    "is_valid_training_sample",
    "parse_vlm_result",
    "sample_hash",
    "write_classes",
    "write_positive_sample",
]
