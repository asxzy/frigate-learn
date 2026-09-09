"""Evaluation package: metrics + golden set (P0), benchmark (P6)."""

from .golden import GoldenDataset, GoldenSample, time_bucket_for_timestamp
from .metrics import (
    DetectionEvaluator,
    DetectionMetrics,
    Evaluator,
    GroundTruth,
    Prediction,
    evaluate,
    iou,
)

__all__ = [
    "GoldenDataset",
    "GoldenSample",
    "time_bucket_for_timestamp",
    "Prediction",
    "GroundTruth",
    "DetectionMetrics",
    "DetectionEvaluator",
    "Evaluator",
    "evaluate",
    "iou",
]