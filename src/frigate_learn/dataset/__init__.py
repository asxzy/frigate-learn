"""Dataset package: YOLO format helpers (P0) + versioned builder (P5)."""

from .manifest import ManifestDuplicateError, append_record, iter_manifest, read_manifest
from .splits import assign_split, make_splits
from .yolo import (
    ImageSize,
    NormalizedBox,
    YoloLine,
    bbox_to_yolo,
    normalize_bbox,
    parse_yolo_line,
    read_yolo_label,
    validate_normalized,
    write_yolo_label,
    yolo_to_bbox,
)

__all__ = [
    "ImageSize",
    "NormalizedBox",
    "YoloLine",
    "normalize_bbox",
    "validate_normalized",
    "bbox_to_yolo",
    "yolo_to_bbox",
    "parse_yolo_line",
    "read_yolo_label",
    "write_yolo_label",
    "append_record",
    "iter_manifest",
    "read_manifest",
    "ManifestDuplicateError",
    "assign_split",
    "make_splits",
]