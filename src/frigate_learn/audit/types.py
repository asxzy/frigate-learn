"""Core value types shared across the audit pipeline.

All boxes passed through the pipeline are in *crop-image* pixel space
(absolute ``xyxy``). The dataset adapter is responsible for converting
whatever the raw Frigate dataset declares (normalized/absolute, ``xywh``/
``xyxy``, frame-relative vs crop-relative) into that one convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _num(value: Any, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number, got {value!r}") from exc


@dataclass
class BoundingBox:
    """Axis-aligned box in absolute pixel coordinates (``x1,y1,x2,y2``).

    Coordinates are in the coordinate space of the crop image being audited.
    The box is stored as given (no silent reordering); degenerate boxes have
    zero area and all geometry helpers treat them safely.
    """

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        self.x1 = _num(self.x1, "x1")
        self.y1 = _num(self.y1, "y1")
        self.x2 = _num(self.x2, "x2")
        self.y2 = _num(self.y2, "y2")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    def is_valid(self) -> bool:
        return self.x2 > self.x1 and self.y2 > self.y1

    def to_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    def to_xyxy_norm(self, width: int, height: int) -> tuple[float, float, float, float]:
        if width <= 0 or height <= 0:
            raise ValueError("image dimensions must be positive")
        return (
            self.x1 / width,
            self.y1 / height,
            self.x2 / width,
            self.y2 / height,
        )

    def to_cxcywh_norm(self, width: int, height: int) -> tuple[float, float, float, float]:
        if width <= 0 or height <= 0:
            raise ValueError("image dimensions must be positive")
        return (self.cx / width, self.cy / height, self.width / width, self.height / height)

    @classmethod
    def from_xyxy_norm(
        cls,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        width: int,
        height: int,
    ) -> BoundingBox:
        if width <= 0 or height <= 0:
            raise ValueError("image dimensions must be positive")
        return cls(x1 * width, y1 * height, x2 * width, y2 * height)

    @classmethod
    def from_cxcywh_norm(
        cls,
        cx: float,
        cy: float,
        w: float,
        h: float,
        width: int,
        height: int,
    ) -> BoundingBox:
        if width <= 0 or height <= 0:
            raise ValueError("image dimensions must be positive")
        cx_px = cx * width
        cy_px = cy * height
        w_px = w * width
        h_px = h * height
        return cls(cx_px - w_px / 2.0, cy_px - h_px / 2.0, cx_px + w_px / 2.0, cy_px + h_px / 2.0)

    @classmethod
    def from_xywh_px(cls, x: float, y: float, w: float, h: float) -> BoundingBox:
        return cls(x, y, x + w, y + h)

    def to_xywh_px(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.width, self.height)

    def clipped(self, width: float, height: float) -> BoundingBox:
        return BoundingBox(
            min(max(self.x1, 0.0), float(width)),
            min(max(self.y1, 0.0), float(height)),
            min(max(self.x2, 0.0), float(width)),
            min(max(self.y2, 0.0), float(height)),
        )

    def shifted(self, dx: float, dy: float) -> BoundingBox:
        return BoundingBox(self.x1 + dx, self.y1 + dy, self.x2 + dx, self.y2 + dy)

    def as_dict(self) -> dict[str, float]:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}


@dataclass
class Mask:
    """Binary segmentation mask in crop-image space (``bool`` HxW ndarray)."""

    data: np.ndarray

    def __post_init__(self) -> None:
        arr = np.asarray(self.data)
        if arr.ndim != 2:
            raise ValueError(f"mask must be 2D, got shape {arr.shape}")
        self.data = arr.astype(bool, copy=False)

    @property
    def height(self) -> int:
        return int(self.data.shape[0])

    @property
    def width(self) -> int:
        return int(self.data.shape[1])

    @property
    def area(self) -> int:
        return int(np.count_nonzero(self.data))

    def tight_bbox(self) -> BoundingBox | None:
        ys, xs = np.nonzero(self.data)
        if ys.size == 0:
            return None
        return BoundingBox(float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))

    def to_uint8(self) -> np.ndarray:
        return (self.data.astype(np.uint8)) * 255


@dataclass(frozen=True)
class SamCandidate:
    """One SAM 3.1 detection (text prompt + geometric prompt) before selection."""

    class_name: str
    score: float
    bbox: BoundingBox
    mask_area: int
    index: int
    mask: np.ndarray

    def as_dict(self) -> dict[str, Any]:
        return {
            "class_name": self.class_name,
            "score": self.score,
            "bbox": self.bbox.to_list(),
            "mask_area": self.mask_area,
            "index": self.index,
        }


@dataclass
class SamResult:
    """SAM 3.1 output for one Frigate object.

    ``class_name`` is SAM own hypothesis (an independent observation). It may
    differ from the immutable Frigate class; the decision engine handles that.
    ``bbox`` is the SAM-refined box in crop space; ``mask`` is the SAM mask.
    """

    class_name: str
    bbox: BoundingBox
    mask: Mask
    confidence: float | None
    raw_metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        tight = self.mask.tight_bbox()
        return {
            "class_name": self.class_name,
            "bbox": self.bbox.to_list(),
            "confidence": self.confidence,
            "mask_area": self.mask.area,
            "mask_shape": [self.mask.height, self.mask.width],
            "mask_tight_bbox": tight.to_list() if tight is not None else None,
            "raw_metadata": self.raw_metadata,
        }

    def is_valid(self) -> bool:
        return bool(self.class_name) and self.bbox.is_valid() and self.mask.area > 0


@dataclass(frozen=True)
class VlmResult:
    """Strictly-parsed independent VLM judgement for one object.

    The VLM is never told the expected class. It reports only its own
    reading of the crop plus the detection box: whether the box covers the
    object and what label the object is (empty string = no object found).
    Class agreement with SAM/Frigate is computed later by the decision
    engine, never by the model.
    """

    bbox_covers_object: bool
    class_label: str
    raw: dict[str, Any]

    @property
    def object_present(self) -> bool:
        return bool(self.class_label.strip())

    def as_dict(self) -> dict[str, Any]:
        return {
            "bbox_covers_object": self.bbox_covers_object,
            "class_label": self.class_label,
        }

    @property
    def all_conditions_met(self) -> bool:
        return self.bbox_covers_object and self.object_present

    def failing_conditions(self) -> list[str]:
        failed = []
        if not self.object_present:
            failed.append("object_present")
        if not self.bbox_covers_object:
            failed.append("bbox_covers")
        return failed


@dataclass(frozen=True)
class GeometryMetrics:
    """Deterministic geometry statistics for one sample (crop space)."""

    bbox_iou: float
    edge_deltas: dict[str, float]
    mask_area: int
    mask_bbox_ratio: float
    mask_frigate_containment: float
    mask_sam_containment: float
    frigate_bbox_area: float
    sam_bbox_area: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "bbox_iou": self.bbox_iou,
            "edge_deltas": dict(self.edge_deltas),
            "mask_area": self.mask_area,
            "mask_bbox_ratio": self.mask_bbox_ratio,
            "mask_frigate_containment": self.mask_frigate_containment,
            "mask_sam_containment": self.mask_sam_containment,
            "frigate_bbox_area": self.frigate_bbox_area,
            "sam_bbox_area": self.sam_bbox_area,
        }


@dataclass
class FrigateObject:
    """One Frigate crop + one Frigate object for the reconciliation pipeline.

    ``bbox`` is already converted to crop-image space (absolute ``xyxy``) by the
    dataset adapter; ``bbox_format`` records the convention actually applied.
    ``extra`` carries opaque source provenance (camera, event id, original
    bbox, crop origin, ...) which is echoed into the provenance metadata.
    """

    sample_id: str
    image_path: str
    class_name: str
    class_id: int | None
    bbox: BoundingBox
    bbox_format: str = "xyxy_px"
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "image_path": self.image_path,
            "class_name": self.class_name,
            "class_id": self.class_id,
            "bbox": self.bbox.to_list(),
            "bbox_format": self.bbox_format,
            "extra": dict(self.extra),
        }


@dataclass
class Decision:
    """Final automated decision for one sample.

    ``status`` is ``KEEP``, ``DROP`` or ``PENDING`` (VLM transport failure;
    retried by ``--resume``). ``reason`` is a stable machine code.
    """

    status: str
    reason: str | None
    frigate_label: str
    sam_class: str | None = None
    training_label: str | None = None
    bbox_source: str | None = None
    mask_source: str | None = None
    failing_conditions: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "frigate_label": self.frigate_label,
            "sam_class": self.sam_class,
            "training_label": self.training_label,
            "bbox_source": self.bbox_source,
            "mask_source": self.mask_source,
            "failing_conditions": list(self.failing_conditions),
        }


def object_to_dict(obj: FrigateObject) -> dict[str, Any]:
    return obj.as_dict()


__all__ = [
    "BoundingBox",
    "Decision",
    "FrigateObject",
    "GeometryMetrics",
    "Mask",
    "SamCandidate",
    "SamResult",
    "VlmResult",
    "object_to_dict",
]
