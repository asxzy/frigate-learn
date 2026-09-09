"""YOLO detection-format utilities (Phase 0).

Pure helpers over the classic ``class cx cy w h`` text label format with
normalized coordinates. Everything here is detector-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_EPS = 1e-6


@dataclass(frozen=True)
class ImageSize:
    width: int
    height: int


@dataclass(frozen=True)
class YoloLine:
    """One line of a YOLO label file (normalized center-wh coordinates)."""

    class_id: int
    cx: float
    cy: float
    w: float
    h: float

    def to_text(self) -> str:
        return f"{self.class_id} {self.cx:.6f} {self.cy:.6f} {self.w:.6f} {self.h:.6f}"

    def to_xyxy(self) -> tuple[float, float, float, float]:
        return (self.cx - self.w / 2, self.cy - self.h / 2, self.cx + self.w / 2, self.cy + self.h / 2)


@dataclass(frozen=True)
class NormalizedBox:
    """Axis-aligned box in normalized coordinates x1 <= x2, y1 <= y2."""

    x1: float
    y1: float
    x2: float
    y2: float

    def to_yolo_center_wh(self) -> tuple[float, float, float, float]:
        return bbox_to_yolo(self.x1, self.y1, self.x2, self.y2)


def validate_normalized(x1: float, y1: float, x2: float, y2: float) -> bool:
    """Reject boxes outside [0, 1] (with tiny tolerance) or with inverted axes."""
    low = -_EPS
    high = 1.0 + _EPS
    if not (low <= x1 <= high and low <= y1 <= high and low <= x2 <= high and low <= y2 <= high):
        return False
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return False
    return True


def normalize_bbox(x1: float, y1: float, x2: float, y2: float, size: ImageSize | None = None) -> NormalizedBox:
    """Convert pixel coordinates to normalized [0,1] box coordinates.

    Raises ``ValueError`` for empty/invalid boxes. When ``size`` is provided,
    out-of-range coordinates raise instead of silently clamping.
    """
    factors: tuple[float, float] = (1.0, 1.0)
    if size is not None:
        if size.width <= 0 or size.height <= 0:
            raise ValueError(f"invalid image size {size}")
        factors = (float(size.width), float(size.height))

    nx1, ny1 = x1 / factors[0], y1 / factors[1]
    nx2, ny2 = x2 / factors[0], y2 / factors[1]

    if size is not None:
        low, high = -_EPS, 1.0 + _EPS
        if not (low <= nx1 <= high and low <= ny1 <= high):
            raise ValueError(f"normalized coords out of range: {(nx1, ny1, nx2, ny2)}")
    if (nx2 - nx1) <= 0 or (ny2 - ny1) <= 0:
        raise ValueError(f"degenerate box: {(nx1, ny1, nx2, ny2)}")
    return NormalizedBox(nx1, ny1, nx2, ny2)


def bbox_to_yolo(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float, float, float]:
    """Normalized xyxy -> center-wh (all in [0,1])."""
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return cx, cy, (x2 - x1), (y2 - y1)


def yolo_to_bbox(cx: float, cy: float, w: float, h: float) -> tuple[float, float, float, float]:
    """Center-wh -> normalized xyxy."""
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def parse_yolo_line(line: str, num_classes: int | None = None) -> YoloLine:
    parts = line.strip().split()
    if len(parts) != 5:
        raise ValueError(f"expected 5 fields, got {len(parts)}: {line!r}")
    class_id = int(parts[0])
    cx, cy, w, h = (float(p) for p in parts[1:])
    if w <= 0 or h <= 0:
        raise ValueError(f"non-positive width/height in {line!r}")
    if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
        raise ValueError(f"center outside unit square in {line!r}")
    if num_classes is not None and not (0 <= class_id < num_classes):
        raise ValueError(f"class id {class_id} out of range [0,{num_classes})")
    return YoloLine(class_id=class_id, cx=cx, cy=cy, w=w, h=h)


def read_yolo_label(path: Path, num_classes: int | None = None) -> list[YoloLine]:
    if not path.exists():
        return []
    lines: list[YoloLine] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        lines.append(parse_yolo_line(raw, num_classes=num_classes))
    return lines


def write_yolo_label(path: Path, lines: list[YoloLine]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(line.to_text() for line in lines)
    path.write_text(body + ("\n" if body else ""), encoding="utf-8")


__all__ = [
    "ImageSize",
    "NormalizedBox",
    "YoloLine",
    "validate_normalized",
    "normalize_bbox",
    "bbox_to_yolo",
    "yolo_to_bbox",
    "parse_yolo_line",
    "read_yolo_label",
    "write_yolo_label",
]