"""Deterministic geometry utilities for the audit pipeline.

Everything in this module is pure math on boxes/masks — no model inference,
no randomness, no floating-point tricks that change between runs. The VLM is
never asked to compute coordinates, IoU, percentages, or pixel measurements.

Convention: boxes are absolute pixel ``xyxy`` in crop-image space. Degenerate
boxes (zero/negative area) are handled safely (0-area intersections and 0.0
ratios) so a mis-annotated source box cannot crash the pipeline.
"""

from __future__ import annotations

import numpy as np

from .types import BoundingBox, GeometryMetrics, Mask


def intersection_area(a: BoundingBox, b: BoundingBox) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def union_area(a: BoundingBox, b: BoundingBox) -> float:
    return a.area + b.area - intersection_area(a, b)


def bbox_iou(a: BoundingBox, b: BoundingBox) -> float:
    """IoU of two boxes; 0.0 when either box is degenerate (no overlap possible)."""
    inter = intersection_area(a, b)
    union = union_area(a, b)
    return inter / union if union > 0.0 else 0.0


def bbox_containment(inner: BoundingBox, outer: BoundingBox) -> float:
    """Fraction of ``inner`` area covered by ``outer`` (0..1)."""
    if inner.area <= 0.0:
        return 0.0
    return intersection_area(inner, outer) / inner.area


def box_edge_deltas(reference: BoundingBox, other: BoundingBox) -> dict[str, float]:
    """Per-edge movement ``other - reference`` in pixels.

    Positive ``left_delta``/``top_delta`` mean the other box starts further
    right/down; positive ``right_delta``/``bottom_delta`` mean it extends
    further right/down. Exact and deterministic.
    """
    return {
        "left_delta": other.x1 - reference.x1,
        "top_delta": other.y1 - reference.y1,
        "right_delta": other.x2 - reference.x2,
        "bottom_delta": other.y2 - reference.y2,
    }


def mask_area(mask: Mask) -> int:
    return mask.area


def mask_to_tight_bbox(mask: Mask) -> BoundingBox | None:
    return mask.tight_bbox()


def mask_bbox_ratio(mask: Mask, box: BoundingBox) -> float:
    """``mask_area / box.area``; 0.0 for an empty mask or degenerate box."""
    if box.area <= 0.0:
        return 0.0
    return mask.area / box.area


def mask_containment(mask: Mask, box: BoundingBox) -> float:
    """Fraction of mask pixels that lie inside ``box`` (0..1).

    An empty mask yields 0.0 (no containment evidence). This is the mask
    containment sanity check: values near 1.0 mean the mask is confined to the
    box; values near 0.0 mean the mask drifted far outside it.
    """
    if mask.area == 0:
        return 0.0
    h, w = mask.data.shape
    xs = np.arange(w, dtype=np.int64)
    ys = np.arange(h, dtype=np.int64)
    xv, yv = np.meshgrid(xs, ys)
    inside = (
        (xv >= box.x1)
        & (xv < box.x2)
        & (yv >= box.y1)
        & (yv < box.y2)
    )
    return float(np.count_nonzero(mask.data & inside) / mask.area)


def compute_geometry(
    frigate_bbox: BoundingBox,
    sam_bbox: BoundingBox,
    mask: Mask,
) -> GeometryMetrics:
    """All deterministic metrics for one sample, in one call."""
    return GeometryMetrics(
        bbox_iou=bbox_iou(frigate_bbox, sam_bbox),
        edge_deltas=box_edge_deltas(frigate_bbox, sam_bbox),
        mask_area=mask.area,
        mask_bbox_ratio=mask_bbox_ratio(mask, sam_bbox),
        mask_frigate_containment=mask_containment(mask, frigate_bbox),
        mask_sam_containment=mask_containment(mask, sam_bbox),
        frigate_bbox_area=frigate_bbox.area,
        sam_bbox_area=sam_bbox.area,
    )


def clip_box_to_image(box: BoundingBox, width: int, height: int) -> BoundingBox:
    return box.clipped(width, height)


def convert_normalized_xyxy(
    box: BoundingBox,
    width: int,
    height: int,
) -> BoundingBox:
    """Reinterpret a normalized ``xyxy`` box (0..1) as absolute pixels."""
    return BoundingBox(
        box.x1 * width,
        box.y1 * height,
        box.x2 * width,
        box.y2 * height,
    )


def convert_xywh_to_xyxy(x: float, y: float, w: float, h: float) -> BoundingBox:
    return BoundingBox(x, y, x + w, y + h)


def convert_cxcywh_to_xyxy(cx: float, cy: float, w: float, h: float) -> BoundingBox:
    return BoundingBox(cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


def convert_frame_to_crop(
    frame_box: BoundingBox,
    crop_origin_x: float,
    crop_origin_y: float,
) -> BoundingBox:
    """Shift a frame-absolute box into crop coordinates (origin = crop top-left).

    Callers must clamp the result to the crop bounds afterwards — the box may
    legitimately extend beyond the crop edge.
    """
    return frame_box.shifted(-crop_origin_x, -crop_origin_y)


__all__ = [
    "bbox_containment",
    "bbox_iou",
    "box_edge_deltas",
    "clip_box_to_image",
    "compute_geometry",
    "convert_cxcywh_to_xyxy",
    "convert_frame_to_crop",
    "convert_normalized_xyxy",
    "convert_xywh_to_xyxy",
    "intersection_area",
    "mask_area",
    "mask_bbox_ratio",
    "mask_containment",
    "mask_to_tight_bbox",
    "union_area",
]
