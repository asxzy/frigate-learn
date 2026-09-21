"""Training export writer + provenance builder.

Positive samples are written as YOLO-style annotations:

    <training>/positive/images/<sample_id>.jpg
    <training>/positive/labels/<sample_id>.txt
    <training>/positive/masks/<sample_id>.png
    <training>/positive/provenance.jsonl
    <training>/classes.txt

The label uses the original Frigate class and the SAM-refined bbox; the mask
is the SAM mask. Hard negatives (optional) land under
``<training>/hard_negative/`` with empty label files.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from PIL import Image

from .types import (
    BoundingBox,
    Decision,
    FrigateObject,
    GeometryMetrics,
    SamResult,
    VlmResult,
)


def _write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(line + chr(10) for line in lines), encoding="utf-8")


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + chr(10))


def write_classes(training_root: Path, ontology: list[str]) -> Path:
    """Frigate vocabulary as classes.txt (one name per line, id = line index)."""
    path = training_root / "classes.txt"
    _write_lines(path, list(ontology))
    return path


def yolo_label_line(
    class_id: int,
    bbox: BoundingBox,
    image_width: int,
    image_height: int,
) -> str:
    """Normalized cxcywh YOLO label line for one box."""
    cx = bbox.cx / image_width
    cy = bbox.cy / image_height
    w = max(0.0, bbox.width) / image_width
    h = max(0.0, bbox.height) / image_height
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def write_positive_sample(
    training_root: Path,
    sample_id: str,
    crop: Image.Image,
    class_id: int,
    sam_result: SamResult,
    provenance: dict[str, Any],
) -> Path:
    """Write one accepted sample; returns the labels path."""
    positive = training_root / "positive"
    (positive / "images").mkdir(parents=True, exist_ok=True)
    (positive / "labels").mkdir(parents=True, exist_ok=True)
    (positive / "masks").mkdir(parents=True, exist_ok=True)
    crop.convert("RGB").save(positive / "images" / f"{sample_id}.jpg", quality=92)
    width, height = crop.size
    line = yolo_label_line(class_id, sam_result.bbox, width, height)
    _write_lines(positive / "labels" / f"{sample_id}.txt", [line])
    Image.fromarray(sam_result.mask.to_uint8(), mode="L").save(
        positive / "masks" / f"{sample_id}.png"
    )
    _append_jsonl(positive / "provenance.jsonl", provenance)
    return positive / "labels" / f"{sample_id}.txt"


def write_hard_negative(
    training_root: Path,
    sample_id: str,
    crop: Image.Image,
    provenance: dict[str, Any],
    *,
    kind: str = "hard_negative",
) -> Path:
    """Write a negative sample (empty label); returns the labels path."""
    negative = training_root / "hard_negative"
    (negative / "images").mkdir(parents=True, exist_ok=True)
    (negative / "labels").mkdir(parents=True, exist_ok=True)
    crop.convert("RGB").save(negative / "images" / f"{sample_id}.jpg", quality=92)
    labels_path = negative / "labels" / f"{sample_id}.txt"
    _write_lines(labels_path, [])
    record = dict(provenance)
    record["kind"] = kind
    _append_jsonl(negative / "provenance.jsonl", record)
    return labels_path


def sample_background_negative_box(
    image_width: int,
    image_height: int,
    object_box: BoundingBox,
    seed: int,
    *,
    max_iou: float = 0.2,
    min_box_side: int = 32,
    attempts: int = 16,
) -> BoundingBox | None:
    """Deterministic random background box inside the crop.

    Used only for optional in-crop background sampling (off by default). The
    sampled box is constrained to the crop and must weakly overlap the
    audited object; the seed makes the draw reproducible per sample.
    """
    rng = random.Random(seed)
    max_side = min(image_width, image_height, 256)
    for _ in range(attempts):
        side = rng.randint(min_box_side, max(min_box_side, max_side))
        x = rng.randint(0, max(0, image_width - side))
        y = rng.randint(0, max(0, image_height - side))
        box = BoundingBox(float(x), float(y), float(x + side), float(y + side))
        from .geometry import bbox_iou

        if bbox_iou(box, object_box) <= max_iou:
            return box
    return None


def build_provenance(
    sample_id: str,
    obj: FrigateObject,
    sample_hash: str,
    sam_result: SamResult | None,
    geometry: GeometryMetrics | None,
    vlm_result: VlmResult | None,
    vlm_error: str | None,
    decision: Decision,
    *,
    error_kind: str | None = None,
) -> dict[str, Any]:
    """Machine-readable provenance in the canonical 17-shape."""
    source = {
        "type": "frigate",
        "class_name": obj.class_name,
        "class_id": obj.class_id,
        "bbox": obj.bbox.to_list(),
    }
    source.update(obj.extra)
    sam_block: dict[str, Any] = {"class_name": None, "bbox": None, "confidence": None}
    if sam_result is not None:
        sam_block = {
            "class_name": sam_result.class_name,
            "bbox": sam_result.bbox.to_list(),
            "confidence": sam_result.confidence,
            "mask_area": sam_result.mask.area,
        }
    geometry_block = geometry.as_dict() if geometry is not None else None
    vlm_block: dict[str, Any] = {"error": vlm_error} if vlm_error else {}
    if error_kind is not None:
        vlm_block["error_kind"] = error_kind
    if vlm_result is not None:
        vlm_block = vlm_result.as_dict()
    return {
        "sample_id": sample_id,
        "sample_hash": sample_hash,
        "source": source,
        "sam": sam_block,
        "geometry": geometry_block,
        "vlm": vlm_block,
        "decision": decision.as_dict(),
    }


__all__ = [
    "build_provenance",
    "sample_background_negative_box",
    "write_classes",
    "write_hard_negative",
    "write_positive_sample",
    "yolo_label_line",
]
