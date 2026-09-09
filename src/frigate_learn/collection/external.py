"""External dataset import (Phase 10).

Ramps up rare classes by importing annotated images that are NOT from the Frigate
cameras (COCO subsets, Open Images exports, ...) into the same sample pool as
``source='external'``. Images are copied into ``data/images/<day>/<camera>/`` and
one ``samples`` row + annotations per object are written, so every downstream
phase (dedup, verification, dataset build) treats them uniformly.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..config import AppConfig
from ..db import Database
from ..logutil import info
from ..models import Annotation, Sample, utcnow
from .dedup import dhash_file


@dataclass
class ImportSummary:
    images: int = 0
    objects: int = 0
    skipped_missing: int = 0
    skipped_unmapped: int = 0
    samples_written: int = 0


def _day_dir(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y%m%d")


def _hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _normalize_box(box_xywh: list[float], width: float, height: float):
    x, y, w, h = (float(v) for v in box_xywh)
    if width <= 0 or height <= 0 or w <= 0 or h <= 0:
        return None
    x1 = max(0.0, x / width)
    y1 = max(0.0, y / height)
    x2 = min(1.0, (x + w) / width)
    y2 = min(1.0, (y + h) / height)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def import_coco(
    config: AppConfig,
    db: Database,
    annotations_path: Path,
    images_dir: Path,
    *,
    camera: str,
    class_map: dict[str, str] | None = None,
    source: str = "external",
) -> ImportSummary:
    """Import a COCO-format annotations JSON plus its image folder.

    ``class_map`` maps COCO category names -> configured class names; objects
    whose category is absent are skipped. By default the COCO name is used as-is.
    """
    payload = json.loads(Path(annotations_path).read_text(encoding="utf-8"))
    categories = {c["id"]: str(c["name"]) for c in payload.get("categories", [])}
    images = {img["id"]: img for img in payload.get("images", [])}
    anns_by_image: dict[int, list] = {}
    for ann in payload.get("annotations", []):
        anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)

    summary = ImportSummary()
    images_dir = Path(images_dir)
    for image_id in sorted(images):
        meta = images[image_id]
        file_name = str(meta.get("file_name") or meta.get("id"))
        src = images_dir / file_name
        if not src.is_file():
            # allow values like "subdir/file.jpg" relative to images_dir
            alt = images_dir / file_name
            if not alt.is_file():
                summary.skipped_missing += 1
                continue
        width = float(meta.get("width") or 1920)
        height = float(meta.get("height") or 1080)

        mapped = []
        for ann in anns_by_image.get(image_id, []):
            cat_id = int(ann.get("category_id", -1))
            coco_name = categories.get(cat_id)
            if coco_name is None:
                summary.skipped_unmapped += 1
                continue
            label = class_map.get(coco_name, coco_name) if class_map else coco_name
            if label not in config.classes:
                summary.skipped_unmapped += 1
                continue
            box = _normalize_box(ann.get("bbox", []), width, height)
            if box is None:
                summary.skipped_unmapped += 1
                continue
            mapped.append((label, box, float(ann.get("score") or 1.0)))

        if not mapped:
            continue

        sample_id = str(uuid.uuid4())
        timestamp = float(src.stat().st_mtime)
        dest_dir = config.images_dir() / _day_dir(timestamp) / camera
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{sample_id}{src.suffix.lower() or '.jpg'}"
        try:
            shutil.copy2(src, dest)
        except OSError:
            summary.skipped_missing += 1
            continue

        phash = None
        try:
            phash = dhash_file(dest)
        except Exception:
            pass

        with db.session() as session:
            session.add(
                Sample(
                    id=sample_id,
                    camera=camera,
                    timestamp=timestamp,
                    event_id=None,
                    frame_index=0,
                    image_path=str(dest),
                    image_hash=_hash_file(dest),
                    perceptual_hash=phash,
                    source=source,
                    frigate_label=mapped[0][0],
                    frigate_score=mapped[0][2],
                    status="collected",
                    created_at=utcnow(),
                )
            )
            session.flush()
            for label, box, conf in mapped:
                session.add(
                    Annotation(
                        id=str(uuid.uuid4()),
                        sample_id=sample_id,
                        source=source,
                        label=label,
                        x1=box[0],
                        y1=box[1],
                        x2=box[2],
                        y2=box[3],
                        confidence=conf,
                        verified=1,  # human/curated external labels are trusted
                    )
                )
            session.commit()
        summary.images += 1
        summary.objects += len(mapped)
        summary.samples_written += 1

    info(
        "coco import finished",
        images=summary.images,
        objects=summary.objects,
        camera=camera,
    )
    return summary


__all__ = ["ImportSummary", "import_coco"]