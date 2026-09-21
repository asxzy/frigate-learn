"""Dataset adapters.

The audit pipeline only consumes :class:`FrigateObject` items — one crop, one
object, one label, one box — in *crop-image* absolute ``xyxy`` pixel space.
Adapters convert from whatever the raw Frigate dataset actually declares.

Two adapters ship:

* :class:`ManifestDatasetAdapter` — a documented directory layout
  (``raw/images/*`` + ``raw/annotations/manifest.json``) that explicitly
  declares every coordinate convention. This is the recommended interchange
  format when exporting crops out of Frigate.
* :class:`FrigateDatabaseAdapter` — reads the frigate-learn SQLite database
  (+ ``data/images``) directly: samples rows carry ``frigate_label`` and a
  normalized ``xyxy`` box (Frigate 0.18 event boxes are normalized
  ``[x, y, w, h]`` relative to the original frame).

Coordinate handling (never guessed):

* ``bbox_format`` — ``xyxy`` | ``xywh`` | ``cxcywh``
* ``normalized`` — box values are fractions of the declared space size
* ``coordinate_space`` — ``image`` (bbox relative to the crop) or ``frame``
  (bbox relative to the original frame; requires ``crop_origin``)
* conversion always ends with clamping to the crop bounds.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from .geometry import (
    clip_box_to_image,
    convert_cxcywh_to_xyxy,
    convert_frame_to_crop,
    convert_normalized_xyxy,
    convert_xywh_to_xyxy,
)
from .types import BoundingBox, FrigateObject


class AdapterError(ValueError):
    """The dataset declares something the adapter cannot convert."""


class DatasetAdapter(ABC):
    """Yields one Frigate object at a time, already in crop-local pixels."""

    @abstractmethod
    def iter_objects(self) -> Iterator[FrigateObject]:
        """Iterate objects; boxes are absolute ``xyxy`` in crop-image space."""
        ...

    @abstractmethod
    def ontology(self) -> list[str]:
        """Ordered Frigate label vocabulary (class id = index)."""
        ...

    def class_id_of(self, class_name: str) -> int | None:
        try:
            return self.ontology().index(class_name)
        except ValueError:
            return None

    def __len__(self) -> int:
        return sum(1 for _ in self.iter_objects())


def _parse_bbox(
    raw: Any,
    bbox_format: str,
    normalized: bool,
    coordinate_space: str,
    image: Image.Image,
    crop_origin: tuple[float, float] | None,
    frame_size: tuple[int, int] | None,
) -> tuple[BoundingBox, str]:
    """Convert a manifest-declared bbox to crop-local absolute xyxy pixels.

    Returns ``(box, box_format)``. Raises :class:`AdapterError` when the
    declared conventions cannot be resolved without guessing.
    """
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        raise AdapterError(f"bbox must be a 4-number list, got {raw!r}")
    try:
        x1, y1, x2, y2 = (float(v) for v in raw)
    except (TypeError, ValueError) as exc:
        raise AdapterError(f"bbox entries must be numbers: {raw!r}") from exc

    width = image.width
    height = image.height
    if width <= 0 or height <= 0:
        raise AdapterError(f"image has invalid dimensions {width}x{height}")

    if bbox_format == "xywh":
        box = convert_xywh_to_xyxy(x1, y1, x2, y2)
    elif bbox_format == "cxcywh":
        box = convert_cxcywh_to_xyxy(x1, y1, x2, y2)
    elif bbox_format == "xyxy":
        box = BoundingBox(x1, y1, x2, y2)
    else:
        raise AdapterError(f"unknown bbox_format {bbox_format!r} (xyxy|xywh|cxcywh)")

    if coordinate_space not in ("image", "frame"):
        raise AdapterError(f"unknown coordinate_space {coordinate_space!r} (image|frame)")

    if coordinate_space == "frame":
        if crop_origin is None:
            raise AdapterError(
                "coordinate_space=frame requires crop_origin=[ox, oy] for each object"
            )
        frame_w, frame_h = frame_size or (0, 0)
        if normalized and (frame_w <= 0 or frame_h <= 0):
            raise AdapterError("normalized frame boxes require frame_size=[w, h]")
        if normalized:
            box = convert_normalized_xyxy(box, frame_w, frame_h)
        box = convert_frame_to_crop(box, crop_origin[0], crop_origin[1])
    elif normalized:
        box = convert_normalized_xyxy(box, width, height)

    box = clip_box_to_image(box, width, height)
    return box, "xyxy_px"


def _match_image(
    image_path: Path,
    suffixes: Sequence[str] = (".jpg", ".jpeg", ".png", ".webp"),
) -> Path:
    """Resolve an image path, possibly without an extension."""
    if image_path.is_file():
        return image_path
    for suffix in suffixes:
        candidate = image_path.with_suffix(suffix)
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"image not found: {image_path}")


class ManifestDatasetAdapter(DatasetAdapter):
    """Reads ``raw/images`` + ``raw/annotations/manifest.json``.

    Manifest schema (all fields optional unless noted):
    """

    def __init__(
        self,
        root: str | Path,
        *,
        manifest_rel: str = "annotations/manifest.json",
        image_root: str | None = None,
    ) -> None:
        self.root = Path(root)
        manifest_candidates = [self.root / manifest_rel, self.root / "manifest.json"]
        manifest_path = next((p for p in manifest_candidates if p.is_file()), None)
        if manifest_path is None:
            raise AdapterError(
                f"no manifest found under {self.root} (looked for {manifest_rel} and manifest.json)"
            )
        self.manifest_path = manifest_path
        self._manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
        self._image_root = self.root / (image_root or str(self._manifest.get("image_root", "images")))
        self._bbox_format = str(self._manifest.get("bbox_format", "xyxy"))
        self._normalized = bool(self._manifest.get("normalized", False))
        self._coordinate_space = str(self._manifest.get("coordinate_space", "image"))
        self._frame_size = self._manifest.get("frame_size")
        if isinstance(self._frame_size, (list, tuple)) and len(self._frame_size) == 2:
            self._frame_size = tuple(int(v) for v in self._frame_size)
        else:
            self._frame_size = None
        self._class_map = self._manifest.get("class_map")
        if self._class_map is not None and not isinstance(self._class_map, dict):
            raise AdapterError("class_map must be a JSON object {name: id}")
        self._objects = self._manifest.get("objects", [])
        if not isinstance(self._objects, list):
            raise AdapterError("manifest objects must be a list")
        self._ontology: list[str] = []
        self._ids_seen: set[str] = set()
        self.skip_reasons: dict[str, int] = {}

    def _count_skip(self, reason: str) -> None:
        self.skip_reasons[reason] = self.skip_reasons.get(reason, 0) + 1

    def ontology(self) -> list[str]:
        if self._ontology:
            return list(self._ontology)
        if self._class_map:
            ordered = sorted(self._class_map.items(), key=lambda kv: int(kv[1]))
            self._ontology = [name for name, _ in ordered]
        else:
            seen: list[str] = []
            for obj in self._objects:
                name = str(obj.get("class_name", ""))
                if name and name not in seen:
                    seen.append(name)
            self._ontology = seen
        return list(self._ontology)

    def iter_objects(self) -> Iterator[FrigateObject]:
        ontology = self.ontology()
        for index, raw in enumerate(self._objects):
            if not isinstance(raw, dict):
                raise AdapterError(f"objects[{index}] is not an object")
            sample_id = str(raw.get("id", "")).strip()
            if not sample_id:
                raise AdapterError(f"objects[{index}] missing id")
            if sample_id in self._ids_seen:
                raise AdapterError(f"duplicate sample id {sample_id!r}")
            self._ids_seen.add(sample_id)
            class_name = str(raw.get("class_name", "")).strip()
            if not class_name:
                raise AdapterError(f"sample {sample_id}: missing class_name")
            image_rel = str(raw.get("image", ""))
            if not image_rel:
                raise AdapterError(f"sample {sample_id}: missing image path")
            image_path = _match_image(self._image_root / image_rel)
            try:
                with Image.open(image_path) as img:
                    img.load()
                    image = img
            except (OSError, ValueError):
                self._count_skip("image_open_failed")
                continue
            crop_origin = raw.get("crop_origin")
            if crop_origin is not None:
                crop_origin = tuple(float(v) for v in crop_origin)
            bbox, bbox_format = _parse_bbox(
                raw.get("bbox"),
                bbox_format=str(raw.get("bbox_format") or self._bbox_format),
                normalized=bool(raw.get("normalized", self._normalized)),
                coordinate_space=str(raw.get("coordinate_space") or self._coordinate_space),
                image=image,
                crop_origin=crop_origin,
                frame_size=self._frame_size,
            )
            class_id: int | None
            if self._class_map is not None:
                class_id = self._class_map.get(class_name)
            else:
                class_id = ontology.index(class_name) if class_name in ontology else None
            extra = dict(raw.get("extra") or {})
            extra.setdefault("manifest_index", index)
            extra.setdefault("crop_origin", list(crop_origin) if crop_origin else None)
            yield FrigateObject(
                sample_id=sample_id,
                image_path=str(image_path),
                class_name=class_name,
                class_id=class_id,
                bbox=bbox,
                bbox_format=bbox_format,
                extra=extra,
            )


class FrigateDatabaseAdapter(DatasetAdapter):
    """Reads the frigate-learn SQLite dataset (samples table + images).

    Convention (this repo):

    * images live under ``<data>/images/<YYYYMMDD>/<camera>/<id>.jpg``
    * ``frigate_x1..y2`` hold the Frigate event box as **normalized xyxy**
      relative to the original frame (Frigate 0.18 events are normalized
      ``[x, y, w, h]``; the collectors normalize to ``xyxy``).
    * with the default full-frame collection the stored image *is* the frame,
      so normalized xyxy maps straight onto image pixels.
    * the DB does **not** record crop origins; if the dataset was collected
      with ``collection.region_crop`` you must pass ``assume_crop=True``, which
      interprets the stored box as normalized relative to the stored crop.
    """

    def __init__(
        self,
        db_path: str | Path,
        images_root: str | Path | None = None,
        *,
        assume_crop: bool = False,
        statuses: Sequence[str] = (),
        quality: Sequence[str] = (),
    ) -> None:
        from sqlalchemy import bindparam, create_engine, text

        self.db_path = Path(db_path)
        if not self.db_path.is_file():
            raise AdapterError(f"database not found: {self.db_path}")
        self._engine = create_engine(f"sqlite:///{self.db_path}")
        self._bindparam = bindparam
        self._text = text
        self.images_root = Path(images_root) if images_root else None
        self.assume_crop = assume_crop
        self.statuses = list(statuses)
        self.quality = list(quality)
        self.skip_reasons: dict[str, int] = {}
        self._ontology: list[str] | None = None

    def _resolve_image(self, stored: str) -> Path:
        path = Path(stored)
        if self.images_root is not None and not path.is_absolute():
            path = self.images_root / path
        return _match_image(path)

    def ontology(self) -> list[str]:
        if self._ontology is not None:
            return list(self._ontology)
        query = "SELECT frigate_label FROM samples "
        query += "WHERE frigate_label IS NOT NULL AND length(frigate_label) > 0 "
        query += "GROUP BY frigate_label ORDER BY MIN(rowid)"
        with self._engine.connect() as conn:
            rows = conn.execute(self._text(query)).fetchall()
        self._ontology = [str(r[0]) for r in rows]
        return list(self._ontology)

    def iter_objects(self) -> Iterator[FrigateObject]:
        ontology = self.ontology()
        where = [
            "frigate_label IS NOT NULL AND length(frigate_label) > 0",
            "image_path IS NOT NULL",
            ("frigate_x1 IS NOT NULL AND frigate_y1 IS NOT NULL "
             "AND frigate_x2 IS NOT NULL AND frigate_y2 IS NOT NULL"),
        ]
        params: dict[str, Any] = {}
        if self.statuses:
            where.append("status IN :statuses")
            params["statuses"] = self.statuses
        if self.quality:
            where.append("quality IN :quality")
            params["quality"] = self.quality
        sql = (
            "SELECT id, image_path, frigate_label, frigate_x1, frigate_y1, "
            "frigate_x2, frigate_y2, camera, timestamp, event_id, "
            "frigate_score FROM samples WHERE " + " AND ".join(where)
        )
        statement = self._text(sql)
        for key in params:
            statement = statement.bindparams(self._bindparam(key, expanding=True))
        with self._engine.connect() as conn:
            rows = conn.execute(statement, params).fetchall()
            for row in rows:
                (
                    sample_id,
                    stored_image,
                    label,
                    fx1,
                    fy1,
                    fx2,
                    fy2,
                    camera,
                    timestamp,
                    event_id,
                    score,
                ) = row
                if fx2 <= fx1 or fy2 <= fy1:
                    self._count_skip("degenerate_box")
                    continue
                try:
                    image_path = self._resolve_image(str(stored_image))
                except FileNotFoundError:
                    self._count_skip("missing_image")
                    continue
                try:
                    with Image.open(image_path) as img:
                        img.load()
                        width, height = img.size
                except (OSError, ValueError):
                    self._count_skip("image_open_failed")
                    continue
                box = convert_normalized_xyxy(
                    BoundingBox(float(fx1), float(fy1), float(fx2), float(fy2)),
                    width,
                    height,
                )
                box = clip_box_to_image(box, width, height)
                if not box.is_valid():
                    self._count_skip("clipped_to_empty")
                    continue
                class_id = ontology.index(str(label)) if str(label) in ontology else None
                yield FrigateObject(
                    sample_id=str(sample_id),
                    image_path=str(image_path),
                    class_name=str(label),
                    class_id=class_id,
                    bbox=box,
                    bbox_format="xyxy_px",
                    extra={
                        "camera": camera,
                        "timestamp": timestamp,
                        "event_id": event_id,
                        "frigate_score": score,
                        "source_box": [fx1, fy1, fx2, fy2],
                        "assume_crop": self.assume_crop,
                    },
                )

    def _count_skip(self, reason: str) -> None:
        self.skip_reasons[reason] = self.skip_reasons.get(reason, 0) + 1

    def close(self) -> None:
        self._engine.dispose()


__all__ = [
    "AdapterError",
    "DatasetAdapter",
    "FrigateDatabaseAdapter",
    "ManifestDatasetAdapter",
]
