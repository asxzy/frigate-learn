"""Immutable golden dataset (Phase 0).

The golden set is the permanent benchmark for candidate deployment. It lives in
``data/golden/<version>/`` in standard YOLO layout:

````
golden/
├── images/
├── labels/
├── dataset.yaml
└── manifest.jsonl
````

Rules:
- Every image has a stable ``sample_id``.
- The manifest is append-only and never overwritten: re-adding a known
  ``sample_id`` raises ``ManifestDuplicateError``.
- Collection / training jobs may *read* the golden set, never write to it.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..dataset.manifest import ManifestDuplicateError, append_record, iter_manifest
from ..dataset.yolo import YoloLine

MANIFEST_KEYS = (
    "sample_id",
    "image",
    "camera",
    "timestamp",
    "source",
    "split",
    "weather",
    "time_bucket",
    "notes",
)


@dataclass
class GoldenSample:
    record: dict[str, Any]
    image_path: Path
    label_path: Path


@dataclass
class GoldenDataset:
    """Representation + create/load/validate API for one golden version."""

    root: Path
    classes: list[str] = field(default_factory=list)
    manifest: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def create(cls, root: Path, classes: list[str]) -> "GoldenDataset":
        root = Path(root)
        (root / "images").mkdir(parents=True, exist_ok=True)
        (root / "labels").mkdir(parents=True, exist_ok=True)
        write_dataset_yaml(root / "dataset.yaml", classes)
        return cls(root=root, classes=list(classes))

    @classmethod
    def load(cls, root: Path) -> "GoldenDataset":
        root = Path(root)
        ds_yaml = root / "dataset.yaml"
        classes = read_dataset_yaml(ds_yaml) if ds_yaml.exists() else []
        if not (root / "images").is_dir():
            raise ValueError(f"golden dataset missing images/ dir: {root}")
        manifest = list(iter_manifest(root / "manifest.jsonl")) if (root / "manifest.jsonl").exists() else []
        return cls(root=root, classes=classes, manifest=manifest)

    def dataset_yaml(self) -> Path:
        return self.root / "dataset.yaml"

    def manifest_path(self) -> Path:
        return self.root / "manifest.jsonl"

    def images_dir(self) -> Path:
        return self.root / "images"

    def labels_dir(self) -> Path:
        return self.root / "labels"

    def add_image(
        self,
        sample_id: str,
        image_src: Path,
        labels: list[YoloLine],
        *,
        camera: str,
        timestamp: float,
        source: str = "camera",
        split: str = "golden",
        weather: str | None = None,
        time_bucket: str | None = None,
        notes: str | None = None,
        overwrite_ok: bool = False,
    ) -> dict[str, Any]:
        """Copy an image + labels into the golden set and append a manifest row.

        Never silently overwrites: a duplicate ``sample_id`` raises unless
        ``overwrite_ok=True`` (used only by explicit golden-set maintenance).
        """
        ext = image_src.suffix.lower()
        if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
            raise ValueError(f"unsupported image extension: {ext}")

        image_dst = self.images_dir() / f"{sample_id}{ext}"
        label_dst = self.labels_dir() / f"{sample_id}.txt"
        if not overwrite_ok:
            known = {r.get("sample_id") for r in self.manifest}
            if sample_id in known:
                raise ManifestDuplicateError(f"golden sample already exists: {sample_id}")
            # protect against half-written prior attempts
            if image_dst.exists() or label_dst.exists():
                raise ManifestDuplicateError(
                    f"golden sample files already exist on disk: {sample_id}"
                )

        from ..dataset.yolo import write_yolo_label

        shutil.copyfile(image_src, image_dst)
        write_yolo_label(label_dst, labels)

        record: dict[str, Any] = {
            "sample_id": sample_id,
            "image": image_dst.name,
            "camera": camera,
            "timestamp": timestamp,
            "source": source,
            "split": split,
            "weather": weather,
            "time_bucket": time_bucket or time_bucket_for_timestamp(timestamp),
            "notes": notes,
        }
        if not overwrite_ok:
            append_record(self.manifest_path(), record)
        self.manifest.append(record)
        return record

    def samples(self) -> list[GoldenSample]:
        out = []
        for rec in self.manifest:
            image = self.images_dir() / rec["image"]
            label = self.labels_dir() / f"{Path(rec['image']).stem}.txt"
            out.append(GoldenSample(record=rec, image_path=image, label_path=label))
        return out

    def validate(self) -> list[str]:
        """Return a list of problems; empty list means the set is consistent."""
        problems: list[str] = []
        seen: set[str] = set()
        for rec in self.manifest:
            sid = rec.get("sample_id")
            if not sid:
                problems.append("record missing sample_id")
                continue
            if sid in seen:
                problems.append(f"duplicate sample_id: {sid}")
            seen.add(sid)
            image = self.images_dir() / rec.get("image", "")
            label = self.labels_dir() / f"{Path(rec.get('image', '')).stem}.txt"
            if not image.is_file():
                problems.append(f"missing image for {sid}: {image}")
            if not label.is_file():
                problems.append(f"missing label for {sid}: {label}")
        return problems


def write_dataset_yaml(path: Path, class_names: list[str]) -> None:
    payload = {
        "path": str(path.parent),
        "train": "images",
        "val": "images",
        "names": {i: name for i, name in enumerate(class_names)},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def read_dataset_yaml(path: Path) -> list[str]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    names = payload.get("names") or {}
    ordered = sorted(names.items(), key=lambda kv: int(kv[0]))
    return [str(name) for _, name in ordered]


def time_bucket_for_timestamp(ts: float) -> str:
    """Coarse day/night classification driven by local hour. Configurable later."""
    from datetime import datetime

    hour = datetime.fromtimestamp(ts).hour
    if hour < 5:
        _bucket = "night"
    elif hour < 9:
        _bucket = "dawn"
    elif hour < 17:
        _bucket = "day"
    elif hour < 21:
        _bucket = "dusk"
    else:
        _bucket = "night"
    return _bucket


__all__ = [
    "GoldenDataset",
    "GoldenSample",
    "ManifestDuplicateError",
    "write_dataset_yaml",
    "read_dataset_yaml",
    "time_bucket_for_timestamp",
    "MANIFEST_KEYS",
]