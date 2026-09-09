"""Versioned YOLO dataset builder (Phase 5).

Builds a frozen, reproducible training dataset from the SQLite pool into
``data/datasets/<version>/``:

````
datasets/<version>/
├── images/            <sample_id>.<ext>
├── labels/            <sample_id>.txt   (YOLO class cx cy w h)
├── train.txt | val.txt | test.txt        (deterministic split assignment)
├── dataset.yaml
├── build.json         (parameters that produced this build)
└── manifest.jsonl
````

Annotations choose the *best available* label per sample: a verified VLM/human
annotation first, then the original Frigate detection. Only classes in
``config.classes`` survive; everything else is dropped (and counted).
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..config import AppConfig
from ..db import Database
from ..models import Sample
from .manifest import append_record, iter_manifest
from .splits import VALID_SPLITS, assign_split
from .yolo import YoloLine, bbox_to_yolo, write_yolo_label

SPLIT_SEED = "frigate-learn-v1"


@dataclass
class BuildSummary:
    version: str
    root: Path
    total: int = 0
    images_written: int = 0
    skipped_no_box: int = 0
    skipped_class: int = 0
    skipped_cap: int = 0
    verified: int = 0
    split_counts: dict[str, int] = field(default_factory=dict)


class DatasetBuilder:
    def __init__(self, config: AppConfig, db: Database) -> None:
        self.config = config
        self.db = db

    def build(
        self,
        version: str,
        *,
        cameras: list[str] | None = None,
        labels: list[str] | None = None,
        quality: str | None = None,
        verified_only: bool = False,
        max_per_class: int | None = None,
        seed: str | None = None,
        overwrite: bool = False,
    ) -> BuildSummary:
        target = self.config.datasets_dir() / version
        if target.exists() and any(target.iterdir()) and not overwrite:
            raise FileExistsError(
                f"dataset version {version!r} already exists at {target}; "
                "pick a new version or pass overwrite=True"
            )

        classes = list(self.config.classes)
        class_id = {name: i for i, name in enumerate(classes)}
        split_seed = seed or SPLIT_SEED
        summary = BuildSummary(version=version, root=target)

        images_dir = target / "images"
        labels_dir = target / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        cap_counts: dict[str, int] = {}

        rows = self._query_samples(cameras, labels, quality)
        summary.total = len(rows)

        # deterministic order so per-class caps are stable across runs
        rows = sorted(rows, key=lambda s: (s.camera, s.timestamp, s.id))

        for sample in rows:
            anns = self._best_annotations(sample, verified_only=verified_only)
            if not anns:
                summary.skipped_no_box += 1
                continue
            usable = [
                a
                for a in anns
                if a.label in class_id
                and a.x1 is not None
                and a.x2 is not None
                and a.x2 > a.x1
                and a.y2 is not None
                and a.y1 is not None
                and a.y2 > a.y1
            ]
            if not usable:
                summary.skipped_class += 1
                continue

            primary = usable[0].label
            if max_per_class is not None and cap_counts.get(primary, 0) >= max_per_class:
                summary.skipped_cap += 1
                continue
            cap_counts[primary] = cap_counts.get(primary, 0) + 1

            source_path = Path(sample.image_path) if sample.image_path else None
            if source_path is None or not source_path.is_file():
                summary.skipped_no_box += 1
                continue

            ext = source_path.suffix.lower() or ".jpg"
            image_dst = images_dir / f"{sample.id}{ext}"
            try:
                shutil.copy2(source_path, image_dst)
            except OSError:
                summary.skipped_no_box += 1
                continue

            lines = [_to_yolo_line(a, class_id) for a in usable]
            write_yolo_label(labels_dir / f"{sample.id}.txt", lines)

            split = assign_split(sample.id, seed=split_seed)
            summary.split_counts[split] = summary.split_counts.get(split, 0) + 1
            if sample.verified:
                summary.verified += 1
            summary.images_written += 1
            append_record(
                target / "manifest.jsonl",
                {
                    "sample_id": sample.id,
                    "image": f"{sample.id}{ext}",
                    "camera": sample.camera,
                    "timestamp": sample.timestamp,
                    "split": split,
                    "source": sample.source,
                    "verified": bool(sample.verified),
                    "labels": [a.label for a in usable],
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            )

        from ..evaluation.golden import write_dataset_yaml as _write_dataset_yaml

        _write_dataset_yaml(target / "dataset.yaml", classes)
        self._write_split_files(target)
        self._write_build_json(
            target / "build.json",
            {
                "version": version,
                "classes": classes,
                "cameras": cameras,
                "labels": labels,
                "quality": quality,
                "verified_only": verified_only,
                "max_per_class": max_per_class,
                "seed": split_seed,
                "built_at": datetime.now(timezone.utc).isoformat(),
                "summary": {
                    "total": summary.total,
                    "images_written": summary.images_written,
                    "skipped_no_box": summary.skipped_no_box,
                    "skipped_class": summary.skipped_class,
                    "skipped_cap": summary.skipped_cap,
                    "split_counts": summary.split_counts,
                    "verified": summary.verified,
                },
            },
        )
        return summary

    # --- internals --------------------------------------------------------

    def _query_samples(
        self,
        cameras: list[str] | None,
        labels: list[str] | None,
        quality: str | None,
    ):
        query = self.db.session().query(Sample)
        query = query.filter(Sample.status == "collected")
        if quality is not None:
            query = query.filter(Sample.quality == quality)
        else:
            query = query.filter((Sample.quality == "useful") | (Sample.quality.is_(None)))
        if cameras:
            query = query.filter(Sample.camera.in_(cameras))
        if labels:
            query = query.filter(Sample.frigate_label.in_(labels))
        return query.all()

    def _best_annotations(self, sample: Sample, verified_only: bool):
        """Prefer verified (VLM/human) annotations; fall back to Frigate rows."""
        from ..models import Annotation

        with self.db.session() as session:
            anns = (
                session.query(Annotation)
                .filter(Annotation.sample_id == sample.id)
                .order_by(Annotation.verified.desc(), Annotation.created_at.asc())
                .all()
            )
            verified_anns = [a for a in anns if a.verified == 1]
            if verified_only:
                return verified_anns
            return verified_anns or anns[:1]

    @staticmethod
    def _write_split_files(target: Path) -> None:
        splits: dict[str, list[str]] = {s: [] for s in VALID_SPLITS}
        for rec in iter_manifest(target / "manifest.jsonl"):
            splits.setdefault(rec.get("split", "train"), []).append(f"images/{rec['image']}")
        for split, lines in splits.items():
            if not lines:
                continue
            (target / f"{split}.txt").write_text(
                "\n".join(sorted(lines)) + "\n", encoding="utf-8"
            )

    @staticmethod
    def _write_build_json(path: Path, payload: dict) -> None:
        path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _to_yolo_line(ann, class_id: dict[str, int]) -> YoloLine:
    cx, cy, w, h = bbox_to_yolo(ann.x1, ann.y1, ann.x2, ann.y2)
    return YoloLine(class_id=class_id[ann.label], cx=cx, cy=cy, w=w, h=h)


__all__ = ["DatasetBuilder", "BuildSummary"]