"""Versioned YOLO dataset builder (Phase 5).

Builds a frozen, reproducible training dataset from the SQLite pool into
``data/datasets/<version>/``:

````
datasets/<version>/
├── images/            <split>/<sample_id>.<ext>   (split in train|val|test)
├── labels/            <split>/<sample_id>.txt   (YOLO class cx cy w h)
├── train.txt | val.txt | test.txt        (deterministic split assignment)
├── dataset.yaml
├── build.json         (parameters that produced this build)
└── manifest.jsonl
````

Annotations choose the *best available* label per sample: a verified VLM/human
annotation first; the original Frigate detection is used only when
`verified_only=False` (the `--include-unverified` opt-in). Only classes in
``config.classes`` survive; everything else is dropped (and counted).
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..classes import coco_index
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
        verified_only: bool = True,
        max_per_class: int | None = None,
        seed: str | None = None,
        since: str | None = None,
        overwrite: bool = False,
    ) -> BuildSummary:
        target = self.config.datasets_dir() / version
        if target.exists() and any(target.iterdir()) and not overwrite:
            raise FileExistsError(
                f"dataset version {version!r} already exists at {target}; "
                "pick a new version or pass overwrite=True"
            )
        if target.exists() and overwrite:
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()

        model_classes = self.config.model_class_names()
        label_space = self.config.training.label_space
        if label_space == "coco80":
            class_id = {
                name: coco_index(name)
                for name in model_classes
                if name in self.config.classes
            }
        else:
            class_id = {name: i for i, name in enumerate(model_classes)}
        split_seed = seed or SPLIT_SEED
        summary = BuildSummary(version=version, root=target)

        images_dir = target / "images"
        labels_dir = target / "labels"
        for split in VALID_SPLITS:
            (images_dir / split).mkdir(parents=True, exist_ok=True)
            (labels_dir / split).mkdir(parents=True, exist_ok=True)

        cap_counts: dict[str, int] = {}

        rows = self._query_samples(cameras, labels, quality, since)
        summary.total = len(rows)

        # deterministic order so per-class caps are stable across runs
        rows = sorted(rows, key=lambda s: (s.camera, s.timestamp, s.id))

        for sample in rows:
            anns = self._best_annotations(sample, verified_only=verified_only)
            if not anns:
                summary.skipped_no_box += 1
                continue
            boxed = [
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
            if not boxed:
                summary.skipped_class += 1
                continue

            usable = boxed
            if max_per_class is not None:
                kept = []
                for a in boxed:
                    if cap_counts.get(a.label, 0) >= max_per_class:
                        continue
                    cap_counts[a.label] = cap_counts.get(a.label, 0) + 1
                    kept.append(a)
                if not kept:
                    summary.skipped_cap += 1
                    continue
                usable = kept

            source_path = Path(sample.image_path) if sample.image_path else None
            if source_path is None or not source_path.is_file():
                summary.skipped_no_box += 1
                continue

            ext = source_path.suffix.lower() or ".jpg"
            split = assign_split(sample.id, seed=split_seed)
            image_dst = images_dir / split / f"{sample.id}{ext}"
            try:
                shutil.copy2(source_path, image_dst)
            except OSError:
                summary.skipped_no_box += 1
                continue

            lines = [_to_yolo_line(a, class_id) for a in usable]
            write_yolo_label(labels_dir / split / f"{sample.id}.txt", lines)

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

        _write_dataset_yaml(
            target / "dataset.yaml",
            model_classes,
            train="images/train",
            val="images/val",
            test="images/test",
        )
        self._write_split_files(target)
        payload = {
            "version": version,
            "classes": list(self.config.classes),
            "label_space": label_space,
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
        }
        trainable = self.config.trainable_class_mask()
        if trainable is not None:
            payload["trainable"] = trainable
        self._write_build_json(target / "build.json", payload)
        return summary

    # --- internals --------------------------------------------------------

    def _query_samples(
        self,
        cameras: list[str] | None,
        labels: list[str] | None,
        quality: str | None,
        since: str | None = None,
    ):
        from ..models import Annotation

        session = self.db.session()
        query = session.query(Sample)
        query = query.filter(Sample.status == "collected")
        if since is not None:
            query = query.filter(Sample.created_at >= since)
        if quality is not None:
            query = query.filter(Sample.quality == quality)
        else:
            query = query.filter((Sample.quality == "useful") | (Sample.quality.is_(None)))
        if cameras:
            query = query.filter(Sample.camera.in_(cameras))
        if labels:
            with_label = (
                session.query(Annotation.sample_id)
                .filter(Annotation.label.in_(labels))
            )
            query = query.filter(Sample.id.in_(with_label))
        return query.all()

    def _best_annotations(self, sample: Sample, verified_only: bool):
        """Prefer verified (VLM/human) annotations; fall back to Frigate rows.

        Verified annotations win outright; the unverified fallback is deduped
        against them so one physical object is never emitted twice.
        """
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
            return _dedupe_annotations(verified_anns + [a for a in anns if a.verified != 1])

    @staticmethod
    def _write_split_files(target: Path) -> None:
        splits: dict[str, list[str]] = {s: [] for s in VALID_SPLITS}
        for rec in iter_manifest(target / "manifest.jsonl"):
            split = rec.get("split", "train")
            splits.setdefault(split, []).append(f"images/{split}/{rec['image']}")
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


_SOURCE_RANK = {"vlm": 0, "human": 1, "external": 2, "frigate": 3}


def _dedupe_annotations(anns):
    """Keep one box per physical object: suppress lower-priority annotations.

    Priority is verified first, then source (vlm/human > external > frigate),
    then created_at/id for determinism. A lower-priority annotation is dropped
    when an already-selected annotation has the same label and IoU > 0.5.
    Different-label boxes are never suppressed, even when they overlap (a
    person on a motorcycle is still two objects). Boxes are normalized, so IoU
    is computed directly from x1/y1/x2/y2.
    """
    ordered = sorted(
        anns,
        key=lambda a: (
            -int(a.verified),
            _SOURCE_RANK.get(a.source, 10),
            a.created_at or "",
            a.id,
        ),
    )
    selected: list = []
    for ann in ordered:
        if any(_same_label_overlap(a, ann) for a in selected):
            continue
        selected.append(ann)
    return selected


def _same_label_overlap(a, b) -> bool:
    if a.label != b.label:
        return False
    coords = (a.x1, a.y1, a.x2, a.y2, b.x1, b.y1, b.x2, b.y2)
    if any(c is None for c in coords):
        return False
    ax1, ay1, ax2, ay2, bx1, by1, bx2, by2 = coords
    inter_w = min(ax2, bx2) - max(ax1, bx1)
    inter_h = min(ay2, by2) - max(ay1, by1)
    if inter_w <= 0 or inter_h <= 0:
        return False
    inter = inter_w * inter_h
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    if union <= 0:
        return False
    return inter / union > 0.5


__all__ = ["DatasetBuilder", "BuildSummary"]