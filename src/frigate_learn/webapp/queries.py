"""Read-side queries for the webapp dashboard.

Pure DB + artifact reads. No HTTP; every function returns plain dicts shaped
for the SPA. Artifacts are read in place, never cached.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
from datetime import datetime
from pathlib import Path

from sqlalchemy import func

from ..dataset.yolo import read_yolo_label
from ..evaluation.benchmark import pareto_frontier, results_from_json
from ..evaluation.golden import GoldenDataset
from ..inspect.triage import QUALITY_VALUES
from ..models import Annotation, Deployment, Job, Sample
from ..run import PIPELINE, next_build_version

BENCHMARK_FILE = "benchmark-results.json"
TRAINING_DIR = "training"
MAP50_KEY = "metrics/mAP50(B)"
MAX_LIMIT = 200


def overview(config, db) -> dict:
    counts = quality_counts(db)
    with db.session() as session:
        cameras = [
            row[0]
            for row in session.query(Sample.camera)
            .distinct()
            .order_by(Sample.camera)
            .all()
        ]
        jobs = (
            session.query(Job)
            .order_by(Job.started_at.desc())
            .limit(20)
            .all()
        )
    return {
        "steps": list(PIPELINE),
        "auto_enable": list(config.automation.enable),
        "samples": {
            "total": counts["total"],
            "verified": counts["verified"],
            "unverified": counts["unverified"],
            "reviewed": counts["reviewed"],
            "unreviewed": counts["unreviewed"],
            **{f"quality_{q}": counts["by_quality"][q] for q in sorted(QUALITY_VALUES)},
        },
        "cameras": cameras,
        "vlm_enabled": bool(config.vlm.enabled),
        "disk_free_bytes": shutil.disk_usage(config.base_dir)[2],
        "next_build_version": next_build_version(config),
        "pipeline": [_job_item(j) for j in jobs],
    }


def benchmark(config) -> dict:
    path = config.resolve(config.data.root, BENCHMARK_FILE)
    results = []
    updated_at = None
    if path.is_file():
        results = results_from_json(path)
        updated_at = datetime.fromtimestamp(path.stat().st_mtime).isoformat()
    return {
        "golden": config.evaluation.golden_dataset,
        "baseline": config.deployment.baseline,
        "max_latency_ms": config.deployment.max_latency_ms,
        "results": [r.to_dict() for r in results],
        "pareto": [p.name for p in pareto_frontier(results)],
        "results_updated_at": updated_at,
    }


def deployments(db) -> dict:
    with db.session() as session:
        rows = (
            session.query(Deployment)
            .order_by(Deployment.created_at.desc())
            .all()
        )
    return {
        "deployments": [
            {
                "id": d.id,
                "model_name": d.model_name,
                "version": d.version,
                "artifact_path": d.artifact_path,
                "artifact_hash": d.artifact_hash,
                "metrics": _parse_json(d.metrics_json),
                "reasons": _parse_json(d.reasons_json),
                "verdict": d.verdict,
                "deployed": d.deployed,
                "deployed_at": d.deployed_at,
                "created_at": d.created_at,
            }
            for d in rows
        ]
    }


def quality_counts(db) -> dict:
    with db.session() as session:
        total = session.query(func.count(Sample.id)).scalar() or 0
        verified = (
            session.query(func.count(Sample.id))
            .filter(Sample.verified == 1)
            .scalar()
            or 0
        )
        by_quality = {q: 0 for q in sorted(QUALITY_VALUES)}
        for quality, count in (
            session.query(Sample.quality, func.count(Sample.id))
            .filter(Sample.quality.in_(QUALITY_VALUES))
            .group_by(Sample.quality)
            .all()
        ):
            by_quality[quality] = count
        reviewed = (
            session.query(func.count(Sample.id))
            .filter(Sample.frigate_reviewed == 1)
            .scalar()
            or 0
        )
        unreviewed = (
            session.query(func.count(Sample.id))
            .filter(Sample.frigate_reviewed == 0)
            .scalar()
            or 0
        )
        per_class = {
            label: count
            for label, count in (
                session.query(Annotation.label, func.count(Annotation.id))
                .filter(Annotation.verified == 1)
                .group_by(Annotation.label)
                .order_by(Annotation.label)
                .all()
            )
        }
        one_class_boxes = (
            session.query(func.count(func.distinct(Annotation.sample_id)))
            .filter(Annotation.verified == 1)
            .scalar()
            or 0
        )
        problematic = (
            session.query(func.count(Sample.id))
            .filter(Sample.quality.in_(("bad", "duplicate")))
            .scalar()
            or 0
        )
    return {
        "total": total,
        "verified": verified,
        "unverified": total - verified,
        "reviewed": reviewed,
        "unreviewed": unreviewed,
        "by_quality": by_quality,
        "per_class": per_class,
        "one_class_boxes": one_class_boxes,
        "problematic": problematic,
    }


def datasets(config) -> dict:
    versions = []
    datasets_dir = config.datasets_dir()
    if datasets_dir.is_dir():
        for entry in sorted(
            (
                d
                for d in datasets_dir.iterdir()
                if d.is_dir() and d.name.startswith("v")
            ),
            key=lambda d: d.name,
            reverse=True,
        ):
            versions.append(_dataset_version(entry))
    return {"versions": versions, "golden": _golden_segment(config)}


def training_index(config) -> dict:
    runs = []
    root = config.resolve(config.data.root, TRAINING_DIR)
    if root.is_dir():
        for entry in sorted(
            (d for d in root.iterdir() if d.is_dir()),
            key=lambda d: d.name,
            reverse=True,
        ):
            results_csv = entry / "results.csv"
            if not results_csv.is_file():
                continue
            _, rows = _read_csv(results_csv)
            map50s = [r[MAP50_KEY] for r in rows if r.get(MAP50_KEY) is not None]
            model, tag = _split_run(entry.name)
            runs.append(
                {
                    "run": entry.name,
                    "model": model,
                    "tag": tag,
                    "dataset": tag,
                    "results_mtime": datetime.fromtimestamp(
                        results_csv.stat().st_mtime
                    ).isoformat(),
                    "latest_map50": map50s[-1] if map50s else None,
                    "best_map50": max(map50s) if map50s else None,
                    "has_weights": (entry / "weights" / "best.pt").is_file(),
                }
            )
    return {"runs": runs}


def training_run(config, run) -> dict | None:
    results_csv = config.resolve(config.data.root, TRAINING_DIR, run, "results.csv")
    if not results_csv.is_file():
        return None
    columns, rows = _read_csv(results_csv)
    map50s = [r[MAP50_KEY] for r in rows if r.get(MAP50_KEY) is not None]
    _, tag = _split_run(run)
    run_dir = results_csv.parent
    return {
        "run": run,
        "columns": columns,
        "rows": rows,
        "best_map50": max(map50s) if map50s else None,
        "latest_map50": map50s[-1] if map50s else None,
        "epochs": len(rows),
        "weights_exists": (run_dir / "weights" / "best.pt").is_file(),
        "before_after": _read_before_after(run_dir),
        "dataset": tag,
    }


def samples(
    db,
    *,
    status=None,
    quality=None,
    camera=None,
    verified=None,
    reviewed=None,
    limit=50,
    offset=0,
) -> dict:
    limit = max(1, min(MAX_LIMIT, int(limit)))
    offset = max(0, int(offset))
    with db.session() as session:
        query = session.query(Sample)
        if status is not None:
            query = query.filter(Sample.status == status)
        if quality is not None:
            query = query.filter(Sample.quality == quality)
        if camera is not None:
            query = query.filter(Sample.camera == camera)
        if verified is not None:
            query = query.filter(Sample.verified == int(verified))
        if reviewed is not None:
            query = query.filter(Sample.frigate_reviewed == int(reviewed))
        total = query.count()
        rows = (
            query.order_by(Sample.timestamp.desc(), Sample.id)
            .offset(offset)
            .limit(limit)
            .all()
        )
        annotation_counts = _annotation_counts(session, [r.id for r in rows])
        items = [_sample_item(r, annotation_counts.get(r.id, 0) > 0) for r in rows]
    return {"total": total, "limit": limit, "offset": offset, "samples": items}


def sample_detail(db, sample_id) -> dict | None:
    with db.session() as session:
        sample = session.get(Sample, sample_id)
        if sample is None:
            return None
        annotations = (
            session.query(Annotation)
            .filter(Annotation.sample_id == sample_id)
            .order_by(Annotation.created_at)
            .all()
        )
    has_image = _has_image(sample.image_path)
    item = _sample_item(sample, bool(annotations))
    item["created_at"] = sample.created_at
    result = {
        "sample": item,
        "annotations": [_annotation_item(a) for a in annotations],
        "frigate": _frigate_item(sample),
    }
    if has_image:
        result["image_url"] = f"/images/{sample.id}"
        result["thumb_url"] = f"/images/{sample.id}/thumb"
    return result


def _job_item(job) -> dict:
    return {
        "id": job.id,
        "type": job.type,
        "status": job.status,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "error": job.error,
    }


def _parse_json(value) -> object | None:
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _dataset_version(entry: Path) -> dict:
    build_path = entry / "build.json"
    if not build_path.is_file():
        return {"version": entry.name, "error": "build.json missing"}
    try:
        payload = json.loads(build_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"version": entry.name, "error": "build.json unreadable"}
    summary = payload.get("summary") or {}
    return {
        "version": payload.get("version", entry.name),
        "built_at": payload.get("built_at"),
        "images_written": summary.get("images_written"),
        "total": summary.get("total"),
        "verified": summary.get("verified"),
        "skipped_class": summary.get("skipped_class"),
        "split_counts": summary.get("split_counts"),
        "classes": list(payload.get("classes") or []),
    }


def _golden_segment(config) -> dict:
    root = config.golden_dir() / config.evaluation.golden_dataset
    exists = root.is_dir()
    classes: list[str] = []
    image_count = 0
    per_class: dict[str, int] = {}
    if exists:
        try:
            golden = GoldenDataset.load(root)
        except (OSError, ValueError):
            golden = None
        if golden is not None:
            classes = list(golden.classes)
            images_dir = golden.images_dir()
            if images_dir.is_dir():
                image_count = len([f for f in images_dir.iterdir() if f.is_file()])
            num_classes = max(1, len(classes))
            labels_dir = golden.labels_dir()
            if labels_dir.is_dir():
                for label_file in sorted(labels_dir.glob("*.txt")):
                    try:
                        for line in read_yolo_label(
                            label_file, num_classes=num_classes
                        ):
                            label = (
                                classes[line.class_id]
                                if 0 <= line.class_id < len(classes)
                                else f"class-{line.class_id}"
                            )
                            per_class[label] = per_class.get(label, 0) + 1
                    except (OSError, ValueError):
                        continue
    per_class = dict(sorted(per_class.items()))
    return {
        "version": config.evaluation.golden_dataset,
        "exists": exists,
        "classes": classes,
        "image_count": image_count,
        "per_class": per_class,
    }


def _split_run(name: str) -> tuple[str, str]:
    if "-" in name:
        model, _, tag = name.rpartition("-")
        return model or name, tag
    return name, ""


def _to_float_or_str(value: str | None) -> float | str | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return value
    return parsed if math.isfinite(parsed) else None


def _read_before_after(run_dir: Path) -> dict | None:
    path = run_dir / "before_after.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None

    def _segment(key: str) -> dict | None:
        seg = payload.get(key)
        if not isinstance(seg, dict):
            return None
        out: dict[str, float | None] = {}
        for metric in ("map50", "recall", "latency_ms"):
            value = seg.get(metric)
            if isinstance(value, (int, float)) and math.isfinite(value):
                out[metric] = float(value)
            else:
                out[metric] = None
        return out

    before = _segment("before")
    after = _segment("after")
    if before is None or after is None:
        return None
    return {"before": before, "after": after}


def _read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = [{k: _to_float_or_str(v) for k, v in row.items()} for row in reader]
        return list(reader.fieldnames or []), rows


def _has_image(image_path: str | None) -> bool:
    return bool(image_path) and Path(image_path).is_file()


def _annotation_counts(session, sample_ids: list[str]) -> dict[str, int]:
    if not sample_ids:
        return {}
    counts = dict(
        session.query(Annotation.sample_id, func.count(Annotation.id))
        .filter(Annotation.sample_id.in_(sample_ids))
        .group_by(Annotation.sample_id)
        .all()
    )
    return {sample_id: counts.get(sample_id, 0) for sample_id in sample_ids}


def _sample_item(sample, has_annotations: bool) -> dict:
    has_image = _has_image(sample.image_path)
    item = {
        "id": sample.id,
        "camera": sample.camera,
        "timestamp": sample.timestamp,
        "label": sample.frigate_label,
        "score": sample.frigate_score,
        "quality": sample.quality,
        "status": sample.status,
        "verified": sample.verified,
        "reviewed": sample.frigate_reviewed,
        "reviewed_at": sample.reviewed_at,
        "has_image": has_image,
        "has_annotations": has_annotations,
    }
    if has_image:
        item["image_url"] = f"/images/{sample.id}"
        item["thumb_url"] = f"/images/{sample.id}/thumb"
    return item


def _annotation_item(annotation) -> dict:
    return {
        "source": annotation.source,
        "label": annotation.label,
        "box": _box([annotation.x1, annotation.y1, annotation.x2, annotation.y2]),
        "confidence": annotation.confidence,
        "verified": annotation.verified,
    }


def _frigate_item(sample) -> dict | None:
    if sample.frigate_label is None:
        return None
    return {
        "label": sample.frigate_label,
        "score": sample.frigate_score,
        "box": _box(
            [sample.frigate_x1, sample.frigate_y1, sample.frigate_x2, sample.frigate_y2]
        ),
    }


def _box(coords: list[float | None]) -> list[float] | None:
    if any(c is None for c in coords):
        return None
    return [float(c) for c in coords]


__all__ = [
    "overview",
    "benchmark",
    "deployments",
    "quality_counts",
    "datasets",
    "training_index",
    "training_run",
    "samples",
    "sample_detail",
]