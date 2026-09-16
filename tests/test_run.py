"""Pipeline orchestrator tests (Phase 13, dry-run / stubbed ML)."""

from __future__ import annotations

import os

from PIL import Image

from frigate_learn.collection.collector import CollectSummary
from frigate_learn.collection.review_sync import ReviewSyncSummary
from frigate_learn.models import Annotation, Sample, utcnow
from frigate_learn.run import PIPELINE, next_build_version, run_pipeline

IMAGE_BYTES = b"\xff\xd8\xff\xe0fake-jpeg"


class FakeCollector:
    def __init__(self, *args, **kwargs):
        self.client = type("C", (), {"close": lambda self: None})()

    def collect(self, from_ts, to_ts=None, cameras=None, labels=None, severity=None,
                limit=None, concurrency=None, progress=None):
        if progress:
            progress("Reviews found: 1")
        return CollectSummary(
            reviews_found=1, reviews_selected=1, events_found=1, events_new=1,
            new_samples=1, duplicate_samples=0, failures=0, new_annotations=1,
            range_from=from_ts, range_to=to_ts, duration_seconds=0.1,
            job_id="job1",
        )


def _seed_sample(config, db, tmp_path, sample_id="s1"):
    img_dir = config.images_dir() / "20260901" / "front"
    img_dir.mkdir(parents=True, exist_ok=True)
    img = img_dir / "s1.jpg"
    Image.new("RGB", (32, 32), (60, 60, 60)).save(img)
    with db.session() as s:
        s.add(
            Sample(
                id=sample_id, camera="front", timestamp=100.0, event_id="e1",
                frame_index=0, image_path=str(img), source="frigate",
                frigate_label="person", status="collected", created_at=utcnow(),
            )
        )
        s.flush()  # persist the sample before its FK-dependent annotation
        s.add(
            Annotation(
                id=f"a{sample_id}", sample_id=sample_id, source="frigate",
                label="person", x1=0.1, y1=0.1, x2=0.6, y2=0.8,
                confidence=0.9, verified=1, created_at=utcnow(),
            )
        )
        s.commit()
    return img


def test_next_build_version_starts_at_v001(config):
    assert next_build_version(config) == "v001"


def test_next_build_version_increments(config, tmp_path):
    datasets = config.datasets_dir() / "v007"
    datasets.mkdir(parents=True)
    (datasets / "build.json").write_text("{}", encoding="utf-8")
    assert next_build_version(config) == "v008"


def test_run_collect_build_with_stubbed_collector(config, db, tmp_path, monkeypatch):
    _seed_sample(config, db, tmp_path)
    monkeypatch.setattr("frigate_learn.collection.collector.Collector", FakeCollector)

    reports = run_pipeline(config, db, steps=["collect", "verify", "build", "deploy"], dry_run=True)
    by_name = {r.name: r for r in reports}
    assert by_name["collect"].status == "executed"
    assert by_name["verify"].status == "skipped"   # vlm.enabled=False
    assert by_name["build"].status == "executed"
    assert by_name["build"].data["version"] == "v001"
    assert by_name["build"].data["written"] == 1
    assert by_name["deploy"].status == "skipped"   # nothing trained this run


def test_run_until_truncates_steps(config, db, monkeypatch):
    monkeypatch.setattr("frigate_learn.collection.collector.Collector", FakeCollector)
    reports = run_pipeline(
        config, db,
        steps=["collect", "build", "train", "gate"],
        until="build",
        dry_run=True,
    )
    names = [r.name for r in reports]
    assert names == ["collect", "build"]


def _make_existing_version(config, version="v007"):
    datasets = config.datasets_dir() / version
    datasets.mkdir(parents=True, exist_ok=True)
    (datasets / "build.json").write_text("{}", encoding="utf-8")
    return version


def test_run_failed_step_halts_without_keep_going(config, db, tmp_path, monkeypatch):
    _seed_sample(config, db, tmp_path)
    monkeypatch.setattr("frigate_learn.collection.collector.Collector", FakeCollector)
    version = _make_existing_version(config)
    monkeypatch.setattr("frigate_learn.run.next_build_version", lambda c: version)
    reports = run_pipeline(config, db, steps=["collect", "build", "gate"], dry_run=True)
    by_name = {r.name: r for r in reports}
    assert by_name["collect"].status == "executed"
    assert by_name["build"].status == "failed"   # version already exists
    assert len(reports) == 2  # halted after build, no gate


def test_run_keep_going_continues_past_failure(config, db, tmp_path, monkeypatch):
    _seed_sample(config, db, tmp_path)
    monkeypatch.setattr("frigate_learn.collection.collector.Collector", FakeCollector)
    version = _make_existing_version(config)
    monkeypatch.setattr("frigate_learn.run.next_build_version", lambda c: version)
    reports = run_pipeline(
        config, db, steps=["collect", "build", "gate"], dry_run=True, keep_going=True
    )
    by_name = {r.name: r for r in reports}
    assert by_name["build"].status == "failed"
    assert by_name["gate"].status == "skipped"
    assert len(reports) == 3


def test_pipeline_steps_registry_is_ordered():
    assert PIPELINE == [
        "collect", "review-sync", "verify", "build", "train", "benchmark", "gate", "deploy"
    ]


class FakeReviewSyncer:
    def __init__(self, config, db):
        self.client = type("C", (), {"close": lambda self: None})()

    def sync(self, from_ts, to_ts=None, cameras=None, severity=None,
             auto_useful=None, progress=None):
        return ReviewSyncSummary(
            reviews_found=2, reviews_seen=2, samples_matched=1,
            flags_changed=1, auto_useful=1, job_id="rev1", duration_seconds=0.05,
        )


def test_run_review_sync_step_executes(config, db, monkeypatch):
    monkeypatch.setattr(
        "frigate_learn.collection.review_sync.ReviewSyncer", FakeReviewSyncer
    )
    reports = run_pipeline(config, db, steps=["review-sync"], dry_run=True)
    assert len(reports) == 1
    report = reports[0]
    assert report.name == "review-sync"
    assert report.status == "executed"
    assert "auto_useful=1" in report.message


def test_latest_trained_run_none_when_training_dir_missing(config, tmp_path):
    from frigate_learn.evaluation.benchmark import latest_trained_run
    assert latest_trained_run(config) is None


def test_latest_trained_run_picks_newest_with_weights(config, tmp_path):
    from frigate_learn.evaluation.benchmark import latest_trained_run
    root = config.resolve(config.data.root, "training")
    old = root / "yolov8s-v001"
    new = root / "yolov8n-v003"
    for d, mtime in ((old, 10), (new, 20)):
        (d / "weights").mkdir(parents=True)
        (d / "weights" / "best.pt").write_bytes(b"x")
        (d / "results.csv").write_text("epoch\n", encoding="utf-8")
        os.utime(d / "results.csv", (mtime, mtime))
    name, weights = latest_trained_run(config)
    assert name == "yolov8n-v003"
    assert weights == new / "weights" / "best.pt"


def test_latest_trained_run_skips_dir_without_weights(config, tmp_path):
    from frigate_learn.evaluation.benchmark import latest_trained_run
    root = config.resolve(config.data.root, "training")
    (root / "yolov8n-v001" / "weights").mkdir(parents=True)
    (root / "yolov8n-v001" / "results.csv").write_text("epoch\n", encoding="utf-8")
    assert latest_trained_run(config) is None