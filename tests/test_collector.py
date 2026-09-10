"""Collector integration tests (Frigate HTTP replaced by a fake client)."""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

import pytest
from sqlalchemy import text

from frigate_learn.collection.collector import Collector
from frigate_learn.config import AppConfig
from frigate_learn.db import Database
from frigate_learn.frigate.client import FrigateAPIError
from frigate_learn.frigate.events import parse_event
from frigate_learn.frigate.reviews import parse_review
from frigate_learn.models import Annotation, Sample


class FakeFrigate:
    """Duck-typed stand-in for FrigateClient."""

    def __init__(
        self,
        reviews_raw: list[dict],
        events_raw: dict[str, dict],
        fail_ids: set[str] | None = None,
    ) -> None:
        self.reviews = [parse_review(r) for r in reviews_raw]
        self.events = {eid: parse_event(raw) for eid, raw in events_raw.items()}
        self.fail_ids = fail_ids or set()
        self.closed = False
        self.downloads: list[tuple[str, str, Path, object]] = []

    def list_reviews(self, after, before=None, cameras=None, labels=None,
                     severity=None, limit=None, **kw):
        return self.reviews

    def get_event(self, event_id: str) -> object:
        if event_id in self.fail_ids:
            raise FrigateAPIError("boom", status_code=404, url=f"/api/events/{event_id}")
        return self.events[event_id]

    def download_clean_snapshot(self, event_id: str, output_path, timestamp=None) -> str:
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"\xff\xd8\xff\xe0fakejpeg-" + event_id.encode())
        self.downloads.append(("clean", event_id, dest, timestamp))
        return str(dest)

    def download_event_snapshot(self, event_id: str, output_path) -> str:
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"annotated-" + event_id.encode())
        self.downloads.append(("debug", event_id, dest))
        return str(dest)

    def download_region_crop(self, event_id, output_path, height, timestamp=None) -> str:
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"\xff\xd8\xff\xe0fakecrop-" + event_id.encode())
        self.downloads.append(("crop", event_id, dest, {"height": height, "timestamp": timestamp}))
        return str(dest)

    def download_annotated_crop(self, event_id, output_path) -> str:
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"annotated-crop-" + event_id.encode())
        self.downloads.append(("debug-crop", event_id, dest))
        return str(dest)

    def close(self) -> None:
        self.closed = True


def _review(r_id: str, camera: str, start: float, detections: list[str]) -> dict:
    return {
        "id": r_id,
        "camera": camera,
        "severity": "detection",
        "start_time": start,
        "end_time": start + 60,
        "thumb_path": "/x",
        "data": {"detections": detections},
    }


_MISSING = object()


def _event(e_id: str, camera: str, label: str, start: float,
           end_time: object = _MISSING, box=None, score: float = 0.9) -> dict:
    resolved_end = start + 60 if end_time is _MISSING else end_time
    return {
        "id": e_id,
        "camera": camera,
        "label": label,
        "start_time": start,
        "end_time": resolved_end,
        "top_score": 0.8,
        "false_positive": False,
        "zones": [],
        "has_clip": True,
        "has_snapshot": True,
        "box": box or [0.1, 0.2, 0.3, 0.6],
        "data": {"score": score},
    }


def test_happy_path(config, db, tmp_path):
    config.collection.region_crop = False
    fake = FakeFrigate(
        reviews_raw=[
            _review("r1", "front", 200, ["e1"]),
            _review("r2", "front", 100, ["e2", "e3"]),
        ],
        events_raw={
            "e1": _event("e1", "front", "person", 200),
            "e2": _event("e2", "front", "car", 100),
            "e3": _event("e3", "front", "dog", 90),
        },
    )
    collector = Collector(config, db, client=fake)
    summary = collector.collect(from_ts=0)

    assert summary.new_samples == 3
    assert summary.new_annotations == 3
    assert summary.events_found == 3
    assert summary.events_new == 3
    assert summary.duplicate_samples == 0
    assert summary.failures == 0
    assert summary.job_id is not None
    assert fake.closed is True

    with db.session() as s:
        rows = s.query(Sample).all()
    assert len(rows) == 3
    row = {r.event_id: r for r in rows}["e1"]
    assert row.camera == "front"
    assert row.frigate_label == "person"
    assert row.frigate_score == 0.9
    assert (row.frigate_x1, row.frigate_y1, row.frigate_x2, row.frigate_y2) == (0.1, 0.2, 0.4, 0.8)
    assert row.review_id == "r1"
    image = Path(row.image_path)
    assert image.is_file()
    assert row.image_hash == hashlib.sha256(image.read_bytes()).hexdigest()
    assert row.status == "collected"

    # day folder scheme: events are >0 unix so they fall under a fixed day
    assert image.parent.name == "front"


def test_idempotent_second_run(config, db):
    config.collection.region_crop = False
    fake = FakeFrigate(
        reviews_raw=[_review("r1", "front", 200, ["e1", "e2"])],
        events_raw={
            "e1": _event("e1", "front", "person", 200),
            "e2": _event("e2", "front", "car", 150),
        },
    )
    collector = Collector(config, db, client=fake)
    first = collector.collect(from_ts=0)
    assert first.new_samples == 2

    fake2 = FakeFrigate(
        reviews_raw=[_review("r1", "front", 200, ["e1", "e2"])],
        events_raw={
            "e1": _event("e1", "front", "person", 200),
            "e2": _event("e2", "front", "car", 150),
        },
    )
    second = Collector(config, db, client=fake2).collect(from_ts=0)
    assert second.new_samples == 0
    assert second.duplicate_samples == 2
    assert second.events_new == 0

    with db.session() as s:
        count = s.execute(text("SELECT COUNT(*) FROM samples")).scalar()
    assert count == 2


def test_shared_event_deduped_across_reviews(config, db):
    config.collection.region_crop = False
    fake = FakeFrigate(
        reviews_raw=[
            _review("r1", "front", 200, ["e1"]),
            _review("r2", "front", 195, ["e1"]),
        ],
        events_raw={"e1": _event("e1", "front", "person", 200)},
    )
    summary = Collector(config, db, client=fake).collect(from_ts=0)
    assert summary.events_found == 1
    assert summary.new_samples == 1


def test_failures_recorded(config, db):
    config.collection.region_crop = False
    fake = FakeFrigate(
        reviews_raw=[
            _review("r1", "front", 200, ["e_ok", "e_fail"]),
        ],
        events_raw={"e_ok": _event("e_ok", "front", "person", 200)},
        fail_ids={"e_fail"},
    )
    summary = Collector(config, db, client=fake).collect(from_ts=0)
    assert summary.new_samples == 1
    assert summary.failures == 1

    with db.session() as s:
        failures = s.execute(
            text("SELECT event_id, error FROM collection_failures")
        ).all()
        jobs = s.execute(text("SELECT status FROM jobs")).all()
    assert [(f.event_id) for f in failures] == ["e_fail"]
    assert "boom" in failures[0].error
    assert jobs[0].status == "finished"


def test_incomplete_event_skipped_when_completed_only(config, db):
    config.collection.region_crop = False
    fake = FakeFrigate(
        reviews_raw=[_review("r1", "front", 200, ["e_pending"])],
        events_raw={
            "e_pending": _event("e_pending", "front", "person", 200, end_time=None),
        },
    )
    summary = Collector(config, db, client=fake).collect(from_ts=0)
    assert summary.new_samples == 0
    assert summary.failures == 1
    with db.session() as s:
        error = s.execute(
            text("SELECT error FROM collection_failures")
        ).scalar()
    assert "not completed" in error


def test_keep_annotated_snapshots(config, db):
    config.collection.region_crop = False
    config.collection.keep_annotated_snapshots = True
    fake = FakeFrigate(
        reviews_raw=[_review("r1", "front", 200, ["e1"])],
        events_raw={"e1": _event("e1", "front", "person", 200)},
    )
    summary = Collector(config, db, client=fake).collect(from_ts=0)
    assert summary.new_samples == 1
    with db.session() as s:
        row = s.query(Sample).one()
    assert row.debug_image_path is not None
    assert Path(row.debug_image_path).is_file()


def test_labels_cameras_filters_passthrough(config, db):
    config.collection.region_crop = False

    class RecordingFake(FakeFrigate):
        def __init__(self):
            super().__init__([], {})
            self.last_kwargs = None

        def list_reviews(self, *args, **kwargs):
            self.last_kwargs = kwargs
            return []

    fake = RecordingFake()
    collector = Collector(config, db, client=fake)
    summary = collector.collect(
        from_ts=100,
        to_ts=200,
        cameras=["front"],
        labels=["person"],
        severity=["alert"],
    )
    assert summary.range_from == 100
    assert summary.range_to == 200
    assert fake.last_kwargs is not None
    assert fake.last_kwargs["after"] == 100
    assert fake.last_kwargs["before"] == 200
    assert fake.last_kwargs["cameras"] == ["front"]
    assert fake.last_kwargs["labels"] == ["person"]
    assert fake.last_kwargs["severity"] == ["alert"]


def test_region_crop_default_collects_crops_not_full_frames(config, db):
    fake = FakeFrigate(
        reviews_raw=[_review("r1", "front", 200, ["e1"])],
        events_raw={"e1": _event("e1", "front", "person", 200)},
    )
    summary = Collector(config, db, client=fake).collect(from_ts=0)
    assert summary.new_samples == 1
    assert summary.new_annotations == 0
    assert [d[0] for d in fake.downloads] == ["crop"]
    with db.session() as s:
        row = s.query(Sample).one()
        attrs = s.query(Annotation).all()
    assert attrs == []
    assert (row.frigate_x1, row.frigate_y1, row.frigate_x2, row.frigate_y2) == (0.1, 0.2, 0.4, 0.8)
    image = Path(row.image_path)
    assert image.is_file()
    assert image.read_bytes().startswith(b"\xff\xd8\xff\xe0fakecrop-")


def test_region_crop_default_height_from_training(config, db):
    fake = FakeFrigate(
        reviews_raw=[_review("r1", "front", 200, ["e1"])],
        events_raw={"e1": _event("e1", "front", "person", 200)},
    )
    Collector(config, db, client=fake).collect(from_ts=0)
    calls = [d for d in fake.downloads if d[0] == "crop"]
    assert calls[0][3]["height"] == config.training.image_size


def test_region_crop_configured_height(config, db):
    config.collection.region_crop_height = 512
    fake = FakeFrigate(
        reviews_raw=[_review("r1", "front", 200, ["e1"])],
        events_raw={"e1": _event("e1", "front", "person", 200)},
    )
    Collector(config, db, client=fake).collect(from_ts=0)
    calls = [d for d in fake.downloads if d[0] == "crop"]
    assert calls[0][3]["height"] == 512


def test_region_crop_skips_degenerate_box(config, db):
    fake = FakeFrigate(
        reviews_raw=[_review("r1", "front", 200, ["e_bad"])],
        events_raw={
            "e_bad": _event("e_bad", "front", "person", 200, box=[0.0, 0.0, 0.0, 0.0]),
        },
    )
    summary = Collector(config, db, client=fake).collect(from_ts=0)
    assert summary.new_samples == 0
    assert summary.skipped_no_box == 1
    assert fake.downloads == []
    with db.session() as s:
        assert s.query(Sample).count() == 0


def test_region_crop_forces_single_frame(config, db):
    config.sampling.enabled = True
    config.sampling.max_samples_per_event = 3
    fake = FakeFrigate(
        reviews_raw=[_review("r1", "front", 200, ["e1"])],
        events_raw={"e1": _event("e1", "front", "person", 200)},
    )
    summary = Collector(config, db, client=fake).collect(from_ts=0)
    assert summary.new_samples == 1
    crops = [d for d in fake.downloads if d[0] == "crop"]
    assert len(crops) == 1
    assert crops[0][3]["timestamp"] is None


def test_region_crop_debug_uses_annotated_crop(config, db):
    config.collection.keep_annotated_snapshots = True
    fake = FakeFrigate(
        reviews_raw=[_review("r1", "front", 200, ["e1"])],
        events_raw={"e1": _event("e1", "front", "person", 200)},
    )
    summary = Collector(config, db, client=fake).collect(from_ts=0)
    assert summary.new_samples == 1
    kinds = [d[0] for d in fake.downloads]
    assert "debug-crop" in kinds
    assert "clean" not in kinds
    assert "debug" not in kinds