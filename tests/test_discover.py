"""Motion-only discovery tests (Phase 9)."""

from __future__ import annotations

from frigate_learn.collection.discover import (
    _cluster_buckets,
    find_motion_windows,
    store_windows,
)
from frigate_learn.frigate.motion import MotionBucket
from frigate_learn.models import DiscoveryWindow, Sample, utcnow


class FakeMotionClient:
    def __init__(self, buckets: list[MotionBucket]) -> None:
        self.buckets = buckets
        self.calls = []

    def get_motion_activity(self, after, before=None, cameras=None, **kwargs):
        self.calls.append({"after": after, "before": before, "cameras": cameras})
        if cameras:
            wanted = set(cameras)
            return [b for b in self.buckets if b.camera in wanted]
        return self.buckets

    def close(self):
        return None


def _bucket(camera, start, motion=0.6):
    return MotionBucket(start_time=start, motion=motion, camera=camera)


def test_cluster_merges_close_buckets_per_camera():
    buckets = [
        _bucket("front", 100),
        _bucket("front", 110),   # within closing gap of 60
        _bucket("front", 300),   # far away -> new window
        _bucket("deck", 120),
    ]
    windows = _cluster_buckets(buckets, motion_threshold=0.3, closing_gap=60)
    front = sorted((w for w in windows if w.camera == "front"), key=lambda w: w.start)
    deck = [w for w in windows if w.camera == "deck"]
    assert len(windows) == 3
    assert [w.start for w in front] == [100, 300]
    assert front[0].end == 111  # 110 + 1.0 bin width
    assert front[0].buckets == 2
    assert deck[0].start == 120


def test_bucket_below_threshold_dropped():
    buckets = [
        _bucket("front", 100, motion=0.2),
        _bucket("front", 120, motion=0.9),
    ]
    windows = _cluster_buckets(buckets, motion_threshold=0.3, closing_gap=60)
    assert len(windows) == 1
    assert windows[0].buckets == 1


def test_find_marks_covered_from_samples(config, db):
    with db.session() as s:
        s.add(
            Sample(
                id="covered", camera="front", timestamp=105.0, event_id=None,
                source="frigate", status="collected", created_at=utcnow(),
            )
        )
        s.commit()
    client = FakeMotionClient(
        [
            _bucket("front", 100),
            _bucket("front", 110),
            _bucket("deck", 200),
        ]
    )
    windows = find_motion_windows(
        client, config, db, after=0, before=1000, motion_threshold=0.3, closing_gap=60
    )
    by_camera = {w.camera: w for w in windows}
    assert by_camera["front"].covered is True
    assert by_camera["deck"].covered is False


def test_find_passes_filters_to_client_but_cameras_tuple_is_expanded(config, db):
    client = FakeMotionClient([_bucket("front", 100)])
    find_motion_windows(
        client, config, db, after=10, before=20, cameras=["front"], motion_threshold=0.0
    )
    assert client.calls == [{"after": 10, "before": 20, "cameras": ["front"]}]


def test_store_windows_idempotent(config, db):
    from frigate_learn.collection.discover import MotionWindow

    windows = [
        MotionWindow(camera="front", start=100, end=140, peak_motion=0.7, buckets=3),
        MotionWindow(camera="front", start=100, end=140, peak_motion=0.7, buckets=3),
        MotionWindow(camera="deck", start=200, end=240, peak_motion=0.5, buckets=2),
    ]
    assert store_windows(db, windows) == 2
    with db.session() as s:
        rows = s.query(DiscoveryWindow).all()
    assert len(rows) == 2
    assert rows[0].motion_score == 0.7