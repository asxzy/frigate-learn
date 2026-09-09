"""Frigate review parsing tests."""

from __future__ import annotations

import pytest

from frigate_learn.frigate.reviews import (
    extract_event_ids,
    parse_review,
    parse_ts,
)


def test_parse_review_iso_and_float_timestamps():
    from datetime import datetime, timezone

    ISO_TS = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc).timestamp()
    raw = {
        "id": "r1",
        "camera": "front",
        "start_time": "2026-09-01T12:00:00+00:00",
        "end_time": 1786723200.0,
        "severity": "detection",
        "thumb_path": "/api/media/front/thumb.jpg",
        "has_been_reviewed": False,
        "zones": ["home"],
        "data": {"detections": ["e1", "e2"], "objects": ["person"]},
    }
    review = parse_review(raw)
    assert review.id == "r1"
    assert review.camera == "front"
    assert review.severity == "detection"
    assert review.has_been_reviewed is False
    assert review.zones == ["home"]
    assert review.end_time == pytest.approx(1786723200.0)
    assert review.start_time == pytest.approx(ISO_TS)


def test_extract_event_ids_strings():
    review = parse_review({"id": "r", "data": {"detections": ["a", "b", "a"]}})
    assert extract_event_ids(review) == ["a", "b"]


def test_extract_event_ids_dicts():
    review = parse_review(
        {"id": "r", "data": {"detections": [{"id": "x"}, {"id": 5}, {}]}}
    )
    assert extract_event_ids(review) == ["x", "5"]


def test_extract_event_ids_objects_fallback():
    review = parse_review({"id": "r", "data": {"objects": ["obj1"]}})
    assert extract_event_ids(review) == ["obj1"]


def test_extract_event_ids_empty_data():
    review = parse_review({"id": "r"})
    assert extract_event_ids(review) == []


@pytest.mark.parametrize(
    "value",
    [
        1786723200.0,
        1786723200,
        "1786723200",
        "2026-09-01T12:00:00+00:00",
        "2026-09-01T12:00:00Z",
        "2026-09-01T12:00:00",
        None,
    ],
)
def test_parse_ts(value):
    if value is None:
        assert parse_ts(value) is None
        return
    from datetime import datetime, timezone

    if isinstance(value, str) and value.lstrip("-").replace(".", "", 1).isdigit():
        assert parse_ts(value) == float(value)
        return
    if isinstance(value, (int, float)):
        assert parse_ts(value) == float(value)
        return
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    # naive datetime is interpreted as host-local time by parse_ts
    assert parse_ts(value) == parsed.timestamp()


def test_parse_ts_invalid():
    with pytest.raises(ValueError):
        parse_ts("not-a-time")