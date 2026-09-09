"""Frigate event parsing tests."""

from __future__ import annotations

import pytest

from frigate_learn.frigate.events import parse_event


def test_parse_event_box_from_raw():
    evt = parse_event(
        {
            "id": "e1",
            "camera": "front",
            "label": "person",
            "start_time": 1786723200.0,
            "end_time": 1786723270.0,
            "top_score": 0.87,
            "false_positive": False,
            "zones": ["home"],
            "has_clip": True,
            "has_snapshot": True,
            "box": [0.1, 0.2, 0.4, 0.8],
            "data": {"score": 0.9},
        }
    )
    assert evt.id == "e1"
    assert evt.label == "person"
    assert evt.box == (0.1, 0.2, 0.4, 0.8)
    assert evt.score == pytest.approx(0.9)
    assert evt.top_score == pytest.approx(0.87)
    assert evt.completed is True


def test_parse_event_box_from_data():
    evt = parse_event(
        {
            "id": "e2",
            "camera": "front",
            "label": "dog",
            "start_time": 1.0,
            "data": {"box": [0.0, 0.0, 0.5, 0.5], "score": 0.6},
        }
    )
    assert evt.box == (0.0, 0.0, 0.5, 0.5)
    assert evt.completed is False


def test_incomplete_box_gives_none():
    evt = parse_event(
        {
            "id": "e3",
            "camera": "front",
            "label": "car",
            "start_time": 1.0,
            "box": [0.1, 0.2],
        }
    )
    assert evt.box is None
    assert evt.score is None
    assert evt.false_positive is None


def test_incomplete_event():
    evt = parse_event({"id": "e4", "camera": "front", "label": "person", "start_time": 1.0})
    assert evt.end_time is None
    assert evt.completed is False