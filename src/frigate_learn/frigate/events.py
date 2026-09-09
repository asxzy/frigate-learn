"""Frigate 0.18 event parsing.

The events endpoints return objects with (at least) ``id, camera, label,
start_time, end_time, top_score, false_positive, zones, has_clip, has_snapshot,
plus_id, sub_label, box, data``. ``box`` is a normalized ``[x1, y1, x2, y2]``
list in [0,1] — this is the clean bbox source for collected samples.
``data`` holds ``score`` (final confidence), ``region``, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .reviews import parse_ts

Box = tuple[float, float, float, float] | None


@dataclass
class Event:
    id: str
    camera: str
    label: str
    start_time: float
    end_time: float | None
    top_score: float | None
    score: float | None
    false_positive: bool | None
    zones: list[str]
    has_clip: bool | None
    has_snapshot: bool | None
    sub_label: str | None = None
    plus_id: str | None = None
    box: Box = None
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def completed(self) -> bool:
        return self.end_time is not None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_box(value: Any) -> Box:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    box = tuple(_optional_float(v) for v in value)
    if any(v is None for v in box):
        return None
    return box  # type: ignore[return-value]


def parse_event(raw: Mapping[str, Any]) -> Event:
    data = raw.get("data") or {}
    if not isinstance(data, dict):
        data = {}
    box = _optional_box(raw.get("box")) or _optional_box(data.get("box"))
    score = _optional_float(data.get("score"))
    return Event(
        id=str(raw.get("id", "")),
        camera=str(raw.get("camera", "")),
        label=str(raw.get("label", "")),
        start_time=parse_ts(raw.get("start_time")) or 0.0,
        end_time=parse_ts(raw.get("end_time")),
        top_score=_optional_float(raw.get("top_score")),
        score=score if score is not None else _optional_float(raw.get("score")),
        false_positive=_optional_bool(raw.get("false_positive")),
        zones=[str(z) for z in (raw.get("zones") or [])],
        has_clip=_optional_bool(raw.get("has_clip")),
        has_snapshot=_optional_bool(raw.get("has_snapshot")),
        sub_label=raw.get("sub_label"),
        plus_id=raw.get("plus_id"),
        box=box,
        data=data,
    )


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


__all__ = ["Event", "parse_event", "Box"]