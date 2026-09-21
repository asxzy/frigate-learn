"""Frigate 0.18 review/event data models and parsing.

API facts (isolated here):

- ``GET /api/review`` lists review segments as a plain JSON array.
- Each item has ``id/camera/start_time/end_time/severity/thumb_path/data``.
- ``data.detections`` is a list of **event ids** (strings). ``data.objects`` is a
  list of labels. Older 0.16/0.17 formats carried dicts in both lists; the
  parsers below accept both shapes.
- The list endpoint serializes timestamps as ISO-8601 strings; the single-item
  endpoint returns unix floats. `parse_ts()` handles both.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping

SEVERITIES = ("alert", "detection")


@dataclass
class Review:
    id: str
    camera: str
    severity: str
    thumb_path: str
    start_time: float
    end_time: float | None
    data: dict[str, Any] = field(default_factory=dict)
    zones: list[str] = field(default_factory=list)


def parse_ts(value: Any) -> float | None:
    """Tolerantly parse a Frigate timestamp (unix float or ISO-8601 string)."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        try:
            return float(text)  # numeric string
        except ValueError:
            pass
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            text2 = text.rstrip("Z")
            try:
                dt = datetime.fromisoformat(text2)
            except ValueError:
                try:
                    return float(text2)
                except ValueError:
                    raise ValueError(f"cannot parse timestamp: {value!r}")
        if dt.tzinfo is None:
            return dt.timestamp()
        return dt.timestamp()
    raise ValueError(f"cannot parse timestamp: {value!r}")


def _ids_from_list(items: Any) -> list[str]:
    """Extract event ids from ``data.detections`` / ``data.objects``.

    Accepts the 0.18 string-list format and the older dict-list format.
    """
    if not items:
        return []
    ids: list[str] = []
    for item in items:
        if isinstance(item, str):
            ids.append(item)
        elif isinstance(item, Mapping):
            value = item.get("id")
            if value:
                ids.append(str(value))
    return ids


def parse_review(raw: Mapping[str, Any]) -> Review:
    data = raw.get("data") or {}
    if not isinstance(data, dict):
        data = {}
    return Review(
        id=str(raw.get("id", "")),
        camera=str(raw.get("camera", "")),
        severity=str(raw.get("severity", "detection")),
        thumb_path=str(raw.get("thumb_path", "") or ""),
        start_time=parse_ts(raw.get("start_time")) or 0.0,
        end_time=parse_ts(raw.get("end_time")),
        data=data,
        zones=[str(z) for z in (raw.get("zones") or [])],
    )


def extract_event_ids(review: Review) -> list[str]:
    """The event/object ids underlying a review item.

    A Review is NOT equivalent to one object: it can contain many tracked
    objects. This returns each underlying event id for individual processing.
    """
    event_ids = _ids_from_list(review.data.get("detections") or [])
    if not event_ids:
        # older reviews may carry events only under data.objects
        event_ids = _ids_from_list(review.data.get("objects") or [])
    # deterministic order, deduped
    seen: set[str] = set()
    out: list[str] = []
    for eid in event_ids:
        if eid and eid not in seen:
            seen.add(eid)
            out.append(eid)
    return out


__all__ = ["Review", "parse_review", "extract_event_ids", "parse_ts", "SEVERITIES"]