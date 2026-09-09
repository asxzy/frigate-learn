"""Frigate motion-activity parsing (used by P9 missed-object discovery).

``GET /api/review/activity/motion`` returns a list of
``{"start_time": int, "motion": float, "camera": str}`` buckets averaged over a
``scale`` window. Collected now (client + model), consumed by the P9
motion-only candidate sampling phase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .reviews import parse_ts


@dataclass
class MotionBucket:
    start_time: int
    motion: float
    camera: str


def parse_motion_activity(raw: list[Mapping[str, Any]]) -> list[MotionBucket]:
    buckets: list[MotionBucket] = []
    for item in raw or []:
        if not isinstance(item, Mapping):
            continue
        ts = item.get("start_time")
        if ts is None:
            continue
        bucket = MotionBucket(
            start_time=int(parse_ts(ts) or 0),
            motion=float(item.get("motion") or 0.0),
            camera=str(item.get("camera", "")),
        )
        buckets.append(bucket)
    return buckets


__all__ = ["MotionBucket", "parse_motion_activity"]