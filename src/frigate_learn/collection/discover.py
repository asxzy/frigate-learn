"""Missed-object discovery from motion activity (Phase 9).

Frigate only *records* detections that pass its detector; motion without a
detection is the "unknown" signal that the current model may be missing. This
module turns ``/api/review/activity/motion`` buckets into contiguous motion
windows, cross-references them against collected samples, and persists any
window with no collected evidence as a candidate for manual / VLM review.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

from ..config import AppConfig
from ..db import Database
from ..frigate.client import FrigateClient
from ..logutil import info
from ..models import DiscoveryWindow, Sample


@dataclass
class MotionWindow:
    camera: str
    start: float
    end: float
    peak_motion: float
    buckets: int
    covered: bool = False  # an existing collected sample falls inside


__all__ = ["MotionWindow", "find_motion_windows", "store_windows", "recent_discovery_windows"]


def _cluster_buckets(
    buckets, motion_threshold: float, closing_gap: float
) -> list[MotionWindow]:
    """Group motion buckets whose start_time is within ``closing_gap`` into
    windows. Buckets below the motion threshold are dropped first."""
    active: list[MotionWindow] = []
    prev_end: float | None = None
    current: MotionWindow | None = None

    for bucket in sorted(buckets, key=lambda b: (b.camera, b.start_time)):
        if bucket.motion < motion_threshold:
            continue
        if (
            current is not None
            and bucket.camera == current.camera
            and prev_end is not None
            and bucket.start_time - prev_end <= closing_gap
        ):
            current.end = max(current.end, bucket.start_time + 1.0)
            current.peak_motion = max(current.peak_motion, bucket.motion)
            current.buckets += 1
        else:
            if current is not None:
                active.append(current)
            bin_width = 1.0
            current = MotionWindow(
                camera=bucket.camera,
                start=float(bucket.start_time),
                end=float(bucket.start_time + bin_width),
                peak_motion=bucket.motion,
                buckets=1,
            )
        prev_end = current.end
    if current is not None:
        active.append(current)
    return active


def find_motion_windows(
    client: FrigateClient,
    config: AppConfig,
    db: Database,
    *,
    after: float,
    before: float | None = None,
    cameras: Sequence[str] | None = None,
    motion_threshold: float = 0.3,
    closing_gap: float | None = None,
) -> list[MotionWindow]:
    """Fetch motion buckets and return clusters not explained by samples."""
    buckets = client.get_motion_activity(after=after, before=before, cameras=cameras)
    closing_gap = closing_gap or 60.0
    windows = _cluster_buckets(buckets, motion_threshold, closing_gap)
    return _mark_covered(db, windows)


def _mark_covered(db: Database, windows: Sequence[MotionWindow]) -> list[MotionWindow]:
    if not windows:
        return []
    cameras = sorted({w.camera for w in windows})
    timestamps: dict[str, list[float]] = {}
    with db.session() as session:
        for camera in cameras:
            timestamps[camera] = [
                float(r[0])
                for r in session.query(Sample.timestamp).filter(Sample.camera == camera).all()
            ]
    out = []
    for w in windows:
        sample_times = timestamps.get(w.camera, [])
        w.covered = any(w.start <= ts <= w.end for ts in sample_times)
        out.append(w)
    return out


def store_windows(db: Database, windows: Sequence[MotionWindow]) -> int:
    """Persist discovery windows, skipping identical (camera, start) rows."""
    info("storing discovery windows", count=len(windows))
    stored = 0
    with db.session() as session:
        seen = {
            (row.camera, float(row.start_time))
            for row in session.query(DiscoveryWindow).all()
        }
        for w in windows:
            key = (w.camera, w.start)
            if key in seen:
                continue
            seen.add(key)
            session.add(
                DiscoveryWindow(
                    id=str(uuid.uuid4()),
                    camera=w.camera,
                    start_time=w.start,
                    end_time=w.end,
                    motion_score=w.peak_motion,
                    notes="motion without collected evidence"
                    if w.covered is False
                    else "motion cluster",
                    created_at=datetime.now(timezone.utc).isoformat(),
                )
            )
            stored += 1
        session.commit()
    return stored


def recent_discovery_windows(db: Database, camera: str | None = None, limit: int = 100) -> list[DiscoveryWindow]:
    with db.session() as session:
        query = session.query(DiscoveryWindow)
        if camera:
            query = query.filter(DiscoveryWindow.camera == camera)
        return query.order_by(DiscoveryWindow.start_time.desc()).limit(limit).all()


__all__ = ["MotionWindow", "find_motion_windows", "store_windows", "recent_discovery_windows"]