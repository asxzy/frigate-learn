"""Temporal sampling within an event (Phase 3).

Frigate stores one snapshot per event by default. ``sample_timestamps`` picks
up to ``n`` spaced-out frame times inside ``[start, end]`` respecting a minimum
inter-frame gap. `collect` uses these times with the events snapshot endpoint's
``timestamp`` query parameter to store multiple frames per event (one sample row
per frame, ``frame_index`` increments).
"""

from __future__ import annotations

import math


def sample_timestamps(
    start: float,
    end: float | None,
    n: int,
    min_gap: float = 2.0,
) -> list[float]:
    """Choose up to ``n`` timestamps evenly spread within the event window.

    Rules:
    - n <= 1 or no positive range -> ``[start]``.
    - The gap between consecutive choices is at least ``min_gap``.
    - Timestamps are returned ascending and clamped inside [start, end].
    """
    if n <= 1 or end is None or end <= start:
        return [start]
    available = end - start
    max_feasible = int(math.floor(available / min_gap)) + 1
    count = min(n, max_feasible)
    if count <= 1:
        return [start]
    if count == 2:
        return [start, end]
    out: list[float] = []
    for i in range(count):
        frac = i / (count - 1) if count > 1 else 0.0
        t = start + available * frac
        # keep strictly inside the window so Frigate's snapshot lookup succeeds
        out.append(round(min(max(t, start), end), 3))
    return out


__all__ = ["sample_timestamps"]