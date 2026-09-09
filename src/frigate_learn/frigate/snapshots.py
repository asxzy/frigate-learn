"""Frigate media / snapshot endpoints.

Frigate 0.18 does NOT expose ``/api/events/{id}/snapshot-clean.webp`` as assumed
in the design. The clean (unannotated) frame is served by the same endpoint as
the annotated one, with overlays disabled:

    GET /api/events/{id}/snapshot.jpg?bbox=0&timestamp=0

These URL builders are the ONLY place that knows this. Everything else calls
``FrigateClient.download_clean_snapshot(...)`` etc.
"""

from __future__ import annotations

from typing import Any


def clean_snapshot_params(**overrides: Any) -> dict[str, Any]:
    params: dict[str, Any] = {"bbox": 0, "timestamp": 0, "download": 1}
    params.update({k: v for k, v in overrides.items() if v is not None})
    return params


def annotated_snapshot_params(**overrides: Any) -> dict[str, Any]:
    params: dict[str, Any] = {"download": 1}
    params.update({k: v for k, v in overrides.items() if v is not None})
    return params


def clean_snapshot_url(event_id: str) -> str:
    return f"/api/events/{event_id}/snapshot.jpg"


def annotated_snapshot_url(event_id: str) -> str:
    return f"/api/events/{event_id}/snapshot.jpg"


def review_preview_url(review_id: str, fmt: str = "mp4") -> str:
    if fmt not in {"mp4", "gif"}:
        raise ValueError(f"unsupported preview format: {fmt!r}")
    return f"/api/review/{review_id}/preview?format={fmt}"


def motion_activity_url() -> str:
    return "/api/review/activity/motion"


__all__ = [
    "clean_snapshot_params",
    "annotated_snapshot_params",
    "clean_snapshot_url",
    "annotated_snapshot_url",
    "review_preview_url",
    "motion_activity_url",
]