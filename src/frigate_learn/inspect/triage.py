"""Quality triage (Phase 2).

Writes the human browser verdict back into ``samples.quality``. Verdict values
are fixed so downstream phases (dedup, verification, dataset build) can rely on
them: ``useful | bad | duplicate | ignore``.
"""

from __future__ import annotations

from datetime import datetime, timezone

from ..config import AppConfig
from ..db import Database
from ..models import Annotation, Sample

QUALITY_VALUES = {"useful", "bad", "duplicate", "ignore"}


def validate_quality(value: str) -> str:
    if value not in QUALITY_VALUES:
        raise ValueError(
            f"invalid quality {value!r}; expected one of {sorted(QUALITY_VALUES)}"
        )
    return value


def set_sample_quality(config: AppConfig, db: Database, sample_id: str, quality: str) -> bool:
    """Set quality for one sample; returns True when a row changed."""
    if quality is not None:
        validate_quality(quality)
    with db.session() as session:
        sample = session.get(Sample, sample_id)
        if sample is None:
            return False
        sample.quality = quality
        session.commit()
        return True


def unset_quality(config: AppConfig, db: Database, sample_id: str) -> bool:
    """Clear the quality verdict for one sample (back to unset)."""
    return set_sample_quality(config, db, sample_id, None)


def set_batch_quality(
    config: AppConfig,
    db: Database,
    quality: str | None,
    *,
    cameras: list[str] | None = None,
    days: int | None = None,
    labels: list[str] | None = None,
    limit: int | None = None,
    clear: bool = False,
) -> int:
    """Set (or clear) quality across a filtered batch.

    Selection is deterministic: oldest samples first (stable review order).
    Returns the number of rows updated.
    """
    if not clear:
        validate_quality(str(quality))
        target_quality: str | None = quality
    else:
        target_quality = None

    with db.session() as session:
        query = session.query(Sample.id)
        if cameras:
            query = query.filter(Sample.camera.in_(cameras))
        if labels:
            with_label = (
                session.query(Annotation.sample_id)
                .filter(Annotation.label.in_(labels))
            )
            query = query.filter(Sample.id.in_(with_label))
        if days is not None:
            from_ts = datetime.now(timezone.utc).timestamp() - days * 86400.0
            query = query.filter(Sample.timestamp >= from_ts)
        query = query.order_by(Sample.timestamp.asc())
        if limit is not None:
            query = query.limit(limit)
        ids = [row[0] for row in query.all()]
        if not ids:
            return 0
        updated = (
            session.query(Sample)
            .filter(Sample.id.in_(ids))
            .update({"quality": target_quality}, synchronize_session=False)
        )
        session.commit()
        return updated


__all__ = [
    "QUALITY_VALUES",
    "validate_quality",
    "set_sample_quality",
    "unset_quality",
    "set_batch_quality",
]