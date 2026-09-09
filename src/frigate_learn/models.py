"""SQLAlchemy ORM models.

Mapped to the migrations (``0001_initial.sql``). Prefer creating/migrating via
``db.init()``; these models are used only for reads/writes against the engine.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Float, ForeignKey, Integer, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class Base(DeclarativeBase):
    pass


class Sample(Base):
    __tablename__ = "samples"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    camera: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    timestamp: Mapped[float] = mapped_column(Float, nullable=False, index=True)
    event_id: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    frame_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    review_id: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    image_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    debug_image_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    image_hash: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    perceptual_hash: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    source: Mapped[str] = mapped_column(Text, nullable=False, default="frigate")
    verified: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    frigate_label: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)
    frigate_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    frigate_x1: Mapped[float | None] = mapped_column(Float, nullable=True)
    frigate_y1: Mapped[float | None] = mapped_column(Float, nullable=True)
    frigate_x2: Mapped[float | None] = mapped_column(Float, nullable=True)
    frigate_y2: Mapped[float | None] = mapped_column(Float, nullable=True)
    quality: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="collected")
    created_at: Mapped[str] = mapped_column(Text, nullable=False, default=utcnow)


class Annotation(Base):
    __tablename__ = "annotations"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    sample_id: Mapped[str] = mapped_column(
        ForeignKey("samples.id"), nullable=False, index=True
    )
    source: Mapped[str] = mapped_column(Text, nullable=False)
    label: Mapped[str] = mapped_column(Text, nullable=False)
    x1: Mapped[float | None] = mapped_column(Float, nullable=True)
    y1: Mapped[float | None] = mapped_column(Float, nullable=True)
    x2: Mapped[float | None] = mapped_column(Float, nullable=True)
    y2: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    verified: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[str] = mapped_column(Text, nullable=False, default=utcnow)


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    type: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    status: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    started_at: Mapped[str | None] = mapped_column(Text, nullable=True)
    finished_at: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)


class CollectionFailure(Base):
    __tablename__ = "collection_failures"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    event_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    review_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    camera: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[str] = mapped_column(Text, nullable=False, default=utcnow)


class DiscoveryWindow(Base):
    __tablename__ = "discovery_windows"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    camera: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    start_time: Mapped[float] = mapped_column(Float, nullable=False)
    end_time: Mapped[float] = mapped_column(Float, nullable=False)
    motion_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[str] = mapped_column(Text, nullable=False, default=utcnow)


class Deployment(Base):
    __tablename__ = "deployments"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    model_name: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    version: Mapped[str] = mapped_column(Text, nullable=False)
    artifact_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    artifact_hash: Mapped[str | None] = mapped_column(Text, nullable=True)
    metrics_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    verdict: Mapped[str | None] = mapped_column(Text, nullable=True)
    reasons_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    deployed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    deployed_at: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[str] = mapped_column(Text, nullable=False, default=utcnow)


__all__ = [
    "Base",
    "Sample",
    "Annotation",
    "Job",
    "CollectionFailure",
    "DiscoveryWindow",
    "Deployment",
    "utcnow",
]