"""Database migration tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from frigate_learn.db import Database, migration_files, resolve_migrations_dir


def test_init_creates_tables(db):
    names = db.table_names()
    for table in (
        "samples",
        "annotations",
        "jobs",
        "collection_failures",
        "discovery_windows",
        "deployments",
        "schema_migrations",
    ):
        assert table in names


def test_migrations_idempotent(db):
    db.migrate()
    first_version = db.schema_version()
    db.migrate()  # no-op
    assert db.schema_version() == first_version
    assert db.pending_migrations() == []


def test_applied_and_pending(db):
    applied = db.applied_migrations()
    assert len(applied) == 5
    assert applied == [
        "0001_initial.sql",
        "0002_multiframe_phash.sql",
        "0003_review_sync.sql",
        "0004_multi_object.sql",
        "0005_drop_review.sql",
    ]
    assert db.schema_version() == 5


def test_0004_adds_annotation_event_id(db):
    from sqlalchemy import text

    with db.engine.connect() as conn:
        cols = {
            row[1]
            for row in conn.execute(text("PRAGMA table_info(annotations)"))
        }
    assert "event_id" in cols


def test_0004_rejects_duplicate_sample_event(db):
    from sqlalchemy.exc import IntegrityError

    from frigate_learn.models import Annotation, Sample, utcnow

    with db.session() as s:
        s.add(Sample(id="a", camera="front", timestamp=1.0, event_id="evt1",
                     source="frigate", status="collected", created_at=utcnow()))
        s.commit()
        s.add(Annotation(id="a1", sample_id="a", source="frigate", label="person",
                         event_id="evt1", verified=0, created_at=utcnow()))
        s.commit()
        s.add(Annotation(id="a2", sample_id="a", source="frigate", label="car",
                         event_id="evt1", verified=0, created_at=utcnow()))
        with pytest.raises(IntegrityError):
            s.commit()

    with db.session() as s:
        s.add(Annotation(id="a3", sample_id="a", source="vlm", label="person",
                         event_id=None, verified=1, created_at=utcnow()))
        s.add(Annotation(id="a4", sample_id="a", source="vlm", label="car",
                         event_id=None, verified=1, created_at=utcnow()))
        s.commit()


def test_0005_drops_review_columns(db):
    from sqlalchemy import text

    with db.engine.connect() as conn:
        sample_cols = {
            row[1]
            for row in conn.execute(text("PRAGMA table_info(samples)"))
        }
        failure_cols = {
            row[1]
            for row in conn.execute(text("PRAGMA table_info(collection_failures)"))
        }
    for column in ("frigate_reviewed", "reviewed_at", "review_id"):
        assert column not in sample_cols
    assert "review_id" not in failure_cols


def test_0002_adds_sampling_columns(db):
    from sqlalchemy import text

    with db.engine.connect() as conn:
        cols = {
            row[1]
            for row in conn.execute(text("PRAGMA table_info(samples)"))
        }
    for column in ("frame_index", "perceptual_hash", "verified"):
        assert column in cols


def test_fresh_database_has_no_pending(tmp_path):
    db = Database(tmp_path / "fresh.db")
    assert len(db.pending_migrations()) >= 1
    db.migrate()
    assert db.pending_migrations() == []
    assert db.schema_version() >= 1
    db.dispose()


def test_unique_event_index_enforced(db):
    from sqlalchemy.exc import IntegrityError

    from frigate_learn.models import Sample, utcnow

    with db.session() as s:
        s.add(Sample(id="a", camera="front", timestamp=1.0, event_id="evt1",
                     source="frigate", status="collected", created_at=utcnow()))
        s.add(Sample(id="b", camera="front", timestamp=2.0, event_id="evt1",
                     source="frigate", status="collected", created_at=utcnow()))
        with pytest.raises(IntegrityError):
            s.commit()


def test_foreign_key_enforced(db):
    from sqlalchemy.exc import IntegrityError

    from frigate_learn.models import Annotation, utcnow

    with db.session() as s:
        s.add(Annotation(id="x", sample_id="does-not-exist", source="frigate",
                         label="person", verified=0, created_at=utcnow()))
        with pytest.raises(IntegrityError):
            s.commit()


def test_resolve_migrations_bundled():
    mdir = resolve_migrations_dir(None)
    assert mdir.joinpath("0001_initial.sql").exists() or migration_files(mdir)


def test_migration_files_sorted(tmp_path):
    (tmp_path / "0002_b.sql").write_text("--", encoding="utf-8")
    (tmp_path / "0001_a.sql").write_text("--", encoding="utf-8")
    (tmp_path / "README.md").write_text("--", encoding="utf-8")
    files = migration_files(tmp_path)
    names = [f.name for f in files]
    assert names == ["0001_a.sql", "0002_b.sql"]