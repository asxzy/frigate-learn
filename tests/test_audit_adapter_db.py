"""FrigateDatabaseAdapter tests against a real SQLite samples table.

Uses the conventions documented in the adapter: stored normalized xyxy boxes
relative to the (default) full frame; images under ``<root>/images/...``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from PIL import Image

from frigate_learn.audit.adapter import AdapterError, FrigateDatabaseAdapter

SCHEMA = """
CREATE TABLE samples (
    id TEXT PRIMARY KEY,
    camera TEXT,
    timestamp TEXT,
    event_id TEXT,
    image_path TEXT,
    frigate_label TEXT,
    frigate_score REAL,
    frigate_x1 REAL,
    frigate_y1 REAL,
    frigate_x2 REAL,
    frigate_y2 REAL,
    status TEXT,
    quality TEXT
)
"""


def _make_image(path: Path, size=(200, 100), color=(80, 140, 220)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def _db(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "frigate.db"
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    for row in rows:
        cols = list(row)
        col_sql = ", ".join(cols)
        ph_sql = ", ".join("?" for _ in cols)
        conn.execute(
            "INSERT INTO samples (" + col_sql + ") VALUES (" + ph_sql + ")",
            [row[c] for c in cols],
        )
    conn.commit()
    conn.close()
    return path


def test_db_adapter_reads_normalized_boxes(tmp_path):
    images = tmp_path / "images"
    _make_image(images / "20240101" / "cam1" / "s1.jpg")
    _make_image(images / "20240101" / "cam1" / "s2.jpg")
    db = _db(tmp_path, [
        {"id": "s1", "camera": "cam1", "image_path": "images/20240101/cam1/s1.jpg",
         "frigate_label": "person", "frigate_score": 0.8,
         "frigate_x1": 0.1, "frigate_y1": 0.2, "frigate_x2": 0.4, "frigate_y2": 0.6},
        {"id": "s2", "camera": "cam1", "image_path": "images/20240101/cam1/s2.jpg",
         "frigate_label": "car", "frigate_score": 0.9,
         "frigate_x1": 0.5, "frigate_y1": 0.0, "frigate_x2": 1.0, "frigate_y2": 1.0},
    ])
    adapter = FrigateDatabaseAdapter(db, images_root=tmp_path)
    objs = list(adapter.iter_objects())
    assert [o.sample_id for o in objs] == ["s1", "s2"]
    s1 = objs[0]
    # 200x100 image: (0.1*200, 0.2*100, 0.4*200, 0.6*100) = (20, 20, 80, 60)
    assert s1.bbox.to_list() == [20.0, 20.0, 80.0, 60.0]
    assert s1.bbox_format == "xyxy_px"
    assert s1.class_name == "person"
    assert s1.class_id == 0
    assert s1.extra["camera"] == "cam1"
    assert s1.extra["frigate_score"] == 0.8
    assert s1.extra["source_box"] == [0.1, 0.2, 0.4, 0.6]
    s2 = objs[1]
    assert s2.bbox.to_list() == [100.0, 0.0, 200.0, 100.0]
    assert s2.class_id == 1
    assert adapter.ontology() == ["person", "car"]
    assert adapter.class_id_of("car") == 1


def test_db_adapter_skip_reasons(tmp_path):
    images = tmp_path / "images"
    _make_image(images / "20240101" / "cam1" / "a.jpg")
    db = _db(tmp_path, [
        {"id": "a", "image_path": "images/20240101/cam1/a.jpg",
         "frigate_label": "person", "frigate_x1": 0.1, "frigate_y1": 0.1,
         "frigate_x2": 0.3, "frigate_y2": 0.3},
        # degenerate box (x2 < x1)
        {"id": "bad", "image_path": "images/20240101/cam1/a.jpg",
         "frigate_label": "person", "frigate_x1": 0.5, "frigate_y1": 0.5,
         "frigate_x2": 0.2, "frigate_y2": 0.6},
        # missing image
        {"id": "gone", "image_path": "images/20240101/cam1/missing.jpg",
         "frigate_label": "car", "frigate_x1": 0.1, "frigate_y1": 0.1,
         "frigate_x2": 0.3, "frigate_y2": 0.3},
    ])
    adapter = FrigateDatabaseAdapter(db, images_root=tmp_path)
    objs = list(adapter.iter_objects())
    assert [o.sample_id for o in objs] == ["a"]
    assert adapter.skip_reasons.get("degenerate_box") == 1
    assert adapter.skip_reasons.get("missing_image") == 1


def test_db_adapter_corrupt_image_skipped(tmp_path):
    images = tmp_path / "images"
    good = images / "20240101" / "cam1" / "a.jpg"
    _make_image(good)
    bad = images / "20240101" / "cam1" / "bad.jpg"
    bad.write_bytes(good.read_bytes()[:40])
    db = _db(tmp_path, [
        {"id": "a", "image_path": "images/20240101/cam1/a.jpg",
         "frigate_label": "person", "frigate_x1": 0.1, "frigate_y1": 0.1,
         "frigate_x2": 0.3, "frigate_y2": 0.3},
        {"id": "bad", "image_path": "images/20240101/cam1/bad.jpg",
         "frigate_label": "car", "frigate_x1": 0.1, "frigate_y1": 0.1,
         "frigate_x2": 0.3, "frigate_y2": 0.3},
    ])
    adapter = FrigateDatabaseAdapter(db, images_root=tmp_path)
    objs = list(adapter.iter_objects())
    assert [o.sample_id for o in objs] == ["a"]
    assert adapter.skip_reasons.get("image_open_failed") == 1


def test_db_adapter_status_filter(tmp_path):
    images = tmp_path / "images"
    _make_image(images / "20240101" / "cam1" / "a.jpg")
    db = _db(tmp_path, [
        {"id": "ok", "image_path": "images/20240101/cam1/a.jpg",
         "frigate_label": "person", "status": "collected",
         "frigate_x1": 0.0, "frigate_y1": 0.0, "frigate_x2": 0.5, "frigate_y2": 0.5},
        {"id": "nope", "image_path": "images/20240101/cam1/a.jpg",
         "frigate_label": "person", "status": "reviewed",
         "frigate_x1": 0.0, "frigate_y1": 0.0, "frigate_x2": 0.5, "frigate_y2": 0.5},
    ])
    adapter = FrigateDatabaseAdapter(db, images_root=tmp_path, statuses=["collected"])
    objs = list(adapter.iter_objects())
    assert [o.sample_id for o in objs] == ["ok"]


def test_db_adapter_missing_db_file(tmp_path):
    with pytest.raises(AdapterError, match="database not found"):
        FrigateDatabaseAdapter(tmp_path / "nope.db")


def test_db_adapter_empty_label_skipped(tmp_path):
    images = tmp_path / "images"
    _make_image(images / "20240101" / "cam1" / "a.jpg")
    db = _db(tmp_path, [
        {"id": "a", "image_path": "images/20240101/cam1/a.jpg",
         "frigate_label": None, "frigate_x1": 0.0, "frigate_y1": 0.0,
         "frigate_x2": 0.5, "frigate_y2": 0.5},
    ])
    adapter = FrigateDatabaseAdapter(db, images_root=tmp_path)
    assert list(adapter.iter_objects()) == []
    assert adapter.ontology() == []

