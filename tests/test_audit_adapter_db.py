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

SCHEMA_ANNOTATIONS = SCHEMA.replace(
    "    image_path TEXT,\n",
    "    image_path TEXT,\n    review_id TEXT,\n",
) + ";\n" + """
CREATE TABLE annotations (
    id TEXT PRIMARY KEY,
    sample_id TEXT,
    source TEXT,
    label TEXT,
    x1 REAL, y1 REAL, x2 REAL, y2 REAL,
    confidence REAL,
    verified INTEGER,
    event_id TEXT
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


def _db_with_annotations(tmp_path: Path, rows: list[dict], annotations: list[dict]) -> Path:
    path = tmp_path / "annotated.db"
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA_ANNOTATIONS)
    for row in rows:
        cols = list(row)
        conn.execute(
            "INSERT INTO samples (" + ", ".join(cols) + ") VALUES ("
            + ", ".join("?" for _ in cols) + ")",
            [row[c] for c in cols],
        )
    for ann in annotations:
        cols = list(ann)
        conn.execute(
            "INSERT INTO annotations (" + ", ".join(cols) + ") VALUES ("
            + ", ".join("?" for _ in cols) + ")",
            [ann[c] for c in cols],
        )
    conn.commit()
    conn.close()
    return path


def test_db_adapter_iterates_annotations_not_primary_boxes(tmp_path):
    images = tmp_path / "images"
    _make_image(images / "20240101" / "cam1" / "s1.jpg")
    db = _db_with_annotations(tmp_path, [
        {"id": "s1", "camera": "cam1", "timestamp": 1.0, "event_id": "e1",
         "image_path": "images/20240101/cam1/s1.jpg",
         "frigate_label": "person", "frigate_score": 0.9,
         "frigate_x1": 0.1, "frigate_y1": 0.1, "frigate_x2": 0.2, "frigate_y2": 0.2,
         "status": "collected", "quality": "useful"},
    ], [
        {"id": "ann1", "sample_id": "s1", "source": "frigate", "label": "person",
         "x1": 0.1, "y1": 0.2, "x2": 0.4, "y2": 0.6, "confidence": 0.8,
         "verified": 0, "event_id": "e1"},
        {"id": "ann2", "sample_id": "s1", "source": "frigate", "label": "car",
         "x1": 0.5, "y1": 0.1, "x2": 0.9, "y2": 0.7, "confidence": 0.7,
         "verified": 0, "event_id": "e2"},
        {"id": "ann3", "sample_id": "s1", "source": "vlm", "label": "bird",
         "x1": 0.0, "y1": 0.0, "x2": 0.5, "y2": 0.5, "confidence": 0.9,
         "verified": 1, "event_id": None},
    ])
    adapter = FrigateDatabaseAdapter(db, images_root=tmp_path)
    objs = list(adapter.iter_objects())
    assert [o.sample_id for o in objs] == ["ann1", "ann2"]  # vlm annotation excluded
    assert objs[0].image_path == objs[1].image_path
    assert objs[0].class_name == "person"
    assert objs[1].class_name == "car"
    assert objs[0].bbox.to_list() == [20.0, 20.0, 80.0, 60.0]   # 200x100 image
    assert objs[1].bbox.to_list() == [100.0, 10.0, 180.0, 70.0]
    assert objs[0].extra["sample_id"] == "s1"      # original sample preserved
    assert objs[0].extra["event_id"] == "e1"
    assert objs[1].extra["event_id"] == "e2"
    assert objs[0].extra["annotation_id"] == "ann1"
    assert objs[0].class_id == 0
    assert objs[1].class_id == 1
    assert adapter.ontology() == ["person", "car"]  # derived from annotations


def test_db_adapter_source_filter(tmp_path):
    images = tmp_path / "images"
    _make_image(images / "20240101" / "cam1" / "s1.jpg")
    db = _db_with_annotations(tmp_path, [
        {"id": "s1", "camera": "cam1", "timestamp": 1.0, "event_id": "e1",
         "image_path": "images/20240101/cam1/s1.jpg",
         "frigate_label": "person", "frigate_score": 0.9,
         "frigate_x1": 0.1, "frigate_y1": 0.1, "frigate_x2": 0.2, "frigate_y2": 0.2,
         "status": "collected", "quality": "useful"},
    ], [
        {"id": "ann1", "sample_id": "s1", "source": "frigate", "label": "person",
         "x1": 0.1, "y1": 0.1, "x2": 0.5, "y2": 0.5, "confidence": 0.8,
         "verified": 1, "event_id": "e1"},
        {"id": "ann2", "sample_id": "s1", "source": "vlm", "label": "bird",
         "x1": 0.0, "y1": 0.0, "x2": 0.5, "y2": 0.5, "confidence": 0.9,
         "verified": 1, "event_id": None},
    ])
    frigate_only = list(FrigateDatabaseAdapter(db, images_root=tmp_path).iter_objects())
    vlm_only = list(FrigateDatabaseAdapter(db, images_root=tmp_path, source="vlm").iter_objects())
    assert [o.sample_id for o in frigate_only] == ["ann1"]
    assert [o.sample_id for o in vlm_only] == ["ann2"]

