"""Dataset adapter tests: manifest parsing and DB (SQLite) loading.

Both adapters must yield one FrigateObject per crop with the bbox converted
to crop-local absolute xyxy pixels, without ever guessing conventions.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from frigate_learn.audit.adapter import (
    AdapterError,
    ManifestDatasetAdapter,
)


def _make_image(path: Path, size=(320, 240), color=(90, 120, 200)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def _manifest(root: Path, objects: list[dict], **top) -> Path:
    ann = root / "annotations"
    ann.mkdir(parents=True, exist_ok=True)
    data = {"image_root": "images", "objects": objects}
    data.update(top)
    path = ann / "manifest.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_manifest_xyxy_pixels(tmp_path):
    _make_image(tmp_path / "images" / "a.jpg")
    _make_image(tmp_path / "images" / "b.jpg")
    _manifest(tmp_path, [
        {"id": "a", "image": "a.jpg", "class_name": "person", "bbox": [10, 20, 110, 120]},
        {"id": "b", "image": "b.jpg", "class_name": "car", "bbox": [30, 40, 90, 80]},
    ])
    adapter = ManifestDatasetAdapter(tmp_path)
    objs = list(adapter.iter_objects())
    assert [o.sample_id for o in objs] == ["a", "b"]
    assert [o.class_name for o in objs] == ["person", "car"]
    assert adapter.ontology() == ["person", "car"]
    assert adapter.class_id_of("person") == 0
    assert adapter.class_id_of("car") == 1
    assert adapter.class_id_of("bicycle") is None
    a = objs[0]
    assert a.bbox.to_list() == [10.0, 20.0, 110.0, 120.0]
    assert a.bbox_format == "xyxy_px"
    assert a.class_id == 0
    assert a.extra["manifest_index"] == 0


def test_manifest_xywh_conversion_and_clamp(tmp_path):
    _make_image(tmp_path / "images" / "a.jpg", size=(100, 100))
    _manifest(tmp_path, [
        {"id": "a", "image": "a.jpg", "class_name": "person",
         "bbox": [10, 10, 40, 40], "bbox_format": "xywh"},
        {"id": "b", "image": "a.jpg", "class_name": "person",
         "bbox": [50, 50, 200, 200], "bbox_format": "xywh"},
    ])
    adapter = ManifestDatasetAdapter(tmp_path)
    objs = list(adapter.iter_objects())
    assert objs[0].bbox.to_list() == [10.0, 10.0, 50.0, 50.0]
    assert objs[1].bbox.to_list() == [50.0, 50.0, 100.0, 100.0]


def test_manifest_normalized_default(tmp_path):
    _make_image(tmp_path / "images" / "a.jpg", size=(200, 100))
    _manifest(tmp_path, [
        {"id": "a", "image": "a.jpg", "class_name": "person",
         "bbox": [0.25, 0.5, 0.75, 1.0]},
    ], normalized=True)
    adapter = ManifestDatasetAdapter(tmp_path)
    a = next(iter(adapter.iter_objects()))
    assert a.bbox.to_list() == [50.0, 50.0, 150.0, 100.0]


def test_manifest_normalized_default_false(tmp_path):
    _make_image(tmp_path / "images" / "a.jpg", size=(200, 100))
    _manifest(tmp_path, [
        {"id": "a", "image": "a.jpg", "class_name": "person",
         "bbox": [20, 30, 80, 60]},
    ])
    adapter = ManifestDatasetAdapter(tmp_path)
    a = next(iter(adapter.iter_objects()))
    assert a.bbox.to_list() == [20.0, 30.0, 80.0, 60.0]


def test_manifest_frame_coordinates_with_origin(tmp_path):
    _make_image(tmp_path / "images" / "a.jpg", size=(100, 80))
    _manifest(tmp_path, [
        {"id": "a", "image": "a.jpg", "class_name": "person",
         "bbox": [30, 40, 70, 80], "coordinate_space": "frame",
         "crop_origin": [20, 30]},
    ])
    adapter = ManifestDatasetAdapter(tmp_path)
    a = next(iter(adapter.iter_objects()))
    assert a.bbox.to_list() == [10.0, 10.0, 50.0, 50.0]


def test_manifest_frame_coordinates_requires_origin(tmp_path):
    _make_image(tmp_path / "images" / "a.jpg", size=(100, 80))
    _manifest(tmp_path, [
        {"id": "a", "image": "a.jpg", "class_name": "person",
         "bbox": [30, 40, 70, 80], "coordinate_space": "frame"},
    ])
    adapter = ManifestDatasetAdapter(tmp_path)
    with pytest.raises(AdapterError, match="crop_origin"):
        list(adapter.iter_objects())


def test_manifest_malformed_bbox(tmp_path):
    _make_image(tmp_path / "images" / "a.jpg")
    _manifest(tmp_path, [
        {"id": "a", "image": "a.jpg", "class_name": "person", "bbox": [1, 2, 3]},
    ])
    adapter = ManifestDatasetAdapter(tmp_path)
    with pytest.raises(AdapterError, match="4-number"):
        list(adapter.iter_objects())


def test_manifest_duplicate_id_rejected(tmp_path):
    _make_image(tmp_path / "images" / "a.jpg")
    _manifest(tmp_path, [
        {"id": "a", "image": "a.jpg", "class_name": "person", "bbox": [1, 2, 3, 4]},
        {"id": "a", "image": "a.jpg", "class_name": "car", "bbox": [1, 2, 3, 4]},
    ])
    adapter = ManifestDatasetAdapter(tmp_path)
    with pytest.raises(AdapterError, match="duplicate"):
        list(adapter.iter_objects())


def test_manifest_missing_manifest_file(tmp_path):
    with pytest.raises(AdapterError, match="no manifest"):
        ManifestDatasetAdapter(tmp_path)


def test_manifest_class_map_orders_ontology(tmp_path):
    _make_image(tmp_path / "images" / "a.jpg")
    _manifest(tmp_path, [
        {"id": "a", "image": "a.jpg", "class_name": "dog", "bbox": [1, 2, 3, 4]},
        {"id": "b", "image": "a.jpg", "class_name": "cat", "bbox": [1, 2, 3, 4]},
    ], class_map={"dog": 0, "cat": 1})
    adapter = ManifestDatasetAdapter(tmp_path)
    assert adapter.ontology() == ["dog", "cat"]
    assert next(iter(adapter.iter_objects())).class_id == 0

