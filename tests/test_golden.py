"""Golden dataset tests."""

from __future__ import annotations

from datetime import datetime

import pytest

from frigate_learn.dataset.manifest import ManifestDuplicateError
from frigate_learn.dataset.yolo import YoloLine, read_yolo_label
from frigate_learn.evaluation.golden import (
    GoldenDataset,
    read_dataset_yaml,
    time_bucket_for_timestamp,
    write_dataset_yaml,
)


@pytest.fixture
def source_image(tmp_path):
    img = tmp_path / "src.jpg"
    img.write_bytes(b"\xff\xd8\xff\xe0fakejpeg")
    return img


def test_create_layout_and_yaml(tmp_path):
    ds = GoldenDataset.create(tmp_path / "golden", ["person", "car"])
    assert (tmp_path / "golden" / "images").is_dir()
    assert (tmp_path / "golden" / "labels").is_dir()
    assert read_dataset_yaml(tmp_path / "golden" / "dataset.yaml") == ["person", "car"]
    assert ds.classes == ["person", "car"]
    assert ds.validate() == []


def test_load_roundtrip(tmp_path, source_image):
    root = tmp_path / "golden"
    ds = GoldenDataset.create(root, ["person"])
    ds.add_image(
        "s1",
        source_image,
        [YoloLine(0, 0.5, 0.5, 0.2, 0.4)],
        camera="front",
        timestamp=datetime(2026, 1, 1, 12, 0).timestamp(),
        source="camera",
    )
    loaded = GoldenDataset.load(root)
    assert loaded.classes == ["person"]
    assert len(loaded.manifest) == 1
    assert loaded.manifest[0]["sample_id"] == "s1"
    assert loaded.manifest[0]["time_bucket"] == "day"
    assert loaded.samples()[0].image_path.is_file()
    assert read_yolo_label(loaded.samples()[0].label_path) == [YoloLine(0, 0.5, 0.5, 0.2, 0.4)]


def test_add_image_duplicate_rejected(tmp_path, source_image):
    ds = GoldenDataset.create(tmp_path / "golden", ["person"])
    ds.add_image("s1", source_image, [], camera="front", timestamp=1.0)
    with pytest.raises(ManifestDuplicateError):
        ds.add_image("s1", source_image, [], camera="front", timestamp=1.0)
    assert len(ds.manifest) == 1


def test_add_image_overwrite_ok(tmp_path, source_image):
    ds = GoldenDataset.create(tmp_path / "golden", ["person"])
    ds.add_image("s1", source_image, [], camera="front", timestamp=1.0)
    ds.add_image(
        "s1",
        source_image,
        [YoloLine(0, 0.2, 0.2, 0.1, 0.1)],
        camera="front",
        timestamp=1.0,
        overwrite_ok=True,
    )
    labels = read_yolo_label(ds.labels_dir() / "s1.txt")
    assert labels[0].cx == pytest.approx(0.2)


def test_unsupported_extension(tmp_path, source_image):
    ds = GoldenDataset.create(tmp_path / "golden", ["person"])
    bad = tmp_path / "bad.heic"
    bad.write_bytes(b"x")
    with pytest.raises(ValueError):
        ds.add_image("s1", bad, [], camera="front", timestamp=1.0)


def test_validate_detects_missing_files(tmp_path, source_image):
    ds = GoldenDataset.create(tmp_path / "golden", ["person"])
    ds.add_image("s1", source_image, [], camera="front", timestamp=1.0)
    ds.samples()[0].image_path.unlink()
    problems = ds.validate()
    assert any("missing image" in p for p in problems)


def test_multi_object_image_roundtrips_all_boxes(tmp_path, source_image):
    ds = GoldenDataset.create(tmp_path / "golden", ["person", "car"])
    labels = [
        YoloLine(0, 0.5, 0.5, 0.2, 0.4),
        YoloLine(1, 0.2, 0.7, 0.3, 0.3),
    ]
    ds.add_image(
        "s1",
        source_image,
        labels,
        camera="front",
        timestamp=1.0,
    )
    loaded = GoldenDataset.load(tmp_path / "golden")
    assert loaded.validate() == []
    got = read_yolo_label(loaded.samples()[0].label_path)
    assert got == labels


def test_write_read_dataset_yaml(tmp_path):
    path = tmp_path / "ds.yaml"
    write_dataset_yaml(path, ["dog", "cat"])
    assert read_dataset_yaml(path) == ["dog", "cat"]


@pytest.mark.parametrize(
    ("hour", "bucket"),
    [(2, "night"), (7, "dawn"), (12, "day"), (19, "dusk"), (23, "night"), (4, "night")],
)
def test_time_bucket(hour, bucket):
    ts = datetime(2026, 1, 1, hour, 30).timestamp()
    assert time_bucket_for_timestamp(ts) == bucket