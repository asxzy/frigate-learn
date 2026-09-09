"""External COCO import tests (Phase 10)."""

from __future__ import annotations

import json

from PIL import Image

from frigate_learn.collection.external import import_coco
from frigate_learn.models import Annotation, Sample


def _make_image(path, size=32, ext=".jpg"):
    Image.new("RGB", (size, size), (90, 120, 60)).save(path)
    return path


def _coco(images, annotations, categories):
    return {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def test_import_coco_happy_path(config, db, tmp_path):
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    _make_image(img_dir / "a.jpg")
    _make_image(img_dir / "b.jpg")

    body = _coco(
        images=[
            {"id": 1, "file_name": "a.jpg", "width": 32, "height": 32},
            {"id": 2, "file_name": "b.jpg", "width": 32, "height": 32},
        ],
        annotations=[
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [2, 2, 8, 8]},
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [16, 16, 4, 4]},
        ],
        categories=[{"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}],
    )
    ann_file = tmp_path / "instances.json"
    ann_file.write_text(json.dumps(body), encoding="utf-8")

    summary = import_coco(config, db, ann_file, img_dir, camera="external")
    assert summary.images == 1  # only image 1 has mapped annotations
    assert summary.objects == 2
    assert summary.samples_written == 1

    with db.session() as s:
        sample = s.query(Sample).one()
        anns = s.query(Annotation).filter(Annotation.sample_id == sample.id).all()
    assert sample.source == "external"
    assert sample.camera == "external"
    assert sample.frigate_label == "cat"
    assert sample.verified == 0
    assert {a.label for a in anns} == {"cat", "dog"}
    assert all(a.verified == 1 for a in anns)  # curated external labels trusted
    # bbox normalized to [0,1]
    bounds = next(a for a in anns if a.label == "cat")
    assert bounds.x1 == 2 / 32 and bounds.x2 == 10 / 32


def test_import_coco_unmapped_and_missing_objects_skipped(config, db, tmp_path):
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    _make_image(img_dir / "a.jpg")

    body = _coco(
        images=[{"id": 1, "file_name": "a.jpg", "width": 32, "height": 32}],
        annotations=[
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 4, 4]},
            {"id": 2, "image_id": 1, "category_id": 99, "bbox": [0, 0, 4, 4]},  # unknown category
        ],
        categories=[{"id": 1, "name": "cat"}],
    )
    ann_file = tmp_path / "instances.json"
    ann_file.write_text(json.dumps(body), encoding="utf-8")

    summary = import_coco(
        config, db, ann_file, img_dir, camera="ext",
        class_map={"cat": "person", "dog": "car"},
    )
    # 'cat' -> 'person' mapped; category 99 drops; object ratio kept
    assert summary.objects == 1
    assert summary.samples_written == 1
    with db.session() as s:
        ann = s.query(Annotation).one()
    assert ann.label == "person"


def test_import_coco_missing_image_skipped(config, db, tmp_path):
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    body = _coco(
        images=[{"id": 1, "file_name": "ghost.jpg", "width": 32, "height": 32}],
        annotations=[{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 4, 4]}],
        categories=[{"id": 1, "name": "cat"}],
    )
    ann_file = tmp_path / "instances.json"
    ann_file.write_text(json.dumps(body), encoding="utf-8")
    summary = import_coco(config, db, ann_file, img_dir, camera="ext")
    assert summary.skipped_missing == 1
    assert summary.samples_written == 0