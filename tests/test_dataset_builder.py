"""Dataset builder: verified-only default, split-directory layout, label space."""

from __future__ import annotations

import json

import yaml
from PIL import Image

from frigate_learn.dataset.builder import DatasetBuilder
from frigate_learn.dataset.manifest import iter_manifest
from frigate_learn.models import Annotation, Sample, utcnow


def _seed_annotations(config, db, sample_id, labels):
    with db.session() as s:
        for i, label in enumerate(labels):
            s.add(
                Annotation(
                    id=f"a{sample_id}{i}", sample_id=sample_id,
                    source="vlm", label=label, x1=0.1, y1=0.1, x2=0.6, y2=0.8,
                    confidence=0.9, verified=1, created_at=utcnow(),
                )
            )
        s.commit()


def _seed_bare_sample(config, db, sample_id="s2"):
    img_dir = config.images_dir() / "20260901" / "front"
    img_dir.mkdir(parents=True, exist_ok=True)
    img = img_dir / f"{sample_id}.jpg"
    Image.new("RGB", (320, 320), (40, 40, 40)).save(img)
    with db.session() as s:
        s.add(
            Sample(
                id=sample_id, camera="front", timestamp=200.0, event_id="e2",
                frame_index=0, image_path=str(img), status="collected",
                frigate_label="person", created_at=utcnow(),
            )
        )
        s.commit()


def _seed_sample(config, db, sample_id="s1", verified=True):
    img_dir = config.images_dir() / "20260901" / "front"
    img_dir.mkdir(parents=True, exist_ok=True)
    img = img_dir / f"{sample_id}.jpg"
    Image.new("RGB", (320, 320), (30, 30, 30)).save(img)
    with db.session() as s:
        s.add(
            Sample(
                id=sample_id, camera="front", timestamp=100.0, event_id="e1",
                frame_index=0, image_path=str(img), status="collected",
                frigate_label="person", created_at=utcnow(),
            )
        )
        s.flush()
        s.add(
            Annotation(
                id=f"a{sample_id}", sample_id=sample_id,
                source="vlm" if verified else "frigate",
                label="person", x1=0.1, y1=0.1, x2=0.6, y2=0.8,
                confidence=0.9, verified=int(verified), created_at=utcnow(),
            )
        )
        s.commit()


def test_builder_default_is_verified_only(config, db, tmp_path):
    _seed_sample(config, db, verified=False)
    summary = DatasetBuilder(config, db).build("v001")
    assert summary.images_written == 0
    assert summary.skipped_no_box == 1


def test_builder_include_unverified_opt_in(config, db, tmp_path):
    _seed_sample(config, db, verified=False)
    summary = DatasetBuilder(config, db).build("v001", verified_only=False)
    assert summary.images_written == 1


def test_builder_writes_split_directories(config, db, tmp_path):
    _seed_sample(config, db)
    summary = DatasetBuilder(config, db).build("v001")
    assert summary.images_written == 1
    target = config.datasets_dir() / "v001"
    records = list(iter_manifest(target / "manifest.jsonl"))
    assert len(records) == 1
    split = records[0]["split"]
    assert (target / "images" / split / "s1.jpg").is_file()
    assert (target / "labels" / split / "s1.txt").is_file()
    assert not (target / "images" / "s1.jpg").exists()
    yaml_text = (target / "dataset.yaml").read_text(encoding="utf-8")
    assert "train: images/train" in yaml_text
    assert "val: images/val" in yaml_text
    assert "test: images/test" in yaml_text
    assert f"images/{split}/s1.jpg" in (target / f"{split}.txt").read_text(encoding="utf-8")


def test_builder_overwrite_recreates_version(config, db, tmp_path):
    _seed_sample(config, db)
    target = config.datasets_dir() / "v001"
    DatasetBuilder(config, db).build("v001")
    stale = target / "stale.txt"
    stale.write_text("x", encoding="utf-8")
    summary = DatasetBuilder(config, db).build("v001", overwrite=True)
    assert summary.images_written == 1
    records = list(iter_manifest(target / "manifest.jsonl"))
    assert len(records) == 1
    assert not stale.exists()


def test_builder_coco80_label_ids_and_yaml(config, db, tmp_path):
    _seed_sample(config, db)
    _seed_annotations(config, db, "s1", ["cat"])
    DatasetBuilder(config, db).build("v001")
    target = config.datasets_dir() / "v001"
    records = list(iter_manifest(target / "manifest.jsonl"))
    split = records[0]["split"]
    label_path = target / "labels" / split / "s1.txt"
    rows = label_path.read_text(encoding="utf-8").splitlines()
    assert {line.split()[0] for line in rows} == {"0", "15"}
    payload = yaml.safe_load((target / "dataset.yaml").read_text(encoding="utf-8"))
    names = {str(k): v for k, v in payload["names"].items()}
    assert len(names) == 80
    assert names["15"] == "cat"
    assert names["16"] == "dog"
    build = json.loads((target / "build.json").read_text(encoding="utf-8"))
    assert build["label_space"] == "coco80"
    assert len(build["trainable"]) == 80
    assert build["trainable"][0] is True
    assert build["trainable"][15] is True


def test_builder_subset_still_compact(config, db, tmp_path):
    config.training.label_space = "subset"
    _seed_sample(config, db)
    DatasetBuilder(config, db).build("v001")
    target = config.datasets_dir() / "v001"
    records = list(iter_manifest(target / "manifest.jsonl"))
    split = records[0]["split"]
    label_path = target / "labels" / split / "s1.txt"
    rows = label_path.read_text(encoding="utf-8").splitlines()
    assert rows[0].split()[0] == "0"
    payload = yaml.safe_load((target / "dataset.yaml").read_text(encoding="utf-8"))
    names = {str(k): v for k, v in payload["names"].items()}
    assert len(names) == len(config.classes)
    assert names["0"] == "person"
    build = json.loads((target / "build.json").read_text(encoding="utf-8"))
    assert build.get("label_space") == "subset"
    assert "trainable" not in build


def test_builder_filters_non_coco_and_non_configured_labels(config, db, tmp_path):
    _seed_sample(config, db)
    _seed_bare_sample(config, db)
    _seed_annotations(config, db, "s2", ["bird", "bike"])
    summary = DatasetBuilder(config, db).build("v001")
    assert summary.skipped_class == 1
    target = config.datasets_dir() / "v001"
    records = list(iter_manifest(target / "manifest.jsonl"))
    split = records[0]["split"]
    label_path = target / "labels" / split / "s1.txt"
    rows = label_path.read_text(encoding="utf-8").splitlines()
    assert {line.split()[0] for line in rows} == {"0"}


def _seed_multi_box(config, db, sample_id="s1", verified=True, boxes=None, event_ids=None):
    img_dir = config.images_dir() / "20260901" / "front"
    img_dir.mkdir(parents=True, exist_ok=True)
    img = img_dir / f"{sample_id}.jpg"
    Image.new("RGB", (320, 320), (50, 50, 50)).save(img)
    boxes = boxes or [
        ("person", 0.1, 0.1, 0.5, 0.8),
        ("car", 0.5, 0.2, 0.9, 0.6),
    ]
    event_ids = event_ids or [f"e{sample_id}-{i}" for i in range(len(boxes))]
    with db.session() as s:
        s.add(
            Sample(
                id=sample_id, camera="front", timestamp=sample_id_hash(sample_id),
                event_id=f"e{sample_id}", frame_index=0, image_path=str(img),
                status="collected", frigate_label=boxes[0][0], created_at=utcnow(),
            )
        )
        s.flush()
        for i, (label, x1, y1, x2, y2) in enumerate(boxes):
            s.add(
                Annotation(
                    id=f"a{sample_id}-{i}", sample_id=sample_id,
                    source="frigate", event_id=event_ids[i], label=label,
                    x1=x1, y1=y1, x2=x2, y2=y2, confidence=0.8,
                    verified=int(verified), created_at=utcnow(),
                )
            )
        s.commit()


def sample_id_hash(sample_id: str) -> float:
    total = sum(ord(c) for c in sample_id)
    return 100.0 + total


def _read_emitted_labels(config, version="v001"):
    target = config.datasets_dir() / version
    records = list(iter_manifest(target / "manifest.jsonl"))
    split = records[0]["split"]
    label_path = target / "labels" / split / f"{records[0]['sample_id']}.txt"
    return label_path.read_text(encoding="utf-8").splitlines()


def test_builder_emits_all_unverified_boxes_when_opted_in(config, db, tmp_path):
    _seed_multi_box(config, db, verified=False)
    summary = DatasetBuilder(config, db).build("v001", verified_only=False)
    assert summary.images_written == 1
    rows = _read_emitted_labels(config)
    assert len(rows) == 2
    assert {line.split()[0] for line in rows} == {"0", "2"}  # person, car (coco80)


def test_builder_verified_only_still_skips_unverified_frames(config, db, tmp_path):
    _seed_multi_box(config, db, verified=False)
    summary = DatasetBuilder(config, db).build("v001")
    assert summary.images_written == 0
    assert summary.skipped_no_box == 1


def test_builder_dedupes_verified_vs_overlapping_frigate_box(config, db, tmp_path):
    _seed_multi_box(
        config, db,
        boxes=[
            ("person", 0.1, 0.1, 0.6, 0.8),
            ("person", 0.15, 0.15, 0.7, 0.9),
        ],
    )
    with db.session() as s:
        box = s.query(Annotation).filter(Annotation.id == "as1-1").one()
        box.source = "frigate"
        box.verified = 0
        box.event_id = "e-other"
        first = s.query(Annotation).filter(Annotation.id == "as1-0").one()
        first.source = "vlm"
        s.commit()
    summary = DatasetBuilder(config, db).build("v001", verified_only=False)
    assert summary.images_written == 1
    rows = _read_emitted_labels(config)
    assert len(rows) == 1  # the overlapping frigate person box is suppressed


def test_builder_keeps_overlapping_different_labels(config, db, tmp_path):
    _seed_multi_box(
        config, db,
        boxes=[
            ("person", 0.1, 0.1, 0.6, 0.8),
            ("car", 0.15, 0.15, 0.7, 0.9),
        ],
    )
    summary = DatasetBuilder(config, db).build("v001", verified_only=False)
    assert summary.images_written == 1
    rows = _read_emitted_labels(config)
    assert len(rows) == 2
    assert {line.split()[0] for line in rows} == {"0", "2"}


def test_builder_cap_counts_boxes_not_samples(config, db, tmp_path):
    _seed_multi_box(config, db, sample_id="s1", boxes=[
        ("person", 0.1, 0.1, 0.5, 0.8),
        ("person", 0.55, 0.1, 0.9, 0.8),
    ])
    _seed_multi_box(config, db, sample_id="s2", boxes=[
        ("person", 0.1, 0.1, 0.5, 0.8),
        ("car", 0.5, 0.2, 0.9, 0.6),
    ])
    _seed_multi_box(config, db, sample_id="s3", boxes=[
        ("person", 0.1, 0.1, 0.9, 0.9),
    ])
    summary = DatasetBuilder(config, db).build("v001", max_per_class=2)
    assert summary.images_written == 2  # s1 (2 person) + s2 (car only)
    assert summary.skipped_cap == 1     # s3 had only a capped person box
    target = config.datasets_dir() / "v001"
    written = {r["sample_id"]: r for r in iter_manifest(target / "manifest.jsonl")}
    assert set(written) == {"s1", "s2"}
    split = next(iter(written.values()))["split"]
    s2_lines = (target / "labels" / split / "s2.txt").read_text().splitlines()
    assert len(s2_lines) == 1
    assert s2_lines[0].split()[0] == "2"  # only the car box survived the cap
    assert "labels" in written["s2"]
    assert written["s2"]["labels"] == ["car"]


def test_builder_label_filter_routes_through_annotations(config, db, tmp_path):
    _seed_multi_box(config, db, sample_id="s1")
    by_car = DatasetBuilder(config, db).build("v001", labels=["car"])
    assert by_car.images_written == 1
    records = list(iter_manifest(config.datasets_dir() / "v001" / "manifest.jsonl"))
    assert records[0]["sample_id"] == "s1"
    by_person = DatasetBuilder(config, db).build("v002", labels=["person"])
    assert by_person.images_written == 1
    by_truck = DatasetBuilder(config, db).build("v003", labels=["truck"])
    assert by_truck.images_written == 0