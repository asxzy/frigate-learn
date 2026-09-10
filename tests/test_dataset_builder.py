"""Dataset builder: verified-only default and split-directory layout."""

from __future__ import annotations

from PIL import Image

from frigate_learn.dataset.builder import DatasetBuilder
from frigate_learn.dataset.manifest import iter_manifest
from frigate_learn.models import Annotation, Sample, utcnow


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