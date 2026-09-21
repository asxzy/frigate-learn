"""End-to-end: crop-domain multi-object through collect -> verify -> build.

Validates the training-domain invariant from AGENTS.md: the training input is
the cropped region image exactly as Frigate serves it (byte-identical through
the whole pipeline, never a full frame), and multi-object means several
objects inside that one cropped image — collect stores one sample per event,
the VLM verifier writes one annotation per object it sees in the crop, and
the dataset builder emits every box.
"""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageDraw

from frigate_learn.annotation.schema import VLMObject
from frigate_learn.annotation.verifier import Verifier
from frigate_learn.collection.collector import Collector
from frigate_learn.dataset.builder import DatasetBuilder
from frigate_learn.dataset.manifest import iter_manifest
from frigate_learn.frigate.events import parse_event
from frigate_learn.frigate.reviews import parse_review
from frigate_learn.models import Annotation, Sample


def _crop_bytes() -> bytes:
    img = Image.new("RGB", (200, 100), (225, 225, 225))
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, 70, 90], fill=(120, 60, 40))
    draw.rectangle([110, 30, 190, 90], fill=(40, 90, 160))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class CropFakeFrigate:
    def __init__(self, crop_bytes: bytes, event_id: str = "e1") -> None:
        self.crop_bytes = crop_bytes
        self.review = parse_review({
            "id": "r1",
            "camera": "front",
            "severity": "detection",
            "start_time": 200,
            "end_time": 260,
            "thumb_path": "/x",
            "data": {"detections": [event_id]},
        })
        self.event = parse_event({
            "id": event_id,
            "camera": "front",
            "label": "person",
            "start_time": 200,
            "end_time": 260,
            "top_score": 0.8,
            "false_positive": False,
            "zones": [],
            "has_clip": True,
            "has_snapshot": True,
            "box": [0.1, 0.2, 0.3, 0.6],
            "data": {"score": 0.9},
        })
        self.closed = False
        self.downloads: list[tuple[str, Path, object]] = []

    def list_reviews(self, after, before=None, cameras=None, labels=None,
                     severity=None, limit=None, **kw):
        return [self.review]

    def get_event(self, event_id: str):
        return self.event

    def download_region_crop(self, event_id: str, output_path, height, timestamp=None) -> str:
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.crop_bytes)
        self.downloads.append((event_id, dest, height))
        return str(dest)

    def close(self) -> None:
        self.closed = True


class TwoObjectProvider:
    def __init__(self, objects: list[VLMObject]) -> None:
        self.objects = objects
        self.calls = 0

    def verify(self, images):
        self.calls += 1
        return [self.objects] * len(images)


def test_crop_domain_multi_object_collect_verify_build(config, db, tmp_path):
    config.collection.region_crop = True
    config.collection.concurrency = 1
    config.vlm.enabled = True
    config.vlm.base_url = "http://llm:4001/v1"
    config.vlm.api_key = "k"
    config.vlm.batch_size = 2

    crop_bytes = _crop_bytes()
    fake = CropFakeFrigate(crop_bytes)
    summary = Collector(config, db, client=fake).collect(from_ts=0)
    assert summary.new_samples == 1
    assert summary.new_annotations == 0
    assert summary.merged_events == 0

    with db.session() as s:
        sample = s.query(Sample).one()
        assert s.query(Annotation).count() == 0
        assert (sample.frigate_x1, sample.frigate_y1,
                sample.frigate_x2, sample.frigate_y2) == (0.1, 0.2, 0.4, 0.8)
        stored = Path(sample.image_path)
        sample_id = sample.id
    assert stored.read_bytes() == crop_bytes

    provider = TwoObjectProvider([
        VLMObject(label="person", confidence=0.91, bbox=(0.05, 0.1, 0.4, 0.9)),
        VLMObject(label="car", confidence=0.87, bbox=(0.55, 0.3, 0.95, 0.9)),
    ])
    verify_summary = Verifier(config, db, provider=provider).verify()
    assert verify_summary.annotated == 1
    assert verify_summary.objects_written == 2

    with db.session() as s:
        sample = s.get(Sample, sample_id)
        assert sample.verified == 1
        vlm_anns = (
            s.query(Annotation)
            .filter(Annotation.sample_id == sample_id, Annotation.source == "vlm")
            .all()
        )
        assert len(vlm_anns) == 2
        assert {a.label for a in vlm_anns} == {"person", "car"}
        assert all(a.verified == 1 for a in vlm_anns)

    build = DatasetBuilder(config, db).build("v001")
    assert build.images_written == 1
    target = config.datasets_dir() / "v001"
    recs = list(iter_manifest(target / "manifest.jsonl"))
    assert len(recs) == 1
    split = recs[0]["split"]
    assert sorted(recs[0]["labels"]) == ["car", "person"]

    label_lines = (target / "labels" / split / f"{sample_id}.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(label_lines) == 2
    assert sorted(line.split()[0] for line in label_lines) == ["0", "2"]

    built_image = next((target / "images" / split).glob(f"{sample_id}.*"))
    assert built_image.read_bytes() == crop_bytes