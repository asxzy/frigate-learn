"""Verifier tests using a fake VLM provider (Phase 4/8)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from frigate_learn.annotation.verifier import Verifier
from frigate_learn.annotation.schema import VLMObject, VLMValidationError
from frigate_learn.models import Annotation, Sample, utcnow


class FakeProvider:
    def __init__(self, per_image):
        self.per_image = per_image  # list of lists per call
        self.calls = 0

    def verify(self, images):
        self.calls += 1
        return self.per_image


class FlakyProvider(FakeProvider):
    """Rejects multi-image calls; raises on 'bad.jpg' even singly."""

    def __init__(self):
        super().__init__(per_image=None)
        self.bad_calls = 0

    def verify(self, images):
        if len(images) > 1:
            raise VLMValidationError("batch rejected")
        if Path(images[0]).name == "bad.jpg":
            self.bad_calls += 1
            raise VLMValidationError("model refused")
        return [[VLMObject(label="person", confidence=0.9, bbox=(0.1, 0.1, 0.6, 0.6))]]


def _sample_record(db, sample_id="s1", camera="front", label="person", quality=None, verified=0):
    with db.session() as s:
        s.add(
            Sample(
                id=sample_id,
                camera=camera,
                timestamp=100.0,
                event_id="e" + sample_id,
                frame_index=0,
                source="frigate",
                frigate_label=label,
                status="collected",
                quality=quality,
                verified=verified,
                created_at=utcnow(),
            )
        )
        s.commit()
    return sample_id


def _make_image(tmp_path, name):
    path = tmp_path / name
    Image.new("RGB", (32, 32), (10, 10, 10)).save(path)
    return path


def _config_with_vlm(config):
    config.vlm.enabled = True
    config.vlm.base_url = "http://llm:4001/v1"
    config.vlm.api_key = "k"
    config.vlm.batch_size = 2
    return config


def test_default_provider_interpolates_classes(config):
    config.vlm.enabled = True
    config.vlm.model = "Qwen3.8-27B"
    config.vlm.base_url = "http://llm:4001/v1"
    provider = Verifier(config, db=None)._default_provider()
    # The default system prompt contains {classes} (and literal {"objects"});
    # interpolation must not crash on the braces (replace, not format).
    assert "{classes}" not in provider.system_prompt
    assert "person" in provider.system_prompt and "car" in provider.system_prompt


def test_verify_annotates_unverified_samples(config, db, tmp_path):
    _config_with_vlm(config)
    img = _make_image(tmp_path, "a.jpg")
    _sample_record(db, "s1", "front", "person")
    with db.session() as s:
        row = s.get(Sample, "s1")
        row.image_path = str(img)
        s.commit()

    provider = FakeProvider(
        [[VLMObject(label="person", confidence=0.91, bbox=(0.1, 0.1, 0.6, 0.6))]]
    )
    summary = Verifier(config, db, provider=provider).verify()
    assert summary.annotated == 1
    assert summary.objects_written == 1

    with db.session() as s:
        sample = s.get(Sample, "s1")
        ann = s.query(Annotation).filter(Annotation.sample_id == "s1").one()
    assert sample.verified == 1
    assert ann.source == "vlm"
    assert ann.verified == 1
    assert ann.confidence == 0.91


def test_verify_limit_applies_after_order(config, db, tmp_path):
    _config_with_vlm(config)
    img = _make_image(tmp_path, "a.jpg")
    for sid in ("s1", "s2", "s3"):
        _sample_record(db, sid, "front", "person")
        with db.session() as s:
            s.get(Sample, sid).image_path = str(img)
            s.commit()

    provider = FakeProvider([[]])
    summary = Verifier(config, db, provider=provider).verify(limit=2)
    # order_by() before limit() — must not raise InvalidRequestError
    assert summary.processed == 2


def test_verify_isolates_per_image_failures(config, db, tmp_path):
    _config_with_vlm(config)
    good_img = _make_image(tmp_path, "good.jpg")
    bad_img = _make_image(tmp_path, "bad.jpg")
    _sample_record(db, "s1", "front", "person")
    _sample_record(db, "s2", "front", "person")
    with db.session() as s:
        s.get(Sample, "s1").image_path = str(good_img)
        s.get(Sample, "s2").image_path = str(bad_img)
        s.commit()

    summary = Verifier(config, db, provider=FlakyProvider()).verify()
    # the batch call fails, per-image isolation recovers the good one and
    # drops the bad one instead of aborting the whole pass
    assert summary.processed == 2
    assert summary.annotated == 1
    assert summary.failed == 1
    assert summary.dropped == 1
    assert summary.objects_written == 1
    with db.session() as s:
        s1 = s.get(Sample, "s1")
        s2 = s.get(Sample, "s2")
    assert s1.verified == 1
    assert s2.verified == 0
    assert s2.quality == "bad"  # dropped → excluded from verify + builds


def test_verify_dropped_never_reprocessed(config, db, tmp_path):
    _config_with_vlm(config)
    _make_image(tmp_path, "good.jpg")
    _make_image(tmp_path, "bad.jpg")
    _sample_record(db, "s1", "front", "person")
    _sample_record(db, "s2", "front", "person")
    with db.session() as s:
        s.get(Sample, "s1").image_path = str(tmp_path / "bad.jpg")
        s.get(Sample, "s2").image_path = str(tmp_path / "good.jpg")
        s.commit()

    summary = Verifier(config, db, provider=FlakyProvider()).verify()
    assert summary.annotated == 1  # s2 verified
    assert summary.dropped == 1    # s1 dropped

    # second / force runs must never re-pick the dropped sample
    fake = FakeProvider([[VLMObject(label="person", confidence=0.9, bbox=(0.1, 0.1, 0.6, 0.6))]])
    summary2 = Verifier(config, db, provider=fake).verify()
    assert summary2.processed == 0  # nothing left (s1 dropped, s2 done)
    Verifier(config, db, provider=fake).verify(force=True)
    with db.session() as s:
        assert s.get(Sample, "s1").verified == 0
        assert s.get(Sample, "s1").quality == "bad"
        assert s.query(Annotation).filter(Annotation.sample_id == "s1").count() == 0


def test_verify_skips_verified_unless_force(config, db, tmp_path):
    _config_with_vlm(config)
    img = _make_image(tmp_path, "a.jpg")
    _sample_record(db, "s1", "front", "person", verified=1)
    with db.session() as s:
        row = s.get(Sample, "s1")
        row.image_path = str(img)
        s.commit()

    provider = FakeProvider([[]])
    summary = Verifier(config, db, provider=provider).verify()
    assert summary.processed == 0  # verified sample excluded by default
    assert summary.annotated == 0

    summary = Verifier(config, db, provider=provider).verify(force=True)
    assert summary.annotated == 1


def test_verify_skips_bad_quality(config, db, tmp_path):
    _config_with_vlm(config)
    img = _make_image(tmp_path, "a.jpg")
    _sample_record(db, "s1", "front", "person", quality="bad")
    with db.session() as s:
        row = s.get(Sample, "s1")
        row.image_path = str(img)
        s.commit()
    summary = Verifier(config, db, provider=FakeProvider([[]])).verify()
    assert summary.processed == 0


def test_verify_missing_image_skipped(config, db, tmp_path):
    _config_with_vlm(config)
    _sample_record(db, "s1", "front", "person", quality=None)
    summary = Verifier(config, db, provider=FakeProvider([[]])).verify()
    assert summary.skipped == 1
    assert summary.annotated == 0


def test_verify_ignores_labels_outside_config_classes(config, db, tmp_path):
    config = _config_with_vlm(config)
    img = _make_image(tmp_path, "a.jpg")
    _sample_record(db, "s1", "front", "person")
    with db.session() as s:
        row = s.get(Sample, "s1")
        row.image_path = str(img)
        s.commit()
    config.classes = ["person", "car"]
    provider = FakeProvider(
        [[VLMObject(label="alien", confidence=0.9, bbox=(0, 0, 0.5, 0.5))]]
    )
    summary = Verifier(config, db, provider=provider).verify()
    assert summary.annotated == 1
    assert summary.objects_written == 0  # filtered out of the allowed set


def test_verify_records_job(config, db, tmp_path):
    _config_with_vlm(config)
    img = _make_image(tmp_path, "a.jpg")
    _sample_record(db, "s1", "front", "person", quality="useful")
    with db.session() as s:
        row = s.get(Sample, "s1")
        row.image_path = str(img)
        s.commit()
    provider = FakeProvider([[]])
    summary = Verifier(config, db, provider=provider).verify()
    with db.session() as s:
        from frigate_learn.models import Job

        job = s.get(Job, summary.job_id)
    assert job is not None
    assert job.status == "finished"


def test_verify_label_filter_routes_through_annotations(config, db, tmp_path):
    _config_with_vlm(config)
    img = _make_image(tmp_path, "a.jpg")
    _sample_record(db, "s1", "front", "person")
    with db.session() as s:
        row = s.get(Sample, "s1")
        row.image_path = str(img)
        s.add(
            Annotation(
                id="car-ann", sample_id="s1", source="frigate", label="car",
                x1=0.1, y1=0.1, x2=0.6, y2=0.6, verified=0, created_at=utcnow(),
            )
        )
        s.commit()

    provider = FakeProvider([[]])
    by_car = Verifier(config, db, provider=provider).verify(labels=["car"])
    assert by_car.processed == 1
    by_truck = Verifier(config, db, provider=provider).verify(labels=["truck"])
    assert by_truck.processed == 0