"""Inspection + triage tests (Phase 2)."""

from __future__ import annotations

from frigate_learn.inspect.report import collect_records, render_html_report
from frigate_learn.inspect.triage import (
    QUALITY_VALUES,
    set_batch_quality,
    set_sample_quality,
    unset_quality,
)
from frigate_learn.models import Annotation, Sample, utcnow


def _seed(config, db, camera="front"):
    with db.session() as s:
        for idx in range(3):
            s.add(
                Sample(
                    id=f"s{idx}",
                    camera=camera,
                    timestamp=float(100 + idx),
                    event_id=f"e{idx}",
                    source="frigate",
                    frigate_label="person",
                    status="collected",
                    created_at=utcnow(),
                )
            )
        s.commit()


def test_quality_values_are_fixed():
    assert QUALITY_VALUES == {"useful", "bad", "duplicate", "ignore"}


def test_set_and_unset_single(config, db):
    _seed(config, db)
    assert set_sample_quality(config, db, "s0", "useful") is True
    with db.session() as s:
        assert s.get(Sample, "s0").quality == "useful"
    assert unset_quality(config, db, "s0") is True
    with db.session() as s:
        assert s.get(Sample, "s0").quality is None
    assert set_sample_quality(config, db, "missing", "useful") is False


def test_set_batch_quality_filters(config, db):
    _seed(config, db)
    updated = set_batch_quality(config, db, "bad", cameras=["front"], limit=2)
    assert updated == 2
    with db.session() as s:
        values = [r[0] for r in s.query(Sample.quality).all()]
    assert values.count(None) == 1  # third sample untouched (limit)
    assert values.count("bad") == 2


def test_set_batch_quality_clear(config, db):
    _seed(config, db)
    set_batch_quality(config, db, "ignore", cameras=["front"])
    cleared = set_batch_quality(config, db, None, cameras=["front"], clear=True)
    assert cleared == 3
    with db.session() as s:
        assert all(r[0] is None for r in s.query(Sample.quality).all())


def test_set_batch_quality_rejects_unknown_value(config, db):
    _seed(config, db)
    try:
        set_batch_quality(config, db, "nonsense", cameras=["front"])
        raised = False
    except ValueError:
        raised = True
    assert raised
    with db.session() as s:
        assert all(r[0] is None for r in s.query(Sample.quality).all())


def test_batch_quality_label_filter_uses_annotations(config, db):
    _seed(config, db)
    with db.session() as s:
        s.add(
            Annotation(
                id="car-ann", sample_id="s0", source="frigate", label="car",
                x1=0.1, y1=0.1, x2=0.5, y2=0.5, verified=0, created_at=utcnow(),
            )
        )
        s.commit()
    updated = set_batch_quality(config, db, "bad", labels=["car"])
    assert updated == 1
    with db.session() as s:
        assert s.get(Sample, "s0").quality == "bad"
        assert s.get(Sample, "s1").quality is None


def test_collect_records_label_filter_uses_annotations(config, db):
    _seed(config, db)
    with db.session() as s:
        s.add(
            Annotation(
                id="car-ann", sample_id="s2", source="frigate", label="car",
                x1=0.1, y1=0.1, x2=0.5, y2=0.5, verified=0, created_at=utcnow(),
            )
        )
        s.commit()
    records = collect_records(config, db, labels=["car"])
    assert [r.sample_id for r in records] == ["s2"]
    assert collect_records(config, db, labels=["truck"]) == []


def test_collect_records_and_html_report(config, db, tmp_path):
    _seed(config, db)
    records = collect_records(config, db, cameras=["front"])
    assert len(records) == 3
    assert records[0].label == "person"

    out = tmp_path / "report"
    summary = render_html_report(records, out)
    index = out / "index.html"
    assert index.is_file()
    assert "Frigate Learn" in index.read_text(encoding="utf-8")
    assert summary.total == 3


def test_render_html_thumbnail_generated(config, db, tmp_path):
    from PIL import Image

    img_dir = config.images_dir() / "20260901" / "front"
    img_dir.mkdir(parents=True, exist_ok=True)
    img = img_dir / "s0.jpg"
    Image.new("RGB", (320, 240), (10, 10, 10)).save(img)
    with db.session() as s:
        s.add(
            Sample(
                id="s0", camera="front", timestamp=100.0, image_path=str(img),
                source="frigate", frigate_label="person", status="collected",
                created_at=utcnow(),
            )
        )
        s.commit()
    records = collect_records(config, db)
    out = tmp_path / "report"
    render_html_report(records, out)
    thumb = out / "thumbs" / "s0.jpg"
    assert thumb.is_file()