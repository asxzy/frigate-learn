"""ReviewSyncer tests (Frigate review state -> local samples)."""

from __future__ import annotations

from frigate_learn.collection.review_sync import ReviewSyncer
from frigate_learn.frigate.reviews import parse_review
from frigate_learn.models import Job, Sample, utcnow


class FakeReviewsClient:
    def __init__(self, reviews_raw: list[dict]) -> None:
        self.reviews = [parse_review(r) for r in reviews_raw]
        self.closed = False

    def list_reviews(self, after, before=None, cameras=None, labels=None,
                     severity=None, limit=None, **kw):
        return self.reviews

    def close(self) -> None:
        self.closed = True


def _review(r_id: str, start: float, reviewed: bool | None = None) -> dict:
    raw = {
        "id": r_id,
        "camera": "front",
        "start_time": start,
        "end_time": start + 60,
        "severity": "detection",
        "thumb_path": "/x",
        "data": {"detections": ["e1"]},
    }
    if reviewed is not None:
        raw["has_been_reviewed"] = reviewed
    return raw


def _seed_sample(db, sample_id: str, review_id: str, quality: str | None = None) -> None:
    with db.session() as s:
        s.add(
            Sample(
                id=sample_id, camera="front", timestamp=100.0, event_id=sample_id,
                frame_index=0, review_id=review_id, source="frigate",
                frigate_label="person", quality=quality, status="collected",
                created_at=utcnow(),
            )
        )
        s.commit()


def test_sync_stores_flag_and_auto_triages(config, db):
    client = FakeReviewsClient(
        [
            _review("r1", 200, reviewed=True),
            _review("r2", 100, reviewed=False),
            _review("r3", 50),
        ]
    )
    _seed_sample(db, "s1", "r1")
    _seed_sample(db, "s2", "r2")
    _seed_sample(db, "s3", "r3")

    summary = ReviewSyncer(config, db, client=client).sync(from_ts=0)

    assert summary.reviews_found == 3
    assert summary.reviews_seen == 2
    assert summary.samples_matched == 2
    assert summary.flags_changed == 2
    assert summary.auto_useful == 1
    assert client.closed is True

    with db.session() as s:
        rows = {r.id: r for r in s.query(Sample).all()}
    assert rows["s1"].frigate_reviewed == 1
    assert rows["s1"].quality == "useful"
    assert rows["s1"].reviewed_at
    assert rows["s2"].frigate_reviewed == 0
    assert rows["s2"].quality is None
    assert rows["s2"].reviewed_at is None
    assert rows["s3"].frigate_reviewed is None
    assert rows["s3"].quality is None


def test_idempotent_and_preserves_explicit_verdict(config, db):
    client = FakeReviewsClient([_review("r1", 200, reviewed=True)])
    _seed_sample(db, "s1", "r1", quality="useful")
    _seed_sample(db, "s2", "r1")

    first = ReviewSyncer(config, db, client=client).sync(from_ts=0)
    assert first.flags_changed == 2
    assert first.auto_useful == 1

    second = ReviewSyncer(config, db, client=client).sync(from_ts=0)
    assert second.flags_changed == 0
    assert second.auto_useful == 0


def test_auto_useful_opt_out(config, db):
    client = FakeReviewsClient([_review("r1", 200, reviewed=True)])
    _seed_sample(db, "s1", "r1")

    summary = ReviewSyncer(config, db, client=client).sync(from_ts=0, auto_useful=False)

    assert summary.flags_changed == 1
    assert summary.auto_useful == 0
    with db.session() as s:
        row = s.query(Sample).filter(Sample.id == "s1").one()
    assert row.frigate_reviewed == 1
    assert row.quality is None


def test_flip_clears_reviewed_at_keeps_auto_quality(config, db):
    _seed_sample(db, "s1", "r1")

    first = ReviewSyncer(
        config, db, client=FakeReviewsClient([_review("r1", 200, reviewed=True)])
    ).sync(from_ts=0)
    assert first.flags_changed == 1

    second = ReviewSyncer(
        config, db, client=FakeReviewsClient([_review("r1", 200, reviewed=False)])
    ).sync(from_ts=0)
    assert second.flags_changed == 1

    with db.session() as s:
        row = s.query(Sample).filter(Sample.id == "s1").one()
    assert row.frigate_reviewed == 0
    assert row.reviewed_at is None
    assert row.quality == "useful"


def test_job_recorded(config, db):
    client = FakeReviewsClient([_review("r1", 200, reviewed=True)])
    _seed_sample(db, "s1", "r1")

    summary = ReviewSyncer(config, db, client=client).sync(from_ts=0)

    with db.session() as s:
        job = s.query(Job).filter(Job.id == summary.job_id).one()
    assert job.type == "review"
    assert job.status == "finished"
    assert "auto_useful" in job.metadata_json


def test_transport_failure_marks_job_failed(config, db):
    class Boom:
        def list_reviews(self, **kw):
            raise RuntimeError("connection refused")

        def close(self) -> None:
            pass

    _seed_sample(db, "s1", "r1")
    summary = ReviewSyncer(config, db, client=Boom()).sync(from_ts=0)

    assert summary.error
    with db.session() as s:
        job = s.query(Job).filter(Job.id == summary.job_id).one()
    assert job.status == "failed"