"""Frigate Review state sync (human-confirmation channel).

Operators confirm/deny events in Frigate's own Review UI (marking review
segments viewed). This stage ingests that state and maps it onto the local
sample pool:

    has_been_reviewed=True  ->  samples.frigate_reviewed = 1
    has_been_reviewed=False ->  samples.frigate_reviewed = 0
    (flag absent on review) ->  left NULL (unknown)

With ``review_auto_useful`` enabled (default), samples whose review segment is
marked reviewed and whose local verdict is unset are auto-triaged to
``useful``: the human's Review action becomes the batch confirmation the
Triage lightbox used to provide. Box-level labels are untouched — the VLM
verifier still owns those. The raw flag is always stored; the quality
remapping only ever ADDS ``useful`` and never overwrites an explicit verdict.

Frigate stores the reviewed flag PER USER, so the token user must be the same
account that marks reviews in the Frigate UI (see docs).
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from ..config import AppConfig
from ..db import Database
from ..frigate.client import FrigateClient
from ..logutil import debug, info
from ..models import Job, Sample, utcnow

logger = logging.getLogger(__name__)


@dataclass
class ReviewSyncSummary:
    reviews_found: int = 0
    reviews_seen: int = 0
    samples_matched: int = 0
    flags_changed: int = 0
    auto_useful: int = 0
    job_id: str | None = None
    duration_seconds: float = 0.0
    error: str | None = None


class ReviewSyncer:
    def __init__(self, config: AppConfig, db: Database, client: FrigateClient | None = None) -> None:
        self.config = config
        self.db = db
        self.client = client or FrigateClient(
            base_url=config.frigate.base_url,
            token=config.frigate.token,
            timeout_seconds=config.frigate.timeout_seconds,
            max_retries=config.frigate.max_retries,
            retry_backoff=config.frigate.retry_backoff,
        )

    @staticmethod
    def _flag(value: bool | None) -> int | None:
        if value is None:
            return None
        return 1 if value else 0

    def sync(
        self,
        *,
        from_ts: float,
        to_ts: float | None = None,
        cameras: Sequence[str] | None = None,
        severity: Sequence[str] | None = None,
        auto_useful: bool | None = None,
        progress=None,
    ) -> ReviewSyncSummary:
        started = datetime.now(UTC)
        summary = ReviewSyncSummary()
        job_id = str(uuid.uuid4())
        summary.job_id = job_id
        self.db.migrate()
        self._record_job(job_id, status="running")

        apply_useful = (
            self.config.collection.review_auto_useful
            if auto_useful is None
            else bool(auto_useful)
        )
        try:
            reviewed_by_id: dict[str, bool] = {}
            for review in self.client.list_reviews(
                after=from_ts,
                before=to_ts,
                cameras=list(cameras) if cameras else (self.config.collection.cameras or None),
                severity=list(severity) if severity else (self.config.collection.severity or None),
            ):
                summary.reviews_found += 1
                if review.has_been_reviewed is None:
                    continue
                reviewed_by_id[review.id] = review.has_been_reviewed
            summary.reviews_seen = len(reviewed_by_id)
            (progress or _noop)(f"Reviews seen: {summary.reviews_seen}/{summary.reviews_found}")

            if reviewed_by_id:
                self._apply(reviewed_by_id, summary, apply_useful)
            summary.duration_seconds = (
                datetime.now(UTC) - started
            ).total_seconds()
            self._finish_job(job_id, summary)
        except Exception as exc:
            logger.exception("review-sync failed")
            summary.error = str(exc)
            summary.duration_seconds = (datetime.now(UTC) - started).total_seconds()
            self._fail_job(job_id, exc)
        finally:
            self.client.close()
        return summary

    def _apply(self, reviewed_by_id: dict[str, bool], summary: ReviewSyncSummary, apply_useful: bool) -> None:
        review_ids = list(reviewed_by_id.keys())
        with self.db.session() as session:
            rows = (
                session.query(Sample)
                .filter(Sample.review_id.in_(review_ids))
                .all()
            )
            summary.samples_matched = len(rows)
            now = utcnow()
            for sample in rows:
                new_flag = self._flag(reviewed_by_id[sample.review_id])
                if sample.frigate_reviewed != new_flag:
                    sample.frigate_reviewed = new_flag
                    sample.reviewed_at = now if new_flag else None
                    summary.flags_changed += 1
                if apply_useful and new_flag == 1 and sample.quality is None:
                    sample.quality = "useful"
                    summary.auto_useful += 1
            session.commit()
        info(
            "review-sync applied",
            seen=summary.reviews_seen,
            matched=summary.samples_matched,
            changed=summary.flags_changed,
            auto_useful=summary.auto_useful,
        )

    def _record_job(self, job_id: str, status: str) -> None:
        with self.db.session() as session:
            session.add(Job(id=job_id, type="review", status=status, started_at=utcnow()))
            session.commit()

    def _finish_job(self, job_id: str, summary: ReviewSyncSummary) -> None:
        import json

        with self.db.session() as session:
            job = session.get(Job, job_id)
            if job:
                job.status = "finished"
                job.finished_at = utcnow()
                job.metadata_json = json.dumps(
                    {
                        "reviews_found": summary.reviews_found,
                        "reviews_seen": summary.reviews_seen,
                        "samples_matched": summary.samples_matched,
                        "flags_changed": summary.flags_changed,
                        "auto_useful": summary.auto_useful,
                    },
                    sort_keys=True,
                )
                session.commit()

    def _fail_job(self, job_id: str, exc: Exception) -> None:
        with self.db.session() as session:
            job = session.get(Job, job_id)
            if job:
                job.status = "failed"
                job.finished_at = utcnow()
                job.error = f"{type(exc).__name__}: {exc}"[:2000]
                session.commit()


def _noop(message: str) -> None:
    debug("progress", msg=message)


__all__ = ["ReviewSyncSummary", "ReviewSyncer"]