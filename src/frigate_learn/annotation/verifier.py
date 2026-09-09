"""VLM verifier + teacher (Phase 4 / Phase 8).

Takes collected samples, runs their images through a VLM provider, validates the
outputs against the strict schema, and writes the results back as ``verified``
annotations (``source='vlm', verified=1``), flipping ``samples.verified``. The
same pipeline doubles as the P8 "VLM teacher": re-running with ``force=True``
replaces prior VLM annotations with fresh ones.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..config import AppConfig
from ..db import Database
from ..logutil import debug, error, info, warning
from .schema import VLMValidationError
from .vlm import VLMProvider, build_provider
from ..models import Annotation, Job, Sample, utcnow

logger = logging.getLogger(__name__)


@dataclass
class VerifySummary:
    processed: int = 0
    annotated: int = 0
    skipped: int = 0
    failed: int = 0
    dropped: int = 0
    objects_written: int = 0
    job_id: str | None = None
    duration_seconds: float = 0.0
    error: str | None = None


class Verifier:
    def __init__(
        self, config: AppConfig, db: Database, provider: VLMProvider | None = None
    ) -> None:
        self.config = config
        self.db = db
        self.provider = provider or self._default_provider()

    def _default_provider(self) -> VLMProvider:
        vlm = self.config.vlm
        prompt = vlm.system_prompt.replace("{classes}", ", ".join(self.config.classes))
        return build_provider(
            base_url=vlm.base_url,
            api_key=vlm.api_key,
            model=vlm.model,
            provider_name=vlm.provider,
            temperature=vlm.temperature,
            timeout_seconds=vlm.timeout_seconds,
            system_prompt=prompt,
            allowed_labels=self.config.classes,
        )

    # --- public -----------------------------------------------------------

    def verify(
        self,
        *,
        limit: int | None = None,
        cameras: list[str] | None = None,
        labels: list[str] | None = None,
        days: int | None = None,
        force: bool = False,
    ) -> VerifySummary:
        """Verify unverified samples (or all when ``force``)."""
        started = datetime.now(timezone.utc)
        summary = VerifySummary()
        job_id = str(uuid.uuid4())
        summary.job_id = job_id
        self._record_job(job_id, status="running")

        from_ts: float | None = None
        if days is not None:
            from_ts = datetime.now(timezone.utc).timestamp() - days * 86400.0

        with self.db.session() as session:
            query = session.query(Sample)
            if cameras:
                query = query.filter(Sample.camera.in_(cameras))
            if labels:
                query = query.filter(Sample.frigate_label.in_(labels))
            if from_ts is not None:
                query = query.filter(Sample.timestamp >= from_ts)
            query = query.filter(
                (Sample.quality != "bad") | (Sample.quality.is_(None))
            )
            if not force:
                query = query.filter(Sample.verified == 0)
            query = query.order_by(Sample.timestamp.asc())
            if limit is not None:
                query = query.limit(limit)

            candidates = query.all()

        summary.processed = len(candidates)
        if not candidates:
            self._finish_job(job_id, summary)
            return summary

        pictures = [Path(s.image_path) if s.image_path else None for s in candidates]
        summary.skipped = sum(1 for p in pictures if p is None or not p.is_file())

        batch_size = max(1, self.config.vlm.batch_size)
        try:
            for start in range(0, len(candidates), batch_size):
                batch = candidates[start : start + batch_size]
                pairs = [
                    (s, Path(s.image_path))
                    for s in batch
                    if s.image_path and Path(s.image_path).is_file()
                ]
                if not pairs:
                    continue
                _samples, paths = zip(*pairs)
                try:
                    per_image = self.provider.verify(list(paths))
                except VLMValidationError:
                    # One bad response must not abort the whole pass: isolate
                    # the offending image(s), drop them, and keep the good ones.
                    debug("vlm batch rejected; isolating per-image", batch=len(paths))
                    per_image = []
                    for sample, path in pairs:
                        try:
                            per_image.append(self.provider.verify([path])[0])
                        except VLMValidationError as exc:
                            debug("vlm rejection", sample_id=sample.id, error=str(exc))
                            summary.failed += 1
                            self._drop(sample, summary)
                            per_image.append(None)
                for (sample, _path), objects in zip(pairs, per_image):
                    if objects is None:
                        continue
                    try:
                        self._store(sample, objects, summary)
                    except VLMValidationError as exc:
                        debug("vlm rejection", sample_id=sample.id, error=str(exc))
                        summary.failed += 1
                        self._drop(sample, summary)
            summary.duration_seconds = (
                datetime.now(timezone.utc) - started
            ).total_seconds()
            self._finish_job(job_id, summary)
        except Exception as exc:
            logger.exception("verify failed")
            summary.error = str(exc)
            summary.duration_seconds = (datetime.now(timezone.utc) - started).total_seconds()
            self._fail_job(job_id, exc)
        return summary

    # --- internals --------------------------------------------------------

    def _store(
        self, sample: Sample, objects, summary: VerifySummary
    ) -> None:
        allowed = set(self.config.classes)
        accepted = [o for o in objects if o is not None and o.label in allowed]
        with self.db.session() as session:
            sess_sample = session.get(Sample, sample.id)
            if sess_sample is None:
                summary.skipped += 1
                return
            sess_sample.verified = 1
            for obj in accepted:
                session.add(
                    Annotation(
                        id=str(uuid.uuid4()),
                        sample_id=sample.id,
                        source="vlm",
                        label=obj.label,
                        x1=obj.bbox[0],
                        y1=obj.bbox[1],
                        x2=obj.bbox[2],
                        y2=obj.bbox[3],
                        confidence=obj.confidence,
                        verified=1,
                    )
                )
                summary.objects_written += 1
            session.commit()
        summary.annotated += 1
        info("vlm annotated", sample_id=sample.id, objects=len(accepted))

    def _drop(self, sample: Sample, summary: VerifySummary) -> None:
        """Withdraw a VLM-failed sample from the pool.

        Uses the ``bad`` quality verdict: both the verifier's candidate query and
        the dataset builder already exclude ``bad`` samples, so a failure is
        permanent for the current model but reversible via ``triage set``.
        """
        summary.dropped += 1
        with self.db.session() as session:
            row = session.get(Sample, sample.id)
            if row is not None and row.quality != "bad":
                row.quality = "bad"
                session.commit()
        warning("vlm dropped", sample_id=sample.id)

    def _record_job(self, job_id: str, status: str) -> None:
        with self.db.session() as session:
            session.add(Job(id=job_id, type="annotate", status=status, started_at=utcnow()))
            session.commit()

    def _finish_job(self, job_id: str, summary: VerifySummary) -> None:
        import json

        with self.db.session() as session:
            job = session.get(Job, job_id)
            if job:
                job.status = "finished"
                job.finished_at = utcnow()
                job.metadata_json = json.dumps(
                    {
                        "processed": summary.processed,
                        "annotated": summary.annotated,
                        "skipped": summary.skipped,
                        "failed": summary.failed,
                        "dropped": summary.dropped,
                        "objects_written": summary.objects_written,
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


__all__ = ["Verifier", "VerifySummary"]