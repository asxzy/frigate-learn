"""Frigate segment/event batch collector (Phase 1).

Owns its pipeline state: dedup is driven by our own SQLite
``samples.event_id`` unique key + SHA-256 image hashing.

Flow:

    review segments ──▶ event ids ──▶ event detail (label/score/bbox)
                              └──▶ clean snapshot download ──▶ samples row
                              └──▶ annotated snapshot (debug, optional)

Failures are recorded per-item and never abort the batch.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

from sqlalchemy.exc import IntegrityError

from ..config import AppConfig
from ..db import Database
from ..frigate.client import FrigateAPIError, FrigateClient
from ..frigate.reviews import extract_event_ids
from ..logutil import debug, error, info, warning
from ..models import Annotation, CollectionFailure, Job, Sample, utcnow
from .dedup import dhash_file, hamming_distance
from .sampling import sample_timestamps

logger = logging.getLogger(__name__)

ProgressFn = Callable[[str], None]


@dataclass
class CollectSummary:
    reviews_found: int = 0
    reviews_selected: int = 0
    events_found: int = 0
    events_new: int = 0
    new_samples: int = 0
    duplicate_samples: int = 0
    merged_events: int = 0   # extra Frigate annotations merged into an existing same-frame sample
    failures: int = 0
    skipped_no_box: int = 0   # events skipped in crop mode for a degenerate box
    new_annotations: int = 0
    job_id: str | None = None
    range_from: float | None = None
    range_to: float | None = None
    cameras: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    severity: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    error: str | None = None


@dataclass
class EventOutcome:
    """Per-event worker result."""

    status: str          # ok | dup | fail | skip | merged
    stored: int = 0      # samples inserted
    skipped: int = 0     # frames dropped as near-duplicates
    annotations: int = 0 # annotation rows inserted for this event
    merged: int = 0      # extra annotations merged into existing samples
    error: str | None = None


class Collector:
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

    # --- public -----------------------------------------------------------

    def collect(
        self,
        from_ts: float,
        to_ts: float | None = None,
        cameras: Sequence[str] | None = None,
        labels: Sequence[str] | None = None,
        severity: Sequence[str] | None = None,
        limit: int | None = None,
        concurrency: int | None = None,
        progress: ProgressFn | None = None,
        refresh: bool = False,
        job_id: str | None = None,
        record_job: bool = True,
    ) -> CollectSummary:
        started = datetime.now(timezone.utc)
        summary = CollectSummary(
            range_from=from_ts,
            range_to=to_ts if to_ts is not None else started.timestamp(),
            cameras=list(cameras or self.config.collection.cameras),
            labels=list(labels or self.config.collection.labels),
            severity=list(severity or self.config.collection.severity),
        )
        job_id = job_id or str(uuid.uuid4())
        summary.job_id = job_id
        self.db.migrate()
        if record_job:
            self._record_job(job_id, type="collect", status="running", started_at=utcnow())

        workers = concurrency or self.config.collection.concurrency
        try:
            self.images_dir().mkdir(parents=True, exist_ok=True)

            reviews = self._fetch_reviews(from_ts, summary, limit, cameras, labels)
            summary.reviews_found = len(reviews)
            (progress or _noop)(f"Reviews found: {len(reviews)}")

            event_ids = self._resolve_events(reviews, summary)
            (progress or _noop)(f"Events found: {len(event_ids)}")

            new_events = self._prune_existing(event_ids, summary, refresh=refresh)
            (progress or _noop)(f"New events to collect: {len(new_events)}")

            self._download_and_store(
                new_events, summary, workers=workers, progress=progress,
            )

            summary.duration_seconds = (
                datetime.now(timezone.utc) - started
            ).total_seconds()
            self._finish_job(job_id, summary)
        except Exception as exc:  # fatal error -> mark job failed
            logger.exception("collect failed")
            summary.error = str(exc)
            summary.duration_seconds = (datetime.now(timezone.utc) - started).total_seconds()
            self._fail_job(job_id, exc)
        finally:
            self.client.close()
        return summary

    # --- stages -----------------------------------------------------------

    def _fetch_reviews(
        self,
        from_ts: float,
        summary: CollectSummary,
        limit: int | None,
        cameras: Sequence[str] | None,
        labels: Sequence[str] | None,
    ) -> list:
        max_reviews = limit or self.config.collection.max_reviews
        reviews: list = []
        seen: set[str] = set()
        for review in self.client.list_reviews(
            after=from_ts,
            before=summary.range_to,
            cameras=list(cameras) if cameras else (self.config.collection.cameras or None),
            labels=list(labels) if labels else (self.config.collection.labels or None),
            severity=summary.severity or None,
            limit=max_reviews,
        ):
            if review.id in seen:
                continue
            seen.add(review.id)
            reviews.append(review)
        summary.reviews_selected = len(reviews)
        info("collected review list", count=len(reviews))
        return reviews

    def _resolve_events(self, reviews: Sequence, summary: CollectSummary) -> list[str]:
        """Unique event ids referenced across the fetched segments (stable order)."""
        mapping: list[str] = []
        seen: set[str] = set()
        for review in reviews:
            for event_id in extract_event_ids(review):
                if event_id not in seen:
                    seen.add(event_id)
                    mapping.append(event_id)
        summary.events_found = len(mapping)
        return mapping

    def _prune_existing(
        self, event_ids: list[str], summary: CollectSummary, refresh: bool = False
    ) -> list[str]:
        """Remove events already stored as samples (idempotency).

        With ``refresh=True``, already-stored events are *not* pruned: their
        existing samples/annotations/image files are deleted and they are
        re-collected under the current collection settings (e.g. upgrading a
        crop-era pool to full-frame snapshots).
        """
        existing: set[str] = set()
        with self.db.session() as session:
            for event_id in event_ids:
                row = session.query(Sample.id).filter(Sample.event_id == event_id).first()
                if row is not None:
                    existing.add(event_id)
            if event_ids:
                ann_events = (
                    session.query(Annotation.event_id)
                    .filter(Annotation.event_id.in_(event_ids))
                    .all()
                )
                for (eid,) in ann_events:
                    existing.add(eid)
        summary.duplicate_samples = len(existing)
        if refresh:
            deleted = 0
            for event_id in existing:
                deleted += self._delete_event(event_id)
            info("refresh: dropped prior collection", events=len(existing), samples=deleted)
        if refresh:
            # re-collect the previously-stored events too (they were deleted above)
            new_events = list(event_ids)
        else:
            new_events = [eid for eid in event_ids if eid not in existing]
        max_events = self.config.collection.max_events
        if max_events and len(new_events) > max_events:
            new_events = new_events[:max_events]
        summary.events_new = len(new_events)
        return new_events

    def _delete_event(self, event_id: str) -> int:
        """Remove an event's stored contribution (+ image files when orphaned).

        Events merged as extra annotations have no sample of their own: only
        their annotations are removed. A sample is deleted only when it owned
        the event (``samples.event_id``) and no annotations remain.
        """
        deleted = 0
        with self.db.session() as session:
            annotations = (
                session.query(Annotation).filter(Annotation.event_id == event_id).all()
            )
            for annotation in annotations:
                session.delete(annotation)
            owned = (
                session.query(Sample).filter(Sample.event_id == event_id).all()
            )
            for sample in owned:
                remaining = (
                    session.query(Annotation.id)
                    .filter(Annotation.sample_id == sample.id)
                    .count()
                )
                if remaining:
                    continue
                if sample.image_path:
                    Path(sample.image_path).unlink(missing_ok=True)
                if sample.debug_image_path:
                    Path(sample.debug_image_path).unlink(missing_ok=True)
                session.delete(sample)
                deleted += 1
            session.commit()
        return deleted

    def _download_and_store(
        self,
        events: list[str],
        summary: CollectSummary,
        workers: int,
        progress: ProgressFn | None,
    ) -> None:
        done = 0
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {
                pool.submit(
                    self._process_event, event_id
                ): event_id
                for event_id in events
            }
            for future in as_completed(futures):
                event_id = futures[future]
                try:
                    outcome = future.result()
                except Exception as exc:  # worker-level safety net
                    error("event processing raised", event_id=event_id, error=str(exc))
                    self._record_failure(event_id, f"{type(exc).__name__}: {exc}")
                    outcome = EventOutcome(status="fail", error=str(exc))
                summary.duplicate_samples += outcome.skipped
                if outcome.status == "ok":
                    summary.new_samples += outcome.stored
                    summary.new_annotations += outcome.annotations
                    summary.merged_events += outcome.merged
                elif outcome.status == "merged":
                    summary.merged_events += outcome.merged
                elif outcome.status == "skip":
                    summary.skipped_no_box += 1
                elif outcome.status == "fail":
                    summary.failures += 1
                else:
                    summary.duplicate_samples += 1
                done += 1
                if done % 10 == 0 or done == len(events):
                    (progress or _noop)(
                        f"Downloaded {done}/{len(events)} "
                        f"(ok={summary.new_samples} dup={summary.duplicate_samples} fail={summary.failures})"
                    )

    def _frame_times(self, event) -> list[float | None]:
        """Decide which frame times to fetch for one event (P3 temporal sampling).

        Region crops are single-frame only: the event box is a single-frame
        artifact, so a stale box on a resampled frame would crop the wrong area.
        Returns ``[None]`` (the Frigate default frame) in that case, or unless
        sampling is enabled and the event spans enough time to space frames apart.
        """
        if self.config.collection.region_crop:
            return [None]
        samp = self.config.sampling
        if not samp.enabled or samp.max_samples_per_event <= 1:
            return [None]
        times = sample_timestamps(
            event.start_time, event.end_time, samp.max_samples_per_event,
            min_gap=samp.min_seconds_between_samples,
        )
        return [t if t != event.start_time else None for t in times]

    @staticmethod
    def _has_sane_box(box) -> bool:
        """A usable Frigate event box: 4 normalized coords with positive area."""
        if not box or len(box) != 4:
            return False
        x1, y1, x2, y2 = box
        return 0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0

    def _is_duplicate_frame(
        self, camera: str, image_hash: str | None, image_path: Path
    ) -> bool:
        """Exact SHA-256 dupes are always rejected; phash duplicates only when
        enabled. Comparison is scoped to the same camera."""
        candidates: list[tuple[str | None, str | None]] = []
        with self.db.session() as session:
            rows = session.query(Sample.image_hash, Sample.perceptual_hash).filter(
                Sample.camera == camera
            ).all()
            candidates = [(r[0], r[1]) for r in rows]

        if image_hash is not None:
            for known_hash, _ in candidates:
                if known_hash == image_hash:
                    return True
        if self.config.collection.dedup_enabled:
            threshold = self.config.collection.phash_threshold
            try:
                phash = dhash_file(image_path)
            except Exception as exc:
                debug("phash failed", camera=camera, error=str(exc))
                return False
            for _, known_phash in candidates:
                if known_phash and hamming_distance(phash, known_phash) <= threshold:
                    return True
        return False

    def _find_exact_duplicate_sample(self, camera: str, image_hash: str) -> Sample | None:
        """Locate the existing full-frame sample for an exact SHA-256 match."""
        if image_hash is None:
            return None
        with self.db.session() as session:
            return (
                session.query(Sample)
                .filter(Sample.camera == camera, Sample.image_hash == image_hash)
                .order_by(Sample.created_at.asc())
                .first()
            )

    def _merge_event_frame(self, event, sample: Sample) -> bool:
        """Add one Frigate annotation for ``event`` onto an existing sample.

        Returns False when ``(sample, event)`` is already annotated (the merge
        is idempotent — e.g. a later temporal frame of the same event).
        """
        with self.db.session() as session:
            existing = (
                session.query(Annotation.id)
                .filter(
                    Annotation.sample_id == sample.id,
                    Annotation.event_id == event.id,
                )
                .first()
            )
            if existing is not None:
                return False
            box = event.box
            session.add(
                Annotation(
                    id=str(uuid.uuid4()),
                    sample_id=sample.id,
                    event_id=event.id,
                    source="frigate",
                    label=event.label,
                    x1=box[0] if box else None,
                    y1=box[1] if box else None,
                    x2=box[2] if box else None,
                    y2=box[3] if box else None,
                    confidence=event.score if event.score is not None else event.top_score,
                    verified=0,
                )
            )
            session.commit()
        return True

    def _process_event(
        self,
        event_id: str,
    ) -> EventOutcome:
        """Fetch one event, download clean frame(s), store sample(s).

        An event may produce multiple samples when temporal sampling is enabled
        (one row per frame). Near-duplicate frames are skipped, never stored.
        """
        try:
            event = self.client.get_event(event_id)
            if self.config.frigate.completed_events_only and not event.completed:
                raise ValueError(
                    f"event not completed (end_time is None); refusing in-progress snapshot"
                )

            crop = self.config.collection.region_crop
            if crop and not self._has_sane_box(event.box):
                return EventOutcome(status="skip")
            crop_height = self.config.collection.region_crop_height or self.config.training.image_size

            stored = 0
            skipped = 0
            annotations = 0
            merged = 0
            for frame_index, frame_ts in enumerate(self._frame_times(event)):
                sample_id = str(uuid.uuid4())
                image_path = self._image_path(event.camera, event.start_time, sample_id)
                if crop:
                    self.client.download_region_crop(
                        event_id, image_path, height=crop_height, timestamp=frame_ts
                    )
                else:
                    self.client.download_clean_snapshot(event_id, image_path, timestamp=frame_ts)
                image_hash = self._hash_file(image_path)

                if self._is_duplicate_frame(event.camera, image_hash, image_path):
                    image_path.unlink(missing_ok=True)
                    if crop:
                        skipped += 1
                        debug("skipped duplicate crop", event_id=event.id, camera=event.camera)
                        continue
                    target = self._find_exact_duplicate_sample(event.camera, image_hash)
                    if target is None or not self._merge_event_frame(event, target):
                        skipped += 1
                        debug("skipped duplicate frame", event_id=event.id, camera=event.camera)
                        continue
                    merged += 1
                    debug("merged same-frame event", event_id=event.id, sample_id=target.id)
                    continue

                debug_path = None
                if self.config.collection.keep_annotated_snapshots:
                    debug_path = image_path.with_name(f"{sample_id}-debug.jpg")
                    if crop:
                        self.client.download_annotated_crop(event_id, debug_path)
                    else:
                        self.client.download_event_snapshot(event_id, debug_path)

                phash = None
                if self.config.collection.dedup_enabled:
                    try:
                        phash = dhash_file(image_path)
                    except Exception as exc:
                        debug("phash failed", event_id=event.id, error=str(exc))

                box = event.box
                sample = Sample(
                    id=sample_id,
                    camera=event.camera,
                    timestamp=frame_ts if frame_ts is not None else event.start_time,
                    event_id=event.id,
                    frame_index=frame_index,
                    image_path=str(image_path),
                    debug_image_path=str(debug_path) if debug_path else None,
                    image_hash=image_hash,
                    perceptual_hash=phash,
                    source="frigate",
                    frigate_label=event.label,
                    frigate_score=event.score if event.score is not None else event.top_score,
                    frigate_x1=box[0] if box else None,
                    frigate_y1=box[1] if box else None,
                    frigate_x2=box[2] if box else None,
                    frigate_y2=box[3] if box else None,
                    status="collected",
                )
                with self.db.session() as session:
                    session.add(sample)
                    session.flush()  # persist the sample before its FK-dependent annotation
                    if not crop:
                        session.add(
                            Annotation(
                                id=str(uuid.uuid4()),
                                sample_id=sample_id,
                                event_id=event.id,
                                source="frigate",
                                label=event.label,
                                x1=sample.frigate_x1,
                                y1=sample.frigate_y1,
                                x2=sample.frigate_x2,
                                y2=sample.frigate_y2,
                                confidence=sample.frigate_score,
                                verified=0,
                            )
                        )
                        annotations += 1
                    session.commit()
                info(
                    "collected sample",
                    camera=event.camera,
                    event_id=event.id,
                    frame_index=frame_index,
                    sample_id=sample_id,
                    label=event.label,
                    crop=crop,
                )
                stored += 1

            if stored:
                return EventOutcome(status="ok", stored=stored, skipped=skipped,
                                    annotations=annotations, merged=merged)
            if merged:
                return EventOutcome(status="merged", skipped=skipped,
                                    annotations=annotations, merged=merged)
            return EventOutcome(status="dup", stored=0, skipped=skipped)
        except FrigateAPIError as exc:
            if exc.status_code == 404:
                # event/review pruned by Frigate while we were collecting
                warning("event unreachable", event_id=event_id, status="404")
            else:
                error("event failed (api)", event_id=event_id, error=str(exc))
            self._record_failure(event_id, f"{exc}")
            return EventOutcome(status="fail", error=str(exc))
        except IntegrityError as exc:
            # concurrent duplicate insert from a second run in the same window
            error("event duplicate insert", event_id=event_id, error=str(exc))
            return EventOutcome(status="dup", error=str(exc))
        except Exception as exc:
            error("event failed", event_id=event_id, error=str(exc))
            self._record_failure(event_id, f"{type(exc).__name__}: {exc}")
            return EventOutcome(status="fail", error=str(exc))

    # --- storage helpers --------------------------------------------------

    def images_dir(self) -> Path:
        return self.config.images_dir()

    def _image_path(self, camera: str, timestamp: float, sample_id: str) -> Path:
        day = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y%m%d")
        return self.images_dir() / day / camera / f"{sample_id}.jpg"

    @staticmethod
    def _hash_file(path: Path) -> str:
        hasher = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _record_failure(self, event_id: str, error_text: str) -> None:
        try:
            with self.db.session() as session:
                session.add(
                    CollectionFailure(
                        id=str(uuid.uuid4()),
                        event_id=event_id,
                        error=error_text[:2000],
                    )
                )
                session.commit()
        except Exception:  # failure ledger must never crash collection
            logger.exception("could not record collection failure")

    def _record_job(self, job_id: str, type: str, status: str, started_at: str) -> None:
        with self.db.session() as session:
            session.add(Job(id=job_id, type=type, status=status, started_at=started_at))
            session.commit()

    def _finish_job(self, job_id: str, summary: CollectSummary) -> None:
        import json

        metadata = json.dumps(
            {
                "reviews_found": summary.reviews_found,
                "events_found": summary.events_found,
                "new_samples": summary.new_samples,
                "duplicates": summary.duplicate_samples,
                "failures": summary.failures,
                "skipped_no_box": summary.skipped_no_box,
                "range_from": summary.range_from,
                "range_to": summary.range_to,
                "cameras": summary.cameras,
                "labels": summary.labels,
                "severity": summary.severity,
            },
            sort_keys=True,
        )
        with self.db.session() as session:
            job = session.get(Job, job_id)
            if job:
                try:
                    current = json.loads(job.metadata_json) if job.metadata_json else {}
                except (ValueError, TypeError):
                    current = {}
                metadata_obj = json.loads(metadata)
                if isinstance(current, dict):
                    tail = current.get("log_tail")
                    if isinstance(tail, list):
                        metadata_obj["log_tail"] = tail
                job.status = "finished"
                job.finished_at = utcnow()
                job.metadata_json = json.dumps(metadata_obj, sort_keys=True)
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


__all__ = ["Collector", "CollectSummary"]