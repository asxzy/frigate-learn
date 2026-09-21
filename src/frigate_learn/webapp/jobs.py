"""Background job runner for the webapp.

Runs ``run_pipeline`` stages in a daemon thread so HTTP handlers never block,
records progress in the ``jobs`` table, captures the pipeline log tail, and
recovers stale ``running`` rows left by a crashed server. One concurrent job at
a time.
"""

from __future__ import annotations

import json
import logging
import threading
from uuid import uuid4

from sqlalchemy import update

from ..db import Database
from ..models import Job, utcnow
from ..run import PIPELINE, StepReport, run_pipeline

_LOG_TAIL_CAP = 200
_RECENT_TAIL = 50
_STALE_ERROR = "terminated by server restart"


class JobRunningError(Exception):
    """Raised when ``start`` is called while another job is already running."""


class _TailHandler(logging.Handler):
    def __init__(self, append) -> None:
        super().__init__(level=logging.DEBUG)
        self._append = append
        self.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        )

    def emit(self, record) -> None:
        try:
            self._append(self.format(record))
        except Exception:
            pass


class JobManager:
    def __init__(self, config, db: Database) -> None:
        self.config = config
        self.db = db
        self.db.migrate()
        self._thread: threading.Thread | None = None
        self._job_id: str | None = None
        self._tail: list[str] = []
        self._start_lock = threading.Lock()
        self._tail_lock = threading.Lock()
        self.mark_stale_running()

    def mark_stale_running(self) -> None:
        with self.db.session() as session:
            session.execute(
                update(Job)
                .where(Job.status == "running")
                .where(Job.type.in_(("pipeline", "audit")))
                .values(status="failed", error=_STALE_ERROR)
            )
            session.commit()

    def start_audit(
        self,
        *,
        resume: bool = False,
        limit: int | None = None,
        sam_only: bool = False,
    ) -> str:
        if limit is not None and limit < 1:
            raise ValueError("limit must be >= 1")
        with self._start_lock:
            if (self._thread and self._thread.is_alive()) or self._running_job_id():
                raise JobRunningError("a job is already running")
            job_id = uuid4().hex[:12]
            with self.db.session() as session:
                session.add(
                    Job(id=job_id, type="audit", status="running", started_at=utcnow())
                )
                session.commit()
            self._job_id = job_id
            self._tail = []
            self._thread = threading.Thread(
                target=self._run_audit,
                args=(job_id, resume, limit, sam_only),
                daemon=True,
            )
            self._thread.start()
            return job_id

    def start(
        self,
        steps: list[str],
        *,
        dry_run: bool = False,
        days: int | None = None,
        limit: int | None = None,
        keep_going: bool = False,
    ) -> str:
        if not steps:
            raise ValueError("steps must not be empty")
        for step in steps:
            if step not in PIPELINE:
                raise ValueError(f"unknown step: {step}")
        with self._start_lock:
            if (self._thread and self._thread.is_alive()) or self._running_job_id():
                raise JobRunningError("a job is already running")
            job_id = uuid4().hex[:12]
            with self.db.session() as session:
                session.add(
                    Job(id=job_id, type="pipeline", status="running", started_at=utcnow())
                )
                session.commit()
            self._job_id = job_id
            self._tail = []
            self._thread = threading.Thread(
                target=self._run,
                args=(job_id, steps, dry_run, days, limit, keep_going),
                daemon=True,
            )
            self._thread.start()
            return job_id

    def _run(
        self,
        job_id: str,
        steps: list[str],
        dry_run: bool,
        days: int | None,
        limit: int | None,
        keep_going: bool,
    ) -> None:
        logger = logging.getLogger("frigate_learn")
        handler = _TailHandler(self._append_tail)
        logger.addHandler(handler)
        previous_level = logger.level
        logger.setLevel(logging.DEBUG)
        reports: list[StepReport] = []
        error: str | None = None
        try:
            logger.info(
                "job started steps=%s dry_run=%s days=%s limit=%s keep_going=%s",
                steps, dry_run, days, limit, keep_going,
            )
            try:
                reports = run_pipeline(
                    self.config,
                    self.db,
                    steps=steps,
                    dry_run=dry_run,
                    days=days,
                    limit=limit,
                    keep_going=keep_going,
                )
                if any(r.status == "failed" for r in reports):
                    error = next(
                        r.message for r in reports if r.status == "failed"
                    )
                status = "failed" if error is not None else "finished"
            except Exception as exc:
                status = "failed"
                error = str(exc)
            with self._tail_lock:
                tail = list(self._tail)
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous_level)
        with self.db.session() as session:
            job = session.get(Job, job_id)
            if job is None:
                return
            job.status = status
            job.error = error
            job.finished_at = utcnow()
            job.metadata_json = json.dumps(
                {
                    "steps": steps,
                    "dry_run": dry_run,
                    "days": days,
                    "limit": limit,
                    "keep_going": keep_going,
                    "reports": [_report_dict(r) for r in reports],
                    "log_tail": tail,
                }
            )
            session.commit()

    def _run_audit(
        self,
        job_id: str,
        resume: bool,
        limit: int | None,
        sam_only: bool,
    ) -> None:
        from ..audit.runner import AuditRunError, run_audit

        logger = logging.getLogger("frigate_learn")
        handler = _TailHandler(self._append_tail)
        logger.addHandler(handler)
        previous_level = logger.level
        logger.setLevel(logging.DEBUG)
        stats: dict = {}
        error: str | None = None
        try:
            logger.info(
                "job started type=audit resume=%s limit=%s sam_only=%s",
                resume, limit, sam_only,
            )
            try:
                stats = run_audit(
                    self.config,
                    resume=resume,
                    limit=limit,
                    sam_only=sam_only,
                )
                message = (
                    f"processed={stats.get('processed')} "
                    f"accepted={stats.get('accepted')} "
                    f"dropped={stats.get('dropped')} "
                    f"pending={stats.get('pending')}"
                )
                reports = [{"name": "audit", "status": "executed", "message": message}]
                status = "finished"
            except AuditRunError as exc:
                status = "failed"
                error = str(exc)
                reports = [{"name": "audit", "status": "failed", "message": str(exc)}]
            except Exception as exc:
                status = "failed"
                error = str(exc)
                reports = [{"name": "audit", "status": "failed", "message": str(exc)}]
            with self._tail_lock:
                tail = list(self._tail)
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous_level)
        with self.db.session() as session:
            job = session.get(Job, job_id)
            if job is None:
                return
            job.status = status
            job.error = error
            job.finished_at = utcnow()
            job.metadata_json = json.dumps(
                {
                    "kind": "audit",
                    "resume": resume,
                    "limit": limit,
                    "sam_only": sam_only,
                    "reports": reports,
                    "stats": stats,
                    "log_tail": tail,
                }
            )
            session.commit()

    def _append_tail(self, line: str) -> None:
        with self._tail_lock:
            self._tail.append(line)
            if len(self._tail) > _LOG_TAIL_CAP:
                del self._tail[: len(self._tail) - _LOG_TAIL_CAP]
            tail = list(self._tail)
        try:
            with self.db.session() as session:
                job = session.get(Job, self._job_id)
                if job is None:
                    return
                try:
                    meta = json.loads(job.metadata_json) if job.metadata_json else {}
                except (ValueError, TypeError):
                    meta = {}
                if not isinstance(meta, dict):
                    meta = {}
                meta["log_tail"] = tail
                job.metadata_json = json.dumps(meta)
                session.commit()
        except Exception:
            pass

    def current_job_id(self) -> str | None:
        if self._thread and self._thread.is_alive():
            return self._job_id
        return self._running_job_id()

    def _running_job_id(self) -> str | None:
        with self.db.session() as session:
            return (
                session.query(Job.id)
                .filter(Job.status == "running")
                .limit(1)
                .scalar()
            )

    def get(self, job_id: str) -> dict | None:
        with self.db.session() as session:
            job = session.get(Job, job_id)
            if job is None:
                return None
            return _job_item(job)

    def recent(self, limit: int = 20) -> list[dict]:
        limit = max(1, min(200, int(limit)))
        with self.db.session() as session:
            jobs = (
                session.query(Job)
                .order_by(Job.started_at.desc())
                .limit(limit)
                .all()
            )
        return [_job_item(j, truncate_tail=True) for j in jobs]


def _report_dict(report: StepReport) -> dict:
    return {"name": report.name, "status": report.status, "message": report.message}


def _parse_metadata(value: str | None) -> dict:
    if not value:
        return {"reports": [], "log_tail": []}
    try:
        data = json.loads(value)
    except ValueError:
        return {"reports": [], "log_tail": []}
    if not isinstance(data, dict):
        return {"reports": [], "log_tail": []}
    reports = data.get("reports")
    log_tail = data.get("log_tail")
    data["reports"] = reports if isinstance(reports, list) else []
    data["log_tail"] = log_tail if isinstance(log_tail, list) else []
    return data


def _job_item(job: Job, *, truncate_tail: bool = False) -> dict:
    metadata = _parse_metadata(job.metadata_json)
    log_tail = metadata["log_tail"]
    if truncate_tail:
        log_tail = log_tail[-_RECENT_TAIL:]
    return {
        "id": job.id,
        "type": job.type,
        "status": job.status,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "error": job.error,
        "days": metadata.get("days"),
        "limit": metadata.get("limit"),
        "keep_going": metadata.get("keep_going"),
        "resume": metadata.get("resume"),
        "sam_only": metadata.get("sam_only"),
        "reports": metadata["reports"],
        "log_tail": log_tail,
    }


__all__ = ["JobManager", "JobRunningError"]