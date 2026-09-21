"""Structured logging helpers.

Python's stdlib logging doesn't support key=value context natively, so we wrap
it with `key=value` suffixes. This keeps the log lines greppable and
machine-parseable while staying dependency-free.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import sys
import threading
from typing import Any

_logger = logging.getLogger("frigate_learn")


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    _logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    handler.setLevel(level)
    _logger.handlers = [handler]
    _logger.propagate = False
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except (AttributeError, ValueError, OSError):
            pass


def write_job_log_tail(db_path: str, job_id: str, tail: list[str]) -> None:
    """Persist a bounded log tail into a jobs-row metadata_json (sqlite)."""
    con = sqlite3.connect(db_path, timeout=5)
    try:
        row = con.execute(
            "SELECT metadata_json FROM jobs WHERE id=?", (job_id,)
        ).fetchone()
        if row is None or not row[0]:
            return
        try:
            meta = json.loads(row[0])
        except (ValueError, TypeError):
            meta = {}
        if not isinstance(meta, dict):
            meta = {}
        meta["log_tail"] = tail
        con.execute(
            "UPDATE jobs SET metadata_json=? WHERE id=?",
            (json.dumps(meta), job_id),
        )
        con.commit()
    finally:
        con.close()


def job_metadata_log_tail(meta: dict) -> list[str]:
    tail = meta.get("log_tail") if isinstance(meta, dict) else None
    return tail if isinstance(tail, list) else []


class JobLogTailHandler(logging.Handler):
    """Attach to the ``frigate_learn`` logger to stream records into a job row.

    Used by CLI commands (collect/verify/audit) that run outside the webapp's
    JobManager: each record is appended to the jobs row's ``log_tail`` in
    ``metadata_json`` (bounded), so the dashboard shows a live log tail and a
    bounded history after the job finishes.
    """

    def __init__(self, db_path: str, job_id: str, cap: int = 200) -> None:
        super().__init__(level=logging.DEBUG)
        self.db_path = db_path
        self.job_id = job_id
        self.cap = cap
        self._tail: list[str] = []
        self._lock = threading.Lock()
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))

    def emit(self, record) -> None:
        try:
            line = self.format(record)
            with self._lock:
                self._tail.append(line)
                if len(self._tail) > self.cap:
                    del self._tail[: len(self._tail) - self.cap]
                tail = list(self._tail)
            write_job_log_tail(self.db_path, self.job_id, tail)
        except Exception:
            pass


def _fmt(message: str, **kw: Any) -> str:
    if not kw:
        return message
    tail = " ".join(f"{k}={v}" for k, v in kw.items())
    return f"{message} {tail}"


def info(message: str, **kw: Any) -> None:
    _logger.info(_fmt(message, **kw))


def debug(message: str, **kw: Any) -> None:
    _logger.debug(_fmt(message, **kw))


def warning(message: str, **kw: Any) -> None:
    _logger.warning(_fmt(message, **kw))


def error(message: str, **kw: Any) -> None:
    _logger.error(_fmt(message, **kw))


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)