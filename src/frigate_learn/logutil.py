"""Structured logging helpers.

Python's stdlib logging doesn't support key=value context natively, so we wrap
it with `key=value` suffixes. This keeps the log lines greppable and
machine-parseable while staying dependency-free.
"""

from __future__ import annotations

import logging
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