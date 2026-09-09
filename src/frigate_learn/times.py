"""Time-argument parsing for the CLI.

Turns values like ``"7 days ago"``, ``"24 hours ago"``, ``"2026-09-01"`` or
ISO-8601 datetimes into unix timestamps. Naive datetimes are interpreted in the
host's local timezone (the deployment is fixed to America/Edmonton).
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

_RELATIVE = re.compile(
    r"^\s*(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|weeks?|h|m|d|w)\s+ago\s*$",
    re.IGNORECASE,
)

_UNITS: dict[str, timedelta] = {
    "second": timedelta(seconds=1),
    "seconds": timedelta(seconds=1),
    "sec": timedelta(seconds=1),
    "secs": timedelta(seconds=1),
    "minute": timedelta(minutes=1),
    "minutes": timedelta(minutes=1),
    "min": timedelta(minutes=1),
    "mins": timedelta(minutes=1),
    "hour": timedelta(hours=1),
    "hours": timedelta(hours=1),
    "hr": timedelta(hours=1),
    "hrs": timedelta(hours=1),
    "day": timedelta(days=1),
    "days": timedelta(days=1),
    "week": timedelta(weeks=1),
    "weeks": timedelta(weeks=1),
    "h": timedelta(hours=1),
    "m": timedelta(minutes=1),
    "d": timedelta(days=1),
    "w": timedelta(weeks=1),
}


class TimeArgumentError(ValueError):
    """Raised when a time argument cannot be parsed."""


def _local_now() -> datetime:
    return datetime.now().astimezone()


def parse_time_arg(value: str) -> float:
    """Parse a CLI time argument into a unix timestamp (float)."""
    value = (value or "").strip()
    if not value:
        raise TimeArgumentError("empty time argument")

    match = _RELATIVE.match(value)
    if match:
        num = float(match.group("num"))
        unit = match.group("unit").lower()
        base = _UNITS.get(unit)
        if base is None:
            unit = unit.rstrip("s")
            base = _UNITS.get(unit)
        if base is None:
            raise TimeArgumentError(f"unrecognized relative unit in {value!r}")
        return (_local_now() - base * num).timestamp()

    # date-only: YYYY-MM-DD -> local midnight
    try:
        d = datetime.fromisoformat(value)
        return d.timestamp()
    except ValueError:
        pass

    # datetime with a space instead of 'T', and optional empty-time parts:
    try:
        d = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
        return d.timestamp()
    except ValueError:
        pass

    try:
        d = datetime.strptime(value, "%Y-%m-%d %H:%M")
        return d.timestamp()
    except ValueError:
        pass

    raise TimeArgumentError(f"unrecognized time argument: {value!r}")


def parse_time_arg_or(value: str | None, default_ts: float) -> float:
    if value is None:
        return default_ts
    return parse_time_arg(value)