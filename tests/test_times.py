"""Time argument parsing tests."""

from __future__ import annotations

import time
from datetime import datetime, timedelta

import pytest

from frigate_learn.times import TimeArgumentError, parse_time_arg


def within(expected: float, got: float, tolerance: float = 2) -> bool:
    return abs(expected - got) <= tolerance


def test_relative_days():
    now = time.time()
    got = parse_time_arg("7 days ago")
    assert within(now - 7 * 86400, got)


def test_relative_hours_and_minutes():
    now = time.time()
    assert within(now - 24 * 3600, parse_time_arg("24 hours ago"))
    assert within(now - 90 * 60, parse_time_arg("90 minutes ago"))
    assert within(now - 2 * 3600, parse_time_arg("2h ago"))


def test_relative_singular_units():
    now = time.time()
    assert within(now - 3600, parse_time_arg("1 hour ago"))
    assert within(now - 86400, parse_time_arg("1 day ago"))
    assert within(now - 7 * 86400, parse_time_arg("1 week ago"))


def test_iso_date_parses():
    got = parse_time_arg("2026-01-01")
    expected = datetime(2026, 1, 1, 0, 0).timestamp()
    assert got == expected


def test_datetime_forms():
    assert parse_time_arg("2026-01-01 12:00:00") == datetime(2026, 1, 1, 12, 0, 0).timestamp()
    assert parse_time_arg("2026-01-01 12:30") == datetime(2026, 1, 1, 12, 30).timestamp()
    assert parse_time_arg("2026-01-01T12:30:00") == datetime(2026, 1, 1, 12, 30).timestamp()


@pytest.mark.parametrize(
    "bad",
    ["", "yesterday", "7 days", "in 2 days", "7 microfortnights ago", "abc", "2026-13-40"],
)
def test_invalid_raises(bad):
    with pytest.raises(TimeArgumentError):
        parse_time_arg(bad)