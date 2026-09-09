"""Temporal sampling within events (Phase 3)."""

from __future__ import annotations

from frigate_learn.collection.sampling import sample_timestamps


def test_single_sample_when_disabled_or_short():
    assert sample_timestamps(100, 120, 1) == [100]
    assert sample_timestamps(100, 100, 3) == [100]
    assert sample_timestamps(100, None, 3) == [100]


def test_two_samples_span_window():
    assert sample_timestamps(100, 120, 2) == [100, 120]


def test_even_spacing_with_gap_respected():
    times = sample_timestamps(100, 200, 5, min_gap=10)
    assert times[0] == 100
    assert times[-1] == 200
    assert len(times) == 5
    gaps = [b - a for a, b in zip(times, times[1:])]
    assert all(g >= 9.0 for g in gaps)
    assert all(100 <= t <= 200 for t in times)


def test_max_feasible_limited_by_min_gap():
    times = sample_timestamps(100, 106, 10, min_gap=5)
    assert len(times) <= 3  # 0, 5, ... -> floor(6/5)+1 = 2, so 2 samples
    assert times[0] == 100


def test_ascending_and_unique():
    times = sample_timestamps(1234.0, 1294.0, 7, min_gap=3)
    assert times == sorted(set(times))