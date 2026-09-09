"""Deterministic split tests."""

from __future__ import annotations

import pytest

from frigate_learn.dataset.splits import VALID_SPLITS, assign_split, make_splits


def test_valid_split_returned():
    for i in range(100):
        assert assign_split(f"sample-{i}") in VALID_SPLITS


def test_deterministic():
    assert assign_split("abc") == assign_split("abc")
    ids = [f"s{i}" for i in range(50)]
    assert make_splits(ids, seed="x") == make_splits(ids, seed="x")


def test_seed_changes_assignments():
    counts_a = _count(5000, seed="seed-a")
    counts_b = _count(5000, seed="seed-b")
    assert counts_a != counts_b


def _count(n: int, seed: str) -> dict[str, int]:
    out = {"train": 0, "val": 0, "test": 0}
    for i in range(n):
        out[assign_split(f"id-{i}", seed=seed)] += 1
    return out


def test_ratios_approximate():
    counts = _count(20000, seed="ratio-check")
    total = sum(counts.values())
    train = counts["train"] / total
    test = counts["test"] / total
    assert 0.79 <= train <= 0.81
    assert 0.09 <= test <= 0.11


def test_make_splits_partitions_all():
    n = 200
    ids = [f"p{i}" for i in range(n)]
    splits = make_splits(ids)
    flattened = [i for v in splits.values() for i in v]
    assert sorted(flattened) == sorted(ids)


def test_custom_ratios_reject_impossible():
    with pytest.raises(ValueError):
        make_splits(["a"], ratios={"train": 0.0, "val": 0.0, "test": 0.0})