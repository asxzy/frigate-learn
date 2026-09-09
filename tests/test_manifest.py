"""Manifest JSONL tests."""

from __future__ import annotations

import pytest

from frigate_learn.dataset.manifest import (
    ManifestDuplicateError,
    append_record,
    iter_manifest,
    read_manifest,
)


def test_append_and_read(tmp_path):
    path = tmp_path / "manifest.jsonl"
    append_record(path, {"sample_id": "a", "camera": "front"})
    append_record(path, {"sample_id": "b", "camera": "gate"})
    records = read_manifest(path)
    assert [r["sample_id"] for r in records] == ["a", "b"]
    assert [r["sample_id"] for r in iter_manifest(path)] == ["a", "b"]


def test_duplicate_raises(tmp_path):
    path = tmp_path / "manifest.jsonl"
    append_record(path, {"sample_id": "a"})
    with pytest.raises(ManifestDuplicateError):
        append_record(path, {"sample_id": "a", "camera": "gate"})


def test_allow_overwrite(tmp_path):
    path = tmp_path / "manifest.jsonl"
    append_record(path, {"sample_id": "a", "camera": "front"})
    append_record(path, {"sample_id": "a", "camera": "gate"}, allow_overwrite=True)
    records = read_manifest(path)
    assert records == [
        {"sample_id": "a", "camera": "front"},
        {"sample_id": "a", "camera": "gate"},
    ]


def test_missing_sample_id_raises(tmp_path):
    with pytest.raises(ValueError):
        append_record(tmp_path / "m.jsonl", {"camera": "front"})


def test_read_missing_file(tmp_path):
    assert read_manifest(tmp_path / "nope.jsonl") == []