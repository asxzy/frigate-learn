"""JSONL manifest reader/writer.

Manifests hold one metadata record per sample. They are *append-only by
sample_id*: writing a record whose ``sample_id`` already exists raises, so a
``golden`` manifest can never be silently overwritten.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


class ManifestDuplicateError(ValueError):
    """Raised when writing a record whose sample_id already exists."""


def read_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def iter_manifest(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def append_record(path: Path, record: dict[str, Any], *, allow_overwrite: bool = False) -> None:
    """Append a record, refusing to overwrite an existing ``sample_id``.

    When ``allow_overwrite=False`` (default) the full manifest is checked for the
    sample_id first. Manifests are small enough that this is acceptable and keeps
    the "never silently overwrite a golden sample" guarantee explicit.
    """
    sample_id = record.get("sample_id")
    if sample_id is None:
        raise ValueError("manifest record requires a 'sample_id'")
    if not allow_overwrite and _contains_sample(path, sample_id):
        raise ManifestDuplicateError(
            f"sample_id already present in manifest: {sample_id}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _contains_sample(path: Path, sample_id: str) -> bool:
    for record in iter_manifest(path):
        if record.get("sample_id") == sample_id:
            return True
    return False


__all__ = ["read_manifest", "iter_manifest", "append_record", "ManifestDuplicateError"]