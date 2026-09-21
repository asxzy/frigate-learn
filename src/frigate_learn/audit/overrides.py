"""Human override store for audit decisions.

A JSON file (``overrides.json``) in the audit output root holds manual
KEEP/DROP overrides keyed by sample id. The pipeline applies them when
writing decisions; the webapp reads them for the review UI and lets the
reviewer add or remove them.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

OVERRIDES_FILE = "overrides.json"
VALID_STATUSES = ("KEEP", "DROP")


class OverrideError(ValueError):
    """Raised on invalid override input."""


def _path(audit_root: Path | str) -> Path:
    return Path(audit_root) / OVERRIDES_FILE


def load_overrides(audit_root: Path | str) -> dict[str, dict[str, object]]:
    try:
        data = json.loads(_path(audit_root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    cleaned: dict[str, dict[str, object]] = {}
    for sample_id, item in data.items():
        if not isinstance(sample_id, str) or not isinstance(item, dict):
            continue
        if item.get("status") not in VALID_STATUSES:
            continue
        cleaned[sample_id] = {
            "status": item["status"],
            "note": str(item.get("note") or ""),
            "updated_at": item.get("updated_at"),
        }
    return cleaned


def save_override(
    audit_root: Path | str,
    sample_id: str,
    status: str,
    note: str = "",
) -> dict[str, object]:
    if not _valid_sample_id(sample_id):
        raise OverrideError("invalid sample id")
    if status not in VALID_STATUSES:
        raise OverrideError("status must be KEEP or DROP")
    overrides = load_overrides(audit_root)
    entry: dict[str, object] = {
        "status": status,
        "note": note.strip() if note else "",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    overrides[sample_id] = entry
    _write(audit_root, overrides)
    return entry


def remove_override(audit_root: Path | str, sample_id: str) -> bool:
    overrides = load_overrides(audit_root)
    if sample_id not in overrides:
        return False
    del overrides[sample_id]
    _write(audit_root, overrides)
    return True


def _valid_sample_id(sample_id: str) -> bool:
    return bool(sample_id) and sample_id not in (".", "..") and "/" not in sample_id and "\\" not in sample_id


def _write(audit_root: Path | str, overrides: dict) -> None:
    path = _path(audit_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(overrides, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp.replace(path)


__all__ = [
    "OverrideError",
    "OVERRIDES_FILE",
    "load_overrides",
    "remove_override",
    "save_override",
]