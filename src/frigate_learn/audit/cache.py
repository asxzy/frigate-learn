"""Caching for expensive per-sample artifacts.

Every cached artifact lives under ``<audit>/<sample_id>/`` and is validated
against a stable sample hash derived from:

* decoded, resized image content (EXIF/JPEG metadata do not matter)
* the Frigate bbox and class
* the SAM model key, VLM model key, and pipeline version

If any relevant input/model version changes, the corresponding cache entry
is invalidated and recomputed. A rerun never recomputes an unchanged sample.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .types import BoundingBox, Decision, SamResult, VlmResult

CONTENT_HASH_MAX_SIDE = 512


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def content_hash(image: np.ndarray) -> str:
    """Stable digest of decoded image *content*.

    The RGB array is downscaled to a bounded max side so that reruns are
    stable and cheap; only pixels participate (no EXIF, no file metadata).
    """
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    h, w = arr.shape[:2]
    long_side = max(h, w)
    if long_side > CONTENT_HASH_MAX_SIDE:
        scale = CONTENT_HASH_MAX_SIDE / long_side
        img = Image.fromarray(arr).resize(
            (max(1, int(w * scale)), max(1, int(h * scale))),
            Image.BILINEAR,
        )
        arr = np.asarray(img)
    return _sha256_hex(arr.tobytes())


def sample_hash(
    image_hash: str,
    frigate_class: str,
    frigate_bbox: BoundingBox,
    sam_key: str,
    vlm_key: str,
    pipeline_version: int,
) -> str:
    """Stable hash of everything that affects a sample decision."""
    bbox_json = json.dumps(frigate_bbox.to_list(), separators=(',', ':'))
    parts = [
        f"img:{image_hash}",
        f"class:{frigate_class}",
        f"bbox:{bbox_json}",
        f"sam:{sam_key}",
        f"vlm:{vlm_key}",
        f"pipeline:{pipeline_version}",
    ]
    return _sha256_hex("|".join(parts).encode("utf-8"))


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + chr(10),
        encoding="utf-8",
    )


def _meta_ok(meta: dict | None, *, expected_hash: str | None = None, keys: dict | None = None) -> bool:
    if not isinstance(meta, dict):
        return False
    if expected_hash is not None and meta.get("hash") != expected_hash:
        return False
    for key, value in (keys or {}).items():
        if meta.get(key) != value:
            return False
    return True


def sam_result_to_dict(result: SamResult, mask_path: str) -> dict:
    data = result.as_dict()
    data["mask_path"] = mask_path
    return data


def sam_result_from_dict(data: dict, mask_array: np.ndarray) -> SamResult:
    from .types import Mask

    bbox = BoundingBox(*data["bbox"])
    return SamResult(
        class_name=data["class_name"],
        bbox=bbox,
        mask=Mask(mask_array),
        confidence=data.get("confidence"),
        raw_metadata=data.get("raw_metadata") or {},
    )


def load_sam_cache(
    sample_dir: Path,
    expected_hash: str,
    sam_key: str,
) -> tuple[SamResult, dict] | None:
    """Return cached (result, meta) when valid, else None."""
    entry = read_json(sample_dir / "sam.json")
    if entry is None or not _meta_ok(entry.get("meta"), expected_hash=expected_hash, keys={"sam_key": sam_key}):
        return None
    mask_path = sample_dir / (entry.get("result", {}).get("mask_path") or "mask.png")
    if not mask_path.is_file():
        return None
    try:
        mask_img = Image.open(mask_path)
        mask_array = np.asarray(mask_img.convert("L")) > 127
    except OSError:
        return None
    try:
        result = sam_result_from_dict(entry["result"], mask_array)
    except (KeyError, TypeError, ValueError):
        return None
    return result, entry.get("meta", {})


def save_sam_cache(
    sample_dir: Path,
    expected_hash: str,
    sam_key: str,
    result: SamResult,
) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    mask_img = Image.fromarray(result.mask.to_uint8(), mode="L")
    mask_img.save(sample_dir / "mask.png")
    payload = {
        "meta": {
            "hash": expected_hash,
            "sam_key": sam_key,
        },
        "result": sam_result_to_dict(result, "mask.png"),
    }
    write_json(sample_dir / "sam.json", payload)


def reconciliation_is_cached(sample_dir: Path, expected_hash: str, sam_key: str) -> bool:
    if not (sample_dir / "reconciliation.png").is_file():
        return False
    meta = read_json(sample_dir / "reconciliation.json")
    return meta is not None and _meta_ok(meta, expected_hash=expected_hash, keys={"sam_key": sam_key})


def save_reconciliation_cache(sample_dir: Path, expected_hash: str, sam_key: str, image) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    image.save(sample_dir / "reconciliation.png")
    write_json(
        sample_dir / "reconciliation.json",
        {"meta": {"hash": expected_hash, "sam_key": sam_key}},
    )


def load_vlm_cache(
    sample_dir: Path,
    expected_hash: str,
    vlm_key: str,
) -> VlmResult | None:
    entry = read_json(sample_dir / "vlm.json")
    if entry is None or not _meta_ok(entry.get("meta"), expected_hash=expected_hash, keys={"vlm_key": vlm_key}):
        return None
    result = entry.get("result")
    if not isinstance(result, dict):
        return None
    try:
        return VlmResult(
            bbox_covers_object=bool(result["bbox_covers_object"]),
            class_label=str(result.get("class_label") or ""),
            raw=result,
        )
    except KeyError:
        return None


def save_vlm_cache(
    sample_dir: Path,
    expected_hash: str,
    vlm_key: str,
    result: VlmResult,
) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {"hash": expected_hash, "vlm_key": vlm_key},
        "result": result.as_dict(),
    }
    write_json(sample_dir / "vlm.json", payload)


def vlm_input_is_cached(sample_dir: Path, expected_hash: str, sam_key: str) -> bool:
    if not (sample_dir / "vlm.png").is_file():
        return False
    meta = read_json(sample_dir / "vlm_input.json")
    return meta is not None and _meta_ok(meta, expected_hash=expected_hash, keys={"sam_key": sam_key})


def save_vlm_input_cache(sample_dir: Path, expected_hash: str, sam_key: str, image) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    image.save(sample_dir / "vlm.png")
    write_json(
        sample_dir / "vlm_input.json",
        {"meta": {"hash": expected_hash, "sam_key": sam_key}},
    )


def load_decision(
    sample_dir: Path,
    expected_hash: str,
    sam_key: str,
    vlm_key: str,
    pipeline_version: int,
) -> dict | None:
    """Final decision (KEEP/DROP) only; PENDING is never treated as final."""
    entry = read_json(sample_dir / "decision.json")
    if entry is None or not _meta_ok(
        entry.get("meta"),
        expected_hash=expected_hash,
        keys={"sam_key": sam_key, "vlm_key": vlm_key, "pipeline_version": pipeline_version},
    ):
        return None
    status = entry.get("decision", {}).get("status")
    if status not in ("KEEP", "DROP"):
        return None
    return entry


def save_decision(
    sample_dir: Path,
    expected_hash: str,
    sam_key: str,
    vlm_key: str,
    pipeline_version: int,
    decision: Decision,
    provenance: dict[str, Any],
) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "hash": expected_hash,
            "sam_key": sam_key,
            "vlm_key": vlm_key,
            "pipeline_version": pipeline_version,
        },
        "decision": decision.as_dict(),
        "provenance": provenance,
    }
    write_json(sample_dir / "decision.json", payload)


__all__ = [
    "content_hash",
    "load_decision",
    "load_sam_cache",
    "load_vlm_cache",
    "read_json",
    "reconciliation_is_cached",
    "sample_hash",
    "save_decision",
    "save_reconciliation_cache",
    "save_sam_cache",
    "save_vlm_cache",
    "write_json",
]
