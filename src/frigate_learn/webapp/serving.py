"""Image serving for the webapp.

Pure path resolution: ``resolve_image`` returns the on-disk original for a
sample, ``resolve_thumb`` lazily builds a width-capped JPEG thumbnail into the
previews dir. No HTTP; routes in :mod:`frigate_learn.webapp.api` wire these up.
Nothing is ever fetched from the store.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from ..config import AppConfig
from ..db import Database
from ..models import Sample

THUMB_MAX_WIDTH = 480


def resolve_image(config: AppConfig, db: Database, sample_id: str) -> Path | None:
    """Return the original image path for a sample, or None if unavailable."""
    with db.session() as session:
        sample = session.get(Sample, sample_id)
    if sample is None or not sample.image_path:
        return None
    p = Path(sample.image_path).resolve()
    images_dir = config.images_dir().resolve()
    data_root = config.resolve(config.data.root).resolve()
    if not (p.is_relative_to(images_dir) or p.is_relative_to(data_root)):
        return None
    return p if p.is_file() else None


def resolve_thumb(config: AppConfig, db: Database, sample_id: str) -> Path | None:
    """Return the thumb path for a sample, building it lazily from disk."""
    thumb = config.previews_dir() / "thumbs" / f"{sample_id}.jpg"
    original = resolve_image(config, db, sample_id)
    if original is None:
        return None
    if thumb.is_file():
        return thumb
    thumb.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(original) as img:
            img = img.convert("RGB")
            scale = THUMB_MAX_WIDTH / max(1, img.width)
            if scale < 1.0:
                img = img.resize(
                    (int(img.width * scale), int(img.height * scale)),
                    Image.Resampling.LANCZOS,
                )
            img.save(thumb, quality=82)
    except Exception:
        return None
    return thumb if thumb.is_file() else None


__all__ = ["resolve_image", "resolve_thumb", "THUMB_MAX_WIDTH"]