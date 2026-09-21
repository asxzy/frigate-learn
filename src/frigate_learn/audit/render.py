"""Reconciliation image renderer.

Produces a single visual summary of the SAM+VLM audit for one Frigate crop.
The VLM sees this image; the human reviewer can also inspect it.

Visual layout:
* Original crop
* Green box = Frigate bbox
* Blue box = SAM refined bbox
* Translucent red overlay = SAM mask
* Top-left legend panel with class labels and deltas
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .geometry import box_edge_deltas
from .types import BoundingBox, Mask, SamResult

_BOX_WIDTH = 3
_MASK_ALPHA = 100
_LEGEND_BG = (0, 0, 0, 200)
_LEGEND_FG = (255, 255, 255)
_FRIGATE_COLOR = (0, 255, 0)
_SAM_COLOR = (50, 120, 255)
_MASK_COLOR = (255, 80, 80)
_LEGEND_MARGIN = 8


def _try_font(size: int):
    """Best-effort monospace font; falls back to default if unavailable."""
    candidates = [
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Monaco.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    ]
    import os
    for path in candidates:
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                pass
    return ImageFont.load_default()


def _overlay_mask(base: Image.Image, mask: Mask) -> Image.Image:
    """Translucent mask overlay without modifying the original."""
    rgba = np.zeros((mask.height, mask.width, 4), dtype=np.uint8)
    rgba[mask.data, 0] = _MASK_COLOR[0]
    rgba[mask.data, 1] = _MASK_COLOR[1]
    rgba[mask.data, 2] = _MASK_COLOR[2]
    rgba[mask.data, 3] = _MASK_ALPHA
    if (mask.height, mask.width) != base.size:
        rgba_pil = Image.fromarray(rgba, "RGBA").resize(base.size, Image.NEAREST)
    else:
        rgba_pil = Image.fromarray(rgba, "RGBA")
    return Image.alpha_composite(base.convert("RGBA"), rgba_pil)


def _draw_box(draw: ImageDraw.ImageDraw, box: BoundingBox, color: tuple, width: int = _BOX_WIDTH) -> None:
    draw.rectangle([box.x1, box.y1, box.x2, box.y2], outline=color + (255,), width=width)


def _draw_legend(
    draw: ImageDraw.ImageDraw,
    frigate_label: str,
    sam_label: str,
    deltas: dict[str, float],
    font,
) -> None:
    lines = [
        f"Frigate: {frigate_label}",
        f"SAM:     {sam_label}",
        "edge deltas (px):",
        f"  L={deltas['left_delta']:+.1f}  T={deltas['top_delta']:+.1f}",
        f"  R={deltas['right_delta']:+.1f}  B={deltas['bottom_delta']:+.1f}",
    ]
    pad = _LEGEND_MARGIN
    max_w = max(font.getlength(ln) for ln in lines) + pad * 2
    line_h = font.getmetrics()[1] + 4
    total_h = line_h * len(lines) + pad * 2
    bg = Image.new("RGBA", (int(max_w), int(total_h)), _LEGEND_BG)
    bg_draw = ImageDraw.Draw(bg)
    y = pad
    for line in lines:
        bg_draw.text((pad, y), line, fill=_LEGEND_FG + (255,), font=font)
        y += line_h
    return bg, bg.width, bg.height


def render_reconciliation_image(
    crop: Image.Image,
    frigate_bbox: BoundingBox,
    sam_result: SamResult | None,
    frigate_class: str,
    sam_class: str | None = None,
) -> Image.Image:
    """Render the single-object reconciliation image (crop space RGBA).

    The original crop is never modified; a new RGBA image is returned.
    When ``sam_result`` is None (SAM failed), only the Frigate box is drawn.
    """
    base = crop.convert("RGBA")
    if sam_result is not None and sam_result.mask.area > 0:
        base = _overlay_mask(base, sam_result.mask)
    draw = ImageDraw.Draw(base)
    _draw_box(draw, frigate_bbox, _FRIGATE_COLOR)
    if sam_result is not None:
        _draw_box(draw, sam_result.bbox, _SAM_COLOR)
    deltas = box_edge_deltas(frigate_bbox, sam_result.bbox) if sam_result else {}
    if not deltas:
        deltas = {k: 0.0 for k in ("left_delta", "top_delta", "right_delta", "bottom_delta")}
    font = _try_font(14)
    legend, _lw, _lh = _draw_legend(draw, frigate_class, sam_class or "N/A", deltas, font)
    base.paste(legend, (_LEGEND_MARGIN, _LEGEND_MARGIN), legend)
    return base


def render_vlm_input_image(
    crop: Image.Image,
    frigate_bbox: BoundingBox,
) -> Image.Image:
    """Render the blind VLM input: crop with ONLY the Frigate box drawn.

    No SAM box, no mask, no legend, and no class text — the VLM must judge
    the crop on its own, with nothing revealing the expected class.
    """
    base = crop.convert("RGBA")
    draw = ImageDraw.Draw(base)
    _draw_box(draw, frigate_bbox, _FRIGATE_COLOR)
    return base


__all__ = ["render_reconciliation_image", "render_vlm_input_image"]
