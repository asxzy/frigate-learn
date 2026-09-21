"""SAM 3.1 teacher running through MLX on Apple Silicon.

The pipeline talks to :class:`SamTeacher` only; the concrete implementation
(:class:`MlxSam3Teacher`) wraps the ``sam3_mlx`` package (Meta SAM 3 / 3.1
ported to MLX). All sam3_mlx imports are lazy so the rest of the package
imports and unit-tests cleanly on machines without the extra installed.

Class handling
===============

The Frigate class is immutable; SAM is an *independent* observation. SAM 3.1
is open-vocabulary, so the teacher queries the model against a fixed
candidate set drawn exclusively from the Frigate vocabulary (never new
labels) and selects the best concept deterministically (see
``_select_best_candidate``). The selected class name may therefore differ
from the Frigate label — the decision engine treats that as a disagreement.

Deterministic candidate-selection strategy (documented)
=======================================================

1. Prompt order: ``[frigate_class, *others]`` where others follow the
   configured Frigate class order (config ``audit.models.sam.candidate_classes``,
   default: the full configured class list).
2. For every candidate, run SAM 3.1 grounding with the same positive box
   prompt (the crop-local Frigate bbox in normalized cxcywh).
3. Per candidate: keep detections with score >= confidence_threshold and a
   non-empty mask; rank by (score desc, mask area desc, index asc).
4. Across candidates: rank by (score desc, frigate-class priority asc,
   candidate-order asc). The Frigate class wins ties, which only matters for
   exactly-equal scores.
5. No candidate passing the filters => :class:`SamTeacherError` (SAM failure
   => sample dropped).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from PIL import Image

from .types import BoundingBox, Mask, SamCandidate, SamResult


class SamTeacherError(RuntimeError):
    """SAM failed to produce a usable result for this sample.

    ``kind`` distinguishes environmental failures (backend ``import`` missing)
    from model failures (``run``: weights loaded but no candidate qualified),
    so ``--resume`` can refresh stale ``sam_failure`` decisions once the
    backend becomes available.
    """

    def __init__(self, message: str, *, kind: str = "run") -> None:
        super().__init__(message)
        self.kind = kind


def is_sam_available() -> bool:
    """True when the ``sam3_mlx`` backend is importable on this machine."""
    try:
        import sam3_mlx  # noqa: PLC0415
    except ImportError:
        return False
    return True


class SamTeacher(Protocol):
    """Minimal interface the audit pipeline depends on."""

    model_key: str

    def predict(
        self,
        image: Image.Image,
        frigate_class: str,
        frigate_bbox: BoundingBox,
    ) -> SamResult:
        ...


@dataclass
class MlxSam3Teacher:
    """SAM 3.1 teacher backed by ``sam3_mlx`` (MLX, Apple Silicon).

    Configuration:

    * ``model_key`` — stable identifier mixed into the cache hash
      (``sam3-mlx:<version>:<repo-or-checkpoint>:<resolution>``); derived from
      the weight source, resolution and confidence threshold when not set
      explicitly, so config changes invalidate stale caches.
    * ``checkpoint_path`` / ``load_from_hf`` / ``hf_repo`` — weight source
      forwarded to ``sam3_mlx.build_sam3_image_model``.
    * ``resolution`` — ViT input size in pixels (multiple of 14; smaller =
      faster at the cost of detail).
    * ``confidence_threshold`` — per-detection score floor.
    * ``candidate_classes`` — the SAM hypothesis candidate set; empty means
      "use the full Frigate vocabulary" (passed by the pipeline).
    """

    model_key: str = ""
    version: str = "sam3.1"
    checkpoint_path: str | None = None
    load_from_hf: bool = True
    hf_repo: str = "mlx-community/sam3-image"
    resolution: int = 1008
    confidence_threshold: float = 0.5
    candidate_classes: list[str] = field(default_factory=list)
    device: str = "mlx"
    quantize_bits: int = 0          # 0 = keep precision; 4/8 = mx.quantize at load

    _model: Any = None
    _processor: Any = None

    def __post_init__(self) -> None:
        if not self.model_key:
            weights = self.checkpoint_path or (self.hf_repo if self.load_from_hf else "no-weights")
            candidates = ",".join(sorted(self.candidate_classes))
            self.model_key = (
                f"sam3-mlx:{self.version}:{weights}:q{self.quantize_bits}:{self.resolution}:"
                f"{self.confidence_threshold}:{candidates}"
            )

    def _ensure_loaded(self) -> None:
        if self._processor is not None:
            return
        try:
            from sam3_mlx import build_sam3_image_model
            from sam3_mlx.model.sam3_image_processor import Sam3Processor
        except ImportError as exc:
            raise SamTeacherError(
                "sam3_mlx not installed. Install: pip install -e .[audit]",
                kind="import",
            ) from exc
        kwargs: dict[str, Any] = {
            "device": self.device,
            "checkpoint_path": self.checkpoint_path,
            "load_from_HF": self.load_from_hf,
        }
        if self.load_from_hf:
            kwargs["hf_repo"] = self.hf_repo
        model = build_sam3_image_model(**kwargs)
        if self.quantize_bits:
            import mlx.nn as nn

            nn.quantize(
                model,
                bits=self.quantize_bits,
                group_size=64,
                class_predicate=lambda name, mod: (
                    isinstance(mod, nn.Linear)
                    and mod.weight.shape[-1] % 64 == 0
                    and mod.weight.shape[0] % 64 == 0
                ),
            )
        processor = Sam3Processor(
            model,
            resolution=self.resolution,
            device=self.device,
            confidence_threshold=self.confidence_threshold,
        )
        self._model = model
        self._processor = processor

    def predict(
        self,
        image: Image.Image,
        frigate_class: str,
        frigate_bbox: BoundingBox,
    ) -> SamResult:
        """Run SAM 3.1 on one crop and return the best candidate result.

        Raises :class:`SamTeacherError` when the model is unavailable, the
        image cannot be prepared, or no candidate reaches the thresholds.
        """
        self._ensure_loaded()
        try:
            state = self._processor.set_image(np.asarray(image.convert("RGB")))
            width = image.width
            height = image.height
            cx, cy, bw, bh = frigate_bbox.to_cxcywh_norm(width, height)
            box = [float(cx), float(cy), float(bw), float(bh)]
            state = self._processor.add_geometric_prompt(box, True, state)

            candidates = self._candidate_order(frigate_class)
            samples: list[tuple[str, SamCandidate]] = []
            for cand in candidates:
                state = self._processor.set_text_prompt(cand, state)
                detections = self._collect_detections(state, cand, width, height)
                best = self._best_single(detections)
                if best is not None:
                    samples.append((cand, best))
            selected = self._select_best_candidate(samples, frigate_class)
        except SamTeacherError:
            raise
        except Exception as exc:
            raise SamTeacherError(f"SAM 3.1 failed: {exc}") from exc

        if selected is None:
            raise SamTeacherError(
                "SAM 3.1 produced no candidate above the confidence threshold"
            )

        cand_class, cand = selected
        metadata = {
            "model_key": self.model_key,
            "version": self.version,
            "resolution": self.resolution,
            "prompts": {
                "box_norm_cxcywh": box,
                "text_candidates": candidates,
            },
            "candidates": [c.as_dict() for _, c in samples],
        }
        return SamResult(
            class_name=cand_class,
            bbox=cand.bbox,
            mask=Mask(cand.mask),
            confidence=cand.score if cand.score >= 0.0 else None,
            raw_metadata=metadata,
        )

    def _candidate_order(self, frigate_class: str) -> list[str]:
        """Frigate class first, then the configured candidates in order."""
        if not self.candidate_classes:
            return [frigate_class]
        ordered = [c for c in self.candidate_classes if c != frigate_class]
        return [frigate_class, *ordered]

    def _collect_detections(
        self,
        state: dict[str, Any],
        cand: str,
        width: int,
        height: int,
    ) -> list[SamCandidate]:
        """Snapshot one grounding pass into candidates (crop-space pixels)."""
        import mlx.core as mx

        scores_arr = state["scores"]
        boxes_arr = state["boxes"]
        mask_arrs = list(state["masks"])
        if scores_arr is None or boxes_arr is None:
            return []
        mx.eval(scores_arr, boxes_arr, *mask_arrs)
        scores = np.asarray(scores_arr).reshape(-1)
        boxes = np.asarray(boxes_arr).reshape(-1, 4)
        masks = [np.asarray(m, dtype=bool).squeeze() for m in mask_arrs]
        results: list[SamCandidate] = []
        for index in range(len(scores)):
            score = float(scores[index])
            mask = masks[index] if index < len(masks) else None
            if mask is None or mask.ndim != 2 or not np.any(mask):
                continue
            x1, y1, x2, y2 = (float(v) for v in boxes[index])
            box = BoundingBox(
                min(max(x1, 0.0), float(width)),
                min(max(y1, 0.0), float(height)),
                min(max(x2, 0.0), float(width)),
                min(max(y2, 0.0), float(height)),
            ).clipped(width, height)
            if not box.is_valid():
                continue
            results.append(
                SamCandidate(
                    class_name=cand,
                    score=score,
                    bbox=box,
                    mask_area=int(np.count_nonzero(mask)),
                    index=index,
                    mask=mask,
                )
            )
            self._last_mask = mask
        return results

    def _best_single(self, detections: list[SamCandidate]) -> SamCandidate | None:
        """Deterministic per-candidate selection.

        Rank: score desc, mask area desc, index asc.
        """
        if not detections:
            return None
        return min(detections, key=lambda c: (-c.score, -c.mask_area, c.index))

    def _select_best_candidate(
        self,
        samples: list[tuple[str, SamCandidate]],
        frigate_class: str,
    ) -> tuple[str, SamCandidate] | None:
        """Deterministic cross-candidate selection.

        Rank: score desc, frigate priority asc (Frigate class wins ties),
        candidate order asc. The candidate order was already stabilised by
        ``_candidate_order``.
        """
        if not samples:
            return None
        order = self._candidate_order(frigate_class)
        priority = {name: 0 if name == frigate_class else 1 for name in order}
        rank = {name: i for i, name in enumerate(order)}
        return min(samples, key=lambda pair: (-pair[1].score, priority[pair[0]], rank[pair[0]]))

    @property
    def last_mask(self) -> np.ndarray | None:
        """Most recent mask array (debug aid; not part of the contract)."""
        return getattr(self, "_last_mask", None)


__all__ = ["MlxSam3Teacher", "SamTeacher", "SamTeacherError"]
