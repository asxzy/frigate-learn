"""SAM 3.1 teacher: local MLX backend or a remote HTTP endpoint.

The pipeline talks to :class:`SamTeacher` only and never picks an
implementation. Two teachers exist:

* :class:`MlxSam3Teacher` — SAM 3 / 3.1 ported to MLX (Apple Silicon)
  through the ``sam3_mlx`` package. All sam3_mlx imports are lazy so the
  rest of the package imports and unit-tests cleanly on machines without
  the extra installed.
* :class:`HttpSamTeacher` — calls a remote ``frigate-learn sam-server``
  endpoint, so the pipeline can run on a small VM that only performs
  HTTP calls (SAM 3.1 stays on the Apple Silicon host).

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

import base64
import io
import json
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx
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

    def is_available(self) -> bool:
        """True when the model backend is reachable (used for cache resync)."""
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

    def is_available(self) -> bool:
        """True when the ``sam3_mlx`` backend is importable on this machine."""
        return is_sam_available()

def _image_to_png_data_url(image: Image.Image) -> str:
    """Encode a crop as a base64 PNG data URL for the remote endpoint."""
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _safe_json(response: httpx.Response) -> Any:
    """Parse a response body without raising on invalid JSON."""
    try:
        return response.json()
    except json.JSONDecodeError:
        return None


def _error_message(body: Any) -> str | None:
    """Best-effort human message from a fastapi-style error body."""
    if not isinstance(body, dict):
        return None
    detail = body.get("detail")
    if isinstance(detail, str) and detail:
        return detail
    if isinstance(detail, dict):
        message = detail.get("message")
        if isinstance(message, str) and message:
            return message
        return None
    if isinstance(detail, list) and detail:
        messages = []
        for item in detail:
            if isinstance(item, dict) and item.get("msg"):
                messages.append(str(item["msg"]))
        if messages:
            return "; ".join(messages)
    return None


def _decode_mask_png(data: Any) -> np.ndarray:
    """Decode the png-encoded binary mask from a remote SAM response."""
    if not isinstance(data, dict) or data.get("encoding") != "png":
        raise ValueError("mask must be an object with encoding=png")
    b64 = data.get("base64")
    if not isinstance(b64, str) or not b64:
        raise ValueError("mask.base64 is missing")
    try:
        raw = base64.b64decode(b64)
        arr = np.asarray(Image.open(io.BytesIO(raw)).convert("L")) > 127
    except Exception as exc:
        raise ValueError(f"mask decode failed: {exc}") from exc
    if arr.ndim != 2 or not np.any(arr):
        raise ValueError("mask must be a non-empty 2D boolean array")
    return arr


def parse_sam_response(payload: Any) -> SamResult:
    """Strict parser for the remote SAM response schema.

    Expects ``{class_name, bbox, confidence, mask: {encoding, base64},
    model_key, raw_metadata}``. Raises :class:`SamTeacherError` with kind
    ``run`` on any malformed or missing field; nothing is inferred.
    """
    if not isinstance(payload, dict):
        raise SamTeacherError("SAM response must be a JSON object", kind="run")
    class_name = payload.get("class_name")
    if not isinstance(class_name, str) or not class_name:
        raise SamTeacherError("SAM response is missing class_name", kind="run")
    raw_bbox = payload.get("bbox")
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        raise SamTeacherError("SAM response is missing bbox", kind="run")
    try:
        bbox = BoundingBox(*(float(v) for v in raw_bbox))
    except (TypeError, ValueError) as exc:
        raise SamTeacherError(f"SAM response bbox is invalid: {exc}", kind="run") from exc
    if not bbox.is_valid():
        raise SamTeacherError("SAM response bbox is degenerate", kind="run")
    try:
        mask = Mask(_decode_mask_png(payload.get("mask")))
    except ValueError as exc:
        raise SamTeacherError(f"SAM response mask is invalid: {exc}", kind="run") from exc
    metadata = payload.get("raw_metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    server_key = payload.get("model_key")
    if isinstance(server_key, str) and server_key:
        metadata = {**metadata, "server_model_key": server_key}
    confidence = payload.get("confidence")
    if confidence is not None:
        try:
            confidence = float(confidence)
        except (TypeError, ValueError) as exc:
            raise SamTeacherError("SAM response confidence must be a number", kind="run") from exc
    return SamResult(
        class_name=class_name,
        bbox=bbox,
        mask=mask,
        confidence=confidence,
        raw_metadata=metadata,
    )


@dataclass
class HttpSamSettings:
    """Connection settings for a remote SAM endpoint."""

    base_url: str = ""
    api_key: str = ""
    model: str = ""
    timeout_seconds: float = 120.0
    max_retries: int = 2


class HttpSamTeacher:
    """SAM teacher backed by a remote ``frigate-learn sam-server`` endpoint.

    The audit pipeline only talks to :class:`SamTeacher`; this is the
    *small-VM path*: SAM 3.1 (MLX) runs on the Apple Silicon host, and every
    crop is POSTed as a PNG data URL to ``/predict``. The mask comes back
    png-encoded, so geometry, reconciliation rendering and exports behave
    exactly as with the local teacher.

    ``model_key`` is derived from the endpoint URL and the configured
    ``model`` label so caches invalidate when either changes — bump
    ``audit.models.sam.model`` (or ``pipeline_version``) when the server's
    weights change. Transport failures (unreachable/5xx/timeouts) map to
    :class:`SamTeacherError` kind ``import`` and are re-attempted by
    ``--resume`` once the endpoint is reachable again; deterministic
    rejections (no candidate, malformed payload) map to kind ``run``.
    """

    settings: HttpSamSettings

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str = "",
        model: str = "",
        timeout_seconds: float = 120.0,
        max_retries: int = 2,
    ) -> None:
        self.settings = HttpSamSettings(
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            model=model,
            timeout_seconds=float(timeout_seconds),
            max_retries=int(max_retries),
        )

    @property
    def model_key(self) -> str:
        return f"sam3-http:{self.settings.base_url}:{self.settings.model or 'default'}"

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"
        return headers

    def _client(self) -> httpx.Client:
        return httpx.Client(
            base_url=self.settings.base_url,
            headers=self._headers(),
            timeout=httpx.Timeout(self.settings.timeout_seconds, connect=10.0),
        )

    def predict(
        self,
        image: Image.Image,
        frigate_class: str,
        frigate_bbox: BoundingBox,
    ) -> SamResult:
        """Run remote SAM on one crop; returns the best candidate result."""
        payload: dict[str, Any] = {
            "image": _image_to_png_data_url(image),
            "frigate_class": frigate_class,
            "bbox": frigate_bbox.to_list(),
        }
        last_error: Exception | None = None
        for _attempt in range(1 + self.settings.max_retries):
            try:
                with self._client() as client:
                    response = client.post("/predict", json=payload)
            except httpx.HTTPError as exc:
                last_error = exc
                continue
            if response.status_code == 200:
                try:
                    body = response.json()
                except json.JSONDecodeError as exc:
                    raise SamTeacherError(
                        "SAM server returned a non-JSON response", kind="run"
                    ) from exc
                result = parse_sam_response(body)
                if (
                    result.mask.height != image.height
                    or result.mask.width != image.width
                ):
                    raise SamTeacherError(
                        f"SAM mask {result.mask.width}x{result.mask.height} does not match "
                        f"crop {image.width}x{image.height}",
                        kind="run",
                    )
                return result
            if 400 <= response.status_code < 500:
                message = _error_message(_safe_json(response))
                raise SamTeacherError(
                    message or f"SAM server rejected the request (HTTP {response.status_code})",
                    kind="run",
                )
            last_error = SamTeacherError(
                f"SAM server error (HTTP {response.status_code})", kind="import"
            )
        raise SamTeacherError(
            f"SAM server unreachable after {1 + self.settings.max_retries} attempts: {last_error}",
            kind="import",
        )

    def is_available(self) -> bool:
        """True when the remote endpoint answers ``/healthz`` (short probe)."""
        client = httpx.Client(
            base_url=self.settings.base_url,
            headers=self._headers(),
            timeout=httpx.Timeout(5.0, connect=5.0),
        )
        try:
            response = client.get("/healthz")
            return response.status_code == 200
        except httpx.HTTPError:
            return False
        finally:
            client.close()


__all__ = [
    "HttpSamSettings",
    "HttpSamTeacher",
    "MlxSam3Teacher",
    "SamTeacher",
    "SamTeacherError",
    "parse_sam_response",
]
