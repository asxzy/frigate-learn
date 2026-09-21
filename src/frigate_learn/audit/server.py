"""Remote SAM 3.1 server (FastAPI).

Hosts the local MLX teacher (:class:`MlxSam3Teacher`) behind a small HTTP
API so the audit pipeline can run on a machine that has no SAM/MLX stack
(a small VM): ``frigate-learn sam-server`` serves it, and the pipeline
talks to it through :class:`HttpSamTeacher`
(``audit.models.sam.backend: http``).

Endpoints
=========

* ``GET /``          -- server + model identity (``model_key``)
* ``GET /healthz``   -- liveness probe (used by ``HttpSamTeacher.is_available``
  and by ``--resume`` to refresh stale ``sam_failure`` decisions)
* ``POST /predict``  -- one crop: ``{image: <png data url>, frigate_class,
  bbox}`` -> ``{class_name, bbox, confidence, mask: {encoding: png, base64},
  model_key, raw_metadata}``

The MLX processor mutates shared state, so predictions are serialised with
a process-wide lock. Model weights load lazily on the first request.
"""

from __future__ import annotations

import base64
import io
import threading
from typing import Any

from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

from .sam import MlxSam3Teacher, SamTeacherError
from .types import BoundingBox, Mask


class PredictRequest(BaseModel):
    image: str  # data:image/png;base64,...
    frigate_class: str
    bbox: list[float]


def _decode_image(data_url: str) -> Image.Image:
    if not data_url.startswith("data:image/") or ";base64," not in data_url:
        raise HTTPException(
            status_code=422,
            detail={
                "kind": "bad_request",
                "message": "image must be a base64 data URL",
            },
        )
    try:
        _, b64 = data_url.split(",", 1)
        raw = base64.b64decode(b64)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail={"kind": "bad_request", "message": f"image decode failed: {exc}"},
        ) from exc


def _mask_to_png(mask: Mask) -> dict[str, str]:
    buf = io.BytesIO()
    Image.fromarray(mask.to_uint8(), mode="L").save(buf, format="PNG")
    return {
        "encoding": "png",
        "base64": base64.b64encode(buf.getvalue()).decode("ascii"),
    }


def create_sam_server(teacher) -> FastAPI:
    """App factory for the remote SAM endpoint around a local ``SamTeacher``."""
    app = FastAPI(
        title="frigate-learn sam-server",
        version=getattr(teacher, "version", "sam3"),
    )
    lock = threading.Lock()

    @app.get("/")
    def info() -> dict[str, Any]:
        return {
            "name": "frigate-learn sam-server",
            "backend": "sam3-mlx",
            "model_key": teacher.model_key,
            "endpoints": ["/healthz", "/predict"],
        }

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {"status": "ok", "model_key": teacher.model_key}

    @app.post("/predict")
    def predict(payload: PredictRequest) -> dict[str, Any]:
        image = _decode_image(payload.image)
        if len(payload.bbox) != 4:
            raise HTTPException(
                status_code=422,
                detail={
                    "kind": "bad_request",
                    "message": "bbox must be [x1, y1, x2, y2]",
                },
            )
        frigate_bbox = BoundingBox(*payload.bbox)
        try:
            with lock:
                result = teacher.predict(image, payload.frigate_class, frigate_bbox)
        except SamTeacherError as exc:
            if exc.kind == "import":
                raise HTTPException(
                    status_code=500,
                    detail={"kind": "import", "message": str(exc)},
                ) from exc
            raise HTTPException(
                status_code=422,
                detail={"kind": "no_candidate", "message": str(exc)},
            ) from exc
        return {
            "class_name": result.class_name,
            "bbox": result.bbox.to_list(),
            "confidence": result.confidence,
            "mask": _mask_to_png(result.mask),
            "mask_shape": [result.mask.height, result.mask.width],
            "model_key": teacher.model_key,
            "raw_metadata": result.raw_metadata,
        }

    return app


def create_sam_app(config) -> FastAPI:
    """Build the server app from an ``AppConfig`` (audit.models.sam).

    The server hosts the local MLX backend; a config pointing at a remote
    backend is a configuration error on the server host.
    """
    if config.audit.sam.backend != "mlx":
        raise RuntimeError(
            "sam-server hosts the local MLX SAM backend; set "
            "audit.models.sam.backend: mlx on the server host",
        )
    teacher = MlxSam3Teacher(
        checkpoint_path=config.audit.sam.checkpoint or None,
        load_from_hf=config.audit.sam.load_from_hf,
        hf_repo=config.audit.sam.hf_repo,
        quantize_bits=config.audit.sam.quantize_bits,
        resolution=config.audit.sam.resolution,
        confidence_threshold=config.audit.sam.confidence_threshold,
        candidate_classes=list(config.audit.sam.candidate_classes),
    )
    return create_sam_server(teacher)


__all__ = ["PredictRequest", "create_sam_app", "create_sam_server"]
