"""Concrete detector backends for benchmarking (Phase 6).

Only the ultralytics backend is provided out of the box; it imports
``ultralytics`` lazily so collection-only installs (and this repo's test suite)
never need torch. Any object implementing ``ModelBackend.predict`` can be wired
in (onnx, Hailo runtime, a mock for tests, ...).
"""

from __future__ import annotations

from pathlib import Path

from .benchmark import DEFAULT_CONFIDENCE, ModelBackend, Prediction


class UltralyticsBackend(ModelBackend):
    """Wraps an ultralytics ``YOLO`` model (auto-downloaded on first use)."""

    def __init__(
        self,
        weights: str,
        *,
        imgsz: int,
        device: str | None = None,
        name: str | None = None,
    ) -> None:
        self.weights = weights
        self.imgsz = imgsz
        self.device = device
        self._name = name or Path(weights).stem
        self._model = None

    @property
    def name(self) -> str:
        return self._name

    def _load(self):
        if self._model is None:
            from ultralytics import YOLO  # lazy: torch not needed otherwise

            self._model = YOLO(self.weights)
        return self._model

    def predict(self, image_path: Path, confidence: float = DEFAULT_CONFIDENCE) -> list[Prediction]:
        model = self._load()
        results = model.predict(
            source=str(image_path),
            conf=confidence,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []
        result = results[0]
        names = getattr(result, "names", None) or {}
        boxes = getattr(result.boxes, "xywhn", None)
        if boxes is None or len(boxes) == 0:
            return []
        cls_ids = result.boxes.cls.tolist()
        confs = result.boxes.conf.tolist()
        predictions: list[Prediction] = []
        for (cx, cy, w, h), cls_id, conf in zip(boxes.tolist(), cls_ids, confs):
            label = str(names.get(int(cls_id), f"class_{int(cls_id)}"))
            x1, y1 = cx - w / 2, cy - h / 2
            x2, y2 = cx + w / 2, cy + h / 2
            predictions.append(
                Prediction(label=label, box=(x1, y1, x2, y2), confidence=float(conf))
            )
        return predictions


__all__ = ["UltralyticsBackend"]