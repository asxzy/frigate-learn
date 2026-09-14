"""Model backend instantiation tests (input-size alignment)."""

from __future__ import annotations

import pytest

from frigate_learn.evaluation.backends import UltralyticsBackend


def test_ultralytics_backend_requires_explicit_imgsz():
    with pytest.raises(TypeError):
        UltralyticsBackend(weights="yolov8n.pt")


def test_ultralytics_backend_stores_imgsz():
    backend = UltralyticsBackend(weights="yolov8n.pt", imgsz=320)
    assert backend.imgsz == 320