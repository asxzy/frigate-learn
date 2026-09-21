"""Remote SAM teacher (HttpSamTeacher) tests — transport mocked with respx."""

from __future__ import annotations

import base64
import io
import json

import httpx
import numpy as np
import pytest
import respx
from httpx import Response
from PIL import Image

from frigate_learn.audit.sam import HttpSamTeacher, SamTeacherError
from frigate_learn.audit.types import BoundingBox

BASE = "http://sam-host:8001"
PREDICT_URL = BASE + "/predict"
HEALTHZ_URL = BASE + "/healthz"


def _png_b64(mask: np.ndarray) -> str:
    img = Image.fromarray(mask.astype(np.uint8) * 255, "L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _mask(rows: int = 64, cols: int = 64) -> np.ndarray:
    arr = np.zeros((rows, cols), dtype=bool)
    arr[10:30, 15:45] = True
    return arr


def _image() -> Image.Image:
    return Image.new("RGB", (64, 64), (120, 130, 140))


def _predict_payload(**overrides) -> dict:
    payload = {
        "class_name": "person",
        "bbox": [8.0, 8.0, 56.0, 56.0],
        "confidence": 0.88,
        "mask": {"encoding": "png", "base64": _png_b64(_mask())},
        "model_key": "sam3-mlx:sam3.1:repo:q0:1008:0.5:",
        "raw_metadata": {"prompts": {"text_candidates": ["person"]}},
    }
    payload.update(overrides)
    return payload


def make_teacher(**kw) -> HttpSamTeacher:
    defaults = {"base_url": BASE, "model": "sam3-remote", "max_retries": 1}
    defaults.update(kw)
    return HttpSamTeacher(**defaults)


def test_predict_happy_path():
    with respx.mock() as m:
        m.post(PREDICT_URL).mock(return_value=Response(200, json=_predict_payload()))
        result = make_teacher().predict(_image(), "person", BoundingBox(4, 4, 60, 60))
    assert result.class_name == "person"
    assert result.bbox.to_list() == [8.0, 8.0, 56.0, 56.0]
    assert result.confidence == 0.88
    assert (result.mask.height, result.mask.width) == (64, 64)
    assert result.mask.area == 20 * 30
    assert result.raw_metadata["server_model_key"].startswith("sam3-mlx")
    assert result.raw_metadata["prompts"]["text_candidates"] == ["person"]


def test_predict_request_body_shape():
    with respx.mock() as m:
        route = m.post(PREDICT_URL).mock(
            return_value=Response(200, json=_predict_payload())
        )
        make_teacher().predict(_image(), "person", BoundingBox(4, 4, 60, 60))
        body = route.calls[0].request.content
    payload = json.loads(body)
    assert payload["frigate_class"] == "person"
    assert payload["bbox"] == [4.0, 4.0, 60.0, 60.0]
    assert payload["image"].startswith("data:image/png;base64,")


def test_model_key_is_deterministic_and_normalizes_base_url():
    a = HttpSamTeacher("http://sam-host:8001/", model="m1")
    b = HttpSamTeacher("http://sam-host:8001", model="m1")
    c = HttpSamTeacher("http://sam-host:8001", model="m2")
    assert a.model_key == "sam3-http:http://sam-host:8001:m1"
    assert a.model_key == b.model_key
    assert a.model_key != c.model_key


def test_retries_on_5xx_then_succeeds():
    with respx.mock() as m:
        route = m.post(PREDICT_URL)
        route.side_effect = [
            Response(503, json={"detail": "overloaded"}),
            Response(200, json=_predict_payload()),
        ]
        result = make_teacher().predict(_image(), "person", BoundingBox(4, 4, 60, 60))
    assert route.call_count == 2
    assert result.class_name == "person"


def test_transport_failure_is_import_kind_after_retries():
    with respx.mock() as m:
        route = m.post(PREDICT_URL)
        route.side_effect = [Response(503, json={}), Response(503, json={})]
        with pytest.raises(SamTeacherError) as excinfo:
            make_teacher().predict(_image(), "person", BoundingBox(4, 4, 60, 60))
    assert excinfo.value.kind == "import"
    assert route.call_count == 2  # max_retries=1 -> 2 attempts
    assert "unreachable" in str(excinfo.value)


def test_connect_error_becomes_import_kind():
    with respx.mock() as m:
        m.post(PREDICT_URL).mock(side_effect=httpx.ConnectError("refused"))
        with pytest.raises(SamTeacherError) as excinfo:
            make_teacher().predict(_image(), "person", BoundingBox(4, 4, 60, 60))
    assert excinfo.value.kind == "import"


def test_no_candidate_422_is_run_kind():
    body = {
        "detail": {
            "kind": "no_candidate",
            "message": "SAM 3.1 produced no candidate above the confidence threshold",
        }
    }
    with respx.mock() as m:
        m.post(PREDICT_URL).mock(return_value=Response(422, json=body))
        with pytest.raises(SamTeacherError) as excinfo:
            make_teacher().predict(_image(), "person", BoundingBox(4, 4, 60, 60))
    assert excinfo.value.kind == "run"
    assert "no candidate" in str(excinfo.value)


def test_malformed_200_is_run_kind():
    with respx.mock() as m:
        m.post(PREDICT_URL).mock(return_value=Response(200, json={"class_name": 42}))
        with pytest.raises(SamTeacherError) as excinfo:
            make_teacher().predict(_image(), "person", BoundingBox(4, 4, 60, 60))
    assert excinfo.value.kind == "run"


def test_non_json_200_is_run_kind():
    with respx.mock() as m:
        m.post(PREDICT_URL).mock(
            return_value=Response(200, content=b"<html>oops</html>")
        )
        with pytest.raises(SamTeacherError) as excinfo:
            make_teacher().predict(_image(), "person", BoundingBox(4, 4, 60, 60))
    assert excinfo.value.kind == "run"


def test_predict_rejects_mask_size_mismatch():
    payload = _predict_payload()
    payload["mask"] = {"encoding": "png", "base64": _png_b64(_mask(rows=32, cols=32))}
    with respx.mock() as m:
        m.post(PREDICT_URL).mock(return_value=Response(200, json=payload))
        with pytest.raises(SamTeacherError) as excinfo:
            make_teacher().predict(_image(), "person", BoundingBox(4, 4, 60, 60))
    assert excinfo.value.kind == "run"
    assert "mask" in str(excinfo.value)
    assert "crop" in str(excinfo.value)


def test_is_available_checks_healthz():
    with respx.mock() as m:
        m.get(HEALTHZ_URL).mock(return_value=Response(200, json={"status": "ok"}))
        assert make_teacher().is_available() is True


def test_is_available_false_when_down():
    with respx.mock() as m:
        m.get(HEALTHZ_URL).mock(side_effect=httpx.ConnectError("refused"))
        assert make_teacher().is_available() is False


def test_is_available_false_on_non_200():
    with respx.mock() as m:
        m.get(HEALTHZ_URL).mock(return_value=Response(503, json={}))
        assert make_teacher().is_available() is False


def test_bearer_token_sent_when_configured():
    with respx.mock() as m:
        route = m.post(PREDICT_URL).mock(
            return_value=Response(200, json=_predict_payload())
        )
        make_teacher(api_key="sekrit").predict(
            _image(), "person", BoundingBox(4, 4, 60, 60)
        )
        headers = route.calls[0].request.headers
    assert headers["Authorization"] == "Bearer sekrit"
