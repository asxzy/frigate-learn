"""Remote SAM server (FastAPI app factory) tests."""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from frigate_learn.audit.sam import SamTeacherError
from frigate_learn.audit.server import create_sam_app, create_sam_server
from frigate_learn.audit.types import Mask, SamResult


class StubTeacher:
    model_key = "stub:sam:server:1"
    version = "sam3.1"

    def predict(self, image, frigate_class, frigate_bbox):
        if frigate_class == "ghost":
            raise SamTeacherError("no candidate above threshold", kind="run")
        height, width = image.size
        mask = np.zeros((height, width), dtype=bool)
        mask[8:24, 12:40] = True
        return SamResult(
            class_name=frigate_class,
            bbox=frigate_bbox,
            mask=Mask(mask),
            confidence=0.91,
            raw_metadata={"prompts": {"text_candidates": [frigate_class]}},
        )


class BrokenImportTeacher(StubTeacher):
    """Simulates a server host without the sam3_mlx backend installed."""

    def predict(self, image, frigate_class, frigate_bbox):
        raise SamTeacherError("sam3_mlx not installed", kind="import")


def _image_data_url(rows: int = 64, cols: int = 64) -> str:
    buf = io.BytesIO()
    Image.new("RGB", (cols, rows), (90, 120, 150)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _predict_body(**overrides) -> dict:
    body = {
        "image": _image_data_url(),
        "frigate_class": "person",
        "bbox": [2.0, 4.0, 60.0, 62.0],
    }
    body.update(overrides)
    return body


@pytest.fixture()
def client():
    return TestClient(create_sam_server(StubTeacher()))


def test_healthz(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_key"] == "stub:sam:server:1"


def test_info(client):
    body = client.get("/").json()
    assert body["name"] == "frigate-learn sam-server"
    assert body["model_key"] == "stub:sam:server:1"
    assert "/predict" in body["endpoints"]


def test_predict_happy_path(client):
    resp = client.post("/predict", json=_predict_body())
    assert resp.status_code == 200
    body = resp.json()
    assert body["class_name"] == "person"
    assert body["bbox"] == [2.0, 4.0, 60.0, 62.0]
    assert body["confidence"] == 0.91
    assert body["model_key"] == "stub:sam:server:1"
    assert body["mask_shape"] == [64, 64]
    assert body["raw_metadata"]["prompts"]["text_candidates"] == ["person"]
    mask = body["mask"]
    assert mask["encoding"] == "png"
    arr = (
        np.asarray(
            Image.open(io.BytesIO(base64.b64decode(mask["base64"]))).convert("L")
        )
        > 127
    )
    assert arr.shape == (64, 64)
    assert arr.sum() == 16 * 28


def test_predict_mask_round_trips_through_client_parser(client):
    from frigate_learn.audit.sam import parse_sam_response

    body = client.post("/predict", json=_predict_body()).json()
    result = parse_sam_response(body)
    assert result.class_name == "person"
    assert result.mask.area == 16 * 28
    assert result.mask.tight_bbox().to_list() == [12.0, 8.0, 40.0, 24.0]
    assert result.raw_metadata["server_model_key"] == "stub:sam:server:1"


def test_predict_no_candidate_is_422(client):
    resp = client.post("/predict", json=_predict_body(frigate_class="ghost"))
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["kind"] == "no_candidate"


def test_predict_import_kind_is_500():
    broken = TestClient(create_sam_server(BrokenImportTeacher()))
    resp = broken.post("/predict", json=_predict_body())
    assert resp.status_code == 500
    assert resp.json()["detail"]["kind"] == "import"


def test_predict_rejects_non_data_url(client):
    resp = client.post("/predict", json=_predict_body(image="not-a-data-url"))
    assert resp.status_code == 422
    assert resp.json()["detail"]["kind"] == "bad_request"


def test_predict_rejects_bad_bbox_length(client):
    resp = client.post("/predict", json=_predict_body(bbox=[1.0, 2.0]))
    assert resp.status_code == 422


def test_invalid_json_body_is_422(client):
    resp = client.post(
        "/predict", content=b"{not json", headers={"Content-Type": "application/json"}
    )
    assert resp.status_code == 422


def test_create_sam_app_builds_mlx_teacher_for_mlx_backend(tmp_path):
    from frigate_learn.config import build_config

    cfg = build_config(
        {"audit": {"models": {"sam": {"backend": "mlx"}}}}, base_dir=tmp_path
    )
    app = create_sam_app(cfg)
    paths = [getattr(route, "path", "") for route in app.routes]
    assert "/predict" in paths  # heavy model load stays lazy


def test_create_sam_app_rejects_remote_backend(tmp_path):
    from frigate_learn.config import build_config

    cfg = build_config(
        {"audit": {"models": {"sam": {"backend": "http", "base_url": "http://x:1"}}}},
        base_dir=tmp_path,
    )
    with pytest.raises(RuntimeError, match="backend"):
        create_sam_app(cfg)
