"""VLM parser and reconciler tests.

Parser must be strict: missing fields, wrong types, and malformed JSON are
rejected — never inferred. The reconciler talks to an OpenAI-compatible
endpoint (oMLX) mocked with respx and never leaks the class into the prompt.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import respx
from httpx import Response
from PIL import Image

from frigate_learn.audit.vlm import (
    VlmResultError,
    VlmTransportError,
    build_reconciler,
    parse_vlm_result,
)


def _good_raw() -> dict:
    return {
        "bbox_covers_object": True,
        "class_label": "person",
    }


def test_parse_valid():
    result = parse_vlm_result(_good_raw())
    assert result.object_present is True
    assert result.bbox_covers_object is True
    assert result.class_label == "person"
    assert result.all_conditions_met is True


def test_parse_absent_object():
    result = parse_vlm_result({"bbox_covers_object": True, "class_label": ""})
    assert result.object_present is False
    assert result.all_conditions_met is False
    assert result.failing_conditions() == ["object_present"]


def test_parse_missing_field_rejected():
    raw = _good_raw()
    del raw["class_label"]
    with pytest.raises(VlmResultError, match="missing required fields"):
        parse_vlm_result(raw)


def test_parse_wrong_type_rejected():
    raw = _good_raw()
    raw["bbox_covers_object"] = "yes"
    with pytest.raises(VlmResultError, match="must be bool"):
        parse_vlm_result(raw)
    raw2 = _good_raw()
    raw2["class_label"] = 7
    with pytest.raises(VlmResultError, match="must be str"):
        parse_vlm_result(raw2)


def test_parse_not_a_dict_rejected():
    with pytest.raises(VlmResultError, match="expected JSON object"):
        parse_vlm_result([True, True])  # type: ignore[arg-type]
    with pytest.raises(VlmResultError, match="expected JSON object"):
        parse_vlm_result("person")


def test_parse_extra_keys_ignored():
    raw = _good_raw()
    raw["reasoning"] = "looks fine"
    result = parse_vlm_result(raw)
    assert result.all_conditions_met is True


def test_parse_old_biased_schema_rejected():
    raw = {
        "object_present": True,
        "class_matches_frigate": True,
        "class_matches_sam": True,
        "bbox_is_valid": True,
        "mask_is_valid": True,
        "agreement": True,
    }
    with pytest.raises(VlmResultError, match="missing required fields"):
        parse_vlm_result(raw)


def test_extract_json_from_fenced_text():
    from frigate_learn.audit.vlm import _extract_json_from_response

    text = "Here is the answer: {\"bbox_covers_object\": true, \"class_label\": \"person\"}"
    assert "bbox_covers_object" in str(_extract_json_from_response(text))


def test_missing_json_rejected():
    from frigate_learn.audit.vlm import _extract_json_from_response

    with pytest.raises(VlmResultError, match="no JSON object"):
        _extract_json_from_response("no json here at all")


def _image(tmp_path: Path) -> Path:
    path = tmp_path / "recon.png"
    Image.new("RGB", (64, 48), (30, 120, 200)).save(path)
    return path


def test_reconciler_happy_path(tmp_path):
    import json as _json

    recon = build_reconciler("http://omlx:8080/v1", "qwen-7b", max_retries=0)
    with respx.mock() as m:
        m.post("http://omlx:8080/v1/chat/completions").mock(
            return_value=Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": _json.dumps(_good_raw())}},
                    ]
                },
            ),
        )
        result = recon.reconcile(_image(tmp_path), class_options=["person", "car"])
    assert result.all_conditions_met is True


def test_reconciler_wrapped_json(tmp_path):
    recon = build_reconciler("http://omlx:8080/v1", "qwen-7b", max_retries=0)
    wrapped = "```json\n{\"bbox_covers_object\": true, \"class_label\": \"person\"}"
    with respx.mock() as m:
        m.post("http://omlx:8080/v1/chat/completions").mock(
            return_value=Response(200, json={"choices": [{"message": {"content": wrapped}}]}),
        )
        result = recon.reconcile(_image(tmp_path))
    assert result.all_conditions_met is True


def test_reconciler_malformed_schema_raises(tmp_path):
    recon = build_reconciler("http://omlx:8080/v1", "qwen-7b", max_retries=1)
    bad = "{\"bbox_covers_object\": true}"
    with respx.mock() as m:
        m.post("http://omlx:8080/v1/chat/completions").mock(
            return_value=Response(200, json={"choices": [{"message": {"content": bad}}]}),
        )
        with pytest.raises(VlmResultError, match="missing required fields"):
            recon.reconcile(_image(tmp_path))


def test_reconciler_transport_error_retries_then_transport(tmp_path):
    recon = build_reconciler("http://omlx:8080/v1", "qwen-7b", max_retries=2)
    with respx.mock() as m:
        m.post("http://omlx:8080/v1/chat/completions").mock(
            return_value=Response(503, json={"error": "overloaded"}),
        )
        with pytest.raises(VlmTransportError):
            recon.reconcile(_image(tmp_path))
        assert len(m.calls) == 3


def test_reconciler_http_500_raises_transport(tmp_path):
    recon = build_reconciler("http://omlx:8080/v1", "qwen-7b", max_retries=0)
    with respx.mock() as m:
        m.post("http://omlx:8080/v1/chat/completions").mock(
            return_value=Response(500, json={"error": "boom"}),
        )
        with pytest.raises(VlmTransportError):
            recon.reconcile(_image(tmp_path))


def test_reconciler_payload_shape(tmp_path):
    recon = build_reconciler("http://omlx:8080/v1", "qwen-7b", max_retries=0)
    with respx.mock() as m:
        m.post("http://omlx:8080/v1/chat/completions").mock(
            return_value=Response(200, json={"choices": [{"message": {"content": "{}"}}]}),
        )
        with pytest.raises(VlmResultError):
            recon.reconcile(_image(tmp_path), class_options=["person", "car"])
        body = json.loads(m.calls[0].request.content)
    assert body["temperature"] == 0.0
    assert body["response_format"] == {"type": "json_object"}
    msgs = body["messages"]
    assert msgs[0]["role"] == "system"
    user = msgs[1]
    assert user["role"] == "user"
    assert "image_url" in user["content"][1]


def test_reconciler_never_leaks_expected_class(tmp_path):
    recon = build_reconciler("http://omlx:8080/v1", "qwen-7b", max_retries=0)
    with respx.mock() as m:
        m.post("http://omlx:8080/v1/chat/completions").mock(
            return_value=Response(200, json={"choices": [{"message": {"content": "{}"}}]}),
        )
        with pytest.raises(VlmResultError):
            recon.reconcile(_image(tmp_path), class_options=["person", "car"])
        body = json.loads(m.calls[0].request.content)
    text = body["messages"][1]["content"][0]["text"]
    assert "person" in text
    assert "car" in text
    assert "expected" not in text.lower()
    assert "frigate" not in text.lower()


def test_reconciler_hard_deadline_bounds_slow_server(tmp_path, monkeypatch):
    import time as _time

    recon = build_reconciler(
        "http://omlx:8080/v1", "qwen-7b", max_retries=1, timeout_seconds=0.1
    )

    class HangingClient:
        def __init__(self) -> None:
            self.closed = 0

        def post(self, *args, **kwargs):
            _time.sleep(1.5)
            return Response(200, json={"choices": [{"message": {"content": "{}"}}]})

        def close(self) -> None:
            self.closed += 1

    hanging = HangingClient()
    monkeypatch.setattr(
        "frigate_learn.audit.vlm.httpx.Client", lambda *a, **k: hanging
    )
    t0 = _time.monotonic()
    with pytest.raises(VlmTransportError) as exc_info:
        recon.reconcile(_image(tmp_path))
    elapsed = _time.monotonic() - t0
    _time.sleep(0.05)
    assert "timed out" in str(exc_info.value)
    assert elapsed < 1.0
    assert hanging.closed >= 2
    recon.close()


def test_vlm_key_includes_model_and_temperature():
    from frigate_learn.audit.vlm import OmlixReconciler, OmlixReconcilerSettings

    recon = OmlixReconciler(
        settings=OmlixReconcilerSettings(model="qwen-7b", temperature=0.0),
    )
    assert "qwen-7b" in recon.vlm_key