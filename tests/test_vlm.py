"""OpenAI-compatible VLM provider tests (transport mocked with respx)."""

from __future__ import annotations

from pathlib import Path

import pytest
import respx
from httpx import Response

from frigate_learn.annotation.schema import VLMValidationError
from frigate_learn.annotation.vlm import build_provider


@pytest.fixture
def image(tmp_path) -> Path:
    from PIL import Image

    path = tmp_path / "shot.jpg"
    Image.new("RGB", (64, 48), (200, 90, 30)).save(path)
    return path


def make_provider(base_url="http://llm:4001/v1", **kw):
    return build_provider(
        base_url=base_url,
        api_key="asxzy",
        model="gpt-4o-mini",
        provider_name="openai",
        allowed_labels=["person", "car"],
        max_retries=1,
        backoff_seconds=0.01,
        **kw,
    )


def test_verify_happy_path(image):
    with respx.mock() as m:
        m.post("http://llm:4001/v1/chat/completions").mock(
            return_value=Response(
                200,
                json={
                    "choices": [{
                        "message": {
                            "content": '```json\n{"objects": [{"label": "person", "confidence": 0.9, "bbox": [0.1, 0.1, 0.5, 0.5]}]}\n```'
                        }
                    }]
                },
            )
        )
        provider = make_provider()
        result = provider.verify([image, image])
    assert len(result) == 2
    assert result[0][0].label == "person"


def test_retries_on_503_then_succeeds(image):
    with respx.mock() as m:
        route = m.post("http://llm:4001/v1/chat/completions")
        route.side_effect = [
            Response(503, json={"error": "overloaded"}),
            Response(200, json={"choices": [{"message": {"content": '{"objects": []}'}}]}),
        ]
        provider = make_provider()
        result = provider.verify([image])
    assert route.call_count == 2
    assert result[0] == []


def test_rejects_out_of_schema_output(image):
    with respx.mock() as m:
        m.post("http://llm:4001/v1/chat/completions").mock(
            return_value=Response(200, json={"choices": [{"message": {"content": '{"objects": 42}'}}]})
        )
        provider = make_provider()
        with pytest.raises(VLMValidationError):
            provider.verify([image])


def test_http_error_becomes_validation_error(image):
    with respx.mock() as m:
        m.post("http://llm:4001/v1/chat/completions").mock(return_value=Response(400, json={}))
        provider = make_provider()
        with pytest.raises(VLMValidationError):
            provider.verify([image])


def test_missing_base_url_rejected():
    with pytest.raises(VLMValidationError):
        make_provider(base_url="")