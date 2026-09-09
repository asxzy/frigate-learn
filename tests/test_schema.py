"""VLM output schema validation tests (Phase 4)."""

from __future__ import annotations

import pytest

from frigate_learn.annotation.schema import (
    VLMValidationError,
    extract_json_from_response,
    validate_vlm_output,
)


def test_valid_output_pass():
    objs = validate_vlm_output(
        {"objects": [{"label": "person", "confidence": 0.9, "bbox": [0.1, 0.1, 0.5, 0.5]}]},
        allowed_labels=["person", "car"],
    )
    assert len(objs) == 1
    assert objs[0].label == "person"
    assert objs[0].bbox == (0.1, 0.1, 0.5, 0.5)


def test_bare_list_accepted():
    objs = validate_vlm_output(
        [{"label": "person", "confidence": 0.8, "bbox": [0, 0, 0.4, 0.4]}]
    )
    assert len(objs) == 1


def test_empty_objects_allowed():
    objs = validate_vlm_output({"objects": []})
    assert objs == []


@pytest.mark.parametrize(
    "payload",
    [
        None,
        "text",
        {"objects": "not-a-list"},
        {"label": "person"},  # missing 'objects'
        {"objects": [{"label": "", "confidence": 0.5, "bbox": [0, 0, 1, 1]}]},
        {"objects": [{"label": "person", "confidence": 1.5, "bbox": [0, 0, 1, 1]}]},
        {"objects": [{"label": "person", "confidence": 0.5, "bbox": [0, 0]}]},  # 2 coords
        {"objects": [{"label": "person", "confidence": 0.5, "bbox": [0.4, 0.4, 0.2, 0.5]}]},  # inverted
        {"objects": [{"label": "person", "confidence": 0.5, "bbox": [0, 0, 1, 1], "extra": True}]},
    ],
)
def test_invalid_outputs_rejected(payload):
    with pytest.raises(VLMValidationError):
        validate_vlm_output(payload)


def test_out_of_allowed_labels_rejected():
    with pytest.raises(VLMValidationError):
        validate_vlm_output(
            {"objects": [{"label": "zebra", "confidence": 0.5, "bbox": [0, 0, 1, 1]}]},
            allowed_labels=["person"],
        )


def test_confidence_one_point_zero_is_allowed():
    validate_vlm_output(
        {"objects": [{"label": "person", "confidence": 1.0, "bbox": [0, 0, 1, 1]}]}
    )


def test_extract_json_from_response_chat_wrappers():
    content = 'Sure! Here you go:\n```json\n{"objects": [{"label": "dog", "confidence": 0.5, "bbox": [0, 0, 1, 1]}]}\n```'
    data = extract_json_from_response(content)
    assert data["objects"][0]["label"] == "dog"


def test_extract_json_with_prose_prefix():
    content = 'The image contains {"objects": []} as no objects were found.'
    assert extract_json_from_response(content) == {"objects": []}


def test_extract_json_drops_trailing_garbage():
    content = '{"objects": [{"label": "person", "confidence": 0.4, "bbox": [0.1, 0.1, 0.5, 0.5]}]}, {"second": true}'
    data = extract_json_from_response(content)
    assert data["objects"][0]["label"] == "person"
    assert "second" not in data


def test_extract_json_with_trailing_prose():
    content = '{"objects": []} and that is my final answer'
    assert extract_json_from_response(content) == {"objects": []}


def test_extract_json_rejects_nonsense():
    with pytest.raises(VLMValidationError):
        extract_json_from_response("no json here at all")