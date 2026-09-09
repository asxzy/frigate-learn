"""VLM verification (Phase 4).

The verifier contract: a vision-language model inspects a training image and
returns zero or more detections. Output is constrained to a strict JSON shape so
results are machine-usable without a human in the loop. Only labels from the
configured class list are accepted; anything else is a validation error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

DETECTION_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["objects"],
    "properties": {
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["label", "confidence", "bbox"],
                "properties": {
                    "label": {"type": "string", "minLength": 1},
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                    "bbox": {
                        "type": "array",
                        "minItems": 4,
                        "maxItems": 4,
                        "items": {"type": "number"},
                    },
                },
                "additionalProperties": False,
            },
        }
    },
    "additionalProperties": False,
}

# JSON Schema with a *hard* exclusive maximum for confidence (values < 1.0).
# Kept separate so tooling reading DETECTION_OUTPUT_SCHEMA sees the soft form.
STRICT_DETECTION_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["objects"],
    "properties": {
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["label", "confidence", "bbox"],
                "properties": {
                    "label": {"type": "string", "minLength": 1},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "bbox": {
                        "type": "array",
                        "minItems": 4,
                        "maxItems": 4,
                        "items": {"type": "number"},
                    },
                },
                "additionalProperties": False,
            },
        }
    },
    "additionalProperties": False,
}


@dataclass(frozen=True)
class VLMObject:
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]  # normalized x1 y1 x2 y2


class VLMValidationError(ValueError):
    """Raised when the model's output does not conform to the strict schema."""


def _clamp_box(box: Sequence[float]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = [max(0.0, min(1.0, float(v))) for v in box]
    if x2 <= x1 or y2 <= y1:
        raise VLMValidationError(f"degenerate bbox: {box}")
    if x2 - x1 < 1e-4:
        return (x1, y1, x1 + 1e-4, y2)
    return (x1, y1, x2, y2)


def validate_vlm_output(
    data: Any,
    *,
    allowed_labels: Sequence[str] | None = None,
    strict: bool = False,
) -> list[VLMObject]:
    """Validate model output against the strict schema.

    Accepts either ``{"objects": [...]}`` or a bare list of objects. Raises
    ``VLMValidationError`` on any violation; returns a list of ``VLMObject``.
    """
    from jsonschema import Draft7Validator

    if isinstance(data, list):
        data = {"objects": data}
    if not isinstance(data, Mapping):
        raise VLMValidationError("output must be a JSON object with an 'objects' array")

    schema = STRICT_DETECTION_OUTPUT_SCHEMA if strict else DETECTION_OUTPUT_SCHEMA
    errors = sorted(Draft7Validator(schema).iter_errors(data), key=lambda e: list(e.path))
    if errors:
        details = "; ".join(f"{'/'.join(map(str, e.path)) or 'root'}: {e.message}" for e in errors[:5])
        raise VLMValidationError(f"VLM output rejected: {details}")

    allowed = set(allowed_labels) if allowed_labels is not None else None
    objects: list[VLMObject] = []
    for item in data["objects"]:
        label = str(item["label"]).strip()
        if not label:
            raise VLMValidationError("empty label")
        if allowed is not None and label not in allowed:
            raise VLMValidationError(f"label {label!r} not in allowed set {sorted(allowed)}")
        conf = float(item["confidence"])
        if not (0.0 <= conf <= 1.0):
            raise VLMValidationError(f"confidence out of range: {conf}")
        objects.append(VLMObject(label=label, confidence=conf, bbox=_clamp_box(item["bbox"])))
    return objects


def extract_json_from_response(content: str) -> Any:
    """Strip markdown fences / prose and parse the embedded JSON object."""
    import json
    import re

    if not content:
        raise VLMValidationError("empty model response")
    text = content.strip()
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if fenced:
        text = fenced[-1].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        # maybe the model returned a bare array
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1:
            raise VLMValidationError("no JSON found in model response")
    candidate = text[start : end + 1]
    # Models sometimes append prose or a second object after the JSON; keep the
    # longest prefix that still parses (e.g. drop a trailing ", {...}").
    while candidate:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            if exc.pos == 0 or "Extra data" not in exc.msg:
                raise VLMValidationError(f"invalid JSON in model response: {exc}") from exc
            candidate = candidate[: exc.pos]
    raise VLMValidationError("no JSON found in model response")


__all__ = [
    "VLMObject",
    "VLMValidationError",
    "DETECTION_OUTPUT_SCHEMA",
    "STRICT_DETECTION_OUTPUT_SCHEMA",
    "validate_vlm_output",
    "extract_json_from_response",
]