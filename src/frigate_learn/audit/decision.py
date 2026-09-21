"""Decision engine: the single conservative acceptance rule.

The rule is centralized in :func:`is_valid_training_sample`. A sample is
accepted ONLY when every condition holds:

* SAM produced a valid result (non-empty mask, valid box, class present)
* the VLM's independent judgement passes: the box covers the object and the
  object is present (a class label was produced); the VLM reading is made
  blind — it is never told the expected class or SAM's guess
* the independent SAM and VLM class readings agree (canonicalized against
  the label vocabulary, with common synonyms)
* (optional, configurable) deterministic geometry gates pass — e.g. the
  SAM mask must align with the Frigate bbox

Individual components never override the final rule; if agreement cannot be
established the sample is dropped. There is no UNCERTAIN bucket.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .types import Decision, GeometryMetrics, SamResult, VlmResult

REASON_SAM_FAILURE = "sam_failure"
REASON_SAM_GEOMETRY = "sam_geometry"
REASON_VLM_FAILURE = "vlm_failure"
REASON_VLM_MALFORMED = "vlm_malformed"
REASON_VLM_CONDITIONS = "vlm_conditions"

_CLASS_SYNONYMS: dict[str, frozenset[str]] = {
    "person": frozenset({"person", "people", "human", "man", "woman", "pedestrian"}),
    "car": frozenset({"car", "sedan", "automobile", "suv", "pickup"}),
    "bicycle": frozenset({"bicycle", "bike", "cycle"}),
    "motorcycle": frozenset({"motorcycle", "motorbike", "motor bike"}),
    "bird": frozenset({"bird"}),
    "cat": frozenset({"cat", "kitty"}),
    "dog": frozenset({"dog", "puppy"}),
    "suitcase": frozenset({"suitcase", "luggage"}),
}


def canonical_label(label: str, ontology: list[str] | None) -> str:
    """Map a label to a canonical vocabulary member ("" when unmatched).

    Without a vocabulary the raw normalized label is returned so plain
    equality still holds; with a vocabulary, synonyms are resolved and
    out-of-vocabulary labels collapse to "".
    """
    text = (label or "").strip().lower()
    if not text:
        return ""
    if ontology is None:
        return text
    for cls in ontology:
        norm = cls.strip().lower()
        if not norm:
            continue
        members = _CLASS_SYNONYMS.get(norm, frozenset({norm}))
        if text in members:
            return norm
    return ""


@dataclass
class GeometryThresholds:
    """Deterministic geometry gates for the acceptance rule.

    All gates are opt-in: the defaults do not reject anything (they only
    *record* geometry). Set real values in configuration once you have
    validated them against a sample of your dataset. Documented example
    values are given next to each field.
    """

    min_mask_area: int = 0
    min_bbox_iou: float = 0.0
    mask_bbox_ratio_min: float = 0.0
    mask_bbox_ratio_max: float = math.inf
    min_mask_frigate_containment: float = 0.0

    def violations(self, geometry: GeometryMetrics) -> list[str]:
        """Names of the geometry gates that fail (empty when all pass)."""
        failed: list[str] = []
        if geometry.mask_area < self.min_mask_area:
            failed.append("mask_area")
        if geometry.bbox_iou < self.min_bbox_iou:
            failed.append("bbox_iou")
        if geometry.mask_bbox_ratio < self.mask_bbox_ratio_min:
            failed.append("mask_bbox_ratio_min")
        if geometry.mask_bbox_ratio > self.mask_bbox_ratio_max:
            failed.append("mask_bbox_ratio_max")
        if geometry.mask_frigate_containment < self.min_mask_frigate_containment:
            failed.append("mask_frigate_containment")
        return failed

    def as_dict(self) -> dict:
        return {
            "min_mask_area": self.min_mask_area,
            "min_bbox_iou": self.min_bbox_iou,
            "mask_bbox_ratio_min": self.mask_bbox_ratio_min,
            "mask_bbox_ratio_max": self.mask_bbox_ratio_max,
            "min_mask_frigate_containment": self.min_mask_frigate_containment,
        }


def sam_result_is_valid(sam_result: SamResult | None) -> bool:
    """SAM produced a usable result (teacher guarantees non-empty mask)."""
    return sam_result is not None and sam_result.is_valid()


def is_valid_training_sample(
    sam_result: SamResult | None,
    vlm_result: VlmResult | None,
    frigate_label: str,
    *,
    geometry: GeometryMetrics | None = None,
    gates: GeometryThresholds | None = None,
    ontology: list[str] | None = None,
) -> bool:
    """The one conservative acceptance rule.

    ALL of the following must hold:
    - SAM produced a valid result
    - VLM produced a strictly parsed, independent result
    - the VLM box/presence conditions pass
    - the independent SAM and VLM class readings agree
    - optional geometry gates pass (when ``gates`` is provided)
    """
    if not sam_result_is_valid(sam_result):
        return False
    if vlm_result is None or not vlm_result.all_conditions_met:
        return False
    vlm_label = canonical_label(vlm_result.class_label, ontology)
    sam_label = canonical_label(sam_result.class_name, ontology)
    if not vlm_label or vlm_label != sam_label:
        return False
    return gates is None or geometry is None or not gates.violations(geometry)


def decide_sample(
    sam_result: SamResult | None,
    vlm_result: VlmResult | None,
    geometry: GeometryMetrics | None,
    frigate_label: str,
    *,
    gates: GeometryThresholds | None = None,
    vlm_error: str | None = None,
    ontology: list[str] | None = None,
) -> Decision:
    """Produce the final per-sample decision with a stable reason code.

    * ``KEEP`` is the only accepted status.
    * ``DROP`` carries a reason: a SAM/VLM failure or a condition
      disagreement.
    * ``PENDING`` means the VLM transport failed (server down or timeout);
      `--resume` retries it. It is never a decision about content.
    """
    if not sam_result_is_valid(sam_result):
        return Decision(
            status="DROP",
            reason=REASON_SAM_FAILURE,
            frigate_label=frigate_label,
        )

    if vlm_result is None:
        if vlm_error == "transport":
            return Decision(
                status="PENDING",
                reason=REASON_VLM_FAILURE,
                frigate_label=frigate_label,
                sam_class=sam_result.class_name,
            )
        reason = REASON_VLM_MALFORMED if vlm_error == "malformed" else REASON_VLM_FAILURE
        return Decision(
            status="DROP",
            reason=reason,
            frigate_label=frigate_label,
            sam_class=sam_result.class_name,
        )

    gates = gates if gates is not None else GeometryThresholds()
    geometry_violations = gates.violations(geometry) if geometry is not None else []
    failing = list(vlm_result.failing_conditions())
    if vlm_result.object_present:
        vlm_label = canonical_label(vlm_result.class_label, ontology)
        sam_label = canonical_label(sam_result.class_name, ontology)
        if not vlm_label or vlm_label != sam_label:
            failing.append("class_mismatch")

    if failing or geometry_violations:
        conditions = failing + [f"geometry:{v}" for v in geometry_violations]
        return Decision(
            status="DROP",
            reason=REASON_VLM_CONDITIONS if failing else REASON_SAM_GEOMETRY,
            frigate_label=frigate_label,
            sam_class=sam_result.class_name,
            failing_conditions=conditions,
        )
    return Decision(
        status="KEEP",
        reason=None,
        frigate_label=frigate_label,
        sam_class=sam_result.class_name,
        training_label=frigate_label,
        bbox_source="sam",
        mask_source="sam",
    )


__all__ = [
    "REASON_SAM_FAILURE",
    "REASON_SAM_GEOMETRY",
    "REASON_VLM_CONDITIONS",
    "REASON_VLM_FAILURE",
    "REASON_VLM_MALFORMED",
    "GeometryThresholds",
    "canonical_label",
    "decide_sample",
    "is_valid_training_sample",
    "sam_result_is_valid",
]
