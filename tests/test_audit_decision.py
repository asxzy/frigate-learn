"""Decision engine tests: the single conservative acceptance rule.

KEEP only when SAM valid AND the VLM's independent judgement passes (box
covers the object, object present) AND the SAM and VLM class readings agree
(canonicalized against the vocabulary) plus optional geometry gates.
Everything else must DROP; transport failures are PENDING. No auto-relabeling,
no UNCERTAIN bucket.
"""

from __future__ import annotations

import numpy as np

from frigate_learn.audit.decision import (
    REASON_SAM_FAILURE,
    REASON_SAM_GEOMETRY,
    REASON_VLM_CONDITIONS,
    REASON_VLM_FAILURE,
    REASON_VLM_MALFORMED,
    GeometryThresholds,
    canonical_label,
    decide_sample,
    is_valid_training_sample,
)
from frigate_learn.audit.geometry import compute_geometry
from frigate_learn.audit.types import BoundingBox, Mask, SamResult, VlmResult


def _mask(shape=(100, 100), block=(20, 20, 80, 80)) -> Mask:
    m = np.zeros(shape, dtype=bool)
    y1, x1, y2, x2 = block
    m[y1:y2, x1:x2] = True
    return Mask(m)


def _sam_result(**kw) -> SamResult:
    defaults = {
        "class_name": "person",
        "bbox": BoundingBox(10, 10, 90, 90),
        "mask": _mask(),
        "confidence": 0.9,
    }
    defaults.update(kw)
    return SamResult(**defaults)


def _vlm(**kw) -> VlmResult:
    defaults = {
        "bbox_covers_object": True,
        "class_label": "person",
        "raw": {},
    }
    defaults.update(kw)
    return VlmResult(**defaults)


def _geometry():
    return compute_geometry(
        BoundingBox(0, 0, 100, 100),
        BoundingBox(10, 10, 90, 90),
        _mask(),
    )


def test_agree_keeps():
    sam = _sam_result()
    vlm = _vlm()
    decision = decide_sample(sam, vlm, _geometry(), "person")
    assert decision.status == "KEEP"
    assert decision.training_label == "person"
    assert decision.frigate_label == "person"
    assert decision.bbox_source == "sam"
    assert decision.mask_source == "sam"
    assert is_valid_training_sample(sam, vlm, "person") is True


def test_sam_failure_drops():
    decision = decide_sample(None, None, None, "person")
    assert decision.status == "DROP"
    assert decision.reason == REASON_SAM_FAILURE
    assert is_valid_training_sample(None, _vlm(), "person") is False
    bad_sam = _sam_result(mask=Mask(np.zeros((10, 10), dtype=bool)))
    assert decide_sample(bad_sam, _vlm(), None, "person").status == "DROP"


def test_class_disagreement_drops():
    sam = _sam_result(class_name="person")
    vlm = _vlm(class_label="car")
    decision = decide_sample(sam, vlm, _geometry(), "person")
    assert decision.status == "DROP"
    assert decision.reason == REASON_VLM_CONDITIONS
    assert "class_mismatch" in decision.failing_conditions
    assert is_valid_training_sample(sam, vlm, "person") is False


def test_synonym_labels_agree_via_ontology():
    ontology = ["person", "car"]
    vlm = _vlm(class_label="human")
    sam = _sam_result(class_name="person")
    assert canonical_label("human", ontology) == "person"
    assert decide_sample(
        sam, vlm, _geometry(), "person", ontology=ontology
    ).status == "KEEP"


def test_out_of_vocabulary_label_drops():
    ontology = ["person", "car"]
    vlm = _vlm(class_label="unidentified flying object")
    decision = decide_sample(
        _sam_result(), vlm, _geometry(), "person", ontology=ontology
    )
    assert decision.status == "DROP"
    assert "class_mismatch" in decision.failing_conditions


def test_object_absent_drops():
    vlm = _vlm(class_label="")
    decision = decide_sample(_sam_result(), vlm, _geometry(), "person")
    assert decision.status == "DROP"
    assert "object_present" in decision.failing_conditions


def test_bbox_not_covering_drops():
    vlm = _vlm(bbox_covers_object=False)
    decision = decide_sample(_sam_result(), vlm, _geometry(), "person")
    assert decision.status == "DROP"
    assert "bbox_covers" in decision.failing_conditions
    assert is_valid_training_sample(_sam_result(), vlm, "person") is False


def test_vlm_absent_drops():
    decision = decide_sample(_sam_result(), None, _geometry(), "person")
    assert decision.status == "DROP"
    assert decision.reason == REASON_VLM_FAILURE
    assert is_valid_training_sample(_sam_result(), None, "person") is False


def test_vlm_transport_pending():
    decision = decide_sample(
        _sam_result(), None, _geometry(), "person", vlm_error="transport"
    )
    assert decision.status == "PENDING"
    assert decision.reason == REASON_VLM_FAILURE
    assert is_valid_training_sample(_sam_result(), None, "person") is False


def test_vlm_malformed_is_drop_not_pending():
    decision = decide_sample(
        _sam_result(), None, _geometry(), "person", vlm_error="malformed"
    )
    assert decision.status == "DROP"
    assert decision.reason == REASON_VLM_MALFORMED


def test_geometry_gate_drops():
    sam = _sam_result()
    vlm = _vlm()
    gates = GeometryThresholds(min_mask_area=999999)
    decision = decide_sample(sam, vlm, _geometry(), "person", gates=gates)
    assert decision.status == "DROP"
    assert decision.reason == REASON_SAM_GEOMETRY
    assert "geometry:mask_area" in decision.failing_conditions
    assert is_valid_training_sample(sam, vlm, "person",
                                    geometry=_geometry(), gates=gates) is False


def test_default_gates_do_not_reject():
    sam = _sam_result()
    gates = GeometryThresholds()
    assert gates.violations(_geometry()) == []
    assert decide_sample(sam, _vlm(), _geometry(), "person", gates=gates).status == "KEEP"


def test_failing_conditions_records_all():
    vlm = _vlm(class_label="", bbox_covers_object=False)
    decision = decide_sample(_sam_result(), vlm, _geometry(), "person")
    assert "object_present" in decision.failing_conditions
    assert "bbox_covers" in decision.failing_conditions


def test_ontology_never_changes_label():
    for label in ["person", "cat", "dog"]:
        decision = decide_sample(_sam_result(), _vlm(), _geometry(), label)
        assert decision.training_label == label
        assert decision.frigate_label == label


def test_decision_as_dict_serializable():
    decision = decide_sample(_sam_result(), _vlm(), _geometry(), "person")
    d = decision.as_dict()
    assert d["status"] == "KEEP"
    assert d["training_label"] == "person"
    assert d["bbox_source"] == "sam"
    import json

    json.dumps(d)