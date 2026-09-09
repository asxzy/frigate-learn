"""Detection metrics tests."""

from __future__ import annotations

import pytest

from frigate_learn.evaluation.metrics import (
    DetectionEvaluator,
    GroundTruth,
    Prediction,
    box_area,
    evaluate,
    iou,
)


def test_iou():
    a = (0.0, 0.0, 1.0, 1.0)
    assert iou(a, a) == pytest.approx(1.0)
    # two unit squares overlapping by half their area
    assert iou(a, (0.5, 0.0, 1.5, 1.0)) == pytest.approx(1 / 3)
    assert iou(a, (2.0, 2.0, 3.0, 3.0)) == pytest.approx(0.0)


def test_perfect_detection():
    gt = [GroundTruth("person", (0.1, 0.1, 0.5, 0.8)), GroundTruth("car", (0.2, 0.2, 0.6, 0.6))]
    preds = [
        Prediction("person", (0.1, 0.1, 0.5, 0.8), 0.99),
        Prediction("car", (0.2, 0.2, 0.6, 0.6), 0.98),
    ]
    m = evaluate(preds, gt)
    assert m.precision == pytest.approx(1.0)
    assert m.recall == pytest.approx(1.0)
    assert m.f1 == pytest.approx(1.0)
    assert m.map50 == pytest.approx(1.0)
    assert m.map50_95 == pytest.approx(1.0)
    assert m.true_positives == 2
    assert m.false_positives == 0
    assert m.false_negatives == 0
    assert m.total_gt == 2


def test_no_predictions():
    gt = [GroundTruth("person", (0.1, 0.1, 0.5, 0.8))]
    m = evaluate([], gt)
    assert m.recall == 0.0
    assert m.precision == 0.0
    assert m.true_positives == 0
    assert m.false_negatives == 1


def test_false_positive():
    gt = [GroundTruth("person", (0.1, 0.1, 0.3, 0.3),)]
    preds = [Prediction("person", (0.7, 0.7, 0.9, 0.9), 0.9)]
    m = evaluate(preds, gt)
    assert m.true_positives == 0
    assert m.false_positives == 1
    assert m.precision == 0.0
    assert m.recall == 0.0
    assert m.map50 == 0.0


def test_partial_precision_recall():
    gt = [GroundTruth("person", (0.1, 0.1, 0.3, 0.3)), GroundTruth("person", (0.6, 0.6, 0.9, 0.9))]
    preds = [
        Prediction("person", (0.1, 0.1, 0.3, 0.3), 0.9),  # TP
        Prediction("person", (0.4, 0.4, 0.55, 0.55), 0.7),  # FP
    ]
    m = evaluate(preds, gt)
    assert m.true_positives == 1
    assert m.false_positives == 1
    assert m.false_negatives == 1
    assert m.precision == pytest.approx(0.5)
    assert m.recall == pytest.approx(0.5)


def test_no_ground_truth():
    preds = [Prediction("person", (0.1, 0.1, 0.3, 0.3), 0.9)]
    m = evaluate(preds, [])
    assert m.map50 == 0.0
    assert m.total_gt == 0
    assert "person" in m.per_class
    assert m.per_class["person"].count == 0


def test_per_class_metrics():
    gt = [GroundTruth("person", (0.1, 0.1, 0.3, 0.3))]
    preds = [Prediction("person", (0.1, 0.1, 0.3, 0.3), 0.9)]
    m = evaluate(preds, gt)
    assert m.per_class["person"].recall == pytest.approx(1.0)
    assert m.per_class["person"].count == 1


def test_small_object_recall():
    gt = [
        GroundTruth("person", (0.01, 0.01, 0.02, 0.02)),  # small
        GroundTruth("person", (0.01, 0.5, 0.02, 0.53)),   # small, missed
        GroundTruth("car", (0.1, 0.1, 0.6, 0.6)),         # big
    ]
    preds = [
        Prediction("person", (0.01, 0.01, 0.02, 0.02), 0.9),
        Prediction("car", (0.1, 0.1, 0.6, 0.6), 0.9),
    ]
    m = evaluate(preds, gt)
    assert m.small_object_recall == pytest.approx(0.5)


def test_reference_evaluator_matches_function():
    gt = [GroundTruth("person", (0.1, 0.1, 0.3, 0.3))]
    preds = [Prediction("person", (0.1, 0.1, 0.3, 0.3), 0.9)]
    direct = evaluate(preds, gt)
    wrapped = DetectionEvaluator().evaluate(preds, gt)
    assert direct.map50 == pytest.approx(wrapped.map50)
    assert direct.recall == pytest.approx(wrapped.recall)


def test_map50_equals_iou50_only_eval():
    gt = [GroundTruth("person", (0.1, 0.1, 0.3, 0.3)), GroundTruth("car", (0.2, 0.6, 0.7, 0.9))]
    preds = [
        Prediction("person", (0.11, 0.1, 0.3, 0.31), 0.95),
        Prediction("car", (0.6, 0.6, 0.9, 0.9), 0.6),
        Prediction("person", (0.5, 0.5, 0.7, 0.7), 0.3),
    ]
    full = evaluate(preds, gt)
    iou50 = evaluate(preds, gt, iou_thresholds=[0.5])
    assert full.map50 == pytest.approx(iou50.map50)


def test_box_area():
    assert box_area((0.0, 0.0, 0.5, 0.5)) == pytest.approx(0.25)
    assert box_area((0.0, 0.0, 0.0, 1.0)) == 0.0