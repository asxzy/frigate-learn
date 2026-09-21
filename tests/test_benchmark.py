"""Benchmark-level multi-object ground truth tests.

A golden image carrying several boxes must produce one GroundTruth per box,
and a model must be scored against every box: predicting only one of two
objects can never reach 100% recall.
"""

from __future__ import annotations

from frigate_learn.dataset.yolo import YoloLine
from frigate_learn.evaluation.benchmark import (
    BenchmarkConfig,
    evaluate_backend,
    golden_to_examples,
)
from frigate_learn.evaluation.golden import GoldenDataset
from frigate_learn.evaluation.metrics import Prediction


class FakeBackend:
    name = "fake"

    def __init__(self, per_image_predictions):
        self.per_image_predictions = per_image_predictions

    def predict(self, image_path, confidence=0.25):  # noqa: ARG002
        return self.per_image_predictions


def _build_multi_gt_golden(tmp_path):
    src = tmp_path / "src.jpg"
    src.write_bytes(b"\xff\xd8\xff\xe0fakejpeg")
    ds = GoldenDataset.create(tmp_path / "golden", ["person", "car"])
    ds.add_image(
        "s1",
        src,
        [
            YoloLine(0, 0.5, 0.5, 0.2, 0.2),   # person
            YoloLine(1, 0.15, 0.7, 0.3, 0.3),  # car
        ],
        camera="front",
        timestamp=1.0,
    )
    return ds


def _build_two_person_golden(tmp_path):
    src = tmp_path / "src.jpg"
    src.write_bytes(b"\xff\xd8\xff\xe0fakejpeg")
    ds = GoldenDataset.create(tmp_path / "golden", ["person"])
    ds.add_image(
        "s1",
        src,
        [
            YoloLine(0, 0.25, 0.5, 0.2, 0.2),   # person left
            YoloLine(0, 0.75, 0.5, 0.2, 0.2),   # person right
        ],
        camera="front",
        timestamp=1.0,
    )
    return ds


def _examples(ds):
    examples = golden_to_examples(ds, ["person", "car"])
    assert len(examples) == 1
    return examples


def test_golden_to_examples_returns_every_box(tmp_path):
    ds = _build_multi_gt_golden(tmp_path)
    examples = _examples(ds)
    truths = examples[0].ground_truth
    assert len(truths) == 2
    labels = sorted(t.label for t in truths)
    assert labels == ["car", "person"]


def test_model_predicting_both_boxes_scores_full_recall(tmp_path):
    examples = _examples(_build_multi_gt_golden(tmp_path))
    backend = FakeBackend([
        Prediction("person", (0.4, 0.4, 0.6, 0.6), 0.9),
        Prediction("car", (0.0, 0.55, 0.3, 0.85), 0.8),
    ])
    metrics = evaluate_backend(
        backend, examples, config=BenchmarkConfig(classes=["person", "car"])
    )
    assert metrics.true_positives == 2
    assert metrics.false_negatives == 0
    assert metrics.recall == 1.0
    assert metrics.map50 == 1.0


def test_model_predicting_only_one_box_not_full_recall(tmp_path):
    examples = _examples(_build_multi_gt_golden(tmp_path))
    backend = FakeBackend([
        Prediction("person", (0.4, 0.4, 0.6, 0.6), 0.9),
    ])
    metrics = evaluate_backend(
        backend, examples, config=BenchmarkConfig(classes=["person", "car"])
    )
    assert metrics.true_positives == 1
    assert metrics.false_negatives == 1
    assert metrics.recall == 0.5
    assert metrics.map50 < 1.0


def test_two_same_label_ground_truths_scored_independently(tmp_path):
    examples = _examples(_build_two_person_golden(tmp_path))
    both = FakeBackend([
        Prediction("person", (0.15, 0.4, 0.35, 0.6), 0.9),
        Prediction("person", (0.65, 0.4, 0.85, 0.6), 0.85),
    ])
    full = evaluate_backend(
        both, examples, config=BenchmarkConfig(classes=["person"])
    )
    assert full.true_positives == 2
    assert full.false_negatives == 0
    assert full.recall == 1.0

    one = FakeBackend([
        Prediction("person", (0.15, 0.4, 0.35, 0.6), 0.9),
    ])
    partial = evaluate_backend(
        one, examples, config=BenchmarkConfig(classes=["person"])
    )
    assert partial.true_positives == 1
    assert partial.false_negatives == 1
    assert partial.recall == 0.5