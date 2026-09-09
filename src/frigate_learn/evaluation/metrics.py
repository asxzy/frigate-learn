"""Detector-agnostic object-detection metrics (Phase 0).

The metric layer only knows about *predictions* (label, box, confidence) and
*ground truth* (label, box) — never about a specific detector framework, model
file, or inference engine. Latency/throughput/CPU are optional, benchmark-driven
fields added by P6's benchmark module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Protocol, Sequence, Tuple

Box = Tuple[float, float, float, float]  # normalized x1 y1 x2 y2


@dataclass(frozen=True)
class Prediction:
    label: str
    box: Box
    confidence: float


@dataclass(frozen=True)
class GroundTruth:
    label: str
    box: Box


@dataclass
class ClassMetrics:
    precision: float = 0.0
    recall: float = 0.0
    ap50: float = 0.0
    f1: float = 0.0
    count: int = 0


@dataclass
class DetectionMetrics:
    precision: float  # overall
    recall: float
    f1: float
    map50: float
    map50_95: float
    small_object_recall: float
    false_positive_rate: float
    per_class: Mapping[str, ClassMetrics] = field(default_factory=dict)
    total_gt: int = 0
    total_pred: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    # benchmark fields (filled by the P6 benchmark runner when available)
    latency_ms: float | None = None
    throughput_fps: float | None = None
    cpu_percent: float | None = None

    @property
    def mean(self) -> float:
        return self.map50

    def to_dict(self) -> dict:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "map50": self.map50,
            "map50_95": self.map50_95,
            "small_object_recall": self.small_object_recall,
            "false_positive_rate": self.false_positive_rate,
            "total_gt": self.total_gt,
            "total_pred": self.total_pred,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "latency_ms": self.latency_ms,
            "throughput_fps": self.throughput_fps,
            "per_class": {
                label: {
                    "precision": cm.precision,
                    "recall": cm.recall,
                    "ap50": cm.ap50,
                    "f1": cm.f1,
                    "count": cm.count,
                }
                for label, cm in self.per_class.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "DetectionMetrics":
        per_class = {
            label: ClassMetrics(**{k: v for k, v in row.items()})
            for label, row in (data.get("per_class") or {}).items()
        }
        return cls(
            precision=data.get("precision", 0.0),
            recall=data.get("recall", 0.0),
            f1=data.get("f1", 0.0),
            map50=data.get("map50", 0.0),
            map50_95=data.get("map50_95", 0.0),
            small_object_recall=data.get("small_object_recall", 0.0),
            false_positive_rate=data.get("false_positive_rate", 0.0),
            per_class=per_class,
            total_gt=data.get("total_gt", 0),
            total_pred=data.get("total_pred", 0),
            true_positives=data.get("true_positives", 0),
            false_positives=data.get("false_positives", 0),
            false_negatives=data.get("false_negatives", 0),
            latency_ms=data.get("latency_ms"),
            throughput_fps=data.get("throughput_fps"),
            cpu_percent=data.get("cpu_percent"),
        )


def iou(a: Box, b: Box) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def box_area(box: Box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


# --- matching -------------------------------------------------------------


def _match_class(
    preds: Sequence[Prediction],
    truths: Sequence[GroundTruth],
    iou_threshold: float,
    matched_gt: set[int],
) -> list[int]:
    """Greedy IoU matching for a single class.

    Returns index of the matched ground truth for each prediction (-1 otherwise).
    """
    matches: list[int] = []
    for p in preds:
        best = -1
        best_iou = iou_threshold
        for t_i, t in enumerate(truths):
            if t_i in matched_gt:
                continue
            score = iou(p.box, t.box)
            if score > best_iou:
                best = t_i
                best_iou = score
        if best >= 0:
            matched_gt.add(best)
        matches.append(best)
    return matches


def _ap_from_curve(scores: Sequence[float], matches: Sequence[bool]) -> float:
    """101-point (COCO-style) interpolated average precision."""
    if not scores:
        return 0.0

    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    tp = 0
    fp = 0
    recalls: list[float] = []
    precisions: list[float] = []
    n_gt = matches.count(True)
    for idx in order:
        if matches[idx]:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / (tp + fp) if (tp + fp) else 0.0)
        recalls.append(tp / n_gt if n_gt else 0.0)

    recall_levels = [i / 100.0 for i in range(101)]
    precision_at_recall = []
    for r in recall_levels:
        candidates = [p for p, rec in zip(precisions, recalls) if rec >= r]
        if candidates:
            precision_at_recall.append(max(candidates))
        else:
            # AP starts at 0 when the curve does not reach this recall level
            precision_at_recall.append(0.0)
    return sum(precision_at_recall) / len(recall_levels)


# --- public evaluation ----------------------------------------------------


def evaluate(
    predictions: Sequence[Prediction],
    ground_truth: Sequence[GroundTruth],
    iou_thresholds: Sequence[float] = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95),
    small_object_max_area: float = 0.0025,
    classes: Iterable[str] | None = None,
) -> DetectionMetrics:
    """Compute COCO-style metrics for a set of predictions / ground truth.

    ``small_object_max_area`` is a *fraction of the frame*; boxes whose normalized
    area is below it count as small objects for ``small_object_recall``
    (0.0025 ~ COCO's "<32x32 px on a 640px frame").
    """
    class_names = list(classes) if classes is not None else sorted(
        {p.label for p in predictions} | {g.label for g in ground_truth}
    )
    thresholds = list(iou_thresholds)
    if 0.5 not in thresholds:
        thresholds = sorted(thresholds + [0.5])

    per_class: dict[str, ClassMetrics] = {}
    ap50s: list[float] = []
    aps: list[float] = []
    total_tp = total_fp = total_fn = 0
    small_gt = [g for g in ground_truth if box_area(g.box) < small_object_max_area]
    small_tp = 0
    counted_small = set()  # dedupe small GT across the IoU50 global pass

    for label in class_names:
        preds = sorted(
            (p for p in predictions if p.label == label), key=lambda p: p.confidence, reverse=True
        )
        truths = [g for g in ground_truth if g.label == label]
        confs = [p.confidence for p in preds]

        if not truths:
            per_class[label] = ClassMetrics(count=0)
            continue

        # mAP over all thresholds (fresh matching per threshold)
        class_aps = []
        for threshold in thresholds:
            matches = _match_class(preds, truths, threshold, set())
            class_aps.append(_ap_from_curve(confs, [m >= 0 for m in matches]))
        aps.append(sum(class_aps) / len(class_aps))

        # IoU50 operating point: per-class confusion + AP50
        matches = _match_class(preds, truths, 0.5, set())
        tp = sum(1 for m in matches if m >= 0)
        fp = len(preds) - tp
        fn = len(truths) - tp
        total_tp += tp
        total_fp += fp
        total_fn += fn

        for m in matches:
            if m >= 0 and m not in counted_small and box_area(truths[m].box) < small_object_max_area:
                small_tp += 1
                counted_small.add(m)

        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        cm = ClassMetrics(
            count=len(truths),
            precision=p,
            recall=r,
            ap50=_ap_from_curve(confs, [m >= 0 for m in matches]),
            f1=2 * p * r / (p + r) if (p + r) else 0.0,
        )
        per_class[label] = cm
        ap50s.append(cm.ap50)

    overall_map50 = sum(ap50s) / len(ap50s) if ap50s else 0.0
    overall_map = sum(aps) / len(aps) if aps else 0.0

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / len(ground_truth) if ground_truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return DetectionMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        map50=overall_map50,
        map50_95=overall_map,
        small_object_recall=(small_tp / len(small_gt)) if small_gt else 0.0,
        false_positive_rate=1.0 - precision if (total_tp + total_fp) else 0.0,
        per_class=per_class,
        total_gt=len(ground_truth),
        total_pred=len(predictions),
        true_positives=total_tp,
        false_positives=total_fp,
        false_negatives=total_fn,
    )


# --- framework-independent evaluator interface ----------------------------


class Evaluator(Protocol):
    """Anything that computes evaluation metrics from predictions + labels.

    A concrete detector backend (ultralytics, onnx, Hailo runtime, ...) can be
    wrapped behind a class implementing this protocol — the golden-set
    acceptance gate (P11) only depends on this interface.
    """

    def evaluate(
        self, predictions: Sequence[Prediction], ground_truth: Sequence[GroundTruth]
    ) -> DetectionMetrics: ...


class DetectionEvaluator:
    """Reference evaluator: pure, dependency-free implementation."""

    def __init__(
        self,
        iou_thresholds: Sequence[float] = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95),
        small_object_max_area: float = 0.0025,
        classes: Iterable[str] | None = None,
    ) -> None:
        self.iou_thresholds = iou_thresholds
        self.small_object_max_area = small_object_max_area
        self.classes = classes

    def evaluate(
        self, predictions: Sequence[Prediction], ground_truth: Sequence[GroundTruth]
    ) -> DetectionMetrics:
        return evaluate(
            predictions,
            ground_truth,
            iou_thresholds=self.iou_thresholds,
            small_object_max_area=self.small_object_max_area,
            classes=self.classes,
        )


__all__ = [
    "Box",
    "Prediction",
    "GroundTruth",
    "ClassMetrics",
    "DetectionMetrics",
    "iou",
    "box_area",
    "evaluate",
    "Evaluator",
    "DetectionEvaluator",
]