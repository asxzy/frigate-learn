"""Candidate benchmark runner (Phase 6).

A benchmark measures a detector on a fixed ground-truth set (the golden dataset,
or a dataset version's ``val`` split) and produces a ``CandidateResult``:
metrics + latency/throughput. The Pareto frontier picks the points that no other
candidate dominates on (accuracy, latency); the acceptance gate (P11) consumes
``CandidateResult`` objects.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

from ..logutil import info
from .golden import GoldenDataset
from .metrics import (
    DetectionMetrics,
    Evaluator,
    GroundTruth,
    Prediction,
)

DEFAULT_CONFIDENCE = 0.25


@dataclass
class Example:
    image_path: Path
    ground_truth: list[GroundTruth]


class ModelBackend(Protocol):
    """A detector wrapped for benchmarking: image path -> predictions."""

    def predict(self, image_path: Path, confidence: float = DEFAULT_CONFIDENCE) -> list[Prediction]: ...

    @property
    def name(self) -> str: ...


@dataclass
class CandidateResult:
    name: str
    metrics: DetectionMetrics
    version: str = ""      # dataset/golden version evaluated on
    kind: str = "golden"   # golden | dataset

    @property
    def map50(self) -> float:
        return self.metrics.map50

    @property
    def latency_ms(self) -> float | None:
        return self.metrics.latency_ms

    @property
    def recall(self) -> float:
        return self.metrics.recall

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "kind": self.kind,
            "metrics": self.metrics.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CandidateResult":
        return cls(
            name=data["name"],
            metrics=DetectionMetrics.from_dict(data["metrics"]),
            version=data.get("version", ""),
            kind=data.get("kind", "golden"),
        )


def collect_examples_from_golden(
    golden: GoldenDataset, classes: Sequence[str], limit: int | None = None
) -> list[Example]:
    """Load (image, ground-truth) examples from a golden dataset (deprecated
    alias of ``golden_to_examples``)."""
    return golden_to_examples(golden, classes, limit=limit)


def golden_to_examples(golden: GoldenDataset, classes: Sequence[str], limit: int | None = None) -> list[Example]:
    from ..dataset.yolo import read_yolo_label

    class_name = {i: name for i, name in enumerate(classes)}
    examples: list[Example] = []
    for sample in golden.samples():
        lines = read_yolo_label(sample.label_path, num_classes=len(classes))
        gt = []
        for line in lines:
            name = class_name.get(line.class_id)
            if name is None:
                continue
            x1, y1, x2, y2 = line.to_xyxy()
            gt.append(GroundTruth(label=name, box=(x1, y1, x2, y2)))
        examples.append(Example(image_path=sample.image_path, ground_truth=gt))
        if limit is not None and len(examples) >= limit:
            break
    info("golden examples loaded", count=len(examples))
    return examples


@dataclass
class BenchmarkConfig:
    confidence: float = DEFAULT_CONFIDENCE
    warmup: int = 3
    repeats: int = 30
    small_object_max_area: float = 0.0025
    classes: Sequence[str] = field(default_factory=list)


def evaluate_backend(
    backend: ModelBackend,
    examples: Sequence[Example],
    config: BenchmarkConfig | None = None,
    evaluator: Evaluator | None = None,
) -> DetectionMetrics:
    """Run a backend over examples; measure metrics and single-image latency."""
    config = config or BenchmarkConfig()
    predictions: list[Prediction] = []
    latencies: list[float] = []
    truths: list[GroundTruth] = []

    total = len(examples)
    info("benchmark started", candidate=backend.name, examples=total)
    log_every = max(1, total // 10)

    for _ in range(config.warmup):
        if examples:
            backend.predict(examples[0].image_path, confidence=config.confidence)

    for index, example in enumerate(examples, start=1):
        truths.extend(example.ground_truth)
        start = time.perf_counter()
        preds = backend.predict(example.image_path, confidence=config.confidence)
        latencies.append((time.perf_counter() - start) * 1000.0)
        predictions.extend(preds)
        if index % log_every == 0 or index == total:
            info(
                "benchmark progress",
                candidate=backend.name,
                processed=index,
                total=total,
            )

    metrics = (evaluator or _default_evaluator(config)).evaluate(predictions, truths)
    metrics.latency_ms = statistics.mean(latencies) if latencies else None
    if metrics.latency_ms:
        metrics.throughput_fps = 1000.0 / metrics.latency_ms
    info(
        "benchmark finished",
        candidate=backend.name,
        processed=total,
        map50=round(metrics.map50, 4),
        recall=round(metrics.recall, 4),
        latency_ms=round(metrics.latency_ms, 2) if metrics.latency_ms else None,
    )
    return metrics


def _default_evaluator(config: BenchmarkConfig) -> Evaluator:
    from .metrics import DetectionEvaluator

    return DetectionEvaluator(
        small_object_max_area=config.small_object_max_area,
        classes=list(config.classes) or None,
    )


def benchmark_candidate(
    backend: ModelBackend,
    examples: Sequence[Example],
    *,
    version: str,
    kind: str = "golden",
    config: BenchmarkConfig | None = None,
    evaluator: Evaluator | None = None,
) -> CandidateResult:
    metrics = evaluate_backend(backend, examples, config=config, evaluator=evaluator)
    return CandidateResult(name=backend.name, metrics=metrics, version=version, kind=kind)


def latest_trained_run(config) -> tuple[str, Path] | None:
    root = config.resolve(config.data.root, "training")
    if not root.is_dir():
        return None
    best_entry: tuple[str, Path, float] | None = None
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        weights = entry / "weights" / "best.pt"
        if not weights.is_file():
            continue
        csv = entry / "results.csv"
        mtime = csv.stat().st_mtime if csv.is_file() else weights.stat().st_mtime
        if best_entry is None or mtime > best_entry[2]:
            best_entry = (entry.name, weights, mtime)
    return None if best_entry is None else (best_entry[0], best_entry[1])


# --- Pareto frontier -------------------------------------------------------


@dataclass
class FrontierPoint:
    name: str
    map50: float
    recall: float
    latency_ms: float | None
    dominated_by: list[str] = field(default_factory=list)


def pareto_frontier(results: Sequence[CandidateResult]) -> list[FrontierPoint]:
    """Return the set of candidates not dominated on (map50, recall, -latency).

    A candidate dominates another when it is strictly better or equal on map50,
    recall and *lower* latency, and better on at least one. Lower latency is
    better, so we negate it before comparing.
    """
    def key(r: CandidateResult):
        return (r.map50, r.recall, -(r.latency_ms or float("inf")))

    points: list[FrontierPoint] = []
    for r in results:
        dominated = [
            other.name
            for other in results
            if other.name != r.name
            and key(other)[0] >= key(r)[0]
            and key(other)[1] >= key(r)[1]
            and key(other)[2] >= key(r)[2]
            and (key(other)[0] > key(r)[0] or key(other)[1] > key(r)[1] or key(other)[2] > key(r)[2])
        ]
        points.append(
            FrontierPoint(
                name=r.name,
                map50=r.map50,
                recall=r.recall,
                latency_ms=r.latency_ms,
                dominated_by=dominated,
            )
        )
    frontier = [p for p in points if not p.dominated_by]
    frontier.sort(key=lambda p: (-p.map50, p.latency_ms or 0.0))
    return frontier


# --- JSON helpers ----------------------------------------------------------


def results_to_json(results: Sequence[CandidateResult], path: Path) -> None:
    payload = {"results": [r.to_dict() for r in results]}
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


def results_from_json(path: Path | str) -> list[CandidateResult]:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return [CandidateResult.from_dict(r) for r in data.get("results", [])]


__all__ = [
    "Example",
    "ModelBackend",
    "CandidateResult",
    "BenchmarkConfig",
    "FrontierPoint",
    "evaluate_backend",
    "benchmark_candidate",
    "pareto_frontier",
    "results_to_json",
    "results_from_json",
    "golden_to_examples",
    "collect_examples_from_golden",
    "latest_trained_run",
]