"""Trainer before/after golden evaluation tests."""

from __future__ import annotations

import json

from frigate_learn.evaluation.benchmark import CandidateResult
from frigate_learn.evaluation.metrics import DetectionMetrics
from frigate_learn.training.trainer import (
    _safe_before_after,
    _write_before_after,
    evaluate_on_golden,
)


def test_evaluate_on_golden_missing_golden_dir_returns_none(config):
    assert evaluate_on_golden(config, "/nonexistent/weights.pt", name="yolov8n-v001") is None


def test_evaluate_on_golden_returns_metrics(config, monkeypatch):
    golden = config.golden_dir() / config.evaluation.golden_dataset
    golden.mkdir(parents=True)

    from frigate_learn.evaluation import backends, golden as golden_mod

    class DummyBackend:
        def __init__(self, weights, *, imgsz, device=None, name=None):
            self._name = name or weights

        @property
        def name(self):
            return self._name

        def predict(self, image_path, confidence=0.25):
            return []

    class FakeGolden:
        def samples(self):
            return []

    monkeypatch.setattr(golden_mod.GoldenDataset, "load", lambda path: FakeGolden())
    monkeypatch.setattr(backends, "UltralyticsBackend", DummyBackend)
    monkeypatch.setattr(
        "frigate_learn.evaluation.benchmark.golden_to_examples",
        lambda golden, classes, limit=None: [object()],
    )
    monkeypatch.setattr(
        "frigate_learn.evaluation.benchmark.benchmark_candidate",
        lambda backend, examples, *, version, kind="golden", config=None, evaluator=None: (
            CandidateResult(
                name=backend.name,
                metrics=DetectionMetrics(
                    precision=0.9, recall=0.85, f1=0.87, map50=0.91, map50_95=0.5,
                    small_object_recall=0.8, false_positive_rate=0.1,
                ),
                version=version,
                kind=kind,
            )
        ),
    )
    result = evaluate_on_golden(config, "/w.pt", name="yolov8n-v001")
    assert result == {"name": "yolov8n-v001", "map50": 0.91, "recall": 0.85, "latency_ms": None}


def test_write_before_after_writes_json(config, tmp_path, monkeypatch):
    run_dir = tmp_path / "runs" / "x"
    run_dir.mkdir(parents=True)
    monkeypatch.setattr(
        "frigate_learn.training.trainer.evaluate_on_golden",
        lambda config, weights, *, name: (
            {"name": name, "map50": 0.5, "recall": 0.4, "latency_ms": 12.0}
            if name.endswith("-before")
            else {"name": name, "map50": 0.8, "recall": 0.7, "latency_ms": 10.0}
        ),
    )
    ok = _write_before_after(config, "b.pt", "a.pt", run_dir, name="yolov8n-v003")
    assert ok is True
    payload = json.loads((run_dir / "before_after.json").read_text(encoding="utf-8"))
    assert payload["delta"] == {"map50": 0.3, "recall": 0.3}
    assert payload["before"]["name"] == "yolov8n-v003-before"
    assert payload["after"]["name"] == "yolov8n-v003"


def test_write_before_after_missing_golden_returns_false(config, tmp_path, monkeypatch):
    run_dir = tmp_path / "runs" / "x"
    run_dir.mkdir(parents=True)
    monkeypatch.setattr(
        "frigate_learn.training.trainer.evaluate_on_golden",
        lambda config, weights, *, name: None,
    )
    assert _write_before_after(config, "b.pt", "a.pt", run_dir, name="y") is False
    assert not (run_dir / "before_after.json").exists()


def test_safe_before_after_survives_golden_failure(config, tmp_path, monkeypatch):
    run_dir = tmp_path / "runs" / "x"
    run_dir.mkdir(parents=True)

    def _boom(config, weights, *, name):
        raise RuntimeError("corrupt golden")

    monkeypatch.setattr("frigate_learn.training.trainer.evaluate_on_golden", _boom)
    assert _safe_before_after(config, "b.pt", "a.pt", run_dir, name="y") is False
    assert not (run_dir / "before_after.json").exists()