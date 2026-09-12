"""Trainer before/after golden evaluation tests."""

from __future__ import annotations

import json
from pathlib import Path

from frigate_learn.evaluation.benchmark import CandidateResult
from frigate_learn.evaluation.metrics import DetectionMetrics
from frigate_learn.training.trainer import (
    Trainer,
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


def _stub_dataset(config):
    dataset_yaml = config.datasets_dir() / "v001" / "dataset.yaml"
    dataset_yaml.parent.mkdir(parents=True)
    dataset_yaml.write_text("names:\n  0: person\n", encoding="utf-8")
    return dataset_yaml


class _FakeMaskedTrainer:
    instances = []

    def __init__(self, overrides, trainable):
        self.overrides = overrides
        self.trainable = trainable
        self._run_dir = Path(overrides["project"]) / overrides["name"]
        _FakeMaskedTrainer.instances.append(self)

    def train(self):
        best = self._run_dir / "weights" / "best.pt"
        best.parent.mkdir(parents=True)
        best.write_bytes(b"stub")
        self.best = best
        self.metrics = {"mAP50(B)": 0.9}


def test_train_coco80_uses_masked_trainer(config, db, monkeypatch):
    _stub_dataset(config)
    _FakeMaskedTrainer.instances.clear()
    monkeypatch.setattr(
        "frigate_learn.training.masked.MaskedDetectionTrainer", _FakeMaskedTrainer
    )
    result = Trainer(config, db).train("yolov8n", "v001")
    fake = _FakeMaskedTrainer.instances[0]
    overrides = fake.overrides
    dataset_yaml = config.datasets_dir() / "v001" / "dataset.yaml"
    assert overrides["data"] == str(dataset_yaml)
    assert overrides["name"] == "yolov8n-v001"
    assert overrides["imgsz"] == config.training.image_size
    assert overrides["model"] == "yolov8n.pt"
    assert overrides["seed"] == config.training.seed
    assert fake.trainable == config.trainable_class_mask()
    project = config.resolve(config.data.root, "training")
    assert result.weights_path == project / "yolov8n-v001" / "weights" / "best.pt"
    assert result.metrics == {"mAP50(B)": 0.9}


def test_train_coco80_forwards_lr_and_freeze(config, db, monkeypatch):
    config.training.lr0 = 0.001
    config.training.lrf = 0.05
    config.training.freeze = 10
    config.training.device = "cpu"
    _stub_dataset(config)
    _FakeMaskedTrainer.instances.clear()
    monkeypatch.setattr(
        "frigate_learn.training.masked.MaskedDetectionTrainer", _FakeMaskedTrainer
    )
    Trainer(config, db).train("yolov8n", "v001")
    overrides = _FakeMaskedTrainer.instances[0].overrides
    assert overrides["lr0"] == config.training.lr0
    assert overrides["lrf"] == config.training.lrf
    assert overrides["freeze"] == config.training.freeze
    assert overrides["device"] == config.training.device


def test_train_coco80_omits_none_overrides(config, db, monkeypatch):
    _stub_dataset(config)
    _FakeMaskedTrainer.instances.clear()
    monkeypatch.setattr(
        "frigate_learn.training.masked.MaskedDetectionTrainer", _FakeMaskedTrainer
    )
    Trainer(config, db).train("yolov8n", "v001")
    overrides = _FakeMaskedTrainer.instances[0].overrides
    assert "lr0" not in overrides
    assert "lrf" not in overrides
    assert "device" not in overrides
    assert "freeze" not in overrides
    assert overrides["seed"] == config.training.seed


def test_train_subset_keeps_yolo_path(config, db, monkeypatch):
    config.training.label_space = "subset"
    _stub_dataset(config)
    captured = {}

    class FakeObs:
        pass

    class FakeYOLO:
        def __init__(self, weights):
            captured["weights"] = weights

        def train(self, **kwargs):
            captured["kwargs"] = kwargs
            save_dir = Path(kwargs["project"]) / kwargs["name"]
            best = save_dir / "weights" / "best.pt"
            best.parent.mkdir(parents=True)
            best.write_bytes(b"stub")
            results = FakeObs()
            results.save_dir = save_dir
            results.results_dict = {"mAP50(B)": 0.8}
            return results

    monkeypatch.setattr("ultralytics.YOLO", FakeYOLO)
    result = Trainer(config, db).train("yolov8n", "v001")
    kwargs = captured["kwargs"]
    dataset_yaml = config.datasets_dir() / "v001" / "dataset.yaml"
    assert captured["weights"] == "yolov8n.pt"
    assert kwargs["data"] == str(dataset_yaml)
    assert kwargs["epochs"] == config.training.epochs
    assert kwargs["imgsz"] == config.training.image_size
    assert kwargs["batch"] == config.training.batch
    assert kwargs["project"] == str(config.resolve(config.data.root, "training"))
    assert kwargs["name"] == "yolov8n-v001"
    assert kwargs["exist_ok"] is True
    assert kwargs["seed"] == config.training.seed
    assert "device" not in kwargs
    assert "freeze" not in kwargs
    assert result.metrics == {"mAP50(B)": 0.8}
    expected_best = Path(kwargs["project"]) / "yolov8n-v001" / "weights" / "best.pt"
    assert result.weights_path == expected_best
