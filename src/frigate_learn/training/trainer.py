"""Ultralytics training wrapper (Phase 7).

Thin, deliberate wrapper: ``train`` builds one candidate on a built dataset
version. torch / ultralytics are only imported when actually training, so a
collection-only install (or the test suite) never needs them. ``dry_run``
produces a credible placeholder run directory for smoke tests and empty-machine
dry marks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..config import AppConfig
from ..db import Database
from ..logutil import info
from .candidates import resolve


@dataclass
class TrainResult:
    model_name: str
    dataset_version: str
    output_dir: Path
    weights_path: Path | None
    metrics: dict = field(default_factory=dict)
    dry_run: bool = False


def evaluate_on_golden(config: AppConfig, weights: str, *, name: str) -> dict | None:
    golden_dir = config.golden_dir() / config.evaluation.golden_dataset
    if not golden_dir.is_dir():
        return None
    try:
        from ..evaluation.backends import UltralyticsBackend
        from ..evaluation.benchmark import (
            BenchmarkConfig,
            benchmark_candidate,
            golden_to_examples,
        )
        from ..evaluation.golden import GoldenDataset
    except ImportError:
        return None

    golden = GoldenDataset.load(golden_dir)
    examples = golden_to_examples(golden, config.classes)
    if not examples:
        return None
    backend = UltralyticsBackend(
        weights, imgsz=config.training.image_size, name=name
    )
    result = benchmark_candidate(
        backend,
        examples,
        version=config.evaluation.golden_dataset,
        kind="golden",
        config=BenchmarkConfig(classes=list(config.classes)),
    )
    return {
        "name": result.name,
        "map50": result.map50,
        "recall": result.metrics.recall,
        "latency_ms": result.latency_ms,
    }


def _write_before_after(
    config: AppConfig,
    before_weights: str,
    after_weights: str,
    run_dir: Path,
    *,
    name: str,
) -> bool:
    before = evaluate_on_golden(config, before_weights, name=f"{name}-before")
    after = evaluate_on_golden(config, after_weights, name=name)
    if before is None or after is None:
        return False
    payload = {
        "before": before,
        "after": after,
        "delta": {
            "map50": round(after["map50"] - before["map50"], 4),
            "recall": round(after["recall"] - before["recall"], 4),
        },
    }
    (run_dir / "before_after.json").write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    info(
        "before/after on golden",
        before=f"{before['map50']:.3f}",
        after=f"{after['map50']:.3f}",
        delta=f"{payload['delta']['map50']:+.3f}",
    )
    return True


class Trainer:
    def __init__(self, config: AppConfig, db: Database) -> None:
        self.config = config
        self.db = db

    def train(
        self,
        model_name: str,
        dataset_version: str,
        *,
        epochs: int | None = None,
        imgsz: int | None = None,
        batch: int | None = None,
        device: str | None = None,
        freeze: int | None = None,
        seed: int | None = None,
        dry_run: bool = False,
        tag: str | None = None,
    ) -> TrainResult:
        cfg = self.config.training
        candidate = resolve(model_name)

        dataset_dir = self.config.datasets_dir() / dataset_version
        dataset_yaml = dataset_dir / "dataset.yaml"
        if not dataset_yaml.is_file():
            raise FileNotFoundError(
                f"dataset version {dataset_version!r} missing dataset.yaml at {dataset_yaml}"
            )

        project_dir = cfg.project
        if project_dir:
            project = Path(project_dir)
        else:
            project = self.config.resolve(self.config.data.root, "training")
        name = f"{model_name}-{tag or dataset_version}"
        run_dir = project / name
        run_dir.mkdir(parents=True, exist_ok=True)

        seed = seed or cfg.seed
        epochs = epochs or cfg.epochs
        imgsz = imgsz or cfg.image_size
        batch = batch or cfg.batch
        device = device or cfg.device
        freeze = freeze if freeze is not None else cfg.freeze

        info(
            "training launch",
            model=model_name,
            dataset=dataset_version,
            epochs=epochs,
            imgsz=imgsz,
            dry_run=dry_run,
        )

        if dry_run:
            return self._dry_run_result(candidate.weights, dataset_version, run_dir, tag=tag)

        try:
            from ultralytics import YOLO  # lazy import

            model = YOLO(candidate.weights)
            kwargs: dict = {
                "seed": seed,
            }
            if device is not None:
                kwargs["device"] = device
            if freeze is not None:
                kwargs["freeze"] = freeze
            results = model.train(
                data=str(dataset_yaml),
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                project=str(project),
                name=name,
                exist_ok=True,
                **kwargs,
            )
            best = Path(getattr(results, "save_dir", run_dir)) / "weights" / "best.pt"
            metrics = dict(getattr(results, "results_dict", {}) or {})
            if best.exists():
                _write_before_after(
                    self.config, str(candidate.weights), str(best), run_dir, name=name
                )
        except ImportError as exc:  # pragma: no cover - guarded by CLI precheck
            raise RuntimeError(
                "ultralytics/torch not installed; install the 'ml' extra on the "
                "training machine (pip install -e '.[ml]')"
            ) from exc

        info("training finished", model=model_name, best=str(best) if best.exists() else "n/a")
        return TrainResult(
            model_name=model_name,
            dataset_version=dataset_version,
            output_dir=run_dir,
            weights_path=best if best.exists() else None,
            metrics=metrics,
            dry_run=False,
        )

    @staticmethod
    def _dry_run_result(weights: str, dataset_version: str, run_dir: Path, tag: str | None) -> TrainResult:
        payload = {
            "dry_run": True,
            "model": weights,
            "dataset_version": dataset_version,
            "epochs": None,
            "metrics": {},
            "note": "dry run: no training executed; install the 'ml' extra to train",
        }
        (run_dir / "results.json").write_text(
            json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        return TrainResult(
            model_name=weights,
            dataset_version=dataset_version,
            output_dir=run_dir,
            weights_path=None,
            metrics={},
            dry_run=True,
        )


__all__ = ["Trainer", "TrainResult", "evaluate_on_golden", "_write_before_after"]