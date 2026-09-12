"""Deploy pipeline tests (Phase 12, dry-run only)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from frigate_learn.deploy.hailo import (
    HailoCompilerMissing,
    compile_hailo,
    deploy,
    export_onnx,
    render_frigate_detector_config,
)
from frigate_learn.evaluation.benchmark import CandidateResult
from frigate_learn.evaluation.gate import GateResult, evaluate_gate
from frigate_learn.evaluation.metrics import DetectionMetrics
from frigate_learn.models import Deployment


def test_render_frigate_detector_config():
    snippet = render_frigate_detector_config(
        "yolov8n-custom", "yolov8n-custom.hef", ["person", "car"], imgsz=320
    )
    assert "type: hailo" in snippet
    assert snippet.find("hef_path: /usr/share/frigate/models/yolov8n-custom.hef") != -1
    assert "num_classes: 2" in snippet
    assert "model input: 320x320" in snippet


def test_export_onnx_requires_explicit_imgsz(tmp_path):
    onnx = Path("model.onnx")
    assert export_onnx(onnx, tmp_path, imgsz=320) == onnx
    with pytest.raises(TypeError):
        export_onnx(onnx, tmp_path)


def test_deploy_resolves_imgsz_from_training_config(config, db, tmp_path):
    weights = tmp_path / "best.pt"
    weights.write_text("# fake weights", encoding="utf-8")

    outcome = deploy(
        config, db,
        model_name="yolov8n-nightly",
        weights=weights,
        version="v001",
        dry_run=True,
        out_dir=tmp_path / "models" / "yolov8n-nightly",
    )
    manifest = json.loads(outcome.manifest_path.read_text(encoding="utf-8"))
    assert manifest["imgsz"] == config.training.image_size
    assert f"model input: {config.training.image_size}x{config.training.image_size}" in outcome.config_snippet


def test_deploy_explicit_imgsz_override(config, db, tmp_path):
    weights = tmp_path / "best.pt"
    weights.write_text("# fake weights", encoding="utf-8")

    outcome = deploy(
        config, db,
        model_name="yolov8n-nightly",
        weights=weights,
        version="v001",
        imgsz=640,
        dry_run=True,
        out_dir=tmp_path / "models" / "yolov8n-nightly",
    )
    manifest = json.loads(outcome.manifest_path.read_text(encoding="utf-8"))
    assert manifest["imgsz"] == 640


def test_compile_hailo_dry_run_writes_placeholder(tmp_path):
    onnx = tmp_path / "model.onnx"
    onnx.write_text("# fake onnx", encoding="utf-8")
    hef = compile_hailo(onnx, tmp_path / "out", dry_run=True)
    assert hef.exists()
    assert hef.suffix == ".hef"


def test_compile_hailo_without_binary_raises(tmp_path):
    onnx = tmp_path / "model.onnx"
    onnx.write_text("# fake onnx", encoding="utf-8")
    with pytest.raises(HailoCompilerMissing):
        compile_hailo(onnx, tmp_path / "out", dry_run=False)


def test_deploy_dry_run_writes_manifest_and_hef(config, db, tmp_path):
    weights = tmp_path / "best.pt"
    weights.write_text("# fake weights", encoding="utf-8")

    outcome = deploy(
        config, db,
        model_name="yolov8n-nightly",
        weights=weights,
        version="v001",
        imgsz=640,
        dry_run=True,
        out_dir=tmp_path / "models" / "yolov8n-nightly",
    )
    assert outcome.dry_run is True
    assert outcome.onnx_path.exists()
    assert outcome.hef_path.exists()

    manifest = json.loads(outcome.manifest_path.read_text(encoding="utf-8"))
    assert manifest["model_name"] == "yolov8n-nightly"
    assert manifest["version"] == "v001"
    assert manifest["dry_run"] is True
    assert len(manifest["hef_sha256"]) == 64

    snippet = (outcome.manifest_path.parent / "frigate-detector.yml").read_text(encoding="utf-8")
    assert "type: hailo" in snippet


def test_render_coco80_num_classes_80(config):
    snippet = render_frigate_detector_config(
        "m", "m.hef", config.model_class_names(), imgsz=320
    )
    assert "num_classes: 80" in snippet


def test_deploy_manifest_label_space_and_num_classes(config, db, tmp_path):
    weights = tmp_path / "best.pt"
    weights.write_text("# fake weights", encoding="utf-8")

    outcome = deploy(
        config, db,
        model_name="yolov8n-coco80",
        weights=weights,
        version="v001",
        dry_run=True,
        out_dir=tmp_path / "models" / "yolov8n-coco80",
    )
    manifest = json.loads(outcome.manifest_path.read_text(encoding="utf-8"))
    assert manifest["label_space"] == "coco80"
    assert manifest["num_classes"] == 80
    snippet = (outcome.manifest_path.parent / "frigate-detector.yml").read_text(encoding="utf-8")
    assert "num_classes: 80" in snippet


def test_deploy_subset_compact(config, db, tmp_path):
    config.training.label_space = "subset"
    weights = tmp_path / "best.pt"
    weights.write_text("# fake weights", encoding="utf-8")

    outcome = deploy(
        config, db,
        model_name="yolov8n-subset",
        weights=weights,
        version="v001",
        dry_run=True,
        out_dir=tmp_path / "models" / "yolov8n-subset",
    )
    manifest = json.loads(outcome.manifest_path.read_text(encoding="utf-8"))
    assert manifest["label_space"] == "subset"
    assert manifest["num_classes"] == len(config.classes)
    snippet = (outcome.manifest_path.parent / "frigate-detector.yml").read_text(encoding="utf-8")
    assert f"num_classes: {len(config.classes)}" in snippet


def test_deploy_records_deployment_row_when_gate_given(config, db, tmp_path):
    weights = tmp_path / "best.pt"
    weights.write_text("# fake", encoding="utf-8")
    candidate = CandidateResult(
        name="yolov8n",
        metrics=DetectionMetrics(0.5, 0.6, 0.55, 0.6, 0.3, 0.1, 0.05, latency_ms=4.0),
        version="v001",
    )
    gate = evaluate_gate(candidate, config.deployment, None)
    outcome = deploy(
        config, db, model_name="yolov8n", weights=weights, version="v001",
        gate=gate, result=candidate, dry_run=True,
        out_dir=tmp_path / "models" / "yolov8n",
    )
    assert outcome.deployment_id is not None
    with db.session() as s:
        row = s.get(Deployment, outcome.deployment_id)
    assert row is not None
    assert row.verdict == "PASS"
    assert row.artifact_path == str(outcome.hef_path)