"""Deploy pipeline tests (Phase 12, dry-run + real-stage orchestration)."""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import pytest

from frigate_learn.deploy.hailo import (
    HailoCompileError,
    HailoCompilerMissing,
    compile_hailo,
    deploy,
    export_onnx,
    render_frigate_detector_config,
)
from frigate_learn.evaluation.benchmark import CandidateResult
from frigate_learn.evaluation.gate import evaluate_gate
from frigate_learn.evaluation.metrics import DetectionMetrics
from frigate_learn.models import Deployment


def _write_script(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _fake_hailo(bin_dir: Path, log_path: Path) -> Path:
    return _write_script(
        bin_dir / "hailo",
        "#!/usr/bin/env python3\n"
        "import os, pathlib, sys\n"
        f"log = pathlib.Path({str(log_path)!r})\n"
        "with log.open('a') as fh:\n"
        "    fh.write(' '.join(sys.argv[1:]) + '\\n')\n"
        "stage = sys.argv[1]\n"
        "if stage == 'parser':\n"
        "    net = sys.argv[sys.argv.index('--net-name') + 1]\n"
        "    pathlib.Path(net + '.har').write_bytes(b'har')\n"
        "elif stage == 'optimize':\n"
        "    out = sys.argv[sys.argv.index('--output-har-path') + 1]\n"
        "    pathlib.Path(out).write_bytes(b'har')\n"
        "elif stage == 'compiler':\n"
        "    outdir = sys.argv[sys.argv.index('--output-dir') + 1]\n"
        "    net = sys.argv[2].rsplit('_optimized', 1)[0]\n"
        "    pathlib.Path(outdir, net + '.hef').write_bytes(b'hef')\n"
        "else:\n"
        "    sys.exit(1)\n",
    )


def _fake_docker(bin_dir: Path, log_path: Path) -> Path:
    return _write_script(
        bin_dir / "docker",
        "#!/usr/bin/env python3\n"
        "import os, pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "idx = args.index('-v')\n"
        "host_dir = args[idx + 1].rsplit(':', 1)[0]\n"
        "image = args[args.index('-w') + 2]\n"
        "argv = args[idx + 5:]\n"
        f"log = pathlib.Path({str(log_path)!r})\n"
        "with log.open('a') as fh:\n"
        "    fh.write('|'.join(args[:idx + 1]) + '|' + image + '|' + ' '.join(argv) + '\\n')\n"
        "os.chdir(host_dir)\n"
        "stage = argv[1]\n"
        "if stage == 'parser':\n"
        "    net = argv[argv.index('--net-name') + 1]\n"
        "    pathlib.Path(net + '.har').write_bytes(b'har')\n"
        "elif stage == 'optimize':\n"
        "    out = argv[argv.index('--output-har-path') + 1]\n"
        "    pathlib.Path(out).write_bytes(b'har')\n"
        "elif stage == 'compiler':\n"
        "    outdir = argv[argv.index('--output-dir') + 1]\n"
        "    net = argv[2].rsplit('_optimized', 1)[0]\n"
        "    pathlib.Path(outdir, net + '.hef').write_bytes(b'hef')\n"
        "else:\n"
        "    sys.exit(1)\n",
    )


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


def test_deploy_real_manifest_records_hw_arch_and_hef_pending(config, db, tmp_path):
    config.hailo.hw_arch = "hailo8l"
    weights = tmp_path / "best.pt"
    weights.write_text("# fake", encoding="utf-8")

    outcome = deploy(
        config, db,
        model_name="yolov8n-nightly",
        weights=weights,
        version="v001",
        dry_run=True,
        out_dir=tmp_path / "models" / "yolov8n-nightly",
    )
    manifest = json.loads(outcome.manifest_path.read_text(encoding="utf-8"))
    assert manifest["hw_arch"] == "hailo8l"
    assert manifest["hef_pending"] is True
    assert len(manifest["onnx_sha256"]) == 64


def test_compile_hailo_runs_parser_optimize_compiler_natively(tmp_path, monkeypatch):
    onnx = tmp_path / "best.onnx"
    onnx.write_bytes(b"# fake onnx")
    bin_dir = tmp_path / "bin"
    log_path = tmp_path / "hailo.log"
    _fake_hailo(bin_dir, log_path)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")

    out_dir = tmp_path / "out"
    hef = compile_hailo(onnx, out_dir, dry_run=False)
    assert hef == out_dir / "best.hef"
    assert hef.read_bytes() == b"hef"

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert lines[0].startswith("parser onnx best.onnx --net-name best --hw-arch hailo8")
    assert lines[0].endswith("-y")
    assert "--use-random-calib-set" in lines[1]
    assert "--output-dir" in lines[2]


def test_compile_hailo_uses_calibration_set_when_dir_given(tmp_path, monkeypatch):
    pytest.importorskip("PIL")
    onnx = tmp_path / "best.onnx"
    onnx.write_bytes(b"# fake onnx")
    calib = tmp_path / "calib"
    nested = calib / "20260909" / "drive_way_west"
    nested.mkdir(parents=True)
    from PIL import Image

    Image.new("RGB", (24, 18), (90, 120, 200)).save(nested / "frame.jpg")
    Image.new("RGB", (20, 20), (10, 10, 10)).save(nested / "frame2.png")

    bin_dir = tmp_path / "bin"
    log_path = tmp_path / "hailo.log"
    _fake_hailo(bin_dir, log_path)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")

    out_dir = tmp_path / "out"
    compile_hailo(onnx, out_dir, dry_run=False, calib_dir=calib, imgsz=320, calib_samples=8)

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert "--calib-set-path calib.npy" in lines[1]
    npy = out_dir / "calib.npy"
    assert npy.exists()
    import numpy as np

    data = np.load(npy)
    assert data.shape == (2, 320, 320, 3)
    assert data.dtype == np.float32
    assert data.min() >= 0.0 and data.max() <= 1.0


def test_compile_hailo_dispatches_to_docker_on_macos(tmp_path, monkeypatch):
    onnx = tmp_path / "best.onnx"
    onnx.write_bytes(b"# fake onnx")
    bin_dir = tmp_path / "bin"
    log_path = tmp_path / "docker.log"
    _fake_docker(bin_dir, log_path)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setattr(sys, "platform", "darwin")

    out_dir = tmp_path / "out"
    hef = compile_hailo(
        onnx, out_dir, dry_run=False,
        docker_image="frigate-learn-hailo-dfc:3.34.0",
    )
    assert hef.read_bytes() == b"hef"

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert "--platform|--rm" in lines[0] or "linux/amd64" in lines[0]
    assert "frigate-learn-hailo-dfc:3.34.0" in lines[0]
    assert lines[0].endswith("hailo parser onnx best.onnx --net-name best --hw-arch hailo8 -y")


def test_compile_hailo_docker_requires_docker_binary(tmp_path, monkeypatch):
    onnx = tmp_path / "best.onnx"
    onnx.write_bytes(b"# fake onnx")
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(
        "frigate_learn.deploy.hailo.shutil.which", lambda name: None
    )
    with pytest.raises(HailoCompilerMissing, match="hailo.docker_image"):
        compile_hailo(onnx, tmp_path / "out", dry_run=False, docker_image="img:tag")


def test_compile_hailo_reports_stage_failure(tmp_path, monkeypatch):
    onnx = tmp_path / "best.onnx"
    onnx.write_bytes(b"# fake onnx")
    bin_dir = tmp_path / "bin"
    _write_script(
        bin_dir / "hailo",
        "#!/usr/bin/env python3\nimport sys\nsys.stderr.write('boom')\nsys.exit(3)\n",
    )
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")

    with pytest.raises(HailoCompileError, match="boom"):
        compile_hailo(onnx, tmp_path / "out", dry_run=False)