"""Hailo deployment pipeline (Phase 12).

Turn a trained candidate into a deployable Frigate artifact:

    1. export .pt -> ONNX (lazy ultralytics import; skips when already .onnx)
    2. compile ONNX -> .hef with the Hailo Dataflow compiler CLI
    3. write the model manifest + a ready-to-paste Frigate detector config snippet
    4. record the deployment row in SQLite

The compiler CLI only exists on the training/compilation machine; ``dry_run``
writes a placeholder ``.hef`` so the pipeline and its tests run anywhere.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ..config import AppConfig
from ..db import Database
from ..evaluation.gate import GateResult, record_deployment
from ..logutil import info

HAILO_ARCH = "hailo8"  # Hailo-8 (the Frigate box); not "hailo8l" from its docs


class HailoCompilerMissing(RuntimeError):
    pass


@dataclass
class DeployOutcome:
    model_name: str
    version: str
    onnx_path: Path | None
    hef_path: Path | None
    manifest_path: Path | None
    config_snippet: str
    dry_run: bool
    deployment_id: str | None = None


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def export_onnx(weights: Path, out_dir: Path, imgsz: int) -> Path:
    """Export a trained weights file to ONNX (ultralytics export)."""
    if weights.suffix == ".onnx":
        return weights
    out_dir.mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO  # lazy

    model = YOLO(str(weights))
    model.export(format="onnx", imgsz=imgsz, opset=12, simplify=True)
    onnx = weights.with_suffix(".onnx")
    dest = out_dir / onnx.name
    if onnx.exists():
        shutil.copy2(onnx, dest)
        return dest
    # fallback: something in the same run dir ends with .onnx
    for candidate in weights.parent.rglob("*.onnx"):
        return candidate
    raise FileNotFoundError(f"ultralytics export produced no .onnx near {weights}")


def compile_hailo(onnx_path: Path, out_dir: Path, *, dry_run: bool = False) -> Path:
    """Compile ONNX -> .hef. No compiler/cli on this box -> dry-run placeholder."""
    out_dir.mkdir(parents=True, exist_ok=True)
    hef = out_dir / f"{onnx_path.stem}.hef"
    if dry_run:
        hef.write_text("# dry-run placeholder HEF (run on the compilation host)\n", encoding="utf-8")
        return hef
    compiler = shutil.which("hailo")
    if compiler is None:
        raise HailoCompilerMissing(
            "Hailo Dataflow compiler CLI ('hailo') not found on PATH; "
            "use --dry-run to produce a placeholder, or run on the compilation host"
        )
    result = subprocess.run(
        [compiler, "export", str(onnx_path), "--hw-arch", HAILO_ARCH, "-o", str(hef)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"hailo compile failed: {result.stderr[-2000:]}")
    return hef


def render_frigate_detector_config(
    model_name: str, hef_name: str, classes: Sequence[str], imgsz: int
) -> str:
    """Frigate ``config.yml`` detector block for a compiled Hailo model."""
    num_classes = len(classes)
    lines = [
        "# --- frigate detectors section (paste under 'detectors:') ---",
        f"# model input: {imgsz}x{imgsz} (letterbox target shared by train/eval/export)",
        f"  {model_name}:",
        "    type: hailo",
        f"    hef_path: /usr/share/frigate/models/{hef_name}",
        f"    num_classes: {num_classes}",
    ]
    return "\n".join(lines) + "\n"


def deploy(
    config: AppConfig,
    db: Database,
    *,
    model_name: str,
    weights: Path,
    version: str = "",
    gate: GateResult | None = None,
    result=None,
    imgsz: int | None = None,
    dry_run: bool = False,
    out_dir: Path | None = None,
) -> DeployOutcome:
    models_dir = out_dir or config.resolve(config.data.root, "models") / model_name
    models_dir.mkdir(parents=True, exist_ok=True)

    imgsz = imgsz or config.training.image_size

    onnx_path: Path | None = None
    hef_path: Path | None = None
    if weights.suffix == ".onnx":
        onnx_path = weights
    else:
        if dry_run:
            onnx_path = models_dir / f"{model_name}.onnx"
            onnx_path.write_text("# dry-run placeholder ONNX\n", encoding="utf-8")
        else:
            onnx_path = export_onnx(Path(weights), models_dir, imgsz=imgsz)
    hef_path = compile_hailo(onnx_path, models_dir, dry_run=dry_run)

    manifest = models_dir / "manifest.json"
    manifest.write_text(
        "{\n"
        f'  "model_name": "{model_name}",\n'
        f'  "version": "{version}",\n'
        f'  "imgsz": {imgsz},\n'
        f'  "onnx": "{onnx_path.name}",\n'
        f'  "hef": "{hef_path.name}",\n'
        f'  "hef_sha256": "{_sha256(hef_path)}",\n'
        f'  "dry_run": {"true" if dry_run else "false"}\n'
        "}\n",
        encoding="utf-8",
    )

    snippet = render_frigate_detector_config(
        model_name, hef_path.name, config.classes, imgsz=imgsz
    )
    (models_dir / "frigate-detector.yml").write_text(snippet, encoding="utf-8")

    deployment_id = None
    if gate is not None:
        if result is None:
            from ..evaluation import benchmark as bench
            from ..evaluation.metrics import DetectionMetrics

            result = bench.CandidateResult(
                name=model_name,
                metrics=DetectionMetrics(
                    precision=0.0, recall=0.0, f1=0.0, map50=0.0, map50_95=0.0,
                    small_object_recall=0.0, false_positive_rate=0.0,
                ),
                version=version,
                kind="golden",
            )
        deployment_id = record_deployment(
            config,
            db,
            result=result,
            gate=gate,
            artifact_path=str(hef_path),
            artifact_hash=_sha256(hef_path),
            version=version,
            deployed=False,
        )

    info("deploy manifest written", model=model_name, hef=hef_path.name)
    return DeployOutcome(
        model_name=model_name,
        version=version,
        onnx_path=onnx_path,
        hef_path=hef_path,
        manifest_path=manifest,
        config_snippet=snippet,
        dry_run=dry_run,
        deployment_id=deployment_id,
    )


__all__ = ["DeployOutcome", "HailoCompilerMissing", "deploy", "export_onnx", "compile_hailo", "render_frigate_detector_config"]