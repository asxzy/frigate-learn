"""Hailo deployment pipeline (Phase 12).

Turn a trained candidate into a deployable Frigate artifact:

    1. export .pt -> ONNX (lazy ultralytics import; skips when already .onnx)
    2. compile ONNX -> .hef with the Hailo Dataflow Compiler 3.x CLI:

           hailo parser onnx model.onnx --net-name <net> --hw-arch <arch> -y
           hailo optimize <net>.har --hw-arch <arch> --calib-set-path calib.npy
           hailo compiler <net>_optimized.har --hw-arch <arch> --output-dir .

        The conversion is fully host-side: no Hailo device is ever needed to
        produce the HEF (a device is only required to *run* it). The Dataflow
        Compiler only runs on x86-64 Linux, so on macOS the three stages run
        inside a docker image (``hailo.docker_image``, see
        ``containers/hailo-dfc/`` and ``docs/hailo-deploy.md``). With ``-y`` the
        DFC parser auto-detects ultralytics-style detection heads and appends
        ``nms_postprocess`` to the stored model script, which ``optimize``
        applies *before quantization* — the required placement for embedding
        net-flow NMS metadata into the HEF. Injecting it only to
        ``hailo compiler`` silently produces an external postprocess ONNX that
        HailoRT 4.21 (Frigate's pinned runtime) ignores.
    3. write the model manifest + a ready-to-paste Frigate detector config snippet
    4. record the deployment row in SQLite

``dry_run`` writes a placeholder ``.hef`` so the pipeline and its tests run
anywhere.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..config import AppConfig
from ..db import Database
from ..evaluation.gate import GateResult, record_deployment
from ..logutil import info, warning

HAILO_ARCH = "hailo8"  # Hailo-8 (the Frigate box); not "hailo8l" from its docs
DEFAULT_CALIB_SAMPLES = 64


class HailoCompilerMissing(RuntimeError):
    pass


class HailoCompileError(RuntimeError):
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
    hw_arch: str
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


# --- calibration ---------------------------------------------------------


def _letterbox(im, size: int):
    from PIL import Image

    w, h = im.size
    scale = size / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    im = im.resize((nw, nh))
    padded = Image.new("RGB", (size, size), (114, 114, 114))
    padded.paste(im, ((size - nw) // 2, (size - nh) // 2))
    return padded


def build_calibration_set(
    image_dir: Path, imgsz: int, out_npy: Path, samples: int = DEFAULT_CALIB_SAMPLES
) -> Path:
    """Fold camera frames into the float32 NHWC [0, 1] ``.npy`` calibration set
    the DFC ``optimize`` stage consumes (shape (n, h, w, c))."""
    import numpy as np
    from PIL import Image

    exts = {".jpg", ".jpeg", ".png"}
    frames = sorted(
        p for p in Path(image_dir).rglob("*") if p.suffix.lower() in exts
    )
    if not frames:
        raise FileNotFoundError(f"no .jpg/.png calibration images in {image_dir}")
    arrays = []
    for path in frames[:samples]:
        with Image.open(path) as raw:
            arr = np.asarray(_letterbox(raw, imgsz), dtype=np.float32) / 255.0
        arrays.append(arr)
    if not arrays:
        raise FileNotFoundError(f"no decodable calibration images in {image_dir}")
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, np.stack(arrays))
    return out_npy


# --- compile -------------------------------------------------------------


class _DfcRunner:
    """Runs the DFC 3.x ``hailo`` CLI natively (Linux) or inside docker (macOS)."""

    def __init__(self, docker_image: str | None):
        self.docker_image = docker_image
        self.docker = shutil.which("docker") if docker_image else None
        self.hailo = None if self.docker else shutil.which("hailo")

    @property
    def available(self) -> bool:
        return self.docker is not None or self.hailo is not None

    @property
    def uses_docker(self) -> bool:
        return self.docker is not None

    def run(self, argv: list[str], cwd: Path) -> subprocess.CompletedProcess:
        if self.uses_docker:
            command = [
                "docker", "run", "--rm", "--platform", "linux/amd64",
                "-v", f"{cwd}:/work", "-w", "/work", self.docker_image, *argv,
            ]
            return subprocess.run(command, capture_output=True, text=True, check=False)
        return subprocess.run(
            [self.hailo, *argv[1:]], cwd=cwd, capture_output=True, text=True, check=False
        )


def compile_hailo(
    onnx_path: Path,
    out_dir: Path,
    *,
    dry_run: bool = False,
    calib_dir: Path | None = None,
    hw_arch: str = HAILO_ARCH,
    docker_image: str | None = None,
    calib_samples: int = DEFAULT_CALIB_SAMPLES,
    interactive: bool = False,
    imgsz: int | None = None,
) -> Path:
    """Compile ONNX -> .hef with the Hailo Dataflow Compiler 3.x CLI.

    Native ``hailo`` on PATH is used when present (Linux); otherwise, when a
    ``docker_image`` is configured (macOS hosts), the DFC stages run in an
    amd64 container. No Hailo device is involved in any mode.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    hef = out_dir / f"{onnx_path.stem}.hef"
    if dry_run:
        hef.write_text("# dry-run placeholder HEF (run on the compilation host)\n", encoding="utf-8")
        return hef

    runner = _DfcRunner(docker_image)
    if not runner.available:
        if sys.platform == "darwin":
            raise HailoCompilerMissing(
                "Hailo DFC runs only on x86-64 Linux. Set hailo.docker_image and build "
                "the image (scripts/build-hailo-dfc.sh), or run `frigate-learn deploy` "
                "on a Linux box with the 'hailo' CLI installed."
            )
        raise HailoCompilerMissing(
            "Hailo Dataflow Compiler CLI ('hailo') not found on PATH; "
            "install it on this Linux host or set hailo.docker_image to compile "
            "inside Docker"
        )
    if runner.uses_docker and sys.platform != "darwin" and "linux" not in sys.platform:
        warning("compiling inside docker on a non-macOS host", platform=sys.platform)

    staging = out_dir
    onnx_local = staging / onnx_path.name
    if onnx_local.resolve() != onnx_path.resolve():
        shutil.copy2(onnx_path, onnx_local)

    calib_npy: Path | None = None
    if calib_dir is not None:
        if imgsz is None:
            raise ValueError("imgsz is required when a calibration directory is given")
        calib_npy = staging / "calib.npy"
        build_calibration_set(calib_dir, imgsz, calib_npy, samples=calib_samples)
    else:
        warning("no hailo.calib_images configured; quantization will use random calibration data")

    net = onnx_path.stem
    stages: list[list[str]] = [
        [
            "hailo", "parser", "onnx", onnx_local.name,
            "--net-name", net, "--hw-arch", hw_arch,
        ]
        + ([] if interactive else ["-y"]),
        [
            "hailo", "optimize", f"{net}.har", "--hw-arch", hw_arch,
            *(
                ["--calib-set-path", calib_npy.name]
                if calib_npy is not None
                else ["--use-random-calib-set"]
            ),
            "--output-har-path", f"{net}_optimized.har",
        ],
        [
            "hailo", "compiler", f"{net}_optimized.har",
            "--hw-arch", hw_arch, "--output-dir", ".",
        ],
    ]
    for argv in stages:
        result = runner.run(argv, cwd=staging)
        if result.returncode != 0:
            raise HailoCompileError(f"hailo {argv[1]} failed: {result.stderr[-2000:]}")
        info("hailo stage done", stage=argv[1], net=net)

    if not hef.exists():
        raise HailoCompileError(f"compiler produced no {hef.name} in {staging}")
    if calib_npy is None:
        warning("HEF compiled with random calibration data; accuracy is not trustworthy", hef=hef.name)
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
    calib_dir: Path | None = None,
    hw_arch: str | None = None,
    docker_image: str | None = None,
    calib_samples: int | None = None,
    interactive: bool | None = None,
) -> DeployOutcome:
    models_dir = out_dir or config.resolve(config.data.root, "models") / model_name
    models_dir.mkdir(parents=True, exist_ok=True)

    imgsz = imgsz or config.training.image_size
    hw_arch = hw_arch or config.hailo.hw_arch
    docker_image = docker_image if docker_image is not None else config.hailo.docker_image
    calib_samples = calib_samples or config.hailo.calib_samples
    interactive = interactive if interactive is not None else config.hailo.interactive
    if calib_dir is None and config.hailo.calib_images:
        calib_dir = config.resolve(config.hailo.calib_images)

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
    hef_path = compile_hailo(
        onnx_path,
        models_dir,
        dry_run=dry_run,
        calib_dir=calib_dir,
        hw_arch=hw_arch,
        docker_image=docker_image,
        calib_samples=calib_samples,
        interactive=interactive,
        imgsz=imgsz,
    )
    model_classes = config.model_class_names()

    manifest = models_dir / "manifest.json"
    manifest.write_text(
        "{\n"
        f'  "model_name": "{model_name}",\n'
        f'  "version": "{version}",\n'
        f'  "imgsz": {imgsz},\n'
        f'  "hw_arch": "{hw_arch}",\n'
        f'  "onnx": "{onnx_path.name}",\n'
        f'  "onnx_sha256": "{_sha256(onnx_path)}",\n'
        f'  "hef": "{hef_path.name}",\n'
        f'  "hef_sha256": "{_sha256(hef_path)}",\n'
        f'  "label_space": "{config.training.label_space}",\n'
        f'  "num_classes": {len(model_classes)},\n'
        f'  "dry_run": {"true" if dry_run else "false"},\n'
        f'  "hef_pending": {"true" if dry_run else "false"}\n'
        "}\n",
        encoding="utf-8",
    )

    snippet = render_frigate_detector_config(
        model_name, hef_path.name, model_classes, imgsz=imgsz
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
        hw_arch=hw_arch,
        deployment_id=deployment_id,
    )


__all__ = [
    "DeployOutcome",
    "HailoCompileError",
    "HailoCompilerMissing",
    "build_calibration_set",
    "compile_hailo",
    "deploy",
    "export_onnx",
    "render_frigate_detector_config",
]