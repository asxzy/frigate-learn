"""Configuration loading.

Config is YAML, with ``${VAR}`` / ``$VAR`` / ``${VAR:-default}`` tokens
interpolated from the environment (optionally seeded from a `.env` file).
Relative paths inside the config are anchored to the config file's directory,
so the same config works regardless of where the CLI is invoked from.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

Env = dict[str, str]

_ENV_TOKEN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(:-([^}]*))?\}|\$([A-Za-z_][A-Za-z0-9_]*)")

DEFAULT_CLASSES = ["person", "car", "bicycle", "motorcycle", "bus", "truck", "dog", "cat"]


@dataclass
class FrigateSettings:
    base_url: str = "http://frigate:8971"
    token: str | None = None
    timeout_seconds: float = 30.0
    max_retries: int = 3
    retry_backoff: float = 1.5
    # Only fetch completed events (Frigate refuses snapshots for in-progress
    # events). Kept as config so it can be pinned to whatever the VM serves.
    completed_events_only: bool = True


@dataclass
class CollectionSettings:
    default_days: int = 7
    cameras: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    severity: list[str] = field(default_factory=lambda: ["detection", "alert"])
    max_reviews: int | None = None
    max_events: int | None = None
    concurrency: int = 8
    download_timeout_seconds: float = 60.0
    keep_annotated_snapshots: bool = False
    # P3 near-duplicate suppression. Exact SHA-256 dupes are always skipped;
    # phash_threshold (0..64 Hamming bits) additionally catches near-identical
    # frames from the same camera.
    dedup_enabled: bool = True
    phash_threshold: int = 10
    # Collect server-side region crops (crop=1&height=<H> snapshots) instead of
    # full frames, matching what Frigate feeds the detector. None -> use
    # training.image_size as the crop height.
    region_crop: bool = True
    region_crop_height: int | None = None


@dataclass
class SamplingSettings:
    enabled: bool = False
    max_samples_per_event: int = 3
    min_seconds_between_samples: float = 2.0


@dataclass
class VLMSettings:
    enabled: bool = False
    provider: str = "openai"     # openai (any OpenAI-compatible chat completions endpoint)
    model: str = "gpt-4o-mini"
    base_url: str | None = None  # e.g. http://<llm-gateway>:4001/v1
    api_key: str | None = None
    batch_size: int = 4
    temperature: float = 0.0
    timeout_seconds: float = 120.0
    # "{classes}" is interpolated with the configured class names.
    system_prompt: str = (
        "You are a precise object-detection annotator. Return ONLY a JSON object "
        "of the form {\"objects\": [{\"label\": <class>, \"confidence\": 0..1, "
        "\"bbox\": [x1, y1, x2, y2]} ...]} with normalized coordinates in [0, 1]. "
        "Classes allowed: {classes}. Use an empty objects list if nothing is present."
    )


@dataclass
class TrainingSettings:
    image_size: int = 320
    epochs: int = 50
    model: str = "yolov8n"
    batch: int = 16
    device: str | None = None
    freeze: int | None = None
    project: str | None = None    # output dir for runs (default: data/training)
    seed: int = 1234


@dataclass
class EvaluationSettings:
    golden_dataset: str = "golden-v001"
    candidates: list[str] = field(default_factory=lambda: ["yolov8n", "yolov8s"])


@dataclass
class DeploymentSettings:
    max_latency_ms: float = 20.0
    min_recall_delta: float = -0.01
    min_map50_delta: float = -0.01
    max_cpu_percent: float = 80.0
    baseline: str = "yolov8l"        # candidate name treated as the deployment baseline


@dataclass
class AutomationSettings:
    enable: list[str] = field(default_factory=lambda: ["collect", "build"])
    till: str = "build"
    schedule: str = "0 3 * * *"   # cron used by the generated systemd timer docs


@dataclass
class DataSettings:
    root: str = "data"
    images: str = "images"
    previews: str = "previews"
    datasets: str = "datasets"
    golden: str = "golden"
    database: str = "frigate_learn.db"
    migrations: str = "migrations"


@dataclass
class AppConfig:
    """Fully-resolved application configuration."""

    classes: list[str] = field(default_factory=lambda: list(DEFAULT_CLASSES))
    frigate: FrigateSettings = field(default_factory=FrigateSettings)
    collection: CollectionSettings = field(default_factory=CollectionSettings)
    sampling: SamplingSettings = field(default_factory=SamplingSettings)
    vlm: VLMSettings = field(default_factory=VLMSettings)
    training: TrainingSettings = field(default_factory=TrainingSettings)
    evaluation: EvaluationSettings = field(default_factory=EvaluationSettings)
    deployment: DeploymentSettings = field(default_factory=DeploymentSettings)
    automation: AutomationSettings = field(default_factory=AutomationSettings)
    data: DataSettings = field(default_factory=DataSettings)
    # absolute directory the config file lives in; anchors relative paths
    base_dir: Path = field(default_factory=Path.cwd)

    def resolve(self, *parts: str) -> Path:
        """Resolve a consumer of `data.*` paths relative to the anchor."""
        for p in parts:
            if Path(p).is_absolute():
                return Path(p)
            break
        return self.base_dir.joinpath(*parts)

    def database_path(self) -> Path:
        return self.resolve(self.data.root, self.data.database)

    def images_dir(self) -> Path:
        return self.resolve(self.data.root, self.data.images)

    def previews_dir(self) -> Path:
        return self.resolve(self.data.root, self.data.previews)

    def datasets_dir(self) -> Path:
        return self.resolve(self.data.root, self.data.datasets)

    def golden_dir(self) -> Path:
        return self.resolve(self.data.root, self.data.golden)

    def migrations_dir(self) -> Path:
        return self.resolve(self.data.migrations)


def _interpolate(value: str, env: Env) -> str:
    def _replace(match: re.Match[str]) -> str:
        name = match.group(1) or match.group(4)
        if name is None:
            return match.group(0)
        if match.group(1):  # ${VAR} or ${VAR:-default}
            resolved = env.get(name)
            if match.group(3) is not None:  # ${VAR:-default} -> default if unset/empty
                return resolved if resolved not in (None, "") else match.group(3)
            return resolved or ""
        # $VAR
        return env.get(name, "")

    return _ENV_TOKEN.sub(_replace, value)


def _interpolate_value(value: Any, env: Env) -> Any:
    if isinstance(value, str):
        return _interpolate(value, env)
    if isinstance(value, list):
        return [_interpolate_value(v, env) for v in value]
    if isinstance(value, dict):
        return {k: _interpolate_value(v, env) for k, v in value.items()}
    return value


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str) and value:
        return [v.strip() for v in value.split(",") if v.strip()]
    return []


def _as_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _as_optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _pop(data: dict[str, Any], key: str, default: Any = None) -> Any:
    return data.get(key, default)


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    section = data.get(name) or {}
    if section is None:
        return {}
    return section if isinstance(section, dict) else {}


def build_config(raw: dict[str, Any], base_dir: Path) -> AppConfig:
    """Map a (already-interpolated) raw dict onto dataclasses."""
    cfg = AppConfig(base_dir=Path(base_dir).resolve())

    cfg.classes = _as_str_list(_pop(raw, "classes")) or list(DEFAULT_CLASSES)

    fr = _section(raw, "frigate")
    cfg.frigate.base_url = str(_pop(fr, "base_url", cfg.frigate.base_url)).rstrip("/")
    cfg.frigate.token = _pop(fr, "token", None)
    cfg.frigate.timeout_seconds = float(_pop(fr, "timeout_seconds", cfg.frigate.timeout_seconds))
    cfg.frigate.max_retries = int(_pop(fr, "max_retries", cfg.frigate.max_retries))
    cfg.frigate.retry_backoff = float(_pop(fr, "retry_backoff", cfg.frigate.retry_backoff))
    cfg.frigate.completed_events_only = bool(
        _pop(fr, "completed_events_only", cfg.frigate.completed_events_only)
    )

    col = _section(raw, "collection")
    cfg.collection.default_days = int(_pop(col, "default_days", cfg.collection.default_days))
    cfg.collection.cameras = _as_str_list(_pop(col, "cameras", []))
    cfg.collection.labels = _as_str_list(_pop(col, "labels", []))
    cfg.collection.severity = _as_str_list(_pop(col, "severity", cfg.collection.severity)) or [
        "detection",
        "alert",
    ]
    cfg.collection.max_reviews = _as_optional_int(_pop(col, "max_reviews", None))
    cfg.collection.max_events = _as_optional_int(_pop(col, "max_events", None))
    cfg.collection.concurrency = int(_pop(col, "concurrency", cfg.collection.concurrency))
    cfg.collection.download_timeout_seconds = float(
        _pop(col, "download_timeout_seconds", cfg.collection.download_timeout_seconds)
    )
    cfg.collection.keep_annotated_snapshots = bool(
        _pop(col, "keep_annotated_snapshots", cfg.collection.keep_annotated_snapshots)
    )
    cfg.collection.dedup_enabled = bool(_pop(col, "dedup_enabled", cfg.collection.dedup_enabled))
    cfg.collection.phash_threshold = int(
        _pop(col, "phash_threshold", cfg.collection.phash_threshold)
    )
    cfg.collection.region_crop = bool(_pop(col, "region_crop", cfg.collection.region_crop))
    cfg.collection.region_crop_height = _as_optional_int(
        _pop(col, "region_crop_height", None)
    )

    samp = _section(raw, "sampling")
    cfg.sampling.enabled = bool(_pop(samp, "enabled", cfg.sampling.enabled))
    cfg.sampling.max_samples_per_event = int(
        _pop(samp, "max_samples_per_event", cfg.sampling.max_samples_per_event)
    )
    cfg.sampling.min_seconds_between_samples = float(
        _pop(samp, "min_seconds_between_samples", cfg.sampling.min_seconds_between_samples)
    )

    vlm = _section(raw, "vlm")
    cfg.vlm.enabled = bool(_pop(vlm, "enabled", cfg.vlm.enabled))
    cfg.vlm.provider = str(_pop(vlm, "provider", cfg.vlm.provider))
    cfg.vlm.model = str(_pop(vlm, "model", cfg.vlm.model))
    cfg.vlm.base_url = _pop(vlm, "base_url", None)
    if cfg.vlm.base_url is not None:
        cfg.vlm.base_url = str(cfg.vlm.base_url).rstrip("/")
    cfg.vlm.api_key = _pop(vlm, "api_key", None)
    cfg.vlm.batch_size = int(_pop(vlm, "batch_size", cfg.vlm.batch_size))
    cfg.vlm.temperature = float(_pop(vlm, "temperature", cfg.vlm.temperature))
    cfg.vlm.timeout_seconds = float(_pop(vlm, "timeout_seconds", cfg.vlm.timeout_seconds))
    cfg.vlm.system_prompt = str(
        _pop(vlm, "system_prompt", cfg.vlm.system_prompt)
    )

    tr = _section(raw, "training")
    cfg.training.image_size = int(_pop(tr, "image_size", cfg.training.image_size))
    cfg.training.epochs = int(_pop(tr, "epochs", cfg.training.epochs))
    cfg.training.model = str(_pop(tr, "model", cfg.training.model))
    cfg.training.batch = int(_pop(tr, "batch", cfg.training.batch))
    cfg.training.device = _pop(tr, "device", None)
    cfg.training.freeze = _as_optional_int(_pop(tr, "freeze", None))
    cfg.training.project = _pop(tr, "project", None)
    cfg.training.seed = int(_pop(tr, "seed", cfg.training.seed))

    ev = _section(raw, "evaluation")
    cfg.evaluation.golden_dataset = str(_pop(ev, "golden_dataset", cfg.evaluation.golden_dataset))
    cfg.evaluation.candidates = _as_str_list(
        _pop(ev, "candidates", cfg.evaluation.candidates)
    ) or list(cfg.evaluation.candidates)

    dep = _section(raw, "deployment")
    cfg.deployment.max_latency_ms = float(
        _pop(dep, "max_latency_ms", cfg.deployment.max_latency_ms)
    )
    cfg.deployment.min_recall_delta = float(
        _pop(dep, "min_recall_delta", cfg.deployment.min_recall_delta)
    )
    cfg.deployment.min_map50_delta = float(
        _pop(dep, "min_map50_delta", cfg.deployment.min_map50_delta)
    )
    cfg.deployment.max_cpu_percent = float(
        _pop(dep, "max_cpu_percent", cfg.deployment.max_cpu_percent)
    )
    cfg.deployment.baseline = str(_pop(dep, "baseline", cfg.deployment.baseline))

    auto = _section(raw, "automation")
    cfg.automation.enable = (
        _as_str_list(_pop(auto, "enable", cfg.automation.enable))
        or list(cfg.automation.enable)
    )
    cfg.automation.till = str(_pop(auto, "till", cfg.automation.till))
    cfg.automation.schedule = str(_pop(auto, "schedule", cfg.automation.schedule))

    data = _section(raw, "data")
    cfg.data.root = str(_pop(data, "root", cfg.data.root))
    cfg.data.images = str(_pop(data, "images", cfg.data.images))
    cfg.data.previews = str(_pop(data, "previews", cfg.data.previews))
    cfg.data.datasets = str(_pop(data, "datasets", cfg.data.datasets))
    cfg.data.golden = str(_pop(data, "golden", cfg.data.golden))
    cfg.data.database = str(_pop(data, "database", cfg.data.database))
    cfg.data.migrations = str(_pop(data, "migrations", cfg.data.migrations))

    return cfg


def load_dotenv(path: Path | None = None):
    """Lightweight .env loader (no external dependency required to be robust)."""
    specs: list[Path] = []
    if path is not None:
        specs.append(Path(path))
    else:
        for candidate in (Path.cwd() / ".env",):
            specs.append(candidate)
    loaded: Env = {}
    for spec in specs:
        try:
            text = spec.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                loaded[key] = value
    return loaded


def _merged_env(dotenv: Env | None) -> Env:
    env = dict(os.environ)
    if dotenv:
        for k, v in dotenv.items():
            env.setdefault(k, v)
    return env


def load_config(
    path: str | Path | None = None,
    env_file: str | Path | None = None,
    overrides: Env | None = None,
) -> AppConfig:
    """Load YAML config, interpolate environment, and build an `AppConfig`.

    Config resolution order:
      1. explicit ``path`` argument
      2. ``FRIGATE_LEARN_CONFIG`` env var
      3. ``config.yaml`` in the current directory
      4. ``config.example.yaml`` in the project root (safe defaults fallback)
    """
    if path is None:
        path = os.environ.get("FRIGATE_LEARN_CONFIG") or "config.yaml"

    signal = Path(path)
    if not Path(path).expanduser().is_file():
        candidate = Path.cwd() / "config.example.yaml"
        if candidate.is_file():
            signal = candidate
        else:
            raise FileNotFoundError(
                f"Config file not found: {path}. Pass --config or set FRIGATE_LEARN_CONFIG."
            )

    config_path = Path(path).expanduser() if Path(path).is_file() else signal
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    env = _merged_env(load_dotenv(env_file) if env_file else load_dotenv())
    if overrides:
        env.update(overrides)

    base_dir = config_path.resolve().parent
    raw = _interpolate_value(raw, env)
    return build_config(raw, base_dir)