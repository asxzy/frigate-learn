"""Candidate model registry (Phase 6).

The Pareto frontier + acceptance gate compare *candidates* against the current
baseline detector. Candidates are tiny YOLO models that fit Hailo-8 memory while
staying far cheaper than the YOLOv8l baseline. Only names here (or explicit
``.pt``/``.onnx`` files) are accepted by ``benchmark`` / ``train``/ ``deploy``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Candidate:
    name: str
    weights: str          # ultralytics model id (auto-downloaded) or a path
    hint_params: float    # ~parameters, million
    hint_latency_ms: float  # rough Hailo-expected latency, used only for ordering hints


CANDIDATES: dict[str, Candidate] = {
    "yolov8n": Candidate("yolov8n", "yolov8n.pt", 3.2, 3.5),
    "yolov8s": Candidate("yolov8s", "yolov8s.pt", 11.2, 7.0),
    "yolov8m": Candidate("yolov8m", "yolov8m.pt", 25.9, 12.5),
    "yolov8l": Candidate("yolov8l", "yolov8l.pt", 43.7, 22.0),  # current baseline
    "yolo11n": Candidate("yolo11n", "yolo11n.pt", 2.6, 2.5),
    "yolo11s": Candidate("yolo11s", "yolo11s.pt", 9.4, 6.0),
    "yolo11m": Candidate("yolo11m", "yolo11m.pt", 20.1, 10.5),
}


def resolve(name: str) -> Candidate:
    """Resolve a candidate name to a spec; also accepts an explicit file path."""
    if name in CANDIDATES:
        return CANDIDATES[name]
    if str(name).endswith((".pt", ".onnx")):
        candidate = Candidate(name, name, float("nan"), float("nan"))
        return candidate
    raise KeyError(
        f"unknown candidate {name!r}; available: {', '.join(sorted(CANDIDATES))}"
    )


def list_candidates() -> list[Candidate]:
    return sorted(CANDIDATES.values(), key=lambda c: c.hint_params)


def weights_exist(candidate: Candidate) -> bool:
    return Path(candidate.weights).is_file()


__all__ = ["Candidate", "CANDIDATES", "resolve", "list_candidates", "weights_exist"]