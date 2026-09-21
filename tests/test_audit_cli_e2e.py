"""CLI-level end-to-end tests for the audit pipeline (no deployment).

Each test drives the real ``frigate-learn audit`` click group against a
real on-disk dataset. SAM is a deterministic stand-in (measured geometry,
no model download) and the VLM is a real local HTTP server implementing
the oMLX OpenAI-compatible chat-completions endpoint, so the full stack
runs: config loading, adapters, pipeline orchestration, caching,
reconciliation rendering, HTTP transport, strict parsing, decisions,
exports and stats.
"""

from __future__ import annotations

import base64
import io
import json
import sqlite3
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner
from PIL import Image, ImageDraw

from frigate_learn.audit.cli import audit_group
from frigate_learn.audit.types import BoundingBox, Mask, SamResult


class StubSamTeacher:
    """Deterministic stand-in for MlxSam3Teacher with the same call surface."""

    model_key = "stub-sam:rect:1"

    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)

    def predict(self, image, frigate_class, frigate_bbox):
        width, height = image.size
        x1 = int(frigate_bbox.x1) + 2
        y1 = int(frigate_bbox.y1) + 2
        x2 = min(width, int(frigate_bbox.x2) - 2)
        y2 = min(height, int(frigate_bbox.y2) - 2)
        if x2 <= x1 or y2 <= y1:
            x1, y1 = int(frigate_bbox.x1), int(frigate_bbox.y1)
            x2, y2 = min(width, int(frigate_bbox.x2)), min(height, int(frigate_bbox.y2))
        mask = np.zeros((height, width), dtype=bool)
        mask[y1:y2, x1:x2] = True
        box = BoundingBox(float(x1), float(y1), float(x2), float(y2))
        return SamResult(
            class_name=frigate_class,
            bbox=box,
            mask=Mask(mask),
            confidence=0.95,
        )


@dataclass
class VlmScenario:
    """State a fake VLM server consults for its responses.

    The fake acts blind like the real inspector: it answers per-call from
    ``class_seq`` (never from the image text, which contains no class hint).
    Calls whose sequence number is in ``absent_on`` report no object.
    """

    class_seq: list = field(default_factory=lambda: ["person"])
    absent_on: set = field(default_factory=lambda: set())
    transport_fail_on: set = field(default_factory=lambda: set())
    calls: list = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def verdict_for(self, seq: int) -> dict:
        if seq in self.absent_on:
            return {"bbox_covers_object": True, "class_label": ""}
        label = self.class_seq[seq % len(self.class_seq)]
        return {"bbox_covers_object": True, "class_label": label}


def _make_handler(scenario: VlmScenario):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            with scenario.lock:
                seq = len(scenario.calls)
                scenario.calls.append(body)
            if seq in scenario.transport_fail_on:
                data = b"{\"error\":\"simulated 503\"}"
                self.send_response(503)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            fields = scenario.verdict_for(seq)
            payload = {"choices": [{"message": {"content": json.dumps(fields, sort_keys=True)}}]}
            data = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format, *args):
            pass

    return Handler


@pytest.fixture()
def fake_vlm():
    """Start a real local VLM (OpenAI-compatible) server on an ephemeral port."""
    scenario = VlmScenario()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(scenario))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}/v1"
    try:
        yield scenario, base_url
    finally:
        server.shutdown()
        thread.join(timeout=5)


@dataclass
class SamScenario:
    """Health state for the fake remote SAM server (togglable mid-test)."""

    healthy: bool = True


def _make_sam_handler(scenario: SamScenario):
    """HTTP fake of the `frigate-learn sam-server` endpoint.

    Mirrors the local stub geometry: the SAM box is the Frigate box inset by
    two pixels and the mask is that box filled. Supports a 503 "down" state
    (healthz and predict) for the availability / resume-refresh tests.
    """

    class Handler(BaseHTTPRequestHandler):
        def _json(self, code, payload):
            data = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.rstrip("/") == "/healthz":
                code = 200 if scenario.healthy else 503
                self._json(code, {"status": "ok" if scenario.healthy else "down"})
            else:
                self._json(404, {"detail": {"kind": "not_found"}})

        def do_POST(self):
            if self.path != "/predict":
                self._json(404, {"detail": {"kind": "not_found"}})
                return
            if not scenario.healthy:
                self._json(503, {"detail": "server down"})
                return
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            data_url = body["image"]
            b64 = data_url.split(",", 1)[1]
            img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
            width, height = img.size
            bbox = [float(v) for v in body["bbox"]]
            x1 = int(bbox[0]) + 2
            y1 = int(bbox[1]) + 2
            x2 = min(width, int(bbox[2]) - 2)
            y2 = min(height, int(bbox[3]) - 2)
            if x2 <= x1 or y2 <= y1:
                x1, y1 = int(bbox[0]), int(bbox[1])
                x2, y2 = min(width, int(bbox[2])), min(height, int(bbox[3]))
            mask = np.zeros((height, width), dtype=bool)
            mask[y1:y2, x1:x2] = True
            buf = io.BytesIO()
            Image.fromarray(mask.astype(np.uint8) * 255, "L").save(buf, format="PNG")
            payload = {
                "class_name": body["frigate_class"],
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": 0.95,
                "mask": {
                    "encoding": "png",
                    "base64": base64.b64encode(buf.getvalue()).decode("ascii"),
                },
                "model_key": "stub-sam:http:1",
                "raw_metadata": {"server": "fake"},
            }
            self._json(200, payload)

        def log_message(self, format, *args):
            pass

    return Handler


@pytest.fixture()
def fake_sam():
    """Start a real local SAM (frigate-learn sam-server) HTTP fake."""
    scenario = SamScenario()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _make_sam_handler(scenario))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield scenario, base_url
    finally:
        server.shutdown()
        thread.join(timeout=5)

def _make_image(rows: int, cols: int, color: tuple = (180, 190, 200)) -> Image.Image:
    img = Image.new("RGB", (cols, rows), (24, 28, 34))
    draw = ImageDraw.Draw(img)
    draw.rectangle(
        [int(cols * 0.3), int(rows * 0.3), int(cols * 0.55), int(rows * 0.75)],
        fill=color,
    )
    return img


def _make_manifest_dataset(root: Path, objects: list[dict]) -> Path:
    """objects: id, label, rows, cols, box ([x1,y1,x2,y2] normalized)."""
    images = root / "images"
    images.mkdir(parents=True, exist_ok=True)
    manifest_objects = []
    for index, obj in enumerate(objects):
        rel = f"obj{index}.png"
        _make_image(obj["rows"], obj["cols"]).save(images / rel)
        manifest_objects.append({
            "id": obj["id"],
            "class_name": obj["label"],
            "image": rel,
            "bbox": obj["box"],
            "bbox_format": "xyxy",
            "normalized": True,
            "coordinate_space": "image",
        })
    annotations = root / "annotations"
    annotations.mkdir(parents=True, exist_ok=True)
    (annotations / "manifest.json").write_text(json.dumps({
        "image_root": "images",
        "bbox_format": "xyxy",
        "normalized": True,
        "coordinate_space": "image",
        "class_map": {"person": 0, "car": 1, "bird": 2},
        "objects": manifest_objects,
    }), encoding="utf-8")
    return root


def _make_db_dataset(root: Path, objects: list[dict]) -> Path:
    """objects: id, label, rows, cols, box; images stored under relative paths."""
    root.mkdir(parents=True, exist_ok=True)
    images = root / "images"
    conn = sqlite3.connect(root / "frigate_learn.db")
    conn.execute(
        "CREATE TABLE samples (id TEXT, image_path TEXT, frigate_label TEXT, "
        "frigate_x1 REAL, frigate_y1 REAL, frigate_x2 REAL, frigate_y2 REAL, "
        "camera TEXT, timestamp REAL, event_id TEXT, frigate_score REAL)"
    )
    for obj in objects:
        day, cam, sid, label = obj["day"], obj["cam"], obj["id"], obj["label"]
        img_dir = images / day / cam
        img_dir.mkdir(parents=True, exist_ok=True)
        _make_image(obj["rows"], obj["cols"]).save(img_dir / f"{sid}.jpg", quality=90)
        rel = f"{day}/{cam}/{sid}.jpg"
        x1, y1, x2, y2 = obj["box"]
        conn.execute(
            "INSERT INTO samples VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (sid, rel, label, x1, y1, x2, y2, cam, 1788750000.0, f"{sid}-evt", 0.9),
        )
    conn.commit()
    conn.close()
    return root


def _write_config(
    path: Path,
    *,
    input_dir: str,
    output: str,
    training: str,
    vlm_base: str,
    extra: str = "",
    sam_extra: str = "",
) -> None:
    lines = [
        "classes: [person, car, bird]",
        "audit:",
    ]
    lines.append(f'  input: "{input_dir}"')
    lines.append(f'  output: "{output}"')
    lines.append(f'  training: "{training}"')
    sam_lines = [
        "      load_from_hf: false",
        "      confidence_threshold: 0.4",
    ]
    if sam_extra:
        sam_lines.append(sam_extra)
    lines += [
        "  pipeline_version: 7",
        "  max_consecutive_vlm_errors: 3",
        "  hard_negatives_enabled: true",
        "  negative_sampling_enabled: true",
        "  samples_per_crop: 1",
        "  models:",
        "    sam:",
        *sam_lines,
        "    vlm:",
    ]
    lines.append(f'      base_url: "{vlm_base}"')
    lines += [
        "      model: fake-vlm",
        "      max_retries: 2",
        "      json_mode: true",
        "  geometry:",
        "    min_bbox_iou: 0.3",
        "    mask_bbox_ratio_min: 0.2",
        "    min_mask_frigate_containment: 0.5",
    ]
    if extra:
        lines.append(extra)
    path.write_text("\n".join(lines), encoding="utf-8")


def _invoke(args: list[str]):
    return CliRunner().invoke(audit_group, args)


def _run_stats_json(output: str) -> dict:
    return json.loads(output)


def test_manifest_end_to_end(tmp_path, fake_vlm, monkeypatch):
    scenario, base_url = fake_vlm
    scenario.class_seq = ["person", "car"]
    scenario.absent_on = {2}
    monkeypatch.setattr("frigate_learn.audit.sam.MlxSam3Teacher", StubSamTeacher)
    dataset = _make_manifest_dataset(tmp_path / "dataset", [
        {"id": "p1", "label": "person", "rows": 96, "cols": 128, "box": [0.30, 0.30, 0.55, 0.75]},
        {"id": "c1", "label": "car", "rows": 64, "cols": 96, "box": [0.20, 0.40, 0.70, 0.60]},
        {"id": "b1", "label": "bird", "rows": 96, "cols": 128, "box": [0.10, 0.10, 0.40, 0.40]},
    ])
    cfg = tmp_path / "config.yaml"
    _write_config(
        cfg,
        input_dir=str(dataset),
        output="out/audit",
        training="out/training",
        vlm_base=base_url,
    )

    result = _invoke(["run", "--config", str(cfg), "--stats-json"])
    assert result.exit_code == 0, result.output
    stats = _run_stats_json(result.output)
    assert stats["total"] == 3
    assert stats["processed"] == 3
    assert stats["sam_ok"] == 3
    assert stats["sam_fail"] == 0
    assert stats["vlm_calls"] == 3
    assert stats["accepted"] == 2
    assert stats["dropped"] == 1
    assert stats["pending"] == 0
    assert stats["agreements"] == 2
    assert stats["disagreements"] == 1
    assert stats["acceptance_rate"] == round(2 / 3, 4)
    assert stats["hard_negatives_written"] >= 1

    audit_out = tmp_path / "out" / "audit"
    training_out = tmp_path / "out" / "training"
    decisions = sorted(audit_out.glob("*/decision.json"))
    assert len(decisions) == 3
    keep = [d for d in decisions if json.loads(d.read_text())["decision"]["status"] == "KEEP"]
    drop = [d for d in decisions if json.loads(d.read_text())["decision"]["status"] == "DROP"]
    assert len(keep) == 2 and len(drop) == 1

    assert (training_out / "classes.txt").read_text().splitlines() == ["person", "car", "bird"]
    positive = training_out / "positive"
    images = sorted(p.name for p in (positive / "images").glob("*.jpg"))
    labels = sorted(p.name for p in (positive / "labels").glob("*.txt"))
    masks = sorted(p.name for p in (positive / "masks").glob("*.png"))
    assert images == ["c1.jpg", "p1.jpg"]
    assert labels == ["c1.txt", "p1.txt"]
    assert masks == ["c1.png", "p1.png"]
    provenance = [json.loads(l) for l in (positive / "provenance.jsonl").read_text().splitlines()]
    assert {p["sample_id"] for p in provenance} == {"c1", "p1"}
    assert all(p["decision"]["status"] == "KEEP" for p in provenance)
    assert all(p["source"]["class_id"] == 1 for p in provenance if p["sample_id"] == "c1")
    label_line = (positive / "labels" / "p1.txt").read_text().strip()
    class_id, cx, cy, w, h = (float(v) for v in label_line.split())
    assert int(class_id) == 0
    assert 0.0 < cx < 1.0 and 0.0 < cy < 1.0 and 0.0 < w < 1.0 and 0.0 < h < 1.0
    mask_img = np.asarray(Image.open(positive / "masks" / "p1.png").convert("L"))
    assert mask_img.max() == 255 and mask_img.min() == 0

    hard = training_out / "hard_negative"
    hard_images = {p.name for p in (hard / "images").glob("*.jpg") if p.name.startswith("b1")}
    assert "b1.jpg" in hard_images
    hard_prov = [json.loads(l) for l in (hard / "provenance.jsonl").read_text().splitlines()]
    kinds = {p["kind"] for p in hard_prov}
    assert kinds <= {"hard_negative", "random_background"}
    b1_prov = next(p for p in hard_prov if p["sample_id"] == "b1")
    assert b1_prov["kind"] == "hard_negative"
    assert b1_prov["decision"]["status"] == "DROP"
    assert "object_present" in b1_prov["decision"]["failing_conditions"]
    assert all("background_box" in p["extra"] for p in hard_prov if p["kind"] == "random_background")
    assert scenario.calls


def test_db_end_to_end_and_class_filter(tmp_path, fake_vlm, monkeypatch):
    scenario, base_url = fake_vlm
    scenario.class_seq = ["person", "car"]
    monkeypatch.setattr("frigate_learn.audit.sam.MlxSam3Teacher", StubSamTeacher)
    dataset = _make_db_dataset(tmp_path / "dataset", [
        {"id": "p1", "label": "person", "rows": 80, "cols": 120, "box": [0.25, 0.30, 0.60, 0.80], "day": "20260907", "cam": "front"},
        {"id": "c1", "label": "car", "rows": 64, "cols": 96, "box": [0.20, 0.40, 0.75, 0.62], "day": "20260906", "cam": "front"},
        {"id": "p2", "label": "person", "rows": 80, "cols": 120, "box": [0.30, 0.20, 0.55, 0.70], "day": "20260906", "cam": "back"},
        {"id": "c2", "label": "car", "rows": 64, "cols": 96, "box": [0.10, 0.45, 0.65, 0.65], "day": "20260907", "cam": "front"},
    ])
    cfg = tmp_path / "config.yaml"
    _write_config(
        cfg,
        input_dir=str(dataset),
        output=str(tmp_path / "out" / "audit"),
        training=str(tmp_path / "out" / "training"),
        vlm_base=base_url,
    )
    result = _invoke(["run", "--config", str(cfg), "--stats-json"])
    assert result.exit_code == 0, result.output
    stats = _run_stats_json(result.output)
    assert stats["total"] == 4 and stats["processed"] == 4 and stats["accepted"] == 4
    assert stats["by_class"]["person"]["accepted"] == 2
    assert stats["by_class"]["car"]["accepted"] == 2
    assert (tmp_path / "out" / "training" / "classes.txt").read_text().splitlines() == ["person", "car"]

    filtered = _invoke([
        "run", "--config", str(cfg),
        "--output", str(tmp_path / "out2" / "audit"),
        "--training-output", str(tmp_path / "out2" / "training"),
        "--class", "person", "--limit", "2", "--stats-json",
    ])
    assert filtered.exit_code == 0, filtered.output
    fstats = _run_stats_json(filtered.output)
    assert fstats["total"] == 2 and fstats["processed"] == 2
    assert fstats["by_class"]["person"]["processed"] == 2
    assert "car" not in fstats["by_class"]


def test_resume_skips_cached_samples(tmp_path, fake_vlm, monkeypatch):
    scenario, base_url = fake_vlm
    scenario.class_seq = ["person", "car"]
    monkeypatch.setattr("frigate_learn.audit.sam.MlxSam3Teacher", StubSamTeacher)
    dataset = _make_manifest_dataset(tmp_path / "dataset", [
        {"id": "p1", "label": "person", "rows": 96, "cols": 128, "box": [0.30, 0.30, 0.55, 0.75]},
        {"id": "c1", "label": "car", "rows": 64, "cols": 96, "box": [0.20, 0.40, 0.70, 0.60]},
    ])
    cfg = tmp_path / "config.yaml"
    _write_config(
        cfg,
        input_dir=str(dataset),
        output="out/audit",
        training="out/training",
        vlm_base=base_url,
    )
    first = _invoke(["run", "--config", str(cfg), "--stats-json"])
    assert first.exit_code == 0, first.output
    calls_after_first = len(scenario.calls)
    assert calls_after_first > 0
    second = _invoke(["run", "--config", str(cfg), "--resume", "--stats-json"])
    assert second.exit_code == 0, second.output
    stats = _run_stats_json(second.output)
    assert stats["total"] == 2
    assert stats["cached_skipped"] == 2
    assert stats["processed"] == 0
    assert stats["accepted"] == 2
    assert len(scenario.calls) == calls_after_first


def test_pending_then_resume_with_server_restored(tmp_path, fake_vlm, monkeypatch):
    scenario, base_url = fake_vlm
    monkeypatch.setattr("frigate_learn.audit.sam.MlxSam3Teacher", StubSamTeacher)
    dataset = _make_db_dataset(tmp_path / "dataset", [
        {"id": "p1", "label": "person", "rows": 80, "cols": 120, "box": [0.25, 0.30, 0.60, 0.80], "day": "20260907", "cam": "front"},
        {"id": "p2", "label": "person", "rows": 80, "cols": 120, "box": [0.30, 0.20, 0.55, 0.70], "day": "20260906", "cam": "back"},
    ])
    scenario.transport_fail_on = {0, 1, 2}
    cfg = tmp_path / "config.yaml"
    _write_config(
        cfg,
        input_dir=str(dataset),
        output="out/audit",
        training="out/training",
        vlm_base=base_url,
        extra="  max_consecutive_vlm_errors: 10",
    )
    result = _invoke(["run", "--config", str(cfg), "--stats-json"])
    assert result.exit_code == 0, result.output
    stats = _run_stats_json(result.output)
    assert stats["total"] == 2
    assert stats["accepted"] == 1
    assert stats["pending"] == 1
    assert stats["vlm_transport"] == 1
    audit_out = tmp_path / "out" / "audit"
    statuses = {json.loads(p.read_text())["decision"]["status"] for p in audit_out.glob("*/decision.json")}
    assert statuses == {"KEEP", "PENDING"}

    scenario.transport_fail_on = set()
    resume = _invoke(["run", "--config", str(cfg), "--resume", "--stats-json"])
    assert resume.exit_code == 0, resume.output
    rstats = _run_stats_json(resume.output)
    assert rstats["total"] == 2
    assert rstats["cached_skipped"] == 1
    assert rstats["processed"] == 1
    assert rstats["accepted"] == 2
    assert rstats["pending"] == 0
    statuses = {json.loads(p.read_text())["decision"]["status"] for p in audit_out.glob("*/decision.json")}
    assert statuses == {"KEEP"}


def test_stats_command_aggregates(tmp_path, fake_vlm, monkeypatch):
    _, base_url = fake_vlm
    monkeypatch.setattr("frigate_learn.audit.sam.MlxSam3Teacher", StubSamTeacher)
    dataset = _make_manifest_dataset(tmp_path / "dataset", [
        {"id": "p1", "label": "person", "rows": 96, "cols": 128, "box": [0.30, 0.30, 0.55, 0.75]},
        {"id": "b1", "label": "bird", "rows": 96, "cols": 128, "box": [0.10, 0.10, 0.40, 0.40]},
    ])
    cfg = tmp_path / "config.yaml"
    _write_config(
        cfg,
        input_dir=str(dataset),
        output="out/audit",
        training="out/training",
        vlm_base=base_url,
    )
    run = _invoke(["run", "--config", str(cfg), "--stats-json"])
    assert run.exit_code == 0, run.output
    run_stats = _run_stats_json(run.output)
    stats = _invoke(["stats", str(tmp_path / "out" / "audit"), "--json"])
    assert stats.exit_code == 0, stats.output
    agg = _run_stats_json(stats.output)
    for key in ("total", "accepted", "dropped", "pending", "processed", "sam_ok", "vlm_calls"):
        assert agg[key] == run_stats[key], key


def test_cli_degrades_when_models_missing(tmp_path):
    dataset = _make_manifest_dataset(tmp_path / "dataset", [
        {"id": "p1", "label": "person", "rows": 96, "cols": 128, "box": [0.30, 0.30, 0.55, 0.75]},
        {"id": "c1", "label": "car", "rows": 64, "cols": 96, "box": [0.20, 0.40, 0.70, 0.60]},
    ])
    cfg = tmp_path / "config.yaml"
    _write_config(
        cfg,
        input_dir=str(dataset),
        output="out/audit",
        training="out/training",
        vlm_base="http://127.0.0.1:59999/v1",
        extra="  max_consecutive_vlm_errors: 10",
    )
    result = _invoke(["run", "--config", str(cfg), "--stats-json"])
    assert result.exit_code == 0, result.output
    stats = _run_stats_json(result.output)
    assert stats["total"] == 2 and stats["processed"] == 2
    assert stats["sam_fail"] == 2 and stats["accepted"] == 0
    assert stats["vlm_calls"] == 0
    decision = json.loads((tmp_path / "out" / "audit" / "p1" / "decision.json").read_text())
    assert decision["decision"]["status"] == "DROP"
    assert decision["decision"]["reason"] == "sam_failure"
    assert decision["provenance"]["sam"]["class_name"] is None


def test_overrides_are_honored(tmp_path, fake_vlm, monkeypatch):
    scenario, base_url = fake_vlm
    monkeypatch.setattr("frigate_learn.audit.sam.MlxSam3Teacher", StubSamTeacher)
    dataset = _make_manifest_dataset(tmp_path / "dataset", [
        {"id": "a1", "label": "person", "rows": 96, "cols": 128, "box": [0.30, 0.30, 0.55, 0.75]},
        {"id": "a2", "label": "car", "rows": 64, "cols": 96, "box": [0.20, 0.40, 0.70, 0.60]},
    ])
    output = tmp_path / "out" / "audit"
    training = tmp_path / "out" / "training"
    cfg = tmp_path / "config.yaml"
    _write_config(
        cfg,
        input_dir=str(dataset),
        output=str(output),
        training=str(training),
        vlm_base=base_url,
    )
    from frigate_learn.audit.overrides import save_override

    save_override(output, "a1", "DROP", "human rejects")
    save_override(output, "a2", "KEEP", "human keeps the car")

    result = _invoke(["run", "--config", str(cfg), "--stats-json"])
    assert result.exit_code == 0, result.output
    stats = _run_stats_json(result.output)
    assert stats["total"] == 2
    assert stats["accepted"] == 1
    assert stats["dropped"] == 1

    a1 = json.loads((output / "a1" / "decision.json").read_text(encoding="utf-8"))
    assert a1["decision"]["status"] == "DROP"
    assert a1["decision"]["reason"] == "human_override"
    assert a1["provenance"]["override"]["status"] == "DROP"
    assert a1["provenance"]["override"]["original_status"] == "KEEP"

    a2 = json.loads((output / "a2" / "decision.json").read_text(encoding="utf-8"))
    assert a2["decision"]["status"] == "KEEP"
    assert a2["provenance"]["override"]["status"] == "KEEP"
    assert a2["provenance"]["override"]["original_status"] == "DROP"
    assert (training / "positive" / "images" / "a2.jpg").is_file()
    assert not (training / "positive" / "images" / "a1.jpg").exists()


def test_remote_sam_end_to_end(tmp_path, fake_vlm, fake_sam):
    """The small-VM path: SAM comes from a remote HTTP endpoint, never a
    local sam3_mlx import (no monkeypatch needed)."""
    scenario, vlm_base = fake_vlm
    scenario.class_seq = ["person", "car"]
    sam_scenario, sam_base = fake_sam
    dataset = _make_manifest_dataset(tmp_path / "dataset", [
        {"id": "p1", "label": "person", "rows": 96, "cols": 128, "box": [0.30, 0.30, 0.55, 0.75]},
        {"id": "c1", "label": "car", "rows": 64, "cols": 96, "box": [0.20, 0.40, 0.70, 0.60]},
    ])
    cfg = tmp_path / "config.yaml"
    _write_config(
        cfg,
        input_dir=str(dataset),
        output="out/audit",
        training="out/training",
        vlm_base=vlm_base,
        sam_extra=f'      backend: "http"\n      base_url: "{sam_base}"\n      model: "stub-remote"',
    )

    result = _invoke(["run", "--config", str(cfg), "--stats-json"])
    assert result.exit_code == 0, result.output
    stats = _run_stats_json(result.output)
    assert stats["total"] == 2 and stats["processed"] == 2
    assert stats["sam_ok"] == 2 and stats["sam_fail"] == 0
    assert stats["vlm_calls"] == 2
    assert stats["accepted"] == 2
    assert sam_scenario.healthy
    audit_out = tmp_path / "out" / "audit"
    for sid in ("p1", "c1"):
        assert (audit_out / sid / "sam.json").is_file()
        assert (audit_out / sid / "mask.png").is_file()
        decision = json.loads((audit_out / sid / "decision.json").read_text())
        assert decision["decision"]["status"] == "KEEP"
    assert (tmp_path / "out" / "training" / "positive" / "masks" / "p1.png").is_file()


def test_remote_sam_resume_skips_cached_with_server_down(tmp_path, fake_vlm, fake_sam):
    """--resume must stay offline-capable: the SAM cache key is derived from
    config, not from probing the server."""
    scenario, vlm_base = fake_vlm
    scenario.class_seq = ["person"]
    sam_scenario, sam_base = fake_sam
    dataset = _make_manifest_dataset(tmp_path / "dataset", [
        {"id": "p1", "label": "person", "rows": 96, "cols": 128, "box": [0.30, 0.30, 0.55, 0.75]},
    ])
    cfg = tmp_path / "config.yaml"
    _write_config(
        cfg,
        input_dir=str(dataset),
        output="out/audit",
        training="out/training",
        vlm_base=vlm_base,
        sam_extra=f'      backend: "http"\n      base_url: "{sam_base}"\n      model: "stub-remote"',
    )
    first = _invoke(["run", "--config", str(cfg), "--stats-json"])
    assert first.exit_code == 0, first.output
    assert _run_stats_json(first.output)["sam_ok"] == 1

    sam_scenario.healthy = False
    second = _invoke(["run", "--config", str(cfg), "--resume", "--stats-json"])
    assert second.exit_code == 0, second.output
    rstats = _run_stats_json(second.output)
    assert rstats["total"] == 1
    assert rstats["cached_skipped"] == 1
    assert rstats["processed"] == 0
    assert rstats["accepted"] == 1


def test_remote_sam_failure_refreshes_once_server_returns(tmp_path, fake_vlm, fake_sam):
    """Transport failures are environmental (kind=import): a cached
    sam_failure DROP is refreshed by --resume once the endpoint is reachable
    again (is_available() probe on /healthz)."""
    scenario, vlm_base = fake_vlm
    scenario.class_seq = ["person", "car"]
    sam_scenario, sam_base = fake_sam
    dataset = _make_manifest_dataset(tmp_path / "dataset", [
        {"id": "p1", "label": "person", "rows": 96, "cols": 128, "box": [0.30, 0.30, 0.55, 0.75]},
        {"id": "c1", "label": "car", "rows": 64, "cols": 96, "box": [0.20, 0.40, 0.70, 0.60]},
    ])
    cfg = tmp_path / "config.yaml"
    sam_lines = (
        f'      backend: "http"\n'
        f'      base_url: "{sam_base}"\n'
        f'      model: "stub-remote"\n'
        f'      max_retries: 0\n'
        f'      timeout_seconds: 5'
    )
    _write_config(
        cfg,
        input_dir=str(dataset),
        output="out/audit",
        training="out/training",
        vlm_base=vlm_base,
        sam_extra=sam_lines,
    )

    sam_scenario.healthy = False
    first = _invoke(["run", "--config", str(cfg), "--stats-json"])
    assert first.exit_code == 0, first.output
    stats = _run_stats_json(first.output)
    assert stats["total"] == 2 and stats["processed"] == 2
    assert stats["sam_fail"] == 2 and stats["accepted"] == 0
    decision = json.loads((tmp_path / "out" / "audit" / "p1" / "decision.json").read_text())
    assert decision["decision"]["status"] == "DROP"
    assert decision["decision"]["reason"] == "sam_failure"
    assert decision["provenance"]["vlm"]["error_kind"] == "import"

    sam_scenario.healthy = True
    second = _invoke(["run", "--config", str(cfg), "--resume", "--stats-json"])
    assert second.exit_code == 0, second.output
    rstats = _run_stats_json(second.output)
    assert rstats["cached_skipped"] == 0
    assert rstats["processed"] == 2
    assert rstats["sam_ok"] == 2 and rstats["sam_fail"] == 0
    assert rstats["accepted"] == 2


def test_remote_sam_http_backend_requires_base_url(tmp_path, fake_vlm):
    _, vlm_base = fake_vlm
    dataset = _make_manifest_dataset(tmp_path / "dataset", [
        {"id": "p1", "label": "person", "rows": 96, "cols": 128, "box": [0.30, 0.30, 0.55, 0.75]},
    ])
    cfg = tmp_path / "config.yaml"
    _write_config(
        cfg,
        input_dir=str(dataset),
        output="out/audit",
        training="out/training",
        vlm_base=vlm_base,
        sam_extra='      backend: "http"',
    )
    result = _invoke(["run", "--config", str(cfg)])
    assert result.exit_code != 0
    assert "base_url" in result.output


__all__ = ["SamScenario", "StubSamTeacher", "VlmScenario", "_make_handler", "_make_sam_handler", "_write_config", "fake_sam", "fake_vlm"]