"""End-to-end pipeline tests with fakes for SAM and the VLM.

Verifies: KEEP writes exports + decision, DROP writes decision, resume skips
finished samples (and reuses caches), SAM-only mode never calls the VLM,
caching prevents recomputation, hard negatives land in the right directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest as _pytest
from PIL import Image

from frigate_learn.audit.adapter import ManifestDatasetAdapter
from frigate_learn.audit.pipeline import AuditPipeline, PipelineAbort, PipelineOptions
from frigate_learn.audit.types import BoundingBox, Mask, SamResult, VlmResult


class FakeSam:
    """SamTeacher stand-in; configurable per-class outcomes."""

    model_key = "fake-sam:test:1"

    def __init__(self, outcomes=None, fail_kind: str = "run"):
        self.outcomes = dict(outcomes or {})
        self.fail_kind = fail_kind
        self.calls: list[tuple] = []

    def predict(self, image, frigate_class, frigate_bbox):
        self.calls.append((frigate_class, frigate_bbox.to_list()))
        outcome = self.outcomes.get(frigate_class, "ok")
        if outcome == "fail":
            from frigate_learn.audit.sam import SamTeacherError

            raise SamTeacherError("sam exploded", kind=self.fail_kind)
        mask = np.zeros((64, 64), dtype=bool)
        mask[10:50, 10:50] = True
        return SamResult(
            class_name=outcome if isinstance(outcome, str) and outcome not in ("ok", "fail") else frigate_class,
            bbox=BoundingBox(8, 8, 56, 56),
            mask=Mask(mask),
            confidence=0.9,
        )


class FakeVlm:
    """VisualReconciler stand-in; configurable verdict."""

    def __init__(self, verdict: dict | None = None):
        self.verdict = verdict or {
            "bbox_covers_object": True,
            "class_label": "person",
        }
        self.calls = 0
        self.seen_paths: list[str] = []

    @property
    def vlm_key(self) -> str:
        return "fake-vlm:test:1"

    def reconcile(self, image_path, class_options=None):
        self.calls += 1
        self.seen_paths.append(str(image_path))
        return VlmResult(**self.verdict, raw=dict(self.verdict))

    def raise_transport(self):
        self.verdict = None


def _make_dataset(tmp_path: Path, n: int = 3, label="person") -> Path:
    images = tmp_path / "images"
    objects = []
    for i in range(n):
        p = images / f"img{i}.jpg"
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), (40 + i * 20, 90, 160)).save(p)
        objects.append({
            "id": f"s{i}",
            "image": f"img{i}.jpg",
            "class_name": label,
            "bbox": [0.1, 0.1, 0.9, 0.9],
            "normalized": True,
        })
    ann = tmp_path / "annotations"
    ann.mkdir(exist_ok=True)
    (ann / "manifest.json").write_text(
        json.dumps({"image_root": "images", "objects": objects}), encoding="utf-8"
    )
    return tmp_path


def _pipeline(tmp_path: Path, sam=None, vlm=None, **kw):
    adapter = ManifestDatasetAdapter(tmp_path)
    return AuditPipeline(
        adapter=adapter,
        sam_teacher=sam or FakeSam(),
        reconciler=vlm or FakeVlm(),
        audit_root=tmp_path / "audit",
        training_root=tmp_path / "training",
        **kw,
    )


def test_pipeline_keep_writes_exports(tmp_path):
    pipeline = _pipeline(_make_dataset(tmp_path, n=2))
    stats = pipeline.run()
    assert stats.accepted == 2
    assert stats.dropped == 0
    assert stats.sam_ok == 2
    assert stats.vlm_calls == 2
    for i in range(2):
        image = tmp_path / "training" / "positive" / "images" / f"s{i}.jpg"
        label = tmp_path / "training" / "positive" / "labels" / f"s{i}.txt"
        mask = tmp_path / "training" / "positive" / "masks" / f"s{i}.png"
        assert image.is_file()
        assert label.is_file()
        assert mask.is_file()
        # YOLO label: class_id 0, normalized cxcywh within image
        values = label.read_text().strip().split()
        assert values[0] == "0"
        assert 0.0 <= float(values[1]) <= 1.0
    classes = tmp_path / "training" / "classes.txt"
    assert classes.read_text().strip() == "person"
    prov = json.loads((tmp_path / "training" / "positive" / "provenance.jsonl").read_text().splitlines()[0])
    assert prov["source"]["type"] == "frigate"
    assert prov["decision"]["status"] == "KEEP"
    assert prov["sam"]["class_name"] == "person"
    assert prov["geometry"]["mask_area"] > 0


def test_pipeline_drop_writes_decision(tmp_path):
    vlm = FakeVlm(verdict={
        "bbox_covers_object": True,
        "class_label": "",
    })
    pipeline = _pipeline(_make_dataset(tmp_path, n=1), vlm=vlm)
    stats = pipeline.run()
    assert stats.accepted == 0
    assert stats.dropped == 1
    decision = json.loads((tmp_path / "audit" / "s0" / "decision.json").read_text())
    assert decision["decision"]["status"] == "DROP"
    assert "object_present" in decision["decision"]["failing_conditions"]
    assert not (tmp_path / "training" / "positive" / "images" / "s0.jpg").exists()


def test_pipeline_sam_failure_drops(tmp_path):
    sam = FakeSam(outcomes={"person": "fail"})
    pipeline = _pipeline(_make_dataset(tmp_path, n=1), sam=sam)
    stats = pipeline.run()
    assert stats.sam_fail == 1
    assert stats.dropped == 1
    decision = json.loads((tmp_path / "audit" / "s0" / "decision.json").read_text())
    assert decision["decision"]["reason"] == "sam_failure"
    assert decision["provenance"]["vlm"]["error"] == "sam"


def test_pipeline_resume_skips_finished(tmp_path):
    dataset = _make_dataset(tmp_path, n=2)
    sam = FakeSam()
    vlm = FakeVlm()
    pipeline = _pipeline(dataset, sam=sam, vlm=vlm)
    pipeline.run()
    # fresh fakes so cache misses would be visible
    sam2 = FakeSam()
    vlm2 = FakeVlm()
    pipeline2 = _pipeline(dataset, sam=sam2, vlm=vlm2)
    stats2 = pipeline2.run(options=PipelineOptions(resume=True))
    assert len(sam2.calls) == 0
    assert vlm2.calls == 0
    assert stats2.cached_skipped == 2
    assert stats2.accepted == 2


def test_pipeline_resume_refreshes_stale_sam_failure(tmp_path, monkeypatch):
    dataset = _make_dataset(tmp_path, n=1)
    sam = FakeSam(outcomes={"person": "fail"}, fail_kind="import")
    pipeline = _pipeline(dataset, sam=sam)
    stats = pipeline.run()
    assert stats.sam_fail == 1
    decision = json.loads((tmp_path / "audit" / "s0" / "decision.json").read_text())
    assert decision["provenance"]["vlm"]["error_kind"] == "import"
    # backend becomes available: resume must re-attempt, not skip the DROP
    monkeypatch.setattr("frigate_learn.audit.pipeline.is_sam_available", lambda: True)
    sam2 = FakeSam()
    vlm2 = FakeVlm()
    pipeline2 = _pipeline(dataset, sam=sam2, vlm=vlm2)
    stats2 = pipeline2.run(options=PipelineOptions(resume=True))
    assert len(sam2.calls) == 1
    assert stats2.cached_skipped == 0
    assert stats2.accepted == 1


def test_pipeline_resume_keeps_run_sam_failure_final(tmp_path, monkeypatch):
    dataset = _make_dataset(tmp_path, n=1)
    sam = FakeSam(outcomes={"person": "fail"}, fail_kind="run")
    pipeline = _pipeline(dataset, sam=sam)
    pipeline.run()
    monkeypatch.setattr("frigate_learn.audit.pipeline.is_sam_available", lambda: True)
    sam2 = FakeSam(outcomes={"person": "fail"}, fail_kind="run")
    pipeline2 = _pipeline(dataset, sam=sam2)
    stats2 = pipeline2.run(options=PipelineOptions(resume=True))
    assert len(sam2.calls) == 0
    assert stats2.cached_skipped == 1
    assert stats2.dropped == 1

class AvailableSam(FakeSam):
    """FakeSam with a controllable ``is_available`` probe (remote mode)."""

    def __init__(self, available: bool = True, **kw):
        super().__init__(**kw)
        self._available = available

    def is_available(self) -> bool:
        return self._available


def test_pipeline_resume_stale_sam_failure_respects_teacher_availability(tmp_path):
    """Remote-mode staleness: import-kind sam_failure is refreshed only once
    the teacher's own availability probe turns true."""
    dataset = _make_dataset(tmp_path, n=1)
    first = _pipeline(
        dataset, sam=AvailableSam(outcomes={"person": "fail"}, fail_kind="import")
    )
    first.run()

    down = _pipeline(
        dataset,
        sam=AvailableSam(outcomes={"person": "fail"}, fail_kind="import", available=False),
    )
    stats = down.run(options=PipelineOptions(resume=True))
    assert len(down.sam_teacher.calls) == 0
    assert stats.cached_skipped == 1
    assert stats.dropped == 1

    up = _pipeline(dataset, sam=AvailableSam(available=True))
    stats = up.run(options=PipelineOptions(resume=True))
    assert len(up.sam_teacher.calls) == 1
    assert stats.cached_skipped == 0
    assert stats.accepted == 1




def test_pipeline_cache_avoids_recompute_on_plain_rerun(tmp_path):
    dataset = _make_dataset(tmp_path, n=1)
    sam = FakeSam()
    pipeline = _pipeline(dataset, sam=sam)
    pipeline.run()
    sam2 = FakeSam()
    vlm2 = FakeVlm()
    pipeline2 = _pipeline(dataset, sam=sam2, vlm=vlm2)
    # no resume flag; per-artifact caches make the second run cheap
    stats = pipeline2.run()
    assert len(sam2.calls) == 0
    assert vlm2.calls == 0
    assert stats.accepted == 1


def test_pipeline_sam_only_never_calls_vlm(tmp_path):
    dataset = _make_dataset(tmp_path, n=1)
    sam = FakeSam()
    vlm = FakeVlm()
    pipeline = _pipeline(dataset, sam=sam, vlm=vlm)
    stats = pipeline.run(options=PipelineOptions(sam_only=True))
    assert vlm.calls == 0
    assert len(sam.calls) == 1
    assert stats.sam_ok == 1
    recon = tmp_path / "audit" / "s0" / "reconciliation.png"
    assert recon.is_file()
    # no decision file in sam-only mode
    assert not (tmp_path / "audit" / "s0" / "decision.json").exists()


def test_pipeline_class_filter(tmp_path):
    dataset = _make_dataset(tmp_path, n=2, label="person")
    pipeline = _pipeline(dataset, sam=FakeSam(), vlm=FakeVlm())
    stats = pipeline.run(options=PipelineOptions(classes=["car"]))
    assert stats.total == 0
    assert stats.accepted == 0


def test_pipeline_limit(tmp_path):
    pipeline = _pipeline(_make_dataset(tmp_path, n=4), sam=FakeSam(), vlm=FakeVlm())
    stats = pipeline.run(options=PipelineOptions(limit=2))
    assert stats.accepted == 2
    assert stats.total == 2


def test_pipeline_hard_negatives(tmp_path):
    dataset = _make_dataset(tmp_path, n=1)
    vlm = FakeVlm(verdict={
        "bbox_covers_object": True,
        "class_label": "",
    })
    pipeline = _pipeline(dataset, sam=FakeSam(), vlm=vlm, hard_negatives_enabled=True)
    pipeline.run()
    neg_image = tmp_path / "training" / "hard_negative" / "images" / "s0.jpg"
    assert neg_image.is_file()
    label = tmp_path / "training" / "hard_negative" / "labels" / "s0.txt"
    assert label.read_text().strip() == ""
    record = json.loads((tmp_path / "training" / "hard_negative" / "provenance.jsonl").read_text().splitlines()[0])
    assert record["kind"] == "hard_negative"


def test_pipeline_vlm_transport_pending_and_abort(tmp_path):
    dataset = _make_dataset(tmp_path, n=3)
    vlm = FakeVlm()
    vlm.verdict = None
    class _FlakyVlm(FakeVlm):
        def reconcile(self, image_path, class_options=None):
            self.calls += 1
            from frigate_learn.audit.vlm import VlmTransportError

            raise VlmTransportError("server down")


    pipeline = _pipeline(dataset, sam=FakeSam(), vlm=_FlakyVlm())
    stats = pipeline.run(options=PipelineOptions(max_consecutive_vlm_errors=5))
    assert stats.pending == 3
    assert stats.vlm_transport == 3
    # same run with a tighter error budget aborts instead

    with _pytest.raises(PipelineAbort):
        _pipeline(dataset, sam=FakeSam(), vlm=_FlakyVlm()).run(options=PipelineOptions(max_consecutive_vlm_errors=3))


def test_pipeline_resume_reprocesses_old_schema_vlm(tmp_path):
    dataset = _make_dataset(tmp_path, n=1)
    pipeline = _pipeline(dataset, sam=FakeSam(), vlm=FakeVlm())
    pipeline.run()
    sample_dir = tmp_path / "audit" / "s0"
    decision_path = sample_dir / "decision.json"
    entry = json.loads(decision_path.read_text())
    old_vlm = {
        "object_present": True,
        "class_matches_frigate": True,
        "class_matches_sam": True,
        "bbox_is_valid": True,
        "mask_is_valid": True,
        "agreement": True,
    }
    entry["provenance"]["vlm"] = old_vlm
    decision_path.write_text(json.dumps(entry))
    vlm_json = sample_dir / "vlm.json"
    ve = json.loads(vlm_json.read_text())
    ve["result"] = old_vlm
    vlm_json.write_text(json.dumps(ve))
    sam2 = FakeSam()
    vlm2 = FakeVlm()
    pipeline2 = _pipeline(dataset, sam=sam2, vlm=vlm2)
    stats2 = pipeline2.run(options=PipelineOptions(resume=True))
    assert len(sam2.calls) == 0
    assert vlm2.calls == 1
    assert stats2.cached_skipped == 0
    assert stats2.accepted == 1
    fresh = json.loads(decision_path.read_text())
    assert fresh["provenance"]["vlm"]["bbox_covers_object"] is True


def test_stats_scan_audit_dir(tmp_path):
    from frigate_learn.audit.stats import scan_audit_dir

    pipeline = _pipeline(_make_dataset(tmp_path, n=2), sam=FakeSam(), vlm=FakeVlm())
    pipeline.run()
    stats = scan_audit_dir(tmp_path / "audit")
    assert stats.total == 2
    assert stats.accepted == 2
    assert stats.by_class["person"]["accepted"] == 2

