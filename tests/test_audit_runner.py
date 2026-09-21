"""Audit runner entry-point tests (the path shared by CLI and webapp)."""

from __future__ import annotations

import pytest

from frigate_learn.audit.runner import AuditRunError, build_adapter, run_audit
from frigate_learn.config import build_config


def test_run_audit_missing_input_raises(tmp_path):
    cfg = build_config({}, tmp_path)
    cfg.audit.input = "does-not-exist"
    with pytest.raises(AuditRunError, match="input path does not exist"):
        run_audit(cfg)


def test_build_adapter_rejects_unknown_root(tmp_path):
    with pytest.raises(AuditRunError, match="manifest.json"):
        build_adapter(tmp_path)


def test_run_audit_forwards_options_to_pipeline(tmp_path, monkeypatch):
    cfg = build_config({}, tmp_path)
    (tmp_path / "data").mkdir(parents=True)
    (tmp_path / "data" / "frigate_learn.db").write_bytes(b"sqlite")
    captured = {}

    class FakeAdapter:
        pass

    class FakeSam:
        model_key = "fake"

    class FakeVlm:
        pass

    class FakePipeline:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def run(self, options):
            captured["options"] = options
            from frigate_learn.audit.pipeline import AuditStats

            return AuditStats(total=5, accepted=4, dropped=1)

    monkeypatch.setattr(
        "frigate_learn.audit.runner.build_adapter", lambda p: FakeAdapter()
    )
    monkeypatch.setattr(
        "frigate_learn.audit.runner.build_sam_teacher", lambda a: FakeSam()
    )
    monkeypatch.setattr(
        "frigate_learn.audit.runner.build_reconciler", lambda a: FakeVlm()
    )
    monkeypatch.setattr(
        "frigate_learn.audit.runner.build_gates", lambda a: None
    )
    monkeypatch.setattr(
        "frigate_learn.audit.pipeline.AuditPipeline", FakePipeline
    )

    stats = run_audit(cfg, resume=True, limit=12, class_filter=("person",))
    assert stats["total"] == 5
    assert stats["accepted"] == 4
    assert captured["options"].resume is True
    assert captured["options"].limit == 12
    assert captured["options"].classes == ["person"]
    assert captured["init"]["hard_negatives_enabled"] is False