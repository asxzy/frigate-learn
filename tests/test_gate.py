"""Acceptance gate tests (Phase 11)."""

from __future__ import annotations

from pathlib import Path

from frigate_learn.config import DeploymentSettings
from frigate_learn.evaluation.benchmark import CandidateResult
from frigate_learn.evaluation.gate import (
    evaluate_gate,
    latest_deployment,
    record_deployment,
    save_gate_results,
)
from frigate_learn.evaluation.metrics import DetectionMetrics


def _result(name, map50=0.5, recall=0.6, latency=5.0, fpr=0.05) -> CandidateResult:
    return CandidateResult(
        name=name,
        metrics=DetectionMetrics(
            precision=0.5, recall=recall, f1=0.5, map50=map50, map50_95=0.3,
            small_object_recall=0.1, false_positive_rate=fpr, latency_ms=latency,
        ),
        version="golden-v001",
        kind="golden",
    )


def _limits(**kw) -> DeploymentSettings:
    base = DeploymentSettings()
    for k, v in kw.items():
        setattr(base, k, v)
    return base


def test_passes_when_within_envelope():
    baseline = _result("yolov8l", map50=0.5, recall=0.6)
    candidate = _result("yolov8n", map50=0.62, recall=0.65, latency=3.2)
    gate = evaluate_gate(candidate, _limits(), baseline)
    assert gate.passed is True


def test_no_baseline_skips_metric_deltas():
    gate = evaluate_gate(_result("yolov8n", map50=0.2, recall=0.2), _limits(), None)
    assert gate.passed is True  # latency ok, no metric floor yet


def test_fails_on_latency_breach():
    gate = evaluate_gate(
        _result("yolov8m", map50=0.5, latency=200.0),
        _limits(max_latency_ms=20.0),
        None,
    )
    assert gate.passed is False
    assert any("latency" in r for r in gate.reasons)


def test_fails_on_recall_regression_below_floor():
    baseline = _result("yolov8l", map50=0.5, recall=0.7)
    candidate = _result("yolov8n", map50=0.55, recall=0.6)  # -0.10 < -0.01
    gate = evaluate_gate(candidate, _limits(min_recall_delta=-0.01), baseline)
    assert gate.passed is False
    assert any("recall" in r for r in gate.reasons)


def test_fails_on_map50_regression():
    baseline = _result("yolov8l", map50=0.6, recall=0.6)
    candidate = _result("yolov8n", map50=0.5, recall=0.6)  # -0.10 < -0.01
    gate = evaluate_gate(candidate, _limits(), baseline)
    assert gate.passed is False


def test_fails_on_fp_rate_regression_above_floor():
    baseline = _result("yolov8l", fpr=0.05)
    candidate = _result("yolov8n", fpr=0.30)  # +0.25 > +0.05
    gate = evaluate_gate(candidate, _limits(), baseline)
    assert gate.passed is False
    assert any("fp rate regressed" in r for r in gate.reasons)


def test_fp_rate_within_delta_passes():
    baseline = _result("yolov8l", fpr=0.05)
    candidate = _result("yolov8n", fpr=0.10)  # +0.05 <= +0.05
    gate = evaluate_gate(candidate, _limits(), baseline)
    assert gate.passed is True
    assert any("fp rate" in r for r in gate.reasons)


def test_fp_rate_floor_configurable():
    baseline = _result("yolov8l", fpr=0.05)
    candidate = _result("yolov8n", fpr=0.20)  # +0.15 > +0.10
    gate = evaluate_gate(candidate, _limits(max_fp_rate_delta=0.10), baseline)
    assert gate.passed is False


def test_record_and_latest_deployment(config, db):
    gate = evaluate_gate(_result("yolov8n"), _limits(), None)
    record_deployment(
        config, db, _result("yolov8n", map50=0.6), gate,
        artifact_path="/tmp/model.hef", artifact_hash="abc", version="v001",
    )
    latest = latest_deployment(db)
    assert latest is not None
    assert latest["verdict"] == "PASS"
    assert latest["deployed"] is False
    assert latest["metrics"]["map50"] == 0.6


def test_latest_deployment_none_when_empty(db):
    assert latest_deployment(db) is None


def test_save_gate_results(tmp_path):
    from frigate_learn.evaluation.gate import GateResult

    gate = GateResult(passed=True, candidate="yolov8n", reasons=["latency ok"])
    out = tmp_path / "gate.json"
    save_gate_results(out, gate)
    import json

    assert json.loads(out.read_text(encoding="utf-8"))["passed"] is True


def test_results_from_json_accepts_str_path(tmp_path):
    from frigate_learn.evaluation.benchmark import results_from_json, results_to_json

    results = [_result("yolov8n", map50=0.4, recall=0.5, latency=3.0)]
    out = tmp_path / "benchmark-results.json"
    results_to_json(results, out)
    loaded = results_from_json(str(out))  # click passes str; must work
    assert [r.name for r in loaded] == ["yolov8n"]
    assert loaded[0].map50 == 0.4