"""Acceptance gate (Phase 11).

A candidate passes only when it stays inside the deployment envelope:
- latency <= ``deployment.max_latency_ms``
- recall not worse than the baseline by more than ``min_recall_delta``
- mAP50 not worse than the baseline by more than ``min_map50_delta``
- CPU (when measured) <= ``max_cpu_percent``

With no baseline (first deploy), the metric deltas are skipped and the gate is
latency/CPU-only — a fresh model must at least fit the latency budget.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..config import AppConfig, DeploymentSettings
from ..db import Database
from ..logutil import info
from ..models import Deployment, utcnow
from .benchmark import CandidateResult


@dataclass
class GateResult:
    passed: bool
    candidate: str
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"passed": self.passed, "candidate": self.candidate, "reasons": self.reasons}

    @classmethod
    def from_dict(cls, data: dict) -> "GateResult":
        return cls(
            passed=bool(data.get("passed")),
            candidate=data.get("candidate", ""),
            reasons=list(data.get("reasons", [])),
        )


def evaluate_gate(
    candidate: CandidateResult,
    limits: DeploymentSettings,
    baseline: CandidateResult | None = None,
) -> GateResult:
    reasons: list[str] = []
    ok = True

    # latency envelope
    if candidate.latency_ms is not None:
        if candidate.latency_ms <= limits.max_latency_ms:
            reasons.append(
                f"latency {candidate.latency_ms:.1f}ms <= {limits.max_latency_ms:.1f}ms"
            )
        else:
            reasons.append(
                f"latency {candidate.latency_ms:.1f}ms > limit {limits.max_latency_ms:.1f}ms"
            )
            ok = False
    else:
        reasons.append("latency not measured (skipped)")

    if candidate.metrics.cpu_percent is not None:
        if candidate.metrics.cpu_percent <= limits.max_cpu_percent:
            reasons.append(f"cpu {candidate.metrics.cpu_percent:.0f}% <= {limits.max_cpu_percent:.0f}%")
        else:
            reasons.append(
                f"cpu {candidate.metrics.cpu_percent:.0f}% > limit {limits.max_cpu_percent:.0f}%"
            )
            ok = False

    if baseline is None:
        reasons.append("no baseline: metric deltas skipped")
    else:
        recall_delta = candidate.recall - baseline.recall
        map50_delta = candidate.map50 - baseline.map50
        if recall_delta >= limits.min_recall_delta:
            reasons.append(f"recall {candidate.recall:.3f} vs {baseline.recall:.3f} (delta {recall_delta:+.3f})")
        else:
            reasons.append(
                f"recall regressed {recall_delta:+.3f} below floor {limits.min_recall_delta:+.3f}"
            )
            ok = False
        if map50_delta >= limits.min_map50_delta:
            reasons.append(f"mAP50 {candidate.map50:.3f} vs {baseline.map50:.3f} (delta {map50_delta:+.3f})")
        else:
            reasons.append(
                f"mAP50 regressed {map50_delta:+.3f} below floor {limits.min_map50_delta:+.3f}"
            )
            ok = False

    return GateResult(passed=ok, candidate=candidate.name, reasons=reasons)


def record_deployment(
    config: AppConfig,
    db: Database,
    result: CandidateResult,
    gate: GateResult,
    *,
    artifact_path: str | None = None,
    artifact_hash: str | None = None,
    version: str = "",
    deployed: bool = False,
) -> str:
    """Persist one evaluation + verdict into the ``deployments`` ledger."""
    deployment_id = str(uuid.uuid4())
    with db.session() as session:
        session.add(
            Deployment(
                id=deployment_id,
                model_name=result.name,
                version=version,
                artifact_path=artifact_path,
                artifact_hash=artifact_hash,
                metrics_json=json.dumps(result.metrics.to_dict(), sort_keys=True),
                verdict="PASS" if gate.passed else "FAIL",
                reasons_json=json.dumps({"reasons": gate.reasons}, sort_keys=True),
                deployed=1 if deployed else 0,
                deployed_at=datetime.now(timezone.utc).isoformat() if deployed else None,
                created_at=utcnow(),
            )
        )
        session.commit()
    info("deployment recorded", model=result.name, verdict=gate.to_dict()["passed"])
    return deployment_id


def latest_deployment(db: Database, model_name: str | None = None) -> dict | None:
    """Return the most recent deployment row (optional per model)."""
    from ..models import Deployment as DeploymentModel

    with db.session() as session:
        query = session.query(DeploymentModel)
        if model_name:
            query = query.filter(DeploymentModel.model_name == model_name)
        row = query.order_by(DeploymentModel.created_at.desc()).first()
        if row is None:
            return None
        return {
            "id": row.id,
            "model_name": row.model_name,
            "version": row.version,
            "verdict": row.verdict,
            "deployed": bool(row.deployed),
            "deployed_at": row.deployed_at,
            "metrics": json.loads(row.metrics_json or "{}"),
        }


def save_gate_results(results_path: Path, gate: GateResult) -> None:
    results_path.write_text(
        json.dumps(gate.to_dict(), sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


__all__ = [
    "GateResult",
    "evaluate_gate",
    "record_deployment",
    "latest_deployment",
    "save_gate_results",
]