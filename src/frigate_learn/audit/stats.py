"""Audit statistics collection + reporting.

``scan_audit_dir`` rebuilds the aggregate report from previously written
``decision.json`` files (used by ``audit-dataset --stats`` without running
the models). ``render_stats`` prints the summary in the layout the spec
requires (totals + per-class breakdown).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .pipeline import AuditStats


def scan_audit_dir(audit_root: str | Path) -> AuditStats:
    """Aggregate stats from existing decision.json files under audit_root."""
    root = Path(audit_root)
    stats = AuditStats()
    if not root.is_dir():
        return stats
    for decision_path in sorted(root.glob("*/decision.json")):
        try:
            entry = json.loads(decision_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        stats.total += 1
        provenance = entry.get("provenance", {})
        source = provenance.get("source", {})
        label = str(source.get("class_name", ""))
        if not label:
            label = str(entry.get("decision", {}).get("frigate_label", ""))
        cs = stats.class_stats(label)
        decision = entry.get("decision", {})
        status = decision.get("status")
        reason = decision.get("reason")
        if status == "KEEP":
            stats.agreements += 1
            stats.accepted += 1
            stats.sam_ok += 1
            stats.vlm_calls += 1
            cs["processed"] += 1
            cs["accepted"] += 1
        elif status == "PENDING":
            stats.pending += 1
            stats.sam_ok += 1
            stats.vlm_transport += 1
            cs["processed"] += 1
            cs["pending"] += 1
        else:
            stats.dropped += 1
            stats.disagreements += 1
            if provenance.get("sam", {}).get("class_name"):
                stats.sam_ok += 1
            if reason in ("vlm_failure", "vlm_malformed"):
                stats.vlm_fail += 1
                cs["vlm_fail"] += 1
            elif reason and "vlm" in reason:
                stats.vlm_calls += 1
            cs["processed"] += 1
            cs["dropped"] += 1
            if reason == "sam_failure":
                stats.sam_fail += 1
                cs["sam_fail"] += 1
    return stats


def render_stats(stats: AuditStats) -> str:
    """Human-readable report with totals and per-class breakdown."""
    d = stats.as_dict()
    lines: list[str] = []
    lines.append("Audit statistics")
    lines.append("-" * 32)
    val_total = d["total"]
    val_proc = d["processed"]
    val_cached = d["cached_skipped"]
    val_imgfail = d["image_fail"]
    val_samok = d["sam_ok"]
    val_samfail = d["sam_fail"]
    val_vlmcalls = d["vlm_calls"]
    val_vlmfail = d["vlm_fail"]
    val_vlmtrans = d["vlm_transport"]
    val_agree = d["agreements"]
    val_disagree = d["disagreements"]
    val_accepted = d["accepted"]
    val_dropped = d["dropped"]
    val_pending = d["pending"]
    val_hn = d["hard_negatives_written"]
    rate = d["acceptance_rate"]
    lines.append(f"Total samples:          {val_total}")
    lines.append(f"Processed:              {val_proc}")
    lines.append(f"  cached / skipped:     {val_cached}")
    lines.append(f"  image load failures:  {val_imgfail}")
    lines.append(f"SAM successful:         {val_samok}")
    lines.append(f"SAM failures:           {val_samfail}")
    lines.append(f"VLM calls:              {val_vlmcalls}")
    lines.append(f"VLM failures:           {val_vlmfail}")
    lines.append(f"VLM transport (pending):{val_vlmtrans}")
    lines.append(f"SAM/VLM agreements:     {val_agree}")
    lines.append(f"SAM/VLM disagreements:  {val_disagree}")
    lines.append(f"Accepted samples:       {val_accepted}")
    lines.append(f"Dropped samples:        {val_dropped}")
    lines.append(f"Pending samples:        {val_pending}")
    lines.append(f"Hard negatives written: {val_hn}")
    pct = rate * 100.0
    lines.append(f"Acceptance rate:        {pct}")
    if stats.by_class:
        lines.append("")
        lines.append("By Frigate class:")
        for label in sorted(stats.by_class):
            cs = stats.by_class[label]
            lines.append(f"  {label}:")
            p = cs["processed"]
            a = cs["accepted"]
            dr = cs["dropped"]
            pe = cs["pending"]
            lines.append(f"      processed: {p}")
            lines.append(f"      accepted:  {a}")
            lines.append(f"      dropped:   {dr}")
            if pe:
                lines.append(f"      pending:   {pe}")
    return "\n".join(lines)


def aggregate_to_dict(audit_root: str | Path) -> dict[str, Any]:
    """JSON dict summary (used by the CLI --stats JSON output)."""
    return scan_audit_dir(audit_root).as_dict()


__all__ = ["aggregate_to_dict", "render_stats", "scan_audit_dir"]
