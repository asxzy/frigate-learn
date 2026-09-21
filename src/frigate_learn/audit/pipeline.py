"""Per-sample audit pipeline + batch runner.

For every Frigate object the pipeline:

1. loads the existing crop (never crops again, never loads full frames)
2. runs SAM 3.1 (cached) to get mask, refined bbox and class hypothesis
3. computes deterministic geometry metrics
4. renders the full-overlay reconciliation image (cached, for humans)
5. renders the blind VLM input (crop + Frigate box only, no labels) and
   runs the VLM inspector on it (cached) — the VLM never sees the class
6. aggregates: SAM vs VLM class agreement + geometry gates against the
   Frigate bbox, then applies the one conservative rule (KEEP / DROP /
   PENDING)
7. writes provenance + optionally exports positives / hard negatives

``--resume`` skips samples with a valid final decision; PENDING samples are
reprocessed. Caches are invalidated when image content, class, bbox, model
or pipeline version change.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from . import cache
from .. import logutil
from .adapter import DatasetAdapter
from .decision import GeometryThresholds, decide_sample
from .export import (
    build_provenance,
    sample_background_negative_box,
    write_classes,
    write_hard_negative,
    write_positive_sample,
)
from .geometry import compute_geometry
from .overrides import load_overrides
from .render import render_reconciliation_image, render_vlm_input_image
from .sam import SamTeacher, SamTeacherError, is_sam_available
from .types import Decision, FrigateObject, SamResult, VlmResult
from .vlm import VisualReconciler, VlmResultError, VlmTransportError


class PipelineAbort(RuntimeError):
    """Abort the whole run (e.g. VLM server is down repeatedly)."""


@dataclass
class PipelineOptions:
    limit: int | None = None
    classes: list[str] = field(default_factory=list)
    sam_only: bool = False
    resume: bool = False
    max_consecutive_vlm_errors: int = 5


@dataclass
class AuditStats:
    total: int = 0
    cached_skipped: int = 0
    image_fail: int = 0
    sam_ok: int = 0
    sam_fail: int = 0
    vlm_calls: int = 0
    vlm_fail: int = 0
    vlm_transport: int = 0
    agreements: int = 0
    disagreements: int = 0
    accepted: int = 0
    dropped: int = 0
    pending: int = 0
    hard_negatives_written: int = 0
    by_class: dict[str, dict[str, int]] = field(default_factory=dict)

    def class_stats(self, label: str) -> dict[str, int]:
        return self.by_class.setdefault(
            label,
            {"processed": 0, "accepted": 0, "dropped": 0, "pending": 0, "sam_fail": 0, "vlm_fail": 0},
        )

    @property
    def processed(self) -> int:
        return self.total - self.cached_skipped - self.image_fail

    def as_dict(self) -> dict[str, Any]:
        accepted = self.accepted
        processed = self.processed
        acceptance_rate = accepted / processed if processed else 0.0
        return {
            "total": self.total,
            "processed": processed,
            "cached_skipped": self.cached_skipped,
            "image_fail": self.image_fail,
            "sam_ok": self.sam_ok,
            "sam_fail": self.sam_fail,
            "vlm_calls": self.vlm_calls,
            "vlm_fail": self.vlm_fail,
            "vlm_transport": self.vlm_transport,
            "agreements": self.agreements,
            "disagreements": self.disagreements,
            "accepted": accepted,
            "dropped": self.dropped,
            "pending": self.pending,
            "hard_negatives_written": self.hard_negatives_written,
            "acceptance_rate": round(acceptance_rate, 4),
            "by_class": {k: dict(v) for k, v in self.by_class.items()},
        }


@dataclass
class AuditPipeline:
    """Configuration object; ``run`` executes the batch."""

    adapter: DatasetAdapter
    sam_teacher: SamTeacher
    reconciler: VisualReconciler
    audit_root: Path
    training_root: Path
    pipeline_version: int = 1
    gates: GeometryThresholds = field(default_factory=GeometryThresholds)
    hard_negatives_enabled: bool = False
    negative_sampling_enabled: bool = False
    samples_per_crop: int = 0

    def run(self, options: PipelineOptions | None = None) -> AuditStats:
        """Process the whole dataset; returns aggregated stats."""
        options = options or PipelineOptions()
        self.audit_root.mkdir(parents=True, exist_ok=True)
        self.training_root.mkdir(parents=True, exist_ok=True)
        write_classes(self.training_root, self.adapter.ontology())
        stats = AuditStats()
        consecutive_vlm_errors = 0
        processed_count = 0
        logutil.info(
            "audit run started",
            pipeline_version=self.pipeline_version,
            resume=options.resume,
            sam_only=options.sam_only,
            hard_negatives=self.hard_negatives_enabled,
            vlm_model=getattr(self.reconciler, "settings", None) and getattr(self.reconciler.settings, "model", ""),
        )
        for obj in self.adapter.iter_objects():
            label = obj.class_name
            if options.classes and label not in options.classes:
                continue
            if options.limit is not None and processed_count >= options.limit:
                break
            stats.total += 1
            cs = stats.class_stats(label)
            outcome = self._process_object(obj, options, stats, cs)
            logutil.info(
                "sample",
                sample=obj.sample_id[:12],
                label=label,
                outcome=outcome,
                done=processed_count,
                total=stats.total,
            )
            if outcome == "cached":
                stats.cached_skipped += 1
                continue
            if outcome == "image_fail":
                stats.image_fail += 1
                continue
            processed_count += 1
            if outcome == "sam_only":
                continue
            if outcome == "pending":
                consecutive_vlm_errors += 1
                if consecutive_vlm_errors >= options.max_consecutive_vlm_errors:
                    raise PipelineAbort(
                        "aborting after " + str(consecutive_vlm_errors) + " consecutive VLM transport failures (last sample " + str(self._last_sample_id) + "; VLM server down?"
                    )
            else:
                consecutive_vlm_errors = 0
        rate = stats.accepted / stats.processed if stats.processed else 0.0
        logutil.info(
            "audit run finished",
            accepted=stats.accepted,
            dropped=stats.dropped,
            pending=stats.pending,
            cached_skipped=stats.cached_skipped,
            sam_fail=stats.sam_fail,
            vlm_fail=stats.vlm_fail,
            acceptance_rate=round(rate, 4),
        )
        return stats

    _last_sample_id: str | None = field(default=None, init=False)

    def _apply_override(
        self,
        decision: Decision,
        override: dict[str, object] | None,
    ) -> tuple[Decision, dict[str, object] | None]:
        if not override or override.get("status") not in ("KEEP", "DROP"):
            return decision, None
        return (
            replace(
                decision,
                status=str(override["status"]),
                reason=None if override["status"] == "KEEP" else "human_override",
            ),
            {
                "status": override["status"],
                "note": override.get("note") or "",
                "updated_at": override.get("updated_at"),
                "original_status": decision.status,
                "original_reason": decision.reason,
            },
        )

    def _process_object(
        self,
        obj: FrigateObject,
        options: PipelineOptions,
        stats: AuditStats,
        class_stats: dict[str, int],
    ) -> str:
        """Process one object; returns outcome tag for the stats loop."""
        self._last_sample_id = obj.sample_id
        sample_dir = self.audit_root / obj.sample_id
        sam_key = self.sam_teacher.model_key
        vlm_key = getattr(self.reconciler, "vlm_key", "") or "vlm"
        overrides = load_overrides(self.audit_root)

        try:
            crop = Image.open(obj.image_path).convert("RGB")
        except OSError:
            stats.image_fail += 1
            return "image_fail"

        image_hash = cache.content_hash(np.asarray(crop))
        expected_hash = cache.sample_hash(
            image_hash,
            obj.class_name,
            obj.bbox,
            sam_key,
            vlm_key,
            self.pipeline_version,
        )

        if options.resume and not options.sam_only:
            final = cache.load_decision(
                sample_dir,
                expected_hash,
                sam_key,
                vlm_key,
                self.pipeline_version,
            )
            if final is not None and not self._cached_sam_failure_stale(final) \
                    and not self._cached_vlm_stale(final):
                self._account_final(final, stats, class_stats)
                return "cached"

        sam_result, sam_error = self._run_sam(
            crop, obj, sample_dir, expected_hash, sam_key, sam_only=options.sam_only
        )
        class_stats["processed"] += 1
        if sam_result is None:
            stats.sam_fail += 1
            class_stats["sam_fail"] += 1
            decision = decide_sample(None, None, None, obj.class_name)
            decision, override_meta = self._apply_override(
                decision, overrides.get(obj.sample_id)
            )
            error_kind = getattr(sam_error, "kind", "run") if sam_error is not None else None
            provenance = build_provenance(
                obj.sample_id, obj, expected_hash, None, None, None, "sam", decision,
                error_kind=error_kind,
            )
            if override_meta:
                provenance["override"] = override_meta
            self._save_decision(sample_dir, expected_hash, sam_key, vlm_key, decision, provenance)
            if decision.status == "KEEP":
                stats.accepted += 1
                class_stats["accepted"] += 1
                return "accepted"
            stats.dropped += 1
            class_stats["dropped"] += 1
            stats.disagreements += 1
            return "dropped"

        stats.sam_ok += 1
        geometry = compute_geometry(obj.bbox, sam_result.bbox, sam_result.mask)
        self._ensure_reconciliation(
            crop, obj, sam_result, sample_dir, expected_hash, sam_key
        )
        if options.sam_only:
            return "sam_only"

        self._ensure_vlm_input(crop, obj, sample_dir, expected_hash, sam_key)
        vlm_result, vlm_error = self._run_vlm(
            sample_dir, expected_hash, vlm_key,
        )
        if vlm_result is None:
            if vlm_error != "transport":
                stats.vlm_fail += 1
                class_stats["vlm_fail"] += 1
            decision = decide_sample(
                sam_result, None, geometry, obj.class_name,
                gates=self.gates, vlm_error=vlm_error,
                ontology=self._ontology(),
            )
        else:
            stats.vlm_calls += 1
            decision = decide_sample(
                sam_result, vlm_result, geometry, obj.class_name, gates=self.gates,
                ontology=self._ontology(),
            )
        decision, override_meta = self._apply_override(
            decision, overrides.get(obj.sample_id)
        )
        provenance = build_provenance(
            obj.sample_id, obj, expected_hash, sam_result, geometry, vlm_result, vlm_error, decision,
        )
        if override_meta:
            provenance["override"] = override_meta
        self._save_decision(
            sample_dir, expected_hash, sam_key, vlm_key, decision, provenance,
        )

        status = decision.status
        if status == "KEEP":
            stats.agreements += 1
            stats.accepted += 1
            class_stats["accepted"] += 1
            self._export_positive(obj, crop, decision, sam_result, provenance)
            stats.hard_negatives_written += self._maybe_random_backgrounds(
                obj, crop, expected_hash, sam_result, provenance
            )
            return "accepted"
        if status == "PENDING":
            stats.pending += 1
            stats.vlm_transport += 1
            class_stats["pending"] += 1
            return "pending"
        stats.dropped += 1
        stats.disagreements += 1
        class_stats["dropped"] += 1
        if self.hard_negatives_enabled and self._is_hard_negative(decision):
            self._write_hard_negative(obj, crop, provenance)
            stats.hard_negatives_written += 1
        return "dropped"

    def _cached_vlm_stale(self, entry: dict) -> bool:
        """Cached decisions from the old biased VLM schema are stale.

        Verdicts recorded before the VLM became blind (six self-confirming
        booleans) cannot be reused: resume must re-inspect those samples so
        every decision reflects the independent SAM vs VLM aggregation.
        SAM-failure entries never reached the VLM and keep their own
        staleness rule.
        """
        vlm_block = (entry.get("provenance") or {}).get("vlm") or {}
        if not isinstance(vlm_block, dict):
            return True
        if vlm_block.get("error") == "sam":
            return False
        return "bbox_covers_object" not in vlm_block and "class_label" not in vlm_block

    def _cached_sam_failure_stale(self, entry: dict) -> bool:
        """sam_failure is final only when the backend was really there.

        A cached DROP with reason ``sam_failure`` produced while ``sam3_mlx``
        was missing (unknown or ``import`` kind) is stale once the backend is
        importable: resume must re-attempt it. Model failures (``run`` kind)
        are deterministic for the given weights and stay final.
        """
        decision = entry.get("decision") or {}
        if decision.get("status") != "DROP" or decision.get("reason") != "sam_failure":
            return False
        vlm_block = (entry.get("provenance") or {}).get("vlm") or {}
        kind = vlm_block.get("error_kind") or "import"
        if kind == "run":
            return False
        # Local backend: sam3_mlx importable? Remote backend: endpoint up?
        available = getattr(self.sam_teacher, "is_available", None)
        if callable(available):
            return bool(available())
        return is_sam_available()

    def _ontology(self) -> list[str]:
        try:
            return list(self.adapter.ontology())
        except Exception:
            return []

    def _run_sam(
        self,
        crop: Image.Image,
        obj: FrigateObject,
        sample_dir: Path,
        expected_hash: str,
        sam_key: str,
        *,
        sam_only: bool,
    ) -> tuple[SamResult | None, SamTeacherError | None]:
        cached = cache.load_sam_cache(sample_dir, expected_hash, sam_key)
        if cached is not None:
            return cached[0], None
        try:
            result = self.sam_teacher.predict(crop, obj.class_name, obj.bbox)
        except SamTeacherError as exc:
            return None, exc
        cache.save_sam_cache(sample_dir, expected_hash, sam_key, result)
        return result, None

    def _ensure_reconciliation(
        self,
        crop: Image.Image,
        obj: FrigateObject,
        sam_result: SamResult,
        sample_dir: Path,
        expected_hash: str,
        sam_key: str,
    ) -> None:
        if cache.reconciliation_is_cached(sample_dir, expected_hash, sam_key):
            return
        image = render_reconciliation_image(
            crop, obj.bbox, sam_result, obj.class_name, sam_result.class_name,
        )
        cache.save_reconciliation_cache(sample_dir, expected_hash, sam_key, image)

    def _ensure_vlm_input(
        self,
        crop: Image.Image,
        obj: FrigateObject,
        sample_dir: Path,
        expected_hash: str,
        sam_key: str,
    ) -> None:
        if cache.vlm_input_is_cached(sample_dir, expected_hash, sam_key):
            return
        image = render_vlm_input_image(crop, obj.bbox)
        cache.save_vlm_input_cache(sample_dir, expected_hash, sam_key, image)

    def _run_vlm(
        self,
        sample_dir: Path,
        expected_hash: str,
        vlm_key: str,
    ) -> tuple[VlmResult | None, str | None]:
        cached = cache.load_vlm_cache(sample_dir, expected_hash, vlm_key)
        if cached is not None:
            return cached, None
        vlm_path = sample_dir / "vlm.png"
        try:
            result = self.reconciler.reconcile(str(vlm_path), self._ontology())
        except VlmTransportError:
            return None, "transport"
        except VlmResultError:
            return None, "malformed"
        cache.save_vlm_cache(sample_dir, expected_hash, vlm_key, result)
        return result, None

    def _save_decision(
        self,
        sample_dir: Path,
        expected_hash: str,
        sam_key: str,
        vlm_key: str,
        decision: Decision,
        provenance: dict[str, Any],
    ) -> None:
        cache.save_decision(
            sample_dir, expected_hash, sam_key, vlm_key, self.pipeline_version,
            decision, provenance,
        )

    def _account_final(
        self,
        entry: dict[str, Any],
        stats: AuditStats,
        class_stats: dict[str, int],
    ) -> None:
        decision = entry.get("decision", {})
        status = decision.get("status")
        provenance = entry.get("provenance", {})
        label = provenance.get("source", {}).get("class_name") or str(decision.get("frigate_label", "") or "")
        cs = stats.class_stats(label) if label else class_stats
        if status == "KEEP":
            stats.agreements += 1
            stats.accepted += 1
            cs["processed"] += 1
            cs["accepted"] += 1
        elif status == "DROP":
            stats.dropped += 1
            stats.disagreements += 1
            cs["processed"] += 1
            cs["dropped"] += 1
        else:
            stats.pending += 1
            cs["processed"] += 1
            cs["pending"] += 1

    def _export_positive(
        self,
        obj: FrigateObject,
        crop: Image.Image,
        decision: Decision,
        sam_result: SamResult,
        provenance: dict[str, Any],
    ) -> None:
        class_id = obj.class_id
        if class_id is None:
            ontology = self.adapter.ontology()
            class_id = ontology.index(obj.class_name) if obj.class_name in ontology else None
        if class_id is None:
            return
        write_positive_sample(
            self.training_root, obj.sample_id, crop, class_id, sam_result, provenance,
        )

    def _is_hard_negative(self, decision: Decision) -> bool:
        """A confirmed false positive: VLM says no object is present."""
        return "object_present" in decision.failing_conditions

    def _write_hard_negative(
        self,
        obj: FrigateObject,
        crop: Image.Image,
        provenance: dict[str, Any],
    ) -> None:
        write_hard_negative(self.training_root, obj.sample_id, crop, provenance)

    def _maybe_random_backgrounds(
        self,
        obj: FrigateObject,
        crop: Image.Image,
        expected_hash: str,
        sam_result: SamResult,
        provenance: dict[str, Any],
    ) -> int:
        """Write optional in-crop random background negatives; returns count."""
        if not self.negative_sampling_enabled or self.samples_per_crop <= 0:
            return 0
        width, height = crop.size
        written = 0
        for i in range(self.samples_per_crop):
            box = sample_background_negative_box(
                width,
                height,
                sam_result.bbox,
                int(expected_hash[:8], 16),
            )
            if box is None:
                continue
            bg_prov = dict(provenance)
            bg_prov["extra"] = {"background_box": box.to_list(), "index": i}
            write_hard_negative(
                self.training_root,
                f"{obj.sample_id}-bg{i}",
                crop,
                bg_prov,
                kind="random_background",
            )
            written += 1
        return written

    @property
    def last_sample_id(self) -> str | None:
        return self._last_sample_id


_SAM_FAILURE_REASON = "sam_failure"


__all__ = [
    "AuditPipeline",
    "AuditStats",
    "PipelineAbort",
    "PipelineOptions",
]
