"""Collection package: batch collection (P1) + sampling/dedup (P3)."""

from .collector import CollectSummary, Collector

__all__ = ["Collector", "CollectSummary"]