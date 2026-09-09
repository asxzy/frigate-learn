"""Deterministic dataset splits.

Splits are assigned from a stable hash of the *sample_id* so the same manifest
always produces the same split — no random state, no ordering dependence, and
re-running the builder is reproducible. This is what the golden dataset uses
(100% of its samples are, by construction, in the ``golden`` split).
"""

from __future__ import annotations

import hashlib
from typing import Iterable

VALID_SPLITS = ["train", "val", "test"]


def _bucket(sample_id: str, seed: str) -> int:
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def assign_split(
    sample_id: str,
    ratios: dict[str, float] | None = None,
    seed: str = "frigate-learn-v1",
) -> str:
    """Assign a sample to ``train|val|test`` deterministically."""
    if ratios is None:
        ratios = {"train": 0.8, "val": 0.1, "test": 0.1}
    total = sum(ratios.values())
    if total <= 0:
        raise ValueError("ratios must sum to a positive number")
    # Pure-integer bucketing: fraction of the 64-bit space in [0,1).
    bucket = _bucket(sample_id, seed)
    fraction = bucket / (1 << 64)
    cumulative = 0.0
    for split in VALID_SPLITS:
        if split not in ratios:
            continue
        cumulative += ratios[split] / total
        if fraction < cumulative:
            return split
    return next(s for s in VALID_SPLITS if ratios.get(s, 0) > 0)


def make_splits(
    sample_ids: Iterable[str],
    ratios: dict[str, float] | None = None,
    seed: str = "frigate-learn-v1",
    flatten: bool = False,
) -> dict[str, list[str]]:
    """Partition sample ids into `{split: [ids]}` deterministically.

    ``ratios`` default is ``{"train": 0.8, "val": 0.1, "test": 0.1}``.
    ``flatten=False`` (default) keeps ids in their original relative order;
    pass ``flatten=True`` for fully deterministic (hash-sorted) output.
    """
    if ratios is None:
        ratios = {"train": 0.8, "val": 0.1, "test": 0.1}
    ids = list(sample_ids)
    if flatten:
        ids.sort(key=lambda s: (assign_split(s, ratios, seed), s))
    result: dict[str, list[str]] = {split: [] for split in VALID_SPLITS}
    for sample_id in ids:
        result[assign_split(sample_id, ratios, seed)].append(sample_id)
    return {k: v for k, v in result.items() if v}


__all__ = ["assign_split", "make_splits", "VALID_SPLITS"]