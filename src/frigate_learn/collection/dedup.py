"""Perceptual hashing for near-duplicate detection (Phase 3).

Uses a 64-bit dHash (difference hash): grayscale -> 9x8 -> compare adjacent
pixels. The Hamming distance between two dHashes is a robust proxy for visual
similarity independent of encoding/mislabel, which exact SHA-256 dedup misses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image


def dhash_bytes(data: bytes, size: int = 8) -> str:
    """Return the 64-bit dHash of already-decoded image bytes as a hex string."""
    with Image.open(__import__("io").BytesIO(data)) as img:
        return dhash_from_image(img, size=size)


def dhash_file(path: Path, size: int = 8) -> str:
    with Image.open(path) as img:
        return dhash_from_image(img, size=size)


def dhash_from_image(img: Image.Image, size: int = 8) -> str:
    """Compute dHash(size*size bits) from an open PIL image."""
    gray = img.convert("L").resize((size + 1, size))
    pixels = list(gray.get_flattened_data())
    bits: list[str] = []
    for row in range(size):
        base = row * (size + 1)
        for col in range(size):
            bits.append("1" if pixels[base + col] >= pixels[base + col + 1] else "0")
    return "".join(bits)


def hamming_distance(a: str, b: str) -> int:
    """Number of differing bits (pad shorter string with leading zeros)."""
    if a == b:
        return 0
    width = max(len(a), len(b))
    a = a.zfill(width)
    b = b.zfill(width)
    return sum(x != y for x, y in zip(a, b))


def near_hashes(candidate: str, existing: Iterable[str], threshold: int = 10) -> list[str]:
    """Return every existing hash within ``threshold`` Hamming bits (<= inclusive)."""
    found = []
    for other in existing:
        if other is None or not other:
            continue
        if hamming_distance(candidate, other) <= threshold:
            found.append(other)
    return found


__all__ = ["dhash_bytes", "dhash_file", "dhash_from_image", "hamming_distance", "near_hashes"]