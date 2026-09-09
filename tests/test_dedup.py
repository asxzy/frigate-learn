"""Perceptual-hash near-duplicate detection tests (Phase 3)."""

from __future__ import annotations

from PIL import Image

from frigate_learn.collection.dedup import (
    dhash_bytes,
    dhash_from_image,
    dhash_file,
    hamming_distance,
    near_hashes,
)


def _solid_color(color: tuple[int, int, int], size: int = 64) -> Image.Image:
    img = Image.new("RGB", (size, size), color)
    return img


def _checker(size: int = 64) -> Image.Image:
    """Alternating stripes: local high/low checkerboard gives mixed dHash bits."""
    pixels = []
    for y in range(size):
        for x in range(size):
            on = ((x // 4) + (y // 4)) % 2 == 0
            pixels.append((255, 255, 255) if on else (0, 0, 0))
    img = Image.new("RGB", (size, size))
    img.putdata(pixels)
    return img


def test_dhash_is_64_bit():
    phash = dhash_from_image(_checker())
    assert len(phash) == 64
    assert set(phash) <= {"0", "1"}
    assert "0" in phash and "1" in phash  # checkerboard produces both bit values


def test_same_image_same_hash():
    first = dhash_from_image(_checker())
    second = dhash_from_image(_checker())
    assert first == second


def test_from_binary_and_dhash_file(tmp_path):
    path = tmp_path / "img.jpg"
    _checker().save(path)
    phash = dhash_file(path)
    assert phash == dhash_bytes(path.read_bytes())
    assert len(phash) == 64


def test_hamming_distance():
    a = "0" * 64
    b = "0" * 63 + "1"
    c = "10" * 32
    assert hamming_distance(a, b) == 1
    assert hamming_distance(a, a) == 0
    assert hamming_distance(a, c) == 32


def test_near_hashes():
    a = dhash_from_image(_checker())
    b = dhash_from_image(_checker())  # identical -> distance 0
    c = dhash_from_image(_solid_color((200, 200, 200)))  # far away (all-1s)

    close = near_hashes(a, [b, c], threshold=3)
    assert b in close and c not in close
    assert near_hashes(a, [a], threshold=0) == [a]
    assert near_hashes(a, [], threshold=10) == []
    assert near_hashes(a, [None, ""], threshold=10) == []