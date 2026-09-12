"""COCO-80 catalog and mask helper tests."""

from __future__ import annotations

import pytest

from frigate_learn.classes import COCO_80, coco_index, collect_indices, trainable_mask


def test_coco80_has_80_entries():
    assert len(COCO_80) == 80
    assert len(set(COCO_80)) == 80


def test_coco80_spot_indices():
    assert coco_index("person") == 0
    assert coco_index("car") == 2
    assert coco_index("dog") == 16
    assert coco_index("cat") == 15
    assert coco_index("bicycle") == 1
    assert coco_index("motorcycle") == 3
    assert coco_index("bus") == 5
    assert coco_index("truck") == 7
    assert COCO_80.index("toothbrush") == 79


def test_coco_index_unknown_name_raises():
    with pytest.raises(ValueError):
        coco_index("deer")


def test_collect_indices_default_classes():
    names = ["person", "car", "bicycle", "motorcycle", "bus", "truck", "dog", "cat"]
    assert collect_indices(names) == [0, 2, 1, 3, 5, 7, 16, 15]


def test_trainable_mask():
    mask = trainable_mask(80, ["person", "cat"])
    assert len(mask) == 80
    assert [i == 0 or i == 15 for i in range(80)] == mask
    with pytest.raises(ValueError):
        trainable_mask(80, ["deer"])
    assert trainable_mask(4, ["person", "car"]) == [True, False, True, False]