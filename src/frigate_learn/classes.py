"""COCO-80 catalog and mask helpers.

The catalog is the COCO-2017 train/val 80-class order used by the Frigate
labelmap; index == position. ``coco_index`` and ``collect_indices`` map class
names to that order, ``trainable_mask`` marks which positions the detector
should back-propagate cls loss through during frozen-head fine-tuning.
"""

from __future__ import annotations

from collections.abc import Iterable

COCO_80: tuple[str, ...] = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)


def coco_index(name: str) -> int:
    return COCO_80.index(name)


def collect_indices(names: Iterable[str]) -> list[int]:
    return [coco_index(n) for n in names]


def trainable_mask(nc: int, trainable: Iterable[str]) -> list[bool]:
    indices = set(collect_indices(trainable))
    return [i in indices for i in range(nc)]


__all__ = ["COCO_80", "coco_index", "collect_indices", "trainable_mask"]