"""Deterministic geometry engine tests: IoU, containment, conversions, clipping."""

from __future__ import annotations

import numpy as np

from frigate_learn.audit.geometry import (
    bbox_containment,
    bbox_iou,
    box_edge_deltas,
    clip_box_to_image,
    compute_geometry,
    convert_cxcywh_to_xyxy,
    convert_frame_to_crop,
    convert_normalized_xyxy,
    convert_xywh_to_xyxy,
    intersection_area,
    union_area,
)
from frigate_learn.audit.types import BoundingBox, Mask


def test_intersection_area_disjoint_and_overlap():
    a = BoundingBox(0, 0, 10, 10)
    b = BoundingBox(20, 20, 30, 30)
    assert intersection_area(a, b) == 0.0
    c = BoundingBox(5, 5, 15, 15)
    assert intersection_area(a, c) == 25.0


def test_union_area_and_iou():
    a = BoundingBox(0, 0, 10, 10)
    c = BoundingBox(5, 5, 15, 15)
    assert union_area(a, c) == 175.0
    assert abs(bbox_iou(a, c) - 25.0 / 175.0) < 1e-9
    assert bbox_iou(a, BoundingBox(50, 50, 60, 60)) == 0.0
    # degenerate box => 0 IoU, no exception
    assert bbox_iou(a, BoundingBox(5, 5, 5, 5)) == 0.0


def test_bbox_containment_fraction():
    outer = BoundingBox(0, 0, 100, 100)
    inner = BoundingBox(10, 20, 30, 40)
    assert bbox_containment(inner, outer) == 1.0
    # inner half inside, half outside the outer box
    straddling = BoundingBox(50, 0, 150, 100)
    assert abs(bbox_containment(straddling, outer) - 0.5) < 1e-9
    # degenerate inner
    assert bbox_containment(BoundingBox(5, 5, 5, 5), outer) == 0.0


def test_box_edge_deltas_dict():
    frigate = BoundingBox(0, 0, 100, 100)
    sam = BoundingBox(10, 20, 90, 120)
    d = box_edge_deltas(frigate, sam)
    assert d["left_delta"] == 10
    assert d["top_delta"] == 20
    assert d["right_delta"] == -10
    assert d["bottom_delta"] == 20


def test_clip_box_to_image():
    box = BoundingBox(-5, 10, 105, 120)
    clipped = clip_box_to_image(box, width=100, height=110)
    assert clipped.x1 == 0.0
    assert clipped.y2 == 110.0
    assert clipped.is_valid()
    empty = clip_box_to_image(BoundingBox(200, 200, 300, 300), width=100, height=100)
    assert empty.is_valid() is False


def test_mask_helpers():
    m = np.zeros((10, 10), dtype=bool)
    m[2:7, 3:8] = True
    mask = Mask(m)
    assert mask.area == 25
    tb = mask.tight_bbox()
    assert tb is not None
    assert (tb.x1, tb.y1, tb.x2, tb.y2) == (3.0, 2.0, 8.0, 7.0)
    empty = Mask(np.zeros((5, 5), dtype=bool))
    assert empty.area == 0
    assert empty.tight_bbox() is None
    from frigate_learn.audit.geometry import mask_area, mask_to_tight_bbox

    assert mask_area(mask) == 25
    assert mask_to_tight_bbox(mask) is not None
    assert mask_to_tight_bbox(empty) is None


def test_mask_bbox_ratio():
    from frigate_learn.audit.geometry import mask_bbox_ratio

    mask = Mask(np.zeros((20, 20), dtype=bool))
    mask.data[5:15, 5:15] = True   # 100 px
    box = BoundingBox(0, 0, 20, 20)   # 400 px
    assert abs(mask_bbox_ratio(mask, box) - 0.25) < 1e-9
    assert mask_bbox_ratio(mask, BoundingBox(0, 0, 0, 0)) == 0.0


def test_mask_containment():
    from frigate_learn.audit.geometry import mask_containment

    mask = Mask(np.zeros((20, 20), dtype=bool))
    mask.data[0:10, 0:10] = True
    assert mask_containment(mask, BoundingBox(0, 0, 10, 10)) == 1.0
    # half-size box covers 25 of the 100 mask pixels
    assert abs(mask_containment(mask, BoundingBox(0, 0, 5, 5)) - 0.25) < 1e-9
    assert mask_containment(mask, BoundingBox(100, 100, 110, 110)) == 0.0
    assert mask_containment(Mask(np.zeros((5, 5), dtype=bool)), BoundingBox(0, 0, 5, 5)) == 0.0


def test_conversions():
    n = convert_normalized_xyxy(BoundingBox(0.1, 0.2, 0.3, 0.4), width=100, height=200)
    assert (n.x1, n.y1, n.x2, n.y2) == (10.0, 40.0, 30.0, 80.0)
    w = convert_xywh_to_xyxy(10, 20, 30, 40)
    assert (w.x1, w.y1, w.x2, w.y2) == (10.0, 20.0, 40.0, 60.0)
    c = convert_cxcywh_to_xyxy(50, 60, 20, 30)
    assert (c.x1, c.y1, c.x2, c.y2) == (40.0, 45.0, 60.0, 75.0)


def test_convert_frame_to_crop():
    # frame-relative pixel box into a crop whose top-left is (25, 50)
    local = convert_frame_to_crop(BoundingBox(30, 60, 70, 90), 25, 50)
    assert (local.x1, local.y1, local.x2, local.y2) == (5.0, 10.0, 45.0, 40.0)
    # result may go negative; callers clamp afterwards
    neg = convert_frame_to_crop(BoundingBox(10, 10, 40, 40), 20, 30)
    assert (neg.x1, neg.y1) == (-10.0, -20.0)


def test_compute_geometry_full():
    frigate = BoundingBox(0, 0, 100, 100)
    sam = BoundingBox(10, 10, 90, 90)
    m = np.zeros((100, 100), dtype=bool)
    m[20:80, 20:80] = True
    g = compute_geometry(frigate, sam, Mask(m))
    assert g.bbox_iou > 0
    assert g.mask_area == 3600
    assert g.mask_bbox_ratio > 0
    assert g.mask_frigate_containment > 0
    assert g.mask_sam_containment > 0
    assert g.frigate_bbox_area == 10000.0
    assert g.sam_bbox_area == 6400.0
    assert g.edge_deltas["left_delta"] == 10.0
    d = g.as_dict()
    assert "bbox_iou" in d and "edge_deltas" in d and "mask_area" in d


def test_compute_geometry_degenerate_sam():
    frigate = BoundingBox(0, 0, 100, 100)
    sam = BoundingBox(10, 10, 10, 10)
    g = compute_geometry(frigate, sam, Mask(np.zeros((100, 100), dtype=bool)))
    assert g.bbox_iou == 0.0
    assert g.mask_bbox_ratio == 0.0
    assert g.mask_area == 0


def test_mask_type_area_and_tight_bbox():
    m = np.zeros((10, 10), dtype=bool)
    m[1:4, 2:6] = True
    mask = Mask(m)
    assert mask.area == 12
    tb = mask.tight_bbox()
    assert tb is not None
    assert (tb.x1, tb.y1, tb.x2, tb.y2) == (2.0, 1.0, 6.0, 4.0)
    uint8 = mask.to_uint8()
    assert uint8.dtype == np.uint8
    assert uint8.max() == 255
    assert uint8.min() == 0
