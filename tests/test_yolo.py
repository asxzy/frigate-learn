"""YOLO label format tests."""

from __future__ import annotations

import pytest

from frigate_learn.dataset.yolo import (
    ImageSize,
    YoloLine,
    bbox_to_yolo,
    normalize_bbox,
    parse_yolo_line,
    read_yolo_label,
    validate_normalized,
    write_yolo_label,
    yolo_to_bbox,
)


def test_normalize_pixel_box():
    box = normalize_bbox(100, 50, 200, 150, ImageSize(width=1000, height=500))
    assert (box.x1, box.y1, box.x2, box.y2) == pytest.approx((0.1, 0.1, 0.2, 0.3))


def test_normalize_without_size_is_identity():
    box = normalize_bbox(0.25, 0.25, 0.5, 0.75)
    assert (box.x1, box.y1, box.x2, box.y2) == pytest.approx((0.25, 0.25, 0.5, 0.75))


def test_degenerate_box_raises():
    with pytest.raises(ValueError):
        normalize_bbox(0.2, 0.2, 0.2, 0.4)
    with pytest.raises(ValueError):
        normalize_bbox(0.4, 0.2, 0.2, 0.4)


def test_out_of_range_with_size_raises():
    with pytest.raises(ValueError):
        normalize_bbox(-10, 0, 100, 100, ImageSize(width=100, height=100))


def test_center_wh_roundtrip():
    xyxy = (0.1, 0.2, 0.5, 0.8)
    cx, cy, w, h = bbox_to_yolo(*xyxy)
    assert yolo_to_bbox(cx, cy, w, h) == pytest.approx(xyxy)


def test_validate_normalized():
    assert validate_normalized(0, 0, 1, 1)
    assert not validate_normalized(0, 0, 0, 1)  # zero width
    assert not validate_normalized(-0.5, 0, 1, 1)
    assert not validate_normalized(0, 0, 1.5, 1)


def test_parse_yolo_line_ok():
    line = parse_yolo_line("3 0.500000 0.500000 0.250000 0.250000")
    assert line == YoloLine(3, 0.5, 0.5, 0.25, 0.25)
    assert line.to_text().startswith("3 0.500000")


@pytest.mark.parametrize(
    "bad",
    [
        "1 2 3",                   # wrong field count
        "a 0.5 0.5 0.2 0.2",       # non-int class
        "1 -0.5 0.5 0.2 0.2",      # invalid center
        "1 0.5 0.5 0 0.2",         # zero width
        "1 0.5 0.5 0.2 -0.2",      # negative height
        "1.5 0.5 0.5 0.2 0.2",     # non-int class
        "5 0.5 0.5 0.2 0.2",       # class oob for num_classes=4
    ],
)
def test_parse_yolo_line_bad(bad):
    with pytest.raises(ValueError):
        parse_yolo_line(bad, num_classes=4)


def test_write_read_roundtrip(tmp_path):
    label = tmp_path / "l.txt"
    lines = [YoloLine(0, 0.5, 0.4, 0.3, 0.2), YoloLine(1, 0.2, 0.2, 0.1, 0.1)]
    write_yolo_label(label, lines)
    assert read_yolo_label(label) == lines


def test_read_missing_file(tmp_path):
    assert read_yolo_label(tmp_path / "absent.txt") == []