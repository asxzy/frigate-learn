"""Masked per-class cls loss and trainer tests (frozen-head fine-tuning)."""

from __future__ import annotations

import pytest

pytest.importorskip("ultralytics")
pytest.importorskip("torch")

from types import SimpleNamespace

import torch
from ultralytics import YOLO

from frigate_learn.training.masked import (
    MaskedDetectionLoss,
    MaskedDetectionTrainer,
    mask_cls_loss,
)


def test_mask_cls_loss_all_trainable_equals_plain():
    bce = torch.rand(2, 9, 5) * 2
    tgt = torch.rand(2, 9, 5)
    mask = torch.ones(5)
    expected = bce.sum() / max(tgt.sum(), 1)
    assert torch.allclose(mask_cls_loss(bce, tgt, mask), expected)


def test_mask_cls_loss_zeroes_frozen_columns():
    bce = torch.rand(2, 9, 4)
    tgt = torch.rand(2, 9, 4)
    mask = torch.tensor([True, True, False, True])
    bce_zero = bce.clone()
    bce_zero[:, :, 2] = 0.0
    tgt_zero = tgt.clone()
    tgt_zero[:, :, 2] = 0.0
    recompute = mask_cls_loss(bce_zero, tgt_zero, torch.ones(4))
    assert torch.allclose(mask_cls_loss(bce, tgt, mask), recompute)

    frozen_only = torch.zeros(2, 9, 4)
    frozen_only[:, :, 2] = 1.0
    frozen_bce = torch.zeros(2, 9, 4)
    frozen_bce[:, :, 2] = 3.0
    assert float(mask_cls_loss(frozen_bce, frozen_only, mask)) == 0.0


def test_mask_loss_zero_gradient_on_frozen():
    pred = torch.randn(2, 9, 4, requires_grad=True)
    tgt = torch.zeros(2, 9, 4)
    tgt[:, 0, 2] = 1.0
    tgt[:, 1, 1] = 1.0
    mask = torch.tensor([True, True, False, True])
    mask_cls_loss(pred, tgt, mask).backward()
    assert float(pred.grad[:, :, 2].abs().sum()) == 0.0
    assert float(pred.grad[:, :, 1].abs().sum()) > 0.0


def test_masked_detection_loss_frozen_class_adds_nothing():
    model = YOLO("yolov8n.yaml").model
    model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
    model.train()
    batch = dict(
        img=torch.rand(1, 3, 64, 64),
        batch_idx=torch.zeros(2, dtype=torch.int32),
        cls=torch.tensor([3, 3], dtype=torch.float32),
        bboxes=torch.tensor(
            [[20.0, 20.0, 20.0, 20.0], [30.0, 30.0, 10.0, 10.0]],
            dtype=torch.float32,
        ),
    )
    no_gt = dict(
        img=batch["img"],
        batch_idx=torch.empty(0, dtype=torch.int32),
        cls=torch.empty(0, dtype=torch.float32),
        bboxes=torch.empty(0, 4, dtype=torch.float32),
    )

    def cls_loss_at(target_batch, *trainable):
        model.criterion = MaskedDetectionLoss(model, trainable=trainable)
        return float(model.loss(target_batch)[1]["cls_loss"])

    frozen = cls_loss_at(batch, 0, 15)
    frozen_no_gt = cls_loss_at(no_gt, 0, 15)
    full = cls_loss_at(batch, *range(80))

    assert 0 < frozen
    assert frozen == pytest.approx(frozen_no_gt)
    assert frozen < full


def test_masked_trainer_get_model_attaches_criterion(monkeypatch, tmp_path):
    from frigate_learn.training.masked import DetectionTrainer

    monkeypatch.setattr(DetectionTrainer, "get_dataset", lambda self: {"nc": 80})

    captured = {}

    def _get_model(self, cfg, weights=None, verbose=True):
        m = YOLO("yolov8n.yaml").model
        m.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
        captured["model"] = m
        return m

    monkeypatch.setattr(DetectionTrainer, "get_model", _get_model)

    trainer = MaskedDetectionTrainer(
        trainable=(0, 15, 16),
        overrides={
            "data": {"nc": 80},
            "model": "yolov8n.yaml",
            "project": str(tmp_path / "runs"),
        },
    )
    model = trainer.get_model("yolov8n.yaml")
    assert model is captured["model"]
    criterion = model.criterion
    assert isinstance(criterion, MaskedDetectionLoss)
    mask = criterion.cls_mask
    assert mask.shape == (1, 1, 80)
    assert mask.sum() == 3
    for i in (0, 15, 16):
        assert mask[0, 0, i] == 1.0
    assert mask[0, 0, 1] == 0.0


def test_masked_trainer_empty_trainable_raises():
    from frigate_learn.training.masked import MaskedDetectionTrainer

    with pytest.raises(ValueError):
        MaskedDetectionTrainer(
            trainable=(), overrides={"data": {"nc": 80}, "model": "yolov8n.yaml"}
        )


def test_masked_loss_rejects_bad_index():
    model = YOLO("yolov8n.yaml").model
    model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
    with pytest.raises(ValueError):
        MaskedDetectionLoss(model, trainable=(0, 999))
    with pytest.raises(ValueError):
        MaskedDetectionLoss(model, trainable=())
    bool_mask = [i in (0, 15, 16) for i in range(80)]
    criterion = MaskedDetectionLoss(model, trainable=bool_mask)
    assert criterion.cls_mask[0, 0, 0] == 1.0
    assert criterion.cls_mask[0, 0, 15] == 1.0
    assert criterion.cls_mask[0, 0, 16] == 1.0
    assert criterion.cls_mask[0, 0, 1] == 0.0
    assert criterion.cls_mask.sum() == 3