"""Masked per-class cls loss and trainer for frozen-head fine-tuning.

``MaskedDetectionLoss.get_assigned_targets_and_loss`` is a verbatim copy of
the installed ultralytics 8.4.146 body (version-verified) with a single
change: the cls term is routed through ``mask_cls_loss`` so frozen head rows
contribute exactly zero and the denominator sums only trainable columns.
"""

# ruff: noqa: E501

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

# ---
try:
    import torch
    from ultralytics.utils.loss import v8DetectionLoss
    from ultralytics.utils.tal import make_anchors
    from ultralytics.models.yolo.detect.train import DetectionTrainer
except ImportError:
    torch = None
    v8DetectionLoss = None
    make_anchors = None
    DetectionTrainer = None

_MISSING_EXTRA = (
    "ultralytics/torch not installed; install the 'ml' extra on the training "
    "machine (pip install -e '.[ml]')"
)

_LossBase = v8DetectionLoss if v8DetectionLoss is not None else object
_TrainerBase = DetectionTrainer if DetectionTrainer is not None else object


def _to_indices(trainable: Iterable[int] | Iterable[bool] | None) -> list[int]:
    items = list(trainable) if trainable is not None else []
    if items and all(isinstance(x, bool) for x in items):
        return [i for i, flag in enumerate(items) if flag]
    return [int(x) for x in items]


def mask_cls_loss(
    bce_loss: Any, target_scores: Any, mask: Any
) -> Any:
    """Classification loss over trainable columns only."""
    mask = mask.view(1, 1, -1)
    return (bce_loss * mask).sum() / max((target_scores * mask).sum(), 1)


class MaskedDetectionLoss(_LossBase):
    def __init__(self, model, *, trainable: Iterable[int]) -> None:
        if v8DetectionLoss is None or torch is None:
            raise RuntimeError(_MISSING_EXTRA)
        super().__init__(model)
        indices = _to_indices(trainable)
        if not indices:
            raise ValueError("trainable must be non-empty")
        bad = [i for i in indices if not 0 <= i < self.nc]
        if bad:
            raise ValueError(f"trainable indices must be within [0, {self.nc}); out-of-range: {bad}")
        self.cls_mask = torch.zeros(1, 1, self.nc, dtype=torch.float32, device=self.device)
        self.cls_mask[:, :, indices] = 1.0

    def get_assigned_targets_and_loss(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> tuple:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size and return foreground mask and
        target indices.
        """
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss with optional class weighting
        bce_loss = self.bce(pred_scores, target_scores.to(dtype))  # (bs, num_anchors, nc)
        if self.class_weights is not None:
            bce_loss *= self.class_weights
        loss[1] = mask_cls_loss(bce_loss, target_scores, self.cls_mask)

        # Bbox loss
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )
        # WARNING: line below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[0] += pred_distri[..., :0].sum()

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            dict(zip(self.loss_names, loss.detach())),
        )  # loss(box, cls, dfl)


class MaskedDetectionTrainer(_TrainerBase):
    def __init__(self, trainable=None, **kwargs) -> None:
        if DetectionTrainer is None or torch is None:
            raise RuntimeError(_MISSING_EXTRA)
        self._trainable = _to_indices(trainable)
        if not self._trainable:
            raise ValueError("trainable must be non-empty")
        super().__init__(**kwargs)

    def set_model_attributes(self):
        super().set_model_attributes()
        self.model.criterion = MaskedDetectionLoss(self.model, trainable=self._trainable)

    def get_model(self, cfg, weights=None, verbose=True):
        return super().get_model(cfg, weights, verbose)


__all__ = ["MaskedDetectionLoss", "MaskedDetectionTrainer", "mask_cls_loss"]