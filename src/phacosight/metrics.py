"""Per-class IoU/Dice accumulated over a whole evaluation split via a
confusion matrix (exact, streaming, resolution-independent)."""

import numpy as np
import torch


class SegMetrics:
    def __init__(self, num_classes: int, class_names: tuple[str, ...] | None = None):
        self.num_classes = num_classes
        self.class_names = class_names or tuple(str(i) for i in range(num_classes))
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        # pred/target: (N, H, W) int tensors of train IDs
        p = pred.reshape(-1).cpu().numpy()
        t = target.reshape(-1).cpu().numpy()
        valid = (t >= 0) & (t < self.num_classes)
        idx = t[valid] * self.num_classes + p[valid]
        self.confusion += np.bincount(
            idx, minlength=self.num_classes**2
        ).reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict:
        tp = np.diag(self.confusion).astype(np.float64)
        fp = self.confusion.sum(axis=0) - tp
        fn = self.confusion.sum(axis=1) - tp
        denom_iou = tp + fp + fn
        denom_dice = 2 * tp + fp + fn
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = np.where(denom_iou > 0, tp / denom_iou, np.nan)
            dice = np.where(denom_dice > 0, 2 * tp / denom_dice, np.nan)
        return {
            "per_class_iou": dict(zip(self.class_names, iou.tolist())),
            "per_class_dice": dict(zip(self.class_names, dice.tolist())),
            "miou": float(np.nanmean(iou)),
            "mdice": float(np.nanmean(dice)),
        }
