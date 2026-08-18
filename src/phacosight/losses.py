"""CE + log-soft-Dice loss, the recipe used across the Cataract-1K group's
segmentation papers (LensID, DeepPyram): L = CE + lambda * (-log softDice)."""

import torch
import torch.nn.functional as F
from torch import nn


class CrossEntropyLogDice(nn.Module):
    def __init__(self, num_classes: int, dice_weight: float = 0.8, smooth: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: (N, C, H, W); target: (N, H, W) int64 train IDs
        ce = F.cross_entropy(logits, target)
        probs = torch.softmax(logits, dim=1)
        one_hot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = (probs * one_hot).sum(dims)
        cardinality = probs.sum(dims) + one_hot.sum(dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return ce - self.dice_weight * torch.log(dice.mean())
