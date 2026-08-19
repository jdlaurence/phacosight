"""Segmentation model factory.

SegFormer (HuggingFace transformers) is the production family — it won the
Stage-1 bake-off (`docs/stage1-segmentation-bakeoff.md`) and both deployment
checkpoints are segformer_b2. The losing candidates (EfficientViT, PIDNet) and
their vendored code were removed in the post-E7 simplification pass; see git
history if they are ever needed again.

Factories return a module whose forward yields logits at input resolution
(N, num_classes, H, W) from an input batch (N, 3, H, W).
"""

import torch
import torch.nn.functional as F
from torch import nn


class _HFSegWrapper(nn.Module):
    """SegformerForSemanticSegmentation outputs logits at H/4; upsample to input."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        logits = self.model(pixel_values=pixel_values).logits
        return F.interpolate(
            logits, size=pixel_values.shape[-2:], mode="bilinear", align_corners=False
        )


def _build_segformer(variant: str, num_classes: int) -> nn.Module:
    from transformers import SegformerForSemanticSegmentation

    checkpoint = {
        "segformer_b0": "nvidia/mit-b0",
        "segformer_b2": "nvidia/mit-b2",
        "segformer_b5": "nvidia/mit-b5",
    }[variant]
    model = SegformerForSemanticSegmentation.from_pretrained(
        checkpoint, num_labels=num_classes, ignore_mismatched_sizes=True
    )
    return _HFSegWrapper(model)


_BUILDERS = {
    "segformer_b0": _build_segformer,
    "segformer_b2": _build_segformer,
    "segformer_b5": _build_segformer,
}


def build_model(name: str, num_classes: int) -> nn.Module:
    if name not in _BUILDERS:
        raise ValueError(f"Unknown model {name!r}; choose from {sorted(_BUILDERS)}")
    return _BUILDERS[name](name, num_classes)
