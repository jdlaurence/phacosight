"""Model factory for the segmentation bake-off.

Candidates (see the experimentation plan):
- segformer_b2 / segformer_b0: HuggingFace transformers, fully supported here.
- efficientvit_b1: mit-han-lab/efficientvit — optional dependency, install with
  ``pip install git+https://github.com/mit-han-lab/efficientvit.git``.
- pidnet_s: no maintained package; vendor the model file from
  https://github.com/XuJiacong/PIDNet into third_party/pidnet before use.

All factories return a module whose forward yields logits at input resolution
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


def _build_efficientvit(variant: str, num_classes: int) -> nn.Module:
    try:
        from efficientvit.seg_model_zoo import create_efficientvit_seg_model
    except ImportError as e:
        raise ImportError(
            "EfficientViT is not installed. Run: pip install "
            "git+https://github.com/mit-han-lab/efficientvit.git"
        ) from e
    # cityscapes-pretrained head is replaced to num_classes by the zoo API
    return create_efficientvit_seg_model(
        name=f"efficientvit-seg-{variant.split('_')[-1]}-cityscapes", n_classes=num_classes
    )


def _build_pidnet(variant: str, num_classes: int) -> nn.Module:
    try:
        from third_party.pidnet.pidnet import get_pred_model
    except ImportError as e:
        raise ImportError(
            "PIDNet source not vendored. Copy models/pidnet.py (and its imports) "
            "from https://github.com/XuJiacong/PIDNet into third_party/pidnet/."
        ) from e
    return get_pred_model(name=variant.replace("_", "-"), num_classes=num_classes)


_BUILDERS = {
    "segformer_b0": _build_segformer,
    "segformer_b2": _build_segformer,
    "segformer_b5": _build_segformer,
    "efficientvit_b1": _build_efficientvit,
    "efficientvit_b2": _build_efficientvit,
    "pidnet_s": _build_pidnet,
    "pidnet_l": _build_pidnet,
}


def build_model(name: str, num_classes: int) -> nn.Module:
    if name not in _BUILDERS:
        raise ValueError(f"Unknown model {name!r}; choose from {sorted(_BUILDERS)}")
    return _BUILDERS[name](name, num_classes)
