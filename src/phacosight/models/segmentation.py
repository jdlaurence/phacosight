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


class _UpsampleWrapper(nn.Module):
    """For models emitting logits below input resolution (e.g. stride-8)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        logits = self.model(pixel_values)
        return F.interpolate(
            logits, size=pixel_values.shape[-2:], mode="bilinear", align_corners=False
        )


def _build_efficientvit(variant: str, num_classes: int) -> nn.Module:
    try:
        from efficientvit.models import efficientvit as ev
    except ImportError as e:
        raise ImportError(
            "EfficientViT is not installed. Run: pip install "
            "git+https://github.com/mit-han-lab/efficientvit.git"
        ) from e
    from huggingface_hub import hf_hub_download

    short = variant.split("_")[-1]  # b1, b2
    model = getattr(ev, f"efficientvit_seg_{short}")(dataset="cityscapes")
    ckpt = hf_hub_download("han-cai/efficientvit-seg", f"efficientvit_seg_{short}_cityscapes.pt")
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(sd.get("state_dict", sd))
    # swap the 19-class Cityscapes classifier for ours (stride-8 logits → upsample)
    final = model.head.output_ops[0].op_list[1]
    old = final.conv
    final.conv = nn.Conv2d(old.in_channels, num_classes, kernel_size=1, bias=old.bias is not None)
    return _UpsampleWrapper(model)


def _build_pidnet(variant: str, num_classes: int) -> nn.Module:
    import sys
    from pathlib import Path

    repo_root = str(Path(__file__).resolve().parents[3])  # third_party/ lives here
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    try:
        from third_party.pidnet.pidnet import get_pred_model
    except ImportError as e:
        raise ImportError(
            "PIDNet source not vendored. Copy models/pidnet.py (and its imports) "
            "from https://github.com/XuJiacong/PIDNet into third_party/pidnet/."
        ) from e
    # single-head inference variant (augment=False), stride-8 logits; trains
    # from scratch — the ImageNet init lives on the authors' Google Drive.
    return _UpsampleWrapper(get_pred_model(name=variant.replace("_", "-"), num_classes=num_classes))


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
