"""Segmentation dataset over the rasterized mask PNGs + albumentations pipelines.

Masks store the train ID as the pixel value (see ``phacosight.labels``), so
all mask resizing must be nearest-neighbor — albumentations does this for the
``mask`` target by default.
"""

from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

# ImageNet statistics — both SegFormer and the CNN candidates are pretrained with these.
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def train_transforms(size: int) -> A.Compose:
    # Recipe from the LensID/Cataract-1K group papers (see experimentation plan):
    # motion blur + Gaussian blur + random contrast/brightness + shift/scale/rotate.
    return A.Compose(
        [
            A.Resize(size, size),
            A.HorizontalFlip(p=0.5),
            A.Affine(scale=(0.9, 1.1), rotate=(-15, 15), translate_percent=0.05, p=0.5),
            A.OneOf([A.MotionBlur(blur_limit=7), A.GaussianBlur(blur_limit=(3, 7))], p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.Normalize(mean=_MEAN, std=_STD),
            ToTensorV2(),
        ]
    )


def eval_transforms(size: int) -> A.Compose:
    return A.Compose([A.Resize(size, size), A.Normalize(mean=_MEAN, std=_STD), ToTensorV2()])


class CataractSegDataset(Dataset):
    """Rows of a fold DataFrame (``imgs``/``masks`` path columns) → tensors."""

    def __init__(self, df: pd.DataFrame, transforms: A.Compose):
        self.imgs = [Path(p) for p in df["imgs"]]
        self.masks = [Path(p) for p in df["masks"]]
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        image = np.asarray(Image.open(self.imgs[idx]).convert("RGB"))
        mask = np.asarray(Image.open(self.masks[idx]).convert("L"))
        out = self.transforms(image=image, mask=mask)
        return {
            "pixel_values": out["image"].float(),
            "labels": out["mask"].long(),
        }


# Instrument train IDs per mask variant, keyed by the mask folder name
# (mirrors phacosight.labels; anatomy masks lump all tools into ID 4).
_INSTRUMENT_IDS = {
    "mask_anatomy_inst": (4,),
    "mask_instruments": (1,),
    "mask_instruments_MultiClass": tuple(range(1, 7)),
}


def instrument_oversample_weights(df: pd.DataFrame, background_weight: float = 0.25) -> torch.Tensor:
    """Per-frame sampling weights that down-weight frames without instruments.

    Idle/background-only frames dominate the recordings; rare tools appear in
    few frames. Frames whose mask contains any instrument pixel get weight 1,
    the rest ``background_weight``, normalized to sum to 1 (for
    ``WeightedRandomSampler``). Instrument IDs are inferred from the mask
    folder name; unknown folders fall back to "any non-background pixel".
    """
    weights = []
    for p in df["masks"]:
        p = Path(p)
        ids = _INSTRUMENT_IDS.get(p.parent.name)
        arr = np.asarray(Image.open(p).convert("L"))
        has_instrument = bool(np.isin(arr, ids).any()) if ids else bool((arr > 0).any())
        weights.append(1.0 if has_instrument else background_weight)
    w = torch.tensor(weights, dtype=torch.double)
    return w / w.sum()
