"""CPU smoke tests: dataset → transforms → loss → metrics with synthetic data."""

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from cataract_video.data.seg_dataset import (
    CataractSegDataset,
    eval_transforms,
    instrument_oversample_weights,
    train_transforms,
)
from cataract_video.labels import ANATOMY_INSTRUMENT, INSTRUMENT_MULTICLASS, TASKS
from cataract_video.losses import CrossEntropyLogDice
from cataract_video.metrics import SegMetrics


@pytest.fixture
def synthetic_frame(tmp_path):
    """Three fake 1024x768 frames + masks in the upstream layout."""
    rng = np.random.default_rng(0)
    rows = []
    case = tmp_path / "case_9999"
    (case / "img").mkdir(parents=True)
    (case / ANATOMY_INSTRUMENT.mask_dir).mkdir()
    for i in range(3):
        img = case / "img" / f"case9999_{i:02d}.png"
        mask = case / ANATOMY_INSTRUMENT.mask_dir / f"case9999_{i:02d}.png"
        Image.fromarray(rng.integers(0, 255, (768, 1024, 3), dtype=np.uint8)).save(img)
        # frame 0 is background-only; others contain all 5 classes
        mask_arr = np.zeros((768, 1024), dtype=np.uint8)
        if i > 0:
            mask_arr = rng.integers(0, ANATOMY_INSTRUMENT.num_classes, (768, 1024)).astype(np.uint8)
        Image.fromarray(mask_arr, mode="L").save(mask)
        rows.append({"imgs": img, "masks": mask})
    return pd.DataFrame(rows)


def test_dataset_shapes_and_dtypes(synthetic_frame):
    for tfm in (train_transforms(256), eval_transforms(256)):
        ds = CataractSegDataset(synthetic_frame, tfm)
        sample = ds[1]
        assert sample["pixel_values"].shape == (3, 256, 256)
        assert sample["pixel_values"].dtype == torch.float32
        assert sample["labels"].shape == (256, 256)
        assert sample["labels"].dtype == torch.int64
        assert int(sample["labels"].max()) < ANATOMY_INSTRUMENT.num_classes


def test_mask_values_survive_resize_exactly(synthetic_frame):
    # nearest-neighbor mask resize must not invent interpolated class IDs
    ds = CataractSegDataset(synthetic_frame, eval_transforms(200))
    labels = ds[2]["labels"]
    assert set(labels.unique().tolist()) <= set(range(ANATOMY_INSTRUMENT.num_classes))


def test_oversample_weights(synthetic_frame):
    w = instrument_oversample_weights(synthetic_frame)
    assert w.shape == (3,)
    assert torch.isclose(w.sum(), torch.tensor(1.0, dtype=torch.double))
    assert w[0] < w[1]  # background-only frame down-weighted


def test_loss_decreases_on_perfect_logits():
    criterion = CrossEntropyLogDice(num_classes=5, dice_weight=0.8)
    target = torch.randint(0, 5, (2, 32, 32))
    perfect = torch.nn.functional.one_hot(target, 5).permute(0, 3, 1, 2).float() * 20
    random = torch.randn(2, 5, 32, 32)
    assert criterion(perfect, target) < criterion(random, target)


def test_loss_backward():
    criterion = CrossEntropyLogDice(num_classes=5)
    logits = torch.randn(2, 5, 16, 16, requires_grad=True)
    criterion(logits, torch.randint(0, 5, (2, 16, 16))).backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_metrics_perfect_and_disjoint():
    m = SegMetrics(3, ("bg", "a", "b"))
    t = torch.randint(0, 3, (1, 64, 64))
    m.update(t, t)
    out = m.compute()
    assert out["miou"] == pytest.approx(1.0)
    m2 = SegMetrics(2, ("bg", "a"))
    m2.update(torch.zeros(1, 8, 8, dtype=torch.long), torch.ones(1, 8, 8, dtype=torch.long))
    assert m2.compute()["per_class_iou"]["a"] == 0.0


def test_task_registry_consistency():
    assert set(TASKS) == {"anatomy_instrument", "instrument_binary", "instrument_multiclass"}
    assert INSTRUMENT_MULTICLASS.num_classes == 7  # bg + 6 grouped tools
