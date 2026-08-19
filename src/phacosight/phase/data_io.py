"""Shared feature-cache I/O for the phase scripts (train/eval/confidence/external).

The stride rules live here once: 1 fps heads fed 5 fps sequences silently collapse
(MS-TCN++ receptive fields are in samples — the 2026-08-14 incident class), so every
consumer reads ``frame_stride`` from checkpoints rather than guessing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def load_case(feat_dir: Path | str, case: str, extra_dirs=(), stride: int = 1):
    """(features [T, D], labels [T]) at the checkpoint's stride; extra feature dirs
    (e.g. tool features) are length-checked and concatenated on the channel axis."""
    d = np.load(Path(feat_dir) / f"{case}.npz")
    feats = [d["features"].astype(np.float32)]
    for e in extra_dirs:
        x = np.load(Path(e) / f"{case}.npz")["features"].astype(np.float32)
        assert len(x) == len(feats[0]), f"{case}: extra feature length mismatch"
        feats.append(x)
    return np.concatenate(feats, axis=1)[::stride], d["labels"].astype(np.int64)[::stride]


def run_stride(run_dirs) -> tuple[int, float]:
    """(frame_stride, effective fps) shared by all runs — mixed-rate ensembles are a bug."""
    seen = set()
    for rd in run_dirs:
        cfg = torch.load(Path(rd) / "fold0" / "val_best.pt",
                         map_location="cpu", weights_only=False)["config"]
        seen.add((int(cfg.get("frame_stride", 1)), float(cfg.get("features_fps", 5.0))))
    assert len(seen) == 1, f"runs disagree on stride/fps: {seen}"
    stride, ffps = seen.pop()
    return stride, ffps / stride
