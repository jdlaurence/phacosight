#!/usr/bin/env python
"""Cache frozen DINOv2 features + labels for the 56 phase videos.

One .npz per (video, config) with:
- features [T, 2048] fp16: CLS ⊕ mean-patch (registers variant, so mean-patch
  is free of the high-norm artifact tokens of plain DINOv2 — PI review B3a)
- grid [T, 7, 9, 1024] fp16: average-pooled patch grid, cached so spatial /
  attention-pooling heads never require re-extraction (B3b)
- labels [T] (13 classes, idle=0), times [T] seconds, provenance fields (B3c)

Decodes sequentially with OpenCV at --fps; --size must be multiples of 14
(default 518x392 preserves the videos' 4:3). Shard across GPUs:
    CUDA_VISIBLE_DEVICES=0 python scripts/extract_phase_features.py --shard 0/2 &
    CUDA_VISIBLE_DEVICES=1 python scripts/extract_phase_features.py --shard 1/2 &
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phacosight.phase.timeline import case_paths, load_segments, phase_cases

MEAN = torch.tensor([0.485, 0.456, 0.406])
STD = torch.tensor([0.229, 0.224, 0.225])


@torch.no_grad()
def extract_case(model, case: str, phase_root: Path, fps: float, size: tuple[int, int],
                 batch: int, device) -> dict:
    video, ann = case_paths(phase_root, case)
    cap = cv2.VideoCapture(str(video))
    vfps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = n_frames / vfps
    stride = vfps / fps

    feats, grids, times, buf = [], [], [], []
    next_pick = 0.0
    idx = 0
    patch_hw = (size[1] // 14, size[0] // 14)  # (rows, cols)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx >= next_pick:
            next_pick += stride
            times.append(idx / vfps)
            rgb = cv2.cvtColor(cv2.resize(frame, size), cv2.COLOR_BGR2RGB)
            buf.append(torch.from_numpy(rgb))
            if len(buf) == batch:
                f, g = embed(model, buf, device, patch_hw)
                feats.append(f), grids.append(g)
                buf = []
        idx += 1
    if buf:
        f, g = embed(model, buf, device, patch_hw)
        feats.append(f), grids.append(g)
    cap.release()

    features = torch.cat(feats).numpy().astype(np.float16)
    grid = torch.cat(grids).numpy().astype(np.float16)
    # label each sampled time directly (robust to ragged stride)
    segs = load_segments(ann)
    t = np.array(times)
    labels = np.zeros(len(t), dtype=np.int64)
    for row in segs.itertuples():
        labels[(t >= row.start_sec) & (t < row.end_sec)] = row.phase_id
    assert len(labels) == len(features) == len(grid)
    return {"features": features, "grid": grid, "labels": labels,
            "times": t.astype(np.float32), "duration": duration, "video_fps": vfps}


@torch.no_grad()
def embed(model, buf: list[torch.Tensor], device, patch_hw: tuple[int, int]):
    x = torch.stack(buf).to(device).float().div_(255)
    x = (x.permute(0, 3, 1, 2) - MEAN.view(1, 3, 1, 1).to(device)) / STD.view(1, 3, 1, 1).to(device)
    with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        out = model(pixel_values=x)
    n_special = 1 + getattr(model.config, "num_register_tokens", 0)  # CLS + registers
    patches = out.last_hidden_state[:, n_special:].float()
    cls = out.pooler_output.float()  # [B, 1024]
    flat = torch.cat([cls, patches.mean(dim=1)], dim=-1).cpu()
    b, _, d = patches.shape
    grid = patches.reshape(b, *patch_hw, d).permute(0, 3, 1, 2)
    grid = torch.nn.functional.adaptive_avg_pool2d(grid, (7, 9)).permute(0, 2, 3, 1).cpu()
    return flat, grid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-root", default="data/cataract1k/phase")
    parser.add_argument("--out", default="data/features/phase_dinov2l")
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--size", default="518x392", help="WxH, multiples of 14")
    parser.add_argument("--model", default="facebook/dinov2-with-registers-large")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--shard", default="0/1", help="i/n: process cases i::n")
    args = parser.parse_args()

    from transformers import AutoModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(args.model).to(device).eval()
    w, h = map(int, args.size.split("x"))
    i, n = map(int, args.shard.split("/"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = phase_cases(args.phase_root)[i::n]
    for case in cases:
        out = out_dir / f"{case}.npz"
        if out.exists():
            print(f"{case}: exists, skipping")
            continue
        t0 = time.time()
        data = extract_case(model, case, Path(args.phase_root), args.fps, (w, h),
                            args.batch, device)
        try:
            sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True, cwd=Path(__file__).parents[1]).stdout.strip()
        except OSError:
            sha = "unknown"
        data |= {"backbone": args.model, "size": args.size, "fps": args.fps, "git_sha": sha}
        np.savez_compressed(out, **data)
        print(f"{case}: {data['features'].shape} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
