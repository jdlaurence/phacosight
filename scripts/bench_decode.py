#!/usr/bin/env python
"""Benchmark sequential video decode: torchcodec CPU vs CUDA/NVDEC.

Establishes the decode-layer numbers assumed in the plan (~1,000+ fps NVDEC at
1024x768 → ~10-15 s per 7-min video). Run on the A40 machine; on a machine
without CUDA it benchmarks the CPU path only.

    python scripts/bench_decode.py path/to/video.mp4 [--stride 6]

--stride simulates task routing (e.g. keep every 6th frame for 5 fps tasks):
frames are still decoded sequentially; only retained frames are materialized.
"""

import argparse
import time


def bench(video: str, device: str, stride: int, max_frames: int | None) -> None:
    from torchcodec.decoders import VideoDecoder

    decoder = VideoDecoder(video, device=device)
    meta = decoder.metadata
    n = len(decoder)
    if max_frames:
        n = min(n, max_frames)
    t0 = time.perf_counter()
    kept = 0
    for i in range(n):
        frame = decoder[i] if i % stride == 0 else None  # noqa: F841 — decode-and-drop
        if i % stride == 0:
            kept += 1
    dt = time.perf_counter() - t0
    print(
        f"[{device}] {n} frames ({meta.width}x{meta.height} @ {meta.average_fps:.1f} fps), "
        f"stride {stride} → kept {kept} | {dt:.1f} s = {n / dt:.0f} fps decode"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    try:
        import torch
        import torchcodec  # noqa: F401
    except ImportError:
        raise SystemExit("pip install torchcodec  (plus torch)")

    bench(args.video, "cpu", args.stride, args.max_frames)
    if torch.cuda.is_available():
        bench(args.video, "cuda", args.stride, args.max_frames)
    else:
        print("[cuda] skipped — no CUDA device (run this on the A40 box)")


if __name__ == "__main__":
    main()
