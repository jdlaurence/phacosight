#!/usr/bin/env python
"""FP16 inference throughput for the bake-off candidates on the deployment
condition: batch 1, 512x512, single stream, one A40 (PI review B2 — the
'>=30 fps' half of the winner criterion must be measured, not assumed).

    .venv/bin/python scripts/bench_seg.py --models segformer_b2 efficientvit_b1 pidnet_s
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phacosight.models.segmentation import build_model  # noqa: E402


@torch.no_grad()
def bench(name: str, size: int, batch: int, iters: int, warmup: int) -> dict:
    device = torch.device("cuda")
    model = build_model(name, 5).to(device).eval()
    x = torch.randn(batch, 3, size, size, device=device)
    with torch.autocast("cuda", dtype=torch.float16):
        for _ in range(warmup):
            model(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    return {
        "model": name, "image_size": size, "batch": batch,
        "fps": batch * iters / dt, "ms_per_frame": 1000 * dt / (batch * iters),
        "gpu": torch.cuda.get_device_name(0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        default=["segformer_b2", "efficientvit_b1", "pidnet_s"])
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--out", default=None, help="optional JSON output path")
    args = parser.parse_args()

    results = [bench(m, args.size, args.batch, args.iters, args.warmup) for m in args.models]
    for r in results:
        print(f"{r['model']:>18}: {r['fps']:7.1f} fps  ({r['ms_per_frame']:.2f} ms/frame, "
              f"batch {r['batch']}, {r['image_size']}px, FP16 autocast)")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
