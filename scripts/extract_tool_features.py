#!/usr/bin/env python
"""Cache per-frame tool/anatomy presence features from the Stage-1 segmentation
models for the 56 phase videos (Stage 2 E3).

Checkpoint per video comes from TrainIDs_PhaseRecognition/seg_checkpoint_map.csv
(PI review B1/B2): overlap cases use the seg fold that held them out; clean
cases a pinned round-robin fold. Anatomy = best.pt (those runs predate last.pt;
best-last gap <=0.005, caveat documented), multiclass = last.pt.

Per frame, per model, per class: [max prob, mean prob, argmax pixel fraction]
-> (5 + 7) x 3 = 36 dims. Frames are sampled at exactly the times stored in the
DINO cache so the two feature sets align index-for-index.

    CUDA_VISIBLE_DEVICES=0 python scripts/extract_tool_features.py --shard 0/2 &
    CUDA_VISIBLE_DEVICES=1 python scripts/extract_tool_features.py --shard 1/2 &
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phacosight.labels import ANATOMY_INSTRUMENT, INSTRUMENT_MULTICLASS
from phacosight.models.segmentation import build_model
from phacosight.phase.timeline import case_paths

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

CKPTS = {
    "anatomy": ("runs/segformer_b2_anatomy_instrument", "best.pt", ANATOMY_INSTRUMENT),
    "multiclass": ("runs/segformer_b2_instrument_multiclass", "last.pt", INSTRUMENT_MULTICLASS),
}


def load_models(seg_fold: int, device) -> dict:
    models = {}
    for name, (run, fname, task) in CKPTS.items():
        ckpt = torch.load(Path(run) / f"fold{seg_fold}" / fname,
                          map_location=device, weights_only=False)
        m = build_model("segformer_b2", task.num_classes).to(device).eval()
        m.load_state_dict(ckpt["model"])
        models[name] = m
    return models


@torch.no_grad()
def probs_features(models: dict, batch: torch.Tensor) -> np.ndarray:
    """[B, 36]: per model per class (max prob, mean prob, argmax fraction)."""
    feats = []
    for name in ("anatomy", "multiclass"):
        with torch.autocast("cuda", dtype=torch.float16):
            p = torch.softmax(models[name](batch), dim=1)  # [B, C, H, W]
        amax = p.argmax(dim=1)
        C = p.shape[1]
        frac = torch.stack([(amax == c).float().mean(dim=(1, 2)) for c in range(C)], dim=1)
        feats += [p.amax(dim=(2, 3)), p.mean(dim=(2, 3)), frac]
    return torch.cat(feats, dim=1).float().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cataract1k", "cataracts"], default="cataract1k")
    parser.add_argument("--phase-root", default="data/cataract1k/phase")
    parser.add_argument("--cataracts-root", default="data/cataracts")
    parser.add_argument("--dino-cache", default="data/features/phase_dinov2l")
    parser.add_argument("--out", default="data/features/phase_tools")
    parser.add_argument("--map", default="TrainIDs_PhaseRecognition/seg_checkpoint_map.csv")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--shard", default="0/1")
    args = parser.parse_args()

    device = torch.device("cuda")
    i, n = map(int, args.shard.split("/"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                         cwd=Path(__file__).parents[1]).stdout.strip()

    from types import SimpleNamespace

    if args.dataset == "cataracts":
        # E7-prep: fold-0 zero-shot for every video (deployment convention); out-of-domain.
        from phacosight.data.cataracts import cataracts_cases
        rows = [SimpleNamespace(case=stem, seg_fold=0, overlap=False, video=video)
                for stem, video, _, _ in cataracts_cases(args.cataracts_root)]
    else:
        ckpt_map = pd.read_csv(args.map)
        rows = [SimpleNamespace(case=r.case, seg_fold=r.seg_fold, overlap=r.overlap,
                                video=case_paths(args.phase_root, r.case)[0])
                for r in ckpt_map.itertuples()]

    # group by fold so each checkpoint pair loads once
    rows = rows[i::n]
    by_fold: dict[int, list] = {}
    for r in rows:
        by_fold.setdefault(int(r.seg_fold), []).append(r)

    mean, std = MEAN.to(device), STD.to(device)
    for fold, fold_rows in sorted(by_fold.items()):
        models = load_models(fold, device)
        for r in fold_rows:
            out = out_dir / f"{r.case}.npz"
            if out.exists():
                print(f"{r.case}: exists, skipping")
                continue
            t0 = time.time()
            times = np.load(Path(args.dino_cache) / f"{r.case}.npz")["times"]
            cap = cv2.VideoCapture(str(r.video))
            vfps = cap.get(cv2.CAP_PROP_FPS)
            want = np.round(times * vfps).astype(int)
            feats, buf = [], []
            idx = 0
            wi = 0
            while wi < len(want):
                ok, frame = cap.read()
                if not ok:
                    break
                if idx == want[wi]:
                    if args.dataset == "cataracts":  # pre-registered 4:3 center-crop
                        fh, fw = frame.shape[:2]
                        cw = (fh * 4) // 3
                        x0 = (fw - cw) // 2
                        frame = frame[:, x0:x0 + cw]
                    rgb = cv2.cvtColor(cv2.resize(frame, (args.size, args.size)),
                                       cv2.COLOR_BGR2RGB)
                    buf.append(torch.from_numpy(rgb))
                    wi += 1
                    if len(buf) == args.batch:
                        x = (torch.stack(buf).to(device).float().div_(255)
                             .permute(0, 3, 1, 2) - mean) / std
                        feats.append(probs_features(models, x))
                        buf = []
                idx += 1
            if buf:
                x = (torch.stack(buf).to(device).float().div_(255)
                     .permute(0, 3, 1, 2) - mean) / std
                feats.append(probs_features(models, x))
            cap.release()
            features = np.concatenate(feats).astype(np.float16)
            assert len(features) == len(times), (r.case, len(features), len(times))
            np.savez_compressed(out, features=features, times=times,
                                seg_fold=fold, overlap=bool(r.overlap), git_sha=sha)
            print(f"{r.case}: {features.shape} (fold {fold}) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
