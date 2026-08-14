#!/usr/bin/env python
"""Bulk phase-timeline inference over the unlabeled Cataract-1K videos.

Deployment stack per the Stage 2 round-2 PI review (C5): tool-fusion MS-TCN++
x (4 folds x 3 seeds) ensemble -> temperature -> Viterbi with a grammar
re-estimated from the 56 labeled videos at the inference rate (1 fps).
Legitimate on unlabeled videos: no evaluation happens here, so all folds may
ensemble. Segmentation features use the fixed fold-0 checkpoints (unlabeled
videos are in no seg fold).

Writes data/library/phase_timelines/<case>.json:
  segments: [{phase, start_s, end_s, confidence, disagreement}]
  plus per-video meta and provenance. Skips the 56 phase-annotated cases
  (ground truth exists) and already-processed videos (idempotent).

    CUDA_VISIBLE_DEVICES=0 python scripts/analyze_phase_bulk.py --shard 0/2 &
    CUDA_VISIBLE_DEVICES=1 python scripts/analyze_phase_bulk.py --shard 1/2 &
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cataract_video.labels import ANATOMY_INSTRUMENT, INSTRUMENT_MULTICLASS
from cataract_video.models.segmentation import build_model
from cataract_video.phase.decoding import transition_matrix, viterbi
from cataract_video.phase.heads import HEADS
from cataract_video.phase.metrics import segments as label_segments
from cataract_video.phase.timeline import NUM_CLASSES, PHASES, phase_cases

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
TOOLS_RUNS = ["runs/phase_mstcnpp_tools", "runs/phase_mstcnpp_tools_seed1",
              "runs/phase_mstcnpp_tools_seed2"]
TEMPERATURE = 1.04  # mean of val-fit fold temperatures (confidence.json)


def load_stack(device):
    seg = {}
    for name, (run, fname, task) in {
        "anatomy": ("runs/segformer_b2_anatomy_instrument", "best.pt", ANATOMY_INSTRUMENT),
        "multiclass": ("runs/segformer_b2_instrument_multiclass", "last.pt", INSTRUMENT_MULTICLASS),
    }.items():
        ckpt = torch.load(Path(run) / "fold0" / fname, map_location=device, weights_only=False)
        m = build_model("segformer_b2", task.num_classes).to(device).eval()
        m.load_state_dict(ckpt["model"])
        seg[name] = m
    from transformers import AutoModel
    dino = AutoModel.from_pretrained("facebook/dinov2-with-registers-large").to(device).eval()

    heads = []
    for run in TOOLS_RUNS:
        for fold in range(4):
            ckpt = torch.load(Path(run) / f"fold{fold}" / "val_best.pt", weights_only=False)
            h = HEADS["mstcnpp"](2084, NUM_CLASSES).to(device).eval()
            h.load_state_dict(ckpt["model"])
            heads.append(h)
    return dino, seg, heads


@torch.no_grad()
def video_features(video: Path, dino, seg, device, fps_out: float, batch: int = 24):
    cap = cv2.VideoCapture(str(video))
    vfps = cap.get(cv2.CAP_PROP_FPS)
    stride = max(1.0, vfps / fps_out)
    mean, std = MEAN.to(device), STD.to(device)
    feats, times, buf_d, buf_s = [], [], [], []
    next_pick, idx = 0.0, 0

    def flush():
        xd = (torch.stack(buf_d).to(device).float().div_(255).permute(0, 3, 1, 2) - mean) / std
        xs = (torch.stack(buf_s).to(device).float().div_(255).permute(0, 3, 1, 2) - mean) / std
        with torch.autocast("cuda", dtype=torch.float16):
            out = dino(pixel_values=xd)
            n_special = 1 + getattr(dino.config, "num_register_tokens", 0)
            cls = out.pooler_output.float()
            pm = out.last_hidden_state[:, n_special:].float().mean(dim=1)
            tf = []
            for name in ("anatomy", "multiclass"):
                p = torch.softmax(seg[name](xs), dim=1)
                amax = p.argmax(dim=1)
                frac = torch.stack([(amax == c).float().mean(dim=(1, 2))
                                    for c in range(p.shape[1])], dim=1)
                tf += [p.amax(dim=(2, 3)).float(), p.mean(dim=(2, 3)).float(), frac.float()]
        feats.append(torch.cat([cls, pm, *tf], dim=1).cpu())

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx >= next_pick:
            next_pick += stride
            times.append(idx / vfps)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            buf_d.append(torch.from_numpy(cv2.resize(rgb, (518, 392))))
            buf_s.append(torch.from_numpy(cv2.resize(rgb, (512, 512))))
            if len(buf_d) == batch:
                flush(); buf_d, buf_s = [], []
        idx += 1
    if buf_d:
        flush()
    cap.release()
    duration = idx / vfps if vfps else 0.0
    return torch.cat(feats), np.array(times), duration, vfps


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", default="data/full")
    parser.add_argument("--out", default="data/library/phase_timelines")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--shard", default="0/1")
    args = parser.parse_args()

    device = torch.device("cuda")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                         cwd=Path(__file__).parents[1]).stdout.strip()

    labeled = set(phase_cases("data/phase"))
    videos = sorted(p for p in Path(args.videos).glob("case_*.mp4")
                    if p.stem not in labeled)
    i, n = map(int, args.shard.split("/"))
    videos = videos[i::n]
    print(f"shard {args.shard}: {len(videos)} videos")

    # grammar at the inference rate, from all labeled videos
    stride5 = round(5.0 / args.fps)
    train_labels = [np.load(f"data/features/phase_dinov2l/{c}.npz")["labels"][::stride5]
                    for c in sorted(labeled)]
    log_trans = transition_matrix(train_labels, NUM_CLASSES)

    dino, seg, heads = load_stack(device)
    for video in videos:
        out = out_dir / f"{video.stem}.json"
        if out.exists():
            continue
        t0 = time.time()
        try:
            x, times, duration, vfps = video_features(video, dino, seg, device, args.fps)
        except Exception as e:  # corrupt video should not kill the pass
            print(f"{video.stem}: FAILED decode ({type(e).__name__}: {e})")
            continue
        xt = x.unsqueeze(0).to(device)
        member_probs = torch.stack([torch.softmax(h(xt)[-1, 0], -1) for h in heads])
        probs = member_probs.mean(0)
        probs = torch.softmax(torch.log(probs + 1e-12) / TEMPERATURE, -1).cpu().numpy()
        member_pred = member_probs.argmax(-1).cpu().numpy()
        disagree = (member_pred != member_pred[0]).any(0).astype(float)
        pred = viterbi(np.log(probs + 1e-12), log_trans)
        p_pred = probs[np.arange(len(pred)), pred]

        segs = []
        for cls, s, e in label_segments(pred):
            segs.append({
                "phase": PHASES[cls],
                "start_s": round(float(times[s]), 2),
                "end_s": round(float(times[e - 1] + 1.0 / args.fps), 2),
                "confidence": round(float(p_pred[s:e].mean()), 4),
                "disagreement": round(float(disagree[s:e].mean()), 4),
            })
        out.write_text(json.dumps({
            "case": video.stem, "duration_s": round(duration, 2), "video_fps": vfps,
            "inference_fps": args.fps, "segments": segs,
            "provenance": {"git_sha": sha, "stack": "tools-fusion x 12-member ensemble, "
                           f"T={TEMPERATURE}, viterbi@{args.fps}fps, seg fold0"},
        }, indent=1))
        print(f"{video.stem}: {len(segs)} segments, {duration:.0f}s video "
              f"in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
