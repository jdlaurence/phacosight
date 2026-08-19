#!/usr/bin/env python
"""Evaluate a phase run's val-selected checkpoints with decoding variants.

For each fold: argmax, mode-smoothing, and Viterbi with a grammar learned from
that fold's TRAIN videos only. Reports full metrics per decoding, the
margin-idle vs gap-idle diagnostic, and the pre-registered falsification probe
(the 10 most-fragmented ground-truth timelines must improve, not lose repeats).

    python scripts/eval_phase.py --run runs/phase_mstcnpp_dinov2l --out results.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phacosight.phase.decoding import mode_smooth, transition_matrix, viterbi
from phacosight.phase.heads import HEADS
from phacosight.phase.metrics import (PhaseMetrics, edit_score, segments,
                                      video_f1_at_50, video_macro_f1)
from phacosight.phase.timeline import NUM_CLASSES, PHASES


def load_case(feat_dir: Path, case: str, extra_dirs: list[Path] = (), stride: int = 1):
    d = np.load(feat_dir / f"{case}.npz")
    feats = [d["features"].astype(np.float32)]
    for e in extra_dirs:
        feats.append(np.load(Path(e) / f"{case}.npz")["features"].astype(np.float32))
    # frame_stride matters: 1 fps heads fed 5 fps sequences silently collapse
    # (MS-TCN++ receptive fields are in samples — the 2026-08-14 incident class)
    return np.concatenate(feats, axis=1)[::stride], d["labels"].astype(np.int64)[::stride]


@torch.no_grad()
def fold_logits(run_dir: Path, fold: int, feat_dir: Path, split_dir: Path, device):
    ckpt = torch.load(run_dir / f"fold{fold}" / "val_best.pt",
                      map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    extra_dirs = [Path(p) for p in cfg.get("extra_features", [])]
    stride = int(cfg.get("frame_stride", 1))
    eff_fps = cfg.get("features_fps", 5.0) / stride
    cases = {s: pd.read_csv(split_dir / f"fold{fold}_{s}.csv")["case"].tolist()
             for s in ("train", "test")}
    sample_x, _ = load_case(feat_dir, cases["test"][0], extra_dirs, stride)
    model = HEADS[cfg["head"]](sample_x.shape[1], NUM_CLASSES,
                               **cfg.get("head_kwargs", {})).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    out = {}
    for c in cases["test"]:
        x, y = load_case(feat_dir, c, extra_dirs, stride)
        logits = model(torch.from_numpy(x).unsqueeze(0).to(device))[-1, 0]
        out[c] = (torch.log_softmax(logits, -1).cpu().numpy(), y)
    train_labels = [load_case(feat_dir, c, stride=stride)[1] for c in cases["train"]]
    return out, train_labels, eff_fps


def idle_split_accuracy(pred: np.ndarray, true: np.ndarray) -> dict:
    """Idle accuracy split into margin (pre/post surgery) vs mid-video gaps."""
    action = np.flatnonzero(true != 0)
    if len(action) == 0:
        return {}
    margin = np.zeros(len(true), dtype=bool)
    margin[: action[0]] = True
    margin[action[-1] + 1:] = True
    out = {}
    for name, sel in (("margin", margin & (true == 0)), ("gap", ~margin & (true == 0))):
        if sel.any():
            out[name] = (float((pred[sel] == 0).mean()), int(sel.sum()))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--features", default="data/features/phase_dinov2l")
    parser.add_argument("--split-dir", default="TrainIDs_PhaseRecognition")
    parser.add_argument("--num-folds", type=int, default=4)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--mode-window", type=int, default=9)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir, feat_dir, split_dir = Path(args.run), Path(args.features), Path(args.split_dir)

    decodings = ("argmax", "mode", "viterbi")
    agg = None
    idle_acc = {d: {"margin": [], "gap": []} for d in decodings}
    per_video = []

    for fold in range(args.num_folds):
        logits, train_labels, eff_fps = fold_logits(run_dir, fold, feat_dir, split_dir, device)
        if agg is None:  # metric fps comes from the checkpoints, not a CLI guess
            agg = {d: PhaseMetrics(NUM_CLASSES, PHASES, fps=eff_fps) for d in decodings}
        log_trans = transition_matrix(train_labels, NUM_CLASSES)
        for case, (lp, y) in logits.items():
            preds = {
                "argmax": lp.argmax(-1),
                "mode": mode_smooth(lp.argmax(-1), args.mode_window),
                "viterbi": viterbi(lp, log_trans),
            }
            row = {"case": case, "fold": fold, "true_segments": len(segments(y))}
            for d, p in preds.items():
                agg[d].update(p, y)
                row[f"edit_{d}"] = edit_score(p, y)
                row[f"segs_{d}"] = len(segments(p))
                row[f"macro_f1_{d}"] = video_macro_f1(p, y)
                row[f"f1_50_{d}"] = video_f1_at_50(p, y)
                for k, (acc, n) in idle_split_accuracy(p, y).items():
                    idle_acc[d][k].append((acc, n))
            per_video.append(row)

    result = {"run": str(run_dir), "decodings": {}}
    for d in decodings:
        m = agg[d].compute()
        m["idle_margin_acc"] = float(np.average(
            [a for a, _ in idle_acc[d]["margin"]], weights=[n for _, n in idle_acc[d]["margin"]]))
        m["idle_gap_acc"] = float(np.average(
            [a for a, _ in idle_acc[d]["gap"]], weights=[n for _, n in idle_acc[d]["gap"]]))
        result["decodings"][d] = m
        print(f"{d:>8}: acc {m['accuracy']:.4f} macroF1 {m['macro_f1']:.4f} "
              f"edit {m['edit']:.1f} f1@50 {m['f1@50']:.1f} segratio {m['seg_ratio']:.2f} "
              f"bnd {m['boundary_median_s']:.2f}s idle(margin/gap) "
              f"{m['idle_margin_acc']:.3f}/{m['idle_gap_acc']:.3f}")

    # falsification probe: 10 most-fragmented GT timelines
    frag = sorted(per_video, key=lambda r: -r["true_segments"])[:10]
    probe = {
        "cases": [r["case"] for r in frag],
        "edit_argmax": float(np.mean([r["edit_argmax"] for r in frag])),
        "edit_viterbi": float(np.mean([r["edit_viterbi"] for r in frag])),
        "true_seg_mean": float(np.mean([r["true_segments"] for r in frag])),
        "viterbi_seg_mean": float(np.mean([r["segs_viterbi"] for r in frag])),
    }
    result["fragmented10_probe"] = probe
    result["per_video"] = per_video
    print(f"fragmented-10 probe: edit argmax {probe['edit_argmax']:.1f} -> "
          f"viterbi {probe['edit_viterbi']:.1f}; segments true {probe['true_seg_mean']:.1f} "
          f"vs viterbi {probe['viterbi_seg_mean']:.1f}")

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
