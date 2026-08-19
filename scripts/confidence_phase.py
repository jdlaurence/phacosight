#!/usr/bin/env python
"""Calibrated confidence for phase timelines: per-fold seed ensemble +
temperature scaling (fit on val) + per-segment confidence and risk-coverage.

Honest protocol: a test video is only ever scored by models from the fold that
held it out (the seed ensemble is within-fold, never across folds — cross-fold
models trained on the video). Deployment on NEW videos can use all folds.

    python scripts/confidence_phase.py --runs runs/phase_mstcnpp_dinov2l \
        runs/phase_mstcnpp_dinov2l_seed1 runs/phase_mstcnpp_dinov2l_seed2
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phacosight.phase.decoding import transition_matrix, viterbi
from phacosight.phase.heads import HEADS
from phacosight.phase.metrics import PhaseMetrics, segments
from phacosight.phase.timeline import NUM_CLASSES, PHASES


def load_case(feat_dir: Path, case: str, extra_dirs: list[Path] = (), stride: int = 1):
    d = np.load(feat_dir / f"{case}.npz")
    feats = [d["features"].astype(np.float32)]
    for e in extra_dirs:
        feats.append(np.load(Path(e) / f"{case}.npz")["features"].astype(np.float32))
    # stride from the checkpoints: 1 fps heads fed 5 fps sequences silently collapse
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


@torch.no_grad()
def seed_logits(run_dirs, fold, cases, feat_dir, device, stride=1):
    """{case: [n_seeds, T, C] logits}. Extra-feature spec comes from each checkpoint."""
    out = {c: [] for c in cases}
    for rd in run_dirs:
        ckpt = torch.load(Path(rd) / f"fold{fold}" / "val_best.pt",
                          map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        extra = [Path(p) for p in cfg.get("extra_features", [])]
        x0, _ = load_case(feat_dir, cases[0], extra, stride)
        model = HEADS[cfg["head"]](x0.shape[1], NUM_CLASSES,
                                   **cfg.get("head_kwargs", {})).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        for c in cases:
            x, _ = load_case(feat_dir, c, extra, stride)
            out[c].append(model(torch.from_numpy(x).unsqueeze(0).to(device))[-1, 0].cpu())
    return {c: torch.stack(v) for c, v in out.items()}


def fit_temperature(logits_list, labels_list, device) -> float:
    """Scalar T minimizing NLL of the ensemble-mean probabilities on val."""
    log_t = torch.zeros(1, requires_grad=True, device=device)
    probs = torch.cat([torch.softmax(l, -1).mean(0) for l in logits_list]).to(device)
    y = torch.from_numpy(np.concatenate(labels_list)).to(device)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=50)

    def closure():
        opt.zero_grad()
        # temperature on the ensemble log-probs
        scaled = torch.log_softmax(torch.log(probs + 1e-12) / log_t.exp(), -1)
        loss = torch.nn.functional.nll_loss(scaled, y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp())


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            out += sel.mean() * abs(correct[sel].mean() - conf[sel].mean())
    return float(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--features", default="data/features/phase_dinov2l")
    parser.add_argument("--split-dir", default="TrainIDs_PhaseRecognition")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_dir, split_dir = Path(args.features), Path(args.split_dir)

    stride, eff_fps = run_stride(args.runs)
    print(f"runs at frame_stride {stride} -> {eff_fps} fps")
    frame_conf, frame_correct, frame_conf_raw = [], [], []
    frame_disagree, frame_err = [], []
    seg_rows = []
    agg = PhaseMetrics(NUM_CLASSES, PHASES, fps=eff_fps)

    for fold in range(4):
        splits = {s: pd.read_csv(split_dir / f"fold{fold}_{s}.csv")["case"].tolist()
                  for s in ("train", "val", "test")}
        val_logits = seed_logits(args.runs, fold, splits["val"], feat_dir, device, stride)
        val_labels = [load_case(feat_dir, c, stride=stride)[1] for c in splits["val"]]
        T = fit_temperature(list(val_logits.values()), val_labels, device)

        train_labels = [load_case(feat_dir, c, stride=stride)[1] for c in splits["train"]]
        log_trans = transition_matrix(train_labels, NUM_CLASSES)

        test_logits = seed_logits(args.runs, fold, splits["test"], feat_dir, device, stride)
        for c in splits["test"]:
            sl = test_logits[c]  # [S, T, C]
            probs_raw = torch.softmax(sl, -1).mean(0).numpy()
            probs = torch.softmax(torch.log(torch.from_numpy(probs_raw) + 1e-12) / T, -1).numpy()
            y = load_case(feat_dir, c, stride=stride)[1]
            pred = viterbi(np.log(probs + 1e-12), log_trans)
            agg.update(pred, y)
            p_pred = probs[np.arange(len(pred)), pred]
            frame_conf.append(p_pred)
            frame_conf_raw.append(probs_raw[np.arange(len(pred)), pred])
            frame_correct.append(pred == y)
            seed_preds = sl.argmax(-1).numpy()  # [S, T]
            dis = (seed_preds != seed_preds[0]).any(0).astype(float)
            frame_disagree.append(dis)
            frame_err.append(pred != y)
            for cls, s, e in segments(pred):
                match = ((y[s:e] == cls).mean())
                seg_rows.append({"case": c, "fold": fold, "cls": int(cls),
                                 "len_s": (e - s) / eff_fps,
                                 "conf": float(p_pred[s:e].mean()),
                                 "disagree": float(dis[s:e].mean()),
                                 "purity": float(match)})
        print(f"fold {fold}: T={T:.3f}")

    conf = np.concatenate(frame_conf)
    conf_raw = np.concatenate(frame_conf_raw)
    correct = np.concatenate(frame_correct).astype(float)
    dis = np.concatenate(frame_disagree)
    err = np.concatenate(frame_err).astype(float)

    m = agg.compute()
    print(f"\nensemble+viterbi: acc {m['accuracy']:.4f} macroF1 {m['macro_f1']:.4f} "
          f"edit {m['edit']:.1f} f1@50 {m['f1@50']:.1f}")
    print(f"frame ECE raw {ece(conf_raw, correct):.4f} -> temp-scaled {ece(conf, correct):.4f}")
    # disagreement as an error signal
    from numpy import trapezoid
    order = np.argsort(-dis)
    frac_err_in_top10 = err[order[: len(order) // 10]].mean() / max(err.mean(), 1e-9)
    print(f"seed disagreement: top-10%-disagreement frames carry "
          f"{frac_err_in_top10:.1f}x the average error rate")

    # segment-level risk-coverage: keep segments above confidence percentile
    df = pd.DataFrame(seg_rows)
    df["ok"] = df["purity"] >= 0.5
    print("\nsegment risk-coverage (drop lowest-confidence segments):")
    for keep in (1.0, 0.95, 0.9, 0.8):
        thr = df["conf"].quantile(1 - keep)
        kept = df[df["conf"] >= thr]
        print(f"  keep {keep:4.0%}: segment purity>=0.5 rate {kept['ok'].mean():.4f} "
              f"({len(kept)} segments)")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"ensemble_metrics": m, "ece_raw": ece(conf_raw, correct),
             "ece_temp": ece(conf, correct), "segments": seg_rows}, indent=2))


if __name__ == "__main__":
    main()
