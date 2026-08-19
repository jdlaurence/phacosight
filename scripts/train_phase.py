#!/usr/bin/env python
"""Train a temporal head on cached phase features. One (config, fold) per run.

    python scripts/train_phase.py --config configs/phase_mstcnpp.yaml --fold 0

Full-video sequences, batch of one video. Checkpoint selection on the VAL
split (never test); test is evaluated once at the end with the val-best and
last checkpoints. Writes runs/<run_name>/fold<k>/{metrics.json,val_best.pt,last.pt}.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phacosight.phase.heads import HEADS, mstcn_loss
from phacosight.phase.metrics import PhaseMetrics
from phacosight.phase.timeline import NUM_CLASSES, PHASES


def load_split(split_dir: Path, fold: int, name: str) -> list[str]:
    return pd.read_csv(split_dir / f"fold{fold}_{name}.csv")["case"].tolist()


def load_features(feat_dir: Path, cases: list[str], extra_dirs: list[Path],
                  stride: int = 1) -> dict:
    out = {}
    for c in cases:
        d = np.load(feat_dir / f"{c}.npz")
        feats = [d["features"].astype(np.float32)]
        for e in extra_dirs:
            x = np.load(e / f"{c}.npz")["features"].astype(np.float32)
            assert len(x) == len(feats[0]), f"{c}: extra feature length mismatch"
            feats.append(x)
        out[c] = (np.concatenate(feats, axis=1)[::stride], d["labels"].astype(np.int64)[::stride])
    return out


def class_weights(data: dict, cases: list[str]) -> torch.Tensor:
    """1/sqrt(count); ignore_index frames (-100, composite corpus) are excluded (E7 B3a)."""
    counts = np.zeros(NUM_CLASSES)
    for c in cases:
        y = data[c][1]
        counts += np.bincount(y[y >= 0], minlength=NUM_CLASSES)
    w = 1.0 / np.sqrt(np.maximum(counts, 1))
    return torch.tensor(w / w.mean(), dtype=torch.float32)


@torch.no_grad()
def evaluate(model, data: dict, cases: list[str], device, fps: float = 5.0) -> dict:
    model.eval()
    metrics = PhaseMetrics(NUM_CLASSES, PHASES, fps=fps)
    for c in cases:
        x, y = data[c]
        logits = model(torch.from_numpy(x).unsqueeze(0).to(device))[-1, 0]
        metrics.update(logits.argmax(-1).cpu().numpy(), y)
    return metrics.compute()


def run_fold(cfg: dict, fold: int) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_dir = Path(cfg["split_dir"])
    feat_dir = Path(cfg["features"])
    extra = [Path(p) for p in cfg.get("extra_features", [])]
    splits = {s: load_split(split_dir, fold, s) for s in ("train", "val", "test")}
    data = load_features(feat_dir, sum(splits.values(), []), extra,
                         stride=cfg.get("frame_stride", 1))
    eff_fps = cfg.get("features_fps", 5.0) / cfg.get("frame_stride", 1)

    # E7-b composite corpus: augmentation cases join every fold's TRAIN set only.
    # Val/test stay pure C1K; the eval grammar (eval_phase) is built from the C1K
    # split CSVs and never sees these cases (pre-registered B3b).
    aug_cases: list[str] = []
    weight_log = {}
    if cfg.get("aug_features"):
        from phacosight.data.cataracts import cataracts_cases
        keep = set(cfg.get("aug_splits", ["train", "dev"]))
        aug_cases = [stem for stem, _, _, s in
                     cataracts_cases(cfg.get("aug_root", "data/cataracts")) if s in keep]
        data |= load_features(Path(cfg["aug_features"]), aug_cases,
                              [Path(p) for p in cfg.get("aug_extra_features", [])],
                              stride=cfg.get("frame_stride", 1))
    train_cases = splits["train"] + aug_cases

    in_dim = next(iter(data.values()))[0].shape[1]
    model = HEADS[cfg["head"]](in_dim, NUM_CLASSES, **cfg.get("head_kwargs", {})).to(device)
    weights = class_weights(data, train_cases).to(device) if cfg.get("class_weighted", True) else None
    if aug_cases and weights is not None:
        # pre-registered B3a logging: make the loss-weighting confound visible
        weight_log = {"c1k_only": class_weights(data, splits["train"]).tolist(),
                      "composite": weights.cpu().tolist()}
        print(f"[fold {fold}] class weights c1k->composite: " + ", ".join(
            f"{PHASES[i]}: {a:.2f}->{b:.2f}" for i, (a, b) in
            enumerate(zip(weight_log["c1k_only"], weight_log["composite"]))))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 5e-4),
                            weight_decay=cfg.get("weight_decay", 1e-4))
    epochs = cfg.get("epochs", 60)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    out_dir = Path(cfg.get("out_dir", "runs")) / cfg["run_name"] / f"fold{fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.get("seed", 0) + fold)

    select_on = cfg.get("select_on", "macro_f1")
    best = {"val": {select_on: -1.0}}
    history = []
    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        order = rng.permutation(train_cases)
        total = 0.0
        for c in order:
            x, y = data[c]
            xt = torch.from_numpy(x).unsqueeze(0).to(device)
            yt = torch.from_numpy(y).unsqueeze(0).to(device)
            opt.zero_grad(set_to_none=True)
            loss = mstcn_loss(model(xt), yt, class_weights=weights,
                              tmse_weight=cfg.get("tmse_weight", 0.15))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.item()
        sched.step()

        val = evaluate(model, data, splits["val"], device, fps=eff_fps)
        row = {"epoch": epoch, "train_loss": total / len(order),
               "val": val, "seconds": time.time() - t0}
        history.append(row)
        if val[select_on] > best["val"][select_on]:
            best = row
            torch.save({"model": model.state_dict(), "config": cfg, "fold": fold,
                        "epoch": epoch}, out_dir / "val_best.pt")
        print(f"[fold {fold}] epoch {epoch}: loss {row['train_loss']:.3f} "
              f"val f1 {val['macro_f1']:.4f} edit {val['edit']:.1f}")

    torch.save({"model": model.state_dict(), "config": cfg, "fold": fold,
                "epoch": epochs - 1}, out_dir / "last.pt")
    # test: once, with the val-selected model (and last for reference)
    test_last = evaluate(model, data, splits["test"], device, fps=eff_fps)
    model.load_state_dict(torch.load(out_dir / "val_best.pt", weights_only=False)["model"])
    test_val_best = evaluate(model, data, splits["test"], device, fps=eff_fps)

    result = {"fold": fold, "val_best_epoch": best["epoch"], "val_best": best["val"],
              "test_val_best": test_val_best, "test_last": test_last,
              "history": history, "provenance": cfg.get("_provenance", {}),
              "aug_cases": aug_cases, "class_weight_log": weight_log}
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    print(f"[fold {fold}] TEST (val-selected): acc {test_val_best['accuracy']:.4f} "
          f"macroF1 {test_val_best['macro_f1']:.4f} edit {test_val_best['edit']:.1f} "
          f"f1@50 {test_val_best['f1@50']:.1f}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", default="0", help="fold index or 'all'")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    repo = Path(__file__).resolve().parents[1]
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, cwd=repo).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                                    text=True, cwd=repo).stdout.strip())
    except OSError:
        sha, dirty = "unknown", True
    cfg["_provenance"] = {"git_sha": sha, "git_dirty": dirty, "seed": cfg.get("seed", 0)}
    torch.manual_seed(cfg.get("seed", 0))

    folds = range(cfg.get("num_folds", 4)) if args.fold == "all" else [int(args.fold)]
    results = [run_fold(cfg, f) for f in folds]
    if len(results) > 1:
        for key in ("macro_f1", "edit", "f1@50", "accuracy"):
            vals = [r["test_val_best"][key] for r in results]
            print(f"{key}: mean {np.mean(vals):.4f} ± {np.std(vals):.4f}")


if __name__ == "__main__":
    main()
