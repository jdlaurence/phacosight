#!/usr/bin/env python
"""Segmentation bake-off training: one (model, task, fold) run per invocation.

Usage:
    # single GPU / CPU
    python scripts/train_seg.py --config configs/seg_segformer_b2.yaml --fold 0
    # both A40s via DDP (preferred on the cluster)
    torchrun --standalone --nproc_per_node=2 scripts/train_seg.py --config ... --fold 0
    python scripts/train_seg.py --config ... --fold all   # sequential 5-fold

Writes per-fold checkpoints and a metrics JSON under runs/<run-name>/fold<k>/.
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, WeightedRandomSampler

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cataract_video.data.folds import load_fold
from cataract_video.data.seg_dataset import (
    CataractSegDataset,
    eval_transforms,
    instrument_oversample_weights,
    train_transforms,
)
from cataract_video.labels import TASKS, PUPIL_CLASS_NAME
from cataract_video.losses import CrossEntropyLogDice
from cataract_video.metrics import SegMetrics
from cataract_video.models.segmentation import build_model


def ddp_setup() -> tuple[int, int]:
    """Initialize process group under torchrun; no-op otherwise. Returns (rank, world)."""
    if "RANK" not in os.environ:
        return 0, 1
    if not dist.is_initialized():
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    local_rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return dist.get_rank(), dist.get_world_size()


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_fold(cfg: dict, fold: int, rank: int = 0, world: int = 1) -> dict:
    task = TASKS[cfg["task"]]
    device = pick_device()
    amp = device.type == "cuda"
    main = rank == 0

    split_dir = Path(cfg["fold_csv_dir"])
    prefix = cfg["fold_csv_prefix"]  # e.g. "Cat1k_anatomy_instrument"
    train_df = load_fold(
        split_dir / f"{prefix}_{fold}_train.csv",
        cfg["data_root"], mask_dir=task.mask_dir, require_exists=True,
    )
    test_df = load_fold(
        split_dir / f"{prefix}_{fold}_test.csv",
        cfg["data_root"], mask_dir=task.mask_dir, require_exists=True,
    )

    size = cfg.get("image_size", 512)
    train_ds = CataractSegDataset(train_df, train_transforms(size))
    test_ds = CataractSegDataset(test_df, eval_transforms(size))

    # Each DDP rank evaluates a disjoint slice of the test set; the confusion
    # matrices are all-reduced in evaluate(), so metrics stay exact (no
    # DistributedSampler padding duplicates).
    if world > 1:
        test_ds = CataractSegDataset(
            test_df.iloc[rank::world].reset_index(drop=True), eval_transforms(size)
        )

    sampler = None
    if cfg.get("oversample_rare_instruments", True):
        # With-replacement sampling is independent per rank: each rank draws its
        # own 1/world share of an epoch (seeded differently in main()).
        weights = instrument_oversample_weights(train_df)
        sampler = WeightedRandomSampler(weights, num_samples=len(train_ds) // world)
    elif world > 1:
        sampler = DistributedSampler(train_ds, shuffle=True)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.get("batch_size", 8), sampler=sampler,
        shuffle=sampler is None, num_workers=cfg.get("num_workers", 4),
        pin_memory=amp, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.get("batch_size", 8),
        num_workers=cfg.get("num_workers", 4), pin_memory=amp,
    )

    model = build_model(cfg["model"], task.num_classes).to(device)
    if world > 1:
        model = DistributedDataParallel(
            model, device_ids=[device.index] if device.type == "cuda" else None
        )
    criterion = CrossEntropyLogDice(task.num_classes, dice_weight=cfg.get("dice_weight", 0.8))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.get("lr", 6e-5), weight_decay=cfg.get("weight_decay", 0.01)
    )
    epochs = cfg.get("epochs", 40)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(
        optimizer, total_iters=epochs * len(train_loader), power=1.0
    )
    scaler = torch.amp.GradScaler(enabled=amp)

    out_dir = Path(cfg.get("out_dir", "runs")) / cfg["run_name"] / f"fold{fold}"
    out_dir.mkdir(parents=True, exist_ok=True)

    best = {"miou": -1.0}
    history = []  # full per-epoch record; winner decisions use last epochs, never "best" (test-selected)
    for epoch in range(epochs):
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)
        model.train()
        t0 = time.time()
        total_loss = 0.0
        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(pixel_values)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()

        metrics = evaluate(model, test_loader, task, device, amp)
        metrics["epoch"] = epoch
        metrics["train_loss"] = total_loss / max(1, len(train_loader))
        metrics["epoch_seconds"] = time.time() - t0
        if main:
            print(
                f"[fold {fold}] epoch {epoch}: loss {metrics['train_loss']:.4f} "
                f"mIoU {metrics['miou']:.4f} "
                f"pupil IoU {metrics['per_class_iou'].get(PUPIL_CLASS_NAME, float('nan')):.4f}"
            )
        if metrics["miou"] > best["miou"]:
            best = metrics
            if main:
                state = model.module if isinstance(model, DistributedDataParallel) else model
                torch.save(
                    {"model": state.state_dict(), "config": cfg, "fold": fold, "metrics": metrics},
                    out_dir / "best.pt",
                )
        history.append(metrics)
        if main and epoch == epochs - 1:
            # the decision/deployment artifact (PI N1): unlike best.pt, carries
            # no test-selection bias
            state = model.module if isinstance(model, DistributedDataParallel) else model
            torch.save(
                {"model": state.state_dict(), "config": cfg, "fold": fold, "metrics": metrics},
                out_dir / "last.pt",
            )
        if main:
            (out_dir / "metrics.json").write_text(
                json.dumps(
                    {"best": best, "last": metrics, "history": history,
                     "provenance": cfg.get("_provenance", {})},
                    indent=2,
                )
            )
    if world > 1:
        dist.barrier()
    return best


@torch.no_grad()
def evaluate(model, loader, task, device, amp: bool) -> dict:
    model.eval()
    metrics = SegMetrics(task.num_classes, task.class_names)
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(pixel_values)
        metrics.update(logits.argmax(dim=1), batch["labels"])
    if dist.is_initialized():
        # each rank saw a disjoint slice of the test set; sum the confusions
        confusion = torch.from_numpy(metrics.confusion).to(device)
        dist.all_reduce(confusion)
        metrics.confusion = confusion.cpu().numpy()
    return metrics.compute()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", default="0", help="fold index or 'all'")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    rank, world = ddp_setup()
    seed = cfg.get("seed", 0)
    torch.manual_seed(seed + rank)  # decorrelates per-rank sampling
    repo = Path(__file__).resolve().parents[1]
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=repo,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, cwd=repo,
        ).stdout.strip())
    except OSError:
        sha, dirty = "unknown", True
    cfg["_provenance"] = {"git_sha": sha, "git_dirty": dirty, "seed": seed, "world_size": world}

    folds = range(cfg.get("num_folds", 5)) if args.fold == "all" else [int(args.fold)]
    results = {}
    for fold in folds:
        results[fold] = run_fold(cfg, fold, rank=rank, world=world)
    if len(results) > 1 and rank == 0:
        mious = [r["miou"] for r in results.values()]
        print(f"\n5-fold mIoU: mean {sum(mious)/len(mious):.4f}, per-fold {mious}")
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
