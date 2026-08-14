#!/usr/bin/env python
"""Collect bake-off results per the PI-approved decision rule.

Per PI review of 2026-08-14:
- B1: decisions use LAST-epoch metrics per fold (plus mean-of-last-5 when the
  per-epoch history is available), never the test-selected "best" block, which
  is reported only as an optimistic upper bound.
- C1: mIoU reported both with and without the background class.
- C3: the decision metrics are instrument IoU and pupil IoU (they gate the
  kinematics and miosis features); mIoU is the tiebreaker. Winner margins are
  judged against per-fold paired differences, not a comparison of two means.

    .venv/bin/python scripts/collect_seg_results.py --runs runs \
        --models segformer_b2_anatomy_instrument efficientvit_b1_anatomy_instrument pidnet_s_anatomy_instrument
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load_run(run_dir: Path) -> dict:
    folds = {}
    for fold_dir in sorted(run_dir.glob("fold*")):
        m = json.loads((fold_dir / "metrics.json").read_text())
        last = m["last"]
        hist = m.get("history")
        per_iou = last["per_class_iou"]
        fg = [v for k, v in per_iou.items() if k != "background"]
        folds[fold_dir.name] = {
            "miou": last["miou"],
            "miou_fg": float(np.mean(fg)),
            "instrument_iou": per_iou.get("instrument"),
            "pupil_iou": per_iou.get("pupil"),
            "per_class_iou": per_iou,
            "miou_last5": float(np.mean([h["miou"] for h in hist[-5:]])) if hist else None,
            "best_miou_upper_bound": m["best"]["miou"],
        }
    return folds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--models", nargs="+", required=True)
    args = parser.parse_args()

    all_runs = {m: load_run(Path(args.runs) / m) for m in args.models}

    keys = ("instrument_iou", "pupil_iou", "miou_fg", "miou")
    print(f"{'model':<40}" + "".join(f"{k:>18}" for k in keys))
    for m, folds in all_runs.items():
        row = f"{m:<40}"
        for k in keys:
            vals = [f[k] for f in folds.values() if f[k] is not None]
            row += f"{np.mean(vals):>10.4f}±{np.std(vals):.3f}" if vals else f"{'—':>18}"
        print(row + f"   ({len(folds)} folds)")

    # paired per-fold differences on the decision metrics
    models = list(all_runs)
    print("\npaired per-fold differences (decision metrics):")
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a, b = models[i], models[j]
            shared = sorted(set(all_runs[a]) & set(all_runs[b]))
            for k in ("instrument_iou", "pupil_iou"):
                d = [all_runs[a][f][k] - all_runs[b][f][k] for f in shared
                     if all_runs[a][f][k] is not None and all_runs[b][f][k] is not None]
                if d:
                    wins = sum(x > 0 for x in d)
                    print(f"  {a} - {b} | {k}: mean {np.mean(d):+.4f}, "
                          f"per-fold {[f'{x:+.3f}' for x in d]}, wins {wins}/{len(d)}")

    print("\nNOTE: 'best' blocks are test-selected upper bounds — not comparable, not quotable.")


if __name__ == "__main__":
    main()
