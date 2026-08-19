#!/usr/bin/env python
"""E7-0: frozen Stage-2 deployment stack, zero-shot on CATARACTS (default: official
test split, never trained on) — sizes the cross-center domain gap before any
augmentation training (docs/cataracts-harmonization-plan.md).

Stack = 3-seed x 4-fold MS-TCN++ tool-fusion ensemble at 1 fps, temperature 1.04,
Viterbi with the C1K deployment grammar (app/assets/transition_matrix_1fps.npy) —
exactly the bulk/app deployment configuration. Features come from the cached
5 fps CATARACTS extractions (pre-registered 4:3 crop), strided to 1 fps.

Labels: frozen CATARACTS->C1K map; IGNORE frames are spliced out of both pred and
true before metrics (frame metrics unaffected; segmental metrics computed on the
spliced sequences — convention recorded in the output and held fixed for E7-c).

    python scripts/eval_cataracts_zeroshot.py --out runs/e70_zeroshot/results.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phacosight.data.cataracts import IGNORE_INDEX, cataracts_cases
from phacosight.phase.decoding import mode_smooth, viterbi
from phacosight.phase.heads import HEADS
from phacosight.phase.metrics import PhaseMetrics, edit_score, f1_at_k, segments
from phacosight.phase.timeline import NUM_CLASSES, PHASES

TOOLS_RUNS = ["runs/phase_mstcnpp_tools_1fps_seed0", "runs/phase_mstcnpp_tools_1fps_seed1",
              "runs/phase_mstcnpp_tools_1fps_seed2"]
TEMPERATURE = 1.04  # deployment value (analyze_phase_bulk)


def video_macro_f1(pred: np.ndarray, true: np.ndarray) -> float:
    """Per-video macro-F1 over classes present in that video's (spliced) ground truth
    — the pre-registered per-video convention for the E7-c paired stats."""
    f1s = []
    for c in np.unique(true):
        tp = int(((pred == c) & (true == c)).sum())
        fp = int(((pred == c) & (true != c)).sum())
        fn = int(((pred != c) & (true == c)).sum())
        f1s.append(2 * tp / max(1, 2 * tp + fp + fn))
    return float(np.mean(f1s))


def video_f1_at_50(pred: np.ndarray, true: np.ndarray) -> float:
    tp, fp, fn = f1_at_k(pred, true, 0.50)
    denom = 2 * tp + fp + fn
    return float(100 * 2 * tp / denom) if denom else 0.0


def load_heads(device, runs=TOOLS_RUNS) -> list:
    heads = []
    for run in runs:
        for fold in range(4):
            ckpt = torch.load(Path(run) / f"fold{fold}" / "val_best.pt",
                              map_location="cpu", weights_only=False)
            cfg = ckpt["config"]
            eff = cfg.get("features_fps", 5.0) / cfg.get("frame_stride", 1)
            assert eff == 1.0, f"{run}/fold{fold} not a 1 fps head"
            h = HEADS["mstcnpp"](2084, NUM_CLASSES).to(device).eval()
            h.load_state_dict(ckpt["model"])
            heads.append(h)
    return heads


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--runs", default=None,
                        help="comma-separated head run dirs (default: deployment TOOLS_RUNS); "
                             "e.g. the E7-b aug seeds for the E7-c arm")
    parser.add_argument("--experiment", default="E7-0")
    parser.add_argument("--features", default="data/features/cataracts_dinov2l")
    parser.add_argument("--tools", default="data/features/cataracts_tools")
    parser.add_argument("--cataracts-root", default="data/cataracts")
    parser.add_argument("--stride", type=int, default=5, help="5 fps cache -> 1 fps inference")
    parser.add_argument("--trans", default="app/assets/transition_matrix_1fps.npy")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = args.runs.split(",") if args.runs else TOOLS_RUNS
    heads = load_heads(device, runs)
    log_trans = np.load(args.trans)
    cases = [c for c in cataracts_cases(args.cataracts_root) if c[3] == args.split]
    print(f"{args.experiment} on CATARACTS {args.split} ({len(cases)} videos), "
          f"{len(heads)} heads, T={TEMPERATURE}")

    decodings = ("argmax", "mode", "viterbi")
    agg = {d: PhaseMetrics(NUM_CLASSES, PHASES, fps=1.0) for d in decodings}
    per_video = []
    for stem, _, _, _ in cases:
        d = np.load(Path(args.features) / f"{stem}.npz")
        tools = np.load(Path(args.tools) / f"{stem}.npz")["features"]
        x = np.concatenate([d["features"].astype(np.float32),
                            tools.astype(np.float32)], axis=1)[::args.stride]
        y = d["labels"].astype(np.int64)[::args.stride]
        xt = torch.from_numpy(x).unsqueeze(0).to(device)
        probs = torch.stack([torch.softmax(h(xt)[-1, 0], -1) for h in heads]).mean(0)
        probs = torch.softmax(torch.log(probs + 1e-12) / TEMPERATURE, -1).cpu().numpy()
        lp = np.log(probs + 1e-12)
        keep = y != IGNORE_INDEX
        preds = {"argmax": lp.argmax(-1),
                 "mode": mode_smooth(lp.argmax(-1), 9),
                 "viterbi": viterbi(lp, log_trans)}
        row = {"case": stem, "n": int(keep.sum()), "ignored": int((~keep).sum()),
               "true_segments": len(segments(y[keep]))}
        for name, p in preds.items():
            agg[name].update(p[keep], y[keep])
            row[f"acc_{name}"] = float((p[keep] == y[keep]).mean())
            row[f"edit_{name}"] = edit_score(p[keep], y[keep])
            row[f"segs_{name}"] = len(segments(p[keep]))
            row[f"macro_f1_{name}"] = video_macro_f1(p[keep], y[keep])
            row[f"f1_50_{name}"] = video_f1_at_50(p[keep], y[keep])
        per_video.append(row)
        print(f"  {stem}: acc(viterbi) {row['acc_viterbi']:.3f} edit {row['edit_viterbi']:.1f} "
              f"segs {row['segs_viterbi']}/{row['true_segments']} (ignored {row['ignored']})")

    import subprocess
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                         cwd=Path(__file__).parents[1]).stdout.strip()
    result = {"experiment": args.experiment, "split": args.split, "temperature": TEMPERATURE,
              "grammar": args.trans, "ignore_frames_spliced": True,
              "per_video_macro_f1_convention": "classes present in that video's spliced GT",
              "git_sha": sha, "features": args.features, "tools": args.tools,
              "heads": runs, "decodings": {}, "per_video": per_video}
    for name in decodings:
        m = agg[name].compute()
        m["confusion"] = agg[name].confusion.tolist()  # rows = true, cols = pred (PI #5)
        result["decodings"][name] = m
        print(f"{name:>8}: acc {m['accuracy']:.4f} macroF1 {m['macro_f1']:.4f} "
              f"edit {m['edit']:.1f} f1@50 {m['f1@50']:.1f} segratio {m['seg_ratio']:.2f} "
              f"bnd {m['boundary_median_s'] if m['boundary_median_s'] is None else round(m['boundary_median_s'], 2)}s")
    v = result["decodings"]["viterbi"]["per_class_f1"]
    print("per-class F1 (viterbi):",
          {k: round(f, 3) for k, f in v.items() if result['decodings']['viterbi']['per_class_support'][k] > 0})

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
