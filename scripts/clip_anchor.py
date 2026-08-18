#!/usr/bin/env python
"""External anchor: score the timeline model under the paper's clip protocol.

The Cataract-1K paper benchmarks 3-second action-clip classification with
Viscoelastic and Anterior_Chamber Flushing merged (shared visuals). Here each
annotated action segment of each test video is cut into non-overlapping 3 s
windows; the model's clip prediction is the majority Viterbi-decoded frame
label in the window (idle counts as an error — harsher than a clip classifier
that can't predict idle at all).

Convention caveats vs the paper (label the number accordingly): our model sees
full-video context (their clips are classified in isolation) but our split is
36 train vs their 32, and Table 5's exact numbers are unverifiable from the
public HTML. This is an anchor, not a head-to-head.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from phacosight.phase.decoding import transition_matrix, viterbi
from phacosight.phase.timeline import NUM_CLASSES, PHASE_ID, PHASES, case_paths, load_segments
from eval_phase import fold_logits

MERGE = {PHASE_ID["Anterior_Chamber Flushing"]: PHASE_ID["Viscoelastic"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="runs/phase_mstcnpp_dinov2l")
    parser.add_argument("--features", default="data/features/phase_dinov2l")
    parser.add_argument("--split-dir", default="TrainIDs_PhaseRecognition")
    parser.add_argument("--clip-sec", type=float, default=3.0)
    parser.add_argument("--fps", type=float, default=5.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    win = int(args.clip_sec * args.fps)
    per_class = Counter()
    correct_class = Counter()

    for fold in range(4):
        logits, train_labels = fold_logits(Path(args.run), fold, Path(args.features),
                                           Path(args.split_dir), device)
        log_trans = transition_matrix(train_labels, NUM_CLASSES)
        for case, (lp, _) in logits.items():
            pred = viterbi(lp, log_trans)
            pred = np.array([MERGE.get(int(p), int(p)) for p in pred])
            _, ann = case_paths("data/phase", case)
            for row in load_segments(ann).itertuples():
                cls = MERGE.get(row.phase_id, row.phase_id)
                s = int(row.start_sec * args.fps)
                e = int(row.end_sec * args.fps)
                for w0 in range(s, e - win + 1, win):
                    clip_pred = np.bincount(pred[w0 : w0 + win]).argmax()
                    per_class[cls] += 1
                    correct_class[cls] += int(clip_pred == cls)

    total = sum(per_class.values())
    acc = sum(correct_class.values()) / total
    recalls = {PHASES[c]: correct_class[c] / n for c, n in sorted(per_class.items())}
    print(f"clip protocol (3 s, merged visco/ACF): {total} clips, accuracy {acc:.4f}")
    for name, r in recalls.items():
        print(f"  {name:<28}{r:.3f}  (n={per_class[PHASE_ID[name] if name in PHASE_ID else 0]})")
    print(f"macro recall over merged classes: {np.mean(list(recalls.values())):.4f}")


if __name__ == "__main__":
    main()
