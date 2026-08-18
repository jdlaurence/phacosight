#!/usr/bin/env python
"""Generate committed phase splits + the E3 segmentation-checkpoint map.

Per PI pre-registration review (docs/reviews/, Stage 2):
- 4 folds; per fold 36 train / 6 val / 14 test videos, video-disjoint; the four
  test folds partition all 56 videos.
- Stratified by (Anterior_Chamber-Flushing presence x seg-set overlap x
  duration) so no fold is blind to the class missing from 10 videos and E3
  feature provenance doesn't cluster (B4).
- Every val set must contain >=1 segment of all 13 classes (asserted).
- seg_checkpoint_map.csv: for every phase video, the single Stage-1
  segmentation fold checkpoint allowed to compute its tool features (B1/B2):
  overlap cases -> the seg fold whose TEST set holds the case (never trained on
  it); clean cases -> seeded round-robin. One homogeneous rule, committed.

Deterministic; writes TrainIDs_PhaseRecognition/.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phacosight.phase.timeline import (
    NUM_CLASSES, PHASE_ID, case_paths, load_segments, phase_cases,
)

N_FOLDS = 4
N_VAL = 6
SEED = 0
SEG_SPLIT_DIR = Path("TrainIDs_SemanticSegmentation_FiveFold/TrainIDs_Cataract_1k_Anatomy_Instruments")
ACF = "Anterior_Chamber Flushing"


def seg_test_fold_map() -> dict[str, int]:
    """case -> seg fold whose test split contains it (the held-out checkpoint)."""
    out = {}
    for fold in range(5):
        df = pd.read_csv(SEG_SPLIT_DIR / f"Cat1k_anatomy_instrument_{fold}_test.csv")
        for p in df["imgs"]:
            case = Path(p).parts[-3]
            assert out.get(case, fold) == fold, f"{case} in two seg test folds"
            out[case] = fold
    return out


def main() -> None:
    root = Path("data/phase")
    out_dir = Path("TrainIDs_PhaseRecognition")
    out_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)

    cases = phase_cases(root)
    seg_map = seg_test_fold_map()
    info = {}
    for c in cases:
        video, ann = case_paths(root, c)
        cap = cv2.VideoCapture(str(video))
        dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        phases = set(load_segments(ann)["phase_id"]) | {0}
        info[c] = {
            "duration": dur,
            "phases": phases,
            "has_acf": PHASE_ID[ACF] in phases,
            "overlap": c in seg_map,
        }

    # --- fold assignment: deal within (has_acf, overlap) strata by duration ---
    fold_of = {}
    for key in {(i["has_acf"], i["overlap"]) for i in info.values()}:
        stratum = sorted((c for c in cases if (info[c]["has_acf"], info[c]["overlap"]) == key),
                         key=lambda c: info[c]["duration"])
        for start in range(0, len(stratum), N_FOLDS):
            block = stratum[start : start + N_FOLDS]
            for c, f in zip(block, rng.permutation(N_FOLDS)[: len(block)]):
                fold_of[c] = int(f)

    # --- per-fold val selection with full class coverage (asserted) ---
    for k in range(N_FOLDS):
        test = sorted(c for c in cases if fold_of[c] == k)
        rest = sorted(set(cases) - set(test))
        val = None
        for attempt in range(1000):
            cand = sorted(rng.choice(rest, N_VAL, replace=False))
            covered = set().union(*(info[c]["phases"] for c in cand))
            if len(covered) == NUM_CLASSES and any(not info[c]["has_acf"] for c in cand):
                val = cand
                break
        assert val is not None, f"fold {k}: no val set covering all classes found"
        train = sorted(set(rest) - set(val))
        assert not (set(train) & set(val)) and not (set(train) & set(test))
        assert len(train) + len(val) + len(test) == len(cases)
        for name, group in (("train", train), ("val", val), ("test", test)):
            pd.DataFrame({"case": group}).to_csv(out_dir / f"fold{k}_{name}.csv", index=False)
        n_acf = sum(info[c]["has_acf"] for c in test)
        n_ovl = sum(info[c]["overlap"] for c in test)
        print(f"fold {k}: {len(train)}/{len(val)}/{len(test)} — test: {n_acf} ACF, "
              f"{n_ovl} seg-overlap, median dur "
              f"{np.median([info[c]['duration'] for c in test]):.0f}s")

    all_test = sorted(c for k in range(N_FOLDS)
                      for c in pd.read_csv(out_dir / f"fold{k}_test.csv")["case"])
    assert all_test == cases, "test folds must partition all cases"

    # --- E3 checkpoint map ---
    clean = sorted(c for c in cases if c not in seg_map)
    rr = {c: int(f) for c, f in zip(clean, rng.permutation(len(clean)) % 5)}
    rows = [{"case": c,
             "seg_fold": seg_map.get(c, rr.get(c)),
             "overlap": c in seg_map} for c in cases]
    pd.DataFrame(rows).to_csv(out_dir / "seg_checkpoint_map.csv", index=False)
    n_over = sum(r["overlap"] for r in rows)
    print(f"seg_checkpoint_map.csv: {n_over} overlap (held-out fold), "
          f"{len(rows) - n_over} clean (round-robin)")


if __name__ == "__main__":
    main()
