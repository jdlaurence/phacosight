"""CATARACTS-2020 phase ground truth → C1K 13-class taxonomy.

Implements the pre-registered label map of ``docs/cataracts-harmonization-plan.md``
(PI-reviewed 2026-08-18). **Map FROZEN 2026-08-18** after the sampled-clip review
recorded in the plan doc: idle→idle (2–3.5% instrument-in-eye of 200 sampled, under
the 10% rule), IDs 2/11 IGNORE confirmed, ID 12 amended to IGNORE, ID 18 merge to
Tonifying/Antibiotics confirmed. Any further amendment requires a plan-doc update.
(2026-08-19: target phase names respelled to the canonical forms in
``phacosight.phase.timeline`` — a display-spelling change only, class IDs and
mapping semantics untouched.)

GT format: per-video CSV ``Frame,Steps`` at native video frame rate, frames
numbered from 1, label 0 = Idle, labels 1-18 index ``STEPS``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from phacosight.phase.timeline import PHASE_ID

IGNORE_INDEX = -100  # torch cross-entropy default; masked in loss and metrics

# IDs 1-18 in evaluation_cataracts2020.py order (0 = Idle).
STEPS = (
    "Idle",
    "Toric Marking",
    "Implant Ejection",
    "Incision",
    "Viscodilatation",
    "Capsulorhexis",
    "Hydrodissetion",  # upstream typo preserved: these strings mirror the eval script
    "Nucleus Breaking",
    "Phacoemulsification",
    "Vitrectomy",
    "Irrigation/Aspiration",
    "Preparing Implant",
    "Manual Aspiration",
    "Implantation",
    "Positioning",
    "OVD Aspiration",
    "Suturing",
    "Sealing Control",
    "Wound Hydratation",
)

# CATARACTS step ID -> C1K phase name (None = IGNORE_INDEX).
STEP_TO_PHASE: dict[int, str | None] = {
    0: "idle",                     # pending R1 idle-rule check at map freeze
    1: None,                       # Toric Marking — no C1K equivalent
    2: None,                       # Implant Ejection — provisional (alt: Lens Implantation)
    3: "Incision",
    4: "Viscoelastic",
    5: "Capsulorhexis",
    6: "Hydrodissection",
    7: "Phacoemulsification",      # Nucleus Breaking folded into phaco (plan table ID 7)
    8: "Phacoemulsification",
    9: None,                       # Vitrectomy — complication management
    10: "Irrigation/Aspiration",
    11: None,                      # Preparing Implant — off-eye closeups, OOD for C1K idle
    12: None,                      # Manual Aspiration — only in the 2 vitrectomy videos; atypical cannula
    13: "Lens Implantation",
    14: "Lens Positioning",
    15: "Viscoelastic Suction",    # OVD Aspiration
    16: None,                      # Suturing — no C1K equivalent
    17: None,                      # Sealing Control — no C1K equivalent
    18: "Tonifying/Antibiotics",   # Wound Hydratation — provisional merge
}

STEP_TO_C1K_ID = np.array(
    [IGNORE_INDEX if name is None else PHASE_ID[name] for _, name in sorted(STEP_TO_PHASE.items())],
    dtype=np.int64,
)


def load_steps(csv_path: Path | str) -> np.ndarray:
    """Per-frame CATARACTS step IDs, validated: contiguous frames from 1, IDs in range."""
    df = pd.read_csv(csv_path)
    if list(df.columns) != ["Frame", "Steps"]:
        raise ValueError(f"{csv_path}: unexpected columns {list(df.columns)}")
    frames = df["Frame"].to_numpy()
    if frames[0] != 1 or not np.array_equal(frames, np.arange(1, len(frames) + 1)):
        raise ValueError(f"{csv_path}: Frame column not contiguous from 1")
    steps = df["Steps"].to_numpy(dtype=np.int64)
    if steps.min() < 0 or steps.max() >= len(STEPS):
        raise ValueError(f"{csv_path}: step IDs outside 0..{len(STEPS) - 1}")
    return steps


def to_c1k(steps: np.ndarray) -> np.ndarray:
    """Map per-frame step IDs to C1K phase IDs (IGNORE_INDEX where unmappable)."""
    return STEP_TO_C1K_ID[steps]


def labels_at_times(steps: np.ndarray, video_fps: float, times: np.ndarray) -> np.ndarray:
    """C1K labels sampled at ``times`` (seconds), matching the feature-cache convention
    (frame index = time * fps, as in extract_phase_features)."""
    idx = np.clip(np.round(times * video_fps).astype(np.int64), 0, len(steps) - 1)
    return to_c1k(steps[idx])


def steps_at_times(steps: np.ndarray, video_fps: float, times: np.ndarray) -> np.ndarray:
    """Raw CATARACTS step IDs at ``times`` — cached alongside mapped labels so a map
    amendment at freeze never requires re-extracting features."""
    idx = np.clip(np.round(times * video_fps).astype(np.int64), 0, len(steps) - 1)
    return steps[idx]


def cataracts_cases(root: Path | str) -> list[tuple[str, Path, Path, str]]:
    """(stem, video_path, gt_csv_path, split) for all 50 videos, sorted by stem.
    Split names follow the GT directories (train/dev/test); 2020 dev/test keep 2017
    ``test*`` filenames, so split identity comes from which GT dir holds the CSV."""
    root = Path(root)
    out = []
    for split in ["train", "dev", "test"]:
        for csv_path in sorted((root / "ground_truth/CATARACTS_2020" / f"{split}_gt").glob("*.csv")):
            video = root / "videos/micro" / f"{csv_path.stem}.mp4"
            if not video.exists():
                raise FileNotFoundError(f"no video for GT {csv_path}")
            out.append((csv_path.stem, video, csv_path, split))
    return sorted(out)
