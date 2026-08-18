"""Phase label timelines from the annotation CSVs.

Labels come from the `sec`/`endSec` columns only — the `frame`/fps metadata is
untrustworthy (case_4863's metadata fps is 240 vs the real ~60). Class 0 is
idle: any time not inside an annotated action segment, including the pre/post
surgery margins.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# index == class ID; canonical surgical order, idle first. Names verbatim from
# the CSVs (including the dataset's own spellings).
PHASES = (
    "idle",
    "Incision",
    "Viscoelastic",
    "Capsulorhexis",
    "Hydrodissection",
    "Phacoemulsification",
    "Irrigation/Aspiration",
    "Capsule Pulishing",
    "Lens Implantation",
    "Lens positioning",
    "Viscoelastic_Suction",
    "Anterior_Chamber Flushing",
    "Tonifying/Antibiotics",
)
PHASE_ID = {name: i for i, name in enumerate(PHASES)}
NUM_CLASSES = len(PHASES)


def load_segments(annotation_csv: Path | str) -> pd.DataFrame:
    """Rows of (phase_id, start_sec, end_sec), sorted, validated."""
    df = pd.read_csv(annotation_csv)
    unknown = set(df["comment"]) - set(PHASES)
    if unknown:
        raise ValueError(f"{annotation_csv}: unknown phase names {sorted(unknown)}")
    out = pd.DataFrame(
        {
            "phase_id": df["comment"].map(PHASE_ID),
            "start_sec": df["sec"].astype(float),
            "end_sec": df["endSec"].astype(float),
        }
    ).sort_values("start_sec").reset_index(drop=True)
    empty = out["end_sec"] == out["start_sec"]
    if empty.any():
        # e.g. case_5063 has one zero-duration Viscoelastic_Suction row
        print(f"load_segments: dropping {empty.sum()} zero-duration segment(s) in {annotation_csv}")
        out = out[~empty].reset_index(drop=True)
    bad = out["end_sec"] < out["start_sec"]
    if bad.any():
        raise ValueError(f"{annotation_csv}: {bad.sum()} segments with end<start")
    return out


def frame_labels(segments: pd.DataFrame, duration_sec: float, fps: float) -> np.ndarray:
    """Rasterize segments onto a [T] int64 label array sampled at `fps`; gaps = idle."""
    t = np.arange(0.0, duration_sec, 1.0 / fps)
    labels = np.zeros(len(t), dtype=np.int64)
    for row in segments.itertuples():
        labels[(t >= row.start_sec) & (t < row.end_sec)] = row.phase_id
    return labels


def phase_cases(phase_root: Path | str) -> list[str]:
    return sorted(p.name for p in (Path(phase_root) / "annotations").glob("case_*"))


def case_paths(phase_root: Path | str, case: str) -> tuple[Path, Path]:
    root = Path(phase_root)
    return (
        root / "videos" / f"{case}.mp4",
        root / "annotations" / case / f"{case}_annotations_phases.csv",
    )
