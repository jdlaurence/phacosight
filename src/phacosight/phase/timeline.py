"""Phase label timelines from the annotation CSVs.

Labels come from the `sec`/`endSec` columns only — the `frame`/fps metadata is
untrustworthy (case_4863's metadata fps is 240 vs the real ~60). Class 0 is
idle: any time not inside an annotated action segment, including the pre/post
surgery margins.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# index == class ID; canonical surgical order, idle first. Names are the
# project's cleaned-up spellings; the dataset CSVs' verbatim names (and any
# artifact written before the 2026-08-19 cleanup) map in via PHASE_ALIASES.
PHASES = (
    "idle",
    "Incision",
    "Viscoelastic",
    "Capsulorhexis",
    "Hydrodissection",
    "Phacoemulsification",
    "Irrigation/Aspiration",
    "Capsule Polishing",
    "Lens Implantation",
    "Lens Positioning",
    "Viscoelastic Suction",
    "Anterior Chamber Flushing",
    "Tonifying/Antibiotics",
)
PHASE_ID = {name: i for i, name in enumerate(PHASES)}
NUM_CLASSES = len(PHASES)

# Legacy name -> canonical. Keys are the Cataract-1K CSVs' verbatim spellings
# (typo, underscores, casing), which are also what pre-cleanup timeline/norms
# artifacts stored. Class IDs and order are unchanged by the rename.
PHASE_ALIASES = {
    "Capsule Pulishing": "Capsule Polishing",
    "Lens positioning": "Lens Positioning",
    "Viscoelastic_Suction": "Viscoelastic Suction",
    "Anterior_Chamber Flushing": "Anterior Chamber Flushing",
}


def canonical_phase(name: str) -> str:
    """Map a dataset-CSV or legacy-artifact phase name to its canonical form."""
    return PHASE_ALIASES.get(name, name)


def load_segments(annotation_csv: Path | str) -> pd.DataFrame:
    """Rows of (phase_id, start_sec, end_sec), sorted, validated."""
    df = pd.read_csv(annotation_csv)
    names = df["comment"].map(canonical_phase)
    unknown = set(names) - set(PHASES)
    if unknown:
        raise ValueError(f"{annotation_csv}: unknown phase names {sorted(unknown)}")
    out = pd.DataFrame(
        {
            "phase_id": names.map(PHASE_ID),
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
