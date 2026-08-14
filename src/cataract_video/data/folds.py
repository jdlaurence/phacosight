"""Load the committed cross-validation split CSVs and remap paths locally.

The committed CSVs reference the dataset two ways:

- ``TrainIDs_SemanticSegmentation_FiveFold``: absolute paths from the authors'
  cluster (``/storage/homefs/.../case_5015/img/...``) or repo-relative paths.
- ``TrainIDs_Semantic Segmentation``: repo-relative paths
  (``semantic_segmentation_images_annotations/Images_and_Supervisely_Annotations/case_.../...``).

Either way the stable suffix is ``case_<id>/<subdir>/<file>.png``, so we cut at
the ``case_<id>`` component and graft it onto a local ``data_root`` (the
directory that contains the ``case_*`` folders). ``mask_dir`` optionally swaps
the mask subfolder so one CSV family can serve any of the three mask variants.
"""

import re
from pathlib import Path

import pandas as pd

_CASE_RE = re.compile(r"case_\d+")


def _relocate(path: str, data_root: Path | str, mask_dir: str | None = None) -> Path:
    parts = Path(path).parts
    idx = next((i for i, p in enumerate(parts) if _CASE_RE.fullmatch(p)), None)
    if idx is None:
        raise ValueError(f"no case_<id> component in path: {path!r}")
    rel = list(parts[idx:])
    if mask_dir is not None and len(rel) >= 2 and rel[1].startswith("mask"):
        rel[1] = mask_dir
    return Path(data_root, *rel)


def load_fold(
    csv_path: Path | str,
    data_root: Path | str,
    mask_dir: str | None = None,
    require_exists: bool = False,
) -> pd.DataFrame:
    """Read a split CSV and return a DataFrame with local ``imgs``/``masks`` paths."""
    df = pd.read_csv(csv_path, usecols=["imgs", "masks"])
    df["imgs"] = df["imgs"].map(lambda p: _relocate(p, data_root))
    df["masks"] = df["masks"].map(lambda p: _relocate(p, data_root, mask_dir=mask_dir))
    if require_exists:
        missing = [p for col in ("imgs", "masks") for p in df[col] if not Path(p).exists()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} of {2 * len(df)} files from {csv_path} are missing "
                f"under {data_root} (first: {missing[0]}). Download the dataset and/or "
                "generate the masks with the upstream Dataset_codes scripts."
            )
    return df


def fold_case_ids(df: pd.DataFrame) -> set[str]:
    """Patient/case IDs referenced by a split (for leakage checks)."""
    ids = set()
    for p in df["imgs"]:
        m = _CASE_RE.search(str(p))
        if m:
            ids.add(m.group(0))
    return ids
