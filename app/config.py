"""Path/config resolution for the app.

Everything hangs off DATA_ROOT (default <repo>/data, override with
$PHACOSIGHT_DATA_ROOT). Every consumer tolerates missing pieces: a fresh clone
with no dataset still serves the UI, uploads, and any timelines present —
degraded gracefully rather than crashing.
"""

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("PHACOSIGHT_DATA_ROOT", REPO / "data"))

TIMELINE_DIR = DATA_ROOT / "library" / "phase_timelines"
NORMS_PATH = DATA_ROOT / "library" / "phase_norms.json"
UPLOAD_DIR = DATA_ROOT / "library" / "uploads"
PHASE_DIR = DATA_ROOT / "phase"            # labeled subset (optional)
FULL_DIR = DATA_ROOT / "full"              # raw archive videos (optional)
ASSETS = REPO / "app" / "assets"           # small committed artifacts (grammar, ...)
RUNS = REPO / "runs"                       # model checkpoints

TIMELINE_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
