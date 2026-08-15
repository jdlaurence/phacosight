# Incident: first bulk-pass outputs invalid (rate-mismatched heads)

*2026-08-14. Caught by the pre-planned no-label verification gate before any downstream
consumer (norms, library) used the outputs. All 342 affected timelines deleted.*

**Symptom.** First 342 bulk timelines showed 75% flagged segments, median confidence 0.57,
0/342 canonical phase ordering, durations 2–4× ground truth. Initial hypothesis was domain
shift (older 512×324 videos).

**Control that settled it.** Running the exact bulk stack on *labeled* videos scored
accuracy 0.44 / macro-F1 0.18 — where the same models score ~0.94 in CV. Not domain shift:
a pipeline bug.

**Root cause.** The bulk pass ran at 1 fps using the 5 fps-trained tool-fusion checkpoints.
MS-TCN++ receptive fields are defined in *samples*, not seconds — a 5 fps head sees 1 fps
input as 5× time-compressed and collapses. The validated "fusion@1fps = 0.943" number came
from heads trained *and* evaluated at 1 fps; deployment used the wrong checkpoints.

**Fixes (all in `scripts/analyze_phase_bulk.py` + configs):**
1. Official tools@1fps seed trio trained; bulk stack now loads those.
2. **Rate assertion**: every loaded head's `features_fps/frame_stride` must equal the
   inference fps — the bug class is now impossible to ship silently.
3. **Self-check gate**: before writing any bulk output, the full stack must reproduce
   ≥0.85 accuracy on a labeled video (`case_4687`); the script refuses to run otherwise.
   This guards rate mismatches, feature-order bugs, normalization drift, checkpoint mixups.

**Lessons.** (a) The verification-before-consumption gate earned its place — the invalid
outputs never touched a product artifact. (b) "Validated component" ≠ "validated
deployment": the ablation validated a (head, rate) *pair*; deployment recomposed the pair
incorrectly. Deployment stacks now self-check end-to-end against known ground truth at
startup, always.
