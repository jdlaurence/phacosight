# PhacoSight — physician web app

Browse and review model-analyzed cataract surgeries, compare phase durations
against cohort norms, explore each phase's cohort profile with example clips,
track a physician's progress over time, and upload new recordings for analysis
(with delete/re-analyze and a live queue). On the case page, an AI anatomy
overlay (SegFormer: cornea/pupil/lens/instrument) renders on paused frames and
as per-phase keyframes — on demand via `/api/overlay`, no precompute.

## Run locally (single command)

```
git clone https://github.com/jdlaurence/phacosight.git && cd phacosight
python -m venv .venv && .venv/bin/pip install -e ".[app]" torch torchvision transformers
.venv/bin/python -m app          # → http://localhost:7860
```

Hardware is auto-detected: with a CUDA GPU, uploaded-video analysis runs in
about a minute per surgery; without one, browsing/search work identically and
analysis falls back to CPU (slow but functional). `--host 0.0.0.0` exposes the
app on the network; `--data-root` / `$PHACOSIGHT_DATA_ROOT` points at the data
directory (default `<repo>/data`).

A bare clone (no dataset, no checkpoints) still serves the full UI and any
timelines present. Inference additionally needs the model checkpoints under
`runs/` (see the repo docs); the decoding grammar ships in `app/assets/` and
the deployment stack self-checks against a labeled video whenever the labeled
subset is present.

## Data model

- `data/library/phase_timelines/<case>.json` — one file per analyzed surgery:
  segments (phase, start/end, confidence, disagreement), duration, provenance,
  and for uploads: `source: "uploaded"`, `physician`, `surgery_date`.
- Base-archive and labeled-set videos are distinguished by `source`
  (`library` / `labeled`); **uploads never enter cohort norms** — enforced in
  both the app's norms service and `scripts/build_phase_norms.py`.

## Cloud extension seam (future: public site + hosted inference)

The web layer depends only on the `InferenceService` interface in
`app/inference.py` (`submit(path, meta) -> job_id`, `jobs[job_id]`). To move
inference off-box (Colab, RunPod, a GPU worker pool):

1. Implement the same interface with a remote transport (upload video → run
   the identical stack — `scripts/analyze_phase_bulk.py` is self-contained —
   → POST the timeline JSON back to `TIMELINE_DIR`).
2. Everything else (norms, library, progress, UI) is pure file-backed reads
   and needs no changes.
3. Keep the self-check: any remote worker must reproduce ≥0.85 accuracy on the
   reference video before its outputs are accepted.
