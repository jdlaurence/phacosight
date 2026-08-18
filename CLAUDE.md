# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project goal

**PhacoSight** is a **cataract surgery video analysis system for resident physician education** (this repo began as the Cataract-1K dataset-release repo — see below — and retains its preprocessing scripts and split CSVs). The system processes recordings of cataract surgeries and produces:

1. **Semantic segmentation** of anatomy (cornea, pupil, lens) and surgical instruments per frame.
2. **Phase recognition and timing** — identify the twelve action phases plus idle periods, and extract per-phase durations and transitions.
3. **Generated reports** summarizing the procedure (phase timeline, instrument usage, notable events/irregularities such as pupil contraction or IOL rotation).
4. **Detailed feedback and annotated images/frames** for resident physicians reviewing their own surgeries.
5. **Per-phase instrument-movement feedback** — kinematic metrics (path length, velocity, smoothness, decentration) for each surgical step, compared against cohort norms.
6. **Phase-indexed video library** — analyzed videos are indexed by surgical phase so past surgeries can be searched by step (e.g. "all main incisions") and matching clips retrieved.

The Cataract-1K dataset (described below) is the training/evaluation data for this work: its phase-annotation CSVs support the timing/recognition components, its pixel-level annotations support segmentation, and its irregularity subsets support event detection. New model training, inference, and report-generation code should build on the existing preprocessing scripts and committed cross-validation splits rather than reinventing them — in particular, reuse the class-ID conventions and patient-wise fold splits documented below so results remain comparable to the paper's benchmarks.

## PI review (required step)

`.claude/agents/pi.md` defines a Principal Investigator reviewer agent (expert ML practitioner + ophthalmic-surgery domain expert). **Invoke the `pi` agent to review the pipeline and results whenever new experimental results are in** — end of a training run or bake-off stage, new evaluation numbers, dataset changes — and before acting on any experiment-derived decision (picking a winner model, changing the recipe, promoting to the next stage). Address its blocking findings before relying on the results.

## Development (new model code)

The experimentation plan (architectures, performance budget, experiment roadmap) lives at `docs/experimentation-plan.md`. New code lives in `src/phacosight/` (src layout), configs in `configs/`, entry points in `scripts/`.

- Environment: `.venv` (on the cluster: Python 3.12, torch+cu130, 2× A40 GPUs).
- Tests: `.venv/bin/python -m pytest tests/ -q` — includes CPU smoke tests with synthetic data; no dataset download needed.
- Train segmentation on both GPUs (preferred): `.venv/bin/torchrun --standalone --nproc_per_node=2 scripts/train_seg.py --config configs/seg_segformer_b2.yaml --fold 0` (or `--fold all`). Plain `.venv/bin/python scripts/train_seg.py ...` still works single-GPU/CPU.
- Download data (needs Synapse account + `SYNAPSE_AUTH_TOKEN`): `python scripts/download_data.py --sets segmentation phase` → `data/`. On the cluster, `data/` is a symlink to `/mnt/data/projects/jd_exp/cataract` — datasets live on that mount, not in the home folder.
- Class-ID conventions are codified in `src/phacosight/labels.py` — they mirror the upstream mask scripts exactly (note: multiclass instruments = 10 tools grouped into 6 classes).
- `src/phacosight/data/folds.py` remaps the committed fold CSVs' cluster/relative paths to a local data root; `data_root` in configs points at the local `Images_and_Supervisely_Annotations` directory.
- EfficientViT and PIDNet candidates need extra setup (pip git install / vendoring into `third_party/`) — see `src/phacosight/models/segmentation.py` docstring.

## The Cataract-1K dataset (upstream contents)

`Dataset_codes/` and the `TrainIDs_*` directories are retained from the dataset-release repo for **Cataract-1K** (paper: https://arxiv.org/pdf/2312.06295.pdf), a cataract surgery video dataset for scene segmentation, phase recognition, and irregularity detection; the full upstream dataset description is preserved at `docs/cataract-1k-dataset.md`. Those upstream materials are preprocessing scripts and cross-validation split CSVs only — **the actual dataset (videos, images, annotations) is not in this repo**; it is downloaded separately from Synapse (links in README.md).

The upstream scripts have no build system or test suite. They are standalone Python files run directly (`python <script>.py`) and depend on numpy, pandas, and Pillow; `frame_rate_changer.py` also shells out to `ffmpeg`.

## How the scripts work

All scripts use **hardcoded relative paths** at the top of the file (e.g. `phase_recognition_annotations/`, `semantic_segmentation_images_annotations/Images_and_Supervisely_Annotations/`). They assume the downloaded dataset has been extracted into the working directory with those exact folder names. To run one, either place the data accordingly or edit the path variables — there are no CLI arguments.

### `Dataset_codes/phase recognition dataset codes/`
Turns per-video phase-annotation CSVs (start/end frame per phase, fps) into training clips:
- `action_frame_extractor.py` / `idle_frame_extractor.py` — cut action-phase / idle segments out of the surgery videos using the annotation CSVs.
- `frame_rate_changer.py` — re-encodes clips to 30 fps via ffmpeg.
- `dataset_creation.py` — reorganizes extracted clips into a `Training_Dataset/` folder grouped by phase name.

### `Dataset_codes/semantic segmentation dataset codes/`
Two script families operating on Supervisely JSON polygon annotations (per case: `case_<id>/img/` and `case_<id>/ann/`):
- `json_to_*mask*.py` — rasterize polygons into grayscale PNG masks where the pixel value is the class train-ID. Three task variants, each writing to its own mask folder inside each case: anatomy+instruments (`mask_anatomy_inst`: background=0, Cornea=1, Pupil=2, Lens=3, any instrument=4), binary instruments (`mask_instruments`), and multi-class instruments (`mask_instruments_MultiClass`). `json_to_visual_mask_AllClassess.py` makes color visualization masks instead of training masks. Masks are drawn in a fixed order (Cornea → Pupil → Lens → instruments) so later classes overwrite earlier ones; images are 1024×768.
- `subdataset_generator_*.py` — build patient-wise cross-validation splits: 30 cases split into 5 folds of 6 cases, written as `<name>_<fold>_train.csv` / `_test.csv` with `imgs`/`masks` path columns.

The exact class-title strings from the Supervisely JSON matter: anatomy titles are `Cornea`, `Pupil`, `Lens`, and the 10 instrument titles are listed in the `instrument_names` variable in each mask script.

## Split CSVs (pre-generated, committed)

- `TrainIDs_SemanticSegmentation_FiveFold/` — the five-fold splits described in the paper (`*_<fold>_train.csv` / `*_<fold>_test.csv`). Note: `imgs`/`masks` columns contain absolute paths from the authors' cluster and must be remapped to local paths before use.
- `TrainIDs_Semantic Segmentation/` (note the space in the folder name) — four-fold train/validation/test splits (`*_train_fold<N>.csv` etc.) with relative paths.

Both directories have parallel subfolders for the anatomy+instruments task and the instruments-only task.

## Citation/license constraints

Dataset is CC BY 4.0 and any use requires citing the Cataract-1K publication (BibTeX in README.md and `docs/cataract-1k-dataset.md`). Keep the citation and download sections of README.md intact when editing it.

## Physician web app

`app/` is the FastAPI + vanilla-JS tool for physicians: browse/review analyzed surgeries (interactive timeline synced to video, per-phase durations vs cohort percentiles, confidence flags), search the phase library with clip download, and upload new videos for GPU analysis (uses the validated deployment stack with its startup self-check; uploads never enter cohort norms). Run on the cluster: `.venv/bin/uvicorn app.server:app --host 0.0.0.0 --port 7860`, then `ssh -L 7860:localhost:7860 <cluster>` and open http://localhost:7860.
