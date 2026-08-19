# Cataract Surgery Video Analysis — Architecture Experimentation Plan

## Context

The Cataract-1K repo is being extended into a system that ingests a recorded cataract surgery (~7 min, 1024×768, 30 fps) and produces, for resident-physician education:

1. Per-frame semantic segmentation of anatomy (cornea, pupil, lens) + instruments
2. Surgical phase timeline (12 action phases + idle) with durations
3. Irregularity detection (sudden pupil contraction/miosis, IOL rotation/instability)
4. A template-based report with metrics, feedback, and annotated frames
5. **Per-phase instrument-movement feedback** — for each step of the procedure, kinematic metrics (path length, velocity, smoothness, decentration) with feedback against cohort norms *(added 2026-08-14 from physician feedback)*
6. **Phase-indexed surgical video library** — the physician can upload/analyze new videos, and later query across all previously analyzed videos by surgical step (e.g. "show me all main incisions") and get the matching clips *(added 2026-08-14 from physician feedback)*

**Hard constraint:** inference for a 7-minute video must complete in "a few minutes" on an A40-class GPU (48 GB, Ampere, ≈RTX 3090 compute). This plan selects candidate architectures per task, based on a deep survey of 2023–2026 literature (citations at bottom), and defines the experiments to choose between them. **Segmentation is the first experimental priority** — it feeds annotated frames, pupil tracking, and instrument-presence signals.

Key conclusion from the research: **the speed budget is generous once you sample frames per task** — the binding constraint is accuracy with small training sets (30 segmentation videos / 56 phase videos), so architecture choices favor data efficiency, with speed as a tiebreaker.

## Performance budget

A 7-min 30 fps video = ~12,600 frames. No task needs all of them:

| Task | Sampling rate | Frames per video | Rationale |
|---|---|---|---|
| Video decode | all (GPU NVDEC, sequential) | 12,600 | NVDEC does ~1,000+ fps at this res → ~10–15 s |
| Phase recognition | 3–5 fps | ~1,300–2,100 | 1 fps is the Cholec80 standard; cataract phases are short (lens implantation ≈ 4 s) so 3–5 fps protects them |
| Pupil/limbus tracking | 5–10 fps, phase-gated | ~2,100–4,200 | Miosis filter cutoff is 0.1 Hz; higher rate helps instrument-obstruction handling; IOL rotation happens in <7 s |
| Full-scene segmentation for report frames | ~0.2–1 fps + keyframes | ~100–400 | Dataset annotations are 1 frame/5 s; report exhibits need no more |

⇒ the heaviest per-frame model needs only ~20–40 fps sustained. Every candidate below clears this on an A40 with FP16, so accuracy dominates the choice. End-to-end target: **~2–4 min/video**.

## Dataset facts that constrain design

- Phase labels (56 videos) and segmentation labels (30 videos, 2,256 frames) are **disjoint video subsets** → joint multi-task training needs alternating-batch machinery for modest gains. **Decision: separate task heads/models; optionally share a self-supervised-pretrained backbone later.**
- The ~944 unannotated videos are 512×324 @ 25 fps (annotated ones are 1024×768) — still usable for SSL pretraining, pseudo-labeling, and building phase-duration norm distributions for reports.
- Use the repo's committed patient-wise 5-fold splits (`upstream/TrainIDs_SemanticSegmentation_FiveFold/`, paths need remapping from the authors' cluster) for comparability with published benchmarks.
- Known accuracy anchors on Cataract-1K: pupil Dice ≈ 94–98 (easy), instruments Dice ≈ 77–83 (bottleneck); best published mIoU 0.88 (CatSeg 2026, but only ~6–8 fps); the paper's own phase benchmark: ResNet50+BiGRU best.

## Candidate architectures

### Task 1: Semantic segmentation (first priority)

One combined anatomy + multi-class-instrument semantic model (~14 classes) at 512×512–768×768. No published modern-efficient-backbone results on Cataract-1K exist — likely easy wins over the paper's VGG-era baselines.

| Candidate | Why | Est. on A40 |
|---|---|---|
| **SegFormer-B2** (primary) | ~30M params, Cityscapes 81.7 mIoU; transformer robustness to motion blur/specular reflection (the dominant corruptions here); expect ~0.85+ mIoU at ~8× CatSeg's speed | ~30–60 fps FP16 |
| **EfficientViT-B1-seg** (speed candidate) | Cityscapes 80.5 mIoU at 175 img/s on A100 @1024×2048; open question whether linear attention handles transparent IOL/thin shafts | ~150+ fps |
| **PIDNet-S** (CNN control) | 78.6 mIoU; explicit boundary-attention branch attractive for pupil-edge fidelity | ~100+ fps |

Rejected as primary: Mask2Former/CatSeg (5–15 fps, heavy), YOLO-seg alone (poor fit for concentric anatomy, coarse masks on thin shafts), SAM2-only (class-agnostic). **Phase-2 option:** Cutie-small/SAM2-Tiny mask propagation between keyframes to temporally smooth pupil/IOL signals (SASVi showed consistency gains: IoU_OF 0.575→0.634).

Training recipe (from LensID/Cataract-1K group): 512×512, augmentation = motion blur + Gaussian blur + random contrast/brightness + shift/scale/rotate, loss = CE + log-soft-Dice (λ=0.8), ImageNet init, oversample frames with rare instruments using the dataset's per-fold instance tables.

### Task 2: Phase recognition

Strongest evidence for our setting: a controlled cataract-surgery study swapping only frozen features into an identical MS-TCN++ head found **frozen DINOv3 ViT-L beats ImageNet ResNet-50 by ~7–9 pp accuracy**. Laparoscopy foundation models (EndoViT, GSViT) show negligible gains — wrong domain. End-to-end video transformers hold the best cataract-domain numbers (GLSFormer: 92.9 acc / 81.9 Jaccard on Cataract-101) but risk overfitting 40 training videos.

| Candidate | Why | Cost on A40 |
|---|---|---|
| **Frozen DINOv2/v3 ViT-L features @ 3–5 fps + MS-TCN++ / SR-Mamba-style bidirectional head** (primary) | Most data-efficient; features cached once (~1 min/video), then every temporal-head experiment trains in minutes; offline/bidirectional fits post-op report use case | features ~1 min/video once; head <1 s/video |
| **Surgformer / GLSFormer** (end-to-end TimeSformer, K400 init) (accuracy challenger) | Best published cataract-domain results; mitigate overfitting with the GLSFormer recipe (8–16 frames @1 fps, heavy augmentation) | ~1–2 min/video @1 fps output |
| **DACAT / SurgicalMamba-style online head** (only if streaming ever matters) | Cholec80 SOTA (96.1 acc), O(1)/frame — an advantage irrelevant to offline reports | <1 min/video |

Backbone ablation within the primary: frozen DINOv2/v3 ViT-L vs fine-tuned ConvNeXt-T/ResNet50 vs OphCLIP/OphNet (the only surgical-microscope-domain foundation models). Report both relaxed and strict boundary metrics; expect viscoelastic↔AC-flushing confusion and idle boundaries to dominate errors. Handle class imbalance with stochastic balanced clip sampling (LensID trick) + macro-F1 evaluation.

### Task 3: Irregularity detection (built on Task 1 outputs, not new architectures)

**Pupil contraction/miosis** (canonical published pipeline, PLOS ONE 2021 + Ophthalmology Science 2024):
1. Segment pupil + iris/limbus at 5–10 fps (phase-gated: phaco onward).
2. Normalize: pupil width ÷ limbus width (cancels zoom/working distance; limbus ≈ 11.5–12 mm gives physical scale). Width is more stable than area under vertical eye motion.
3. Detect/compensate instrument obstruction of the pupil margin (mask-overlap test; interpolate).
4. Zero-phase Butterworth low-pass (~0.1 Hz) → extrema detection → flag excursions > ~0.12 × median reference. Published recall: 90.6% for medium/strong reactions — adequate for a flagging system.
5. Report onset time, magnitude (%), recovery, and surgical phase.

**IOL rotation/instability** (LensID baseline + rotation extension):
1. Gate to lens-implantation phase onward (phase model provides this).
2. Segment lens + pupil per frame; LensID metrics: instability = normalized lens-center↔pupil-center distance over time; unfolding delay = relative lens/pupil area over time.
3. Rotation angle (no published Cataract-1K baseline — novel contribution): principal-axis (ellipse-fit/PCA) tracking of the lens mask incl. haptics → θ(t), unwrapped and smoothed → deg/s and cumulative rotation. Fallback/upgrade: point tracking (CoTracker-style) on haptic tips. Measure relative to iris landmarks to cancel eye cyclotorsion.
4. Validate on the dataset's pupil-reaction and IOL-rotation subsets.

### Task 4: Per-phase instrument-movement feedback (built on Task 1 + Task 2 outputs)

Generalizes the phaco-only PhacoTrainer metrics to **every phase of the procedure**. No new architectures — this is post-processing of segmentation masks gated by the phase timeline:

1. Per frame (at the segmentation sampling rate, raised to ~5–10 fps during instrument-active phases), extract instrument **tip position** from the multi-class instrument mask: skeletonize the mask, take the endpoint farthest from the image border (instrument shafts enter from the periphery; the distal endpoint is the working tip). Track per instrument class; handle two-instrument phases (phaco + chopper, I/A + second instrument) as separate tracks.
2. Normalize coordinates by the limbus circle (center + radius from Task 1 anatomy masks) so metrics are zoom/centration-invariant and comparable across videos.
3. Per phase, compute the kinematic metric set from the surgical-skill literature: **path length, mean/peak tip velocity, movement smoothness (jerk / spectral arc length), tip decentration from pupil center, time-in-motion vs holding, instrument-exchange count, bimanual coordination** (inter-tip distance stats for two-handed phases). PhacoTrainer validated path length/velocity against expert ratings; CatSkill's centration/stability metrics complement them.
4. Feedback generation: compare each metric against the **cohort norm distribution for that phase** (built from the annotated videos first, extended via pseudo-labels later — same norms machinery as phase durations). Rubric-based template text per metric band (e.g. ">90th percentile path length during capsulorhexis — consider more deliberate movements"), grounded in OSACSS/ICO-OSCAR rubric language.
5. Report artifacts: per-phase tip-trajectory plots overlaid on a representative frame, metric table vs cohort percentiles, flagged outlier phases.

Risks/open items: tip localization on thin/transparent instruments (validate against a hand-checked subset); velocity estimates are sensitive to segmentation dropout — median-filter tracks and report confidence; no public Cataract-1K kinematics baseline exists, so validation is qualitative + expert review by the physician user.

### Task 5: Phase-indexed video library & retrieval

Every analyzed video's outputs are persisted so the physician can later query by surgical step across all previous surgeries ("show me all main incisions"):

- **Store:** per-video `metrics.json` + phase timeline + per-phase kinematics written to a **SQLite index** (videos, phases with start/end timestamps, metrics, event flags) alongside the raw video and derived artifacts on disk. SQLite is enough for hundreds–thousands of videos; no server dependency.
- **Retrieval:** query by phase name (and later by attributes: duration percentile, flagged events, date range) → returns matching segments. Clips are cut **on demand** with ffmpeg stream-copy (`-ss/-to -c copy`, no re-encode, sub-second per clip) rather than pre-cutting ~13 clips per video — keeps storage at ~1× the source videos.
- **Ingestion workflow:** an `analyze` entry point (`scripts/analyze_video.py <video>`) that runs the full pipeline on a new upload, writes report + index rows, and is idempotent per video hash. This is the same orchestrator as Stage 4 — the library is a thin persistence layer on top of it.
- **Interface:** start with a CLI + generated HTML gallery page (phase-filterable, thumbnails linking to clips). A small web UI (upload form + search page, e.g. FastAPI + static frontend) is the likely eventual form for the physician — deferred until the pipeline is stable; the SQLite schema is the contract that makes the UI a frontend-only task later.
- Phase-recognition accuracy directly bounds retrieval quality — segmental metrics (edit score, F1@k) from Stage 2 are the relevant gate, since over-segmentation would fragment the library's clips.

## Pipeline architecture

```
video ──► NVDEC sequential decode (torchcodec, device="cuda", ~10–15 s/video)
              │  stride-routed frames, 2 CUDA streams (decode ∥ inference)
              ├─► every 6–10th frame @224–384 ──► frame features ──► temporal head ──► phase timeline
              ├─► every 3–6th frame @512² ──► segmentation (FP16/TensorRT) ──► masks
              │
        Pass 2 (phase-gated re-decode of phaco→end only)
              └─► 10 fps pupil/limbus/lens series ──► miosis + IOL analysis
                                                          │
        metrics.json ◄── phase durations, CatSkill/kinematic metrics, events ◄──┘
              ├─► template renderer (Jinja2 → HTML/PDF, matplotlib charts, annotated frames)
              └─► library index (SQLite: video, phase segments, metrics) ──► phase search / clip retrieval (ffmpeg stream-copy)
```

- Decode sequentially once and route by stride — never random-seek on GPU.
- torchcodec is the PyTorch-blessed decode path (benchmarks ahead of decord/torchvision); DALI as fallback if we want a fused decode→resize→normalize graph.
- Separate models per task at their own resolutions/rates — this scheduling freedom is where the time budget is won.

## Report contents (template-based, per user decision)

Deterministic `metrics.json` → Jinja2/HTML(→PDF) renderer with matplotlib charts. Metrics grounded in published skill-assessment work (CatSkill, PhacoTrainer, WetCat — each correlates with OSACSS/ICO-OSCAR expert rubrics):

1. Phase timeline + per-phase durations vs cohort norms (56 annotated videos give the norm distribution; the 944 unlabeled videos can extend it via pseudo-labels later)
2. Idle-time fraction and instrument-exchange count
3. Eye centration/stability/focus per phase (CatSkill's LCP1/LCFC/LFL — cheap by-products of limbus segmentation)
4. Per-phase instrument kinematics for **all phases** (Task 4): path length, velocity, smoothness, decentration, bimanual coordination — each vs cohort percentile, with rubric-based feedback text and tip-trajectory overlays
5. Pupil-diameter trajectory with flagged miosis events (time, magnitude, phase)
6. Post-implantation IOL stability/rotation trace
7. Annotated keyframes per phase from segmentation masks

## Experiment roadmap

**Stage 0 — Scaffolding (prerequisite):**
- Download Segmentation + Phase Recognition + irregularity sets from Synapse (links in README).
- New `src/` package + `configs/`; generate masks with the repo's existing `upstream/Dataset_codes/semantic segmentation dataset codes/` scripts; remap fold-CSV paths to local (small path-mapping utility).
- Decode benchmark: torchcodec NVDEC vs CPU on one video; lock the decode layer.

**Stage 1 — Segmentation bake-off (first priority):**
- Train SegFormer-B2, EfficientViT-B1, PIDNet-S on the anatomy+instruments task, 5-fold patient-wise, shared recipe above (mmseg or HF Transformers).
- Metrics: mIoU/Dice per class per fold; pupil IoU tracked separately (gates irregularity work); per-model FP16 fps on the A40.
- Winner criteria: best mIoU subject to ≥30 fps; then train the multi-class-instruments variant with the winner.
- Export winner to TensorRT/FP16; measure real throughput.

**Stage 2 — Phase recognition bake-off:**
- Cache frozen DINOv2/v3 ViT-L features @ 3–5 fps for all 56 videos (one-time, ~1 GPU-hr).
- Train MS-TCN++ and a bidirectional-Mamba/SR-Mamba-style head on cached features; ablate backbone (DINO vs fine-tuned ConvNeXt-T vs OphCLIP if weights available).
- Metrics: accuracy, macro-F1, segmental edit/F1@k, relaxed + strict boundaries; compare against the paper's ResNet50+BiGRU baseline.
- Only if the two-stage approach plateaus below the paper's baseline: run Surgformer/GLSFormer challenger.

**Stage 3 — Irregularity detection:**
- Implement pupil-series pipeline (obstruction compensation, Butterworth, change detection); validate recall/precision on the pupil-reaction subset.
- Implement LensID-style IOL instability + principal-axis rotation; validate on the IOL-rotation subset.

**Stage 3b — Instrument kinematics (Task 4):**
- Tip extraction (skeleton endpoint) + limbus-normalized tracks; validate tip localization against a small hand-annotated frame set.
- Per-phase metric computation (path length, velocity, smoothness, decentration, bimanual stats); build per-phase cohort norm distributions from the annotated videos.
- Rubric-based feedback text mapping (metric percentile bands → template sentences), reviewed with the physician.

**Stage 4 — Integration + report:**
- Two-pass orchestrator behind a single `scripts/analyze_video.py` entry point (idempotent per video hash); define the `metrics.json` schema (the contract); CatSkill/PhacoTrainer metric extraction; template renderer.
- End-to-end wall-clock benchmark on full videos on the A40 (target ≤4 min).

**Stage 5 — Video library & retrieval (Task 5):**
- SQLite schema (videos / phase segments / metrics / events) written by the Stage 4 orchestrator.
- Query CLI + on-demand ffmpeg stream-copy clip extraction; phase-filterable HTML gallery of analyzed surgeries.
- Later (with physician input on workflow): small web UI for upload + search on top of the same index.

**Later / stretch:** DINO-style SSL pretraining on the 944 unlabeled videos (SelfSupSurg showed up to +7 pp phase F1); pseudo-label those videos for phase-duration norms at scale (Dual Invariance Self-Training — by the Cataract-1K authors — is the recipe); Cutie/SAM2-Tiny temporal smoothing of pupil/IOL masks; capsulorhexis geometry metrics (needs a rhexis-boundary class or fit).

## Verification

- **Stage 1:** 5-fold mIoU vs published anchors (paper baselines ≈0.80, CatSeg 0.88); pupil Dice ≥0.95; winner ≥30 fps FP16 on A40.
- **Stage 2:** beat the dataset paper's ResNet50+BiGRU on its own split protocol; sanity-check confusion matrix against expected viscoelastic/AC-flushing confusion.
- **Stage 3:** recall ≥0.9 on medium/strong pupil reactions (published bar); qualitative validation of rotation traces on the IOL-rotation subset (no published baseline exists).
- **Stage 3b:** tip-localization error ≤ ~10 px on a hand-checked frame subset; kinematic traces qualitatively reviewed by the physician on a handful of videos (no published Cataract-1K kinematics baseline exists).
- **Stage 4:** process 3 held-out full videos end-to-end in ≤4 min each on the A40; report renders with all metrics populated; visual spot-check of annotated frames.
- **Stage 5:** analyze N videos, then verify a phase query (e.g. "all main incisions") returns exactly the segments in the ground-truth/predicted timelines; retrieved clips play and land on the correct step; re-running `analyze_video.py` on an already-indexed video is a no-op.

## Key sources

- Cataract-1K: [Sci Data 2024](https://www.nature.com/articles/s41597-024-03193-4) / [arXiv 2312.06295](https://arxiv.org/abs/2312.06295); follow-ups: [CatSeg](https://www.sciencedirect.com/science/article/pii/S1746809426006336), [SASVi](https://arxiv.org/abs/2502.09653), [RobustSurg](https://arxiv.org/pdf/2512.02188)
- Segmentation: [SegFormer](https://arxiv.org/abs/2105.15203) · [EfficientViT](https://github.com/mit-han-lab/efficientvit) · [PIDNet](https://arxiv.org/abs/2206.02066) · [DeepPyram](https://arxiv.org/abs/2109.05352) · [SAM2](https://arxiv.org/html/2408.00714v2) / [Surgical SAM 2](https://arxiv.org/abs/2408.07931) / [Cutie](https://arxiv.org/abs/2310.12982)
- Phase: [Surgformer](https://github.com/isyangshu/Surgformer) · [GLSFormer](https://github.com/nisargshah1999/GLSFormer) · [SKiT](https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_SKiT_a_Fast_Key_Information_Video_Transformer_for_Online_Surgical_ICCV_2023_paper.pdf) · [SR-Mamba](https://arxiv.org/abs/2407.08333) · [DACAT](https://github.com/kk42yy/DACAT) · [frozen-DINOv3 cataract study](https://arxiv.org/pdf/2604.10514) · [OphCLIP](https://arxiv.org/abs/2411.15421) · [OphNet](https://github.com/minghu0830/OphNet-benchmark)
- Irregularities: [LensID](https://arxiv.org/abs/2107.00875) · [pupil reaction detection, PLOS ONE 2021](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0258390) · [intraoperative pupil analysis, Ophth. Science 2024](https://www.ophthalmologyscience.org/article/S2666-9145(24)00133-7/fulltext)
- Skill metrics: [CatSkill](https://pmc.ncbi.nlm.nih.gov/articles/PMC12084080/) · [PhacoTrainer](https://pmc.ncbi.nlm.nih.gov/articles/PMC8606857/) · [WetCat](https://arxiv.org/html/2506.08896)
- Engineering: [torchcodec](https://pytorch.org/blog/torchcodec/) · [NVDEC app note](https://docs.nvidia.com/video-technologies/video-codec-sdk/12.0/nvdec-application-note/index.html)
- Semi/self-supervised: [SelfSupSurg](https://ar5iv.labs.arxiv.org/html/2207.00449) · [Dual Invariance Self-Training](https://arxiv.org/html/2501.17628) · [SemiVT-Surge](https://arxiv.org/pdf/2506.01471)
