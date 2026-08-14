# Stage 2 — Phase Recognition: Design (pre-registered)

*2026-08-14. PI pre-registration review: REVISE → revisions below applied (see
`docs/reviews/2026-08-14-stage2-preregistration.md`); cleared to cache + E0. Goal: full-video
phase timelines good enough to power reports, per-phase norms, the phase-indexed library, and
Stage-3b priors.*

**Revisions from PI review (all applied before caching):**
- **B1/B2:** overlap is **27** phase∩seg cases (29 clean; seg-only: 5180/5299/5325). One
  homogeneous tool-feature rule, committed as `TrainIDs_PhaseRecognition/seg_checkpoint_map.csv`:
  overlap → the seg fold whose *test* split holds the case; clean → seeded round-robin.
  Anatomy features use `best.pt` (its runs predate `last.pt`; measured best−last ≤0.005,
  caveat recorded); multiclass uses `last.pt`.
- **B3:** backbone = **`facebook/dinov2-with-registers-large`** (mean-patch pooling free of
  artifact tokens); cache additionally stores a 7×9×1024 pooled patch grid per frame so
  spatial/attention heads never require re-extraction; cache files carry backbone/size/fps/SHA
  provenance. DINOv3-L is gated (manual HF approval) → E5 stretch only.
- **B4:** splits stratified by (ACF-presence × seg-overlap × duration); every val set asserted
  to cover all 13 classes; ACF-containing and overlap videos balanced across test folds.
- **C1:** zero-duration segment (case_5063) dropped loudly; class strings pinned in
  `phase/timeline.py`.
- **C2:** margins stay idle in train and eval; idle diagnostics will split margin-idle vs
  gap-idle.
- **C4:** metrics extended with median boundary-timing error (s) and per-phase duration MAE
  (s); A-vs-B claims use n=56 per-video paired stats (test folds partition all videos).
- **Ambition rungs adopted:** E6 self-training on the 944 unlabeled videos (also yields
  phase-duration cohort norms); 4-fold ensemble + disagreement as per-segment confidence with
  val temperature calibration; seg-encoder features as backbone ablation; duration-aware
  (HSMM-style) decoding rather than plain transition Viterbi.
- **Pre-registered bar for "impressive"** (fold-mean after decoding): edit ≥85, F1@50 ≥80,
  segment-count ratio 1.0±0.1, median boundary error ≤1 s, duration MAE ≤2 s on major phases,
  near-zero hallucinated segments; macro-F1 ≥0.80 over 13 classes. **Falsification probe:**
  decoding must improve (not erase) the 10 most-fragmented ground-truth timelines.

## Task definition

**Primary: temporal phase segmentation of untrimmed videos.** Input: one video at 5 fps.
Output: per-frame label over **13 classes** = 12 action phases + idle. Product-facing metrics
are segmental (a fragmented timeline breaks the video library even at high frame accuracy).

**Secondary (external comparability): the paper's clip protocol.** The Cataract-1K paper
benchmarks 3-second *clip classification* (10 frames per clip), 32/24 video split, with
Viscoelastic + Anterior-Chamber-Flushing merged due to shared visuals — an easier task than
timeline segmentation. We additionally evaluate our model under a clip-style protocol (majority
ground-truth label per 3 s window vs majority prediction, merged classes) for a rough external
anchor, with the usual convention caveats.

## Data facts (audited 2026-08-14)

- 56 videos, 512×384 @ 59.94/60 fps (README's 1024×768@30 is wrong for this subset), median
  360 s (241–955 s). All annotation CSVs pair with an existing video.
- case_4863's metadata fps is corrupt (240 vs real 60.22) → all label math uses the CSVs'
  `sec`/`endSec` columns, sanity-checked against video duration per case.
- Action phases cover median 84% of video time; the rest is idle (instrument exchanges) plus
  pre/post-surgery margins. Class imbalance is severe: Phacoemulsification 102 min vs
  Incision 7.6 min of 306 annotated minutes.
- **28 of the 56 phase cases are not in the segmentation set; the other 28 overlap it** —
  matters for any feature derived from the Stage-1 segmentation models (below).

## Splits (video-wise = patient-wise)

4-fold cross-validation, seeded and committed as CSVs: per fold, **36 train / 6 val / 14
test** videos, disjoint. **All model/checkpoint selection and hyperparameter choices use val
only** (institutionalizing the B1 lesson); test is touched once per fold per final model.
Fold assignment stratified by video duration so no fold concentrates the long/short surgeries.

## Features (cached once, then every head experiment is minutes)

- **Backbone: frozen DINOv2 ViT-L/14** (`facebook/dinov2-large`), FP16, no fine-tuning.
- **Resolution 518×392** (multiple of 14, preserves the videos' 4:3 aspect — no squeeze);
  224² kept as an ablation. Thin instruments are the main phase cue, so we do not down-res
  by default.
- **5 fps** sampling (protects the shortest phases: some Incision segments <2 s at margins);
  3 fps as an ablation.
- Cached per video: `[T, 2048]` = CLS ⊕ mean-patch, plus timestamps and labels. ~100k frames
  total, an estimated 15–25 min on 2 A40s — one-time cost.

## Heads (trained on cached features, full-video sequences)

- **H1: MS-TCN++** (~150-line clean implementation): multi-stage dilated TCN with the standard
  CE + truncated-MSE smoothing loss; class-weighted CE for imbalance.
- **H2: BiGRU** (2×256): the paper family's strongest head, on modern features.
- **Decoding (creative, classical):** beyond argmax — median filtering, and **Viterbi decoding
  with a transition matrix + duration priors estimated from train folds only**. Cataract
  surgery has a strong canonical phase grammar; decoding with it directly attacks
  over-segmentation, which is the metric that gates the video library.
- **E3 (creative, uses Stage 1): tool-presence fusion.** Append per-frame instrument/anatomy
  probabilities from the Stage-1 SegFormer models (12-dim) to the DINO features. Phases are
  largely *defined* by tool presence, and the segmentation models are strong. **Leakage
  control:** 28 phase videos overlap segmentation training cases; for those, tool features are
  computed with the fold checkpoint whose *test* set contains that case (i.e., a model that
  never trained on it); the 28 non-overlap videos are clean for any checkpoint.

## Metrics

Per fold, on test videos, with val-selected checkpoints:
- Frame: accuracy, macro-F1 (13 classes), per-class F1 with support counts.
- Segmental: edit score, F1@{10,25,50}; per-video segment-count ratio (predicted/true) as an
  over-segmentation indicator.
- Paper-protocol clip accuracy/F1 (merged classes) as the external anchor.
- Provenance discipline carried over: last-epoch + val-selected checkpoints saved, per-epoch
  history, git SHA + dirty flag, seed.

## Experiment ladder

- **E0 – sanity floor:** linear probe (logistic regression) on cached features, per-frame.
  Verifies features/labels/splits before any temporal modeling. Expect well above the 33%
  majority-class floor; if not, stop and debug.
- **E1 – heads:** H1 vs H2 on DINOv2-L@518×392@5fps; pick by val macro-F1 + edit.
- **E2 – ablations:** 224² features; 3 fps; CLS-only vs CLS⊕mean-patch.
- **E3 – tool fusion** (leakage-clean as above).
- **E4 – decoding:** median vs Viterbi-with-grammar on the best of E1–E3; final test-fold
  numbers reported once.
- **E5 (stretch):** DINOv3 backbone if weights are accessible; SSL-adapted features later.

## Risks / open questions for review

1. Idle at video margins: pre-Incision prep and post-Tonifying time is labeled idle like
   between-phase gaps — acceptable, or should margins be excluded from eval?
2. Viterbi with a hard grammar could fail on anatomically atypical surgeries (repeated
   phases are common: Viscoelastic appears twice in most timelines) — transition matrix is
   estimated from data, not hand-authored, and decoding is compared against plain smoothing
   before adoption.
3. The clip-protocol external anchor is convention-mismatched in our favor (we see more
   context); it will be labeled as such.
4. Two-GPU use: feature extraction shards videos across GPUs; head training is light enough
   for single-GPU per fold — folds run in parallel, 2 at a time.
