# PI Review — Stage 2 Phase-Recognition Design (pre-registration)

*2026-08-14. All data facts recomputed from disk (56 annotation CSVs, video headers, fold CSVs, run directories). Verdict: **REVISE** — right shape, but four blockers to fix before the cache is built and splits are committed. Disposition at bottom: all applied same day.*

## 2. Blocking findings

**B1. Overlap accounting wrong: 27 overlap / 29 clean, not 28/28.** Seg-only cases: 5180, 5299, 5325. Every overlap case maps to exactly one seg test fold (verified), so a held-out checkpoint exists for each. *Fix:* correct the doc; commit the case→held-out-fold assignment as a CSV next to the phase fold CSVs.

**B2. E3 checkpoint rule names a nonexistent artifact and is silent on clean videos.** `segformer_b2_anatomy_instrument/fold*/` has only `best.pt` (predates the N1 fix); multiclass has `last.pt`. Not phase-label leakage (selection was on seg mIoU; bias symmetric; best−last ≤0.005) but must be documented. Clean videos need a pinned rule too — otherwise feature quality correlates with overlap status, which is confounded with recording era. *Fix:* one homogeneous rule — every video gets exactly one 24-case checkpoint: overlap → held-out fold (`last.pt` multiclass, `best.pt` anatomy with caveat), clean → seeded round-robin. Committed in the same CSV.

**B3. Feature-cache spec — three changes before the one-time build.**
(a) Use the **registers** variant (`facebook/dinov2-with-registers-large`): plain DINOv2-L has high-norm artifact patch tokens (Darcet et al. 2023) and mean-patch over 1036 tokens is exactly the pooling they corrupt.
(b) Cache a pooled patch grid (e.g. 7×9×1024 fp16) alongside CLS⊕mean — same forward pass, ~13 GB, enables spatial/attention heads without re-extraction. Cheapest insurance against redoing the expensive step.
(c) Check DINOv3-L access (the cited frozen-feature evidence is a DINOv3 study); write backbone/resolution/fps/normalization/git-SHA into each cache file.

**B4. Duration-only stratification produces rare-phase-blind folds.** Anterior_Chamber Flushing absent from 10/56 videos; Capsule Pulishing and Lens positioning each absent from 1. *Fix:* stratify on (duration × ACF-presence × seg-overlap); assert every val set contains ≥1 segment of all 13 classes; balance overlap videos across folds.

## 3. Concerns (non-blocking)

**C1.** Annotation warts: one zero-duration segment (case_5063 Viscoelastic_Suction) — drop loudly; 12 segments <2 s (min 0.8 s) → 5 fps default confirmed right; pin exact class strings (incl. "Capsule Pulishing") in a labels module.
**C2.** Margins: keep as idle in train and eval — verified safe (all videos start with Incision, end with Tonifying; pre-margins are genuine prep, max 46.1 s). Add margin-idle vs gap-idle diagnostic (874 mid-video gaps, 50.2 min, median 2.6 s).
**C3.** Val=6 videos is a noisy selection signal → prefer last-epoch under decay-to-zero LR; val for model-level choices only.
**C4.** Add median boundary-timing error (s) and per-phase duration MAE (s); report per-class F1 with support and pool rare classes; use n=56 per-video paired stats (test folds partition all videos).
**C5.** Confirm which classes the paper merged (three viscoelastic-adjacent classes exist) before computing the clip anchor; keep the convention-mismatch label.

## 4. Big picture

Right task (timeline segmentation primary; clip protocol demoted to labeled anchor). sec/endSec decision verified correct (case_4863's frame column is fictional; endSec matches real duration).

**Where to be more ambitious, cheaply (in order of value/GPU-hour):**
1. Promote self-training on the 944 unlabeled videos to a concrete E6 (up to +7 pp in cited work; ~10× the 56-video cache; same pass yields phase-duration cohort norms for Task 4).
2. 4-fold ensemble + disagreement as per-segment confidence + val temperature calibration — free at inference; "review this segment manually" is worth more clinically than +1 pp macro-F1.
3. Seg-encoder features as a backbone ablation (surgery-tuned, on disk, same B2 leakage rule).
4. Push Viterbi to duration-aware (HSMM-style) decoding; grammar must be estimated, not hand-authored (repeats are the norm: 110 Viscoelastic_Suction, 125 Tonifying segments across 56 videos).

**Bar for "impressive"** (fold-mean after decoding): edit ≥85, F1@50 ≥80, segment-count ratio 1.0±0.1, median boundary error ≤1 s, per-phase duration MAE ≤2 s on major phases, near-zero hallucinated segments, macro-F1 ≥0.80 over 13 classes. That would sit at/above GLSFormer-class numbers on a harder task. Dramatically beating it = red flag to investigate.

**Falsification probe to pre-register:** evaluate grammar decoding separately on the 10 most-fragmented ground-truth timelines (e.g. case_5316: 18 segments) — decoding must improve, not erase, real repeats.

## 5. What was done well

Task reframing (timeline vs clip) correct and honestly anchored; sec/endSec audit found the real failure mode; E3 leakage-control structurally sound (unique held-out fold verified for all 27); B1-lesson institutionalized (val-only selection, E0 sanity floor); 4-fold geometry right for N=56 (buys n=56 paired stats).

**Disposition requested:** fix B1-B4, then proceed to caching and E0 without further review.

## Disposition (applied same day, before caching)

- B1/B2 → doc corrected (27/29); `TrainIDs_PhaseRecognition/seg_checkpoint_map.csv` committed (27 held-out + 29 round-robin); anatomy `best.pt` caveat recorded.
- B3 → registers backbone; 7×9×1024 grid cached; provenance fields in every cache file; DINOv3 confirmed gated → E5 stretch.
- B4 → stratified splits with val-coverage assertion (all pass; ACF 11-12 per test fold, overlap 6-7).
- C1 → zero-duration drop implemented (loud); C2 margins kept, diagnostic planned; C4 metrics implemented in `phase/metrics.py`.
- Ambition rungs (E6 self-training, ensemble-confidence, seg-encoder ablation, HSMM decoding) and the quantitative bar + falsification probe adopted into the design doc.
