# Stage 2 — Phase Recognition: Results

*2026-08-14. Task: offline 13-class (12 phases + idle-as-a-class) timeline segmentation of
untrimmed videos at 5 fps; 4-fold video-disjoint CV over all 56 phase-annotated videos
(stratified splits, val-only selection, test touched once per model); n=56 per-video paired
statistics. PI-reviewed at pre-registration, E1, and round 2
(`docs/reviews/2026-08-14-stage2-*.md`); all numbers below survived adversarial reproduction.*

## Headline (deployment stack: tool fusion × 3-seed ensemble + Viterbi)

| Metric | Value | Pre-registered bar |
|---|---|---|
| macro-F1 (13 classes) | **0.954** (single seed; seed-mean 0.953) | ≥0.80 |
| frame accuracy | 0.959 | — |
| segmental edit score | **94.1** | ≥85 |
| segmental F1@50 | **94.5** | ≥80 |
| segment-count ratio | 0.99 | 1.0 ± 0.1 |
| median boundary error | **0.20 s** (= the 5 fps sampling floor) | ≤1 s |
| clip-protocol anchor (paper convention, 5,687 clips) | 95.7% acc / 0.948 macro recall | — |

Idle is scored as a real 13th class (harder than the standard convention, which excludes
background). Fragmented-10 probe passed: grammar decoding improves the most-fragmented
timelines (edit 91.5→92.0) with mild repeat-compression (36.8 predicted vs 39.3 true
segments per video).

## What moved the needle

| Configuration (seed-means) | macro-F1 | edit (viterbi) |
|---|---|---|
| E0 linear probe (no temporal context, fold 0) | 0.645 | 11 |
| BiGRU on frozen DINOv2-reg-L | 0.925 | 87.5 |
| MS-TCN++ on frozen DINOv2-reg-L | 0.940 | 92.1 |
| **+ tool-presence fusion (E3)** | **0.952** | **94.1** |
| **+ 3-seed ensemble + temperature (deployment stack)** | **0.954** (acc 0.960) | 93.8 |

- **Tool fusion** (36-dim per-frame presence features from the Stage-1 segmentation models,
  leakage-clean per the committed checkpoint map): honest seed-mean gain **+1.20 pp**
  macro-F1 (official seeds 0.9489/0.9524/0.9555 vs base 0.9349/0.9405/0.9455 — every
  tools seed beats every base seed); beats base on all folds and 24/33 non-tied videos
  (Wilcoxon p=2.9e-4, n=56).
  Attribution: **Hydrodissection +9.3 pp** (the hydro-cannula signal fixes the visually
  weakest phase) plus training stabilization (base runs' catastrophic hydro folds vanish).
  It is *not* ensembling in disguise — single fused model beats the base 3-seed ensemble.
  Viscoelastic did not move (both OVD phases use the same cannula; it is now the weakest
  action class, F1 0.915). A fully-clean ablation (multiclass-only features, no caveated
  checkpoint) reaches 0.949 — the anatomy-`best.pt` footnote does not carry the gain.
- **Learned-grammar Viterbi**: transition matrix estimated from train folds only (repeats
  are the norm — a hand-authored left-to-right grammar would be wrong). Segment ratio
  1.04 → 0.99, edit +2.
- **Confidence** (within-fold 3-seed ensembles — cross-fold models saw the video — with
  temperature fit on val). Deployment stack (fusion ensemble): frame ECE 0.010 → **0.008**
  (T ≈ 1.0–1.06); top-10%-disagreement frames carry **4.5×** the average error; dropping
  the lowest-confidence 5% of segments lifts segment purity(≥0.5) from 97.4% → **99.0%**.
  Frame confidence is the probability of the Viterbi-decoded label (conservative where
  decoding overrides frames), not an HMM posterior.

## Per-phase duration accuracy (fusion + Viterbi; what the report will quote)

All action phases ≤2 s MAE except Phacoemulsification 3.2 s (2.9% relative on a ~110 s
phase). Hydrodissection: 4.53 s → **1.80 s** under fusion. **Idle remains ±~6 s per video**
— idle-time report lines must carry that uncertainty. Gap-idle (between-step exchanges,
median 2.6 s) accuracy 0.914 vs margin-idle 0.960.

## Scaling / bulk-pass pricing

1 fps costs 0.9 pp on the base model (0.940 → 0.931 seed-mean vs seed0) and 1.0 pp on
fusion (0.953 → 0.943) — and **fusion@1fps (0.943) beats DINO-only@5fps (0.940)**. The
944-video pass (phase library + duration cohort norms) therefore runs the fusion stack at
1 fps: ~4–6 GPU-hrs total including the segmentation pass, which the product needs anyway
(instrument-usage reporting) — making tool features free at deployment.

## Caveats and open falsifications

- **Offline ≠ online**: full-video bidirectional context; no streaming claim.
- **Single clinic, single device, scripted procedure** — external validity untested. Open
  probe: frozen pipeline on another center's videos (e.g. Cataract-101) before strong
  product claims.
- Clip anchor is convention-mismatched in our favor (full-video context; 36 vs 32 train
  videos; the paper's exact Table 5 numbers are not retrievable) — anchor, not head-to-head.
- F1@k matching is a slightly generous variant of the Farha reference; measured difference
  0.000 on fold 0; strict matching will be used for any external number.
- Boundary error floor (0.20 s) is the sampling grid, not model precision.

## Reproduce

```
python scripts/make_phase_splits.py                     # splits + seg checkpoint map
python scripts/extract_phase_features.py --shard 0/2    # DINOv2-reg-L cache (+1/2)
python scripts/extract_tool_features.py --shard 0/2     # Stage-1 seg presence features
python scripts/train_phase.py --config configs/phase_mstcnpp_tools.yaml --fold all
python scripts/eval_phase.py --run runs/phase_mstcnpp_tools
python scripts/confidence_phase.py --runs runs/phase_mstcnpp_tools ..._seed1 ..._seed2
python scripts/clip_anchor.py
```
