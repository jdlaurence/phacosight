# Stage 1 — Segmentation Bake-off Results

*2026-08-14. 3 models × 5 patient-wise folds (committed FiveFold splits), anatomy+instruments task,
shared recipe (512², CE + log-soft-Dice λ=0.8, AdamW 6e-5, 40 epochs, LensID augmentations,
rare-instrument oversampling), 2× A40 DDP. PI-reviewed pre-run and post-run
(see `docs/reviews/2026-08-14-stage1-*.md`); decision protocol per those reviews:
last-epoch metrics, paired per-fold comparison on instrument + pupil IoU.*

## Winner: SegFormer-B2

Last-epoch metrics, mean ± std over 5 folds:

| Model | Instrument IoU | Pupil IoU | fg-mIoU | mIoU (incl. bg) | FP16 fps (b1, 512²) |
|---|---|---|---|---|---|
| **SegFormer-B2** | **0.813 ± 0.012** | **0.930 ± 0.004** | **0.878** | **0.900** | 86 |
| EfficientViT-B1 | 0.791 ± 0.012 | 0.913 ± 0.008 | 0.855 | 0.882 | 139 |
| PIDNet-S† | 0.533 ± 0.032 | 0.793 ± 0.019 | 0.679 | 0.739 | 163 |

- **Paired per-fold**: SegFormer beats EfficientViT on **5/5 folds** on both decision metrics
  (instrument +0.022 mean, paired t p=0.004; pupil +0.017, p=0.002). Per-case (30 test cases):
  25/30 instrument, 28/30 pupil (Wilcoxon p≈4e-7 / 3.5e-8). Worst cases are shared between
  models (case_5015 instruments, case_5017 pupil) — case difficulty, not model pathology.
- **Speed is a non-constraint**: all three clear the ≥30 fps deployment gate by 3–5×
  (`runs/bench_seg_fp16_b1_512.json`), so accuracy decides.
- SegFormer won despite EfficientViT's Cityscapes-dense-prediction pretraining advantage
  (SegFormer used ImageNet-only mit-b2 with a random decode head).

† **PIDNet-S scope**: trains from scratch (no accessible ImageNet init), inference-only head
(no auxiliary P/D boundary supervision), transformer-native recipe. The only supported claim is
"not competitive under the shared practical recipe from scratch" — no architecture-level
conclusion. It lost by far more than the fold-noise threshold that would have triggered a fair
rerun, so it is dropped.

## Comparison to published anchors — internal only

Instrument Dice ≈0.90 vs the paper's 77–83 and fg-mIoU 0.878 ≈ CatSeg's 0.88 are **not
convention-matched**: we pool confusion matrices over the split (per-image-averaged Dice is
systematically lower on instruments), evaluate at 512² aspect-squeezed rather than native
768×1024, and use a 2021+ pretrained backbone. Pupil Dice 0.964 lands inside the published
94–98 band. Do not quote against external numbers without re-evaluating per-image at native
resolution.

## Reproduce

```
torchrun --standalone --nproc_per_node=2 scripts/train_seg.py --config configs/seg_<model>.yaml --fold all
python scripts/bench_seg.py
python scripts/collect_seg_results.py --models segformer_b2_anatomy_instrument efficientvit_b1_anatomy_instrument pidnet_s_anatomy_instrument
```

`metrics.json` per fold carries `best` (test-selected upper bound — not quotable), `last`
(decision metrics), per-epoch `history`, and provenance (git SHA + dirty flag, seed, world size).

## Next (PI-endorsed)

Train the multiclass-instruments variant (6 tool groups + background) with SegFormer-B2
(`configs/seg_segformer_b2_multiclass.yaml`). Expect larger per-class fold variance (rare
tools have few frames per fold); keep the per-case breakdown habit. If rare-tool IoU is weak,
the product fallback for Stage-3b kinematics is binary-instrument masks + per-phase instrument
priors from phase recognition.
