# PI Review — Stage 1 Segmentation Bake-off Results (post-hoc)

*2026-08-14. Follows up the pre-results review at `2026-08-14-stage1-pipeline-pre-review.md`. All numbers recomputed from disk by the reviewer, not taken from the summary.*

## 1. Verdict

**APPROVE** — SegFormer-B2 is the Stage 1 winner by a decision-grade margin under the corrected (post-B1/B2/B3/C1/C3) protocol; the reviewer ran the per-case breakdown (30 test cases, both contenders, idle A40) and it confirms the win at case level with no hidden failure case. Proceed to the multiclass-instruments variant.

## 2. Blocking findings

None.

## 3. Concerns (non-blocking)

**N1. No final-epoch checkpoint is saved — only the test-selected `best.pt`.**
`scripts/train_seg.py:161-168` writes only `best.pt` (selected on test-fold mIoU). The *decision* correctly uses last-epoch metrics (B1 honored), but the only *artifact* on disk is the test-max checkpoint, so any qualitative eval, per-case analysis, or checkpoint reuse silently inherits mild test selection. Measured best−last mIoU gaps are ≤0.0051 across all 15 runs (9/15 ≤0.001; best epoch is 36–39 in 12/15) — negligible here, but add a `last.pt` save for all future training.

**N2. PIDNet-rerun provenance records a stale git SHA.**
`runs/pidnet_s_anatomy_instrument/fold*/metrics.json` says `git_sha: a1fbb17`, but the rerun launched at 13:05:37 with the import fix only committed as `f7bca5d` at 13:05:51 — the run started from a dirty tree and the recorded SHA does not contain the code that ran. Harmless this time (import-path-only fix), but provenance should record a dirty-tree flag or it can't be trusted.

**N3. External anchor comparison needs convention-matching before any public claim.**
Last-epoch instrument IoU 0.8131 ⇒ Dice ≈0.897, above the published 77–83 anchor; fg-mIoU 0.8776 ≈ CatSeg's 0.88. Three convention gaps plausibly explain the surplus: (a) pooled-confusion aggregation vs per-image-averaged Dice (systematically lower on instruments); (b) 512² aspect-squeezed eval vs native 768×1024; (c) 2021+ pretrained backbone vs VGG-era baselines. Pupil Dice 0.964 sits inside the expected 94–98 — the class that *should* be easy is exactly as easy as published, the pattern honest numbers show. Do not quote against CatSeg/paper numbers without re-evaluating per-image at native resolution.

**N4. SegFormer runs predate the history/provenance logging** — reviewer reconstructed its 40-epoch history from the training log; no evidence gap, but the asymmetry is worth remembering.

## 4. Big picture — the six questions

1. **Pre-review disposition honored?** Yes — verified in code and artifacts (decision on `last`, `best` labeled unquotable; bench at deployment condition; PIDNet properly scoped; C3 paired decision metrics; C6 provenance modulo N2).
2. **Margin decision-grade?** Yes — paired per-fold 5/5 on instrument (+0.0220, t=6.08, p=0.004) and pupil (+0.0169, t=6.83, p=0.002); per-case: 25/30 instrument (Wilcoxon p=3.9e-7), 28/30 pupil (p=3.5e-8). SegFormer won despite EfficientViT's Cityscapes dense-prediction pretraining advantage, which makes the win more credible.
3. **Convergence?** Clean — |last − mean-of-last-5| mIoU ≤0.0007 (SegFormer), ≤0.0004 (EfficientViT), ≤0.0059 (PIDNet); the polynomial-to-zero LR schedule makes last-epoch a stable estimator; best−last inflation confirmed real but ≤0.005.
4. **Per-case breakdown?** Done by the reviewer (from best.pt; ≤0.005 bounds the bias). SegFormer per-case instrument IoU mean 0.810, min 0.747 (case_5015); pupil mean 0.930, min 0.870 (case_5017). Worst cases shared between models — case difficulty, not model pathology. Nothing further needed before declaring the winner.
5. **Anything smelling wrong?** No — per-class ordering matches surgical anatomy (background ≫ pupil > lens > cornea > instrument); fold spreads consistent with ~450 pooled frames/fold; PIDNet's larger spread is what from-scratch on 24 videos looks like.
6. **Reason not to proceed?** No — proceed to multiclass SegFormer-B2 with three cheap fixes: save `last.pt` (N1); dirty-tree provenance flag (N2); expect much larger per-class fold variance on the 6-way split and keep the per-case habit. If rare-tool IoU is weak, the Stage-3b fallback is binary masks + per-phase instrument priors — a mediocre multiclass result does not block the product path.

## 5. What was done well

- The pre-review disposition was implemented faithfully, not performatively — verified rather than trusted.
- The PIDNet crash was handled correctly: full rerun after the fix, no partial results blended in.
- The bench harness is minimal and honest — the exact deployment condition, synchronized timing, saved as an artifact.
- Last-epoch decisions cost nothing (gaps ≤0.005) — the honest protocol was nearly free, and no asterisk hangs over the winner.

**Endorsed: SegFormer-B2 wins Stage 1. Proceed to the multiclass-instruments variant with the three small fixes folded in.**

## Disposition (actions taken same day)

- **N1** → `train_seg.py` now saves `last.pt` at the final epoch (rank 0).
- **N2** → provenance now records `git_dirty` from `git status --porcelain`; multiclass run launched from a clean tree.
- **N3** → results doc (`docs/stage1-segmentation-bakeoff.md`) marks external comparisons as convention-unmatched; no public claims.
- **N4** → noted; all future runs carry history + provenance.
- Multiclass config: `configs/seg_segformer_b2_multiclass.yaml`; per-case breakdown to be repeated for it.
