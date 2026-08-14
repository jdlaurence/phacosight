# PI Review — Stage 1 Segmentation Bake-off Pipeline (pre-results)

*2026-08-14, reviewed mid-run (SegFormer-B2 folds in flight). Reviewer: PI agent (`.claude/agents/pi.md`).*

## 1. Verdict

**REVISE** — the training runs themselves are sound and should be left running (nothing here requires restarting or wastes the GPU-hours in flight), but the winner-decision procedure must be fixed before any results are relied on: checkpoint/headline selection is currently done on the test folds, the fps half of the winner criterion has no measurement harness, and PIDNet-S as configured cannot support any architecture-level conclusion.

## 2. Blocking findings

**B1. Checkpoint and headline-metric selection on the test fold.**
`scripts/train_seg.py:159-166` — `if metrics["miou"] > best["miou"]` selects `best.pt` and the reported `"best"` block in `metrics.json` by **test-fold mIoU, max over 40 epochs**. There is no validation split; the FiveFold CSVs are train/test only. The log shows epoch-to-epoch mIoU noise of ±0.02 (e.g. fold 0: 0.8848 → 0.8564 → 0.8800 across epochs 19–21), so max-over-epochs inflates each fold's estimate by roughly 0.5–1 pp, and the inflation differs by model (noisier training curves get a bigger max-statistic bonus) — it can flip a close three-way comparison.
*Fix, no rerun needed:* the code already writes `"last"` (final-epoch) metrics to `metrics.json`, and the log contains full per-epoch history. Decide the winner on **last-epoch (or mean-of-last-5-epochs) metrics per fold**, treat `"best"` strictly as an optimistic upper bound, and never quote `"best"` against published anchors. If a selected checkpoint is wanted for deployment, select it on a val subset (the committed 4-fold `TrainIDs_Semantic Segmentation` family has val splits) or just ship the final epoch.

**B2. The fps half of the winner criterion is unmeasurable as things stand.**
The criterion is "best mIoU subject to ≥30 fps FP16 on A40" (docs/experimentation-plan.md:145,176), but no model-throughput benchmark exists — `scripts/bench_decode.py` benchmarks video decode only. `epoch_seconds` in metrics.json is not an fps measurement.
*Fix:* before declaring a winner, add a small benchmark that measures each candidate at **512×512, FP16, batch 1** (the deployment condition — the pipeline is stride-routed single-stream, not batched) on the A40, after training finishes. Without it, "subject to ≥30 fps" is an unverified assumption.

**B3. PIDNet-S is configured so that only one conclusion about it is valid.**
Three compounding handicaps: (a) trains **from scratch** — no ImageNet init (`src/cataract_video/models/segmentation.py:92-93`, comment acknowledges the init "lives on the authors' Google Drive"); (b) built via `get_pred_model(..., augment=False)` (`third_party/pidnet/pidnet.py:218-225`) — the **inference-only variant with no auxiliary P/D heads**, so the boundary-attention training scheme that motivated including PIDNet ("explicit boundary branch attractive for pupil-edge fidelity", plan line 48) is never exercised; (c) AdamW lr 6e-5 is the SegFormer-native recipe — PIDNet's published recipe is SGD lr 1e-2 + OHEM + boundary loss. On 30 videos, a from-scratch CNN under a transformer recipe with its training heads amputated will lose, and that outcome carries **zero information**.
*Fix:* scope the claim — the bake-off may conclude "PIDNet-S *under the shared practical recipe* is not competitive," never "the CNN control loses." If PIDNet lands within fold noise of the leaders despite the handicaps, that's a strong signal warranting one fair rerun (ImageNet init + aux heads + its native recipe). If it loses badly, drop it and say why honestly in the writeup.

## 3. Concerns (non-blocking)

**C1. Background class inflates mIoU vs published anchors.** `SegMetrics` averages over all 5 classes including background (IoU 0.9896 on fold 0), pulling the fold-0 mean to 0.8855; excluding background it is 0.8595. The paper's ≈0.80 and CatSeg's 0.88 anchors must be checked for convention before any "we beat CatSeg" claim — at minimum report both means, and note eval is at 512×512 aspect-squeezed FP16, not native 768×1024 (thin instrument shafts are the class most sensitive to this). Internal three-way comparison is unaffected (same protocol for all).

**C2. Pretraining asymmetry between the two pretrained candidates.** EfficientViT-B1 loads **Cityscapes-segmentation-finetuned** weights (`segmentation.py:72-75`); SegFormer-B2 loads ImageNet-only `nvidia/mit-b2` with a randomly initialized decode head (confirmed by the MISSING-keys load report in bakeoff.log). Dense-prediction pretraining is a real advantage on 30 videos. Note it in the writeup; if EfficientViT wins narrowly, this is a candidate explanation.

**C3. Winner metric should not be raw mIoU.** Downstream, pupil IoU gates miosis detection and **instrument IoU gates the Stage-3b kinematics feature** (tip localization) — and instrument is the known bottleneck class (fold 0: 0.785 vs cornea 0.854, background 0.990). A +0.3 pp mIoU win driven by cornea/background while losing 2 pp on instruments is the wrong pick for the product. Decide on per-class instrument + pupil IoU with mIoU as tiebreaker, and require the winner's margin to exceed fold-to-fold spread (paired per-fold comparison across the 5 folds, not a comparison of two means).

**C4. Unsynced BatchNorm under DDP for the CNN candidates.** PIDNet and EfficientViT use plain `nn.BatchNorm2d` (`third_party/pidnet/pidnet.py:11`); no `convert_sync_batchnorm` in `train_seg.py`. At batch 8/GPU this is a small effect, but it's another asymmetry (SegFormer is nearly BN-free). One-line fix if anything is rerun.

**C5. `--fold all` restarts HF/hub model downloads and mask-weight scans per fold per rank** — cosmetic, but note `instrument_oversample_weights` re-reads ~1,800 masks at each fold start on each rank (`seg_dataset.py:73-90`).

**C6. No git SHA or seed recorded in metrics.json** (config dict is saved in the checkpoint — good; add `git rev-parse HEAD` and the effective seed for provenance).

## 4. Big picture

The bake-off asks approximately the right question. The plan's own analysis is correct that speed is nearly a non-constraint (heaviest use is 5–10 fps phase-gated pupil tracking; all three candidates will clear 30 fps at 512² FP16 on an A40), so **the real decision variable is instrument-class accuracy under 30-video data scarcity** — make that explicit in the decision rule (C3) rather than headline mIoU. Given B3, the honest framing of this stage is "SegFormer-B2 vs EfficientViT-B1, with PIDNet as a recipe-sanity control," and that's fine — the next decision (multi-class instrument variant with the winner) doesn't need a fair PIDNet.

Fold-0 trajectory (0.885 mIoU incl. background ⇒ instrument Dice 0.879, pupil Dice 0.963) sits at or slightly above the published anchors (instruments 77–83 Dice, pupil 94–98) — plausible for a modern backbone vs the paper's VGG-era baselines, not red-flag territory, but part of the margin is the best-on-test selection (B1) and the background/resolution conventions (C1); quote the corrected numbers.

The cheapest high-value follow-up after the winner is chosen: a per-*case* (not per-frame) IoU breakdown on the test folds — with 6 test patients per fold, one bad case can hide inside a healthy pooled confusion matrix, and case-level variance is the honest preview of how this survives a new surgeon or device.

## 5. What was done well

- **Splits are verified clean** — recomputed patient-disjointness independently: all 5 folds have zero train/test case overlap, zero frame-name overlap, test folds partition all 30 cases exactly (24/6 per fold, 2,256 total frames — matches the dataset).
- **DDP evaluation is exact, not approximate** — sharding the test set by `iloc[rank::world]` and all-reducing integer confusion matrices (`train_seg.py:87-90,185-189`) avoids both the classic DistributedSampler-padding duplicate bug and lossy metric averaging. The per-rank `WeightedRandomSampler` with `seed+rank` decorrelation is also correctly reasoned.
- **Mask generation is faithful where it matters** — `scripts/generate_masks.py` reproduces the upstream anatomy+instrument rasterization exactly (same 1024×768 canvas, same Cornea→Pupil→Lens→instruments overwrite order, same 10 tool titles), and its deliberate deviations (fixing the upstream "rev" script's misaligned lowercase multiclass mapping) are documented in the docstring. Spot-checks of masks on disk pass (correct size, IDs ⊆ {0..4}, image pairing).
- **Nearest-neighbor mask resizing, streaming confusion-matrix metrics, and saving both `best` and `last`** — the last of these is precisely what makes fixing B1 free.

## Disposition (actions taken same day)

- **B1** → `scripts/collect_seg_results.py` decides on last-epoch (and mean-of-last-5 where history exists) metrics; `train_seg.py` now records full per-epoch `history` in metrics.json for future runs. "best" is labeled an upper bound.
- **B2** → `scripts/bench_seg.py` measures batch-1 512² FP16 throughput per candidate; to be run on an idle A40 after training.
- **B3** → claim scoped as recommended; fair PIDNet rerun deferred until/unless it lands within fold noise.
- **C3** → decision metrics in `collect_seg_results.py` are instrument IoU + pupil IoU with paired per-fold differences; mIoU (fg and incl-background, C1) reported alongside.
- **C6** → `train_seg.py` records git SHA, seed, world size in metrics.json.
- **C4/C5** → deferred; C4 to be included in any rerun. Mid-run protocol changes were deliberately avoided (logging-only edits).
