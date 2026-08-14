# PI Review — Multiclass-Instruments Run & Stage-3b Path

*2026-08-14. All numbers recomputed from disk; per-case, per-frame, and confusion-structure evals run by the reviewer on the idle A40s from `last.pt`. Follows `2026-08-14-stage1-results-review.md`.*

## 1. Verdict

**APPROVE WITH CONCERNS** — the run is methodologically clean and the numbers are trustworthy at the aggregate level, but the quoted headline (0.817) is the test-selected mean, the per-fold std for knife/injector is statistical fiction (4–14 test frames per fold), and the forceps "floor" is really one rare tool (Katena Forceps, IoU 0.400) hiding inside a healthy class. None of this blocks Stage-3b — the confusion structure actively favors committing to multiclass masks.

## 2. Blocking findings

None.

## 3. Concerns (non-blocking)

**C1. Headline "mIoU 0.817" was the test-selected `best` mean — a B1-discipline slip in the summary, not the code.** Last-epoch mean = **0.8136**; pooled = **0.8150**. Every per-class number was correctly last-epoch; only the headline reverted. Quote 0.814 / 0.8150.

**C2. Per-fold mean±std is meaningless for rare classes — report pooled-over-folds.** Test support per fold: knife 4–12 frames, lens injector 11–14, forceps 18–41. The fold-3 knife "dip" (0.670) rests on 4 frames from 3 cases; fold-4 contains a single near-total-miss frame inside an otherwise 0.82–0.93 fold. Training history shows knife converged fine — the estimate is simply low-precision. Pooled: knife 0.8053, injector 0.8050, forceps 0.6859, gauge 0.7393, mIoU 0.8150. Do not react to the fold-3 dip; it does not exist at pooled level.

**C3. The forceps floor is one tool: Katena Forceps.** Per-frame by Supervisely title: Capsulorhexis Forceps mean frame-IoU 0.769 (1/108 frames <0.5) vs Katena Forceps 0.400 (13/28 frames <0.5, six at 0.000). Every catastrophic forceps case is a Katena case. Clinically the right failure to have: the rhexis forceps (gates rhexis-control feedback) is solid; Katena is a brief fixation instrument at incision with ~22 train frames per fold.

**C4. "Rare-instrument oversampling" oversells the code** — weights are instrument-present (1.0) vs not (0.25) only; a Katena frame samples like a spatula frame. Class-aware weights (∝ inverse frequency of rarest class present) are a five-line change if anyone reruns; not worth a rerun now (~22 Katena train frames caps the gain; temporal smoothing at deployment will help more).

**C5. Gauge/cannula/spatula (0.739) is a boundary problem, not identity** — 13.1% of true pixels go to background (thin shafts eroded at 512² aspect-squeezed), 0.4% to other tools. Benign for tip/centroid kinematics; note if shaft-angle features are wanted.

## 4. Big picture

- **Trustworthy?** Yes. Provenance verified (git_sha 4dd462c, git_dirty false, seed 0; N1 `last.pt` present all folds, N2 honored). Instruments-CSV splits re-verified patient-disjoint. Convergence clean (|last − last-5-mean| ≤ 0.004; best−last ≤ 0.010). case5319_112.png drop immaterial.
- **Stage-3b on multiclass masks? Yes — and drop the separate binary fallback; the multiclass model contains it.** (a) Collapsing multiclass predictions to binary gives instrument IoU 0.8133 ≈ Stage-1 anatomy model's 0.8131 — the 7-way head cost nothing in geometry; (b) among pixels instrument in both truth and prediction, only 0.8% carry the wrong tool label. Architecture: geometry from the union of tool channels; identity by per-frame majority vote; phase priors as a consistency check (chiefly Katena at incision, where the model predicts nothing rather than a wrong tool).
- **Knife dip / forceps floor?** Sample-size artifact and a single rare tool (C2, C3). No recipe change; report honestly.
- **Next: Stage 2 (phase recognition) is the critical path.** Three of six product features are blocked only on it, and Stage-3b identity wants its phase priors. Carry the last-epoch/history/provenance discipline and per-class-support reporting into the phase model (idle vs rare phases will replay this exact review). First Stage-3b validation must run on video: temporal identity stability, not per-frame IoU.
- **Falsification probe:** run the model over one full surgery and count identity switches per tool track before building kinematic metrics. Nothing measured today measures that.

## 5. What was done well

- N1/N2 folded in exactly as requested; run launched from a genuinely clean tree at the commit containing the dataset-wart fix — provenance trustworthy end-to-end.
- `load_fold` missing-file handling has the right shape: loud per-row warning, hard failure >50%, no silent skip.
- The mask-dir remapping design meant the multiclass run needed a 20-line config and zero new code — what the simplicity dogma buys.
- Per-class Dice and IoU recorded per epoch; every question answerable from artifacts on disk plus two short GPU scripts.

**Endorsed: quote 0.814 (last-epoch) / 0.8150 (pooled) with pooled per-class numbers and the Katena caveat; commit Stage-3b to multiclass masks (union geometry + majority-vote identity + phase-prior consistency check); proceed to Stage 2.**

## Disposition (same day)

- **C1** → results doc quotes 0.814 / 0.8150; pooled per-class table adopted.
- **C2/C3** → pooled numbers + Katena caveat in `docs/stage1-segmentation-bakeoff.md`; no recipe reaction to the fold-3 dip.
- **C4** → noted for any future segmentation rerun (class-aware weights).
- **C5** → noted; tip/centroid kinematics unaffected.
- Stage-3b architecture recorded (union geometry + majority-vote identity + phase-prior check); identity-switch probe queued for when Stage-2 decode infra exists. Stage 2 is next.
