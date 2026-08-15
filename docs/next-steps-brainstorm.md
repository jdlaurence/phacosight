# Next steps: raising accuracy and generalizability (brainstorm)

*2026-08-14, post-Stage-2. Current deployment stack: frozen DINOv2-reg-L + tool-presence
fusion, MS-TCN++ ensemble, learned-grammar Viterbi — macro-F1 0.954 / edit 94 in-domain.
Each item lists expected value and rough cost. PI-ranked disposition to follow.*

## A. Raw accuracy (in-domain)

Where the remaining ~5% lives: the viscoelastic↔viscoelastic-suction pair (same cannula —
visually identical, distinguished only by surgical context), idle-gap boundaries, and a
small tail of atypical videos (case_4750-type: 1 of 56 below 0.6 accuracy, flaggable).

1. **Attention pooling over the cached 7×9 patch grids** (cheap; grids were cached for
   exactly this). Mean-pooling dilutes a ~20-patch instrument ~50×; a small learned
   attention pool or a conv stem over the grid gives the head *where* as well as *what*.
   No re-extraction needed. Est. +0.3–0.8 pp, hours of work, minutes of training.
2. **Duration-aware (HSMM) decoding** — replace frame-transition Viterbi with explicit
   per-phase duration distributions (estimated from train folds). Directly targets the
   burst-phase merging and residual boundary placement. Cheap, pure CPU.
3. **Two-pass boundary refinement** — detect transitions at 5 fps, re-decode ±2 s windows
   at 15–30 fps. Breaks the 0.2 s boundary floor where reports need precision; ~free at
   inference (windows are tiny).
4. **Backbone fine-tuning (LoRA or last-2-blocks) on phase supervision** — the biggest
   single accuracy lever left; the frozen features were never told phases exist. Risk:
   overfitting 56 videos (mitigate: heavy augmentation, val-gated, keep the frozen path
   as fallback). ~1–2 GPU-days of experiments.
5. **Self-training on the 940 processed videos** — pseudo-labels already exist with
   validated confidence; retrain heads on high-confidence segments. Cited works suggest
   up to +7 pp in low-data regimes; from our 0.95 base expect +0.5–1 pp, plus better
   calibration on the older-video domain. Cheap (heads train in minutes).
6. **Forward–backward (HMM posterior) confidence** instead of frame-prob-of-decoded-label —
   more principled uncertainty at boundaries; enables confidence-aware decoding. Cheap.
7. **Label adjudication pass** — where the model and GT disagree *consistently across
   seeds/folds* (e.g. gap conventions in burst phases), some disagreements are annotation
   noise. A physician hour spent adjudicating the top-50 disagreement segments both
   cleans eval and may teach us the model is better than measured.
8. **Temporal tool features** — the current 36-dim fusion is per-frame; adding short-window
   tool-presence dynamics (appearance/disappearance edges) encodes instrument exchanges,
   which is what actually defines several boundaries. Small feature change, rerun heads.

## B. Generalizability (other cameras/clinics)

The architecture is already shaped for transfer — a domain-general frozen backbone with a
tiny task head — and the in-house evidence is encouraging (the older 512×324/640×360 era
transferred with a 12% flag rate). But nothing yet tests a truly different device/clinic.

1. **In-house leave-one-era-out validation** (free, do first): treat the recording eras in
   the unlabeled set as pseudo-domains; measure head-ensemble agreement and flag-rate
   deltas across eras. Quantifies "shift sensitivity" without any new labels.
2. **True cross-clinic probe**: CATARACTS (Brest, France — genuinely different device and
   team; public). Note Cataract-101 is from the *same* Klagenfurt clinic as Cataract-1K —
   useful as a same-clinic/different-years probe but it is NOT a cross-clinic test; the
   PI's earlier suggestion should be upgraded accordingly. Run frozen, report flag rates
   and eyeballed timelines; any labels available → real numbers.
3. **Appearance-robust training** (cheap): re-extract features once with test-time-style
   jitter OR train heads on feature-space augmentations; add resolution/compression jitter
   to the tool-feature seg models (they are the most optics-sensitive component — DINO is
   the robust one).
4. **Flag rate as a deployment shift detector** (already validated: r = −0.94 vs accuracy):
   formalize "new clinic onboarding" — run the pipeline on a clinic's first N videos,
   flag-rate distribution tells us whether zero-shot use is safe or adaptation is needed.
5. **Few-shot clinic adaptation protocol**: because only the heads are task-trained,
   adapting to a new clinic = retrain/calibrate heads (minutes) on 5–10 locally annotated
   videos + re-fit temperature. Design and script this as a product feature ("bring-up
   kit"), not research.
6. **Test-time adaptation, lightweight**: per-video feature standardization; optionally
   entropy-minimization steps on the head only (guard-railed by the self-check + flags).
7. **Conformal segment confidence** — distribution-free coverage guarantees on segment
   correctness, calibrated per-deployment; turns "confidence 0.95" into a defensible
   clinical statement.
8. **SSL backbone adaptation** (heavy, later): continue DINO pretraining on a new clinic's
   unlabeled recordings if a partner site has volume; only if 1–6 prove insufficient.

## Suggested sequencing

Quick wins first (A1, A2, A6, B1 — days, mostly CPU), then the two structural bets (A4
backbone fine-tune, A5 self-training) which interact and should be one PI-pre-registered
experiment round, then the outward-facing work (B2 probe, B4/B5 onboarding kit) which is
what converts "impressive in-house" into "trustworthy elsewhere."

---

## PI review disposition (2026-08-14, REVISE — corrections adopted)

**Corrected error budget (recomputed from the full 4-fold confusion matrix; my original
diagnosis was partly wrong):**
- The viscoelastic↔viscoelastic-suction pair accounts for **0 frames** of confusion (fully
  separated by temporal position, before the grammar acts). Viscoelastic's depressed F1 is
  adjacent-boundary leakage, not the same-cannula ambiguity.
- **73.3% of all error frames are idle-involved**; **case_4750 alone is ~20.6%** of the total
  residual (excluding it, the other 55 videos sit at 96.7% frame accuracy).
- Burst-phase convention errors: Tonifying↔AC-Flushing 365 fr; rhexis→hydro 179 fr.

**case_4750 forensics (row 1, done same day):** genuinely atypical recording — heavily
decentered eye, view dominated by conjunctiva/hemorrhage, instruments barely visible.
Not a GT error; a real hard case, correctly flagged at video level (mean conf 0.78 vs
≥0.91 all others). Note: some individual frames are confidently wrong — video-level flags
are the trustworthy unit.

**Killed/deferred:** A3 boundary refinement (0.2 s floor already below every product need);
A4 backbone fine-tuning (targets the wrong residual, invalidates all feature caches,
narrows the domain-general representation; revisit only if an external probe shows a
*featural* failure); B6-entropy-minimization (reinforces confident errors under shift).
A5 self-training reframed as **era adaptation** (pseudo-label filtering removes exactly the
informative segments; measure with leave-one-era-out, not in-domain CV).

**Added:** M-1 fusion graceful degradation (corrupt/zero tool features at eval; tool-dropout
head training; auto-fallback to DINO-only heads — the seg models are the optics-sensitive
component and fusion under shift could be worse than base); M-2 a "report-correct" product
acceptance metric (per-phase totals in tolerance + anchors present/ordered + flags where
needed) evaluated alongside macro-F1; M-3 physician adjudication of top-50 disagreement
segments gates the whole accuracy agenda (much of the 73% idle bucket may be annotation
convention).

**Cross-clinic probe corrected:** Cataract-101 is same-clinic (Klagenfurt) — usable as a
same-clinic/different-era probe only. **CATARACTS (Brest)** is the true external probe;
request access early; freeze the phase mapping before looking at outputs. The zero-label
validation kit (anchor gates, flag-rate detector, self-check) transfers without any label
mapping.

**Ranked next 1–2 weeks (product-critical first):** (1) case_4750 forensics ✓ + physician
adjudication pack; (2) HSMM/min-duration decoding + forward–backward posterior confidence
(one pre-registered round; targets idle MAE ~5 s→3–4 s); (3) leave-one-era-out validation +
onboarding memo; (4) CATARACTS access request now; (5) M-1 fusion degradation test;
(6) attention pooling + tool-edge features (mild score-chasing, <1 GPU-hr); (7) CATARACTS
zero-label probe. Where accuracy stops mattering: in-domain frame-F1 beyond ~0.96 buys
residents nothing — the product-visible residuals are idle timing, honest flag handling,
and external validity.
