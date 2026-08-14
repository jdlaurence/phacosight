# PI Review — Stage 2 Round 2 (E3 fusion, confidence, clip anchor, 1 fps)

*2026-08-14. Verdict: **APPROVE** — all numbers reproduce; fusion gain real and seed-robust; pre-registered bar met on every criterion except idle/phaco duration MAE (reported as such). Reviewer ran three falsification ablations on idle GPUs: tools seeds 1/2, multiclass-only tool features, tools@1fps. Stage 2 cleared for write-up; bulk pass approved (fusion @ 1 fps).*

## Concerns (all non-blocking; disposition at bottom)

- **C1.** Headline fusion comparison was seed0-vs-seed0 (+1.75 pp); honest seed-mean gain is **+1.31 pp** (base 3-seed mean 0.9403; reviewer's tools seeds: 0.9588, 0.9489 → tools mean 0.9534). Effect unambiguously real: every tools seed beats every base seed; paired per-video stats reproduced exactly. → run tools seeds 1/2 officially, quote seed-means.
- **C2.** Attribution corrected: gain is (a) **Hydrodissection +9.3 pp** (0.861→0.954) via the multiclass hydro-cannula signal, plus (b) training stabilization (base runs have catastrophic hydro folds, gone under fusion). Viscoelastic did NOT move (same instrument in both OVD phases — fusion can't separate them; now the weakest action class). Base 3-seed ensemble alone recovers hydro to 0.945, but tools single (0.954) still beats base ensemble (0.948) — fusion is not ensembling in disguise. Duration MAE: hydro 4.53→1.80 s, phaco 7.0→3.2 s.
- **C3.** Leakage clean: checkpoint map re-verified defect-free; the anatomy `best.pt` caveat empirically de-fanged by ablation — **multiclass-only features (fully clean) reach 0.9489** vs 0.9349 base. Keep footnote + cite ablation.
- **C4.** Confidence stack sound (within-fold-only ensembles; T on val ensemble log-probs, identical parameterization at test). Notes: frame confidence is the probability of the Viterbi-decoded label, not an HMM posterior (state it; forward–backward marginals later); also report purity≥0.8; patch `confidence_phase.load_case` for extra features before evaluating the tools deployment stack.
- **C5.** **Deployment recipe: ship the fusion model** — the product needs the seg pass anyway, so tool features are free at inference. Stack: tools fusion × 3-seed ensemble + val temperature + Viterbi + disagreement flags (~0.955–0.96 expected).
- **C6.** Clip anchor reproduced to the digit; keep labeled anchor-not-head-to-head. Reviewer's tools@1fps: **0.9431 ± 0.0096** — fusion at 1/5 cost beats DINO-only at 5 fps (0.9403). Bulk-pass pricing answered.

## Big picture

"Solid, verging on impressive": **yes** — fusion+Viterbi = edit 94.1 / F1@50 94.5 / seg-ratio 0.99 / boundary 0.20 s / macro-F1 0.954 against the pre-registered bar. Remaining gap is **external validity** (single clinic/device); the Cataract-101 frozen-pipeline probe stays in limitations. Bulk pass on the 944 videos: launch now, fusion @ 1 fps, full 4-fold × 3-seed ensemble (legitimate on unlabeled data), confidence attached from day one. Stage 3: go; resist further ablations.

## Disposition (same day)

- C1 → official tools seeds 1/2 launched; writeup quotes seed-means.
- C4(iii) → `confidence_phase.py` patched for extra features; deployment-stack confidence rerun after seeds.
- Writeup checklist adopted in full (results section of `docs/stage2-phase-results.md`).
- Bulk pass queued with the C5 deployment stack.
