# PI Review — Stage 2 E0/E1 Phase-Recognition Results

*2026-08-14. Verdict: **APPROVE WITH CONCERNS** — adversarial red-flag investigation clean (no leakage, no eval bug, no label artifact); numbers real, explained by offline protocol on a single-clinic scripted procedure. Concerns are about claims, not validity.*

## Adversarial sweep (all verified with receipts)

- Splits video-disjoint, test folds partition all 56 cases (direct CSV verification).
- No test peeking: val-only selection (`train_phase.py:106-122`); Viterbi grammar from train folds only (`eval_phase.py:49`).
- No label artifact: cached labels recomputed from raw CSVs for 3 cases (incl. case_5063, case_5316) — 0 mismatches; annotations are millisecond-precision so the 0.20 s boundary median is a genuine 5 fps floor.
- Independent recompute: fold-0 val_best.pt rescored on CPU with reference-standard (Farha) metrics — acc 0.9472 / macro-F1 0.9332 / edit 89.91 / F1@50 90.68, identical to the saved metrics.json.
- Metric conventions standard or harder (idle included as a 13th class in segment metrics; the standard eval excludes background).
- Mechanism for beating the bar: bar was calibrated to *online* published results; E1 is offline with full bidirectional context. E0 (0.747 frame acc without temporal context) shows frames aren't trivially separable; +20 pp from temporal modeling is the canonical pattern. Fragmented-10 probe passed as pre-registered.

## Concerns

- **C1.** "Exceeds the bar" is false for duration MAE: Hydrodissection 4.53 s (20.8% of a 21.8 s phase; also worst F1 0.861), phaco 7.0 s (6.4%), idle 6.9 s. Writeup must show the full duration table with relative % and name hydrodissection as the weak phase.
- **C2.** F1@k matching is a slightly generous variant vs the Farha reference — measured effect zero (90.679 identical to 3 decimals); align or footnote before external comparison.
- **C3.** MS-TCN++ over BiGRU: means statistically indistinguishable at n=4 (macro-F1 +0.010, p=0.54); the correct claim is worst-fold robustness + cost. BiGRU's fold-3 dip is a val-selection artifact (its test_last = 0.921).
- **C4.** Val selection leaves ~1 pp on the table (test_last ≥ test_val_best in 5/8 runs); deployment should use the ensemble, sidestepping single-checkpoint selection noise.
- **C5.** Idle is the weakest structural element (gap-idle acc 0.895; duration MAE 6.9 s): state idle-time report lines as ±~7 s.
- **C6.** E0 was fold-0 only; no error bars on it.

## Big picture

Accuracy goal effectively met for offline segmentation on this clinic's distribution; residual timing error is boundary-definition + 5 fps resolution, not architecture. The remaining gap to "impressive" is **trustworthiness and scale**, not accuracy. Ranked effort:
1. Calibration/confidence (ensemble + temperature + disagreement flags) — highest clinical value.
2. Bulk inference on the 944 unlabeled videos for the library + duration cohort norms (product first; self-training a second-order bonus).
3. E3 tool fusion — finish, but expect within-noise phase deltas; value is instrument-usage reporting + hydrodissection/idle boundaries.
4. Paper-protocol clip anchor — required before any external comparison.
5. E2: keep only the 1 fps vs 5 fps ablation (prices the 944-video pass).

Falsification: domain shift untested; cheapest probe is frozen-pipeline timelines on another center's videos (e.g. Cataract-101) before strong product claims.

## What was done well

Pre-registration→execution fidelity verified item by item; codebase economy (<500 lines across the phase package) is what made verification possible; reporting test_last alongside val_best enabled the selection-noise diagnosis.

## Disposition (same day)

- C1/C3 → Stage 2 writeup corrected (full duration table + relative %, hydrodissection named; model choice phrased as robustness/cost).
- C2 → footnoted; strict-reference matching to be used for any external number.
- Proceeding per ranking: confidence ensemble (per-fold seed ensembles for honest eval), clip anchor, E3 fusion review when landed, 1-vs-5 fps ablation, then the 944-video pass.
