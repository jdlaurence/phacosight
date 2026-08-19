# E7 — CATARACTS augmentation: results

*2026-08-18. Pre-registration: `docs/cataracts-harmonization-plan.md` (PI-reviewed twice:
pre-registration REVISE→applied; E7-0 APPROVE W/ CONCERNS→applied). Status: primary
endpoint and guardrail decided; falsification-probe fallback (v2) in progress; **no
deployment-stack adoption yet** — pending v2, calibration re-fit, and PI results review.*

## Headline

| Arm | CATARACTS test (viterbi) | C1K guardrail (viterbi, seed-mean) |
|---|---|---|
| E7-0 frozen stack (zero-shot) | acc 0.778, **macro-F1 0.678**, edit 71.3, F1@50 59.9 | macro-F1 0.9388, edit 91.0, F1@50 91.9 (baseline seeds) |
| E7-b augmented (3-seed ensemble) | acc 0.891, **macro-F1 0.880**, edit 87.9, F1@50 84.0 | macro-F1 0.9419, edit 90.8, F1@50 91.7 |

- **Primary endpoint (pre-registered): PASS, decisively.** Paired per-video (n=20, all
  three metrics, viterbi): improvement on **20/20 videos** for macro-F1 (+0.212 mean),
  edit (+16.6), and F1@50 (+24.4); Wilcoxon p < 1e-4 each; aggregate macro-F1 gain
  **+20.2pp** against the 0.7pp bar.
- **C1K non-inferiority guardrail: PASS.** Seed-mean macro-F1 *improves* (+0.31pp,
  margin allowed a 0.30pp drop); paired Wilcoxon (n=56, per-video seed-means) finds no
  significant harm on macro-F1 (p=0.83), edit (p=0.39), or F1@50 (p=0.39); boundary
  median unchanged (0.0 s), per-phase duration MAE improves 2.25→2.07 s.
- **The gains land exactly where the E7-0 diagnosis demanded.** Zero-shot collapse was
  idle-attraction on short instrument-defined phases; augmentation rebuilt precisely
  those classes on CATARACTS test: Hydrodissection 0.37→0.80, Viscoelastic 0.42→0.78,
  Incision 0.51→0.80, Capsulorhexis 0.68→0.92, Lens Implantation 0.61→0.88, Lens
  positioning 0.66→0.95.

## Falsification probe: FIRED (as pre-registered), fallback running

- **Capsule Pulishing (C1K, absent-source class): 0.9273 → 0.9131 (−1.42pp)**, beyond
  the 0.7pp seed-spread threshold and consistent across seeds (base seeds 0.9258–0.9291).
  The other absent-source class, Anterior_Chamber Flushing, *improved* +2.45pp; all
  merge-target classes improved. Product-scale context (corrected per PI results review
  #1 — the confusion matrices are 3-seed sums): ≈29 polishing-recall frames **per
  deployed model** ≈ 29 s total across all 56 C1K videos at 1 fps (~0.5 s/video),
  flowing to idle — i.e. a slight polishing-duration under-report, dwarfed by the
  2.07 s per-phase duration MAE.
- **Confusion trace (new `eval_phase` confusion export):** the regression does **not**
  flow through I/A (polishing→I/A misses fell 9→4). True-polishing misses grew as
  **idle (110→165)** and **Viscoelastic (15→42)** — the bidirectional OVD-cannula ↔
  polishing-cannula aliasing the E7-0 review predicted (its zero-shot mirror: 32% of
  true Viscoelastic predicted as polishing).
- **Fallback applied per pre-registration:** of the two registered targets (I/A,
  Viscodilatation), the trace selected **Viscodilatation → IGNORE** (v2; implemented as
  `aug_label_overrides` remapping cached `steps_raw` at load time — no re-extraction);
  3 seeds × 4 folds retrained, all evals re-run.

### v2 outcome: fallback FAILED — and falsified the aliasing hypothesis

| Arm | C1K macro-F1 | C1K polishing F1 | CATARACTS test macro-F1 / edit / F1@50 |
|---|---|---|---|
| base | 0.9388 | 0.9273 | 0.678 / 71.3 / 59.9 (zero-shot) |
| E7-b (aug) | **0.9419** | 0.9131 | **0.880 / 87.9 / 84.0** |
| v2 (Viscodilatation→IGNORE) | 0.9388 | **0.9055** | 0.808 / 74.8 / 72.2 |

- Polishing did **not** recover under v2 — it worsened (0.9131 → 0.9055). Refinement
  (PI results review #2): the Viscodilatation aliasing itself was *real* — v2 reverted
  the targeted polishing→Viscoelastic confusion cell (42 → 12) — but removing those
  12k frames raised idle's share of the augmentation supervision, and polishing→idle
  misses grew further (165 → 208), net negative. **Idle-attraction is the carrier** of
  the regression; span-label surgery on action phases cannot fix it.
- v2 pays a large external price (−7.3pp macro-F1, −13.1 edit vs E7-b; still beats
  E7-0, 18/20 videos) and loses E7-b's C1K gain. **v2 is strictly dominated by E7-b.**
- Remaining explanation for the polishing regression: the idle-attraction term
  (110→165 missed-as-idle) and/or generic representation shift from composite
  training — neither has a pre-registered remedy.

**Decision now with PI (pre-registered chain reached "escalate to Strategy C", but its
premise — span-label aliasing — was falsified by v2):**
(a) **Adopt E7-b as-is** — guardrail passed with a net C1K *gain*; the probe caught a
real −1.42pp regression on one small class (~87 s of recall across 56 videos) that
map surgery demonstrably cannot fix; document and monitor.
(b) **v3: CATARACTS idle → IGNORE** — the direct test of the remaining idle-attraction
mechanism; discards 27% of augmentation frames; ~40 min to train; requires a
pre-registration amendment since it is outside the registered fallback chain.
(c) **Strategy C (per-dataset heads)** — the registered escalation, but architecturally
heavy against a falsified premise.

## Incident note (evaluation infrastructure)

The first C1K guardrail pass was **invalid and discarded**: `eval_phase.py` never
applied `frame_stride`, feeding 5 fps sequences to 1 fps heads (baseline seeds scored
macro-F1 0.50 vs their true 0.94) — the 2026-08-14 incident class resurfacing in the
eval path. Fixed by reading stride + metric fps from each fold's checkpoint config;
fixed harness reproduces baseline expectations (seed0 viterbi macro-F1 0.9389, edit
90.8). The CATARACTS harness was never affected (it asserts head rate). Per-video
macro-F1/F1@50 conventions are now shared between both harnesses via `phase/metrics.py`.

## Protocol notes

- All CATARACTS-side segmental numbers use IGNORE-splicing and are internal to the
  E7-0↔E7-c pairing — never comparable to CATARACTS-2020 official baselines.
- Both external arms decode with the committed deployment grammar
  (`app/assets/transition_matrix_1fps.npy`) at T=1.04 for symmetry; the augmented
  ensemble's own calibration (Stage-2 confidence pipeline re-fit) is a pending step
  before any deployment.
- Artifacts: `runs/e70_zeroshot/results.json`, `runs/e7c_augmented/results.json`
  (per-video paired fields, confusion matrices, provenance), guardrail eval JSONs
  regenerated with confusion export.

## PI results review (APPROVE W/ CONCERNS) — adoption and follow-ups

Full review: `docs/reviews/2026-08-18-e7-results-review.md`. Rulings: **E7-b adopted**;
v3 (idle→IGNORE) authorized as a NON-gating mechanism probe under a pre-committed
supersession rule (see plan amendments); notools waived / novit deferred (formal
amendments, recorded in the plan); Strategy C rejected (falsified premise).

**case_4750 inspection (PI #3, done):** the seed-consistent −10pp per-video macro-F1 is
a small-class artifact on the corpus's known-pathological video (prior forensics:
decentered, hemorrhage-dominated, instruments barely visible). Under E7-b its frame
accuracy *improves* (0.569→0.593; Viscoelastic recall 0.40→0.67, Phaco 0.12→0.31), but
Hydrodissection (38 frames) and Capsule Pulishing (15 frames) collapse to zero recall —
two zeroed small classes dominate a per-video macro over present classes. ~50 frames
total; the video remains correctly confidence-flagged at video level (the trustworthy
unit per the Stage-2 forensics). No new mechanism: the same idle-attraction +
polishing→Viscoelastic pattern on the hardest eye.

**Monitoring plan (per PI #5):** per-class Capsule Pulishing F1 reported on every future
eval; polishing-adjacent idle spans spot-checked in generated reports at the next bulk
pass; case_4750 tracked as the canonical hard-case sentinel.

## Pending before deployment-stack promotion

1. ~~v3 mechanism probe~~ done — **no supersession; E7-b stands** (rule applied
   mechanically). v3 (CATARACTS idle→IGNORE): C1K polishing recovered to 0.9207
   (needed ≥0.9258; base 0.9273) — **confirming idle-attraction as the carrier**, with
   a −0.66pp residual near seed spread — but the external endpoint collapsed
   (macro-F1 0.664, edit 43.4; CATARACTS idle F1 0.012): masking a center's idle
   supervision makes its timelines undecodable. Mechanism explained → the deferred
   novit ablation is waived (its trigger condition — v3 failing to explain — not met).
   Final accounting: E7-b's −1.42pp polishing cost is the price of learning CATARACTS
   idle, which is also what bought the +20.2pp external gain; monitored per plan.
2. Calibration re-fit for the adopted ensemble (temperature, ECE, risk–coverage vs
   Stage-2) — the app's confidence flags must not ship under the base temperature.
3. Bulk-pass re-run + cohort-norms rebuild with the promoted ensemble (model-version
   consistency: analyses must not be compared against old-model norms).
4. Deployment self-check updated to the promoted checkpoints; grammar asset stays the
   committed C1K-only `transition_matrix_1fps.npy` (no CATARACTS sequences — correct).
5. ~~R5 license closure~~ done: CATARACTS citation block (Al Hajj et al. MedIA 2019 +
   IEEE DataPort doi:10.21227/ac97-8m18) in README/CLAUDE.md.
6. ~~case_4750 inspection~~ done (above); monitoring plan recorded.
