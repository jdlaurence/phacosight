# PI review — E7 results / adoption decision (2026-08-18)

Review of `docs/e7-cataracts-augmentation-results.md` and all E7 artifacts.
Disposition: **APPROVE W/ CONCERNS — adopt E7-b (option a)**; v3 authorized non-gating;
all concerns addressed same day (see results doc + plan amendments). Verbatim below.

---

## 1. Verdict

**APPROVE WITH CONCERNS** — and on the decision, **(a): adopt E7-b as-is**, with v3
authorized as a *non-gating* pre-registered amendment. The primary endpoint, guardrail,
probe, and v2 comparison all reproduce exactly from the artifacts; the remaining
findings are documentation, licensing, and one per-video regression to inspect — none
invalidates the result.

## 2. Verification performed (all reproduced)

- **Primary endpoint** (viterbi, n=20): macro-F1 delta +0.2124 mean, **20/20 videos
  improved**, Wilcoxon p=9.5e-7; edit +16.61, 20/20, p=4.4e-5; F1@50 +24.43, 20/20,
  p=9.5e-7. Aggregate 0.678→0.880. The gain exceeds the 0.7pp bar by ~29×.
- **Guardrail** (viterbi): seed-mean macro-F1 0.9388→0.9419 (+0.31pp, margin allowed
  −0.30pp); paired Wilcoxon (n=56): p=0.83/0.39/0.39 — no significant harm. Duration
  MAE 2.25→2.07 s confirmed.
- **Probe**: Capsule Pulishing 0.9273→0.9131 (−1.42pp), seed-consistent (aug max
  0.9225 < base min 0.9258). Flushing +2.45pp. Confusion rows confirmed: idle 110→165,
  Viscoelastic 15→42, I/A 9→4.
- **v2 domination**: C1K 0.9388, polishing 0.9055, external 0.808/74.8/72.2; per-video,
  v2 beats E7-b on ≤1/20 videos on every metric. Strict domination confirmed.
- **No leakage**: aug_cases = exactly the 30 train+dev videos in every fold; test never
  trained on; selection on val only; grammars C1K-only; class weights exclude ignore
  with before/after logged; the frozen map matches the pre-registration table.
- **Incident fix**: `eval_phase.py` now reads `frame_stride` from checkpoint configs;
  base seed0 reproduces 0.9389/90.8 as claimed.

## 3. Blocking findings

None. The result stands.

## 4. Concerns (non-blocking)

1. **"87 s across 56 videos" is 3× overstated** — the confusion matrices are 3-seed
   *sums*; per deployed model it is ~29 recall-frames ≈ 29 s (~0.5 s/video). Fix the
   doc sentence.
2. **"Aliasing premise falsified" is half-right.** Under v2 the targeted
   polishing→Viscoelastic cell *reverted* (42→12) — the Viscodilatation aliasing was
   real — but polishing→idle grew further (165→208), net worse: removing 12k
   Viscodilatation frames raised idle's share of augmentation supervision. Affirmative
   evidence that **idle-attraction is the carrier**; record the refinement — it is why
   v3 is well-motivated as a mechanism probe.
3. **case_4750 regresses −10.0pp seed-mean per-video macro-F1 (0.652→0.552),
   consistent across seeds** — largest per-video move either direction, on the hardest
   C1K video, unmentioned in the doc. Inside a passed guardrail (33/56 videos improve),
   but deserves a 20-minute look (which phases, which confusions) before promotion.
4. **R5 license closure is overdue** — CATARACTS license terms + Al Hajj et al.
   citation were required "before first trained artifact uses the data"; trained
   artifacts exist. Close now.
5. **Minor**: `eval_phase.py` builds metric aggregators from fold 0's `eff_fps` and
   reuses them — correct today, silently wrong if folds ever diverge; add an assert.

## 5. Rulings

**(2) Adoption: option (a).** Guardrail passed with margin; the probe caught a real but
small regression (~29 s of recall per model corpus-wide, flowing to idle — a slight
polishing-duration under-report, dwarfed by the 2.07 s duration MAE) that map surgery
demonstrably cannot fix. Against that: +20.2pp external macro-F1 on 20/20 videos —
exactly the domain-shift robustness the product needs. **Strategy C rejected**: its
premise (conflicting label semantics) is what v2 disproved; per-dataset heads would be
accidental complexity. **v3 authorized but must not gate adoption**: pre-register the
amendment with the decision rule written before results (supersede E7-b only if
polishing recovers to within base seed spread AND external macro-F1 within 0.7pp of
E7-b AND the guardrail passes). Run it during the calibration re-fit.

**(3) Conditional ablations: formally amend, don't silently drop.** notools — **waive**
(its target is moot: external endpoint passed by 20pp; the C1K-side regression cannot
be tool-feature-driven — in-domain features, polishing→I/A decreased). novit —
**defer behind v3** (second-order suspect: complication videos' unusual idle frames).
Record both dispositions as written amendments.

**(4) Before deployment-stack promotion:**
1. Calibration re-fit (temperature, ECE, risk–coverage) — do not ship the aug ensemble
   under the base temperature.
2. Model-version consistency: re-run the bulk pass with the promoted ensemble and
   rebuild cohort norms — new analyses must not be compared against old-model norms.
3. Deployment self-check updated to the promoted checkpoints; grammar asset stays the
   committed C1K-only `transition_matrix_1fps.npy`.
4. Close R5 (concern 4).
5. Fix the doc arithmetic (concern 1), add the v2 refinement (concern 2), document the
   polishing regression and case_4750 with a monitoring plan (per-class polishing F1 on
   every future eval; spot-check polishing-adjacent idle spans in generated reports).
6. Inspect case_4750 (concern 3).

## 6. What was done well

- Pre-registration discipline held under pressure: the fallback was run even though
  E7-b already looked adoptable, its failure reported at full volume, v2's domination
  stated plainly rather than buried. Exactly how a falsification probe should be used.
- The eval-infrastructure incident was handled correctly: invalid pass discarded, root
  cause named, fix verified against known baselines, blast radius documented.
- `train_phase.py`'s aug integration is admirably minimal (~25 lines; labels remapped
  from cached `steps_raw` so map amendments never re-extract; weight confound logged).
