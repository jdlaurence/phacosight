# PI pre-registration review — CATARACTS harmonization plan (2026-08-18)

Review of `docs/cataracts-harmonization-plan.md` (draft of same date). Disposition:
**REVISE** → all blocking findings addressed by document edits, recorded in the plan's
header. Verbatim review below.

---

## 1. Verdict

**REVISE** — the label map and strategy choice are sound and the data-facts all check out
against disk, but E7 as written has a train/eval contamination in the generalizability arm
and two unpinned degrees of freedom that defeat the purpose of pre-registering. All fixes
are document edits; nothing needs GPU time to resolve.

**Data-facts audit (all verified):** split sizes 25/5/20 confirmed on disk at
`data/cataracts/ground_truth/CATARACTS_2020/`; the 18 step names and their order (IDs 1–18,
including the "Hydrodissetion" typo) match `evaluation_cataracts2020.py:33-36`; I recomputed
the per-class train histograms from the 25 train CSVs and **all 19 counts in the plan's
table are exact** (train total 494,868 ≈ "≈495k"); Frame columns are contiguous in all 50
CSVs (start at 1, not 0); C1K taxonomy claim matches `src/phacosight/phase/timeline.py:16-30`;
512×384@60 matches `docs/stage2-phase-design.md:50`. The audit discipline here is genuinely
good.

## 2. Blocking findings

**B1. E7-c evaluates E7-b on its own training data.** The plan trains E7-b on "CATARACTS
(all 50 videos, mapped labels)" and then evaluates E7-a vs E7-b "zero-shot on CATARACTS dev
(5 videos)". Those 5 dev videos are inside E7-b's training set, so the comparison is
invalid, the "cross-center robustness" conclusion is rigged in E7-b's favor, and E7-c is one
of the three win conditions in the success bar — the intended "second-center eval" for
publication would be train-on-test. **Fix (removes data, adds nothing):** train E7-b on
CATARACTS train+dev (30 videos) and hold out the official 20-video test split for E7-c.
Cleaner claim ("official CATARACTS test split, never trained on") and 4× the eval videos —
n=5 dev was too thin for a robustness claim anyway (dev has e.g. only 40 Sealing Control
frames; per-class stats on 5 videos are noise).

**B2. The success bar is not a decision rule yet.** (a) "non-inferior" has no margin — with
Stage-2 seed spread of ~0.7pp macro-F1 (0.9489–0.9555, `docs/stage2-phase-results.md:38`),
pin what drop counts as inferior (e.g. seed-mean drop ≤0.3pp and Wilcoxon on the paired
n=56 per-video deltas not significant against E7-b); (b) "better on ≥1 of: macro-F1,
minority-phase F1, E7-c" is three chances at a nominal win with no primary endpoint —
forking paths. **Fix:** designate one primary (E7-c external performance — see Big
picture), the others exploratory, and write the numeric margins into the doc before the
freeze.

**B3. Composite-corpus interactions with class weights and the Viterbi grammar are
unpinned, and both will silently change under the current code.** (a)
`scripts/train_phase.py:47-52` computes `class_weights` as 1/sqrt(train-fold counts) via
`np.bincount` — adding CATARACTS changes every class weight (confounding "more data" with
"different loss weighting"), and an `ignore_index` label will break or corrupt the bincount
as written. (b) `scripts/eval_phase.py:93` estimates the Viterbi transition matrix from
`train_labels`; if CATARACTS sequences enter it, IGNORE holes (suturing, vitrectomy spans
stitched out) create spurious transitions in the grammar the deployed decoder uses. **Fix,
pre-register:** grammar estimated from C1K train folds only (matches deployment; simplest);
class weights computed over the composite corpus excluding `ignore_index` frames (state
this explicitly, and log the before/after weight vector so the confound is at least
visible).

## 3. Concerns (non-blocking)

**C1. IGNORE arithmetic inconsistent with the table.** "~33k train frames (~6.7%)" is
{1,2,9,16,17} and omits ID 11 (Preparing Implant, 12,222 frames), which the table marks
IGNORE as primary. With 11 included: six steps, 45,226 frames, 9.1%. Fix the numbers to
match whichever map is frozen.

**C2. "Reuse the exact Stage-2 preprocessing" (R4) is ill-defined across aspect ratios.**
C1K is 4:3 (512×384); CATARACTS is 16:9 (1920×1080). Both the DINOv2 feature extraction
(518×392) and the zero-shot seg tool features need a pre-registered crop/letterbox policy —
center-crop to 4:3 vs squash changes what the model sees. Decide before extraction, not
after. The inventory pass should also check for UI overlays/vignetting in the Brest
recordings.

**C3. Absent-class aliasing, not just imbalance.** The `Capsule Pulishing` /
`Anterior_Chamber Flushing` risk is framed as "imbalance shift", but the mechanism is
worse: CATARACTS has no polishing class, so polishing-looking activity (same I/A handpiece)
is labeled `Irrigation/Aspiration` in the mapped corpus — direct "same visuals, different
label" conflict that actively trains against C1K's polishing class. Add both classes to the
falsification probe alongside the merge targets, with a pre-registered fallback (Strategy C
or IGNORE-ing the conflicting spans) if their F1 regresses.

**C4. Idle decision rule should be pre-registered, not post hoc.** CATARACTS idle is 27% of
train frames (133,957) and idle is already Stage-2's weakest output (±6 s/video,
`docs/stage2-phase-results.md:60`). R1 names the risk; write the rule now: e.g. "if >X% of
sampled CATARACTS idle frames show an instrument in the eye, map CATARACTS idle → IGNORE."

**C5. Vitrectomy videos are complicated cases end-to-end.** IGNORE-ing the vitrectomy span
(correct call) doesn't make the rest of those videos typical — post-rupture I/A and
implantation look different. Record which videos contain step 9 and consider an ablation
excluding them if merge-target diagnostics look off.

**C6. Minor wording:** "fold-0 `last.pt`" is imprecise — the deployment convention is
fold-0 with multiclass `last.pt` but anatomy `best.pt` (`scripts/extract_tool_features.py:7-8`,
`scripts/analyze_phase_bulk.py:55`).

## 4. Big picture

Right question, right strategy, but the endpoint ordering is inverted. C1K in-domain
headroom is essentially gone — macro-F1 0.954, boundary error at the sampling floor — so
E7-b beating E7-a on C1K metrics is neither likely nor the point. The value of CATARACTS is
exactly the open falsification Stage-2 itself recorded (`docs/stage2-phase-results.md:75-77`:
"frozen pipeline on another center's videos before strong product claims"). Make external
performance on the held-out CATARACTS test split the primary endpoint and C1K
non-inferiority the guardrail, not the other way around. And note the cheapest informative
experiment costs zero training: once features are extracted, run the frozen E7-a stack
zero-shot on CATARACTS test with mapped labels. That number is required as the E7-c
baseline anyway, and it sizes the domain gap before any augmentation training is committed
— if the frozen model already transfers well, the augmentation argument changes; if it
collapses, you learn which classes break. The label map itself is well-judged: the four
unambiguous IGNOREs are correct calls (Vitrectomy especially — 13,860 frames mislabeled as
anything would be poison), and IGNORE-over-wrong-merge as the stated tiebreak (R2) is the
right prior. Strategy A over B/C is right for one added dataset.

## 5. What was done well

- The data-facts section is fully accurate — every one of the 19 train-frame counts, split
  sizes, and step names matched an independent recount from disk.
- IGNORE-as-masked-in-loss while keeping sequences contiguous is the correct mechanic for
  temporal models; many teams get this wrong by dropping frames.
- Refusing to extract or train before the video inventory pass (given the garbled S3
  manifest) and before this review is the right discipline, and R5's license-before-artifact
  gating matches the repo convention.
