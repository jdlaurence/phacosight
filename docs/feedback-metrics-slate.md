# Personalized Feedback for Training Ophthalmologists — Candidate Slate

*2026-08-17. Synthesis of four research tracks (docs/research/): clinical education
rubrics, automated-metrics literature, commercial landscape, SOTA tech. Purpose:
a decision-ready menu of feedback capabilities to prototype. PI-reviewed ranking
appended after review.*

## What the research converged on (the frame for every choice)

1. **The evidence that this product category works is strong**: EyeSi's metric-gated
   feedback is associated with a 38% PCR reduction in UK trainees; video coaching RCTs
   improve OSATS; and the McCannel finding — rhexis-targeted feedback reduced only
   rhexis-related complications — says feedback must be **per-step**, which is exactly
   what a phase-indexed system delivers.
2. **The market gap is real and specific**: simulators have metrics but no real cases;
   ZEISS's Surgery Optimizer has real cases but only phase KPIs and is hardware-locked;
   the general surgical-AI platforms don't serve ophthalmology; PhacoTrainer proved the
   metrics but never became a normed product. Nobody ships anatomy-referenced,
   cohort-normed skill metrics from real cataract video.
3. **The cheapest metrics are the best-validated**: pure eye/anatomy-motion metrics
   (which need only the masks we already produce) posted the *strongest* correlations
   with expert ratings (pupil path r=−0.71) and the best skill classification
   (CatSkill AUC 0.865). Tool-presence-only is a published negative result (AUC 0.55);
   spatial tip kinematics carry the signal (r up to −0.77 on specific rubric items).
4. **Interpretable metric pipelines beat end-to-end learned scores** for now:
   cross-site transfer of learned skill classifiers fails even with adaptation; metric
   pipelines + norms transfer with the flag-rate safety net we've already validated.
5. **LLM narration of structured metrics is ready; VLMs watching raw video are not**
   (structured conditioning doubled clinically-admissible feedback; GPT-4o raw-video
   phase accuracy is 36%). No one has published LLM feedback from cataract kinematics.

## The slate

### Tier 1 — ship-grade foundations (weeks; mostly our existing outputs)

**A. Eye stability & visualization suite** (CatSkill-style). Limbus/pupil path length,
centration vs frame center, Purkinje-1 neutrality, zoom variability (limbus-diameter SD),
eye-out-of-frame time, focus quality — per phase, cohort-normed.
*Evidence*: r=−0.71 vs expert ratings; AUC 0.865 attending-vs-resident; maps to
ICO-OSCAR global items 15–16, 20. *Build*: post-processing of existing masks + a
Purkinje bright-spot detector. *Risk*: low. **Cost: ~1–2 weeks.**

**B. Per-phase instrument-tip kinematics.** Path length, movement count, workspace
area, tip decentration vs pupil center, velocity (flagged noisy), instrument-exchange
counts — limbus-normalized, per phase, cohort-normed. The "EyeSi odometer for real
surgeries."
*Evidence*: PhacoTrainer r=−0.37…−0.77 per OSACSS item; monocular video ≥ hardware
tracking (Hisey). *Build*: mask-skeleton tip seeding + Track-On2 tracker (MIT license,
0.5 GB); validate tip error on ~50 hand-labeled clips first. *Risk*: medium
(tip-localization quality is the known bottleneck). **Cost: ~2–3 weeks incl. validation.**

**C. Narrative formative report (LLM over structured metrics).** 1–2 paragraph
trainer-style debrief per surgery: metrics + cohort percentiles + rubric anchors +
flags in, text out; confidence-gated; attending-reviewable. Local Qwen3-VL/8B
(Apache-2.0) — no PHI leaves the machine.
*Evidence*: structured conditioning doubled admissible generations; automated-feedback
RCT effect exists; radiology analog 87% helpful. First-in-cataract opportunity.
*Build*: prompt engineering + template fallback + admissibility review loop.
*Risk*: medium (needs blinded physician validation before residents see it). **Cost: ~1–2 weeks to pilot.**

**D. Confidence-gated delivery (conformal intervals).** Split conformal intervals on
every metric/percentile from the 1,000-surgery cohort; suppress feedback lines when
intervals are wide; extends our validated flag system to the metric layer.
*Evidence*: conformal precedent in surgical trajectories; nobody has conformal skill
scores (novel). *Risk*: low. **Cost: days.**

**E. Teaching moments + peer comparison.** Rule-based flags (duration outliers vs
norms, unusual transitions, long idle gaps, low-confidence stretches) each linking to
the video moment; side-by-side retrieval of a high-confidence expert clip of the same
phase from the library; cohort tip-trajectory envelopes (median±IQR heatmap) overlaid
on the resident's frame — the honest alternative to generative "expert counterfactuals."
*Build*: on existing library + norms (+B for envelopes). *Risk*: low. **Cost: ~1 week.**

### Tier 2 — one new model / bigger bets

**F. Capsulorhexis morphometrics.** Diameter, circularity, centration, edge smoothness,
grasp/re-grasp counts for the rhexis — the step with the best outcome links (errant
rhexis → vitreous loss; overlap → PCO) and explicit ICO-OSCAR anchors.
*Evidence*: segmentation proven (Dice 86–92; pupil-prompted SAM-LoRA works and we have
pupil masks); geometry formulas published (WetCat); metric→skill link promising but
thin. *Build*: the one justified new segmentation model + geometry code. *Risk*:
medium. **Cost: ~2–4 weeks.**

**G. Rubric-mapped evidence view.** Map each computed metric onto its ICO-OSCAR:phaco
item (16/20 items are video-observable) so faculty see "evidence per rubric line" in
the language programs already use; supports the ICO's score-then-debrief workflow and
trainee self-assessment. *Build*: mapping + UI. *Risk*: low; *caution*: present as
evidence-for-items, never auto-scores, until locally validated. **Cost: ~1 week after A/B.**

**H. Irregularity & complication flags.** Miosis detection (published pipeline, 90.6%
recall), IOL rotation/instability (adopt LensID's haptic-keypoint approach), peaked-pupil
/ pupil-shape risk cues (nearly free from our pupil mask; CataractCompDetect pattern:
geometric risk score → VLM verify). This is Stage 3 of the original plan wearing its
product hat. *Risk*: medium (validation sets small). **Cost: staged, ~3–6 weeks.**

**I. Dexterity/steadiness score** *(the user's earlier idea, now evidence-checked)*:
4–12 Hz spectral tremor power + SPARC smoothness from tip tracks + time. *Evidence*:
strong in other domains, **no cataract validation** — ship only alongside a local
physician-rating validation study; report as "exploratory" until then. **Cost: small
on top of B; validation is the real cost.**

**J. OVFM frozen-probe heads** (ophthalmic video foundation model with skill +
complication heads, prospectively validated in wet-lab). Highest ceiling of any
learned component; license currently unspecified — resolve before touching. **Cost:
~1–2 weeks compute-cheap probe once cleared.**

### Tier 3 — defer or don't build

- **End-to-end learned skill score** from video: cross-site transfer fails; revisit
  later as SAIS-style training on our phase-clipped library, benchmarked on
  Cataract-LMM's 170 rated rhexis clips (CC-BY-NC — research only).
- **Metric 3D trajectories** from monocular microscope video: not defensible.
- **Video-predicted gaze**: unvalidated; behavioral proxies (A) cover it.
- **Diffusion "expert counterfactual" video**: confabulation + medicolegal risk;
  retrieval (E) is the honest substitute.
- **Raw-video VLM feedback**: premature by an order of magnitude on current benchmarks.
- **Video-based PCR anticipation**: unpublished anywhere — a paper opportunity, not a
  product feature.

## Cross-cutting decisions the slate implies

- **External benchmark**: Cataract-LMM (3,000 videos, 2 centers, rated rhexis clips)
  for validating metrics beyond our clinic — also answers the PI's standing
  cross-center probe with a bigger, ophthalmic-specific set.
- **Validation pattern for anything resident-facing**: blinded physician
  admissibility/helpfulness review (the field's bar), plus the SAIS lesson — audit
  scores for cohort bias, ship explanation moments with every score.
- **License spine for the public/commercial path**: Apache/MIT end-to-end is
  achievable (Track-On2, TAPNext, SAM2.1, DA3Mono, Qwen3-VL); avoid CC-BY-NC
  dependencies in the product path.
- **Sequencing logic**: A+D+E are pure wins on existing outputs; B unlocks I and the
  envelopes in E; C turns everything into the actual product experience; F is the
  highest-value new perception; G packages it in educators' language.

---

## PI review disposition (2026-08-17, APPROVE WITH CONCERNS)

**Blocking (both adopted):**
- **B1 — operator attribution**: teaching cases mix hands (attending does the rhexis, takes over when cases degrade); metrics that silently blend operators would credit residents with attending work and vice versa — "worse than no feedback." A per-phase operator field (resident/attending/assisted) at upload, displayed on every metric, un-attributed cases excluded from resident comparisons. Ships before any Tier-1 prototype.
- **B2 — kinematics gate**: the pre-registered temporal identity-stability probe (tool-track identity switches over full videos) runs BEFORE the 50-clip tip validation, both inside the B spike — mask identity flips corrupt tracks upstream of tracker accuracy.

**Corrections adopted:** duration-vs-norms is the strongest replicated global correlate (ρ=−0.77) and already built — day-one reports lead with it; A-before-B is a cost/risk argument, not a validity gap. LLM narration is "ready to pilot behind attending review," not "ready" (best published config: 42% admissible). Ruzicki negative result is AUC 0.55–0.69; EyeSi 38% is an unadjusted association (motivation, never a promised effect); H's evidence is MSICS n=53 (different operation) — faculty-facing only until locally validated. Cohort percentiles are **"vs expert reference," not "vs peers"** — labeled everywhere; resident peer-norms require a separate table keyed off attribution. Efficiency metrics are displayed, never graded (gameable by rushing); tremor/SPARC never ranks residents. Cataract-LMM license contradiction to be resolved at source.

**Recommended first increment (2–4 weeks):** B1 attribution field → A (eye stability suite) → D (conformal gating) → E-minus-envelopes (teaching moments + expert-clip retrieval) → C template-first (LLM variant attending-only) → B de-risk spike (identity probe + tip validation) → M-1 fusion-degradation test. Second: full B → E envelopes + exploratory I; F rhexis model; G packaging. G's "evidence-for-items, never auto-scores" is inviolable.

**The one physician validation study:** blinded concurrent-validity + admissibility review on ~15–20 library surgeries — physician scores ICO-OSCAR blinded to our metrics, rates every feedback line (admissible/helpful/misleading/harmful), adjudicates duration disagreements (absorbs M-3). Yields local metric↔rating correlations, C-vs-template comparison, and the license to quote norms to residents. Nothing resident-facing ships before it. Also clock minutes-per-review (adoption risk).

**Red-team de-risk (run first, GPU-cheap, zero physician time):** reproduce skill discrimination on Cataract-LMM's 170 expert-rated rhexis clips using OUR extraction stack — if our metrics can't separate their low-vs-high clips, stop and fix perception before building feedback UI.
