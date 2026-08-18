# CATARACTS → PhacoSight label harmonization (pre-registered)

*2026-08-18. PI pre-registration review: REVISE → revisions applied (see
`docs/reviews/2026-08-18-cataracts-preregistration.md`); cleared to E7-prep (inventory +
feature extraction). The label map itself stays unfrozen until the sampled-clip decisions
(R2, idle rule in R1) are made and recorded here. First augmentation dataset for the
composite corpus (see README Data section).*

**Revisions from PI review (all applied):**
- **B1:** E7-b trains on CATARACTS train+dev (30 videos); the official 20-video test split
  is held out end-to-end and carries the primary endpoint (was: train on all 50, eval on 5
  dev — train/eval contamination).
- **B2:** primary endpoint designated (external CATARACTS-test performance), C1K
  non-inferiority is the guardrail with numeric margins; everything else exploratory.
- **B3:** Viterbi grammar pinned to C1K train folds only; class weights computed over the
  composite excluding `ignore_index`, with before/after weight vectors logged.
- **C1–C6:** IGNORE arithmetic corrected (six steps, 9.1%); 4:3 center-crop policy
  pre-registered; absent-class aliasing added to the falsification probe with a
  pre-registered fallback; idle decision rule pre-registered; vitrectomy-video ablation
  noted; tool-checkpoint wording fixed.

## Data facts (audited 2026-08-18; re-audit video properties after extraction)

- **Source:** CATARACTS (Al Hajj et al., Brest University Hospital), IEEE DataPort open
  access. 50 microscope videos (a parallel `videos/tray/` instrument-tray view exists in
  the archive; unused). **Inventory audited on disk 2026-08-18:** all 50 videos
  1920×1080; fps is *variable and non-integer* — ≈29.17–29.20 fps in four distinct exact
  rationals (7500/257, 90000/3083, 45000/1541, 18000/617) — so all time↔frame math must
  use each video's own fps (the extractor already does). **GT↔video frame counts match
  exactly for all 50 pairs.** Naming: 2020 dev/test splits keep 2017 filenames — dev =
  {test01, test07, test14, test16, test19}; test\_gt = the other twenty `test*` videos.
  Vitrectomy (step 9) videos, recorded per C5: train19 (train; 40-min case, 71,066
  frames), test07 (dev), test02 (test; 55,711 frames) — the complicated cases are also
  the longest. The stray procfs tree in the archive was excluded at extraction.
- **Phase GT (CATARACTS_2020):** per-frame `Frame,Steps` CSVs at native frame rate,
  50 videos = 25 train / 5 dev / 20 test, **test GT included** → all 50 usable.
  Frame counts sum to ≈ full-length videos (≈495k labeled frames in train alone).
- **Taxonomy:** label 0 = Idle + 18 steps (IDs 1–18 index the `steps` list in
  `evaluation_cataracts2020.py`).
- **Also downloaded:** CATARACTS_2017 tool-usage GT (21 tool classes, 50 videos) — not part
  of this plan, but relevant later as weak supervision for tool-feature alignment.
- **Cataract-1K reference taxonomy:** 13 classes = idle + 12 phases, pinned in
  `src/phacosight/phase/timeline.py::PHASES`. C1K phase videos are 512×384 @ ~60 fps
  (Stage 2 audit) — CATARACTS is higher-resolution, different center/optics: real domain
  shift, useful for generalizability claims.

## Proposed label map (CATARACTS step → C1K phase)

| ID | CATARACTS step | train frames | → C1K class | confidence / notes |
|---:|---|---:|---|---|
| 0 | Idle | 133,957 | `idle` | High — both defined as "no active step"; check instrument-visibility convention on sampled frames |
| 1 | Toric Marking | 895 | **IGNORE** | No C1K equivalent (toric IOL alignment marking) |
| 2 | Implant Ejection | 2,834 | **IGNORE** (alt: Lens Implantation) | Semantics unclear vs Preparing Implant/Implantation — decide after watching sampled clips |
| 3 | Incision | 18,451 | `Incision` | High |
| 4 | Viscodilatation | 12,324 | `Viscoelastic` | High (OVD injection) |
| 5 | Capsulorhexis | 30,398 | `Capsulorhexis` | High |
| 6 | Hydrodissetion [sic] | 10,423 | `Hydrodissection` | High |
| 7 | Nucleus Breaking | 31,628 | `Phacoemulsification` | Medium — C1K folds chopping into phaco; merging loses no C1K-side semantics |
| 8 | Phacoemulsification | 49,026 | `Phacoemulsification` | High |
| 9 | Vitrectomy | 13,860 | **IGNORE** | Complication management; no C1K phase; sizable, do not mislabel |
| 10 | Irrigation/Aspiration | 49,062 | `Irrigation/Aspiration` | High |
| 11 | Preparing Implant | 12,222 | **IGNORE** (alt: `idle`) | Off-eye activity; C1K would likely call this idle — verify on sampled frames before choosing |
| 12 | Manual Aspiration | 11,068 | **IGNORE** (amended at freeze) | Sampled clips: occurs *only* in the two vitrectomy videos (train19, test07), single manual cannula unlike the C1K I/A handpiece — complication-correlated, unsafe merge |
| 13 | Implantation | 9,576 | `Lens Implantation` | High |
| 14 | Positioning | 27,143 | `Lens positioning` | High |
| 15 | OVD Aspiration | 31,098 | `Viscoelastic_Suction` | High |
| 16 | Suturing | 10,328 | **IGNORE** | No C1K equivalent |
| 17 | Sealing Control | 5,087 | **IGNORE** | Wound-seal check; no C1K phase |
| 18 | Wound Hydratation | 35,488 | `Tonifying/Antibiotics` | Medium — stromal hydration ≈ C1K tonifying; C1K class also covers antibiotics |

C1K classes with **no CATARACTS source**: `Capsule Pulishing`, `Anterior_Chamber Flushing`
— CATARACTS augmentation contributes zero positives for these; watch their per-class metrics
for regression (imbalance shift).

**IGNORE mechanics:** ignored frames get `ignore_index` in the loss (never a wrong label,
never dropped from the timeline — temporal models need contiguous sequences). Frozen map:
seven IGNORE steps (IDs 1, 2, 9, 11, 12, 16, 17) totaling 56,294 train frames (11.4%).

## Map freeze (2026-08-18) — sampled-clip review results

Review materials: seeded (0) contact sheets from **train+dev videos only** (test never
sampled): 200 stratified idle frames + 12 frames per ambiguous step. Decisions:

- **Idle rule (R1): idle → idle stands.** 4 clear instrument-in-eye frames + 3 borderline
  of 200 (2–3.5%) — under the pre-registered 10% threshold. Incidental: rare black/
  camera-pan frames and a keratoscope-like disc rested on the cornea between steps
  (Brest practice) occur within idle spans; all consistent with "no active step".
- **ID 2 (Implant Ejection) → IGNORE confirmed.** All sampled frames are from the first
  ~3 minutes of surgery (84–184 s; only 3 train+dev videos have the step) showing fine
  hooks at the limbus — visually incompatible with the `Lens Implantation` alternate;
  mapping it there would have injected early-surgery frames into a late phase.
- **ID 11 (Preparing Implant) → IGNORE confirmed.** Many frames are full-frame off-eye
  closeups (gloved hands loading the IOL cartridge); C1K idle keeps the camera on the
  eye, so the `idle` alternate would have polluted idle with out-of-distribution frames.
- **ID 12 (Manual Aspiration) → IGNORE, amended from `Irrigation/Aspiration`.** Occurs
  exclusively in the two vitrectomy (complication) videos, using a single manual cannula
  visually unlike the C1K I/A handpiece. Complication-correlated appearance + atypical
  instrument = unsafe merge; IGNORE-over-wrong-merge tiebreak applied.
- **ID 18 (Wound Hydratation) → `Tonifying/Antibiotics` confirmed.** Sampled frames show
  cannula-at-incision stromal hydration late in surgery with the IOL in place — the
  expected match.

## Strategy options

- **A. Map-to-C1K (recommended).** Convert CATARACTS labels to the 13-class C1K taxonomy
  with IGNORE masking; train the existing winning recipe on C1K+CATARACTS; the deployed
  13-class head, app, reports, and cohort norms are untouched. Cheapest, directly tests
  "does more data help the product model."
- **B. Union taxonomy (~22 classes).** Retrains everything; breaks app/report/norms
  semantics; C1K videos would have zero labels for CATARACTS-only steps (asymmetric
  supervision). Not worth it for one added dataset.
- **C. Shared backbone, per-dataset heads.** Right call if label semantics turn out to
  conflict (same visuals, different labels). Hold in reserve pending A's diagnostics.

## Pre-registered experiment (E7)

**Split roles, fixed up front:** CATARACTS train+dev (30 videos) = additional training
data; CATARACTS official test (20 videos) = external evaluation only, never trained on,
never used for any selection. C1K folds/seeds unchanged from Stage 2.

1. **E7-prep:** verify videos on disk (inventory: resolution/fps/duration vs GT frame
   counts; GT↔video pairing; check for UI overlays/vignetting in the Brest recordings;
   record which videos contain step 9 Vitrectomy). **Geometry policy (pre-registered):**
   center-crop 1920×1080 → 4:3 (1440×1080) to match C1K aspect, then the exact Stage-2
   resize — no squash, no new knobs. Extract DINOv2-with-registers-large features at the
   Stage-2 cache fps for all 50 videos, with Stage-2 provenance fields. Tool features: C1K
   fold-0 seg checkpoints applied zero-shot (deployment convention: multiclass `last.pt`,
   anatomy `best.pt`, as in the bulk pass) — recorded as out-of-domain.
2. **E7-0 (domain-gap sizing, zero training):** the frozen Stage-2 deployment stack run
   zero-shot on CATARACTS official test (mapped labels, IGNORE masked). This is the
   baseline for the primary endpoint and is informative before any training is committed:
   if the frozen model already transfers, the augmentation argument changes; if it
   collapses, the per-class breakdown says where.
3. **E7-a (C1K baseline):** existing winning config, C1K folds, official seeds — numbers
   already committed in `docs/stage2-phase-results.md`; no rerun.
4. **E7-b (augmented):** same config + CATARACTS train+dev (30 videos, mapped labels)
   added to each C1K fold's *training* set only; C1K test folds unchanged. Same seeds.
   **Pinned mechanics (B3):** Viterbi transition grammar estimated from C1K train folds
   only (matches deployment; no IGNORE-hole transitions); class weights computed over the
   composite train set excluding `ignore_index` frames, with before/after weight vectors
   logged in the run dir.
5. **E7-c (primary):** E7-b evaluated on CATARACTS official test; compared against E7-0.

**Decision rule:**
- **Primary endpoint:** E7-b beats E7-0 on CATARACTS official test — paired per-video
  stats (n=20) on macro-F1, edit, and F1@50; call it a win only if all three improve and
  the macro-F1 gain exceeds the Stage-2 seed spread (0.7pp).
- **Guardrail (C1K non-inferiority):** seed-mean C1K macro-F1 drop ≤0.3pp *and* Wilcoxon
  on paired per-video deltas (n=56) not significantly worse (α=0.05), on the Stage-2
  product metrics (edit, F1@50, boundary error, duration MAE). Guardrail failure vetoes
  adoption regardless of the primary.
- **Exploratory (no adoption weight):** C1K macro-F1 / minority-phase F1 gains.
- **Falsification probe:** per-class diagnostics for the merge targets
  (`Phacoemulsification`, `Irrigation/Aspiration`, `Tonifying/Antibiotics`) **and the two
  absent-source classes** (`Capsule Pulishing`, `Anterior_Chamber Flushing`) — the latter
  face active aliasing, not just imbalance: CATARACTS labels polishing-like I/A activity
  as `Irrigation/Aspiration`, training against C1K's polishing class. **Pre-registered
  fallback:** if either absent-source class regresses beyond the Stage-2 seed spread,
  first IGNORE the conflicting CATARACTS I/A spans; if that fails, escalate to Strategy C.
  Calibration must not degrade (Stage-2 confidence pipeline re-run on E7-b).
- **Conditional ablations (run only if the probe fires):** E7-b-notools (drop
  out-of-domain tool features); E7-b-novit (exclude the vitrectomy-containing videos —
  complicated cases look atypical even outside the IGNOREd span).

## Risks / open questions for PI

- **R1:** Idle-definition mismatch (C1K: "no instrument visible"; CATARACTS: "no active
  step"). **Pre-registered decision rule:** sample 200 CATARACTS idle frames stratified
  across the 30 train+dev videos; if >10% show an instrument inside the eye, map CATARACTS
  idle → IGNORE (else keep idle → idle). Result recorded here at map freeze.
- **R2:** Mapping decisions for IDs 2, 11, 12, 18 (above) — sampled-clip review before
  freezing the map; a wrong merge is worse than an IGNORE.
- **R3:** Out-of-domain tool features (seg model never saw Brest optics) may be noise for
  CATARACTS sequences; conditional ablation E7-b-notools (above).
- **R4:** Geometry policy pinned in E7-prep (4:3 center-crop, then Stage-2 preprocessing);
  no other normalization knobs without pre-registration.
- **R5:** License/citation: record CATARACTS terms (IEEE DataPort open access; Al Hajj et
  al. 2019 CATARACTS paper citation; confirm CC-BY status from the DataPort page) in
  README + CLAUDE.md per the repo's license-tracking convention **before** first trained
  artifact uses the data.
