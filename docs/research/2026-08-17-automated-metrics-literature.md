# Automated Video-Based Skill Assessment Literature (2018–2026)

*Research agent report, 2026-08-17 (condensed; URLs in original). Part of the feedback-metrics survey.*

## Cataract-specific
- **PhacoTrainer 2025 (key validity paper)**: 57 videos, OSACSS raters. Mask-derived metrics (per-tool path length in limbus units, max velocity, covered area, probe decentration, eye decentration, zoom SD). Extraction validity: area r=0.988, path 0.957, **velocity 0.769 (noisiest)**. vs total OSACSS: **pupil path r=-0.714, limbus path -0.664**, probe path -0.587; task-specific up to r=-0.77 (phaco path vs cracking item). NOTE: pure anatomy-motion metrics beat instrument metrics.
- **CatSkill 2025**: 430 surgeries, 12.6M frames; limbus/palpebral/Purkinje-1 segmentation + obstruction compensation. LCP1 (eye tilt), LCFC (centration), LFL (focus) → RF **AUC 0.865** attending-vs-resident; metric VARIABILITY also discriminates. Purkinje-1 needs only a bright-blob detector; scale via limbus≈11.5-12mm.
- **Ruzicki 2022 (negative result)**: tool-PRESENCE-only skill classification ≈ chance (AUC 0.55-0.69) despite excellent tool detection → spatial kinematics carry the signal.
- **Hisey 2025**: monocular video object-detection kinematics ≥ 6-DOF EM hardware tracking vs ICO-OSCAR (n small) → video suffices.
- **JHU rhexis lineage**: manual tip tracks + TCN AUC 0.863; predicted tips 0.788 (tip quality is the bottleneck); **cross-dataset transfer of end-to-end skill classifiers FAILS** (IJCARS 2025) → interpretable metric pipelines are the safer product basis.
- **Cataract-LMM (2025, CC-BY)**: 3,000 videos, 2 centers; 170 rhexis clips skill-rated (ICC 0.87); TimeSformer 82.5% low-vs-high; path length ↔ skill. **Natural external benchmark for our metrics.**
- **WetCat (same lab as Cataract-1K)**: ready-made metric formulas (circularity 4πA/P², limbus-normed centration, 4.5-5.5mm ideal, Fourier smoothness, eye stability ±10% limbus, non-dominant-hand movement stats); rhexis seg via pupil-prompted SAM-LoRA Dice 85.8.
- **Morita 2020**: real-time tip-to-anatomy distance risk indicator separates residents/experienced.

## Transferable general-surgery
- JIGSAWS lineage → multi-aspect models ρ 0.71-0.87; **combining metric aspects beats any single one**.
- **SAIS (Nature BME 2023)**: contrastive gesture-level skill decoding, AUC>0.8 across 3 hospitals — the design to copy IF we later train a learned rater on phase-clipped segments (our library is the substrate).
- Smoothness: **SPARC > jerk** for noisy 5-10fps tracks; no cataract validation yet (novelty + caution).
- Error/event detection in cataract is a gap (re-grasp counts, eye-out-of-frame, exchanges — cheap from our outputs).

## Rhexis automation
Segmentation proven (BigCat-Capsulotomy UNet Dice 92.1; WetCat SAM-LoRA 85.8 — pupil prompts, which we have). Metric→skill validity thin (CPI n=2). Cataract-1K lacks a rhexis class → one new model, WetCat provides formulas + data.

## Complications beyond our plan
- **CataractCompDetect (2025)**: masks → geometric risk scores (peaked-pupil sector for vitreous loss, pupil straight-edge for PCR, iris-color-beyond-boundary for prolapse) → VLM verification; avg F1 70.6 (n=53 MSICS). Architecture compatible with our stack; peaked-pupil metric nearly free from our pupil mask.
- Miosis: PLOS ONE 2021 recall 90.6% (our planned pipeline); Michigan 2024 confirms obstruction compensation is load-bearing.
- IOL: LensID lens Dice 92.6; their haptic/hook Faster-R-CNN solves the rotation-angle problem (lens disk ~rotationally symmetric) — adopt keypoints over principal-axis.

## Ranked by least-new-modeling
1. **Eye-motion metrics** (limbus/pupil path, centration, zoom SD, out-of-frame time) — cheapest AND strongest validated (ship first).
2. Per-phase tool-tip kinematics (validated; velocity noise-sensitive).
3. Purkinje-1 + focus (CatSkill) — near-free add-ons.
4. Rhexis morphometrics — the one justified new model.
5. Learned end-to-end scores + SPARC — research-grade until validated locally; benchmark on Cataract-LMM.
