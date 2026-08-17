# Commercial & Deployed Tools Landscape: Cataract Surgical Video Analysis and Feedback

*Research agent report, 2026-08-17. Verbatim; part of the feedback-metrics landscape survey.*

## 1. Ophthalmology-specific tools

### EyeSi Surgical (VRmagic, now Haag-Streit) — the incumbent for training, but simulator-only
- The dominant VR ophthalmic surgical simulator (~$150,000/unit; one analysis estimated ~34 years to recoup from OR-time savings).
- Scores per-task (max 100) across five domains: target achievement, efficiency, tissue treatment, instrument handling, microscope handling. Sub-metrics include an instrument-tip odometer (path length → average speed), tissue-injury deductions, time, target completion. Tiered courseware with scoring gates.
- **Transfer evidence (the strongest in the field)**: Royal College NOD Report 6 (Ferris et al., BJO 2019): across 17,831 trainee cases, unadjusted PCR rate fell 38% (4.2% → 2.6%) among 1st/2nd-year trainees with EyeSi access (before/after/no-access: 3.5%/2.6%/3.8%). Thomsen 2017: simulator performance correlates with real-life performance. 2023 meta-analysis: skill-acquisition evidence graded low certainty; error reduction very low certainty.
- PhacoTracking (Eye 2018) tried to bridge simulator metrics to real OR video — the field itself recognizes the missing link is metrics from real surgeries.
- **Gap: zero feedback on actual OR cases.**

### HelpMeSee — MSICS simulator (global ophthalmology)
Haptic VR simulator for manual small-incision cataract surgery; validated automated metrics test (Sci Rep 2023); centers worldwide. Simulation only.

### Orbis + FundamentalVR (2024)
Low-cost portable VR cataract training with cloud assessment data, for resource-limited settings. Simulation only.

### ZEISS — the closest commercial analogue
- ARTEVO digital microscopes + CALLISTO eye cockpit + the ZEISS Cataract Workflow.
- **ZEISS Surgery Optimizer** (~2022–2024): cloud app ingesting videos from CALLISTO-connected microscopes; AI phase segmentation, click-to-phase navigation, KPI dashboard (own + clinic), side-by-side comparison against curated reference videos. Positioned as a non-medical device. **Locked to ZEISS hardware.**
- Does NOT do: segmentation-derived kinematics, cohort-normed per-phase feedback, irregularity/event detection, narrative reports, resident-education framing. KPIs ≈ phase timings + case metadata.

### Alcon — planning/guidance, not skill feedback
ARGOS, ORA/VerifEye+, NGENUITY 1.5 overlays, CENTURION, SMARTCataract planning. Preop planning + intraop guidance + OR efficiency; no postoperative video analysis or skill metrics.

### Bausch + Lomb — device telemetry only (eyeTELLIGENCE / Stellaris Elite settings sync).

### Capture/review niche
MicroREC (Custom Surgical): smartphone-microscope capture + library/telementoring app, 65+ countries, no AI analysis. CataractCoach/Eyetube/Cybersight: human coaching content.

## 2. General surgical video/AI platforms — none serve ophthalmology

- **Theator**: automatic capture, step tagging, critical-event detection, Surgery-to-Text (documentation/billing), trainee skills scoring; Oracle Health partnership (2026). Laparoscopic/robotic MIS only.
- **Proximie**: telepresence + Intelligence Suite (event detection, OR-productivity analytics). Workflow-centric, no microsurgical skill metrics.
- **CSATS (J&J)**: human crowdsourced+expert video ratings (GEARS/OSATS-style); robotics/lap; diminished post-2023 layoffs; days of turnaround.
- **Touch Surgery Enterprise (Medtronic)**: AI capture + step segmentation + benchmarking, "instrument time-in-view"; ~£15,000/unit/year (UK G-Cloud); laparoscopic-centric; cataract content is training simulations only.
- **Caresyntax**: video assessments, peer benchmarking, credentialing; general surgery.
- **ExplORer (→GHX)**, **Incision**, **OR Black Box**: workflow/e-learning/safety; not ophthalmic skill analytics.

Summary: general platforms deliver phase timelines, duration benchmarks, tagged libraries, occasional human-rated scores, narrative op-notes. None deliver instrument-kinematic, anatomy-referenced skill metrics — and none serve ophthalmology.

## 3. Academic / open tools

- **PhacoTrainer (Wang, Stanford)** — closest academic analogue: DL over uploaded cataract videos; step recognition (TVST 2021), tool tracking (TVST 2023), computes path length, max velocity, area covered, phaco-probe decentration, eye decentration, zoom changes — separates attendings from trainees; 2025 work auto-generates performance ratings. Status: preliminary research dashboard; no norms, no product, no reports.
- **PhacoTracking** (Eye 2018): per-segment instrument motion analysis in real OR video; research only.
- **Klagenfurt group**: datasets and papers only (Cataract-1K/101, LensID); no public app.
- **CAMMA (Strasbourg)**: laparoscopy-dominant; cataract = challenge datasets + OphCLIP pretraining; no deployed tool.

## 4. Gap analysis — differentiated capabilities

1. **Per-phase instrument kinematics from real OR video, cohort-normed** (path length, velocity, smoothness, decentration vs segmented pupil/limbus, percentile vs experience-matched peers). No commercial product does this; PhacoTrainer proved discriminative validity but has no norms/product. "EyeSi's odometer, but for the trainee's actual surgeries" — and EyeSi's 38% PCR-reduction evidence is the argument the metrics matter.
2. **Anatomy-grounded event/irregularity detection** (miosis, IOL rotation, rhexis irregularity, prolonged idle) tied to the phase timeline.
3. **Automated narrative training reports** — formative/educational, not documentation/billing (Theator's target).
4. **Hardware-agnostic deployment** — ingests any microscope recording (even smartphone capture); vs CALLISTO-locked Surgery Optimizer, capture-box-locked Touch Surgery, $150k simulators.
5. **Phase-indexed longitudinal library with peer retrieval** — "all my capsulorhexes this year vs attending exemplars" + kinematic trend lines across residency.

(Original report includes full source URLs: Haag-Streit/EyeSi pages, Ferris BJO 2019, Thomsen 2017, 2023 meta-analysis, HelpMeSee Sci Rep 2023, Orbis/FundamentalVR PR, ZEISS Surgery Optimizer + Ophthalmology Times Europe, Alcon ESCRS 2024, B+L eyeTELLIGENCE PR, Theator/Oracle PR, Proximie, CSATS, Touch Surgery fact sheet + G-Cloud, Caresyntax, MicroREC, PhacoTrainer TVST 2021/2023 + 2025 ratings paper, PhacoTracking Eye 2018, Cataract-1K Sci Data 2024, OphCLIP arXiv.)
