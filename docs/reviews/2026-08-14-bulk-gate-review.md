# PI Review — Gate: consumption of regenerated bulk phase timelines (334 videos)

*2026-08-14. Verdict: **APPROVE WITH CONCERNS** — outputs fit for the library and cohort norms under three consumption rules. All stats reproduced under independent recomputation; a new labeled-video control resolved the duration question in the pipeline's favor.*

## Verified

- **Provenance clean**: all 334 timelines carry the fix commit SHA (10370aa); zero pre-fix outputs; the smoke-test-era cases were regenerated.
- **Guards verified in code**: head-rate assertion and ≥0.85 self-check hard-exit before any write; reviewer replicated the self-check at 0.993.
- **Stats reproduce exactly**: 5,219 action segments, 12.0% flagged, median confidence 0.948; 327/334 videos have all four anchors, 327/327 canonical order; idle fraction and segments-per-video match labeled GT distributions.
- **Flags are a real error detector**: per-video low-confidence fraction correlates with held-out frame accuracy at **r = −0.94**.
- **Duration "inflation" resolved (decisive control)**: running the deployment stack on all 56 labeled videos with held-out heads reproduces the per-segment inflation (Viterbi merges GT's burst annotations) while **per-video totals are accurate** (Tonifying 19.0 vs 18.5 s; Visc-Suction 27.5 vs 28.2 s). The residual bulk-vs-labeled difference (~10–30% on some phases, both resolution eras, no tail-bleed, no era confidence collapse) is a genuine cohort shift. I/A runs *shorter* in bulk — the cohort isn't uniformly slower.

## Consumption rules (adopted)

1. **Norms use per-video phase totals**, never per-segment durations for burst-like phases; label norms with cohort provenance.
2. **Quarantine the 7 anchor-incomplete videos** (one 49 s fragment predicted all-idle — correct behavior; three long heavily-flagged atypical recordings; three single-anchor misses). 2% cost.
3. **Flagged segments excluded from norms**; the library surfaces per-segment confidence and filters flagged segments out of retrieval by default. Expect a few percent substantially-wrong timelines among the kept 327, concentrated in high-flag videos (in-domain equivalent: 2/56 below 0.85 acc, both flaggable).

## Recommendations

- Promote the held-out labeled-video timeline control into the standard toolkit; rerun on any stack change.
- Before norms are quoted to residents: extend the bulk pass to the remaining ~2/3 (pure compute; user-capped for now) and hand-verify ~5 whole timelines by video scrubbing (frame spot-checks can't catch boundary placement).
- Decide how idle gaps inside merged burst phases are handled in clip retrieval.

## Disposition (same day)

- `scripts/build_phase_norms.py` implements all three rules (anchor gate on raw predictions; per-phase all-clean contribution; flagged excluded; cohorts separate + pooled with provenance). Quarantine reproduces the reviewer's exact 7.
- Remaining-2/3 extension and whole-timeline scrubbing left as the user's call (compute cap respected).
