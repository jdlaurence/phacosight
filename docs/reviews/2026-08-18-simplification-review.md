# PI review — post-E7 simplification pass (2026-08-18)

Review of commit `e759908` (configs 25→5, SegFormer-only factory, third_party removed,
shared `phase/data_io.py`) and the proposed runs/ artifact deletion. Disposition:
**APPROVE W/ CONCERNS** — commit verified behavior-preserving by independent re-runs
(eval_phase baseline seed0 and confidence_phase on the aug seeds both reproduced to all
printed digits); concerns targeted the deletion proposal and were adopted:

1. **`runs/phase_mstcnpp_dinov2l` spared** — it backs the published clip-anchor number
   (`clip_anchor.py` default) and is the ablation base for tools-fusion/1 fps
   comparisons; deletion would be permanent (runs/ checkpoints are gitignored).
2. **Delete checkpoints, not run directories** — per-fold `metrics.json` and the
   eval/confidence JSONs are the paper's receipts. Adopted and extended: `.gitignore`
   now tracks all small JSONs under runs/ (111 files, ~13 MB), making the provenance
   chain durable in git.
3. Deletion list otherwise sound; keep-list confirmed load-bearing (aug seeds =
   deployment ensemble; baseline tools_1fps seeds = external-harness default and
   reproduction gate).
4. Stale `TEMPERATURE = 1.04` comment in `eval_cataracts_zeroshot.py` reworded (frozen
   E7-era value for reproducing recorded results; deployment uses 1.016).
5. Docstring example commands repointed at the adopted aug seeds.

Noted well-done: the self-check gate propagating into the app meant the refactor could
not silently break deployment; `run_stride` + checkpoint-embedded configs made
independent verification trivial; grammar asset committed for bare-clone decodability.
Big-picture: artifact governance was the real remaining risk (now addressed by tracked
receipts); next real experiment question remains external robustness beyond CATARACTS.
