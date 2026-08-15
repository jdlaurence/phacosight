#!/usr/bin/env python
"""Phase-duration cohort norms from labeled GT + gated bulk timelines.

Consumption rules from the PI gate review (docs/reviews/, 2026-08-14 bulk gate):
- Norms are built on PER-VIDEO PHASE TOTALS, never per-segment durations
  (Viterbi merges the burst-annotated phases; totals are the validated quantity).
- Flagged segments (conf<0.7 or disagreement>0.5) are excluded.
- Anchor-incomplete bulk videos are quarantined (all four of Incision,
  Capsulorhexis, Phacoemulsification, Lens Implantation must be present).
- Cohorts are kept separate AND pooled, with provenance — the bulk cohort runs
  ~10-30% longer on some phases than the labeled one.

Writes data/library/phase_norms.json.
"""

import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cataract_video.phase.timeline import PHASES, case_paths, load_segments, phase_cases

ANCHORS = {"Incision", "Capsulorhexis", "Phacoemulsification", "Lens Implantation"}
PCTS = (10, 25, 50, 75, 90)


def percentiles(vals: list[float]) -> dict:
    a = np.array(vals)
    return {f"p{p}": round(float(np.percentile(a, p)), 1) for p in PCTS} | {"n": len(a)}


def main() -> None:
    # labeled cohort: GT per-video totals
    labeled = defaultdict(dict)
    for c in phase_cases("data/phase"):
        _, ann = case_paths("data/phase", c)
        for r in load_segments(ann).itertuples():
            p = PHASES[r.phase_id]
            labeled[p][c] = labeled[p].get(c, 0.0) + (r.end_sec - r.start_sec)

    # bulk cohort: predicted totals, gated
    bulk = defaultdict(dict)
    quarantined = []
    for f in sorted(Path("data/library/phase_timelines").glob("*.json")):
        d = json.loads(f.read_text())
        if d.get("source") == "uploaded":  # physician uploads never enter norms
            continue
        segs = [s for s in d["segments"] if s["phase"] != "idle"]
        # video-level gate: all four anchors present in the raw prediction
        if not ANCHORS <= {s["phase"] for s in segs}:
            quarantined.append(d["case"])
            continue
        # a video contributes to a phase only if ALL that phase's segments are
        # unflagged — partial sums would undercount burst phases
        by_phase = defaultdict(list)
        for s in segs:
            by_phase[s["phase"]].append(s)
        for p, ss in by_phase.items():
            if all(s["confidence"] >= 0.7 and s["disagreement"] <= 0.5 for s in ss):
                bulk[p][d["case"]] = sum(s["end_s"] - s["start_s"] for s in ss)

    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                         cwd=Path(__file__).parents[1]).stdout.strip()
    n_bulk = len(set().union(*(set(v) for v in bulk.values())))
    out = {
        "provenance": {
            "git_sha": sha, "basis": "per-video phase totals; flagged segments excluded",
            "labeled_videos": 56, "bulk_videos_kept": n_bulk,
            "bulk_quarantined": quarantined,
            "note": "bulk cohort measured ~10-30% longer on some phases than labeled; "
                    "cohorts reported separately and pooled",
        },
        "phases": {},
    }
    for p in PHASES[1:]:
        lab = list(labeled[p].values())
        blk = list(bulk[p].values())
        out["phases"][p] = {
            "labeled": percentiles(lab) if lab else None,
            "bulk": percentiles(blk) if blk else None,
            "pooled": percentiles(lab + blk) if lab + blk else None,
        }
    path = Path("data/library/phase_norms.json")
    path.write_text(json.dumps(out, indent=1))
    print(f"norms over {56 + n_bulk} videos ({len(quarantined)} quarantined: {quarantined})")
    for p in ("Capsulorhexis", "Phacoemulsification", "Tonifying/Antibiotics"):
        row = out["phases"][p]
        print(f"  {p:<26} labeled p50 {row['labeled']['p50']:>6}s | bulk p50 {row['bulk']['p50']:>6}s "
              f"| pooled p10-p90 {row['pooled']['p10']}-{row['pooled']['p90']}s (n={row['pooled']['n']})")


if __name__ == "__main__":
    main()
