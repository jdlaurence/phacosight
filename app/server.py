"""Physician-facing web app: browse analyzed surgeries, inspect timelines and
metrics against cohort norms, search the phase library, upload new videos for
GPU analysis.

    .venv/bin/uvicorn app.server:app --host 0.0.0.0 --port 7860
    # then ssh -L 7860:localhost:7860 <cluster> and open http://localhost:7860
"""

import json
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from cataract_video.phase.timeline import PHASES, case_paths, load_segments, phase_cases  # noqa: E402

from .inference import UPLOAD_DIR, InferenceService  # noqa: E402

TIMELINES = REPO / "data" / "library" / "phase_timelines"
NORMS_PATH = REPO / "data" / "library" / "phase_norms.json"
CLIP_CACHE = Path(tempfile.gettempdir()) / "cataract_clips"
FLAG = {"conf": 0.7, "dis": 0.5}

app = FastAPI(title="Cataract surgery analysis")
service = InferenceService()


# ---------------------------------------------------------------- data access
def video_path(case: str) -> Path:
    for p in (REPO / "data/full" / f"{case}.mp4",
              REPO / "data/phase/videos" / f"{case}.mp4",
              UPLOAD_DIR / f"{case}.mp4"):
        if p.exists():
            return p
    raise HTTPException(404, f"no video file for {case}")


def load_timeline(case: str) -> dict:
    p = TIMELINES / f"{case}.json"
    if not p.exists():
        raise HTTPException(404, f"{case} not analyzed")
    return json.loads(p.read_text())


def labeled_gt(case: str) -> list | None:
    ann = REPO / "data/phase/annotations" / case / f"{case}_annotations_phases.csv"
    if not ann.exists():
        return None
    return [{"phase": PHASES[r.phase_id], "start_s": r.start_sec, "end_s": r.end_sec}
            for r in load_segments(ann).itertuples()]


class Norms:
    """Raw per-video phase-total distributions (labeled GT + gated bulk)."""

    def __init__(self):
        self.dist: dict[str, list[float]] = defaultdict(list)
        for c in phase_cases(REPO / "data/phase"):
            _, ann = case_paths(REPO / "data/phase", c)
            totals = defaultdict(float)
            for r in load_segments(ann).itertuples():
                totals[PHASES[r.phase_id]] += r.end_sec - r.start_sec
            for p, v in totals.items():
                self.dist[p].append(v)
        quarantined = set()
        if NORMS_PATH.exists():
            quarantined = set(json.loads(NORMS_PATH.read_text())
                              ["provenance"]["bulk_quarantined"])
        for f in TIMELINES.glob("*.json"):
            d = json.loads(f.read_text())
            if d["case"] in quarantined or d.get("source") == "uploaded":
                continue
            by_phase = defaultdict(list)
            for s in d["segments"]:
                if s["phase"] != "idle":
                    by_phase[s["phase"]].append(s)
            for p, ss in by_phase.items():
                if all(s["confidence"] >= FLAG["conf"] and s["disagreement"] <= FLAG["dis"]
                       for s in ss):
                    self.dist[p].append(sum(s["end_s"] - s["start_s"] for s in ss))
        self.dist = {p: sorted(v) for p, v in self.dist.items()}

    def percentile(self, phase: str, value: float) -> float | None:
        v = self.dist.get(phase)
        if not v:
            return None
        return round(100 * float(np.searchsorted(v, value) / len(v)), 1)


norms = Norms()


def video_metrics(d: dict) -> dict:
    """Per-phase totals, idle time, cohort percentiles, flags."""
    segs = d["segments"]
    totals = defaultdict(float)
    flagged = []
    for i, s in enumerate(segs):
        totals[s["phase"]] += s["end_s"] - s["start_s"]
        if s["phase"] != "idle" and (s["confidence"] < FLAG["conf"]
                                     or s["disagreement"] > FLAG["dis"]):
            flagged.append(i)
    action = [s for s in segs if s["phase"] != "idle"]
    confs = [s["confidence"] for s in action] or [0.0]
    phases = []
    for p in PHASES[1:]:
        if p in totals:
            v = totals[p]
            band = norms.dist.get(p, [])
            phases.append({
                "phase": p, "total_s": round(v, 1),
                "percentile": norms.percentile(p, v),
                "cohort_p25": round(float(np.percentile(band, 25)), 1) if band else None,
                "cohort_p50": round(float(np.percentile(band, 50)), 1) if band else None,
                "cohort_p75": round(float(np.percentile(band, 75)), 1) if band else None,
                "n_cohort": len(band),
            })
    return {
        "phases": phases,
        "idle_s": round(totals.get("idle", 0.0), 1),
        "n_segments": len(segs),
        "flagged_idx": flagged,
        "flag_fraction": round(len(flagged) / max(1, len(action)), 3),
        "median_confidence": round(float(np.median(confs)), 3),
        "needs_review": len(flagged) / max(1, len(action)) > 0.25
                        or float(np.mean(confs)) < 0.8,
    }


# --------------------------------------------------------------------- routes
@app.get("/api/videos")
def list_videos():
    out = []
    for f in sorted(TIMELINES.glob("*.json")):
        d = json.loads(f.read_text())
        action = [s for s in d["segments"] if s["phase"] != "idle"]
        n_flag = sum(1 for s in action if s["confidence"] < FLAG["conf"]
                     or s["disagreement"] > FLAG["dis"])
        out.append({
            "case": d["case"], "duration_s": d["duration_s"],
            "n_segments": len(d["segments"]),
            "flag_fraction": round(n_flag / max(1, len(action)), 3),
            "source": d.get("source", "library"),
            "has_gt": labeled_gt(d["case"]) is not None,
        })
    for c in phase_cases(REPO / "data/phase"):  # labeled videos: GT-only entries
        if not (TIMELINES / f"{c}.json").exists():
            gt = labeled_gt(c)
            out.append({"case": c, "duration_s": round(max(s["end_s"] for s in gt), 1),
                        "n_segments": len(gt), "flag_fraction": 0.0,
                        "source": "labeled", "has_gt": True})
    return out


@app.get("/api/videos/{case}")
def get_video(case: str):
    try:
        d = load_timeline(case)
    except HTTPException:
        gt = labeled_gt(case)
        if gt is None:
            raise
        dur = max(s["end_s"] for s in gt)
        d = {"case": case, "duration_s": dur, "segments": [], "source": "labeled"}
    d["metrics"] = video_metrics(d) if d["segments"] else None
    d["ground_truth"] = labeled_gt(case)
    d["phase_names"] = list(PHASES)
    return d


@app.get("/api/norms")
def get_norms():
    return {p: {"n": len(v),
                "p10": v[int(0.10 * len(v))], "p25": v[int(0.25 * len(v))],
                "p50": v[int(0.50 * len(v))], "p75": v[int(0.75 * len(v))],
                "p90": v[min(len(v) - 1, int(0.90 * len(v)))]}
            for p, v in norms.dist.items()}


@app.get("/api/search")
def search(phase: str, min_conf: float = 0.9, limit: int = 60):
    hits = []
    for f in sorted(TIMELINES.glob("*.json")):
        d = json.loads(f.read_text())
        for s in d["segments"]:
            if s["phase"] == phase and s["confidence"] >= min_conf:
                hits.append({"case": d["case"], **s})
    hits.sort(key=lambda s: -s["confidence"])
    return {"total": len(hits), "hits": hits[:limit]}


@app.get("/api/frame")
def frame(case: str, t: float, h: int = 240):
    cap = cv2.VideoCapture(str(video_path(case)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * cap.get(cv2.CAP_PROP_FPS)))
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise HTTPException(404, "frame out of range")
    scale = h / img.shape[0]
    img = cv2.resize(img, (int(img.shape[1] * scale), h))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return Response(buf.tobytes(), media_type="image/jpeg",
                    headers={"Cache-Control": "max-age=86400"})


@app.get("/api/clip")
def clip(case: str, start: float, end: float):
    if end - start <= 0 or end - start > 300:
        raise HTTPException(400, "clip must be 0-300 s")
    CLIP_CACHE.mkdir(exist_ok=True)
    out = CLIP_CACHE / f"{case}_{start:.1f}_{end:.1f}.mp4"
    if not out.exists():
        r = subprocess.run(
            ["ffmpeg", "-y", "-ss", str(start), "-to", str(end), "-i",
             str(video_path(case)), "-c", "copy", str(out)],
            capture_output=True, timeout=60)
        if r.returncode != 0:
            raise HTTPException(500, "ffmpeg failed")
    return FileResponse(out, media_type="video/mp4",
                        filename=f"{case}_{PHASES[0] and ''}{start:.0f}s.mp4")


@app.get("/api/stream/{case}")
def stream(case: str, request: Request):
    """Range-aware video streaming so the browser can seek."""
    path = video_path(case)
    size = path.stat().st_size
    range_header = request.headers.get("range")
    if not range_header:
        return FileResponse(path, media_type="video/mp4")
    start_s, _, end_s = range_header.replace("bytes=", "").partition("-")
    start = int(start_s)
    end = min(int(end_s) if end_s else start + 4 * 2**20, size - 1)

    def reader():
        with open(path, "rb") as f:
            f.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = f.read(min(1 << 20, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    return StreamingResponse(
        reader(), status_code=206, media_type="video/mp4",
        headers={"Content-Range": f"bytes {start}-{end}/{size}",
                 "Accept-Ranges": "bytes", "Content-Length": str(end - start + 1)})


@app.post("/api/upload")
async def upload(file: UploadFile):
    if not file.filename.lower().endswith(".mp4"):
        raise HTTPException(400, "mp4 only")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(file.filename).stem.replace(" ", "_")
    dest = UPLOAD_DIR / f"{stem}.mp4"
    i = 1
    while dest.exists():
        dest = UPLOAD_DIR / f"{stem}_{i}.mp4"
        i += 1
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    job_id = service.submit(dest)
    return {"job_id": job_id, "case": dest.stem}


@app.get("/api/jobs/{job_id}")
def job(job_id: str):
    j = service.jobs.get(job_id)
    if not j:
        raise HTTPException(404, "unknown job")
    return {k: v for k, v in j.items() if k != "path"}


app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True))
