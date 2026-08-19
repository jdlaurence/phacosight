"""Physician-facing web app: browse analyzed surgeries, inspect timelines and
metrics against cohort norms, search the phase library, track progress over
time, upload new videos for analysis.

Run: `python -m app`  (see app/__main__.py for options)
"""

import json
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import FULL_DIR, NORMS_PATH, PHASE_DIR, REPO, TIMELINE_DIR, UPLOAD_DIR
from .inference import InferenceService

sys.path.insert(0, str(REPO / "src"))
from phacosight.phase.timeline import PHASES, case_paths, load_segments, phase_cases  # noqa: E402

CLIP_CACHE = Path(tempfile.gettempdir()) / "phacosight_clips"
CLIP_CACHE_MAX_BYTES = 2 * 2**30
FLAG = {"conf": 0.7, "dis": 0.5}
CASE_RE = re.compile(r"[A-Za-z0-9._-]{1,80}")

app = FastAPI(title="PhacoSight")


# ---------------------------------------------------------------- data access
def safe_case(case: str) -> str:
    if not CASE_RE.fullmatch(case) or ".." in case:
        raise HTTPException(400, "invalid case name")
    return case


def video_path(case: str) -> Path:
    safe_case(case)
    for p in (FULL_DIR / f"{case}.mp4", PHASE_DIR / "videos" / f"{case}.mp4",
              UPLOAD_DIR / f"{case}.mp4"):
        if p.exists():
            return p
    raise HTTPException(404, f"no video file for {case}")


class TimelineCache:
    """Mtime-keyed cache of TIMELINE_DIR — list endpoints and norms would
    otherwise re-parse every timeline JSON per request, and one malformed
    upload would 500 all of them."""

    REQUIRED = {"case", "duration_s", "segments"}

    def __init__(self):
        self._entries: dict[str, tuple[float, dict]] = {}
        self._bad: set[str] = set()
        self._lock = threading.Lock()

    def all(self) -> list[dict]:
        with self._lock:
            seen = set()
            for f in TIMELINE_DIR.glob("*.json"):
                seen.add(f.name)
                mtime = f.stat().st_mtime
                hit = self._entries.get(f.name)
                if hit and hit[0] == mtime:
                    continue
                try:
                    d = json.loads(f.read_text())
                    missing = self.REQUIRED - d.keys()
                    if missing:
                        raise ValueError(f"missing keys {sorted(missing)}")
                except (ValueError, OSError) as e:
                    self._entries.pop(f.name, None)
                    if f.name not in self._bad:
                        print(f"[app] skipping malformed timeline {f.name}: {e}")
                        self._bad.add(f.name)
                    continue
                self._bad.discard(f.name)
                self._entries[f.name] = (mtime, d)
            for name in set(self._entries) - seen:
                del self._entries[name]
            return [d for _, (_, d) in sorted(self._entries.items())]


timelines = TimelineCache()


def load_timeline(case: str) -> dict:
    safe_case(case)
    p = TIMELINE_DIR / f"{case}.json"
    if not p.exists():
        raise HTTPException(404, f"{case} not analyzed")
    try:
        return json.loads(p.read_text())
    except ValueError:
        raise HTTPException(500, f"timeline for {case} is malformed")


def labeled_cases() -> list[str]:
    return phase_cases(PHASE_DIR) if (PHASE_DIR / "annotations").exists() else []


@lru_cache(maxsize=None)  # expert GT CSVs are a static dataset
def labeled_gt(case: str) -> list | None:
    safe_case(case)
    ann = PHASE_DIR / "annotations" / case / f"{case}_annotations_phases.csv"
    if not ann.exists():
        return None
    return [{"phase": PHASES[r.phase_id], "start_s": r.start_sec, "end_s": r.end_sec}
            for r in load_segments(ann).itertuples()]


class Norms:
    """Cohort statistics from labeled GT + gated base-library predictions.

    Uploaded videos never contribute. Per phase: per-video totals (`dist`),
    per-segment durations, first-occurrence position, videos-containing count.
    """

    def __init__(self):
        self.dist: dict[str, list[float]] = defaultdict(list)
        self.seg_durs: dict[str, list[float]] = defaultdict(list)
        self.first_frac: dict[str, list[float]] = defaultdict(list)
        self.videos_with: dict[str, int] = defaultdict(int)
        self.n_videos = 0

        for c in labeled_cases():
            gt = labeled_gt(c)
            if not gt:  # case listed but CSV missing/empty
                continue
            self.n_videos += 1
            end = max(s["end_s"] for s in gt)
            totals, first = defaultdict(float), {}
            for s in gt:
                totals[s["phase"]] += s["end_s"] - s["start_s"]
                self.seg_durs[s["phase"]].append(s["end_s"] - s["start_s"])
                first.setdefault(s["phase"], s["start_s"])
            for p, v in totals.items():
                self.dist[p].append(v)
                self.videos_with[p] += 1
                self.first_frac[p].append(first[p] / max(end, 1))

        quarantined = set()
        if NORMS_PATH.exists():
            quarantined = set(json.loads(NORMS_PATH.read_text())
                              ["provenance"]["bulk_quarantined"])
        for d in timelines.all():
            if d["case"] in quarantined or d.get("source") == "uploaded":
                continue
            self.n_videos += 1
            by_phase = defaultdict(list)
            for s in d["segments"]:
                if s["phase"] != "idle":
                    by_phase[s["phase"]].append(s)
            for p, ss in by_phase.items():
                self.videos_with[p] += 1
                if all(s["confidence"] >= FLAG["conf"] and s["disagreement"] <= FLAG["dis"]
                       for s in ss):
                    self.dist[p].append(sum(s["end_s"] - s["start_s"] for s in ss))
                    self.seg_durs[p].extend(s["end_s"] - s["start_s"] for s in ss)
                    self.first_frac[p].append(ss[0]["start_s"] / max(d["duration_s"], 1))
        self.dist = {p: sorted(v) for p, v in self.dist.items()}

    def percentile(self, phase: str, value: float) -> float | None:
        v = self.dist.get(phase)
        if not v:
            return None
        return round(100 * float(np.searchsorted(v, value) / len(v)), 1)


norms = Norms()
_norms_lock = threading.Lock()


def rebuild_norms():
    """Refresh cohort norms after new analyses land (worker on_done hook)."""
    global norms
    with _norms_lock:
        norms = Norms()


service = InferenceService(on_done=rebuild_norms)


def video_metrics(d: dict) -> dict:
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
    }


# --------------------------------------------------------------------- routes
@app.get("/api/videos")
def list_videos():
    out = []
    for d in timelines.all():
        action = [s for s in d["segments"] if s["phase"] != "idle"]
        n_flag = sum(1 for s in action if s["confidence"] < FLAG["conf"]
                     or s["disagreement"] > FLAG["dis"])
        out.append({
            "case": d["case"], "duration_s": d["duration_s"],
            "n_segments": len(d["segments"]),
            "flag_fraction": round(n_flag / max(1, len(action)), 3),
            "source": d.get("source", "library"),
            "physician": d.get("physician"),
            "surgery_date": d.get("surgery_date"),
            "has_gt": labeled_gt(d["case"]) is not None,
        })
    for c in labeled_cases():
        if not (TIMELINE_DIR / f"{c}.json").exists():
            gt = labeled_gt(c)
            out.append({"case": c, "duration_s": round(max(s["end_s"] for s in gt), 1),
                        "n_segments": len(gt), "flag_fraction": 0.0, "source": "labeled",
                        "physician": None, "surgery_date": None, "has_gt": True})
    return out


@app.get("/api/videos/{case}")
def get_video(case: str):
    try:
        d = load_timeline(case)
    except HTTPException:
        gt = labeled_gt(case)
        if gt is None:
            raise
        d = {"case": case, "duration_s": max(s["end_s"] for s in gt),
             "segments": [], "source": "labeled"}
    d["metrics"] = video_metrics(d) if d["segments"] else None
    d["ground_truth"] = labeled_gt(case)
    d["phase_names"] = list(PHASES)
    try:
        video_path(case)
        d["has_video"] = True
    except HTTPException:
        d["has_video"] = False
    return d


@app.get("/api/phase_stats")
def phase_stats(phase: str):
    dist = norms.dist.get(phase, [])
    if not dist:
        raise HTTPException(404, f"no cohort data for {phase}")
    a = np.array(dist)
    segs = np.array(norms.seg_durs.get(phase, [0]))
    counts, edges = np.histogram(a, bins=24)
    return {
        "phase": phase,
        "n_videos": norms.n_videos,
        "videos_with_phase": norms.videos_with.get(phase, 0),
        "presence": round(norms.videos_with.get(phase, 0) / max(1, norms.n_videos), 3),
        "total": {f"p{p}": round(float(np.percentile(a, p)), 1) for p in (10, 25, 50, 75, 90)},
        "segments_per_video": round(len(segs) / max(1, len(a)), 2),
        "segment_median_s": round(float(np.median(segs)), 1),
        "typical_position": round(float(np.median(norms.first_frac.get(phase, [0]))), 3),
        "histogram": {"counts": counts.tolist(),
                      "edges": [round(float(e), 1) for e in edges]},
    }


@app.get("/api/physicians")
def physicians():
    seen = defaultdict(lambda: {"n_videos": 0, "last_date": None})
    for d in timelines.all():
        if d.get("physician"):
            e = seen[d["physician"]]
            e["n_videos"] += 1
            date = d.get("surgery_date") or d.get("uploaded_at")
            if date and (e["last_date"] is None or date > e["last_date"]):
                e["last_date"] = date
    return [{"name": k, **v} for k, v in sorted(seen.items())]


@app.get("/api/progress")
def progress(physician: str):
    rows = []
    for d in timelines.all():
        if d.get("physician") != physician:
            continue
        m = video_metrics(d)
        rows.append({
            "case": d["case"],
            "surgery_date": d.get("surgery_date") or d.get("uploaded_at"),
            "duration_s": d["duration_s"],
            "idle_s": m["idle_s"],
            "flag_fraction": m["flag_fraction"],
            "phases": {p["phase"]: {"total_s": p["total_s"], "percentile": p["percentile"]}
                       for p in m["phases"]},
        })
    rows.sort(key=lambda r: (r["surgery_date"] or "", r["case"]))
    cohort = {p: {"p25": round(float(np.percentile(v, 25)), 1),
                  "p50": round(float(np.percentile(v, 50)), 1),
                  "p75": round(float(np.percentile(v, 75)), 1)}
              for p, v in norms.dist.items() if v}
    return {"physician": physician, "videos": rows, "cohort": cohort}


@app.get("/api/frame")
def frame(case: str, t: float, h: int = 240):
    cap = cv2.VideoCapture(str(video_path(case)))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or fps <= 0:
        cap.release()
        raise HTTPException(415, f"unreadable video for {case}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise HTTPException(404, "frame out of range")
    scale = h / img.shape[0]
    img = cv2.resize(img, (int(img.shape[1] * scale), h))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return Response(buf.tobytes(), media_type="image/jpeg",
                    headers={"Cache-Control": "max-age=86400"})


@app.get("/api/search")
def search(phase: str, min_conf: float = 0.9, limit: int = 60):
    hits = []
    for d in timelines.all():
        for s in d["segments"]:
            # flagged segments (low confidence OR member disagreement) are
            # excluded, matching the UI copy and isFlagged() in app.js
            if (s["phase"] == phase and s.get("confidence", 0) >= min_conf
                    and s.get("confidence", 0) >= FLAG["conf"]
                    and s.get("disagreement", 1) <= FLAG["dis"]):
                hits.append({"case": d["case"], **s})
    hits.sort(key=lambda s: -s["confidence"])
    return {"total": len(hits), "hits": hits[:limit]}


def evict_cache(cache_dir: Path, max_bytes: int):
    files = sorted(cache_dir.glob("*"), key=lambda p: p.stat().st_mtime)
    total = sum(p.stat().st_size for p in files)
    for p in files:
        if total <= max_bytes:
            break
        total -= p.stat().st_size
        p.unlink(missing_ok=True)


@app.get("/api/clip")
def clip(case: str, start: float, end: float):
    if end - start <= 0 or end - start > 300:
        raise HTTPException(400, "clip must be 0-300 s")
    CLIP_CACHE.mkdir(exist_ok=True)
    out = CLIP_CACHE / f"{case}_{start:.1f}_{end:.1f}.mp4"
    if not out.exists():
        try:
            r = subprocess.run(
                ["ffmpeg", "-y", "-ss", str(start), "-to", str(end), "-i",
                 str(video_path(case)), "-c", "copy", str(out)],
                capture_output=True, timeout=60)
        except subprocess.TimeoutExpired:
            out.unlink(missing_ok=True)
            raise HTTPException(504, "clip extraction timed out")
        if r.returncode != 0:
            out.unlink(missing_ok=True)
            raise HTTPException(500, "ffmpeg failed")
        evict_cache(CLIP_CACHE, CLIP_CACHE_MAX_BYTES)
    return FileResponse(out, media_type="video/mp4", filename=f"{case}_{start:.0f}s.mp4")


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
async def upload(file: UploadFile, physician: str = Form(""), surgery_date: str = Form(""),
                 operator: str = Form("")):
    if not file.filename.lower().endswith(".mp4"):
        raise HTTPException(400, "mp4 only")
    stem = re.sub(r"[^A-Za-z0-9._-]", "_", Path(file.filename).stem)[:60] or "upload"
    dest = UPLOAD_DIR / f"{stem}.mp4"
    i = 1
    while dest.exists():
        dest = UPLOAD_DIR / f"{stem}_{i}.mp4"
        i += 1
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    job_id = service.submit(dest, {"physician": physician.strip(),
                                   "surgery_date": surgery_date.strip(),
                                   "operator": operator.strip()})
    return {"job_id": job_id, "case": dest.stem}


def public_job(j: dict) -> dict:
    return {k: v for k, v in j.items() if k not in ("path", "meta")}


@app.get("/api/jobs")
def jobs_list():
    js = sorted(service.jobs.values(), key=lambda j: -j["submitted"])
    return [public_job(j) for j in js[:50]]


@app.get("/api/jobs/{job_id}")
def job(job_id: str):
    j = service.jobs.get(job_id)
    if not j:
        raise HTTPException(404, "unknown job")
    return public_job(j)


@app.delete("/api/videos/{case}")
def delete_video(case: str):
    d = load_timeline(case)
    if d.get("source") != "uploaded":
        raise HTTPException(403, "only uploaded videos can be deleted")
    (TIMELINE_DIR / f"{case}.json").unlink(missing_ok=True)
    (UPLOAD_DIR / f"{case}.mp4").unlink(missing_ok=True)
    rebuild_norms()  # uploads never contribute, but this also refreshes the list cache
    return {"deleted": case}


@app.post("/api/reanalyze/{case}")
def reanalyze(case: str):
    d = load_timeline(case)
    if d.get("source") != "uploaded":
        raise HTTPException(403, "only uploaded videos can be re-analyzed")
    p = UPLOAD_DIR / f"{case}.mp4"
    if not p.exists():
        raise HTTPException(404, "original upload file is gone")
    job_id = service.submit(p, {"physician": d.get("physician") or "",
                                "surgery_date": d.get("surgery_date") or "",
                                "operator": d.get("operator") or ""})
    return {"job_id": job_id, "case": case}


app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True))
