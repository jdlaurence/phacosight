"""One-at-a-time GPU inference service for uploaded videos.

Reuses the validated bulk deployment stack (scripts/analyze_phase_bulk.py):
tool-fusion MS-TCN++ 12-member ensemble + temperature + learned-grammar
Viterbi at 1 fps, including its startup self-check. The stack loads lazily on
the first job and stays resident.
"""

import json
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from analyze_phase_bulk import (  # noqa: E402
    INFERENCE_FPS, TEMPERATURE, load_stack, self_check, video_features,
)
from cataract_video.phase.decoding import transition_matrix, viterbi  # noqa: E402
from cataract_video.phase.metrics import segments as label_segments  # noqa: E402
from cataract_video.phase.timeline import NUM_CLASSES, PHASES, phase_cases  # noqa: E402

UPLOAD_DIR = REPO / "data" / "library" / "uploads"
TIMELINE_DIR = REPO / "data" / "library" / "phase_timelines"


class InferenceService:
    def __init__(self):
        self.jobs: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._queue: list[str] = []
        self._stack = None
        self._log_trans = None
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def submit(self, video_path: Path) -> str:
        job_id = uuid.uuid4().hex[:12]
        self.jobs[job_id] = {"id": job_id, "case": video_path.stem, "status": "queued",
                             "detail": "waiting for GPU worker", "submitted": time.time()}
        with self._lock:
            self._queue.append(job_id)
            self.jobs[job_id]["path"] = str(video_path)
        return job_id

    def _ensure_stack(self):
        if self._stack is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dino, seg, heads = load_stack(device, INFERENCE_FPS)
            labeled = sorted(phase_cases(REPO / "data" / "phase"))
            stride = round(5.0 / INFERENCE_FPS)
            train_labels = [
                np.load(REPO / f"data/features/phase_dinov2l/{c}.npz")["labels"][::stride]
                for c in labeled
            ]
            self._log_trans = transition_matrix(train_labels, NUM_CLASSES)
            acc = self_check(dino, seg, heads, self._log_trans, device, INFERENCE_FPS)
            if acc < 0.85:
                raise RuntimeError(f"deployment stack self-check failed ({acc:.3f})")
            self._stack = (dino, seg, heads, device)

    def _run(self):
        while True:
            with self._lock:
                job_id = self._queue.pop(0) if self._queue else None
            if job_id is None:
                time.sleep(1.0)
                continue
            job = self.jobs[job_id]
            try:
                job.update(status="loading", detail="loading models (first job only)")
                self._ensure_stack()
                self._analyze(job)
                job.update(status="done", detail="analysis complete")
            except Exception as e:  # surface, don't kill the worker
                job.update(status="error", detail=f"{type(e).__name__}: {e}")

    @torch.no_grad()
    def _analyze(self, job: dict):
        dino, seg, heads, device = self._stack
        video = Path(job["path"])
        job.update(status="features", detail="decoding video + extracting features")
        x, times, duration, vfps = video_features(video, dino, seg, device, INFERENCE_FPS)
        job.update(status="inference", detail="temporal ensemble + decoding")
        xt = x.unsqueeze(0).to(device)
        member_probs = torch.stack([torch.softmax(h(xt)[-1, 0], -1) for h in heads])
        probs = member_probs.mean(0)
        probs = torch.softmax(torch.log(probs + 1e-12) / TEMPERATURE, -1).cpu().numpy()
        member_pred = member_probs.argmax(-1).cpu().numpy()
        disagree = (member_pred != member_pred[0]).any(0).astype(float)
        pred = viterbi(np.log(probs + 1e-12), self._log_trans)
        p_pred = probs[np.arange(len(pred)), pred]

        segs = [{
            "phase": PHASES[cls],
            "start_s": round(float(times[s]), 2),
            "end_s": round(float(times[e - 1] + 1.0 / INFERENCE_FPS), 2),
            "confidence": round(float(p_pred[s:e].mean()), 4),
            "disagreement": round(float(disagree[s:e].mean()), 4),
        } for cls, s, e in label_segments(pred)]

        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, cwd=REPO).stdout.strip()
        out = {
            "case": video.stem, "duration_s": round(duration, 2), "video_fps": vfps,
            "inference_fps": INFERENCE_FPS, "segments": segs, "source": "uploaded",
            "provenance": {"git_sha": sha,
                           "stack": f"tools-fusion x 12-member ensemble, T={TEMPERATURE}, "
                                    f"viterbi@{INFERENCE_FPS}fps, seg fold0"},
        }
        TIMELINE_DIR.mkdir(parents=True, exist_ok=True)
        (TIMELINE_DIR / f"{video.stem}.json").write_text(json.dumps(out, indent=1))
        job["result_case"] = video.stem
