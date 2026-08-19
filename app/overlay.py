"""On-demand segmentation overlays for the physician app.

Runs the deployed SegFormer-B2 anatomy+instrument checkpoint (the same one the
phase stack uses for tool-fusion features) on a single decoded frame and
renders an alpha-blended class overlay. Lazy singleton: nothing loads until
the first overlay request, so browse-only and bare-clone deployments are
unaffected. CUDA if available; CPU works at ~1-3 s/frame, acceptable for
on-demand rendering behind the disk cache in server.py.
"""

import sys
import threading
from pathlib import Path

import cv2
import numpy as np
import torch

from .config import REPO

sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from analyze_phase_bulk import MEAN, STD  # noqa: E402  (deployment preprocessing)
from phacosight.labels import ANATOMY_INSTRUMENT  # noqa: E402
from phacosight.models.segmentation import build_model  # noqa: E402

CKPT = REPO / "runs" / "segformer_b2_anatomy_instrument" / "fold0" / "best.pt"

# Display colors keyed by ANATOMY_INSTRUMENT.class_names (None = transparent).
# Served to the frontend via /api/overlay/legend — the palette lives here only.
OVERLAY_COLORS = {
    "background": None,
    "cornea": "#2a78d6",
    "pupil": "#1baf7a",
    "lens": "#9085e9",
    "instrument": "#eb6834",
}
ALPHA = 0.45


def _bgr(hex_color: str) -> tuple[int, int, int]:
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return (b, g, r)


class OverlayService:
    def __init__(self):
        self._model = None
        self._device = None
        self._lock = threading.Lock()

    def available(self) -> bool:
        return CKPT.exists()

    def _ensure(self):
        if self._model is not None:
            return
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(CKPT, map_location=device, weights_only=False)
        m = build_model("segformer_b2", ANATOMY_INSTRUMENT.num_classes).to(device).eval()
        m.load_state_dict(ckpt["model"])
        self._model, self._device = m, device
        print(f"[overlay] segmentation model loaded on {device}")

    @torch.no_grad()
    def mask(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Class-ID mask for one frame. Returns (mask, frame) — the frame is
        center-cropped to 4:3 first, matching the training/deployment
        preprocessing in analyze_phase_bulk.video_features."""
        with self._lock:
            self._ensure()
            fh, fw = frame_bgr.shape[:2]
            if fw * 3 > fh * 4:  # wider than 4:3 (e.g. 16:9 uploads)
                cw = (fh * 4) // 3
                x0 = (fw - cw) // 2
                frame_bgr = frame_bgr[:, x0:x0 + cw]
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            x = torch.from_numpy(cv2.resize(rgb, (512, 512))).to(self._device)
            x = (x.float().div_(255).permute(2, 0, 1).unsqueeze(0)
                 - MEAN.to(self._device)) / STD.to(self._device)
            with torch.autocast(self._device.type, dtype=torch.float16,
                                enabled=self._device.type == "cuda"):
                pred = self._model(x).argmax(1)[0].to(torch.uint8).cpu().numpy()
            mask = cv2.resize(pred, (frame_bgr.shape[1], frame_bgr.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
            return mask, frame_bgr


def render_overlay(frame_bgr: np.ndarray, mask: np.ndarray,
                   alpha: float = ALPHA) -> np.ndarray:
    """Alpha-blend class colors over the frame, with a contour outline per
    class for crisp boundaries. Pure numpy/cv2 — unit-testable without a model."""
    out = frame_bgr.copy()
    for cls, name in enumerate(ANATOMY_INSTRUMENT.class_names):
        color = OVERLAY_COLORS[name]
        if color is None:
            continue
        m = mask == cls
        if not m.any():
            continue
        bgr = np.array(_bgr(color), dtype=np.float32)
        out[m] = (out[m] * (1 - alpha) + bgr * alpha).astype(np.uint8)
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, tuple(int(v) for v in bgr), 2)
    return out
