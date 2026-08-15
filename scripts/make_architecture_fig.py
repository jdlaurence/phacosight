#!/usr/bin/env python
"""Publication-quality pipeline architecture figure (SVG + PDF + PNG).

NeurIPS-style: white ground, thin neutral boxes, one accent per subsystem,
dimension annotations on edges, frozen/trained markers.

    python scripts/make_architecture_fig.py --out docs/figures
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK = "#1a1917"
MUTED = "#6e6b64"
EDGE = "#b9b6ae"
BLUE = "#2a78d6"     # perception (frozen backbone)
ORANGE = "#d9641f"   # tool sensors (Stage-1 models)
GREEN = "#178a5a"    # temporal head (trained here)
VIOLET = "#6a5acd"   # decoding / grammar
FILL = "#ffffff"
FILL_SOFT = "#f7f6f3"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "svg.fonttype": "none",
})


def box(ax, x, y, w, h, title, sub=None, color=EDGE, fill=FILL, title_color=INK,
        lw=1.1, fs=9.0, sub_fs=7.0):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.05",
        facecolor=fill, edgecolor=color, linewidth=lw, zorder=2))
    if sub:
        ax.text(x + w / 2, y + h - 0.30, title, ha="center", va="center", fontsize=fs,
                color=title_color, zorder=3, linespacing=1.2)
        ax.text(x + w / 2, y + 0.30, sub, ha="center", va="center",
                fontsize=sub_fs, color=MUTED, zorder=3, linespacing=1.35)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=fs,
                color=title_color, zorder=3)
    return (x, y, w, h)


def arrow(ax, p0, p1, color=MUTED, lw=1.1, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=8, linewidth=lw, color=color,
        connectionstyle=f"arc3,rad={rad}", zorder=1, shrinkA=2, shrinkB=2))


def right(b):
    x, y, w, h = b
    return (x + w, y + h / 2)


def left(b):
    x, y, w, h = b
    return (x, y + h / 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="docs/figures")
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13.2, 6.6))
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 6.6)
    ax.axis("off")

    ROW = 4.10   # main-flow vertical center
    H = 1.06     # main-flow box height

    # ---- input & decode ----------------------------------------------------
    b_video = box(ax, 0.30, ROW - H / 2, 1.30, H, "surgical video",
                  sub="~7 min\n25–60 fps", fill=FILL_SOFT)
    b_decode = box(ax, 2.02, ROW - H / 2, 1.30, H, "decode",
                   sub="sequential\n1–5 fps")
    arrow(ax, right(b_video), left(b_decode))

    # ---- feature branches (stacked around the main row) --------------------
    b_dino = box(ax, 3.90, 4.80, 2.55, 1.16, "DINOv2-reg ViT-L  ❄",
                 sub="frozen self-supervised features\n518×392 · CLS ⊕ mean-patch",
                 color=BLUE, title_color=BLUE)
    b_seg = box(ax, 3.90, 2.24, 2.55, 1.16, "2 × SegFormer-B2  (Stage 1)",
                sub="anatomy 5-cls · tools 7-cls\nper-class max / mean / area",
                color=ORANGE, title_color=ORANGE)
    arrow(ax, right(b_decode), left(b_dino), rad=-0.22)
    arrow(ax, right(b_decode), left(b_seg), rad=0.22)

    b_cat = box(ax, 7.00, ROW - 0.42, 0.66, 0.84, "⊕", fs=14)
    arrow(ax, right(b_dino), (7.15, ROW + 0.40), rad=0.22)
    arrow(ax, right(b_seg), (7.15, ROW - 0.40), rad=-0.22)
    ax.text(6.82, 5.30, "[T, 2048]", fontsize=6.8, color=MUTED, ha="center")
    ax.text(6.78, 2.86, "[T, 36]", fontsize=6.8, color=MUTED, ha="center")

    # ---- temporal head & decoding ------------------------------------------
    b_head = box(ax, 8.10, ROW - H / 2, 2.20, H, "MS-TCN++ ensemble ●",
                 sub="4 folds × 3 seeds\nmean softmax → temperature",
                 color=GREEN, title_color=GREEN)
    arrow(ax, right(b_cat), left(b_head))
    ax.text(7.80, ROW + 0.68, "[T, 2084]", fontsize=6.8, color=MUTED, ha="center")

    b_dec = box(ax, 10.75, ROW - H / 2, 1.95, H, "Viterbi decoding",
                sub="transition grammar\nlearned from train folds",
                color=VIOLET, title_color=VIOLET)
    arrow(ax, right(b_head), left(b_dec))
    ax.text(10.48, ROW + 0.68, "[T, 13]", fontsize=6.8, color=MUTED, ha="center")

    # ---- output band -------------------------------------------------------
    b_out = box(ax, 5.40, 1.10, 7.30, 0.82,
                "phase timeline — 12 phases + idle, per-segment confidence & disagreement flags",
                fill=FILL_SOFT, lw=1.3, fs=8.6)
    arrow(ax, (b_dec[0] + b_dec[2] / 2, b_dec[1]), (b_dec[0] + b_dec[2] / 2, 1.92))

    # ---- products ----------------------------------------------------------
    prods = [
        ("feedback report", "durations vs cohort norms\nflagged events"),
        ("phase-indexed library", "1,000 surgeries searchable\nclip retrieval on demand"),
        ("irregularity gating", "pupil / IOL analysis\n(Stage 3)"),
    ]
    for i, (t, s) in enumerate(prods):
        x = 5.55 + i * 2.42
        b = box(ax, x, 0.06, 2.26, 0.86, t, sub=s, fs=7.8, sub_fs=6.2)
        arrow(ax, (x + 1.13, 1.10), (x + 1.13, 0.94))

    # ---- legend ------------------------------------------------------------
    ax.text(0.30, 2.90, "❄  frozen", fontsize=7.6, color=BLUE)
    ax.text(0.30, 2.62, "●  trained on Cataract-1K", fontsize=7.6, color=GREEN)
    ax.text(0.30, 2.34, "T = sampled frames", fontsize=7.6, color=MUTED)

    ax.text(0.30, 6.25, "Cataract surgery video analysis — phase-recognition pipeline",
            fontsize=11, color=INK, weight="bold")

    for ext in ("svg", "pdf", "png"):
        fig.savefig(out_dir / f"pipeline_architecture.{ext}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    print("wrote", out_dir / "pipeline_architecture.{svg,pdf,png}")


if __name__ == "__main__":
    main()
