#!/usr/bin/env python
"""Rasterize Supervisely JSON polygons into training-mask PNGs (all 3 variants).

Faithful reimplementation of the upstream ``Dataset_codes/semantic segmentation
dataset codes/json_to_network_mask_*.py`` scripts (same 1024x768 canvas, same
draw order Cornea -> Pupil -> Lens -> instruments so later classes overwrite
earlier ones, same class titles), with three deliberate differences:

- one pass writes all three variants (``mask_anatomy_inst``,
  ``mask_instruments``, ``mask_instruments_MultiClass``);
- binary instrument masks store train ID 1 (not 255) per
  ``phacosight.labels.INSTRUMENT_BINARY``;
- the multiclass grouping uses the Title Case names actually present in the
  JSONs, mapped per ``phacosight.labels.INSTRUMENT_MULTICLASS`` (the
  upstream "rev" script matched lowercase names absent from this export, and
  had a misaligned cartridge/incision-knife mapping).

Stray ``cornea1``/``pupil1`` titles are ignored, exactly as upstream does;
every such frame also carries a proper Cornea/Pupil polygon.

Usage:
    python scripts/generate_masks.py \
        --data-root data/segmentation/Annotations/Images-and-Supervisely-Annotations
"""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

from PIL import Image, ImageDraw

SIZE = (1024, 768)

ANATOMY = {"Cornea": 1, "Pupil": 2, "Lens": 3}

# 10 Supervisely tool titles -> 6 grouped train IDs (see labels.INSTRUMENT_MULTICLASS)
INSTRUMENT_MULTICLASS_IDS = {
    "Slit Knife": 1,
    "Incision Knife": 1,
    "Gauge": 2,
    "Capsulorhexis Cystotome": 2,
    "Spatula": 2,
    "Phacoemulsification Tip": 3,
    "Irrigation-Aspiration": 4,
    "Lens Injector": 5,
    "Katena Forceps": 6,
    "Capsulorhexis Forceps": 6,
}

VARIANTS = ("mask_anatomy_inst", "mask_instruments", "mask_instruments_MultiClass")


def rasterize(json_path: Path) -> str | None:
    objects = json.loads(json_path.read_text())["objects"]
    polys = [
        (o["classTitle"], [tuple(pt) for pt in o["points"]["exterior"]]) for o in objects
    ]

    canvases = {v: Image.new("L", SIZE, 0) for v in VARIANTS}
    draws = {v: ImageDraw.Draw(img) for v, img in canvases.items()}

    # anatomy first, in fixed order, so instruments overwrite it
    for name, train_id in ANATOMY.items():
        for title, ext in polys:
            if title == name and len(ext) >= 2:
                draws["mask_anatomy_inst"].polygon(ext, fill=train_id)
    for title, ext in polys:
        multi_id = INSTRUMENT_MULTICLASS_IDS.get(title)
        if multi_id is not None and len(ext) >= 2:
            draws["mask_anatomy_inst"].polygon(ext, fill=4)
            draws["mask_instruments"].polygon(ext, fill=1)
            draws["mask_instruments_MultiClass"].polygon(ext, fill=multi_id)

    case_dir = json_path.parent.parent
    name = json_path.name[: -len(".json")]  # 'caseXXXX_NN.png'
    for variant, img in canvases.items():
        out = case_dir / variant / name
        out.parent.mkdir(exist_ok=True)
        img.save(out)

    unknown = {t for t, _ in polys} - set(ANATOMY) - set(INSTRUMENT_MULTICLASS_IDS)
    unknown -= {"cornea1", "pupil1"}
    return f"{json_path}: unknown titles {sorted(unknown)}" if unknown else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default="data/segmentation/Annotations/Images-and-Supervisely-Annotations",
        help="directory containing the case_* folders",
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    json_files = sorted(Path(args.data_root).glob("case_*/ann/*.json"))
    if not json_files:
        raise SystemExit(f"no case_*/ann/*.json under {args.data_root}")
    with Pool(args.workers) as pool:
        warnings = [w for w in pool.map(rasterize, json_files) if w]
    for w in warnings:
        print("WARNING", w)
    print(f"rasterized {len(json_files)} frames x {len(VARIANTS)} variants")


if __name__ == "__main__":
    main()
