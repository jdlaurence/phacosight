#!/usr/bin/env python
"""Fetch the trained deployment checkpoints (~260 MB) into runs/.

A fresh clone has code and the decoding grammar but no model weights — this
script downloads the 14 checkpoints the deployment stack loads (2 SegFormer-B2
segmentation folds + 12 MS-TCN++ phase heads), verifies each against the
sha256 in the committed manifest (configs/weights_manifest.json), and places
them at the exact runs/ paths the code expects. Stdlib only; idempotent
(already-verified files are skipped).

    python scripts/download_weights.py                 # fetch from the manifest's base_url
    python scripts/download_weights.py --base-url DIR_OR_URL   # e.g. a local mirror

Maintainers, after changing the deployment stack (DEPLOY_CHECKPOINTS in
analyze_phase_bulk.py): regenerate with --make-manifest, upload the runs/
files (paths preserved) to the manifest's base_url, and commit the manifest.
"""

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "configs" / "weights_manifest.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def missing_files(repo: Path = REPO) -> list[str]:
    """Manifest entries not present locally (no checksum pass — cheap, used by
    the app to fail early with instructions instead of a mid-job crash)."""
    return [f["path"] for f in load_manifest()["files"] if not (repo / f["path"]).exists()]


def fetch(base_url: str | None) -> int:
    man = load_manifest()
    base = (base_url or man["base_url"]).rstrip("/")
    got = 0
    for entry in man["files"]:
        rel, want = entry["path"], entry["sha256"]
        dst = REPO / rel
        if dst.exists() and _sha256(dst) == want:
            print(f"  ok       {rel}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + ".part")
        src = f"{base}/{rel}"
        print(f"  fetching {rel} ({entry['bytes'] / 1e6:.0f} MB)")
        if src.startswith(("http://", "https://")):
            with urllib.request.urlopen(src) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f)
        else:  # local mirror directory
            shutil.copyfile(Path(base) / rel, tmp)
        if _sha256(tmp) != want:
            tmp.unlink()
            raise SystemExit(f"checksum mismatch for {rel} from {src} — aborting "
                             "(corrupt download or wrong --base-url)")
        tmp.replace(dst)  # atomic: never leave a half-written checkpoint
        got += 1
    print(f"done: {got} downloaded, {len(man['files']) - got} already present")
    return got


def make_manifest(base_url: str | None) -> None:
    sys.path.insert(0, str(REPO / "scripts"))
    from analyze_phase_bulk import DEPLOY_CHECKPOINTS  # deferred: needs torch etc.
    old = load_manifest() if MANIFEST.exists() else {}
    files = []
    for rel in DEPLOY_CHECKPOINTS:
        p = REPO / rel
        files.append({"path": rel, "bytes": p.stat().st_size, "sha256": _sha256(p)})
        print(f"  {rel}")
    MANIFEST.write_text(json.dumps({
        "base_url": base_url or old.get("base_url", ""),
        "stack": "E7-b deployment: SegFormer-B2 fold0 (anatomy, multiclass) + "
                 "12x tools-fusion MS-TCN++ 1fps heads",
        "files": files,
    }, indent=1) + "\n")
    print(f"wrote {MANIFEST} ({sum(f['bytes'] for f in files) / 1e6:.0f} MB total)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", help="override the manifest's download source "
                                           "(http(s) URL or local directory)")
    parser.add_argument("--make-manifest", action="store_true",
                        help="maintainer: hash local checkpoints and (re)write the manifest")
    args = parser.parse_args()
    if args.make_manifest:
        make_manifest(args.base_url)
    else:
        fetch(args.base_url)


if __name__ == "__main__":
    main()
