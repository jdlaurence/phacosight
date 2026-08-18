"""Download the CATARACTS dataset (IEEE DataPort open access) onto the data mount.

IEEE DataPort's S3 bucket rejects anonymous requests and direct AWS access is
subscriber-only, but the dataset itself is open access via the website. One-time
manual step (any browser):

1. Log in with a free IEEE account at https://ieee-dataport.org/open-access/cataracts
   ("IEEE Membership is not required").
2. For each file you want, right-click its download button and copy the link —
   a presigned, time-limited S3 URL — into ``data/cataracts/presigned_urls.txt``,
   one URL per line (blank lines and ``#`` comments are ignored).
3. Run::

       .venv/bin/python scripts/download_cataracts.py            # download
       .venv/bin/python scripts/download_cataracts.py --extract  # + unzip

For phase augmentation we need ``videos.zip`` (117.8 GB) and ``ground_truth.zip``
(18.6 MB — includes the CATARACTS_2020 phase annotations); ``evaluation_scripts.zip``
is tiny and worth grabbing. ``images.zip`` (691 GB, the 2018 segmentation frames)
is not needed for the phase work.

Downloads resume if interrupted (``curl -C -``). Presigned links expire after a
few hours — if a download dies with an auth error, paste fresh links and rerun;
completed files are skipped and partial files resume.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.parse import urlparse

DEFAULT_DIR = Path("data/cataracts")


def read_urls(path: Path) -> list[str]:
    if not path.exists():
        raise SystemExit(
            f"{path} not found — paste presigned links there (see module docstring)"
        )
    urls = [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not urls:
        raise SystemExit(f"{path} contains no URLs")
    return urls


def download(url: str, out_dir: Path) -> Path:
    name = Path(urlparse(url).path).name
    final = out_dir / name
    part = out_dir / (name + ".part")
    if final.exists():
        print(f"[skip] {name} already downloaded")
        return final
    print(f"[get ] {name}")
    cmd = [
        "curl", "-L", "--fail", "--retry", "5", "--retry-delay", "10",
        "-C", "-", "-o", str(part), url,
    ]
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise SystemExit(
            f"curl failed for {name} (exit {proc.returncode}) — if this is an "
            "AccessDenied/expired error, paste fresh presigned links and rerun; "
            "the partial file will resume."
        )
    part.replace(final)
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urls-file", type=Path, default=DEFAULT_DIR / "presigned_urls.txt")
    parser.add_argument("--out", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--extract", action="store_true",
                        help="unzip completed archives into --out (skips existing files)")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    got = [download(u, args.out) for u in read_urls(args.urls_file)]

    for f in got:
        if f.suffix == ".zip" and not zipfile.is_zipfile(f):
            print(f"[warn] {f.name} is not a valid zip (truncated download?)", file=sys.stderr)

    if args.extract:
        for f in got:
            if f.suffix != ".zip" or not zipfile.is_zipfile(f):
                continue
            print(f"[unzip] {f.name}")
            subprocess.run(["unzip", "-q", "-n", str(f), "-d", str(args.out)], check=True)

    print(f"done — {len(got)} file(s) in {args.out}")


if __name__ == "__main__":
    main()
