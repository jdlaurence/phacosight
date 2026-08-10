#!/usr/bin/env python
"""Download Cataract-1K subsets from Synapse into data/.

Requires a Synapse account that has accepted the dataset's conditions of use,
and an auth token: https://www.synapse.org/#!PersonalAccessTokens
    pip install synapseclient
    export SYNAPSE_AUTH_TOKEN=...
    python scripts/download_data.py --sets segmentation phase
"""

import argparse
import os
from pathlib import Path

SYNAPSE_IDS = {
    # from README.md "Download" section
    "full": "syn53404507",          # complete Cataract-1K (89.9 GB)
    "phase": "syn53395146",         # phase recognition set (3.87 GB)
    "segmentation": "syn53395479",  # semantic segmentation set (4.74 GB)
    "lens_irregularity": "syn53395131",
    "pupil_reaction": "syn53395402",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sets", nargs="+", default=["segmentation", "phase"],
        choices=sorted(SYNAPSE_IDS),
    )
    parser.add_argument("--dest", default="data")
    args = parser.parse_args()

    try:
        import synapseclient
        import synapseutils
    except ImportError:
        raise SystemExit("pip install synapseclient  (and get a Synapse auth token)")

    token = os.environ.get("SYNAPSE_AUTH_TOKEN")
    if not token:
        raise SystemExit("Set SYNAPSE_AUTH_TOKEN (Synapse → Settings → Personal Access Tokens)")

    syn = synapseclient.Synapse()
    syn.login(authToken=token)
    dest = Path(args.dest)
    dest.mkdir(exist_ok=True)
    for name in args.sets:
        print(f"Downloading {name} ({SYNAPSE_IDS[name]}) → {dest / name}")
        synapseutils.syncFromSynapse(syn, SYNAPSE_IDS[name], path=str(dest / name))


if __name__ == "__main__":
    main()
