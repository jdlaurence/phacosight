"""Single-command launcher: `python -m app` (or `phacosight` once installed).

Auto-detects hardware, reports what inference will run on, and serves the app.
All paths resolve relative to the repo unless overridden, so a fresh clone plus
this command is a working local deployment.

    python -m app                     # http://localhost:7860
    python -m app --port 8000 --host 0.0.0.0
    PHACOSIGHT_DATA_ROOT=/mnt/data/... python -m app
"""

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(prog="phacosight")
    parser.add_argument("--host", default="127.0.0.1",
                        help="bind address (0.0.0.0 to expose on the network)")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--data-root", default=None,
                        help="directory holding library/, phase/, full/ "
                             "(default: <repo>/data or $PHACOSIGHT_DATA_ROOT)")
    args = parser.parse_args()
    if args.data_root:
        os.environ["PHACOSIGHT_DATA_ROOT"] = args.data_root

    # Ops without an MPS kernel fall back to CPU instead of raising; must be
    # set before torch is first imported.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    import torch
    if torch.cuda.is_available():
        gpus = ", ".join(torch.cuda.get_device_name(i)
                         for i in range(torch.cuda.device_count()))
        print(f"[phacosight] inference device: CUDA ({gpus})")
    elif torch.backends.mps.is_available():
        print("[phacosight] inference device: MPS (Apple Silicon)")
    else:
        print("[phacosight] no GPU detected — browsing works normally; "
              "uploaded-video analysis will run on CPU (slow, ~10-20x realtime)")

    import uvicorn
    print(f"[phacosight] serving on http://{args.host}:{args.port}")
    uvicorn.run("app.server:app", host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
