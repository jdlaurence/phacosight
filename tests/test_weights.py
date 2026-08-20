"""Weights distribution: manifest covers the deployment stack, and
scripts/download_weights.py fetches/verifies correctly (local mirror, no network)."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import download_weights as dw  # noqa: E402


def test_manifest_matches_deployment_stack():
    from analyze_phase_bulk import DEPLOY_CHECKPOINTS

    man = json.loads((REPO / "configs" / "weights_manifest.json").read_text())
    paths = [f["path"] for f in man["files"]]
    assert paths == sorted(DEPLOY_CHECKPOINTS)
    assert man["base_url"].startswith("https://")
    for f in man["files"]:
        assert len(f["sha256"]) == 64 and f["bytes"] > 0

    from app.overlay import CKPT  # the overlay model must ship in the bundle too
    assert str(CKPT.relative_to(REPO)) in paths


def _setup(tmp_path, monkeypatch, payload=b"model-weights"):
    rel = "runs/some_model/fold0/best.pt"
    mirror, repo = tmp_path / "mirror", tmp_path / "repo"
    (mirror / rel).parent.mkdir(parents=True)
    (mirror / rel).write_bytes(payload)
    (repo / "configs").mkdir(parents=True)
    (repo / "configs" / "weights_manifest.json").write_text(json.dumps({
        "base_url": str(mirror),
        "files": [{"path": rel, "bytes": len(payload),
                   "sha256": hashlib.sha256(payload).hexdigest()}],
    }))
    monkeypatch.setattr(dw, "REPO", repo)
    monkeypatch.setattr(dw, "MANIFEST", repo / "configs" / "weights_manifest.json")
    return mirror, repo, rel


def test_fetch_verify_idempotent(tmp_path, monkeypatch):
    mirror, repo, rel = _setup(tmp_path, monkeypatch)
    assert dw.missing_files(repo) == [rel]
    assert dw.fetch(None) == 1
    assert (repo / rel).read_bytes() == b"model-weights"
    assert dw.missing_files(repo) == []
    assert dw.fetch(None) == 0  # verified files are not re-downloaded


def test_fetch_rejects_bad_checksum(tmp_path, monkeypatch):
    mirror, repo, rel = _setup(tmp_path, monkeypatch)
    (mirror / rel).write_bytes(b"tampered")
    with pytest.raises(SystemExit, match="checksum mismatch"):
        dw.fetch(None)
    assert not (repo / rel).exists()  # nothing half-written left behind


def test_corrupt_local_file_is_refetched(tmp_path, monkeypatch):
    mirror, repo, rel = _setup(tmp_path, monkeypatch)
    (repo / rel).parent.mkdir(parents=True)
    (repo / rel).write_bytes(b"truncated")
    assert dw.fetch(None) == 1
    assert (repo / rel).read_bytes() == b"model-weights"
