"""Web-app API tests over a synthetic data root — no dataset, videos, or GPU.

The app module is imported once per test module with PHACOSIGHT_DATA_ROOT
pointed at a tmp dir and PHACOSIGHT_NO_WORKER=1 (no inference worker, no
model loads).
"""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root → `app` package


def seg(phase, start, end, conf=0.95, dis=0.0):
    return {"phase": phase, "start_s": float(start), "end_s": float(end),
            "confidence": conf, "disagreement": dis}


def timeline(case, segments, **extra):
    return {"case": case, "duration_s": max(s["end_s"] for s in segments),
            "segments": segments, **extra}


@pytest.fixture(scope="module")
def app_env(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    tldir = root / "library" / "phase_timelines"
    tldir.mkdir(parents=True)

    fixtures = {
        "case_clean": timeline("case_clean", [
            seg("idle", 0, 5), seg("Incision", 5, 20),
            seg("Phacoemulsification", 20, 80, conf=0.92, dis=0.1),
            seg("idle", 80, 100)], source="library"),
        "case_flagged": timeline("case_flagged", [
            seg("Incision", 0, 10, conf=0.5),                 # low confidence
            seg("Phacoemulsification", 10, 50, dis=0.8)],     # high disagreement
            source="library"),
        "case_upload": timeline("case_upload", [seg("Incision", 0, 15, conf=0.99)],
                                source="uploaded", physician="Dr. Test",
                                surgery_date="2026-08-01"),
    }
    for name, d in fixtures.items():
        (tldir / f"{name}.json").write_text(json.dumps(d))
    (tldir / "truncated.json").write_text('{"case": "trunc"')       # malformed
    (tldir / "missing_keys.json").write_text('{"case": "nokeys"}')  # missing keys

    os.environ["PHACOSIGHT_DATA_ROOT"] = str(root)
    os.environ["PHACOSIGHT_NO_WORKER"] = "1"
    for mod in [m for m in sys.modules if m == "app" or m.startswith("app.")]:
        del sys.modules[mod]
    import app.server as server
    from fastapi.testclient import TestClient
    return TestClient(server.app), server, tldir


def test_videos_list_skips_malformed(app_env):
    client, _, _ = app_env
    rows = client.get("/api/videos").json()
    assert sorted(r["case"] for r in rows) == ["case_clean", "case_flagged", "case_upload"]
    clean = next(r for r in rows if r["case"] == "case_clean")
    assert set(clean) >= {"case", "duration_s", "n_segments", "flag_fraction",
                          "source", "physician", "surgery_date", "has_gt"}
    assert clean["flag_fraction"] == 0.0
    flagged = next(r for r in rows if r["case"] == "case_flagged")
    assert flagged["flag_fraction"] == 1.0


def test_get_video_metrics_and_no_video(app_env):
    client, _, _ = app_env
    d = client.get("/api/videos/case_clean").json()
    assert d["has_video"] is False        # no mp4s exist in the tmp root
    assert d["metrics"]["n_segments"] == 4
    assert d["ground_truth"] is None
    assert client.get("/api/videos/does_not_exist").status_code == 404


def test_search_excludes_flagged_segments(app_env):
    client, _, _ = app_env
    res = client.get("/api/search", params={"phase": "Phacoemulsification",
                                            "min_conf": 0.7}).json()
    # case_flagged's segment has conf .95 but disagreement .8 → excluded
    assert [h["case"] for h in res["hits"]] == ["case_clean"]
    res = client.get("/api/search", params={"phase": "Incision", "min_conf": 0.1}).json()
    # min_conf below the flag gate must not resurrect low-confidence segments
    assert sorted(h["case"] for h in res["hits"]) == ["case_clean", "case_upload"]


def test_norms_exclude_uploads(app_env):
    _, server, _ = app_env
    assert server.norms.n_videos == 2                    # upload excluded
    assert server.norms.dist["Incision"] == [15.0]       # flagged Incision gated out
    assert "Phacoemulsification" in server.norms.videos_with


def test_case_name_validation(app_env):
    client, _, _ = app_env
    for bad in ("../../../etc/passwd", "a/b", "x" * 81, ""):
        assert client.get("/api/frame", params={"case": bad, "t": 0}).status_code == 400
    assert client.get("/api/clip", params={"case": "../x", "start": 0,
                                           "end": 5}).status_code == 400


def test_frame_and_clip_guards(app_env):
    client, _, _ = app_env
    # valid name, but no video file on disk
    assert client.get("/api/frame", params={"case": "case_clean", "t": 0}).status_code == 404
    assert client.get("/api/clip", params={"case": "case_clean", "start": 5,
                                           "end": 5}).status_code == 400
    assert client.get("/api/clip", params={"case": "case_clean", "start": 0,
                                           "end": 999}).status_code == 400


def test_physicians_and_progress(app_env):
    client, _, _ = app_env
    docs = client.get("/api/physicians").json()
    assert docs == [{"name": "Dr. Test", "n_videos": 1, "last_date": "2026-08-01"}]
    pr = client.get("/api/progress", params={"physician": "Dr. Test"}).json()
    assert len(pr["videos"]) == 1
    assert pr["videos"][0]["case"] == "case_upload"


def test_phase_stats(app_env):
    client, _, _ = app_env
    st = client.get("/api/phase_stats", params={"phase": "Incision"}).json()
    assert st["n_videos"] == 2 and st["total"]["p50"] == 15.0
    assert client.get("/api/phase_stats",
                      params={"phase": "Lens Implantation"}).status_code == 404


def test_upload_and_jobs_queue(app_env):
    client, server, _ = app_env
    r = client.post("/api/upload",
                    files={"file": ("my video<x>.mp4", b"\x00" * 64, "video/mp4")},
                    data={"physician": "Dr. Test", "surgery_date": "2026-08-19",
                          "operator": "resident"})
    assert r.status_code == 200
    case = r.json()["case"]
    assert server.CASE_RE.fullmatch(case)                 # hostile stem sanitized
    jobs = client.get("/api/jobs").json()
    assert jobs and jobs[0]["case"] == case and jobs[0]["status"] == "queued"
    assert "path" not in jobs[0] and "meta" not in jobs[0]
    assert client.post("/api/upload",
                       files={"file": ("x.mov", b"\x00", "video/mp4")}).status_code == 400


def test_delete_and_reanalyze_upload_only(app_env):
    client, _, tldir = app_env
    # library cases are protected
    assert client.delete("/api/videos/case_clean").status_code == 403
    assert client.post("/api/reanalyze/case_clean").status_code == 403
    # uploaded case with its mp4 gone: reanalyze 404s, delete succeeds
    assert client.post("/api/reanalyze/case_upload").status_code == 404
    assert client.delete("/api/videos/case_upload").status_code == 200
    assert not (tldir / "case_upload.json").exists()
    assert client.get("/api/videos/case_upload").status_code == 404


def test_norms_rebuild_picks_up_new_timeline(app_env):
    client, server, tldir = app_env
    (tldir / "case_new.json").write_text(json.dumps(
        timeline("case_new", [seg("Incision", 0, 30)], source="library")))
    server.rebuild_norms()
    assert server.norms.n_videos == 3
    assert server.norms.dist["Incision"] == [15.0, 30.0]
    rows = client.get("/api/videos").json()
    assert any(r["case"] == "case_new" for r in rows)


def test_overlay_legend_and_endpoint_guards(app_env):
    client, server, _ = app_env
    lg = client.get("/api/overlay/legend").json()
    assert isinstance(lg["available"], bool)
    assert {c["name"] for c in lg["classes"]} == {"cornea", "pupil", "lens", "instrument"}
    assert all(c["color"].startswith("#") for c in lg["classes"])
    r = client.get("/api/overlay", params={"case": "case_clean", "t": 1.0})
    # checkpoint present: 404 (no video file); bare clone: 503 (no checkpoint)
    assert r.status_code == (404 if server.overlay_service.available() else 503)
    assert client.get("/api/overlay",
                      params={"case": "../x", "t": 0}).status_code in (400, 503)


def test_render_overlay_blend():
    import numpy as np
    from app.overlay import ALPHA, OVERLAY_COLORS, render_overlay

    frame = np.full((60, 80, 3), 100, dtype=np.uint8)
    mask = np.zeros((60, 80), dtype=np.uint8)
    mask[10:30, 10:30] = 2  # pupil
    out = render_overlay(frame, mask)
    assert out.shape == frame.shape
    assert (out[40:, 40:] == 100).all()            # background untouched
    assert not (out[15:25, 15:25] == 100).all()    # masked interior blended
    b, g, r = out[20, 20]
    hexc = OVERLAY_COLORS["pupil"]
    exp = [round(100 * (1 - ALPHA) + int(hexc[i:i + 2], 16) * ALPHA) for i in (5, 3, 1)]
    assert abs(int(b) - exp[0]) <= 1 and abs(int(g) - exp[1]) <= 1 and abs(int(r) - exp[2]) <= 1


def test_overlay_model_integration():
    """Runs the real deployed checkpoint on a synthetic frame (skipped on a
    bare clone). Validates load + preprocessing + mask shape, incl. the 16:9
    center-crop path."""
    import numpy as np
    from app.overlay import CKPT, OverlayService

    if not CKPT.exists():
        pytest.skip("segmentation checkpoint not present")
    svc = OverlayService()
    frame = np.random.default_rng(0).integers(0, 255, (768, 1366, 3), dtype=np.uint8)
    mask, cropped = svc.mask(frame)
    assert cropped.shape == (768, 1024, 3)         # 16:9 → 4:3 center crop
    assert mask.shape == cropped.shape[:2]
    assert mask.max() <= 4
