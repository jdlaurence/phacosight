"""CPU tests: phase timelines, segmental metrics, temporal heads."""

import numpy as np
import pandas as pd
import pytest
import torch

from cataract_video.phase.heads import HEADS, mstcn_loss
from cataract_video.phase.metrics import PhaseMetrics, edit_score, f1_at_k, segments
from cataract_video.phase.timeline import NUM_CLASSES, PHASE_ID, frame_labels, load_segments


def test_frame_labels_rasterization(tmp_path):
    csv = tmp_path / "ann.csv"
    pd.DataFrame(
        {"caseId": [1, 1], "comment": ["Incision", "Capsulorhexis"],
         "frame": [0, 0], "endFrame": [0, 0], "sec": [1.0, 4.0], "endSec": [2.0, 6.0]}
    ).to_csv(csv, index=False)
    segs = load_segments(csv)
    labels = frame_labels(segs, duration_sec=8.0, fps=2.0)
    assert len(labels) == 16
    assert labels[0] == 0  # pre-surgery idle
    assert labels[2] == PHASE_ID["Incision"] and labels[3] == PHASE_ID["Incision"]
    assert labels[4] == 0  # gap idle
    assert set(labels[8:12]) == {PHASE_ID["Capsulorhexis"]}
    assert labels[-1] == 0  # post-surgery idle


def test_load_segments_rejects_unknown_phase(tmp_path):
    csv = tmp_path / "bad.csv"
    pd.DataFrame({"comment": ["NotAPhase"], "sec": [0.0], "endSec": [1.0]}).to_csv(csv, index=False)
    with pytest.raises(ValueError):
        load_segments(csv)


def test_segments_and_edit():
    a = np.array([0, 0, 1, 1, 1, 2, 2, 0])
    assert segments(a) == [(0, 0, 2), (1, 2, 5), (2, 5, 7), (0, 7, 8)]
    assert edit_score(a, a) == 100.0
    frag = np.array([0, 1, 0, 1, 1, 2, 2, 0])  # over-segmented
    assert edit_score(frag, a) < 100.0


def test_f1_at_k_perfect_and_shifted():
    true = np.array([0] * 10 + [1] * 10 + [2] * 10)
    tp, fp, fn = f1_at_k(true, true, 0.5)
    assert (tp, fp, fn) == (3, 0, 0)
    shifted = np.array([0] * 12 + [1] * 10 + [2] * 8)  # still >=50% IoU per segment
    tp, fp, fn = f1_at_k(shifted, true, 0.5)
    assert tp == 3 and fn == 0


def test_phase_metrics_aggregate():
    m = PhaseMetrics(3, ("idle", "a", "b"))
    true = np.array([0] * 5 + [1] * 5 + [2] * 5)
    m.update(true, true)
    out = m.compute()
    assert out["accuracy"] == 1.0 and out["macro_f1"] == 1.0
    assert out["edit"] == 100.0 and out["f1@50"] == 100.0
    assert out["seg_ratio"] == 1.0


@pytest.mark.parametrize("name", sorted(HEADS))
def test_heads_shapes_and_loss(name):
    torch.manual_seed(0)
    head = HEADS[name](in_dim=32, num_classes=NUM_CLASSES)
    x = torch.randn(2, 50, 32)
    out = head(x)
    assert out.ndim == 4 and out.shape[1:] == (2, 50, NUM_CLASSES)
    target = torch.randint(0, NUM_CLASSES, (2, 50))
    loss = mstcn_loss(out, target)
    loss.backward()
    assert torch.isfinite(loss)


def test_decoding_smooth_and_viterbi():
    from cataract_video.phase.decoding import mode_smooth, transition_matrix, viterbi

    noisy = np.array([1, 1, 2, 1, 1, 0, 0, 0, 3, 0, 0])
    sm = mode_smooth(noisy, window=3)
    assert sm[2] == 1 and sm[8] == 0  # isolated flickers removed

    # grammar learned from sticky sequences keeps Viterbi from flickering
    train = [np.array([0] * 20 + [1] * 20 + [2] * 20)]
    log_trans = transition_matrix(train, 3)
    probs = np.full((30, 3), 0.1)
    probs[:10, 0] = 0.8
    probs[10:, 1] = 0.8
    probs[15, 2] = 0.9  # single-frame spurious spike
    probs[15, 1] = 0.05
    path = viterbi(np.log(probs), log_trans)
    assert path[15] == 1  # spike suppressed by sticky grammar
    assert (path[:10] == 0).all() and (path[10:] == 1).all()
