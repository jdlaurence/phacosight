"""Timeline decoding: mode smoothing and Viterbi with a learned grammar.

The transition matrix is always *estimated from training folds* — cataract
surgery repeats phases routinely (viscoelastic, tonifying), so a hand-authored
left-to-right grammar would be wrong. Class IDs are categorical: smoothing uses
a sliding-window mode, never a median (which would invent classes between IDs).
"""

import numpy as np


def mode_smooth(pred: np.ndarray, window: int = 5) -> np.ndarray:
    """Sliding-window majority vote (odd window)."""
    assert window % 2 == 1
    half = window // 2
    padded = np.pad(pred, half, mode="edge")
    out = np.empty_like(pred)
    for i in range(len(pred)):
        out[i] = np.bincount(padded[i : i + window]).argmax()
    return out


def transition_matrix(label_seqs: list[np.ndarray], num_classes: int,
                      smoothing: float = 1.0) -> np.ndarray:
    """Row-normalized frame-to-frame transition counts (Laplace-smoothed), log."""
    counts = np.full((num_classes, num_classes), smoothing)
    for y in label_seqs:
        np.add.at(counts, (y[:-1], y[1:]), 1.0)
    return np.log(counts / counts.sum(axis=1, keepdims=True))


def viterbi(log_probs: np.ndarray, log_trans: np.ndarray,
            log_prior: np.ndarray | None = None) -> np.ndarray:
    """MAP path through [T, C] frame log-probabilities under the grammar."""
    T, C = log_probs.shape
    if log_prior is None:
        log_prior = np.full(C, -np.log(C))
    delta = log_prior + log_probs[0]
    back = np.zeros((T, C), dtype=np.int64)
    for t in range(1, T):
        scores = delta[:, None] + log_trans  # [from, to]
        back[t] = scores.argmax(axis=0)
        delta = scores.max(axis=0) + log_probs[t]
    path = np.empty(T, dtype=np.int64)
    path[-1] = delta.argmax()
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path
