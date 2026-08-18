"""Frame and segmental metrics for temporal phase segmentation.

Segmental metrics follow the standard action-segmentation definitions
(MS-TCN/ASFormer papers): edit score = normalized Levenshtein distance between
segment-label sequences; F1@k matches predicted to ground-truth segments of the
same class at IoU >= k.
"""

import numpy as np


def segments(labels: np.ndarray) -> list[tuple[int, int, int]]:
    """[(class, start_idx, end_idx_exclusive)] runs of a label array."""
    if len(labels) == 0:
        return []
    change = np.flatnonzero(np.diff(labels)) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(labels)]])
    return [(int(labels[s]), int(s), int(e)) for s, e in zip(starts, ends)]


def edit_score(pred: np.ndarray, true: np.ndarray) -> float:
    """Levenshtein similarity (0-100) between segment class sequences."""
    a = [c for c, _, _ in segments(pred)]
    b = [c for c, _, _ in segments(true)]
    D = np.zeros((len(a) + 1, len(b) + 1))
    D[:, 0] = np.arange(len(a) + 1)
    D[0, :] = np.arange(len(b) + 1)
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            D[i, j] = min(
                D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + (a[i - 1] != b[j - 1])
            )
    denom = max(len(a), len(b))
    return 100.0 * (1 - D[-1, -1] / denom) if denom else 100.0


def f1_at_k(pred: np.ndarray, true: np.ndarray, iou_threshold: float) -> tuple[int, int, int]:
    """(tp, fp, fn) of segment matches at the IoU threshold (accumulate over videos)."""
    p_segs = segments(pred)
    t_segs = segments(true)
    matched = np.zeros(len(t_segs), dtype=bool)
    tp = fp = 0
    for c, s, e in p_segs:
        best_iou, best_j = 0.0, -1
        for j, (tc, ts, te) in enumerate(t_segs):
            if tc != c or matched[j]:
                continue
            inter = max(0, min(e, te) - max(s, ts))
            union = max(e, te) - min(s, ts)
            iou = inter / union if union else 0.0
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_threshold:
            tp += 1
            matched[best_j] = True
        else:
            fp += 1
    fn = int((~matched).sum())
    return tp, fp, fn


class PhaseMetrics:
    """Accumulates per-video predictions; computes frame + segmental metrics."""

    def __init__(self, num_classes: int, class_names: tuple[str, ...], fps: float = 1.0):
        self.num_classes = num_classes
        self.class_names = class_names
        self.fps = fps
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
        self.f1k = {k: np.zeros(3, dtype=np.int64) for k in (0.10, 0.25, 0.50)}
        self.edits: list[float] = []
        self.seg_ratio: list[float] = []
        self.boundary_err: list[float] = []  # samples, matched segments at IoU>=0.5
        self.dur_err: dict[int, list[float]] = {c: [] for c in range(num_classes)}

    def update(self, pred: np.ndarray, true: np.ndarray) -> None:
        idx = true * self.num_classes + pred
        self.confusion += np.bincount(idx, minlength=self.num_classes**2).reshape(
            self.num_classes, self.num_classes
        )
        for k in self.f1k:
            self.f1k[k] += np.array(f1_at_k(pred, true, k))
        self.edits.append(edit_score(pred, true))
        n_true = max(1, len(segments(true)))
        self.seg_ratio.append(len(segments(pred)) / n_true)
        # boundary timing of matched segments (product metric: report timestamps)
        p_segs = segments(pred)
        for tc, ts, te in segments(true):
            best = max(
                ((min(e, te) - max(s, ts)) / (max(e, te) - min(s, ts)), s, e)
                for c, s, e in p_segs if c == tc
            ) if any(c == tc for c, _, _ in p_segs) else None
            if best and best[0] >= 0.5:
                self.boundary_err += [abs(best[1] - ts), abs(best[2] - te)]
        # per-phase total-duration error (product metric: report duration lines)
        for c in range(self.num_classes):
            true_dur = (true == c).sum()
            if true_dur:
                self.dur_err[c].append(abs(int((pred == c).sum()) - int(true_dur)))

    def compute(self) -> dict:
        tp = np.diag(self.confusion).astype(np.float64)
        support = self.confusion.sum(axis=1)
        prec = np.divide(tp, self.confusion.sum(axis=0), out=np.zeros_like(tp),
                         where=self.confusion.sum(axis=0) > 0)
        rec = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
        f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp),
                       where=(prec + rec) > 0)
        present = support > 0
        out = {
            "accuracy": float(tp.sum() / max(1, self.confusion.sum())),
            "macro_f1": float(f1[present].mean()),
            "per_class_f1": dict(zip(self.class_names, f1.tolist())),
            "per_class_support": dict(zip(self.class_names, support.astype(int).tolist())),
            "edit": float(np.mean(self.edits)),
            "seg_ratio": float(np.mean(self.seg_ratio)),
        }
        for k, (tp_k, fp_k, fn_k) in self.f1k.items():
            denom = 2 * tp_k + fp_k + fn_k
            out[f"f1@{int(k*100)}"] = float(100 * 2 * tp_k / denom) if denom else 0.0
        out["boundary_median_s"] = (
            float(np.median(self.boundary_err) / self.fps) if self.boundary_err else None
        )
        out["duration_mae_s"] = {
            self.class_names[c]: float(np.mean(v) / self.fps)
            for c, v in self.dur_err.items() if v
        }
        return out
