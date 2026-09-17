"""GOT-10k tracking metrics: IoU, success-plot AUC, Average Overlap, SR.

The paper reports tracking quality as **AUC** (area under the OPE success
plot). We compute it exactly the way the GOT-10k / OTB protocol does:

  * per-frame IoU (overlap) between predicted and ground-truth boxes,
  * success rate S(tau) = fraction of frames with IoU > tau,
  * AUC = mean of S(tau) over tau in [0, 1] (21 thresholds),

plus the GOT-10k headline numbers Average Overlap (AO = mean IoU) and success
rates SR@0.5 / SR@0.75. Frames flagged absent/occluded can be excluded via a
mask. All boxes are xyxy; coords may be pixel or normalised as long as pred and
gt share the same space.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-row IoU of two [N,4] xyxy box arrays."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    x0 = np.maximum(a[:, 0], b[:, 0])
    y0 = np.maximum(a[:, 1], b[:, 1])
    x1 = np.minimum(a[:, 2], b[:, 2])
    y1 = np.minimum(a[:, 3], b[:, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a + area_b - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def success_auc(overlaps: np.ndarray, n_thresholds: int = 21) -> float:
    """AUC of the OPE success plot (mean success rate over IoU thresholds)."""
    overlaps = np.asarray(overlaps, dtype=np.float64)
    if overlaps.size == 0:
        return 0.0
    taus = np.linspace(0.0, 1.0, n_thresholds)
    succ = (overlaps[:, None] > taus[None, :]).mean(axis=0)
    return float(succ.mean())


def sequence_metrics(
    pred_boxes: Sequence,
    gt_boxes: Sequence,
    valid: Optional[Sequence] = None,
) -> Dict[str, float]:
    """Metrics for one sequence.

    Args:
        pred_boxes, gt_boxes: [T,4] xyxy in a shared coordinate space.
        valid: optional [T] bool mask of frames to score (drops absent frames).
    Returns dict with auc, ao, sr50, sr75, n_frames.
    """
    pred = np.asarray(pred_boxes, dtype=np.float64).reshape(-1, 4)
    gt = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)
    n = min(len(pred), len(gt))
    pred, gt = pred[:n], gt[:n]
    ov = iou_xyxy(pred, gt)
    if valid is not None:
        mask = np.asarray(valid, dtype=bool)[:n]
        ov = ov[mask]
    if ov.size == 0:
        return {"auc": 0.0, "ao": 0.0, "sr50": 0.0, "sr75": 0.0, "n_frames": 0}
    return {
        "auc": success_auc(ov),
        "ao": float(ov.mean()),
        "sr50": float((ov > 0.5).mean()),
        "sr75": float((ov > 0.75).mean()),
        "n_frames": int(ov.size),
    }


def aggregate_metrics(per_seq: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """Average per-sequence metrics (GOT-10k reports the per-sequence mean)."""
    seqs = [m for m in per_seq if m.get("n_frames", 0) > 0]
    if not seqs:
        return {"auc": 0.0, "ao": 0.0, "sr50": 0.0, "sr75": 0.0, "n_seqs": 0}
    keys = ("auc", "ao", "sr50", "sr75")
    out = {k: float(np.mean([m[k] for m in seqs])) for k in keys}
    out["n_seqs"] = len(seqs)
    return out
