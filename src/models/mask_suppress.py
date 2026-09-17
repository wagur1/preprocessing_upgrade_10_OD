"""Detector-mask protection + background suppression (the R0 preprocessor).

Philosophy, and why it is not another learned editor: every learned preprocessor
this project tried on images lost. The AR checkpoint costs ~12% more bits at
equal detection mAP, and the detection-trained one destroys 25% of the
detector's mAP *before the codec is involved* (mAP(pre)/mAP(x) = 0.75). Both
failures share a mechanism — a learned module may ADD or RESHAPE structure, and
detection mAP does not reward that. The detection literature's large numbers come
from the opposite move: remove background, keep objects (ROI-Packing −44%,
dual-region JPEG −26%). That needs no parameters at all.

So: the frozen detector runs on the SOURCE image (encoder-side analysis — the
encoder has the image and may analyse it freely; the decoder needs no side
information), its boxes are dilated into a protection mask, and everything
outside the mask is heavily blurred. Object regions pass through untouched.

Both functions are deliberately stateless and parameter-free: there is nothing to
overfit the training proxy with, and nothing to select.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def protect_mask(boxes, scores, labels, size: int, score_thresh: float = 0.5,
                 dilate: float = 0.15) -> torch.Tensor:
    """[1,1,S,S] mask, 1 = protect. Boxes are xyxy in the (already resized) frame.

    ``dilate`` is a fraction of each box's own size, so small objects get a
    proportionally small margin. Boxes below ``score_thresh`` are NOT protected:
    a detection the detector is unsure about should not buy bit budget.

    The mask is allocated on the BOXES' device — the detector returns CUDA
    tensors at eval time while the image may be on either device, and a CPU mask
    against a CUDA frame is a RuntimeError that a CPU-only local test cannot see
    (it cost one Kaggle cycle to learn that here).
    """
    boxes_t = torch.as_tensor(boxes)
    scores_t = torch.as_tensor(scores).reshape(-1)
    m = torch.zeros(1, 1, size, size, device=boxes_t.device)
    keep = scores_t >= score_thresh
    for b in boxes_t.reshape(-1, 4)[keep]:
        x1, y1, x2, y2 = [float(v) for v in b]
        w, h = x2 - x1, y2 - y1
        x1, x2 = x1 - dilate * w, x2 + dilate * w
        y1, y2 = y1 - dilate * h, y2 + dilate * h
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(size, int(np.ceil(x2))), min(size, int(np.ceil(y2)))
        if x2 > x1 and y2 > y1:
            m[:, :, y1:y2, x1:x2] = 1.0
    return m


def suppress(x: torch.Tensor, mask: torch.Tensor, sigma: float) -> torch.Tensor:
    """Heavy blur outside the mask; protected regions are byte-identical.

    Separable Gaussian, depthwise (``groups=C``) so RGB is blurred channel-wise.
    A 1-channel kernel against a 3-channel input raises a RuntimeError, which is
    the shape of the bug that cost one debugging cycle in the probe — hence the
    explicit groups argument and the test that pins it.
    """
    if sigma <= 0:
        return x
    k = int(2 * round(2 * sigma) + 1)
    c = x.shape[1]
    ax = torch.arange(k, dtype=x.dtype, device=x.device) - (k - 1) / 2
    g = torch.exp(-ax.pow(2) / (2 * sigma * sigma))
    g = g / g.sum()
    gx = g.view(1, 1, k, 1).expand(c, 1, k, 1).contiguous()
    gy = g.view(1, 1, 1, k).expand(c, 1, 1, k).contiguous()
    b, _, t, h, w = x.shape
    flat = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    pad = k // 2
    blur = F.conv2d(F.pad(flat, (pad, pad, 0, 0), mode="reflect"), gx, groups=c)
    blur = F.conv2d(F.pad(blur, (0, 0, pad, pad), mode="reflect"), gy, groups=c)
    blur = blur.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
    m = mask.expand_as(x)
    return x * m + blur * (1 - m)


def mask_from_detections(det_out: dict, size: int, score_thresh: float = 0.5,
                         dilate: float = 0.15) -> torch.Tensor:
    """Convenience: build the mask from one entry of the detector's output."""
    return protect_mask(det_out["boxes"], det_out["scores"], det_out["labels"],
                        size, score_thresh, dilate)
