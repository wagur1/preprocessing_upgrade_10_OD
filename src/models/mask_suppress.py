"""Parameter-free detector-mask protection and Gaussian background suppression.

Masks are encoder-side analysis; protected pixels pass through unchanged.
Parameter-free does not imply held-out evaluation or immunity to tuning bias.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def protect_mask(boxes, scores, labels, size: int, score_thresh: float = 0.5,
                 dilate: float = 0.15) -> torch.Tensor:
    """Return [1,1,S,S] on boxes' device and floating dtype (float32 for ints).

    Boxes are finite xyxy coordinates in the resized frame. Degenerate boxes
    are ignored; valid boxes are dilated, rounded outward and clipped.
    """
    if size <= 0 or not math.isfinite(score_thresh) or not 0 <= score_thresh <= 1:
        raise ValueError("size must be positive and score_thresh in [0, 1]")
    if not math.isfinite(dilate) or dilate < 0:
        raise ValueError("dilate must be finite and nonnegative")
    boxes_t = torch.as_tensor(boxes)
    if boxes_t.numel() % 4:
        raise ValueError("boxes must contain xyxy quadruples")
    boxes_t = boxes_t.reshape(-1, 4)
    dtype = boxes_t.dtype if boxes_t.is_floating_point() else torch.float32
    scores_t = torch.as_tensor(scores, device=boxes_t.device).reshape(-1)
    labels_t = torch.as_tensor(labels, device=boxes_t.device).reshape(-1)
    if len(boxes_t) != len(scores_t) or len(boxes_t) != len(labels_t):
        raise ValueError("boxes, scores and labels must have matching lengths")
    if not all(torch.isfinite(v).all() for v in (boxes_t, scores_t, labels_t)):
        raise ValueError("detections must be finite")
    m = torch.zeros(1, 1, size, size, device=boxes_t.device, dtype=dtype)
    for b in boxes_t[scores_t >= score_thresh]:
        x1, y1, x2, y2 = [float(v) for v in b]
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            continue
        x1, x2 = x1 - dilate * w, x2 + dilate * w
        y1, y2 = y1 - dilate * h, y2 + dilate * h
        x1, y1 = max(0, math.floor(x1)), max(0, math.floor(y1))
        x2, y2 = min(size, math.ceil(x2)), min(size, math.ceil(y2))
        if x2 > x1 and y2 > y1:
            m[:, :, y1:y2, x1:x2] = 1
    return m


def suppress(x: torch.Tensor, mask: torch.Tensor, sigma: float) -> torch.Tensor:
    """Blur unprotected pixels of floating [B,C,T,H,W] input.

    Accept [H,W], [B,1,H,W] (shared over time), or broadcastable
    [B,1,T,H,W] masks. Singleton B/T axes broadcast independently. Masks move
    to x's device/dtype. Inputs must be finite and masks in [0,1].

    Kernel size is 2*round(2*sigma)+1, matching torchvision gaussian_blur with
    that kernel and sigma. Each axis uses reflection padding when legal, else
    replicate padding (including singleton images). Half precision is computed
    in float32 to support CPU padding/convolution and retain finite results.
    """
    if x.ndim != 5 or not x.is_floating_point() or any(d == 0 for d in x.shape):
        raise ValueError("x must be nonempty floating [B,C,T,H,W]")
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and nonnegative")
    if not torch.isfinite(x).all():
        raise ValueError("x must be finite")
    m = torch.as_tensor(mask, device=x.device)
    if not torch.isfinite(m).all() or ((m < 0) | (m > 1)).any():
        raise ValueError("mask must be finite and in [0,1]")
    if m.ndim == 2:
        m = m[None, None, None]
    elif m.ndim == 4:
        m = m.unsqueeze(2)
    if m.ndim != 5 or m.shape[1] != 1 or m.shape[-2:] != x.shape[-2:]:
        raise ValueError("mask must be [H,W], [B,1,H,W], or [B,1,T,H,W]")
    if m.shape[0] not in (1, x.shape[0]) or m.shape[2] not in (1, x.shape[2]):
        raise ValueError("mask B/T axes must be singleton or match x")
    m = m.to(dtype=x.dtype)
    if sigma == 0:
        return x
    b, c, t, h, w = x.shape
    work_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    flat = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w).to(work_dtype)
    k = int(2 * round(2 * sigma) + 1)
    ax = torch.arange(k, dtype=work_dtype, device=x.device) - k // 2
    g = torch.exp(-0.5 * (ax / sigma).square())
    g = g / g.sum()
    vertical = g.view(1, 1, k, 1).expand(c, 1, k, 1).contiguous()
    horizontal = g.view(1, 1, 1, k).expand(c, 1, 1, k).contiguous()
    pad = k // 2
    blur = F.conv2d(F.pad(flat, (0, 0, pad, pad),
                         mode="reflect" if pad < h else "replicate"), vertical, groups=c)
    blur = F.conv2d(F.pad(blur, (pad, pad, 0, 0),
                         mode="reflect" if pad < w else "replicate"), horizontal, groups=c)
    blur = blur.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4).to(x.dtype)
    # Select fully protected pixels explicitly, preserving their exact bits.
    return torch.where(m == 1, x, x * m + blur * (1 - m))


def mask_from_detections(det_out: dict, size: int, score_thresh: float = 0.5,
                         dilate: float = 0.15) -> torch.Tensor:
    """Build the mask from one entry of the detector's output."""
    return protect_mask(det_out["boxes"], det_out["scores"], det_out["labels"],
                        size, score_thresh, dilate)


def fill_outside(x: torch.Tensor, mask: torch.Tensor, value: float) -> torch.Tensor:
    """Replace unprotected pixels with a constant; protected pixels are bit-exact.

    Same mask conventions as `suppress`. The mask is expected binary (1 =
    protect); intermediate values interpolate between the frame and `value`.
    `value` must lie in [0, 1] to stay inside the range a standard codec expects
    from a [0, 1] float frame — the aggressive end of the published family
    (Li & Rhee's "NROI MASK", ROI-Packing's discard).
    """
    if x.ndim != 5 or not x.is_floating_point() or any(d == 0 for d in x.shape):
        raise ValueError("x must be nonempty floating [B,C,T,H,W]")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("value must be finite and in [0,1]")
    if not torch.isfinite(x).all():
        raise ValueError("x must be finite")
    m = torch.as_tensor(mask, device=x.device)
    if not torch.isfinite(m).all() or ((m < 0) | (m > 1)).any():
        raise ValueError("mask must be finite and in [0,1]")
    if m.ndim == 2:
        m = m[None, None, None]
    elif m.ndim == 4:
        m = m.unsqueeze(2)
    if m.ndim != 5 or m.shape[1] != 1 or m.shape[-2:] != x.shape[-2:]:
        raise ValueError("mask must be [H,W], [B,1,H,W], or [B,1,T,H,W]")
    if m.shape[0] not in (1, x.shape[0]) or m.shape[2] not in (1, x.shape[2]):
        raise ValueError("mask B/T axes must be singleton or match x")
    m = m.to(dtype=x.dtype)
    return torch.where(m == 1, x, x * m + value * (1 - m))
