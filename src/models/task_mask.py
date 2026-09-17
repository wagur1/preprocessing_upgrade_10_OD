"""Task-importance spatial mask (upgrade3, contribution A2).

upgrade2 could only trade in-domain gain for real-codec transfer with a GLOBAL
knob (``gamma`` on total variation): every pixel was pushed to be cheap-to-encode
equally, so cutting background bits also blurred the object the analyzer needs.
A2 makes that trade-off **spatial**: spend the edit/rate budget where the frozen
task network actually looks, and aggressively smooth everywhere else.

The importance map is a **gradient saliency** of the task loss w.r.t. the input
pixels -- exactly "which pixels move the machine's decision":

    m(x) = normalise( | d L_task / d x | ),   detached.

It is computed with a single extra backward through the *frozen* analyzer and
then **detached**, so it only *reweights* the preprocessor's penalties -- no
second-order gradient flows into the preprocessor, keeping training stable and
cheap. This is the pixel-domain analogue of task-driven / wrapper-aware bit
allocation (Reinforced Bit Allocation, arXiv:1910.07392; feature-preserving RDO,
arXiv:2504.02216) and ROI retargeting for machines (EURASIP JIVP 2025), but as a
differentiable loss weight rather than an encoder-side QP map.

Contract: input video ``[B,C,T,H,W]`` in [0,1] -> mask ``[B,1,T,H,W]`` in [0,1],
1 = task-critical (protect), 0 = ignorable (smooth hard).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.enable_grad()
def task_saliency(analyzer, x: torch.Tensor, target, blur: int = 5,
                  eps: float = 1e-6) -> torch.Tensor:
    """Per-clip-normalised gradient-saliency importance map, detached.

    Args:
        analyzer: frozen :class:`TaskAnalyzer` (its ``accuracy_loss`` is used).
        x:        source clip ``[B,C,T,H,W]`` in [0,1].
        target:   whatever ``analyzer.accuracy_loss`` expects.
        blur:     odd box-blur kernel to spread the sparse pixel gradient into
                  coherent regions (0/1 disables).
    Returns:
        ``[B,1,T,H,W]`` mask in [0,1], detached (no grad to the preprocessor).
    """
    xin = x.detach().clone().requires_grad_(True)
    loss, _ = analyzer.accuracy_loss(xin, target)
    grad = torch.autograd.grad(loss, xin, retain_graph=False, create_graph=False)[0]
    sal = grad.abs().mean(dim=1, keepdim=True)  # [B,1,T,H,W], reduce colour
    if blur and blur > 1:
        b, _, t, h, w = sal.shape
        k = blur | 1  # force odd
        flat = sal.reshape(b * t, 1, h, w)
        flat = F.avg_pool2d(flat, k, stride=1, padding=k // 2)
        sal = flat.reshape(b, 1, t, h, w)
    # normalise each clip to [0,1] (robust to the raw gradient scale)
    lo = sal.amin(dim=(1, 2, 3, 4), keepdim=True)
    hi = sal.amax(dim=(1, 2, 3, 4), keepdim=True)
    mask = (sal - lo) / (hi - lo + eps)
    return mask.detach()


def masked_tv(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Total variation of ``x`` weighted by a spatial ``weight`` (same H,W).

    High weight -> that region's high-frequency energy is penalised more. Feed
    ``weight = 1 - mask`` to smooth the *background* while sparing the object."""
    if weight.shape[-2:] != x.shape[-2:]:
        weight = F.interpolate(
            weight.reshape(-1, 1, *weight.shape[-2:]), size=x.shape[-2:],
            mode="bilinear", align_corners=False,
        ).reshape(*weight.shape[:-2], *x.shape[-2:])
    dh = (x[..., 1:, :] - x[..., :-1, :]).abs()
    dw = (x[..., :, 1:] - x[..., :, :-1]).abs()
    wh = weight[..., 1:, :]
    ww = weight[..., :, 1:]
    out = (dh * wh).mean() + (dw * ww).mean()
    if x.ndim == 5 and weight.ndim == 5 and weight.shape[2] == x.shape[2]:
        # Temporal difference, same reason as total_variation(): the inter-frame
        # residual is where a video codec spends most of its bits. Each pair takes
        # the SMALLER of its two endpoint weights so detail bordering a protected
        # region stays protected (proxy_v3's rule, train.py:163-177).
        dt = (x[:, :, 1:] - x[:, :, :-1]).abs()
        wt = torch.minimum(weight[:, :, 1:], weight[:, :, :-1])
        out = out + (dt * wt).mean()
    return out


def _demo() -> None:
    import torch.nn as nn

    from ..tasks.base import TaskAnalyzer  # noqa: E402

    class _Toy(TaskAnalyzer):
        def __init__(self):
            super().__init__()
            self.task_name = "toy"
            self.c = nn.Conv3d(3, 2, 3, padding=1)

        def accuracy_loss(self, x_hat, target):
            # loss concentrates gradient on a bright central patch
            return self.c(x_hat).pow(2).mean(), {}

        @torch.no_grad()
        def predict(self, x_hat):
            return self.c(x_hat).mean()

    torch.manual_seed(0)
    an = _Toy().freeze()
    x = torch.rand(2, 3, 4, 32, 32)
    m = task_saliency(an, x, None, blur=5)
    assert m.shape == (2, 1, 4, 32, 32)
    assert not m.requires_grad and float(m.min()) >= 0.0 and float(m.max()) <= 1.0 + 1e-5
    # masked_tv is differentiable w.r.t. its input and finite
    xp = torch.rand(2, 3, 4, 32, 32, requires_grad=True)
    tv = masked_tv(xp, 1.0 - m)
    tv.backward()
    assert xp.grad is not None and torch.isfinite(tv)
    print("task_mask self-check passed (saliency in [0,1], masked_tv differentiable)")


if __name__ == "__main__":
    _demo()
