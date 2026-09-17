"""Training objective for the machine-vision video preprocessor (upgrade2).

    L = lam_task * L_task + omega * L_distill + beta * L_rate + tau * L_temp
        (+ delta * L_delta) (+ gamma * L_tv) (+ gamma_res * L_tv_res)
        (+ kappa * L_dct) (+ kappa_t * L_dct3d) (+ mu * L_D)

By default there is **no MSE-to-source distortion term**. That term (the
baseline's ``L_D``) pins the reconstruction to the original pixels; against a
codec-*mismatched* proxy (our CompressAI wavelet transform) it fought
compression, which is why that setup never reached negative BD-Rate. Following
Yang et al. (TCSVT 2024) the default objective replaces pixel fidelity with a
*task-aligned* feature distillation term and lets a real rate weight bite.

Zhao et al. (arXiv:2512.15331), however, KEEP ``L_D`` heavily weighted (alpha=10)
with a light rate weight and still win on real block-DCT codecs. So ``L_D`` is
available as an optional ``mu`` term -- enable it together with the
block-transform virtual codec (``codec.kind: virtual``), where pinning pixels is
in the *same* domain as x264/x265 rather than fighting a wavelet proxy.

  * ``L_task``   : accuracy loss from the frozen analyzer on the reconstruction
                   (cross-entropy for recognition; SiamFC logistic for tracking).
  * ``L_distill``: MSE between the frozen analyzer's intermediate features on the
                   *source* and on the *reconstruction* -- keeps semantics the
                   codec would otherwise destroy (helps most at low bitrate).
  * ``L_rate``   : estimated bits-per-pixel from the codec entropy model.
  * ``L_temp``   : temporal consistency -- match the *inter-frame change* of the
                   reconstruction to that of the source. Preserves motion / kills
                   flicker without pinning absolute pixels (the video novelty).
  * ``L_delta``  : (optional, off by default) L1 magnitude of the preprocessor's
                   pixel edit ``|x_pre - x|``. A direct edit-sparsity lever when
                   the rate term alone won't stop the edit adding bits. NOT
                   MSE-to-source: it constrains the *input* edit, not x_hat.
  * ``L_tv``     : (optional, off by default) total variation of the preprocessor
                   output ``x_pre``. A *codec-agnostic* bit-cost proxy: every real
                   codec (DCT/wavelet/block) spends bits on spatial high-frequency
                   energy, so penalising TV pushes cheap-to-encode-on-ANY-codec
                   frames -- the lever for transfer to x264/x265 (unlike beta*bpp,
                   which only reduces the *proxy* codec's bits). L_task keeps the
                   edges the machine actually needs.
  * ``L_D``      : (optional, off by default) MSE between reconstruction and
                   source ``||x_hat - x||^2``. Zhao et al.'s distortion term; use
                   it with the block-transform virtual codec (mu ~ 10, light beta).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import math

import torch
import torch.nn.functional as F


@dataclass
class LossWeights:
    lam_task: float = 1.0
    omega: float = 0.5   # feature distillation (Yang et al. use 0.5)
    beta: float = 0.1    # rate; raise until bpp actually bites
    tau: float = 0.1     # temporal consistency
    delta: float = 0.0   # L1 edit-magnitude |x_pre-x| (edit sparsity; 0 = off)
    gamma: float = 0.0   # total-variation of x_pre (codec-agnostic bit cost; 0 = off)
    gamma_res: float = 0.0  # total-variation of the RESIDUAL x_pre-x (0 = off)
    kappa: float = 0.0      # RPP adaptive-DCT rate proxy on x_pre, PER-FRAME (0 = off)
    kappa_t: float = 0.0    # same, on SPATIO-TEMPORAL blocks (0 = off); additive to kappa
    mu: float = 0.0      # MSE-to-source L_D (paper's distortion term; 0 = off)
    rho: float = 0.0     # UP-VCM S-head distillation MSE(S(x), W_target); 0 = off
    use_task_mask: bool = False  # A2: weight gamma/delta by task saliency (spatial)


def total_variation(x: torch.Tensor) -> torch.Tensor:
    """Mean spatio-TEMPORAL total variation of x. 4D (B,C,H,W) or 5D (B,C,T,H,W).

    Codec-agnostic proxy for encode cost: sum of |neighbour differences| over
    height, width AND time. The temporal term is not decoration -- a video codec
    spends most of its bits on the INTER-FRAME residual, so a TV that stops at
    (h, w) is blind to the dominant cost. A residual that flickers frame to frame
    is expensive on x264/x265 and a spatial-only penalty cannot see it at all.

    Adopted from munnn01/proxy_v3 (`masked_total_variation`, train.py:163-177,
    with permission), which carries all three differences; this function had
    only two until 2026-08-31. Historical `gamma` runs (robust_transfer.yaml
    0.03, universal 0.01) were measured against the spatial-only version."""
    dh = (x[..., 1:, :] - x[..., :-1, :]).abs().mean()
    dw = (x[..., :, 1:] - x[..., :, :-1]).abs().mean()
    tv = dh + dw
    if x.ndim == 5:                      # (B,C,T,H,W): time is dim 2
        tv = tv + (x[:, :, 1:] - x[:, :, :-1]).abs().mean()
    return tv



def _dct_basis(n: int, device, dtype) -> torch.Tensor:
    """Orthogonal-ish 2D-DCT-II basis row matrix ``[n, n]`` (RPP's Eq. 1 form)."""
    k = torch.arange(n, device=device, dtype=dtype)
    return torch.cos(k[:, None] * math.pi / n * (k[None, :] + 0.5))


def adaptive_dct_loss(x: torch.Tensor, block: int = 8, band: float = 0.5) -> torch.Tensor:
    """RPP-style adaptive DCT loss: zero the WEAK high-frequency coefficients only.

    Adapted from Rate-Perception Optimized Preprocessing (arXiv:2301.10455), which
    reports 16.27% mean bitrate saving on FROZEN x264/x265/x266 -- and, unusually,
    a LARGER saving on h265 than h264 (24.6% vs 18.2% VMAF), i.e. the opposite of
    the per-codec asymmetry every subtractive mechanism in this repo has hit.

    Why this is not `gamma`/`gamma_res` again. Total variation penalises ALL
    variation with equal weight, so the only move it leaves the model is to shrink
    the edit uniformly -- measured: a 10x sweep of `gamma_res` mapped monotonically
    onto amplitude (equivalent to strength 0.46 / 0.15 / 0.03) and improved the
    spectrum by a flat 3% at matched amplitude. This term is selective in TWO ways
    instead:
      * a ZigZag band mask protects LOW frequencies absolutely (weight 0), and
      * within the high band, an adaptive threshold T = mean|F| protects the STRONG
        coefficients (edges, contrast) and drives only the sub-mean ones to zero.
    So it removes the detail a codec pays for and a recogniser plausibly does not,
    rather than attenuating everything the model produces.

    ``x`` is the value whose bits you want to reduce -- pass ``x_pre`` (the output),
    NOT the residual: reducing rate BELOW the anchor requires removing source
    detail, and a residual-only penalty can never remove any.
    """
    if x.ndim == 5:                                    # [B,C,T,H,W] -> [B*T,C,H,W]
        x = x.permute(0, 2, 1, 3, 4).flatten(0, 1)
    h, w = x.shape[-2:]
    h, w = h - h % block, w - w % block
    if h < block or w < block:
        return x.new_zeros(())
    blk = (x[..., :h, :w]
           .unfold(-2, block, block).unfold(-2, block, block)  # [...,nH,nW,b,b]
           .reshape(-1, block, block))
    B = _dct_basis(block, x.device, x.dtype)
    freq = B @ blk @ B.transpose(0, 1)                  # 2D DCT per block
    idx = torch.arange(block, device=x.device)
    band_mask = (idx[:, None] + idx[None, :]) >= max(1, int(round(band * 2 * (block - 1))))
    sel = freq.abs() * band_mask                        # low frequencies -> exactly 0
    n_sel = band_mask.sum().clamp_min(1)
    thr = sel.sum(dim=(-2, -1), keepdim=True) / n_sel   # adaptive, per block
    weak = band_mask & (sel < thr)                      # strong HF is PROTECTED
    return (sel * weak).sum() / (blk.shape[0] * n_sel)


def adaptive_dct3d_loss(x: torch.Tensor, tblock: int = 4, sblock: int = 8,
                        band: float = 0.5) -> torch.Tensor:
    """Adaptive DCT on SPATIO-TEMPORAL blocks: the axis ``adaptive_dct_loss`` cannot see.

    ``adaptive_dct_loss`` flattens time into the batch, so it is structurally blind to
    temporal frequency, and an inter-frame residual is where a video codec actually
    spends most of its bits. This term sees that axis.

    RETRACTION, 2026-09-03 -- read this before tuning ``kappa_t``. The term was built
    on "as the spatial penalty tightened the model moved cost onto the temporal axis
    (TVt/RMS 0.4931 -> 0.6964)". A Parseval-exact block-DCT energy audit
    (``u6_big4/spectrum3d.py``) does NOT support that: temporal-AC share of the
    residual FALLS with kappa (18.9% -> 15.1% at tblock=16) and absolute temporal-AC
    energy falls 66.3 -> 28.2 (x1e-5). TVt/RMS is an L1/L2 shape ratio, not an energy
    share -- its rise came from a shrinking denominator. What IS real: the model adds
    temporal-AC energy where this mask does not look. At kappa=10, added energy vs
    source is +127.6% in spatial-LF temporal-AC against +43.2% in the penalised
    spatial-HF cell, and only 20.6% of the added temporal-AC energy sits inside the
    ``band=0.5`` band (57.1% is at v+w 4-6). Widening to ``band=0.0`` raises the loss
    only +35%, because ``T = mean|F|`` protects above-mean coefficients and the escape
    cell's coefficients are above-mean. So v1 prices the wrong cell, and a v2 needs the
    band AND the threshold changed, not just a larger ``kappa_t``.

    Band mask (coefficient address ``(u, v, w)`` = temporal, spatial_h, spatial_w):

    * ``u == 0`` -- temporal DC, i.e. content that is static across the block. Weight
      ZERO: inter prediction codes it almost free, so penalising it would fight real
      content rather than waste.
    * ``u >= 1`` and ``v + w`` below the spatial band -- spatial-LF flicker. Weight
      ZERO, a deliberate scope hole in v1; measure that cell's energy before widening.
    * ``u >= 1`` and ``v + w`` in the spatial-HF band -- flickery texture. PENALISED.
    * inside the band, ``|F| >= T`` with per-block ``T = mean|F|`` -- protected, so
      motion-following edits survive. Same strong-coefficient rule as the 2-D term.

    The basis is the ORTHONORMAL one from ``virtual_codec`` (``D @ D.T == I``), not the
    unnormalised helper used by the 2-D term: this transform mixes a 4-long temporal
    axis with 8-long spatial axes, and unequal per-axis scaling would make ``|F|``
    incomparable across axes and skew the adaptive threshold.
    """
    from .models.virtual_codec import _dct_basis as _ortho_basis

    if x.ndim != 5:
        raise ValueError(f"adaptive_dct3d_loss expects [B,C,T,H,W], got {tuple(x.shape)}")
    t_, h, w = x.shape[2], x.shape[-2], x.shape[-1]
    t_, h, w = t_ - t_ % tblock, h - h % sblock, w - w % sblock
    if t_ < tblock or h < sblock or w < sblock:
        return x.new_zeros(())
    b, c = x.shape[0], x.shape[1]
    blk = (x[:, :, :t_, :h, :w]
           .reshape(b, c, t_ // tblock, tblock, h // sblock, sblock, w // sblock, sblock)
           .permute(0, 1, 2, 4, 6, 3, 5, 7)          # [B,C,nT,nH,nW, t,s,s]
           .reshape(-1, tblock, sblock, sblock))
    Dt = _ortho_basis(tblock).to(x)
    Ds = _ortho_basis(sblock).to(x)
    freq = torch.einsum("ntij,ut,vi,wj->nuvw", blk, Dt, Ds, Ds)

    u = torch.arange(tblock, device=x.device)
    s = torch.arange(sblock, device=x.device)
    spatial_hf = (s[:, None] + s[None, :]) >= max(1, int(round(band * 2 * (sblock - 1))))
    mask = (u[:, None, None] >= 1) & spatial_hf[None, :, :]
    sel = freq.abs() * mask
    n_sel = mask.sum().clamp_min(1)
    thr = sel.sum(dim=(-3, -2, -1), keepdim=True) / n_sel      # adaptive, per block
    weak = mask & (sel < thr)
    return (sel * weak).sum() / (blk.shape[0] * n_sel)


def feature_distillation(analyzer, x_source: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """Scale-normalised MSE between frozen-analyzer features of source vs recon.

    Source features are the (detached) target; gradients flow only through the
    reconstruction path. Returns 0 if the analyzer exposes no features."""
    feats_src = analyzer.features(x_source)
    feats_hat = analyzer.features(x_hat)
    if not feats_src:
        return x_hat.new_zeros(())
    loss = x_hat.new_zeros(())
    for fs, fh in zip(feats_src, feats_hat):
        fs = fs.detach()
        loss = loss + F.mse_loss(fh, fs) / (fs.pow(2).mean() + 1e-6)
    return loss / len(feats_src)


def temporal_consistency(x_source: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """Match reconstruction inter-frame deltas to source inter-frame deltas."""
    if x_source.shape[2] < 2:
        return x_hat.new_zeros(())
    ds = x_source[:, :, 1:] - x_source[:, :, :-1]
    dh = x_hat[:, :, 1:] - x_hat[:, :, :-1]
    return F.mse_loss(dh, ds)


def preprocessing_loss(
    analyzer,
    x_source: torch.Tensor,
    x_hat: torch.Tensor,
    bpp: torch.Tensor,
    target: Any,
    w: LossWeights,
    x_pre: torch.Tensor | None = None,
    task_mask: torch.Tensor | None = None,
    saliency_pred: torch.Tensor | None = None,
    saliency_target: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    """Composite preprocessor loss.

    ``task_mask`` (A2): optional detached ``[B,1,T,H,W]`` importance map in [0,1]
    (1 = task-critical). When given, the edit (``delta``) and TV (``gamma``)
    penalties are weighted by ``1-mask`` so the preprocessor smooths / stops
    spending bits on *background* while sparing the object the analyzer needs.
    Without a mask the penalties are spatially uniform (upgrade2 behaviour).

    ``saliency_pred`` / ``saliency_target`` (UP-VCM ``rho``): MSE between the
    model's self-sufficient importance prediction S(x) and its distillation
    target (teacher saliency, optionally DINOv2-blended — cached on the model
    as ``_last_w`` / ``_last_w_target``). Independent of the edit path: the
    mask is NEVER used to gate the UP-VCM edit, only to teach S.
    """
    from .models.task_mask import masked_tv

    l_task, _ = analyzer.accuracy_loss(x_hat, target)
    l_dist = feature_distillation(analyzer, x_source, x_hat)
    l_temp = temporal_consistency(x_source, x_hat)
    total = w.lam_task * l_task + w.omega * l_dist + w.beta * bpp + w.tau * l_temp
    bg = (1.0 - task_mask) if task_mask is not None else None  # penalise background
    # Edit-magnitude penalty on the *preprocessor output* (not the reconstruction):
    # pushes small/sparse pixel edits so the codec has less added detail to encode.
    # Distinct from MSE-to-source (which pins x_hat and fights compression).
    if w.delta and x_pre is not None:
        edit = (x_pre - x_source).abs()
        l_delta = (edit * bg).mean() if bg is not None else edit.mean()
        total = total + w.delta * l_delta
    else:
        l_delta = x_hat.new_zeros(())
    # Codec-agnostic bit-cost: penalise spatial high-frequency energy of the
    # output so edits are cheap on ANY codec (targets x264/x265 transfer, which
    # beta*bpp -- the proxy codec's own rate -- does not reach). With a task mask
    # the penalty is SPATIAL: smooth background hard, spare the object (A2).
    if w.gamma and x_pre is not None:
        l_tv = masked_tv(x_pre, bg) if bg is not None else total_variation(x_pre)
        total = total + w.gamma * l_tv
    else:
        l_tv = x_hat.new_zeros(())
    # Same idea as gamma, but on the RESIDUAL instead of the output. For an
    # ADDITIVE preprocessor (x_pre = x + s*r) gamma is the wrong target twice
    # over: TV(x_pre) is dominated by the SOURCE video's own texture, so most of
    # the penalty lands on content the preprocessor did not create, and driving
    # it down means blurring the source -- i.e. rebuilding the subtractive family
    # that is already measured R-D neutral. TV(r) penalises only the spectrum of
    # what the model ADDS, which is what makes an additive edit expensive on a
    # DCT codec, and leaves the source alone.
    #
    # SCALE, learned the hard way (gamma sweep 0.01/0.03/0.1, 2026-08-31, 7.5
    # GPU-hours for zero information): TV here is ~6e-3 against a task loss of
    # ~3.3, so a coefficient under ~1 contributes <0.2% of the objective and
    # changes nothing measurable -- three runs 10x apart in gamma produced
    # residuals within 0.7% RMS of each other AND of the gamma=0 baseline. Size
    # this against the task term (c*TV / L_task), never by copying a coefficient
    # from a config written for a different mechanism.
    if w.gamma_res and x_pre is not None:
        res = x_pre - x_source
        l_tv_res = masked_tv(res, bg) if bg is not None else total_variation(res)
        total = total + w.gamma_res * l_tv_res
    else:
        l_tv_res = x_hat.new_zeros(())
    # RPP-style adaptive DCT rate proxy on the OUTPUT. Unlike gamma/gamma_res this
    # can push rate BELOW the anchor, because it removes weak high-frequency detail
    # of the source rather than only shaping what the model adds. Blocks 8 and 16 --
    # the macroblock sizes a real codec transforms on (arXiv:2301.10455).
    if w.kappa and x_pre is not None:
        l_dct = 0.5 * (adaptive_dct_loss(x_pre, block=8)
                       + adaptive_dct_loss(x_pre, block=16))
        total = total + w.kappa * l_dct
    else:
        l_dct = x_hat.new_zeros(())
    # Spatio-temporal sibling of kappa. ADDITIVE, not a replacement: kappa stays at
    # its measured peak (10) so any delta is attributable to the temporal axis alone.
    if w.kappa_t and x_pre is not None:
        l_dct3d = adaptive_dct3d_loss(x_pre)
        total = total + w.kappa_t * l_dct3d
    else:
        l_dct3d = x_hat.new_zeros(())
    # MSE-to-source L_D (Zhao et al.): pins the reconstruction to the source.
    # We dropped it originally because it fought compression -- but that was
    # against a mismatched (wavelet) proxy. The paper KEEPS it heavy (alpha=10)
    # with a light rate weight and still wins on real block-DCT codecs, so it is
    # available here; enable it together with the block-transform virtual codec.
    if w.mu:
        l_d = F.mse_loss(x_hat, x_source)
        total = total + w.mu * l_d
    else:
        l_d = x_hat.new_zeros(())
    # UP-VCM S-head distillation: teach the self-sufficient importance head to
    # reproduce the (teacher+DINO) target. Gradients flow into S only; the
    # target is detached on the model side.
    if w.rho and saliency_pred is not None and saliency_target is not None:
        l_w = F.mse_loss(saliency_pred, saliency_target.detach())
        total = total + w.rho * l_w
    else:
        l_w = x_hat.new_zeros(())
    return {
        "loss": total,
        "loss_task": l_task.detach(),
        "loss_dist": l_dist.detach(),
        "loss_rate": (bpp.detach() if torch.is_tensor(bpp) else torch.as_tensor(bpp)),
        "loss_temp": l_temp.detach(),
        "loss_delta": l_delta.detach(),
        "loss_tv": l_tv.detach(),
        "loss_tv_res": l_tv_res.detach(),
        "loss_dct": l_dct.detach(),
        "loss_dct3d": l_dct3d.detach(),
        "loss_d": l_d.detach(),
        "loss_w_distill": l_w.detach(),
    }
