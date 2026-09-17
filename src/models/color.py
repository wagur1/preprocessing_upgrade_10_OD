"""Differentiable YCbCr 4:2:0 colorspace emulation (upgrade-5.1, contribution C1).

The upgrade-2/3 virtual codec quantised **RGB** planes. The deployment codecs
(x264/x265, ``-pix_fmt yuv420p``) instead:

  1. convert RGB -> YCbCr (BT.601 full-range, ffmpeg's default for yuv420p);
  2. **subsample Cb/Cr 2x2** -- chroma resolution halves at EVERY rate point,
     independent of QP;
  3. quantise chroma with a coarser effective step (the H.26x chroma QP offset
     grows with QP, reaching ~+6..9 QP / ~2x step at QP~50);
  4. upsample chroma and convert back to RGB on the decode side, after the
     damage is baked in.

An edit trained against an RGB proxy never sees this geometry: it happily
spends budget on high-frequency **colour** detail that the real codec destroys
for free at any QP. That is a proxy->real mismatch STE calibration cannot
repair (STE fixes rate/quant geometry, not colourspace geometry), and is the
leading remaining suspect for upgrade-3's small transfer gains (-1.6%/-0.3%
BD-Rate vs the paper's -15%).

This module provides the differentiable building blocks:

  * ``rgb_to_ycbcr`` / ``ycbcr_to_rgb``: exact BT.601 full-range matrix ops;
  * ``rgb_to_yuv420_planes`` / ``yuv420_planes_to_rgb``: the codec-geometry
    split (luma full-res + chroma 2x2-averaged half-res) and its inverse
    (bilinear chroma upsample) -- the pair the VirtualCodec codes on;
  * ``yuv420_roundtrip``: the full colourspace damage without quantisation,
    used by tests to prove the geometry is lossy.

The round trip is deliberately NOT the identity: passing a source through it
reproduces the chroma blur every yuv420p clip suffers. The preprocessor
trained through this proxy learns to keep machine-relevant detail in **luma**,
where the bits actually go.

The chroma quantiser coarsening lives in ``virtual_codec.VirtualCodec`` as
``chroma_step_scale`` on the quant step; this module is purely colourspace
geometry.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# BT.601 full-range (JPEG / ffmpeg full-range yuv420p) constants:
#   Y  = 0.299 R + 0.587 G + 0.114 B
#   Cb = -0.168736 R - 0.331264 G + 0.5 B + 0.5
#   Cr = 0.5 R - 0.418688 G - 0.081312 B + 0.5
_RGB2YCC = (
    (0.299, 0.587, 0.114, 0.0),
    (-0.168736, -0.331264, 0.5, 0.5),
    (0.5, -0.418688, -0.081312, 0.5),
)
_YCC2RGB = (
    (1.0, 0.0, 1.402, -0.701),
    (1.0, -0.344136, -0.714136, 0.529136),
    (1.0, 1.772, 0.0, -0.886),
)


def _color_transform(x: torch.Tensor, m) -> torch.Tensor:
    """Affine 3x3(+offset) colour transform on the channel dim of a 4D/5D
    tensor. 5D [B,3,T,H,W] inputs are folded time-major to [(B*T),3,H,W] first
    -- a plain ``reshape(-1, 3, H, W)`` would interleave channels of different
    frames (memory layout is channel-major)."""
    w = x.new_tensor([row[:3] for row in m])       # [3(out), 3(in)]
    b = x.new_tensor([row[3] for row in m])        # [3(out)]
    if x.ndim == 5:
        bs, c, t, h, wd = x.shape
        flat = x.permute(0, 2, 1, 3, 4).reshape(bs * t, c, h, wd)
        out = torch.einsum("oc,nchw->nohw", w, flat) + b.view(1, 3, 1, 1)
        return out.reshape(bs, t, c, h, wd).permute(0, 2, 1, 3, 4)
    if x.ndim == 4:
        out = torch.einsum("oc,nchw->nohw", w, x) + b.view(1, 3, 1, 1)
        return out
    raise ValueError(f"expected 4D or 5D tensor, got {tuple(x.shape)}")


def rgb_to_ycbcr(x: torch.Tensor) -> torch.Tensor:
    """RGB [0,1] -> YCbCr with all channels in [0,1] (Cb/Cr centred at 0.5).

    Differentiable; shape-preserving for 4D [N,3,H,W] and 5D [B,3,T,H,W]."""
    return _color_transform(x, _RGB2YCC).clamp(0.0, 1.0)


def ycbcr_to_rgb(x: torch.Tensor) -> torch.Tensor:
    """YCbCr [0,1] -> RGB [0,1]. Differentiable; shape-preserving."""
    return _color_transform(x, _YCC2RGB).clamp(0.0, 1.0)


def rgb_to_yuv420_planes(frame: torch.Tensor):
    """RGB frame [B,3,H,W] -> (luma [B,1,H,W], chroma [B,2,ceil(H/2),ceil(W/2)]).

    Chroma is 2x2 average-pooled -- the differentiable stand-in for the
    encoder-side 4:2:0 subsample. Odd H/W: the last chroma row/col is
    replicated to even before pooling (a codec pads to macroblocks anyway)."""
    if frame.ndim != 4 or frame.shape[1] != 3:
        raise ValueError(f"expected [B,3,H,W], got {tuple(frame.shape)}")
    ycc = rgb_to_ycbcr(frame)
    y = ycc[:, 0:1]
    c = ycc[:, 1:3]
    h, w = c.shape[-2:]
    if h % 2 or w % 2:
        c = F.pad(c, (0, w % 2, 0, h % 2), mode="replicate")
    c = F.avg_pool2d(c, 2)
    return y, c


def yuv420_planes_to_rgb(y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`rgb_to_yuv420_planes`: bilinear chroma upsample back to
    luma resolution, then YCbCr -> RGB. Mirrors the decoder-side resample that
    precedes the RGB conversion."""
    h, w = y.shape[-2:]
    c_up = F.interpolate(c, size=(h, w), mode="bilinear", align_corners=False)
    return ycbcr_to_rgb(torch.cat([y, c_up], dim=1))


def yuv420_roundtrip(x: torch.Tensor) -> torch.Tensor:
    """Full colourspace geometry of a yuv420p pipeline, no quantisation:

        RGB -> YCbCr -> 2x2 chroma down -> bilinear up -> YCbCr -> RGB.

    Shape-preserving for 4D/5D tensors; deliberately NOT the identity."""
    def _rt(frame: torch.Tensor) -> torch.Tensor:
        y, c = rgb_to_yuv420_planes(frame)
        return yuv420_planes_to_rgb(y, c)

    if x.ndim == 4:
        return _rt(x)
    if x.ndim == 5:  # fold time into the batch dim
        b, c3, t, h, w = x.shape
        flat = x.permute(0, 2, 1, 3, 4).reshape(b * t, c3, h, w)
        return _rt(flat).reshape(b, t, c3, h, w).permute(0, 2, 1, 3, 4)
    raise ValueError(f"expected 4D or 5D tensor, got {tuple(x.shape)}")


def _demo() -> None:
    torch.manual_seed(0)
    # matrix round trip is near-identity in float32
    x = torch.rand(2, 3, 4, 32, 32)
    err = (ycbcr_to_rgb(rgb_to_ycbcr(x)) - x).abs().max().item()
    assert err < 1e-5, f"colour matrix round trip error {err}"
    # 4:2:0 round trip is NOT identity: chroma high-freq is destroyed ...
    flat = torch.rand(1, 3, 16, 16)
    cost = (yuv420_roundtrip(flat) - flat).abs().mean().item()
    assert cost > 0.0, "subsampling must cost something"
    # ... but grey (achromatic) content is preserved: identical noise on all
    # channels means chroma is exactly flat, only luma carries the perturbation
    g = 0.5 + 0.01 * torch.rand(1, 1, 16, 16)
    grey = g.expand(1, 3, 16, 16)
    gcost = (yuv420_roundtrip(grey) - grey).abs().mean().item()
    assert gcost < 1e-3, f"achromatic damage should be tiny, got {gcost}"
    # shapes preserved for odd sizes
    odd = torch.rand(1, 3, 2, 31, 23)
    out = yuv420_roundtrip(odd)
    assert out.shape == odd.shape, (out.shape, odd.shape)
    # gradients flow through the whole path
    xg = torch.rand(1, 3, 2, 16, 16, requires_grad=True)
    yuv420_roundtrip(xg).mean().backward()
    assert xg.grad is not None and torch.isfinite(xg.grad).all()
    # luma is untouched by the subsample round trip -- exact where the RGB
    # clamp never bites, so probe on low-saturation content (small chroma
    # keeps the reconstruction inside [0,1]; real decoders clamp too)
    low_sat = 0.5 + 0.1 * (torch.rand(1, 3, 2, 16, 16) - 0.5)
    ycc = rgb_to_ycbcr(low_sat)
    rt_ycc = rgb_to_ycbcr(yuv420_roundtrip(low_sat))
    assert torch.allclose(ycc[:, 0:1], rt_ycc[:, 0:1], atol=1e-6), "luma must pass through"
    # plane split/join sizes
    y, c = rgb_to_yuv420_planes(torch.rand(2, 3, 112, 128))
    assert y.shape == (2, 1, 112, 128) and c.shape == (2, 2, 56, 64), (y.shape, c.shape)
    print(f"color self-check passed (matrix rt {err:.2e}, chroma cost {cost:.4f}, "
          f"achromatic cost {gcost:.2e})")


if __name__ == "__main__":
    _demo()
