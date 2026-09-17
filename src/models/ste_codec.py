"""Real-codec-in-the-loop straight-through wrapper (upgrade3, contribution A3).

upgrade2 trained *entirely* through a differentiable proxy (CompressAI or the
block-DCT virtual codec) and hoped the edit transferred to x264/x265. Its own
report shows the transfer stalling near break-even -- the proxy's rate/quant
geometry simply is not the deployment codec's.

The most reliable fix in the literature is **run the real codec in the forward
pass and borrow the proxy's gradient in the backward pass** (straight-through):
Lu et al. (arXiv:2206.05650) measured forward-real-codec at BD-Rate -20.3% vs
-14.6% for a proxy used in *both* directions. This wrapper implements exactly
that so the loss the optimiser sees is computed on the TRUE x264/x265
reconstruction and TRUE coded bpp, while gradients still flow to the
preprocessor through the differentiable proxy:

    x_hat = x_proxy + (x_real  - x_proxy).detach()      # value=real, grad=proxy
    bpp   = bpp_proxy + (bpp_real - bpp_proxy).detach()

Because ffmpeg runs per step this is ~10-50x slower than proxy-only training and
is meant as a **short calibration fine-tune** on top of a proxy-pretrained
checkpoint (see configs/universal_action_recognition.yaml -> stage 2), not for
training from scratch. ``quality -> qp`` inversion lets it slot into the engine's
existing ``codec(x, quality)`` call unchanged.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from ..codecs import StandardCodec, ffmpeg_available


class STECodec(nn.Module):
    """Straight-through: real x264/x265 value, differentiable proxy gradient.

    Args:
        proxy: a differentiable codec (``VirtualCodec`` / ``CompressAICodec``)
            exposing ``forward(x, quality) -> (x_hat, bpp)``.
        codec: ``"h264"`` or ``"h265"`` -- the real deployment codec.
        quality_to_qp: maps each proxy quality id to the real-codec QP to encode
            at (the inverse of the config's ``train.qp_to_quality``).
        preset: x264/x265 speed preset.
    """

    def __init__(
        self,
        proxy: nn.Module,
        codec: str = "h265",
        quality_to_qp: Dict[int, int] | None = None,
        preset: str = "medium",
    ):
        super().__init__()
        if not ffmpeg_available():
            raise RuntimeError("STECodec needs ffmpeg (with libx264/libx265) on PATH")
        self.proxy = proxy
        self.codec = codec
        self.preset = preset
        self.quality_to_qp = {int(k): int(v) for k, v in (quality_to_qp or {}).items()}
        self.qualities: List[int] = list(getattr(proxy, "qualities", []))

    def _qp(self, quality: int) -> int:
        if quality in self.quality_to_qp:
            return self.quality_to_qp[quality]
        # linear fallback: higher quality id -> lower QP, clamped to the x26x range
        return int(max(18, min(51, 51 - 3 * int(quality))))

    def forward(self, x: torch.Tensor, quality: int):
        x_p, bpp_p = self.proxy(x, quality)                       # differentiable
        sc = StandardCodec(codec=self.codec, qp=self._qp(quality), preset=self.preset)
        with torch.no_grad():
            x_r, bpp_r = sc.compress_decompress(x.detach())       # real ffmpeg
            x_r = x_r.to(x_p.device, x_p.dtype)
        # straight-through: forward value = real codec, backward grad = proxy
        x_hat = x_p + (x_r - x_p).detach()
        bpp = bpp_p + (x_p.new_tensor(float(bpp_r)) - bpp_p).detach()
        return x_hat, bpp

    @torch.no_grad()
    def compress_decompress(self, x: torch.Tensor, quality: int):
        """Eval path -> honest real-codec reconstruction + bpp (no proxy)."""
        sc = StandardCodec(codec=self.codec, qp=self._qp(quality), preset=self.preset)
        x_r, bpp_r = sc.compress_decompress(x)
        return x_r.to(x.device), float(bpp_r)


def _demo() -> None:
    from .virtual_codec import VirtualCodec

    if not ffmpeg_available():
        print("ste_codec self-check SKIPPED (ffmpeg not found)")
        return
    torch.manual_seed(0)
    proxy = VirtualCodec(qualities=(1, 3, 5), block=8)
    ste = STECodec(proxy, codec="h264", quality_to_qp={1: 42, 3: 32, 5: 24})
    x = torch.rand(1, 3, 4, 64, 64, requires_grad=True)
    x_hat, bpp = ste(x, 3)
    # forward VALUE must equal the real-codec reconstruction (STE identity)
    sc_val, _ = ste.compress_decompress(x.detach(), 3)
    assert torch.allclose(x_hat.detach(), sc_val, atol=1e-4)
    # but gradient must exist (borrowed from the proxy)
    (x_hat.mean() + bpp).backward()
    assert x.grad is not None and torch.isfinite(bpp)
    print(f"ste_codec self-check passed (real-value/proxy-grad, bpp={float(bpp.detach()):.3f})")


if __name__ == "__main__":
    _demo()
