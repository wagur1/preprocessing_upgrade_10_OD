"""CompressAI codec wrapper -- the replacement for the paper's virtual codec.

The paper builds a hand-crafted *differentiable virtual codec* (intra/inter
prediction -> transform -> quantise -> inverse, with a factorized-prior rate
estimate a la Balle et al.) purely to supply a differentiable rate + distortion
signal during training.

We replace that entire block with a pretrained CompressAI model
(``bmshj2018-factorized``).  Its factorized-prior entropy bottleneck *is* the
Balle et al. model the paper cites, so it provides exactly the same
rate/distortion supervision -- but as a learned, well-tested codec:

    * training  : ``forward`` gives a differentiable reconstruction ``x_hat``
                  and an estimated ``bpp`` (from the entropy model likelihoods).
                  Additive-noise quantisation keeps everything differentiable so
                  gradients flow back to the preprocessor.
    * evaluation: ``compress``/``decompress`` run the *real* range coder and
                  return the *actual* bitrate, so reported bpp is honest.

Video is handled frame-wise: a clip ``[B, C, T, H, W]`` is folded to
``[B*T, C, H, W]`` and each frame is coded as an image (the paper's virtual
codec is likewise applied per frame with the previous frame as reference; here
temporal modelling lives entirely in the preprocessor, and the codec is a pure
image proxy, which is the standard CompressAI setup).
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

def _zoo():
    """Import CompressAI lazily so the block-DCT (virtual) codec path never needs
    compressai installed -- only constructing a CompressAICodec triggers it."""
    from compressai.zoo import bmshj2018_factorized, bmshj2018_hyperprior
    return {
        "bmshj2018-factorized": bmshj2018_factorized,
        "bmshj2018-hyperprior": bmshj2018_hyperprior,
    }

# CompressAI analysis/synthesis transforms downsample by 2^4 (factorized) or
# 2^6 (hyperprior). Pad to 64 so any supported model gets valid dimensions.
_STRIDE = 64


def _bpp_from_likelihoods(likelihoods: dict, num_pixels: int) -> torch.Tensor:
    total = 0.0
    for lk in likelihoods.values():
        total = total + torch.log(lk).sum() / (-math.log(2) * num_pixels)
    return total


class CompressAICodec(nn.Module):
    """Differentiable learned codec used as the training-time virtual codec.

    Args:
        model: CompressAI zoo model name.
        qualities: quality level(s) to instantiate. During training one is
            sampled per batch (mimicking the paper's random quantisation factor
            f_q in [30, 50]); at eval you sweep them to trace a rate curve.
        pretrained: load CompressAI's pretrained weights.
        trainable: if False (default) the codec is frozen -- it is a *proxy*,
            only the preprocessor learns. Gradients still propagate through it.
    """

    def __init__(
        self,
        model: str = "bmshj2018-factorized",
        qualities: List[int] | int = (1, 2, 3, 4, 5, 6, 7, 8),
        pretrained: bool = True,
        trainable: bool = False,
    ):
        super().__init__()
        zoo = _zoo()
        if model not in zoo:
            raise ValueError(f"unknown codec '{model}', choose from {list(zoo)}")
        if isinstance(qualities, int):
            qualities = (qualities,)
        self.model_name = model
        self.qualities = list(qualities)
        # Hold one CompressAI network per quality level.
        self.nets = nn.ModuleDict(
            {str(q): zoo[model](quality=q, pretrained=pretrained) for q in qualities}
        )
        if not trainable:
            for p in self.parameters():
                p.requires_grad_(False)
        self.trainable = trainable
        self._updated = False

    # -- helpers -----------------------------------------------------------
    def _net(self, quality: int) -> nn.Module:
        return self.nets[str(quality)]

    @staticmethod
    def _fold(x: torch.Tensor):
        """[B, C, T, H, W] -> [B*T, C, H, W] (+ shape to unfold later)."""
        b, c, t, h, w = x.shape
        return x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w), (b, t)

    @staticmethod
    def _unfold(x: torch.Tensor, bt) -> torch.Tensor:
        b, t = bt
        n, c, h, w = x.shape
        return x.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)

    @staticmethod
    def _pad(x: torch.Tensor):
        h, w = x.shape[-2:]
        new_h = math.ceil(h / _STRIDE) * _STRIDE
        new_w = math.ceil(w / _STRIDE) * _STRIDE
        pad = (0, new_w - w, 0, new_h - h)
        return F.pad(x, pad, mode="replicate"), (h, w)

    @staticmethod
    def _crop(x: torch.Tensor, hw) -> torch.Tensor:
        h, w = hw
        return x[..., :h, :w]

    # -- training path -----------------------------------------------------
    def forward(self, x: torch.Tensor, quality: int):
        """Differentiable proxy pass.

        Args:
            x: video clip [B, C, T, H, W] in [0, 1].
            quality: which CompressAI quality net to use.
        Returns:
            x_hat: reconstruction [B, C, T, H, W] in [0, 1].
            bpp:   estimated bits-per-pixel (scalar tensor, differentiable).
        """
        frames, bt = self._fold(x)
        frames, hw = self._pad(frames)
        out = self._net(quality)(frames)
        n, _, ph, pw = frames.shape
        # Divide by *original* pixel count (standard convention).
        num_pixels = bt[0] * bt[1] * hw[0] * hw[1]
        bpp = _bpp_from_likelihoods(out["likelihoods"], num_pixels)
        x_hat = self._crop(out["x_hat"], hw).clamp(0.0, 1.0)
        return self._unfold(x_hat, bt), bpp

    # -- evaluation path (real entropy coder) ------------------------------
    @torch.no_grad()
    def compress_decompress(self, x: torch.Tensor, quality: int):
        """Run the *real* range coder. Returns (x_hat, real_bpp).

        Reports the actual coded bitrate, not the entropy estimate.
        """
        if not self._updated:
            for net in self.nets.values():
                net.update(force=True)
            self._updated = True
        net = self._net(quality)
        frames, bt = self._fold(x)
        frames, hw = self._pad(frames)
        enc = net.compress(frames)
        dec = net.decompress(enc["strings"], enc["shape"])
        # total bits across every latent string, for every frame in the clip
        total_bits = 0
        for str_list in enc["strings"]:
            for s in str_list:
                total_bits += len(s) * 8
        num_pixels = bt[0] * bt[1] * hw[0] * hw[1]
        real_bpp = total_bits / num_pixels
        x_hat = self._crop(dec["x_hat"], hw).clamp(0.0, 1.0)
        return self._unfold(x_hat, bt), float(real_bpp)
