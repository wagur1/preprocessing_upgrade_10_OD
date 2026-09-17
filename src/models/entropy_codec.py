"""D8: trained factorized-prior rate model + learnable strength — the entropy-model
component that the 5 learning-failure proofs identified as the missing piece.

Design (informed by the whole u6 campaign):
- The preprocessor stays the D3 smooth parameterisation (convex blend toward
  blur) with the D2 hard saliency gate — structure is proven; only the RATE
  MODEL and the STRENGTH head learn.
- Rate model: per-frequency Laplacian factorized prior (Ballé-style, small:
  one (K,) logits table per frequency position + soft-to-hard quant) trained by
  MLE on its own symbols through the beta*bpp term of the objective — i.e. the
  entropy model learns the DCT symbol distribution it actually sees, not a
  Gaussian-power surrogate. It is NOT regressed against x264/x265 bit counts.
- Strength head: the U-Net's only job is now the per-pixel strength s in [0,1]
  (sigmoid), trained with CE + distill + the LEARNED rate loss.

This module plugs into the existing harness: codec.kind="entropy" in config.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .virtual_codec import VirtualCodec


class LearnedRateCodec(VirtualCodec):
    """VirtualCodec with a trained Laplacian factorized-prior rate model.

    The rate head is a tiny table: for each of the block*K*K DCT positions,
    K Laplacian components (mu, b) parameterise the symbol distribution;
    expected bits = -log2 p(round(y)). Trained online by its own Adam (see
    engine._fit, lr 1e-3) by MINIMISING its own expected bits through the
    objective's beta*bpp term — standard Ballé factorized-prior MLE, which
    converges to the entropy of the symbols the quantiser produces. Cheap
    (table only) and stays "frozen codec" — the prior models the CODEC, not
    the content. There is deliberately NO regression against real x264/x265
    bit counts: the real codec is never in the training forward pass (see
    memory: three separate STE/real-forward attempts failed).
    """

    def __init__(self, *a, n_components: int = 3, **kw):
        super().__init__(*a, **kw)
        bs = self.block
        n_pos = bs * bs
        # Laplacian mixture per DCT position: weights + locs + scales
        self.n_comp = n_components
        self.rate_params = nn.ParameterDict({
            "w": nn.Parameter(torch.zeros(n_pos, n_components)),
            "mu": nn.Parameter(torch.zeros(n_pos, n_components)),
            "b": nn.Parameter(torch.ones(n_pos, n_components) * 0.5),
        })
        self.rate_opt = None

    def _expected_bits(self, y: torch.Tensor) -> torch.Tensor:
        """y: [..., n_pos] quantised indices -> expected bits per coeff."""
        n_pos, K = self.rate_params["w"].shape
        # y: [..., B*B] -> [..., n_pos, 1]
        yq = y.unsqueeze(-1)
        w = F.softmax(self.rate_params["w"], dim=-1)          # [n_pos, K]
        mu = self.rate_params["mu"]                            # [n_pos, K]
        b = F.softplus(self.rate_params["b"]) + 1e-6           # [n_pos, K]
        # Laplacian pmf at integer symbol k: exp(-|k-mu|/b) / (2b)
        pmf = torch.exp(-(yq - mu).abs() / b) / (2.0 * b)      # [..., n_pos, K]
        p = (pmf * w).sum(-1).clamp_min(1e-9)
        return -torch.log2(p)

    def _quant_rate_learned(self, coeff: torch.Tensor, step: float, training: bool):
        """Drop-in replacement for _quant_rate with the learned prior.

        ``coeff`` arrives in the pipeline's block layout
        [N, C*bs*bs, H/bs, W/bs] (see VirtualCodec._dct). Split the channel
        axis into (C, bs*bs) and fold C into the batch: every channel uses
        the same per-position prior (the codec's symbol statistics depend on
        the DCT position, not the plane)."""
        y = coeff / step
        if training:
            y_hat = y + torch.empty_like(y).uniform_(-0.5, 0.5)
        else:
            y_hat = torch.round(y)
        N, Cpos, nH, nW = y_hat.shape
        bs = self.block
        n_pos = bs * bs
        C = Cpos // n_pos
        yv = y_hat.reshape(N * C, n_pos, nH, nW).movedim(1, -1)  # [N*C, nH, nW, n_pos]
        bits = self._expected_bits(yv).sum()
        return y_hat, bits

    # Route every inherited code path through the learned prior: the parent's
    # _code_yuv420/_code_rgb call self._quant_rate; shadow it with the
    # learned version so no call site needs to change.
    def _quant_rate(self, coeff: torch.Tensor, step: float, training: bool):
        return self._quant_rate_learned(coeff, step, training)
