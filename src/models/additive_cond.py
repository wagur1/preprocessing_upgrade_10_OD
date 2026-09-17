"""QP-conditioned additive preprocessor (round (b), docs/RUN_DESIGN_qpc.md).

The unconditioned :class:`AdditivePreprocessor` spends its edit amplitude
uniformly across the rate range: the kappa=10 lineage pays Δbpp +13-14% at QP30
for an accuracy gap the codec did not need help with, and earns its keep only
at QP45/50. Every amplitude lever tried on that model (omega, kappa_t, mu) died
the same death — bits rose faster than gap. This module keeps the exact
Zhao tree (9,795 base params, load-compatible) and adds ONE zero-init FiLM on
the shared 16-ch trunk so the edit can become conditional on the compression
operating point:

    cond = qp_norm(qp) ∈ [0,1]   (1 = QP51 = heavy compression)
    fused' = FiLM(fused, cond)   (γ=β=0 at init → conditioned == unconditioned)

The engine already builds and passes this cond for the U-Net path
(``_rate_cond``/``_qp_norm`` in src/engine.py); no engine change is needed —
this class simply stops ignoring it. ``mask`` stays ignored (spatial targeting
is round b2, one variable at a time).

Design notes:
  * one FiLM AFTER fusion, before to_rgb — the single representation both
    branches feed; per-branch FiLM doubles new params for a scalar condition;
  * zero-init output layer (identity at init) mirrors to_rgb's zero-init;
  * +576 params (10,371 total, +5.9%) — capacity was never the binding
    constraint (the 224-retrain gate-fail showed loss balance shifts, not
    capacity limits).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveCondPreprocessor(nn.Module):
    """Zhao two-branch additive editor + zero-init FiLM rate conditioning.

    10,371 parameters (9,795 Zhao tree + 576 FiLM). Fully convolutional and
    resolution-agnostic; trains at 128, evaluates at any square size.
    """

    def __init__(self, temporal_frames: int = 8, strength: float = 1.0,
                 cond_dim: int = 1):
        super().__init__()
        if temporal_frames < 1:
            raise ValueError(f"temporal_frames must be >= 1, got {temporal_frames}")
        self.tframes = int(temporal_frames)
        self.strength = float(strength)
        self.cond_dim = int(cond_dim)
        # Key names/shape layout must stay load-compatible with best.pt (the
        # additive run's checkpoint strict-loads into the base keys).
        self.spatial_stem = nn.Conv2d(3, 16, 3, padding=1)
        self.spatial_residual = nn.Module()
        self.spatial_residual.body = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 16, 3, padding=1))
        self.temporal_stem = nn.Conv2d(self.tframes * 3, 16, 3, padding=1)
        self.fusion = nn.Module()
        self.fusion.gate = nn.Sequential(
            nn.Conv2d(32, 16, 1), nn.ReLU(), nn.Conv2d(16, 16, 1))
        # Round (b): FiLM on the shared trunk. Linear(1->16) -> LeakyReLU ->
        # Linear(16->32) emitting (gamma, beta); the OUTPUT layer is zero-init
        # so gamma=beta=0 at init and the model is bit-identical to the
        # unconditioned one (exact identity with the zero-init to_rgb).
        self.film = nn.Sequential(
            nn.Linear(self.cond_dim, 16), nn.LeakyReLU(0.1), nn.Linear(16, 32))
        nn.init.zeros_(self.film[2].weight)
        nn.init.zeros_(self.film[2].bias)
        self.to_rgb = nn.Conv2d(16, 3, 3, padding=1)
        nn.init.zeros_(self.to_rgb.weight)
        nn.init.zeros_(self.to_rgb.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: [B,3,T,H,W] in [0,1] -> edited clip, same shape, in [0,1].

        ``cond``: [B, cond_dim] rate condition (normalised QP, 1 = heavy
        compression); ``None`` defaults to zeros = lightest compression.
        ``mask`` is accepted for loop compatibility and ignored (round b2)."""
        if x.ndim != 5 or x.shape[1] != 3:
            raise ValueError(f"expected [B,3,T,H,W], got {tuple(x.shape)}")
        b, c, t, h, w = x.shape
        frames = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)   # [B*T,3,H,W]

        sp = F.relu(self.spatial_stem(frames))
        body = self.spatial_residual.body
        sp = sp + body[2](F.relu(body[0](sp)))

        # Causal window per output frame (identical wiring to additive.py).
        pad = x[:, :, :1].expand(b, c, self.tframes - 1, h, w)
        padded = torch.cat([pad, x], dim=2)                      # [B,C,T+tf-1,H,W]
        win = padded.unfold(2, self.tframes, 1)                  # [B,C,T,H,W,tf]
        stack = win.permute(0, 2, 5, 1, 3, 4).reshape(b * t, self.tframes * c, h, w)
        tp = F.relu(self.temporal_stem(stack))   # per-frame, no broadcast

        gate = torch.sigmoid(
            self.fusion.gate[2](F.relu(self.fusion.gate[0](torch.cat([sp, tp], dim=1)))))
        fused = gate * sp + (1.0 - gate) * tp                    # [B*T,16,H,W]

        if cond is None:
            cond = frames.new_zeros(b, self.cond_dim)
        cond_f = cond.repeat_interleave(t, dim=0).to(fused.dtype)  # [B*T, cond_dim]
        gamma, beta = self.film(cond_f).chunk(2, dim=1)            # [B*T,16] each
        fused = fused * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]

        out = frames + self.strength * self.to_rgb(fused)
        return out.clamp(0.0, 1.0).reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
