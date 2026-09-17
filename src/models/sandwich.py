"""Sandwich pre-processor for VCM (v8 model): PRE before the frozen codec,
POST restoration after decode — the strongest known pre/post design for
frozen-codec machine coding (cf. Sandwiched Compression, arXiv:2402.05887).

v7's UP-VCM is the PRE filter (unchanged, load-compatible). The new POST
filter is a small restoration UNet applied to the DECODED clip before the
analyzer:

    x ─► pre(x) ─► codec (frozen) ─► x̂ ─► post(x̂) ─► analyzer

The POST filter costs ZERO bits (it runs after decode). Its job is to restore
structure the codec destroyed — which lets PRE cut bits more aggressively
than a pre-only design can: accuracy the codec would lose can be bought back
on the decoder side. This is the mechanism pre-only models structurally lack
(the v6 falsification map: pre-only buys accuracy at exactly the anchor's
exchange rate).

Design:
  * PRE: full UPVCMPreprocessor (S + M1 + M2 + M3), zero-init identity gates.
  * POST: 3-level UNet (~96k params), FiLM(cond) on the bottleneck, zero-init
    output conv -> identity at init; a learned scalar gate ``post_strength``
    (zero-init) scales the residual. Input = decoded frames only (no W: at
    decode time S would see codec artifacts, and the restorer must fix ALL
    of the frame anyway).
  * ``forward`` = the PRE path only (engine's existing call unchanged);
    ``post_restore`` is the new half, called by the engine right after the
    codec. ``bypass_post`` lets eval disable POST for the decomposition arm
    (prep-only vs sandwich) without touching weights.

Joint training through the proxy codec: gradients reach PRE and POST in one
backward (loss on post(codec(pre(x)))). The STE stage closes the loop on the
REAL codec: POST then learns to invert real x264/x265 artifacts.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .upvcm import UPVCMPreprocessor, _FiLM


class _PostUNet(nn.Module):
    """3-level restoration UNet with FiLM(QP) on the bottleneck.

    Wider than the M2 editor (restoration is harder than structure shaping:
    it must undo quantisation damage at all rates): ~577k params at base=32
    — still two orders below the analyzers it serves.

    ``temporal=True`` adds a second, additive input branch over the two
    NEIGHBOURING decoded frames. Codec artifacts are strongly temporally
    correlated (I vs P frames, error propagation along the GOP, flicker) and
    the neighbours carry what a single frame cannot: the same content coded at
    a different point in the prediction chain. The branch is zero-initialised,
    so the module is EXACTLY the per-frame restorer at init — a warm start from
    the record checkpoint is behaviour-preserving and every gain is traceable
    to the temporal path. Unlike the gate x module products elsewhere in this
    repo this cannot dead-saddle: it adds into an already-live trunk, so its
    gradient is non-zero at step 0.
    """

    def __init__(self, base: int = 32, cond_dim: int = 1, temporal: bool = False):
        super().__init__()
        self.temporal = bool(temporal)
        self.in_conv = nn.Conv2d(3, base, 3, padding=1)
        if self.temporal:
            # 6 input channels = (prev, next) x RGB; zero-init so the module
            # starts as the per-frame restorer and the temporal path earns its
            # contribution from the loss rather than from a lucky init.
            self.in_conv_t = nn.Conv2d(6, base, 3, padding=1)
            nn.init.zeros_(self.in_conv_t.weight)
            nn.init.zeros_(self.in_conv_t.bias)
        self.enc1 = nn.Sequential(nn.ReLU(inplace=True),
                                  nn.Conv2d(base, base, 3, padding=1))
        self.down1 = nn.Conv2d(base, 2 * base, 3, stride=2, padding=1)
        self.enc2 = nn.Sequential(nn.ReLU(inplace=True),
                                  nn.Conv2d(2 * base, 2 * base, 3, padding=1))
        self.down2 = nn.Conv2d(2 * base, 4 * base, 3, stride=2, padding=1)
        self.mid = nn.Sequential(nn.ReLU(inplace=True),
                                 nn.Conv2d(4 * base, 4 * base, 3, padding=1),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(4 * base, 4 * base, 3, padding=1))
        self.film = _FiLM(cond_dim, 4 * base)
        self.up2 = nn.ConvTranspose2d(4 * base, 2 * base, 2, stride=2)
        self.dec2 = nn.Sequential(nn.ReLU(inplace=True),
                                  nn.Conv2d(4 * base, 2 * base, 3, padding=1))
        self.up1 = nn.ConvTranspose2d(2 * base, base, 2, stride=2)
        self.dec1 = nn.Sequential(nn.ReLU(inplace=True),
                                  nn.Conv2d(2 * base, base, 3, padding=1))
        self.out_conv = nn.Conv2d(base, 3, 3, padding=1)
        # NOT zero-init: zero output conv x zero post_strength gate is a dead
        # saddle (both gradients exactly 0 — the 16-epoch v8 run left POST
        # permanently closed because of this). Small noise + zero gate keeps
        # identity-at-init with alive gradients.
        nn.init.normal_(self.out_conv.weight, std=1e-3)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor,
                neighbours: torch.Tensor | None = None) -> torch.Tensor:
        # x [N,3,H,W], cond [N,cond_dim], neighbours [N,6,H,W] (temporal only)
        e0 = self.in_conv(x)
        if self.temporal:
            if neighbours is None:
                raise ValueError(
                    "temporal POST requires `neighbours` (prev/next frames); "
                    "call post_restore with a [B,3,T,H,W] clip")
            e0 = e0 + self.in_conv_t(neighbours)
        e1 = self.enc1(e0)
        e2 = self.enc2(self.down1(e1))
        b = self.mid(self.down2(e2))
        b = self.film(b, cond)
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)


class SandwichPreprocessor(nn.Module):
    """UP-VCM PRE + restoration POST around a frozen codec.

    Args mirror UPVCMPreprocessor for the PRE half (a UP-VCM checkpoint from
    v7 strict-loads into ``self.pre``); ``post_base`` sizes the POST UNet.

    ``forward`` returns x_pre (the PRE half — the engine's existing call).
    ``post_restore(x_hat, cond)`` is the POST half, called by the engine
    right after the codec. ``bypass_post`` (a plain attribute, not a buffer)
    lets evaluation run the prep-only decomposition arm.
    """

    def __init__(self, s_ch: int = 16, editor_ch: int = 24, cond_dim: int = 1,
                 dino_weight: float = 0.5, dino_name: str = "dinov2_vits14",
                 motion_tau: float = 0.1, post_base: int = 32,
                 w_budget: float = 0.0, post_temporal: bool = False):
        super().__init__()
        self.pre = UPVCMPreprocessor(
            s_ch=s_ch, editor_ch=editor_ch, cond_dim=cond_dim,
            dino_weight=dino_weight, dino_name=dino_name, motion_tau=motion_tau,
            w_budget=w_budget)
        self.post_net = _PostUNet(base=post_base, cond_dim=cond_dim,
                                  temporal=post_temporal)
        self.post_strength = nn.Parameter(torch.zeros(()))
        # eval-time decomposition switch (not part of the state_dict)
        self.bypass_post = False
        # expose the PRE caches for the loss (engine reads _last_w*)
        self._last_w: torch.Tensor | None = None
        self._last_w_target: torch.Tensor | None = None

    # -- PRE half (engine's existing preprocessor interface) ---------------
    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        x_pre = self.pre(x, cond, mask=mask)
        # forward the PRE caches so the engine's rho term keeps working
        self._last_w = self.pre._last_w
        self._last_w_target = self.pre._last_w_target
        return x_pre

    # -- POST half (called by the engine right after the codec) ------------
    def post_restore(self, x_hat: torch.Tensor, cond: torch.Tensor | None
                     ) -> torch.Tensor:
        """Restore the decoded clip. Identity when post_strength==0 (init) or
        ``bypass_post`` is set (the prep-only eval arm)."""
        if self.bypass_post:
            return x_hat
        b, c, t, h, w = x_hat.shape
        frames = x_hat.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        if cond is None:
            cond = frames.new_zeros(b, self.pre.cond_dim)
        cond_f = cond.repeat_interleave(t, dim=0).to(frames.dtype)
        neighbours = None
        if self.post_net.temporal:
            # causal-free ±1 window (edges replicate the boundary frame): the
            # restorer gets the same content coded elsewhere in the GOP, which
            # is exactly what a per-frame filter cannot see.
            prev = torch.cat([x_hat[:, :, :1], x_hat[:, :, :-1]], dim=2)
            nxt = torch.cat([x_hat[:, :, 1:], x_hat[:, :, -1:]], dim=2)
            nb = torch.cat([prev, nxt], dim=1)                 # [B,6,T,H,W]
            neighbours = nb.permute(0, 2, 1, 3, 4).reshape(b * t, 2 * c, h, w)
        delta = self.post_net(frames, cond_f, neighbours)
        out = frames + self.post_strength * delta
        return out.clamp(0.0, 1.0).reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)

    # -- convenience for loading a v7 UP-VCM checkpoint into the PRE half --
    def load_pre_state(self, upvcm_state: dict) -> None:
        self.pre.load_state_dict(upvcm_state, strict=True)
