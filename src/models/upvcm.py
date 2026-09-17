"""UP-VCM: universal pre-filter for Video Coding for Machines (v7 model).

A NEW architecture — deliberately NOT the Zhao two-branch additive editor the
v1-v6 lineage trained. The design bets on three mechanisms the lineage's own
falsification map identified as unmeasured or video-native:

  S  self-sufficient importance head (distilled; deploy needs NO analyzer)
  M1 saliency-gated background decimation  (blur + chroma coarsening)
  M2 W-gated ROI structure editor with FiLM(QP) conditioning
  M3 temporal background stabilisation (static background -> residual ~ 0)

``forward(x, cond, mask)`` keeps the engine's calling convention. ``mask``,
when given (training only), is NOT an edit gate — it is the DISTILL TARGET for
the S head (multi-teacher task saliency, optionally blended with DINOv2 patch
energy via ``dino_saliency``). The model always predicts its own
``W = S(x)``; at eval/deploy time no analyzer and no foundation model is
needed. The cached ``_last_w`` / ``_last_w_target`` feed the ``rho``-weighted
distillation term in ``losses.preprocessing_loss``.

All three modules are wrapped in zero-initialised strength gates, so the model
is an exact identity at initialisation and each mechanism switches on only if
the loss gradient says so (the identity-start discipline proven in v6).

Parameters: ~44k (s_ch=16, editor_ch=24) — still a small preprocessor, ~4.5x the
Zhao editor but three orders below the analyzers it serves.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .color import rgb_to_ycbcr, ycbcr_to_rgb
from .dino_saliency import dino_energy, get_dino


def _gauss_kernel(k: int, sigma: float, device, dtype) -> torch.Tensor:
    ax = torch.arange(k, device=device, dtype=dtype) - (k - 1) / 2
    g = torch.exp(-ax.pow(2) / (2 * sigma * sigma))
    g = g / g.sum()
    k2 = g[:, None] * g[None, :]                              # [k,k]
    return k2.view(1, 1, k, k)


class _FiLM(nn.Module):
    """Zero-init FiLM: gamma=beta=0 at start -> exact identity conditioning."""

    def __init__(self, cond_dim: int, ch: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden), nn.LeakyReLU(0.1), nn.Linear(hidden, 2 * ch))
        nn.init.zeros_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, feat: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # feat [N,C,H,W], cond [N,cond_dim] -> per-sample affine
        gb = self.net(cond)                                  # [N,2C]
        gamma, beta = gb.chunk(2, dim=1)
        return feat * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]


class _EditorBlock(nn.Module):
    """Small 2-level UNet editor with FiLM on the bottleneck (zero-init out)."""

    def __init__(self, ch: int = 24, cond_dim: int = 1):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(3, ch, 3, padding=1), nn.ReLU(inplace=True),
                                  nn.Conv2d(ch, ch, 3, padding=1))
        self.down = nn.Conv2d(ch, ch, 3, stride=2, padding=1)   # /2
        self.enc2 = nn.Sequential(nn.ReLU(inplace=True),
                                  nn.Conv2d(ch, 2 * ch, 3, padding=1), nn.ReLU(inplace=True))
        self.film = _FiLM(cond_dim, 2 * ch)
        self.up = nn.ConvTranspose2d(2 * ch, ch, 2, stride=2)
        self.dec = nn.Sequential(nn.Conv2d(2 * ch, ch, 3, padding=1), nn.ReLU(inplace=True))
        self.out = nn.Conv2d(ch, 3, 3, padding=1)
        # NOT zero-init: a zero output conv times the zero-initialised
        # ``edit_strength`` gate is a dead saddle — both factors sit at exactly
        # zero, so both gradients are exactly zero and the editor subnet never
        # leaves zero. Measured 2026-09-16 on every trained sandwich checkpoint:
        # edit_strength == +0.000000 and |W_out| == 0, i.e. M2 has never been
        # active in any recorded result. This is the same repair the POST half
        # needed in v8 (commit 3a948521e5db): small noise + zero gate keeps the
        # model an exact identity at init while leaving both gradients alive.
        nn.init.normal_(self.out.weight, std=1e-3)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x [N,3,H,W], cond [N,cond_dim]
        e1 = self.enc1(x)
        e2 = self.enc2(self.down(e1))
        e2 = self.film(e2, cond)
        d = self.dec(torch.cat([self.up(e2), e1], dim=1))
        return self.out(d)


class UPVCMPreprocessor(nn.Module):
    """UP-VCM conditional pre-filter (identity at init; analyzer-free at deploy).

    Args:
        s_ch: width of the importance head S.
        editor_ch: width of the ROI editor UNet (M2).
        cond_dim: FiLM condition width (normalised QP).
        dino_weight: blend weight of DINOv2 energy in the distill target
            (0 = teacher saliency only; the DINO load failing degrades to this).
        dino_name: torch.hub DINOv2 model name for the training-time anchor.
        motion_tau: motion-energy scale for M3's static gate.
    """

    def __init__(self, s_ch: int = 16, editor_ch: int = 24, cond_dim: int = 1,
                 dino_weight: float = 0.5, dino_name: str = "dinov2_vits14",
                 motion_tau: float = 0.1, w_budget: float = 0.0):
        super().__init__()
        self.cond_dim = int(cond_dim)
        self.dino_weight = float(dino_weight)
        self.dino_name = str(dino_name)
        self.motion_tau = float(motion_tau)
        self.w_budget = float(w_budget)

        # S: importance head — sees the frame AND its temporal difference, so
        # moving objects are salient even when their texture is flat.
        self.s_net = nn.Sequential(
            nn.Conv2d(6, s_ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(s_ch, s_ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(s_ch, 1, 3, padding=1))

        # M1: background decimation — separable gaussian, luma sigma small,
        # chroma sigma coarse (mimics yuv420 chroma subsampling damage).
        self.dec_strength = nn.Parameter(torch.zeros(()))

        # M2: ROI editor with FiLM(QP).
        self.editor = _EditorBlock(editor_ch, cond_dim)
        self.edit_strength = nn.Parameter(torch.zeros(()))

        # M3: temporal background stabilisation.
        self.stab_strength = nn.Parameter(torch.zeros(()))

        self._dino_failed = False
        # NOTE: the DINOv2 model is NEVER stored on `self` — assigning an
        # nn.Module attribute would register its 21M frozen weights in this
        # module's state_dict. get_dino() caches module-level instead.
        # caches for the loss (set every forward; detached targets)
        self._last_w: torch.Tensor | None = None
        self._last_w_target: torch.Tensor | None = None

        self.register_buffer("_blur_y", _gauss_kernel(5, 0.8, "cpu", torch.float32),
                             persistent=False)
        self.register_buffer("_blur_c", _gauss_kernel(5, 2.0, "cpu", torch.float32),
                             persistent=False)

    # -- helpers ---------------------------------------------------------
    def _frames(self, x: torch.Tensor) -> torch.Tensor:
        """[B,3,T,H,W] -> [B*T,3,H,W]."""
        b, c, t, h, w = x.shape
        return x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)

    def _decimate(self, frames: torch.Tensor) -> torch.Tensor:
        """Luma-preserving decimation: fine blur on Y, coarse blur on Cb/Cr."""
        ycc = rgb_to_ycbcr(frames)
        y, cb, cr = ycc[:, :1], ycc[:, 1:2], ycc[:, 2:3]
        ky = self._blur_y.to(dtype=frames.dtype)
        kc = self._blur_c.to(dtype=frames.dtype)
        y = F.conv2d(F.pad(y, (2, 2, 2, 2), mode="reflect"), ky)
        cb = F.conv2d(F.pad(cb, (2, 2, 2, 2), mode="reflect"), kc)
        cr = F.conv2d(F.pad(cr, (2, 2, 2, 2), mode="reflect"), kc)
        return ycbcr_to_rgb(torch.cat([y, cb, cr], dim=1))

    def _budget_normalise(self, w_map: torch.Tensor) -> torch.Tensor:
        """Rescale W to a fixed spatial mean (the ``w_budget``) over (T, H, W).

        Anti-collapse. With a free scale the head drives W to ~0 — measured on
        every trained checkpoint, logits ~ -60 — which silently turns M1 into a
        global edit, gates M2 off entirely (W * edit ~ 1e-12) and makes M3 a
        global freeze. Fixing the mean keeps the model's CHOICE of where the
        budget goes while removing the option of spending none.

        Dividing by the SUM (not by a floored mean) is what makes it robust: it
        is scale-free, so even a fully saturated sigmoid comes back as a uniform
        map at exactly ``w_budget`` instead of decaying toward zero. A map
        concentrated on very few pixels can drop below the budget after the
        ``clamp`` — that is a different, self-limiting failure (a spike makes
        M1's gate negative, which the loss punishes). ``w_budget = 0`` keeps the
        legacy free scale.
        """
        dims = (2, 3, 4)
        n = w_map.shape[2] * w_map.shape[3] * w_map.shape[4]
        total = w_map.sum(dim=dims, keepdim=True)
        out = w_map / total.clamp_min(torch.finfo(w_map.dtype).tiny) * (self.w_budget * n)
        return out.clamp(0.0, 1.0)

    def _importance(self, x: torch.Tensor) -> torch.Tensor:
        """W = S(x) in [0,1], [B,1,T,H,W]. Motion-aware, analyzer-free."""
        b, c, t, h, w = x.shape
        prev = torch.cat([x[:, :, :1], x[:, :, :-1]], dim=2)
        diff = (x - prev).abs()
        sin = torch.cat([x, diff], dim=1)                    # [B,6,T,H,W]
        sin = sin.permute(0, 2, 1, 3, 4).reshape(b * t, 6, h, w)
        w_map = torch.sigmoid(self.s_net(sin))               # [B*T,1,H,W]
        w_map = w_map.reshape(b, t, h, w).unsqueeze(1)       # [B,1,T,H,W]
        if self.w_budget > 0:
            w_map = self._budget_normalise(w_map)
        return w_map

    def _stabilise(self, x2: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """M3: copy static background from the previous OUTPUT frame.

        Where the frame barely moved and W is low, reuse the previous output
        pixel: the codec's inter prediction then codes ~zero residual there —
        the bit-saving mechanism. ``ref`` is detached (causal state, bounded
        memory); the first frame passes through unchanged.
        """
        b, c, t, h, wd = x2.shape
        outs = []
        ref = None
        for i in range(t):
            cur = x2[:, :, i]
            if ref is None:
                outs.append(cur)
                ref = cur.detach()
                continue
            mot = (cur - ref).abs().mean(dim=1, keepdim=True)        # [B,1,H,W]
            static = 1.0 - (mot / self.motion_tau).clamp(0.0, 1.0)
            gate = self.stab_strength * (1.0 - w[:, :, i]) * static  # zero-init
            out = gate * ref + (1.0 - gate) * cur
            outs.append(out)
            ref = out.detach()
        return torch.stack(outs, dim=2)

    # -- forward ----------------------------------------------------------
    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        """x [B,3,T,H,W] in [0,1] -> edited clip, same shape, in [0,1].

        ``cond``: [B,cond_dim] normalised QP (1 = heavy compression).
        ``mask``: TRAINING-ONLY distill target [B,1,T,H,W] (teacher saliency,
        optionally blended with DINOv2 energy). Never gates the edit.
        """
        if x.ndim != 5 or x.shape[1] != 3:
            raise ValueError(f"expected [B,3,T,H,W], got {tuple(x.shape)}")
        b, c, t, h, w = x.shape
        if cond is None:
            cond = x.new_zeros(b, self.cond_dim)

        w_pred = self._importance(x)                          # [B,1,T,H,W]
        self._last_w = w_pred
        self._last_w_target = None
        if mask is not None:
            tgt = mask
            if 0.0 < self.dino_weight < 1.0 and not self._dino_failed:
                dino = get_dino(self.dino_name, device=x.device)  # module-level cache
                if dino is None:
                    self._dino_failed = True
                else:
                    de = dino_energy(x, dino)
                    tgt = (1 - self.dino_weight) * mask + self.dino_weight * de
            if self.w_budget > 0:
                # match the prediction's normalisation so ``rho`` distils the
                # SHAPE of the saliency map (where to spend) rather than its
                # absolute scale (which the budget now owns)
                tgt = self._budget_normalise(tgt)
            self._last_w_target = tgt.detach()

        frames = self._frames(x)

        # M1: background decimation, gated by (1-W) and the zero-init strength.
        dec = self._decimate(frames)
        w_f = w_pred.permute(0, 2, 1, 3, 4).reshape(b * t, 1, h, w)
        frames1 = frames + self.dec_strength * (1.0 - w_f) * (dec - frames)

        # M2: ROI structure editor, W-gated, FiLM(QP)-conditioned.
        cond_f = cond.repeat_interleave(t, dim=0).to(frames.dtype)
        edit = self.editor(frames1, cond_f)
        frames2 = frames1 + self.edit_strength * w_f * edit

        # M3: temporal background stabilisation over the edited clip.
        x2 = frames2.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
        x3 = self._stabilise(x2, w_pred)

        return x3.clamp(0.0, 1.0)

    def extra_repr(self) -> str:
        return (f"s_ch, editor_ch from submodules; dino_weight={self.dino_weight}, "
                f"motion_tau={self.motion_tau}")
