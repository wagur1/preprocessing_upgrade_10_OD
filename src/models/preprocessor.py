"""Rate- and motion-conditioned U-Net video preprocessor (upgrade2).

This is a clean redesign that drops the two-branch temporal/spatial residual
stack of the baseline (which learned one blurry rate-average and lost to the
codec's own RD curve). Instead we use:

    * a **U-Net backbone** (encoder/decoder + skips) as the pixel editor -- the
      same backbone family Yang et al. (TCSVT 2024) use for the machine-vision
      image preprocessor, which demonstrably reaches negative BD-BR;
    * **FiLM** (Perez et al. 2018) conditioning on the target compression
      operating point (normalised QP), so a *single* model adapts across the
      whole rate range instead of averaging it;
    * **SFT** (Wang et al. CVPR 2018) conditioning on a **motion cue** (temporal
      frame difference), so the preprocessor allocates its edits *spatially*
      toward moving / task-relevant regions -- the video-domain novelty over the
      image preprocessors it builds on.

Video is edited frame-wise (2D convs over [B*T, C, H, W]); the only temporal
signal is the motion cue that drives the SFT, which keeps it cheap on a T4 while
still being motion-aware. Temporal *coherence* of the edit is enforced by the
temporal-consistency loss (see ``src/losses.py``), not by 3D convs.

The residual tail and both modulators are zero-initialised, so the network
starts as an exact identity (stable early training) and only "switches on" its
edits as it learns. Only this module is trained; codec + analyzer are frozen.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _act() -> nn.Module:
    return nn.LeakyReLU(0.1, inplace=True)


class FiLM(nn.Module):
    """Global per-channel affine from the rate condition (zero-init -> identity)."""

    def __init__(self, cond_dim: int, ch: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cond_dim, ch), _act(), nn.Linear(ch, 2 * ch))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.net(cond).chunk(2, dim=1)  # [N, ch] each
        return x * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]


class SFT(nn.Module):
    """Spatially-varying affine from the motion cue (zero-init -> identity)."""

    def __init__(self, ch: int, cue_ch: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cue_ch, ch, 3, padding=1), _act(), nn.Conv2d(ch, 2 * ch, 3, padding=1)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, cue: torch.Tensor) -> torch.Tensor:
        if cue.shape[-2:] != x.shape[-2:]:
            cue = F.interpolate(cue, size=x.shape[-2:], mode="bilinear", align_corners=False)
        gamma, beta = self.net(cue).chunk(2, dim=1)  # [N, ch, H, W] each
        return x * (1.0 + gamma) + beta


class _ConvBlock(nn.Module):
    """conv-act-conv, then FiLM (rate) and SFT (motion) modulation."""

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.film = FiLM(cond_dim, out_ch)
        self.sft = SFT(out_ch)
        self.act = _act()

    def forward(self, x, cond, cue):
        h = self.act(self.conv1(x))
        h = self.act(self.conv2(h))
        h = self.film(h, cond)
        h = self.sft(h, cue)
        return h


class _UNet(nn.Module):
    """3-level U-Net (full, /2, /4). All blocks FiLM+SFT conditioned."""

    def __init__(self, in_ch: int, base_ch: int, cond_dim: int):
        super().__init__()
        c1, c2, c3 = base_ch, base_ch * 2, base_ch * 4
        self.enc1 = _ConvBlock(in_ch, c1, cond_dim)
        self.enc2 = _ConvBlock(c1, c2, cond_dim)
        self.bott = _ConvBlock(c2, c3, cond_dim)
        self.dec2 = _ConvBlock(c3 + c2, c2, cond_dim)
        self.dec1 = _ConvBlock(c2 + c1, c1, cond_dim)
        self.tail = nn.Conv2d(c1, in_ch, 3, padding=1)
        nn.init.zeros_(self.tail.weight)  # start as identity residual
        nn.init.zeros_(self.tail.bias)

    def forward(self, x, cond, cue):
        e1 = self.enc1(x, cond, cue)                       # full
        e2 = self.enc2(F.avg_pool2d(e1, 2), cond, cue)     # /2
        b = self.bott(F.avg_pool2d(e2, 2), cond, cue)      # /4
        d2 = F.interpolate(b, size=e2.shape[-2:], mode="nearest")
        d2 = self.dec2(torch.cat([d2, e2], dim=1), cond, cue)
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="nearest")
        d1 = self.dec1(torch.cat([d1, e1], dim=1), cond, cue)
        return self.tail(d1)


class VideoPreprocessor(nn.Module):
    """Rate- and motion-conditioned U-Net preprocessor with structural saliency
    gating (upgrade-6, contributions D1/D2/D3).

    Args:
        in_ch:    input channels (3 for RGB).
        base_ch:  U-Net width at full resolution.
        res_scale: scales the learned residual before adding to the input.
        cond_dim: rate condition width (1 = normalised QP; sized for later
                  appending an explicit log target-rate for rate control).
        gate:     when True and a task-saliency ``mask`` is supplied to
                  :meth:`forward`, the pixel edit is scaled by ``(1 - mask)``:
                  ``x_pre = x + (1 - M) * edit``. Task-critical pixels (M=1)
                  become an EXACT identity by construction -- accuracy at light
                  QPs can no longer be traded away by any loss term, the failure
                  mode that ended upgrade-5.1 (11 loss variants, all failing
                  the QP30-gap selection; the gate removes the trade-off from
                  the optimisation's reach entirely).
        gate_area: D2 hard mask -- top-`gate_area` fraction of pixels per clip
                  by saliency is protected with an EXACT identity.
        edit_kind: "residual" (default, 5.1 behaviour: free bounded pixel edit)
                  or "smooth" (D3: the U-Net outputs a per-pixel LOW-PASS
                  STRENGTH in [0,1]; the output is a convex blend
                  ``x_pre = x + s * (blur(x) - x)``). D2's diagnosis: the
                  residual edit with mu=10 learns rate-ADDING noise (+5% bpp at
                  every QP) because MSE-to-source penalises real smoothing;
                  the smooth parameterisation makes "cheaper than source" the
                  ONLY possible direction -- blur(x) never adds bits, so
                  rate-adding edits are architecturally impossible.
    """

    def __init__(self, in_ch: int = 3, base_ch: int = 32, res_scale: float = 1.0,
                 cond_dim: int = 1, max_relative_edit: float = 0.25,
                 gate: bool = True, gate_area: float = 0.0,
                 edit_kind: str = "residual"):
        super().__init__()
        if not 0.0 < max_relative_edit <= 1.0:
            raise ValueError("max_relative_edit must be in (0, 1]")
        if not 0.0 <= gate_area < 1.0:
            raise ValueError("gate_area must be in [0, 1)")
        if edit_kind not in ("residual", "smooth"):
            raise ValueError(f"edit_kind must be 'residual' or 'smooth', got {edit_kind!r}")
        self.cond_dim = cond_dim
        self.res_scale = res_scale
        self.max_relative_edit = max_relative_edit
        self.gate = gate
        # D2: hard binarisation of the saliency mask. The D1 soft gate (per-clip
        # min-max-normalised saliency) leaves mask values ~0.3-0.5 on much of the
        # task region, so a strong edit (mu=10) still gets 50-70% through and the
        # QP30-gap only reached -0.251. gate_area > 0 protects the top-`gate_area`
        # fraction of pixels PER CLIP by saliency (scale-free): those become an
        # exact identity, everything else is fully editable.
        self.gate_area = gate_area
        # D3: smoothing-first parameterisation (see class docstring).
        self.edit_kind = edit_kind
        self.unet = _UNet(in_ch, base_ch, cond_dim)

    @staticmethod
    def _gaussian_blur(x: torch.Tensor, kernel_size: int = 5, sigma: float = 2.0) -> torch.Tensor:
        """Differentiable gaussian blur over the last two (H, W) dims.

        Separable 1-D kernels keep it cheap; padding is replicate so border
        pixels stay statistics-preserving. Accepts [..., C, H, W]."""
        k = kernel_size
        coords = torch.arange(k, dtype=x.dtype, device=x.device) - (k - 1) / 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = (g / g.sum()).view(1, 1, 1, k)
        lead, c = x.shape[:-3], x.shape[-3]
        x = x.reshape(-1, c, *x.shape[-2:])
        gk = g.expand(c, 1, 1, k).reshape(c, 1, 1, k)
        x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="replicate"), gk, groups=c)
        gk = g.reshape(1, 1, k, 1).expand(c, 1, k, 1).reshape(c, 1, k, 1)
        x = F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="replicate"), gk, groups=c)
        return x.reshape(*lead, c, *x.shape[-2:])

    @staticmethod
    def _motion_cue(x: torch.Tensor) -> torch.Tensor:
        """[B,C,T,H,W] -> [B,1,T,H,W] per-clip-normalised temporal abs-diff.

        Frame 0 reuses frame 1's diff so the cue is defined for every frame."""
        d = (x[:, :, 1:] - x[:, :, :-1]).abs().mean(dim=1, keepdim=True)  # [B,1,T-1,H,W]
        if d.shape[2] == 0:  # single-frame clip: no motion
            return torch.zeros_like(x[:, :1])
        cue = torch.cat([d[:, :, :1], d], dim=2)  # [B,1,T,H,W]
        peak = cue.amax(dim=(2, 3, 4), keepdim=True)
        return cue / (peak + 1e-6)

    @staticmethod
    def _fold(x):
        b, c, t, h, w = x.shape
        return x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w), (b, t)

    @staticmethod
    def _unfold(x, bt):
        b, t = bt
        n, c, h, w = x.shape
        return x.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: [B,C,T,H,W] in [0,1]. cond: [B, cond_dim] operating point (defaults
        to zeros = top quality). mask: optional detached [B,1,T,H,W] saliency in
        [0,1] (1 = task-critical). When the gate is enabled and a mask is given,
        the edit is scaled by (1 - mask) so task-critical pixels pass through
        unmodified. Returns edited video in [0,1]."""
        if x.ndim != 5 or x.shape[1] != 3:
            raise ValueError(f"expected [B,3,T,H,W], got {tuple(x.shape)}")
        b, c, t, h, w = x.shape
        if cond is None:
            cond = x.new_zeros(b, self.cond_dim)
        cue = self._motion_cue(x)
        frames, bt = self._fold(x)
        cue_f, _ = self._fold(cue)
        cond_f = cond.repeat_interleave(t, dim=0)  # [B*T, cond_dim]
        if self.edit_kind == "smooth":
            # D3: U-Net outputs a per-pixel low-pass strength in [0,1]; the
            # frame becomes a convex blend toward its gaussian blur. s=0 keeps
            # the source, s=1 is full blur; nothing can ever ADD detail.
            s = torch.sigmoid(self.unet(frames, cond_f, cue_f))
            out = frames + s * (self._gaussian_blur(frames) - frames)
        else:
            delta = torch.tanh(self.res_scale * self.unet(frames, cond_f, cue_f))
            # Bound the editor itself, not merely its final pixels. Positive edits
            # consume a fraction of the distance to white; negative edits consume a
            # fraction of the source value. This guarantees [0,1], preserves exact
            # identity at initialization and, crucially, makes an all-black collapse
            # impossible in one preprocessing pass even if the rate loss dominates.
            positive = delta.clamp_min(0.0) * (1.0 - frames)
            negative = delta.clamp_max(0.0) * frames
            out = frames + self.max_relative_edit * (positive + negative)
        if self.gate and mask is not None:
            # D1: structural saliency gating. Scale the *edit*, not the pixels:
            # mask=1 -> exact identity regardless of what the loss wants.
            # D2: optionally binarise first -- protect the top-`gate_area`
            # fraction of pixels per clip (scale-free hard protection).
            if self.gate_area and self.gate_area > 0.0:
                b = mask.shape[0]
                flat = mask.reshape(b, -1)
                thresh = torch.quantile(flat, 1.0 - self.gate_area, dim=1).view(b, 1)
                mask = (flat >= thresh).to(mask.dtype).reshape(mask.shape)
            mask_f, _ = self._fold(mask.to(frames.dtype))
            if mask_f.shape[-2:] != frames.shape[-2:]:
                mask_f = F.interpolate(mask_f, size=frames.shape[-2:],
                                       mode="bilinear", align_corners=False)
            out = frames + (1.0 - mask_f) * (out - frames)
        return self._unfold(out, bt)
