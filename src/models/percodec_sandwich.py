"""v9 model: per-codec POST heads — codec-conditioned restoration.

Evidence from v8's 8-experiment sweep (docs/RESULTS_sandwich.md):
  * STE fine-tuning is CODEC-SPECIFIC: a POST calibrated on x265 lifts h264
    (−5.89 E2) while a POST calibrated on x264 lifts h265 (−3.56 STE) — but
    NEITHER single POST wins both codecs (E1: −4.49/−1.96).
  * TTO inherits the same proxy->x264 bias (+6% h265).
  * Light co-adaptation homogenizes the halves and loses specialization.
Conclusion: the restorer must KNOW which codec's artifacts it is inverting.
The minimal architecture that encodes this: ONE shared restoration trunk +
a codec embedding injected via FiLM alongside the QP condition, so gradient
flow can specialise per codec without duplicating the whole UNet. (A
fully-duplicated per-codec UNet is the ablation arm; this parameterisation
shares the low-level restoration basis — deblocking, deringing — which is
common, and spends capacity only where codecs differ.)

Training (train.py runs as usual): the engine's STE codec is sampled per
step {h264, h265}; ``post_restore`` receives the codec id so FiLM routes.
Eval: the engine passes the codec being evaluated (it already knows — it
loops x264 and x265 arms).

``load_v8_sandwich``: warm-start from a v8 SandwichPreprocessor checkpoint
(PRE + shared trunk copied; codec FiLM zero-init => starts as v8-equivalent).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .upvcm import UPVCMPreprocessor, _FiLM

CODECS = ("h264", "h265")
N_CODECS = len(CODECS)
CODEC_TO_ID = {c: i for i, c in enumerate(CODECS)}


class _PostUNetCodec(nn.Module):
    """Restoration UNet (same geometry as v8's _PostUNet) + codec embedding.

    FiLM condition = [codec_onehot, qp_norm] — zero-init codec FiLM keeps
    v8-warm-start exact (codec block contributes gamma=beta=0 at init).
    """

    def __init__(self, base: int = 32, cond_dim: int = 1):
        super().__init__()
        self.codec_embed = nn.Embedding(N_CODECS, 8)
        nn.init.normal_(self.codec_embed.weight, std=1e-2)
        # cond vector: [codec_embed(8), qp(1)] -> FiLM
        self.film = _FiLM(8 + cond_dim, 4 * base)
        self.in_conv = nn.Conv2d(3, base, 3, padding=1)
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
        self.up2 = nn.ConvTranspose2d(4 * base, 2 * base, 2, stride=2)
        self.dec2 = nn.Sequential(nn.ReLU(inplace=True),
                                  nn.Conv2d(4 * base, 2 * base, 3, padding=1))
        self.up1 = nn.ConvTranspose2d(2 * base, base, 2, stride=2)
        self.dec1 = nn.Sequential(nn.ReLU(inplace=True),
                                  nn.Conv2d(2 * base, base, 3, padding=1))
        self.out_conv = nn.Conv2d(base, 3, 3, padding=1)
        # noise-init (dead-saddle lesson: zero out conv x zero gate = dead)
        nn.init.normal_(self.out_conv.weight, std=1e-3)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor, codec_id: torch.Tensor,
                qp: torch.Tensor) -> torch.Tensor:
        # x [N,3,H,W]; codec_id [N] long; qp [N,1]
        c = torch.cat([self.codec_embed(codec_id), qp], dim=1)  # [N, 8+cond]
        e0 = self.in_conv(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(self.down1(e1))
        b = self.mid(self.down2(e2))
        b = self.film(b, c)
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)


class PerCodecPostSandwich(nn.Module):
    """v9: UP-VCM PRE + codec-conditioned POST around a frozen codec.

    Interface matches v8's SandwichPreprocessor (forward = PRE,
    post_restore = POST) so the engine, eval arms and gates work unchanged —
    except post_restore takes an optional ``codec`` string.
    """

    def __init__(self, s_ch: int = 16, editor_ch: int = 24, cond_dim: int = 1,
                 dino_weight: float = 0.5, dino_name: str = "dinov2_vits14",
                 motion_tau: float = 0.1, post_base: int = 32,
                 w_budget: float = 0.0):
        super().__init__()
        self.pre = UPVCMPreprocessor(
            s_ch=s_ch, editor_ch=editor_ch, cond_dim=cond_dim,
            dino_weight=dino_weight, dino_name=dino_name, motion_tau=motion_tau,
            w_budget=w_budget)
        self.post_net = _PostUNetCodec(base=post_base, cond_dim=cond_dim)
        self.post_strength = nn.Parameter(torch.zeros(()))
        self.bypass_post = False
        self._last_w: torch.Tensor | None = None
        self._last_w_target: torch.Tensor | None = None

    def forward(self, x, cond=None, mask=None):
        x_pre = self.pre(x, cond, mask=mask)
        self._last_w = self.pre._last_w
        self._last_w_target = self.pre._last_w_target
        return x_pre

    def post_restore(self, x_hat: torch.Tensor, cond: torch.Tensor | None,
                     codec: str = "h264") -> torch.Tensor:
        if self.bypass_post:
            return x_hat
        b, c, t, h, w = x_hat.shape
        frames = x_hat.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        if cond is None:
            cond = frames.new_zeros(b, self.pre.cond_dim)
        cid = torch.full((b,), CODEC_TO_ID.get(codec, 0), dtype=torch.long,
                         device=x_hat.device)
        cid_f = cid.repeat_interleave(t, dim=0)
        cond_f = cond.repeat_interleave(t, dim=0).to(frames.dtype)
        delta = self.post_net(frames, cid_f, cond_f)
        out = frames + self.post_strength * delta
        return out.clamp(0.0, 1.0).reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)

    def load_pre_state(self, upvcm_state: dict) -> None:
        self.pre.load_state_dict(upvcm_state, strict=True)

    def load_v8_sandwich(self, v8_state: dict) -> dict:
        """Warm-start from a v8 SandwichPreprocessor state_dict.

        Mapping: PRE + restoration trunk copy name-for-name (same submodule
        names in both classes). The v9 FiLM has a different cond width
        (8+cond vs cond) so v8 film weights are SKIPPED — v9's zero-init FiLM
        final layer makes the whole FiLM contribute gamma=beta=0 at load, so
        post_restore output is exactly v8's at start (verified by test).
        """
        new = self.state_dict()
        skip_prefix = ("post_net.film.", "post_net.codec_embed.")
        copied = 0
        for k, v in v8_state.items():
            if k.startswith(skip_prefix):
                continue
            if k in new and new[k].shape == v.shape:
                new[k] = v
                copied += 1
        missing = [k for k in new if k not in v8_state
                   and not k.startswith(skip_prefix)]
        self.load_state_dict(new)
        return {"copied": copied, "kept_init": len(new) - copied, "missing": missing}
