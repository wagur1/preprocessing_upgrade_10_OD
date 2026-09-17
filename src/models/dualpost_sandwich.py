"""v9-b: TWO fully-separate POST heads (one per codec) — no shared trunk.

The v9-a verdict (docs/RESULTS_percodec.md): shared trunk + codec FiLM
DESTROYED the warm-start specialization (h264 −5.89 → −1.76 in 1079 steps).
The accumulated evidence (E1/E3/v9-a) says per-codec specialization beats
every form of sharing/co-training in this regime.

v9-b takes that conclusion literally:

    PRE (UP-VCM v1) ─► frozen x264 ─► POST_264 (from E2's POST, frozen*)
    PRE (UP-VCM v1) ─► frozen x265 ─► POST_265 (from STE's POST, frozen*)

...with *optional* per-head STE calibration that NEVER mixes codecs: each
head only ever sees its own codec in the loop, so there is no compromise
gradient at all. Routing is an exact switch on the codec id (no embedding to
learn — nothing to get wrong).

Warm-start: POST_264 from the E2 frankenstein-STE checkpoint (the h264 record
holder), POST_265 from the v8 STE checkpoint (the h265 record holder). With
STE steps = 0 (calibration off) this is EXACTLY the two records assembled in
one model — the guaranteed-minimum configuration. STE calibration (per-head,
disjoint) can only be tested as a delta on top.

Default config runs 0 additional STE steps (pure assembly eval): the records
are already measured, and every continued-training variant tested so far
(E1/E3/v9-a) LOST ground. If calibration is wanted: train.ste_per_head=800
gives each head 800 steps on its own codec only.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .sandwich import SandwichPreprocessor


class DualPostSandwich(nn.Module):
    """PRE + two independent POST UNets; exact codec switch at restore time.

    Interface-compatible with SandwichPreprocessor (forward = PRE,
    post_restore = POST with an extra codec=... argument).
    """

    def __init__(self, s_ch: int = 16, editor_ch: int = 24, cond_dim: int = 1,
                 dino_weight: float = 0.5, dino_name: str = "dinov2_vits14",
                 motion_tau: float = 0.1, post_base: int = 32,
                 w_budget: float = 0.0):
        super().__init__()
        # PRE shared by both paths (same as all v8/v9 variants)
        from .upvcm import UPVCMPreprocessor
        self.pre = UPVCMPreprocessor(
            s_ch=s_ch, editor_ch=editor_ch, cond_dim=cond_dim,
            dino_weight=dino_weight, dino_name=dino_name, motion_tau=motion_tau,
            w_budget=w_budget)
        # two INDEPENDENT restorers (same _PostUNet geometry as v8)
        from .sandwich import _PostUNet
        self.post_264 = _PostUNet(base=post_base, cond_dim=cond_dim)
        self.post_265 = _PostUNet(base=post_base, cond_dim=cond_dim)
        self.post_strength_264 = nn.Parameter(torch.zeros(()))
        self.post_strength_265 = nn.Parameter(torch.zeros(()))
        self.bypass_post = False
        self._last_w: torch.Tensor | None = None
        self._last_w_target: torch.Tensor | None = None

    # -- PRE (engine interface unchanged) ---------------------------------
    def forward(self, x, cond=None, mask=None):
        x_pre = self.pre(x, cond, mask=mask)
        self._last_w = self.pre._last_w
        self._last_w_target = self.pre._last_w_target
        return x_pre

    # -- POST: exact codec switch ------------------------------------------
    def post_restore(self, x_hat: torch.Tensor, cond: torch.Tensor | None,
                     codec: str = "h264") -> torch.Tensor:
        if self.bypass_post:
            return x_hat
        b, c, t, h, w = x_hat.shape
        frames = x_hat.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        if cond is None:
            cond = frames.new_zeros(b, self.pre.cond_dim)
        cond_f = cond.repeat_interleave(t, dim=0).to(frames.dtype)
        if codec == "h265":
            delta = self.post_265(frames, cond_f)
            out = frames + self.post_strength_265 * delta
        else:  # h264 (default)
            delta = self.post_264(frames, cond_f)
            out = frames + self.post_strength_264 * delta
        return out.clamp(0.0, 1.0).reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)

    # -- assembly from two record-holder checkpoints ------------------------
    def load_dual(self, pre_state: dict, post264_from: dict,
                  post265_from: dict) -> dict:
        """pre_state: UP-VCM state_dict (v7-v1 best).
        post264_from / post265_from: SandwichPreprocessor state_dicts —
        their ``post_net.*`` weights map onto ``post_264.*`` / ``post_265.*``
        respectively, and ``post_strength`` onto the matching per-head gate.
        """
        self.pre.load_state_dict(pre_state, strict=True)
        report = {}
        for name, src in (("post_264", post264_from), ("post_265", post265_from)):
            new = self.state_dict()
            copied = 0
            for k, v in src.items():
                if k.startswith("post_net."):
                    dst = f"{name}.{k[len('post_net.'):]}"
                    if dst in new and new[dst].shape == v.shape:
                        new[dst] = v
                        copied += 1
                elif k == "post_strength":
                    new[f"post_strength_{name.split('_')[1]}"] = v
                    copied += 1
            self.load_state_dict(new)
            report[name] = copied
        return report


class DualCodecSandwich(nn.Module):
    """v9-b full: PER-CODEC PRE + PER-CODEC POST — the modular union.

    Evidence table (all full-n or best partial, v8 docs):
      h264 record  E2: PRE-v1 + STE-POST  -> -5.89
      h265 record STE: STE-PRE + STE-POST -> -3.56
    Same POST, different PREs — so the union requires routing BOTH halves by
    codec. Deployable: the encoder knows its codec (it IS the encoder), the
    decoder knows its codec (it IS the decoder) — an exact switch is honest
    conditioning, like per-codec presets.

    forward(x, cond, mask, codec) routes PRE; post_restore(..., codec) routes
    POST. With zero additional training this reproduces each record arm
    exactly (same weights as the record runs) — the system-level claim:
    "codec-conditioned modularity beats every shared/co-trained variant"
    (supported by the E1/E3/v9-a negatives).
    """

    def __init__(self, s_ch: int = 16, editor_ch: int = 24, cond_dim: int = 1,
                 dino_weight: float = 0.5, dino_name: str = "dinov2_vits14",
                 motion_tau: float = 0.1, post_base: int = 32,
                 w_budget: float = 0.0):
        super().__init__()
        from .upvcm import UPVCMPreprocessor
        from .sandwich import _PostUNet
        kw = dict(s_ch=s_ch, editor_ch=editor_ch, cond_dim=cond_dim,
                  dino_weight=dino_weight, dino_name=dino_name,
                  motion_tau=motion_tau, w_budget=w_budget)
        self.pre_264 = UPVCMPreprocessor(**kw)
        self.pre_265 = UPVCMPreprocessor(**kw)
        self.post_264 = _PostUNet(base=post_base, cond_dim=cond_dim)
        self.post_265 = _PostUNet(base=post_base, cond_dim=cond_dim)
        self.post_strength_264 = nn.Parameter(torch.zeros(()))
        self.post_strength_265 = nn.Parameter(torch.zeros(()))
        self.bypass_post = False
        self._last_w = None
        self._last_w_target = None

    def forward(self, x, cond=None, mask=None, codec: str = "h264"):
        pre = self.pre_265 if codec == "h265" else self.pre_264
        x_pre = pre(x, cond, mask=mask)
        self._last_w = pre._last_w
        self._last_w_target = pre._last_w_target
        return x_pre

    def post_restore(self, x_hat, cond=None, codec: str = "h264"):
        if self.bypass_post:
            return x_hat
        b, c, t, h, w = x_hat.shape
        frames = x_hat.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        if cond is None:
            cond = frames.new_zeros(b, self.pre_264.cond_dim)
        cond_f = cond.repeat_interleave(t, dim=0).to(frames.dtype)
        if codec == "h265":
            delta = self.post_265(frames, cond_f)
            out = frames + self.post_strength_265 * delta
        else:
            delta = self.post_264(frames, cond_f)
            out = frames + self.post_strength_264 * delta
        return out.clamp(0.0, 1.0).reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)

    def load_record_assembly(self, v1_state: dict, ste_state: dict) -> dict:
        """v1_state: UP-VCM state (v7-v1 best) -> pre_264.
        ste_state: v8 SandwichPreprocessor ckpt state (the STE run) ->
        pre_265 (its PRE) + both POST heads (its POST — the shared record
        holder; kept as two copies so per-head calibration stays possible).
        """
        from .upvcm import UPVCMPreprocessor
        self.pre_264.load_state_dict(v1_state, strict=True)
        new = self.state_dict()
        copied = 0
        for k, v in ste_state.items():
            if k.startswith("pre."):
                dst = f"pre_265.{k[4:]}"
                if dst in new and new[dst].shape == v.shape:
                    new[dst] = v
                    copied += 1
            elif k.startswith("post_net."):
                stem = k[len("post_net."):]
                for dst in (f"post_264.{stem}", f"post_265.{stem}"):
                    if dst in new and new[dst].shape == v.shape:
                        new[dst] = v
                        copied += 1
            elif k == "post_strength":
                new["post_strength_264"] = v
                new["post_strength_265"] = v
                copied += 2
        self.load_state_dict(new)
        return {"copied": copied}
