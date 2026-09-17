"""DINOv2-based semantic energy map for the UP-VCM importance head.

The deploy-time importance prediction is the small distilled ``S`` head inside
``UPVCMPreprocessor`` (analyzer-free, foundation-free). This module supplies a
*training-time only* anchor for the distillation target: DINOv2 patch-token
energy, an analyzer-independent measure of semantic density. Blending it with
the multi-teacher task saliency guards the mask against a single teacher's
class biases (the omega=1.0 teacher-overfit lesson, arXiv:1910.09185).

Never loaded at eval: ``UPVCMPreprocessor`` only calls this when a distill
target is being built (training) and the model is available. A failed load
prints one warning and the target falls back to teacher saliency alone —
the run stays valid, the blend weight is logged in the checkpoint config.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

_DINO_CACHE: dict = {}


def get_dino(name: str = "dinov2_vits14", device="cpu"):
    """Lazy singleton DINOv2 load. Returns None on failure (fallback path)."""
    key = (name, str(device))
    if key in _DINO_CACHE:
        return _DINO_CACHE[key]
    model = None
    try:
        model = torch.hub.load("facebookresearch/dinov2", name, pretrained=True)
        model = model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        print(f"[dino] loaded {name} on {device}")
    except Exception as e:  # network / hub failure -> teacher-only target
        print(f"[dino] load failed ({type(e).__name__}: {e}); "
              "distill target falls back to teacher saliency only")
        model = None
    _DINO_CACHE[key] = model
    return model


@torch.no_grad()
def dino_energy(x: torch.Tensor, model, patch: int = 14,
                max_frames: int = 4) -> torch.Tensor:
    """Patch-token L2 energy of DINOv2 as a per-pixel importance map.

    x: [B,3,T,H,W] in [0,1] -> [B,1,T,H,W] in [0,1] (per-frame min-max).

    Only <=max_frames are scored (time cost); the map is bilinearly upsampled
    in time to T. The input is resized to the nearest patch multiple and
    normalised with ImageNet statistics, as DINOv2 expects.
    """
    b, c, t, h, w = x.shape
    step = max(1, t // max_frames)
    idx = list(range(0, t, step))[:max_frames]
    frames = x[:, :, idx]                                  # [B,3,t',H,W]
    size = (min(h, w) // patch) * patch
    if size < patch:
        raise ValueError(f"frame too small for DINOv2 patch {patch}: {h}x{w}")
    fr = frames.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
    fr = F.interpolate(fr, size=(size, size), mode="bilinear", align_corners=False)
    mean = torch.tensor(_IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    fr = (fr - mean) / std
    feats = model.forward_features(fr)["x_norm_patchtokens"]  # [B*t', N, D]
    e = feats.norm(dim=-1)                                   # [B*t', N]
    g = size // patch
    e = e.view(b, len(idx), g, g)
    e = F.interpolate(e, size=(h, w), mode="bilinear", align_corners=False)
    mn = e.amin(dim=(-2, -1), keepdim=True)
    mx = e.amax(dim=(-2, -1), keepdim=True)
    e = (e - mn) / (mx - mn + 1e-6)                          # [B,t',H,W] in [0,1]
    # time upsample: [B,t',H,W] -> [B,1,T,H,W]
    e = e.permute(0, 2, 3, 1).reshape(b, 1, -1, len(idx))    # [B,1,HW,t']
    e = F.interpolate(e, size=(h * w, t), mode="bilinear", align_corners=False)
    return e.reshape(b, 1, h, w, t).permute(0, 1, 4, 2, 3).contiguous()
