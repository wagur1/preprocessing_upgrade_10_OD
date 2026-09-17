"""Self-contained SiamFC tracker (the default GOT-10k analyzer).

A real fully-convolutional Siamese tracker built on a *frozen* ImageNet
backbone, so it plugs into the paper's framework as a fixed analyzer (only the
preprocessor is trained). It gives:

  * a differentiable **training loss** ``L_Acc`` -- the SiamFC balanced logistic
    loss on the cross-correlation response, so the preprocessor learns to keep
    the target localisable after compression, and
  * a real **inference tracker** (multi-scale search + cosine-window penalty)
    that produces per-frame boxes for genuine GOT-10k AUC.

Geometry (SiamFC conventions):
  exemplar 127 px, search 255 px, backbone total stride ``S`` (8 for resnet18
  up to layer2). Feature sizes fz = 127/S, fx = 255/S; response R = fx - fz + 1.
  A response cell ``r`` corresponds to search-image coordinate
  ``(r + (fz-1)/2) * S``.

The backbone is frozen and ImageNet-pretrained (no tracker-specific training),
so absolute AUC is modest; because the tracker is identical across all
pipelines, the *relative* BD-Rate (prep vs H.264/H.265) is the meaningful
number. Swap in a real pytracking tracker (see ``pytracking_adapter``) for the
paper's exact KYS/DiMP/ATOM/PrDiMP numbers.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def crop_and_resize(
    img: torch.Tensor,
    center: torch.Tensor,
    side: torch.Tensor,
    out_size: int,
) -> torch.Tensor:
    """Differentiable square crop + resize via grid_sample (border padding).

    Args:
        img: [N, C, H, W].
        center: [N, 2] crop centre (cx, cy) in pixels.
        side: [N] square crop side in pixels.
        out_size: output spatial size.
    Returns: [N, C, out_size, out_size].
    """
    n, c, h, w = img.shape
    device = img.device
    lin = torch.linspace(-0.5, 0.5, out_size, device=device)  # [out]
    gy, gx = torch.meshgrid(lin, lin, indexing="ij")           # [out,out]
    gx = gx.unsqueeze(0) * side.view(n, 1, 1) + center[:, 0].view(n, 1, 1)
    gy = gy.unsqueeze(0) * side.view(n, 1, 1) + center[:, 1].view(n, 1, 1)
    # to normalised [-1,1] (align_corners=True)
    gx = 2.0 * gx / max(w - 1, 1) - 1.0
    gy = 2.0 * gy / max(h - 1, 1) - 1.0
    grid = torch.stack([gx, gy], dim=-1)  # [N,out,out,2]
    return F.grid_sample(
        img, grid, mode="bilinear", padding_mode="border", align_corners=True
    )


def _xcorr(z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Batched full cross-correlation. z:[B,C,hz,wz] x:[B,C,hx,wx] -> [B,1,ho,wo]."""
    b, c, hz, wz = z.shape
    _, _, hx, wx = x.shape
    x = x.reshape(1, b * c, hx, wx)
    out = F.conv2d(x, z, groups=b)  # weight [b, c, hz, wz], groups=b -> [1,b,ho,wo]
    return out.reshape(b, 1, out.shape[-2], out.shape[-1])


def _box_to_center_size(boxes_px: torch.Tensor):
    """xyxy pixels -> (center [N,2], wh [N,2])."""
    cx = 0.5 * (boxes_px[:, 0] + boxes_px[:, 2])
    cy = 0.5 * (boxes_px[:, 1] + boxes_px[:, 3])
    w = (boxes_px[:, 2] - boxes_px[:, 0]).clamp(min=1.0)
    h = (boxes_px[:, 3] - boxes_px[:, 1]).clamp(min=1.0)
    return torch.stack([cx, cy], dim=1), torch.stack([w, h], dim=1)


def _exemplar_side(wh: torch.Tensor) -> torch.Tensor:
    """SiamFC context-padded exemplar side in pixels."""
    w, h = wh[:, 0], wh[:, 1]
    context = 0.5 * (w + h)
    return torch.sqrt((w + context) * (h + context))


class SiamFCNet(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet18",
        exemplar_size: int = 127,
        search_size: int = 255,
        pos_radius: float = 2.0,
        response_up: int = 16,
    ):
        super().__init__()
        if backbone != "resnet18":
            raise ValueError("only resnet18 is wired for the SiamFC backbone")
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # up to layer2 -> total stride 8, 128 channels
        self.features = nn.Sequential(
            net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2
        )
        self.total_stride = 8
        self.exemplar_size = exemplar_size
        self.search_size = search_size
        self.pos_radius = pos_radius
        self.response_up = response_up
        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

        self.fz = exemplar_size // self.total_stride
        self.fx = search_size // self.total_stride
        self.resp = self.fx - self.fz + 1

    # -- features ----------------------------------------------------------
    def embed(self, patch: torch.Tensor) -> torch.Tensor:
        return self.features((patch - self.mean) / self.std)

    # -- geometry helpers --------------------------------------------------
    def _search_side(self, wh: torch.Tensor) -> torch.Tensor:
        return _exemplar_side(wh) * (self.search_size / self.exemplar_size)

    def response_coord_to_search_px(self, r: float) -> float:
        return (r + (self.fz - 1) / 2.0) * self.total_stride

    # -- training path -----------------------------------------------------
    def training_loss(
        self, clip: torch.Tensor, boxes_norm: torch.Tensor
    ) -> torch.Tensor:
        """SiamFC balanced logistic loss over a clip.

        clip: [B,C,T,H,W] in [0,1]; boxes_norm: [B,T,4] normalised xyxy.
        Template = compressed frame 0; searches = frames 1..T-1 centred on the
        previous frame's box. Returns a scalar loss.
        """
        b, c, t, h, w = clip.shape
        scale = clip.new_tensor([w, h, w, h])
        boxes_px = boxes_norm * scale  # [B,T,4] pixels

        # exemplar from frame 0
        c0, wh0 = _box_to_center_size(boxes_px[:, 0])
        s_z = _exemplar_side(wh0)
        z = crop_and_resize(clip[:, :, 0], c0, s_z, self.exemplar_size)
        z_feat = self.embed(z)  # [B,C,fz,fz]

        total = clip.new_zeros(())
        count = 0
        for ti in range(1, t):
            ref_c, ref_wh = _box_to_center_size(boxes_px[:, ti - 1])
            cur_c, _ = _box_to_center_size(boxes_px[:, ti])
            s_x = self._search_side(ref_wh)
            x = crop_and_resize(clip[:, :, ti], ref_c, s_x, self.search_size)
            x_feat = self.embed(x)
            resp = _xcorr(z_feat, x_feat)  # [B,1,R,R]

            # target position within the search window, in search-image px
            half = s_x * 0.5
            sc = (cur_c - (ref_c - half.unsqueeze(1))) / s_x.unsqueeze(1) * self.search_size
            # -> response-cell coords
            rc = sc / self.total_stride - (self.fz - 1) / 2.0  # [B,2] (x,y)

            label, weight = self._make_labels(rc, resp.shape[-1], resp.device)
            total = total + _balanced_logistic(resp.squeeze(1), label, weight)
            count += 1
        return total / max(count, 1)

    def _make_labels(self, rc: torch.Tensor, R: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build ±1 labels + balancing weights. rc: [B,2] (col,row) peak coords."""
        b = rc.shape[0]
        ys = torch.arange(R, device=device).view(1, R, 1).float()
        xs = torch.arange(R, device=device).view(1, 1, R).float()
        cx = rc[:, 0].view(b, 1, 1)
        cy = rc[:, 1].view(b, 1, 1)
        dist = torch.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
        pos = (dist <= self.pos_radius).float()  # [B,R,R]
        label = pos * 2.0 - 1.0
        n_pos = pos.sum(dim=(1, 2), keepdim=True).clamp(min=1.0)
        n_neg = (1.0 - pos).sum(dim=(1, 2), keepdim=True).clamp(min=1.0)
        weight = pos * (0.5 / n_pos) + (1.0 - pos) * (0.5 / n_neg)
        return label, weight

    # -- inference ---------------------------------------------------------
    @torch.no_grad()
    def track_sequence(
        self, clip: torch.Tensor, init_box_norm: np.ndarray
    ) -> np.ndarray:
        """Track through a clip. clip:[1,C,T,H,W]; init_box_norm: [4] xyxy norm.

        Returns [T,4] normalised xyxy predicted boxes (frame 0 = init box).
        """
        device = clip.device
        _, c, t, H, W = clip.shape
        scale = np.array([W, H, W, H], dtype=np.float64)
        box_px = np.asarray(init_box_norm, dtype=np.float64) * scale
        cx = 0.5 * (box_px[0] + box_px[2])
        cy = 0.5 * (box_px[1] + box_px[3])
        tw = max(box_px[2] - box_px[0], 1.0)
        th = max(box_px[3] - box_px[1], 1.0)

        center = torch.tensor([[cx, cy]], device=device, dtype=torch.float32)
        wh = torch.tensor([[tw, th]], device=device, dtype=torch.float32)
        s_z = _exemplar_side(wh)
        z = crop_and_resize(clip[:, :, 0], center, s_z, self.exemplar_size)
        z_feat = self.embed(z)

        # scale pyramid + cosine window
        n_scales = 3
        scale_step = 1.0375
        scales = scale_step ** (np.arange(n_scales) - (n_scales - 1) / 2)
        scale_penalty = 0.97
        up = self.response_up
        Rup = self.resp * up
        hann = np.outer(np.hanning(Rup), np.hanning(Rup))
        hann = hann / hann.sum()
        win_influence = 0.18

        boxes = np.zeros((t, 4), dtype=np.float64)
        boxes[0] = init_box_norm
        cur_cx, cur_cy, cur_w, cur_h = cx, cy, tw, th

        for ti in range(1, t):
            base_wh = torch.tensor([[cur_w, cur_h]], device=device, dtype=torch.float32)
            s_x = float(self._search_side(base_wh).item())
            best = None
            for si, sc in enumerate(scales):
                side = s_x * sc
                ctr = torch.tensor([[cur_cx, cur_cy]], device=device, dtype=torch.float32)
                side_t = torch.tensor([side], device=device, dtype=torch.float32)
                x = crop_and_resize(clip[:, :, ti], ctr, side_t, self.search_size)
                x_feat = self.embed(x)
                resp = _xcorr(z_feat, x_feat)  # [1,1,R,R]
                r = F.interpolate(
                    resp, size=(Rup, Rup), mode="bicubic", align_corners=False
                )[0, 0].detach().cpu().numpy()
                r = r - r.min()
                if r.max() > 0:
                    r = r / r.max()
                if si != (n_scales - 1) // 2:
                    r = r * scale_penalty
                peak = r.max()
                if best is None or peak > best[0]:
                    best = (peak, si, r, side)

            _peak, si, r, side = best
            r = (1 - win_influence) * r + win_influence * hann * (r.max() if r.max() > 0 else 1.0)
            pr, pc = np.unravel_index(int(r.argmax()), r.shape)
            # upsampled-cell -> response-cell -> search-image px
            cell_c = pc / up
            cell_r = pr / up
            sx_px = self.response_coord_to_search_px(cell_c)
            sy_px = self.response_coord_to_search_px(cell_r)
            disp_x = (sx_px - self.search_size / 2.0) * side / self.search_size
            disp_y = (sy_px - self.search_size / 2.0) * side / self.search_size
            cur_cx += float(disp_x)
            cur_cy += float(disp_y)
            # damped scale update
            sc = scales[si]
            lr = 0.59
            cur_w *= (1 - lr) + lr * sc
            cur_h *= (1 - lr) + lr * sc
            cur_cx = float(np.clip(cur_cx, 0, W - 1))
            cur_cy = float(np.clip(cur_cy, 0, H - 1))

            boxes[ti] = [
                (cur_cx - cur_w / 2) / W,
                (cur_cy - cur_h / 2) / H,
                (cur_cx + cur_w / 2) / W,
                (cur_cy + cur_h / 2) / H,
            ]
        return np.clip(boxes, 0.0, 1.0)


def _balanced_logistic(
    resp: torch.Tensor, label: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """SiamFC logistic loss: mean over batch of sum_w log(1 + exp(-y*v))."""
    loss = F.softplus(-label * resp)  # log(1+exp(-y*v))
    return (loss * weight).sum(dim=(1, 2)).mean()
