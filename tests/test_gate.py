"""Checks for D1 structural saliency gating (upgrade-6 preprocessor)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.preprocessor import VideoPreprocessor


def test_gate() -> None:
    torch.manual_seed(0)
    pre = VideoPreprocessor(base_ch=8)
    x = torch.rand(2, 3, 4, 16, 16)

    # make the edit non-trivial: un-zero the tail
    with torch.no_grad():
        for p in pre.unet.tail.parameters():
            p.normal_(0, 0.1)
    cond = torch.zeros(2, 1)

    out_free = pre(x, cond)
    assert not torch.allclose(out_free, x), "edit should be non-trivial after tail init"

    # mask = 1 everywhere -> EXACT identity, edit fully suppressed
    mask1 = torch.ones(2, 1, 4, 16, 16)
    out_g1 = pre(x, cond, mask=mask1)
    assert torch.allclose(out_g1, x, atol=1e-6), "mask=1 must give exact identity"

    # mask = 0 everywhere -> gate transparent, equals ungated output
    mask0 = torch.zeros(2, 1, 4, 16, 16)
    out_g0 = pre(x, cond, mask=mask0)
    assert torch.allclose(out_g0, out_free, atol=1e-6), "mask=0 must equal ungated edit"

    # partial mask -> edit scaled per-pixel: out = x + (1-m)*(out_free - x)
    maskp = torch.rand(2, 1, 4, 16, 16)
    out_gp = pre(x, cond, mask=maskp)
    expect = x + (1.0 - maskp) * (out_free - x)
    assert torch.allclose(out_gp, expect, atol=1e-5), "partial mask must scale the edit"

    # gate disabled -> mask ignored (ablation path, 5.1 behaviour)
    pre_off = VideoPreprocessor(base_ch=8, gate=False)
    pre_off.load_state_dict(pre.state_dict())
    out_off = pre_off(x, cond, mask=mask1)
    assert torch.allclose(out_off, out_free, atol=1e-6), "gate=False must ignore the mask"

    # gradient flows through the gate (edit part), mask itself is a constant
    x_g = x.clone().requires_grad_(True)
    mask_half = torch.rand(2, 1, 4, 16, 16)
    mask_half[..., :8] = 1.0  # left half exactly protected
    out = pre(x_g, cond, mask=mask_half)
    out.sum().backward()
    assert x_g.grad is not None and torch.isfinite(x_g.grad).all()
    # protected half: the direct path contributes exactly 1; small deviations
    # (<=~1e-3) come from x feeding unprotected pixels' edits via the shared
    # motion cue -- mathematically correct, forward identity is still exact.
    assert torch.allclose(x_g.grad[..., :8], torch.ones_like(x_g.grad[..., :8]), atol=2e-3)
    assert not torch.allclose(x_g.grad[..., 8:], torch.ones_like(x_g.grad[..., 8:]), atol=2e-3), \
        "unprotected half should carry edit gradients"

    # spatial-size mismatch: mask at half resolution gets upsampled, no crash
    mask_small = torch.rand(2, 1, 4, 8, 8)
    out_ms = pre(x, cond, mask=mask_small)
    assert out_ms.shape == x.shape


def test_gate_area() -> None:
    torch.manual_seed(1)
    pre = VideoPreprocessor(base_ch=8, gate_area=0.25)
    with torch.no_grad():
        for p in pre.unet.tail.parameters():
            p.normal_(0, 0.1)
    x = torch.rand(2, 3, 4, 16, 16)
    cond = torch.zeros(2, 1)
    out_free = pre(x, cond)

    # known saliency: right half high, left half low
    mask = torch.zeros(2, 1, 4, 16, 16)
    mask[..., 8:] = 1.0  # top-50% area by construction
    out = pre(x, cond, mask=mask)
    # gate_area=0.25 protects only the top quarter -> the highest-saliency half
    # is protected, so the right half must be exact identity...
    assert torch.allclose(out[..., 8:], x[..., 8:], atol=1e-6), \
        "hard gate must make the protected (top-area) half exact identity"
    # ...while the unprotected left half keeps the full edit
    assert torch.allclose(out[..., :8], out_free[..., :8], atol=1e-6)

    # area=0 -> soft behaviour (regression guard for D1)
    pre_soft = VideoPreprocessor(base_ch=8, gate_area=0.0)
    pre_soft.load_state_dict(pre.state_dict())
    out_soft = pre_soft(x, cond, mask=mask)
    expect = x + (1.0 - mask) * (out_free - x)
    assert torch.allclose(out_soft, expect, atol=1e-5)

    print("gate_area self-check passed")


def test_edit_kind_smooth() -> None:
    torch.manual_seed(2)
    pre = VideoPreprocessor(base_ch=8, edit_kind="smooth", gate_area=0.5)
    with torch.no_grad():
        # make the strength map non-trivial: bias the tail toward positive
        for p in pre.unet.tail.parameters():
            p.normal_(0, 0.2)
        pre.unet.tail.bias.fill_(1.0)
    x = torch.rand(2, 3, 4, 16, 16)
    cond = torch.zeros(2, 1)

    out = pre(x, cond)
    assert out.shape == x.shape
    assert float(out.min().detach()) >= -1e-6 and float(out.max().detach()) <= 1.0 + 1e-6

    # D3 invariant: output lies BETWEEN x and blur(x) per pixel (convex blend)
    blur = VideoPreprocessor._gaussian_blur(x)
    lo = torch.minimum(x, blur) - 1e-6
    hi = torch.maximum(x, blur) + 1e-6
    assert torch.all(out >= lo) and torch.all(out <= hi), \
        "smooth edit must stay inside [x, blur(x)] — cannot add detail"

    # gradient flows
    x_g = x.clone().requires_grad_(True)
    out = pre(x_g, cond)
    out.sum().backward()
    assert x_g.grad is not None and torch.isfinite(x_g.grad).all()

    # residual mode unchanged (regression): zero-init tail -> exact identity
    pre_r = VideoPreprocessor(base_ch=8, edit_kind="residual")
    out_r = pre_r(x, cond)
    assert torch.allclose(out_r, x, atol=1e-6)

    # gate + smooth compose: mask=1 still exact identity in smooth mode
    mask1 = torch.ones(2, 1, 4, 16, 16)
    out_g = pre(x, cond, mask=mask1)
    assert torch.allclose(out_g, x, atol=1e-6), "gate must force identity in smooth mode too"

    print("edit_kind smooth self-check passed")


if __name__ == "__main__":
    test_gate()
    test_gate_area()
    test_edit_kind_smooth()
