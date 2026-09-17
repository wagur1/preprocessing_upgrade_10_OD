"""Tests for the D8 learned-rate codec (Laplacian factorized prior)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.entropy_codec import LearnedRateCodec


def test_expected_bits_shape_and_finite() -> None:
    torch.manual_seed(0)
    cod = LearnedRateCodec(qualities=(1, 3, 8), block=8, colorspace="yuv420")
    # _expected_bits: y [..., n_pos] (last dim = positions) -> bits same shape
    y = torch.randint(-6, 7, (2, 3, 16, 64))
    bits = cod._expected_bits(y)
    assert bits.shape == y.shape
    assert torch.isfinite(bits).all()
    assert (bits > 0).all(), "bits must be positive"


def test_zero_init_gives_near_uniform_bits() -> None:
    cod = LearnedRateCodec(qualities=(1, 3, 8), block=8, colorspace="yuv420")
    y = torch.zeros(1, 1, 64)
    b0 = cod._expected_bits(y).sum()
    y_far = torch.full((1, 1, 64), 8)
    b_far = cod._expected_bits(y_far).sum()
    assert b_far > b0, "larger magnitude symbols must cost more bits"


def test_learned_prior_can_lower_bits_for_its_mode() -> None:
    cod = LearnedRateCodec(qualities=(1,), block=8, colorspace="yuv420")
    with torch.no_grad():
        cod.rate_params["mu"][:, 0] = 5.0
        cod.rate_params["w"][:, 0] = 10.0  # ~all mass on component 0
    b5 = cod._expected_bits(torch.full((1, 64), 5.0)).mean()
    bm5 = cod._expected_bits(torch.full((1, 64), -5.0)).mean()
    assert b5 < bm5, "prior centred at 5 must rate symbol 5 cheaper"


def test_quant_rate_learned_matches_parent_interface() -> None:
    """The override must accept the PIPELINE layout [N, C*bs*bs, nH, nW]."""
    torch.manual_seed(1)
    cod = LearnedRateCodec(qualities=(1, 3, 8), block=8, colorspace="yuv420")
    bs = cod.block
    # luma-like: [2, 64, 4, 4]; chroma-like: [2, 128, 2, 2]
    for shape in ((2, bs * bs, 4, 4), (2, 2 * bs * bs, 2, 2)):
        coeff = torch.randn(*shape)
        y_hat, bits = cod._quant_rate_learned(coeff, step=0.1, training=False)
        assert y_hat.shape == coeff.shape
        assert torch.isfinite(bits) and float(bits.detach()) > 0
        assert torch.allclose(y_hat, torch.round(coeff / 0.1), atol=1e-6)
        y_hat_t, _ = cod._quant_rate_learned(coeff, step=0.1, training=True)
        assert not torch.allclose(y_hat_t, torch.round(coeff / 0.1))


def test_gradients_flow_to_rate_params() -> None:
    cod = LearnedRateCodec(qualities=(1,), block=8, colorspace="yuv420")
    bs = cod.block
    coeff = torch.randn(2, bs * bs, 4, 4)
    y_hat, bits = cod._quant_rate_learned(coeff, step=0.1, training=True)
    bits.backward()
    for k, p in cod.rate_params.items():
        assert p.grad is not None and torch.isfinite(p.grad).all(), f"{k} no grad"
    assert cod.rate_params["w"].grad.abs().sum() > 0


def test_full_forward_integration() -> None:
    """Engine-path forward: [B,C,T,H,W] -> (x_hat, bpp), grads to input+prior."""
    torch.manual_seed(2)
    cod = LearnedRateCodec(qualities=(1, 3, 8), block=8, colorspace="yuv420")
    x = torch.rand(2, 3, 4, 32, 32)
    x_hat, bpp = cod(x, 3)
    assert x_hat.shape == x.shape
    assert torch.isfinite(bpp) and float(bpp.detach()) > 0
    xg = x.clone().requires_grad_(True)
    xh, b = cod(xg, 3)
    (xh.mean() + b).backward()
    assert xg.grad is not None and torch.isfinite(xg.grad).all()
    for k, p in cod.rate_params.items():
        assert p.grad is not None and torch.isfinite(p.grad).all(), k


if __name__ == "__main__":
    test_expected_bits_shape_and_finite()
    test_zero_init_gives_near_uniform_bits()
    test_learned_prior_can_lower_bits_for_its_mode()
    test_quant_rate_learned_matches_parent_interface()
    test_gradients_flow_to_rate_params()
    test_full_forward_integration()
    print("entropy_codec self-checks passed")
