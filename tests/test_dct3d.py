"""Checks for the 3-D (spatio-temporal) adaptive DCT term (`kappa_t`, src/losses.py).

The 2-D term flattens time into the batch, so it is structurally blind to temporal
frequency. The escape-route framing that motivated this term was RETRACTED on
2026-09-03 (see the losses.py docstring): TVt/RMS is an L1/L2 shape ratio, not an
energy share, and the residual's temporal-AC energy share actually falls with kappa.
What survives is narrower and still worth a term: the model adds temporal-AC energy
mostly OUTSIDE this mask's band (+127.6% spatial-LF vs +43.2% spatial-HF at kappa=10),
so these tests pin the transform's correctness, not the v1 band's usefulness.

Note on the spec these tests implement: "alternating-sign residual -> penalty
maximal" is only true for residuals that also carry SPATIAL high frequency. A
spatially flat field that flips sign every frame is spatial-DC, and the v1 band mask
deliberately leaves that cell unpenalised (the documented scope hole). The tests
below pin the corrected property.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.losses import LossWeights, adaptive_dct3d_loss, adaptive_dct_loss
from src.models.virtual_codec import _dct_basis


def _checker(hw: int = 32, t: int = 16, amp: float = 0.05) -> torch.Tensor:
    """Spatial checkerboard: maximal spatial HF per unit amplitude."""
    v = torch.ones(1, 3, t, hw, hw)
    v[..., ::2, ::2] = -1.0
    v[..., 1::2, 1::2] = -1.0
    return v * amp


def test_default_is_off() -> None:
    assert LossWeights().kappa_t == 0.0, "kappa_t must default to off"


def test_basis_is_orthonormal() -> None:
    """Mixing a 4-long temporal axis with 8-long spatial axes needs D @ D.T == I.

    The unnormalised helper the 2-D term uses would scale the two axes differently,
    making |F| incomparable across them and skewing the adaptive threshold.
    """
    for n in (4, 8):
        D = _dct_basis(n)
        assert torch.allclose(D @ D.T, torch.eye(n), atol=1e-6), f"n={n} not orthonormal"


def test_constant_in_time_costs_nothing() -> None:
    """Temporal DC is free: inter prediction codes static content almost for nothing."""
    flat = torch.full((1, 3, 16, 32, 32), 0.3)
    assert float(adaptive_dct3d_loss(flat)) < 1e-9
    assert float(adaptive_dct3d_loss(_checker())) < 1e-9, \
        "static texture is a strong concentrated coefficient -> protected"


def test_flickering_texture_is_penalised() -> None:
    """The target: spatial HF that also alternates in time."""
    flick = _checker()
    flick[:, :, 1::2] *= -1.0
    assert float(adaptive_dct3d_loss(flick)) > 1e-5


def test_the_2d_term_is_blind_to_it() -> None:
    """This is the escape route, shown on ONE input pair rather than inferred from TVt.

    Stated as an INVARIANCE rather than an absolute threshold, because that is the
    real property: flipping the sign of every other frame leaves each frame's own
    spectrum untouched, so a per-frame transform returns the SAME number for static
    and flickering texture. It cannot charge for flicker even in principle. The 3-D
    term separates them by orders of magnitude.
    """
    static = _checker()
    flick = _checker()
    flick[:, :, 1::2] *= -1.0
    d2s, d2f = float(adaptive_dct_loss(static, block=8)), float(adaptive_dct_loss(flick, block=8))
    d3s, d3f = float(adaptive_dct3d_loss(static)), float(adaptive_dct3d_loss(flick))
    assert abs(d2s - d2f) <= 1e-9 + 1e-6 * max(d2s, 1e-12), \
        f"per-frame term must not distinguish static from flicker: {d2s} vs {d2f}"
    assert d3f > 1e-5 and d3s < 1e-9, f"3-D term must separate them: {d3s} vs {d3f}"


def test_spatially_flat_flicker_is_the_documented_scope_hole() -> None:
    """v1 leaves spatial-LF temporal-AC unpenalised ON PURPOSE. Pin it, so widening
    the mask later is a deliberate decision and not an accident."""
    flat = torch.zeros(1, 3, 16, 32, 32)
    flat[:, :, ::2] = 0.1
    assert float(adaptive_dct3d_loss(flat)) < 1e-9


def test_magnitude_invariant_under_time_reversal() -> None:
    """DCT-II bases are even, so |F| must not depend on time direction.

    A wrong axis order in the einsum would break this while leaving the other tests
    passing, which is exactly the bug class this catches.
    """
    torch.manual_seed(0)
    r = (torch.rand(1, 3, 16, 32, 32) - 0.5) * 0.1
    a, b = float(adaptive_dct3d_loss(r)), float(adaptive_dct3d_loss(r.flip(2)))
    assert abs(a - b) < 1e-6, (a, b)


def test_gradient_flows_and_is_finite() -> None:
    x = (torch.rand(1, 3, 16, 32, 32) * 0.3).requires_grad_(True)
    adaptive_dct3d_loss(x).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert float(x.grad.abs().sum()) > 0, "the term must push on the pixels"


def test_short_clip_and_odd_sizes_do_not_crash() -> None:
    """Crops to whole blocks; a clip shorter than one temporal block costs zero."""
    assert float(adaptive_dct3d_loss(torch.rand(1, 3, 2, 32, 32))) == 0.0
    assert torch.isfinite(adaptive_dct3d_loss(torch.rand(1, 3, 17, 33, 31)))


if __name__ == "__main__":
    for fn in (test_default_is_off, test_basis_is_orthonormal,
               test_constant_in_time_costs_nothing, test_flickering_texture_is_penalised,
               test_the_2d_term_is_blind_to_it,
               test_spatially_flat_flicker_is_the_documented_scope_hole,
               test_magnitude_invariant_under_time_reversal,
               test_gradient_flows_and_is_finite,
               test_short_clip_and_odd_sizes_do_not_crash):
        fn()
    print("3-D DCT self-checks passed")
