"""Checks for the RPP-style adaptive DCT rate proxy (`kappa`, src/losses.py).

The point of this term is that it is SELECTIVE where total variation is not:
low frequencies are untouched, and inside the high band only the sub-mean
coefficients are pushed to zero. A `gamma_res`-style amplitude penalty has
neither property, which is why a 10x sweep of it only shrank the residual
(2026-08-31, three runs, 7.5 GPU-hours).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.losses import LossWeights, adaptive_dct_loss, total_variation


def _clip(hw: int = 32, t: int = 2) -> torch.Tensor:
    return torch.zeros(1, 3, t, hw, hw)


def test_default_is_off() -> None:
    assert LossWeights().kappa == 0.0, "kappa must default to off"


def test_low_frequency_is_untouched() -> None:
    """A pure low-frequency ramp has no penalised energy at all."""
    ramp = torch.linspace(0.2, 0.8, 32).view(1, 1, 1, 1, 32).expand(1, 3, 2, 32, 32)
    assert float(adaptive_dct_loss(ramp.contiguous(), block=8)) < 1e-6


def test_strong_high_frequency_is_protected() -> None:
    """A single strong edge must cost far less than broadband weak texture.

    This is the property `total_variation` cannot express: scaled to the SAME TV,
    one sharp edge (few strong coefficients, all above the block mean) must be
    much cheaper than low-amplitude noise spread over the whole high band.
    """
    torch.manual_seed(0)
    edge = _clip()
    edge[..., 16:] = 0.5                              # one step: strong, sparse HF
    noise = torch.rand_like(edge) - 0.5               # weak, broadband HF
    noise = noise / noise.std() * 0.02
    # normalise both to equal total variation so TV cannot separate them
    noise = noise * (float(total_variation(edge)) / float(total_variation(noise)))
    tv_e, tv_n = float(total_variation(edge)), float(total_variation(noise))
    assert abs(tv_e - tv_n) / tv_e < 0.05, (tv_e, tv_n)
    d_e = float(adaptive_dct_loss(edge, block=8))
    d_n = float(adaptive_dct_loss(noise, block=8))
    assert d_n > 3 * d_e, f"weak broadband HF must cost more: edge {d_e} vs noise {d_n}"


def test_concentrated_energy_is_cheaper_than_spread_energy() -> None:
    """At EQUAL high-frequency energy, few strong coefficients must cost less than
    many weak ones.

    This is the property that matters for bits: a run of near-zero coefficients is
    what entropy coding is cheap on, and it is exactly what an amplitude penalty
    cannot express. Note the threshold is the mean of the selected band, so simply
    scaling one coefficient up is NOT expected to lower the loss (it raises the mean
    and pulls its neighbours under) -- concentration, not magnitude, is the axis.
    """
    xs = torch.arange(32).float()
    def sines(ks, amp):
        v = _clip()
        for k in ks:
            v = v + amp * torch.cos(k * math.pi / 8 * (xs + 0.5)).view(1, 1, 1, 1, 32)
        return v
    # equal L2 energy: one component at amp a, vs 4 components at amp a/2
    conc = sines([7], 0.08)
    spread = sines([5, 6, 7], 0.08 / math.sqrt(3))
    e_c = float((conc - _clip()).pow(2).mean())
    e_s = float((spread - _clip()).pow(2).mean())
    assert abs(e_c - e_s) / e_c < 0.15, (e_c, e_s)          # matched energy
    d_c = float(adaptive_dct_loss(conc, block=8))
    d_s = float(adaptive_dct_loss(spread, block=8))
    assert d_s > d_c, f"spread HF energy must cost more than concentrated: {d_c} vs {d_s}"


def test_gradient_flows_and_is_finite() -> None:
    x = (_clip() + torch.rand(1, 3, 2, 32, 32) * 0.3).requires_grad_(True)
    (adaptive_dct_loss(x, block=8) + adaptive_dct_loss(x, block=16)).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert float(x.grad.abs().sum()) > 0, "the term must actually push on the pixels"


def test_coefficient_is_sized_against_the_task_term() -> None:
    """Same guard as gamma_res: refuse a decorative coefficient.

    MEASURED on a structured 128x128 clip: L_dct ~ 0.058 against L_task ~ 3.3, so
    the biting range is kappa 1-10 (1.8%-17.6%). Anything <= 0.1 is decoration.
    """
    l_dct, l_task = 0.058, 3.3
    share = lambda c: 100 * c * l_dct / l_task
    assert share(0.1) < 0.5, f"0.1 is decorative ({share(0.1):.2f}%)"
    assert 1.0 <= share(1) <= 3.0, f"kappa=1 should just bite ({share(1):.2f}%)"
    assert share(10) > 10.0, f"kappa=10 should dominate visibly ({share(10):.2f}%)"


if __name__ == "__main__":
    test_default_is_off()
    test_low_frequency_is_untouched()
    test_strong_high_frequency_is_protected()
    test_concentrated_energy_is_cheaper_than_spread_energy()
    test_gradient_flows_and_is_finite()
    test_coefficient_is_sized_against_the_task_term()
    print("adaptive DCT self-checks passed")
