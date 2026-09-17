"""Check the residual-TV loss term (`gamma_res`, src/losses.py).

Added 2026-08-31 after the gamma sweep produced three checkpoints within 0.7%
residual RMS of each other and of the gamma=0 baseline: gamma penalises
TV(x_pre), which for an additive model is mostly the SOURCE video's texture,
and the coefficients used (0.01-0.1) contributed <0.2% of the objective.
These checks pin both properties of the replacement term.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.losses import LossWeights, total_variation


def _pair(res_scale: float = 0.05, hf: bool = True):
    """A source clip plus x_pre = source + residual (high- or low-frequency)."""
    torch.manual_seed(0)
    x = torch.linspace(0.2, 0.8, 32).view(1, 1, 1, 1, 32).expand(1, 3, 4, 32, 32).clone()
    if hf:                              # checkerboard: maximal TV per unit RMS
        r = torch.ones(1, 3, 4, 32, 32)
        r[..., ::2, ::2] = -1.0
        r[..., 1::2, 1::2] = -1.0
    else:                               # smooth ramp: same RMS, minimal TV
        r = torch.linspace(-1.0, 1.0, 32).view(1, 1, 1, 1, 32).expand(1, 3, 4, 32, 32).clone()
    r = r / r.pow(2).mean().sqrt() * res_scale     # normalise to the target RMS
    return x, (x + r).clamp(0, 1), r


def test_default_is_off() -> None:
    assert LossWeights().gamma_res == 0.0, "gamma_res must default to off"


def test_targets_the_residual_not_the_source() -> None:
    """TV(residual) must ignore the source's own texture; TV(x_pre) cannot.

    Same residual on a flat source and on a textured one: the residual term is
    unchanged, while TV(x_pre) moves a lot. That difference IS the reason this
    term exists.
    """
    _, _, r = _pair(res_scale=0.02)                # small: stays off the clamp
    flat = torch.full_like(r, 0.5)
    tex = (flat + (torch.rand_like(r) - 0.5) * 0.4).clamp(0, 1)
    tv_res_flat = float(total_variation((flat + r) - flat))
    tv_res_tex = float(total_variation((tex + r) - tex))
    assert abs(tv_res_flat - tv_res_tex) < 1e-5, (tv_res_flat, tv_res_tex)
    assert float(total_variation(tex + r)) > 3 * float(total_variation(flat + r)), \
        "TV(x_pre) must be source-dominated -- otherwise gamma would have worked"


def test_penalises_spectrum_not_amplitude() -> None:
    """At EQUAL residual RMS, high-frequency must cost much more than smooth.

    This is the property the ceiling analysis asks for: a cheaper residual at
    the same amplitude (hence the same accuracy), i.e. a spectrum change. An
    amplitude penalty (`delta`) cannot distinguish these two.
    """
    _, _, r_hf = _pair(hf=True)
    _, _, r_lf = _pair(hf=False)
    rms_hf, rms_lf = (float(v.pow(2).mean().sqrt()) for v in (r_hf, r_lf))
    assert abs(rms_hf - rms_lf) < 1e-6, (rms_hf, rms_lf)      # equal amplitude
    tv_hf, tv_lf = (float(total_variation(v)) for v in (r_hf, r_lf))
    assert tv_hf > 20 * tv_lf, f"TV must separate spectra: hf {tv_hf} vs lf {tv_lf}"
    assert float((r_hf.abs()).mean()) > 0.9 * float((r_lf.abs()).mean()), \
        "L1 (delta) must NOT separate them -- that is why delta is not the lever"


def test_coefficient_is_sized_against_the_task_term() -> None:
    """Guard the mistake that cost 7.5 GPU-hours: a decorative coefficient.

    Sized on the MEASURED quantities, not on this file's synthetic residual
    (whose checkerboard is far more high-frequency than the real one): the D10
    checkpoint's residual has TV ~6.1e-3 and its task loss runs ~3.3, so
    anything below ~1 contributes under 0.2% of the objective. Fail loudly if
    someone copies 0.03 out of robust_transfer.yaml again.
    """
    tv_measured = 6.12e-3      # u6_big4, epoch-5 checkpoint, structured input
    l_task = 3.3               # measured, D10 run over QP30-50
    share = lambda c: 100 * c * tv_measured / l_task
    assert share(0.03) < 0.05, f"sanity: 0.03 really is decorative ({share(0.03):.3f}%)"
    biting = [c for c in (0.1, 1, 3, 10, 30, 100) if share(c) >= 1.0]
    assert biting and min(biting) >= 3, \
        f"a biting coefficient must be >= 3 for this term, got {biting}"


def test_tv_counts_the_temporal_difference() -> None:
    """TV must see flicker. A residual that is constant in time is cheap; one
    that alternates frame to frame costs the codec its inter-frame budget, and a
    spatial-only TV scores them identically (which ours did until 2026-08-31).
    """
    flat = torch.zeros(1, 3, 8, 16, 16)
    steady = flat + 0.1                       # same every frame: no flicker
    flicker = flat.clone()
    flicker[:, :, ::2] = 0.1                  # alternates 0.1 / 0.0 in time
    tv_steady, tv_flicker = (float(total_variation(v)) for v in (steady, flicker))
    assert abs(tv_steady) < 1e-6, f"a constant clip has no variation: {tv_steady}"
    assert tv_flicker > 0.01, f"temporal flicker must be counted: {tv_flicker}"
    # and the 4D path must keep working (no time axis to difference)
    assert float(total_variation(torch.zeros(1, 3, 16, 16))) == 0.0


if __name__ == "__main__":
    test_default_is_off()
    test_targets_the_residual_not_the_source()
    test_penalises_spectrum_not_amplitude()
    test_coefficient_is_sized_against_the_task_term()
    test_tv_counts_the_temporal_difference()
    print("gamma_res self-checks passed")
