"""Checks for the QP-conditioned additive preprocessor (round b, needs torch).

pytest-collectable AND runnable standalone -- same pattern as
tests/test_additive.py. Mirrors its checks where the two models agree and
adds the conditioning-specific ones (identity-at-init with ANY cond,
conditionability of a live residual, strict-load of the unconditioned tree).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.additive import AdditivePreprocessor
from src.models.additive_cond import AdditiveCondPreprocessor

BEST_PT = Path(r"C:\Users\Wagur1\Downloads\preprocessing-final-results\checkpoints\best.pt")


def _trained_like(scale: float = 0.02) -> AdditiveCondPreprocessor:
    """A model with a LIVE residual.

    to_rgb and film[2] are both zero-init, so perturbing to_rgb (the residual
    output projection) is the minimal way to get a non-zero edit; small scale
    keeps the edit away from the [0,1] clamp so strength algebra stays exact.
    """
    pre = AdditiveCondPreprocessor()
    with torch.no_grad():
        pre.to_rgb.weight.normal_(0.0, scale)
        pre.to_rgb.bias.normal_(0.0, scale)
    return pre


def test_param_count() -> None:
    """(a) 10,371 = 9,795 Zhao tree + 576 FiLM (Linear(1,16)+Linear(16,32)+biases)."""
    n = sum(p.numel() for p in AdditiveCondPreprocessor().parameters())
    assert n == 10371, f"param count {n} != 10371"


def test_strict_loads_unconditioned_base_keys() -> None:
    """(b) the base tree stays best.pt-load-compatible: every key of the
    unconditioned 9,795-param checkpoint must exist with the same shape here
    (strict=True fails only on the 6 missing FiLM keys, nothing else)."""
    pre = AdditiveCondPreprocessor()
    own = pre.state_dict()
    if BEST_PT.exists():
        sd = torch.load(BEST_PT, map_location="cpu", weights_only=False)["preprocessor"]
        missing, unexpected = pre.load_state_dict(sd, strict=False)
        assert not unexpected, f"unexpected keys: {unexpected}"
        assert set(missing) == {"film.0.weight", "film.0.bias",
                                "film.2.weight", "film.2.bias"}, f"missing: {missing}"
    else:
        base = AdditivePreprocessor().state_dict()
        for k, v in base.items():
            assert k in own and own[k].shape == v.shape, f"base key drifted: {k}"


def test_identity_at_init_for_any_cond() -> None:
    """(c) zero-init FiLM + zero-init to_rgb: an UNTRAINED model is exactly
    identity regardless of the condition (cond must not leak through zeros)."""
    x = torch.rand(1, 3, 16, 64, 64)
    with torch.no_grad():
        y0 = AdditiveCondPreprocessor()(x, None)
        y1 = AdditiveCondPreprocessor()(x, torch.ones(1, 1))
    assert torch.equal(y0, x), "untrained model (cond=None) must be exact identity"
    assert torch.equal(y1, x), "untrained model (cond=1) must be exact identity"


def test_cond_changes_live_residual_monotonically_somewhere() -> None:
    """(d) conditionability: a LIVE FiLM must change the edit across conds.

    Zero-init FiLM never changes anything, so write a small perturbation into
    film[2] (the gamma/beta head) and require that SOME condition pair changes
    the output. This pins the wiring (cond -> FiLM -> fused -> residual), not
    the direction (that is training's job)."""
    pre = _trained_like()
    with torch.no_grad():
        pre.film[2].weight.normal_(0.0, 0.1)
        pre.film[2].bias.normal_(0.0, 0.1)
    x = torch.rand(2, 3, 8, 48, 48)
    with torch.no_grad():
        y_lo = pre(x, torch.zeros(2, 1))     # QP20-normalised (light)
        y_hi = pre(x, torch.ones(2, 1))      # QP51-normalised (heavy)
    assert not torch.equal(y_lo, y_hi), "cond must reach the residual when FiLM is live"


def test_unconditioned_equals_light_cond() -> None:
    """(e) cond=None and cond=0 must agree exactly (same default semantics)."""
    pre = _trained_like()
    x = torch.rand(1, 3, 8, 48, 48)
    with torch.no_grad():
        y0 = pre(x, None)
        yz = pre(x, torch.zeros(1, 1))
    assert torch.equal(y0, yz), "cond=None must default to zeros"


def test_mask_is_ignored() -> None:
    """(f) spatial gating is round b2: mask must not change the output."""
    pre = _trained_like()
    x = torch.rand(1, 3, 8, 48, 48)
    with torch.no_grad():
        y0 = pre(x)
        y1 = pre(x, None, torch.ones(1, 1, 8, 48, 48))
    assert torch.equal(y0, y1), "mask must be ignored in round (b)"


def test_shape_range_resolution_agnostic() -> None:
    """(g) shape / range / resolution handling mirrors the unconditioned model."""
    pre = _trained_like()
    for t, hw in ((16, 128), (16, 224), (4, 96)):
        x = torch.rand(2, 3, t, hw, hw)
        y = pre(x, torch.full((2, 1), 0.5))
        assert y.shape == x.shape, (y.shape, x.shape)
        assert y.min() >= 0.0 and y.max() <= 1.0, "output must be clamped to [0,1]"


def test_gradients_flow_to_every_param_including_film() -> None:
    """(h) gradient reaches every parameter -- FiLM included (the round's point)."""
    pre = AdditiveCondPreprocessor()
    x = torch.rand(2, 3, 16, 64, 64)
    pre(x, torch.full((2, 1), 0.5)).mean().backward()
    for name, p in pre.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), f"no/NaN grad: {name}"
    # and through the condition path when the residual is live:
    pre2 = _trained_like()
    with torch.no_grad():
        pre2.film[2].weight.normal_(0.0, 0.1)
    cond = torch.zeros(1, 1, requires_grad=True)
    pre2(torch.rand(1, 3, 8, 48, 48), cond).mean().backward()
    assert cond.grad is not None and float(cond.grad.abs()) > 0, \
        "gradient must flow INTO cond (the FiLM input) when film is live"


def test_strength_is_the_operating_point() -> None:
    """(i) 0 -> exact identity; s -> x + s*(y1 - x), at a FIXED cond."""
    pre = _trained_like()
    x = torch.rand(1, 3, 16, 64, 64) * 0.4 + 0.3
    cond = torch.full((1, 1), 0.5)
    with torch.no_grad():
        y1 = pre(x, cond)
        pre.strength = 0.5
        yh = pre(x, cond)
        pre.strength = 0.0
        y0 = pre(x, cond)
    assert not torch.allclose(y1, x, atol=1e-6), "fixture residual must be live"
    assert torch.equal(y0, x), "strength=0 must be exact identity"
    assert torch.allclose(yh, x + 0.5 * (y1 - x), atol=1e-6), \
        "strength=s must scale the residual linearly"


if __name__ == "__main__":
    test_param_count()
    test_strict_loads_unconditioned_base_keys()
    test_identity_at_init_for_any_cond()
    test_cond_changes_live_residual_monotonically_somewhere()
    test_unconditioned_equals_light_cond()
    test_mask_is_ignored()
    test_shape_range_resolution_agnostic()
    test_gradients_flow_to_every_param_including_film()
    test_strength_is_the_operating_point()
    print("additive_cond self-check passed")
