"""Checks for the additive two-branch preprocessor (needs torch).

pytest-collectable (test_* functions) AND runnable standalone -- same pattern as
tests/test_entropy_codec.py. It used to expose only main(), so pytest collected
ZERO tests from this file and the additive model was silently uncovered.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.additive import AdditivePreprocessor

BEST_PT = Path(r"C:\Users\Wagur1\Downloads\preprocessing-final-results\checkpoints\best.pt")


def _trained_like(scale: float = 0.02) -> AdditivePreprocessor:
    """A model with a LIVE residual.

    to_rgb is zero-init, so multiplying its weights is a no-op -- the only way
    to get a non-zero residual out of an untrained instance is to write new
    values in. Small scale keeps the edit away from the [0,1] clamp so the
    strength algebra in test_strength_is_the_operating_point stays exact.
    """
    pre = AdditivePreprocessor()
    with torch.no_grad():
        pre.to_rgb.weight.normal_(0.0, scale)
        pre.to_rgb.bias.normal_(0.0, scale)
    return pre


def test_param_count() -> None:
    """(a) exact parameter count: the additive run's checkpoint has 9,795."""
    n = sum(p.numel() for p in AdditivePreprocessor().parameters())
    assert n == 9795, f"param count {n} != 9795"


def test_strict_loads_best_pt() -> None:
    """(b) structural pin: our module tree must strict-load the real best.pt."""
    if not BEST_PT.exists():
        print("[test_additive] best.pt not found locally -- skipping strict-load pin")
        return
    sd = torch.load(BEST_PT, map_location="cpu", weights_only=False)["preprocessor"]
    AdditivePreprocessor().load_state_dict(sd, strict=True)
    print("[test_additive] strict-loaded best.pt")


def test_shape_range_and_resolution_agnostic() -> None:
    """(c) shape / range / resolution / clip-length handling."""
    pre = _trained_like()
    for t, hw in ((16, 128), (16, 224), (4, 96)):
        x = torch.rand(2, 3, t, hw, hw)
        y = pre(x)
        assert y.shape == x.shape, (y.shape, x.shape)
        assert y.min() >= 0.0 and y.max() <= 1.0, "output must be clamped to [0,1]"


def test_unconditioned_and_ungated() -> None:
    """(d) cond/mask must not change the output."""
    pre = _trained_like()
    x = torch.rand(2, 3, 16, 64, 64)
    y0 = pre(x)
    y1 = pre(x, torch.zeros(2, 1))
    y2 = pre(x, torch.zeros(2, 1), torch.ones(2, 1, 16, 64, 64))
    assert torch.equal(y0, y1) and torch.equal(y0, y2), "cond/mask must be ignored"


def test_gradients_flow_to_every_branch() -> None:
    """(e) gradient reaches every parameter of every branch."""
    pre = AdditivePreprocessor()
    pre(torch.rand(2, 3, 16, 64, 64)).mean().backward()
    for name, p in pre.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), f"no/NaN grad: {name}"


def test_strength_is_the_operating_point() -> None:
    """(f) 0 -> exact identity; s -> x + s*(y1 - x), on a LIVE residual."""
    pre = _trained_like()
    x = torch.rand(1, 3, 16, 64, 64) * 0.4 + 0.3   # away from the clamp
    with torch.no_grad():
        y1 = pre(x)
        pre.strength = 0.5
        yh = pre(x)
        pre.strength = 0.0
        y0 = pre(x)
    assert not torch.allclose(y1, x, atol=1e-6), "fixture residual must be live"
    assert torch.equal(y0, x), "strength=0 must be exact identity"
    assert torch.allclose(yh, x + 0.5 * (y1 - x), atol=1e-6), \
        "strength=s must scale the residual linearly"


def test_identity_at_init() -> None:
    """(g) to_rgb is zero-init, so an UNTRAINED model is exactly identity.

    Inverted from the original "random-init residual must be non-zero": that
    predated the zero-init and had been failing ever since. Exact identity at
    init is the property the additive design actually wants -- an untrained
    preprocessor must not perturb the video at all.
    """
    x = torch.rand(1, 3, 16, 64, 64)
    with torch.no_grad():
        y = AdditivePreprocessor()(x)
    assert torch.equal(y, x), "untrained additive model must be exact identity"


def test_temporal_context_is_causal_and_per_frame() -> None:
    """The temporal branch must be a CAUSAL per-frame window, not a clip constant.

    Until 2026-08-31 the branch stacked the clip's centre 8 frames once and
    broadcast the result to every frame (`repeat_interleave`), so it contributed
    a per-clip constant and the residual could not follow motion. Perturbing one
    input frame pins the corrected behaviour exactly: window i spans frames
    [i-7 .. i], so a change at frame 8 must leave frames 0-7 bit-identical and
    move every frame from 8 to 15.
    """
    pre = _trained_like()
    x = torch.rand(2, 3, 16, 48, 48)
    x2 = x.clone()
    x2[:, :, 8] = torch.rand_like(x2[:, :, 8])
    with torch.no_grad():
        d = (pre(x) - pre(x2)).abs().mean(dim=(0, 1, 3, 4))   # keep the time axis
    assert float(d[:8].max()) == 0.0, f"causality violated: {d[:8].tolist()}"
    assert float(d[8:].min()) > 1e-7, f"no temporal reach: {d[8:].tolist()}"


if __name__ == "__main__":
    test_param_count()
    test_strict_loads_best_pt()
    test_shape_range_and_resolution_agnostic()
    test_unconditioned_and_ungated()
    test_gradients_flow_to_every_branch()
    test_strength_is_the_operating_point()
    test_identity_at_init()
    test_temporal_context_is_causal_and_per_frame()
    print("additive self-check passed")
