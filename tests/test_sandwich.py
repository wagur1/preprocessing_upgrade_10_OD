"""Tests for the v8 sandwich model (src/models/sandwich.py)."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.sandwich import SandwichPreprocessor  # noqa: E402
from src.models.upvcm import UPVCMPreprocessor  # noqa: E402


def _clip(b=2, t=8, s=64):
    return torch.rand(b, 3, t, s, s)


def test_identity_at_init_both_halves():
    torch.manual_seed(0)
    m = SandwichPreprocessor()
    x = _clip()
    cond = torch.full((2, 1), 0.6)
    x_pre = m(x, cond)
    assert torch.allclose(x_pre, x, atol=1e-6), "PRE not identity at init"
    x_hat = torch.rand_like(x)  # stand-in decoded clip
    x_tilde = m.post_restore(x_hat, cond)
    assert torch.allclose(x_tilde, x_hat, atol=1e-6), "POST not identity at init"


def test_post_gate_changes_output():
    m = SandwichPreprocessor()
    x_hat = _clip()
    cond = torch.full((2, 1), 0.6)
    with torch.no_grad():
        m.post_net.out_conv.weight.normal_(0, 0.05)
        m.post_strength.fill_(0.5)
        restored = m.post_restore(x_hat, cond)
        assert not torch.allclose(restored, x_hat, atol=1e-6)
        # bypass switch returns decoded clip untouched (decomposition arm)
        m.bypass_post = True
        assert torch.allclose(m.post_restore(x_hat, cond), x_hat)
        m.bypass_post = False


def test_joint_gradients_reach_pre_and_post():
    m = SandwichPreprocessor()
    x = _clip()
    mask = torch.rand(2, 1, 8, 64, 64)
    cond = torch.rand(2, 1)
    x_pre = m(x, cond, mask=mask)
    x_hat = x_pre + torch.randn_like(x_pre) * 0.05   # stand-in codec noise
    x_tilde = m.post_restore(x_hat, cond)
    loss = (x_tilde - x).pow(2).mean() + \
        (m._last_w - m._last_w_target).pow(2).mean()
    loss.backward()
    # PRE half learns (S head + editor)
    assert m.pre.s_net[0].weight.grad is not None
    assert m.pre.editor.out.weight.grad is not None
    # POST half learns
    assert m.post_net.out_conv.weight.grad is not None
    assert m.post_strength.grad is not None


def test_v7_upvcm_checkpoint_loads_into_pre_half():
    """A v7 UP-VCM checkpoint strict-loads into .pre (warm-start path)."""
    torch.manual_seed(0)
    upvcm = UPVCMPreprocessor()
    v7_state = upvcm.state_dict()
    m = SandwichPreprocessor()
    m.load_pre_state(v7_state)
    x = _clip(b=1, t=4, s=32)
    cond = torch.zeros(1, 1)
    with torch.no_grad():
        assert torch.allclose(m.pre(x, cond), upvcm(x, cond), atol=1e-7)


def test_saliency_caches_forwarded():
    m = SandwichPreprocessor()
    x = _clip()
    mask = torch.rand(2, 1, 8, 64, 64)
    m(x, torch.rand(2, 1), mask=mask)
    assert m._last_w is not None and m._last_w.shape == (2, 1, 8, 64, 64)
    assert m._last_w_target is not None


def test_param_count():
    m = SandwichPreprocessor()
    n = sum(p.numel() for p in m.parameters())
    # ~44k PRE + ~577k POST at defaults (restoration needs the width)
    assert 500_000 < n < 800_000, f"unexpected param count {n}"


def test_post_restore_shape_and_range():
    m = SandwichPreprocessor()
    x_hat = _clip(b=2, t=16, s=128)
    out = m.post_restore(x_hat, torch.rand(2, 1))
    assert out.shape == x_hat.shape
    assert out.min() >= 0.0 and out.max() <= 1.0


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all sandwich tests passed")


def test_post_gate_gradients_alive_at_init():
    """Regression: zero-init out_conv x zero-init post_strength = dead saddle
    (the 16-epoch v8 run left POST permanently closed). Noise-init conv keeps
    identity via the gate but the gate gradient must be nonzero."""
    torch.manual_seed(0)
    m = SandwichPreprocessor()
    x_hat = torch.rand(1, 3, 4, 32, 32)
    out = m.post_restore(x_hat, torch.full((1, 1), 0.6))
    out.pow(2).mean().backward()
    assert m.post_strength.grad is not None
    assert m.post_strength.grad.abs() > 0, "POST gate is a dead saddle again"
