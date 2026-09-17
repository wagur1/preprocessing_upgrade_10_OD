"""Tests for the UP-VCM preprocessor (src/models/upvcm.py)."""

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.upvcm import UPVCMPreprocessor  # noqa: E402
from src.models.dino_saliency import dino_energy  # noqa: E402


def _clip(b=2, t=8, s=64):
    return torch.rand(b, 3, t, s, s)


def test_identity_at_init():
    """All three module strengths are zero-init -> exact identity."""
    torch.manual_seed(0)
    pre = UPVCMPreprocessor()
    for strength in (pre.dec_strength, pre.edit_strength, pre.stab_strength):
        assert strength.item() == 0.0
    x = _clip()
    for cond_val in (0.0, 0.5, 1.0):
        cond = torch.full((2, 1), cond_val)
        out = pre(x, cond)
        assert torch.allclose(out, x, atol=1e-6), "not identity at init"


def test_shapes_and_w_range():
    pre = UPVCMPreprocessor()
    x = _clip(b=2, t=8, s=64)
    out = pre(x, torch.rand(2, 1))
    assert out.shape == x.shape
    w = pre._last_w
    assert w.shape == (2, 1, 8, 64, 64)
    assert w.min() >= 0.0 and w.max() <= 1.0


def test_distill_target_cached_and_detached():
    pre = UPVCMPreprocessor(dino_weight=0.0)  # no DINO -> target == mask
    x = _clip()
    mask = torch.rand(2, 1, 8, 64, 64)
    pre(x, torch.rand(2, 1), mask=mask)
    assert pre._last_w_target is not None
    assert not pre._last_w_target.requires_grad
    assert torch.allclose(pre._last_w_target, mask)
    # no mask -> no target (eval path)
    pre(x, torch.rand(2, 1))
    assert pre._last_w_target is None


def test_gradients_reach_s_and_modules():
    pre = UPVCMPreprocessor()
    x = _clip()
    mask = torch.rand(2, 1, 8, 64, 64)
    out = pre(x, torch.rand(2, 1), mask=mask)
    # loss on output (modules) + distill on S
    loss = out.pow(2).mean() + (pre._last_w - pre._last_w_target).pow(2).mean()
    loss.backward()
    assert pre.s_net[0].weight.grad is not None            # S learns
    assert pre.editor.out.weight.grad is not None          # editor learns
    assert pre.dec_strength.grad is not None               # gates learn
    assert pre.edit_strength.grad is not None
    assert pre.stab_strength.grad is not None
    # ...but "not None" is not enough: a zero tensor also passes that check and
    # is exactly the signature of the dead M2 saddle, so require non-zero.
    assert pre.s_net[0].weight.grad.abs().max() > 0
    assert pre.dec_strength.grad.abs().max() > 0
    assert pre.stab_strength.grad.abs().max() > 0
    assert pre.edit_strength.grad.abs().max() > 0, "M2 gate is stuck at zero"


def test_editor_is_not_born_dead():
    """Regression for the M2 dead saddle (fixed 2026-09-16).

    A zero-init output conv combined with a zero-init ``edit_strength`` gate
    makes BOTH factors exactly zero, so both gradients are exactly zero and the
    editor never switches on. Every checkpoint trained before the fix has
    ``edit_strength == 0.0`` and ``|W_out| == 0`` — the ROI editor has never
    been active in any recorded result. The fix (small noise on the output
    conv, gate still zero) keeps identity-at-init but makes the gate alive at
    step 0 and the whole subnet alive from step 1.
    """
    torch.manual_seed(0)
    pre = UPVCMPreprocessor()
    x = _clip()
    cond = torch.rand(2, 1)

    # identity at init must survive the repair
    with torch.no_grad():
        assert torch.allclose(pre(x, cond), x, atol=1e-6)

    # step 0: the gate sees a non-zero gradient
    pre(x, cond).pow(2).mean().backward()
    assert pre.edit_strength.grad.abs().max() > 0, "gate dead at init"

    # step 1 onward: the subnet (whose gradient carries the gate as a factor)
    # comes alive as soon as the gate has moved off zero
    opt = torch.optim.SGD(pre.parameters(), lr=1e-3)
    opt.step()
    opt.zero_grad()
    pre(x, cond).pow(2).mean().backward()
    assert pre.editor.out.weight.grad.abs().max() > 0, "editor subnet dead"
    assert pre.edit_strength.grad.abs().max() > 0


def test_module_gates_actually_change_output():
    """Manually opening each gate must change the output (mechanism wiring)."""
    pre = UPVCMPreprocessor()
    x = _clip()
    cond = torch.full((2, 1), 0.6)
    with torch.no_grad():
        base = pre(x, cond)
        # M1: decimate background
        pre.dec_strength.fill_(0.5)
        dec = pre(x, cond)
        assert not torch.allclose(dec, base, atol=1e-6)
        pre.dec_strength.zero_()
        # M2: ROI edit
        with torch.no_grad():
            pre.editor.out.weight.normal_(0, 0.05)  # non-zero editor output
        pre.edit_strength.fill_(1.0)
        edit = pre(x, cond)
        assert not torch.allclose(edit, base, atol=1e-6)
        pre.edit_strength.zero_()
        pre.editor.out.weight.zero_()
        # M3: stabilise static background
        pre.stab_strength.fill_(1.0)
        stab = pre(x, cond)
        assert not torch.allclose(stab, base, atol=1e-6)


def test_m3_only_touches_static_background():
    """M3 with full strength: static background == previous frame, moving ROI untouched."""
    pre = UPVCMPreprocessor()
    x = _clip(b=1, t=8, s=32)
    # force W=1 (all-important) -> M3 gate = stab*(1-1)*static = 0 -> no change
    with torch.no_grad():
        pre.s_net[4].weight.zero_()
        pre.s_net[4].bias.fill_(10.0)  # sigmoid(10) ~ 1
    with torch.no_grad():
        pre.stab_strength.fill_(1.0)
        out = pre(x, torch.zeros(1, 1))
    # sigmoid never reaches exactly 1; the gate residual is ~(1-W)*|ref-cur|
    # ~ 5e-5 * 0.7 ~ 3e-6. Without the W-gate the diff would be ~0.25.
    assert torch.allclose(out, x, atol=1e-4), "M3 must not touch W=1 regions"


def test_dino_energy_with_stub_model():
    """dino_energy works with any forward_features-compatible stub."""

    class StubDino(nn.Module):
        def forward_features(self, fr):
            n = fr.shape[0]
            g = fr.shape[-1] // 14
            return {"x_norm_patchtokens": torch.rand(n, g * g, 8, device=fr.device)}

    x = _clip(b=2, t=16, s=64)
    e = dino_energy(x, StubDino())
    assert e.shape == (2, 1, 16, 64, 64)
    assert e.min() >= 0.0 and e.max() <= 1.0


def test_param_count_recorded():
    pre = UPVCMPreprocessor()
    n = sum(p.numel() for p in pre.parameters())
    # ~44k at (s_ch=16, editor_ch=24); guard against accidental blow-up
    assert 30_000 < n < 100_000, f"unexpected param count {n}"
    assert math.isfinite(n)


def test_w_budget_prevents_collapse():
    """With ``w_budget`` the importance map cannot collapse to zero.

    Every trained checkpoint has W ~ 1e-9 (logits ~ -60), which disables the
    spatial selectivity of M1/M2/M3 and makes the M2 gate's gradient ~1e-14.
    The budget normalisation must restore a usable map even from that exact
    state — including full saturation, where the ratio form returns a uniform
    map at the budget instead of decaying to zero.
    """
    torch.manual_seed(0)
    pre = UPVCMPreprocessor(w_budget=0.25)
    x = _clip()
    cond = torch.rand(2, 1)

    # reproduce the trained collapse: constant logits at -60
    with torch.no_grad():
        pre.s_net[4].weight.zero_()
        pre.s_net[4].bias.fill_(-60.0)
    pre(x, cond)
    w = pre._last_w
    assert abs(w.mean().item() - 0.25) < 1e-3, f"budget not enforced: {w.mean().item():.3e}"

    # the distill target is normalised into the same space, so rho distils the
    # SHAPE of the saliency map rather than its absolute scale
    mask = torch.rand(2, 1, 8, 64, 64)
    pre(x, cond, mask=mask)
    assert pre._last_w_target is not None
    assert 0.1 < pre._last_w_target.mean().item() < 0.5

    # legacy behaviour (budget 0) must be untouched
    torch.manual_seed(0)
    legacy = UPVCMPreprocessor()
    with torch.no_grad():
        legacy.s_net[4].weight.zero_()
        legacy.s_net[4].bias.fill_(-60.0)
    legacy(x, cond)
    assert legacy._last_w.mean().item() < 1e-6


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all upvcm tests passed")
