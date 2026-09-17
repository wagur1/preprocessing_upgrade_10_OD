"""Tests for the temporal POST branch (src/models/sandwich.py, `temporal=True`).

Codec artifacts are temporally correlated, but the restorer has always seen one
frame at a time. The temporal branch adds the two neighbours through a
zero-initialised additive input path, which has three properties worth pinning:
it is behaviour-preserving at init (so a warm start from the record checkpoint
is exact), it is never dead (it adds into a live trunk, unlike the gate x module
products elsewhere in this repo), and it actually changes the output once its
weights move.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.sandwich import SandwichPreprocessor, _PostUNet  # noqa: E402


def _clip(b=2, t=6, s=32):
    return torch.rand(b, 3, t, s, s)


def _cond(b=2):
    return torch.full((b, 1), 0.5)


def test_temporal_matches_perframe_at_init():
    """Zero-init branch + identical weights => identical restoration."""
    torch.manual_seed(0)
    plain = _PostUNet(base=8, temporal=False)
    torch.manual_seed(0)
    temp = _PostUNet(base=8, temporal=True)
    # give the temporal net the plain net's weights (new keys stay at init)
    res = temp.load_state_dict(plain.state_dict(), strict=False)
    assert set(res.missing_keys) == {"in_conv_t.weight", "in_conv_t.bias"}

    x = _clip()
    cond = _cond()
    with torch.no_grad():
        a = plain(x.permute(0, 2, 1, 3, 4).reshape(12, 3, 32, 32), cond.repeat_interleave(6, 0))
        prev = torch.cat([x[:, :, :1], x[:, :, :-1]], dim=2)
        nxt = torch.cat([x[:, :, 1:], x[:, :, -1:]], dim=2)
        nb = torch.cat([prev, nxt], dim=1).permute(0, 2, 1, 3, 4).reshape(12, 6, 32, 32)
        b_ = temp(x.permute(0, 2, 1, 3, 4).reshape(12, 3, 32, 32),
                  cond.repeat_interleave(6, 0), nb)
    assert torch.allclose(a, b_, atol=1e-6), "temporal branch is not identity at init"


def test_temporal_branch_is_never_dead():
    """Unlike a gate x module product, an additive branch gets gradient at step 0."""
    torch.manual_seed(0)
    net = _PostUNet(base=8, temporal=True)
    x = _clip()
    prev = torch.cat([x[:, :, :1], x[:, :, :-1]], dim=2)
    nxt = torch.cat([x[:, :, 1:], x[:, :, -1:]], dim=2)
    nb = torch.cat([prev, nxt], dim=1).permute(0, 2, 1, 3, 4).reshape(12, 6, 32, 32)
    out = net(x.permute(0, 2, 1, 3, 4).reshape(12, 3, 32, 32), _cond().repeat_interleave(6, 0), nb)
    out.pow(2).mean().backward()
    assert net.in_conv_t.weight.grad is not None
    assert net.in_conv_t.weight.grad.abs().max() > 0, "temporal branch born dead"


def test_temporal_branch_changes_output_once_open():
    net = _PostUNet(base=8, temporal=True)
    x = _clip()
    cond = _cond().repeat_interleave(6, 0)
    prev = torch.cat([x[:, :, :1], x[:, :, :-1]], dim=2)
    nxt = torch.cat([x[:, :, 1:], x[:, :, -1:]], dim=2)
    nb = torch.cat([prev, nxt], dim=1).permute(0, 2, 1, 3, 4).reshape(12, 6, 32, 32)
    xf = x.permute(0, 2, 1, 3, 4).reshape(12, 3, 32, 32)
    with torch.no_grad():
        closed = net(xf, cond, nb)
        net.in_conv_t.weight.normal_(0, 0.05)
        opened = net(xf, cond, nb)
    assert not torch.allclose(closed, opened, atol=1e-6), "temporal path has no effect"


def test_sandwich_post_restore_temporal_and_single_frame():
    """Sandwich plumbing: neighbours are built from the clip, T=1 must work."""
    torch.manual_seed(0)
    model = SandwichPreprocessor(post_base=8, post_temporal=True)
    with torch.no_grad():
        model.post_strength.fill_(1.0)
        model.post_net.in_conv_t.weight.normal_(0, 0.05)   # open the branch
    x = _clip(t=5)
    out = model.post_restore(x, _cond())
    assert out.shape == x.shape
    assert not torch.allclose(out, x, atol=1e-6)
    # single-frame clip (image tasks / All-Intra): neighbours replicate it
    one = _clip(t=1)
    assert model.post_restore(one, _cond()).shape == one.shape


def test_temporal_flag_is_recorded_and_defaults_off():
    assert SandwichPreprocessor(post_base=8).post_net.temporal is False
    assert SandwichPreprocessor(post_base=8, post_temporal=True).post_net.temporal is True
    # non-temporal path must stay call-compatible with the old signature
    model = SandwichPreprocessor(post_base=8)
    x = _clip(t=4)
    with torch.no_grad():
        model.post_strength.fill_(0.5)
    assert model.post_restore(x, _cond()).shape == x.shape


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all temporal-post tests passed")
