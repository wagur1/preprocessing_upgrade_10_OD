"""Tests for the v9 per-codec POST model."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.percodec_sandwich import PerCodecPostSandwich  # noqa: E402
from src.models.sandwich import SandwichPreprocessor  # noqa: E402
from src.models.upvcm import UPVCMPreprocessor  # noqa: E402


def _clip(b=2, t=8, s=64):
    return torch.rand(b, 3, t, s, s)


def test_identity_at_init():
    torch.manual_seed(0)
    m = PerCodecPostSandwich()
    x = _clip()
    cond = torch.full((2, 1), 0.6)
    assert torch.allclose(m(x, cond), x, atol=1e-6)
    x_hat = torch.rand_like(x)
    for codec in ("h264", "h265"):
        assert torch.allclose(m.post_restore(x_hat, cond, codec=codec), x_hat,
                              atol=1e-6)


def test_codec_routing_changes_output():
    """The whole point of v9: different codecs must give different restores
    once the routing is active. At INIT the codec FiLM is zero (warm-start
    exactness), so this test un-zeros the FiLM final layer first — that is
    the state after a few STE steps of training."""
    m = PerCodecPostSandwich()
    x_hat = _clip()
    cond = torch.full((2, 1), 0.6)
    with torch.no_grad():
        m.post_net.out_conv.weight.normal_(0, 0.05)
        m.post_net.codec_embed.weight.normal_(0, 0.5)   # distinct embeddings
        # activate the codec path: FiLM final layer non-zero
        m.post_net.film.net[2].weight.normal_(0, 0.05)
        m.post_net.film.net[2].bias.normal_(0, 0.05)
        m.post_strength.fill_(0.5)
        r264 = m.post_restore(x_hat, cond, codec="h264")
        r265 = m.post_restore(x_hat, cond, codec="h265")
    assert not torch.allclose(r264, r265, atol=1e-6), "codec routing is a no-op"


def test_gate_gradients_alive_at_init():
    """Dead-saddle regression guard (v7/v8 lesson)."""
    torch.manual_seed(0)
    m = PerCodecPostSandwich()
    x_hat = _clip()
    out = m.post_restore(x_hat, torch.full((2, 1), 0.6), codec="h264")
    out.pow(2).mean().backward()
    assert m.post_strength.grad is not None and m.post_strength.grad.abs() > 0
    assert m.post_net.codec_embed.weight.grad is not None


def test_v8_warmstart_exact_at_load():
    """v8 checkpoint loaded via load_v8_sandwich: codec-agnostic output must
    EQUAL v8's (zero-init codec FiLM contributes nothing)."""
    torch.manual_seed(0)
    v8 = SandwichPreprocessor()
    with torch.no_grad():  # make gates nonzero so identity isn't trivial
        v8.post_net.out_conv.weight.normal_(0, 0.05)
        v8.post_strength.fill_(0.3)
        v8.pre.dec_strength.fill_(-0.4)
    v9 = PerCodecPostSandwich()
    rep = v9.load_v8_sandwich(v8.state_dict())
    # 56 v8 keys total: 4 film keys skipped (different cond width),
    # 1 shape-mismatched → ~51-52 copied. Everything copyable is copied.
    assert rep["copied"] >= 50, rep
    assert rep["missing"] == [], rep["missing"]
    x_hat = _clip(b=1, t=4, s=32)
    cond = torch.full((1, 1), 0.5)
    with torch.no_grad():
        o8 = v8.post_restore(x_hat, cond)
        o9 = v9.post_restore(x_hat, cond, codec="h264")
        o9b = v9.post_restore(x_hat, cond, codec="h265")
    assert torch.allclose(o8, o9, atol=1e-5), "warm-start changed h264 behavior"
    assert torch.allclose(o8, o9b, atol=1e-5), "warm-start changed h265 behavior"


def test_engine_compat_signature():
    """post_restore must accept the engine's keyword call."""
    m = PerCodecPostSandwich()
    x_hat = _clip(b=1, t=4, s=32)
    out = m.post_restore(x_hat, torch.zeros(1, 1), codec="h265")
    assert out.shape == x_hat.shape


def test_param_count():
    m = PerCodecPostSandwich()
    n = sum(p.numel() for p in m.parameters())
    # ~44k PRE + ~577k trunk + codec embed/FiLM delta (~5k)
    assert 600_000 < n < 700_000, f"unexpected param count {n}"


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all v9 tests passed")


def test_dualpost_independent_heads():
    """v9-b: the two POST heads are independent — writing one must not touch
    the other, and the codec switch routes to the right head."""
    import torch as _t
    from src.models.dualpost_sandwich import DualPostSandwich
    _t.manual_seed(0)
    m = DualPostSandwich()
    x_hat = _t.rand(1, 3, 4, 32, 32)
    cond = _t.zeros(1, 1)
    # distinct gates
    with _t.no_grad():
        m.post_264.out_conv.weight.normal_(0, 0.05)
        m.post_265.out_conv.weight.normal_(0, 0.07)  # different init
        m.post_strength_264.fill_(0.5)
        m.post_strength_265.fill_(0.5)
        o264 = m.post_restore(x_hat, cond, codec="h264")
        o265 = m.post_restore(x_hat, cond, codec="h265")
    assert not _t.allclose(o264, o265, atol=1e-6)
    # gradient isolation: loss on h264 arm reaches only post_264
    m2 = DualPostSandwich()
    out = m2.post_restore(x_hat, cond, codec="h264")
    out.pow(2).mean().backward()
    assert m2.post_264.out_conv.weight.grad is not None
    assert m2.post_265.out_conv.weight.grad is None or m2.post_265.out_conv.weight.grad.abs().sum() == 0


def test_dualpost_load_dual_and_identity():
    import torch as _t
    from src.models.dualpost_sandwich import DualPostSandwich
    from src.models.upvcm import UPVCMPreprocessor
    from src.models.sandwich import SandwichPreprocessor
    _t.manual_seed(0)
    m = DualPostSandwich()
    v1 = UPVCMPreprocessor()
    with _t.no_grad():
        v1.dec_strength.fill_(-0.62)
    sA, sB = SandwichPreprocessor(), SandwichPreprocessor()
    with _t.no_grad():
        sA.post_strength.fill_(0.11)
        sB.post_strength.fill_(0.22)
    rep = m.load_dual(v1.state_dict(), sA.state_dict(), sB.state_dict())
    assert rep["post_264"] > 20 and rep["post_265"] > 20
    assert abs(m.post_strength_264.item() - 0.11) < 1e-6
    assert abs(m.post_strength_265.item() - 0.22) < 1e-6
    assert abs(m.pre.dec_strength.item() + 0.62) < 1e-6
    # fresh heads zero-gate => identity
    m2 = DualPostSandwich()
    xh = _t.rand(1, 3, 4, 32, 32)
    assert _t.allclose(m2.post_restore(xh, None, codec="h264"), xh, atol=1e-6)


def test_dualcodec_routes_both_halves():
    """v9-b full: per-codec PRE AND POST — each codec path is independent."""
    import torch as _t
    from src.models.dualpost_sandwich import DualCodecSandwich
    from src.models.upvcm import UPVCMPreprocessor
    from src.models.sandwich import SandwichPreprocessor
    _t.manual_seed(0)
    m = DualCodecSandwich()
    v1 = UPVCMPreprocessor()
    ste = SandwichPreprocessor()
    with _t.no_grad():
        v1.dec_strength.fill_(-0.62)
        ste.pre.dec_strength.fill_(-0.34)
        ste.post_strength.fill_(0.05)
    rep = m.load_record_assembly(v1.state_dict(), ste.state_dict())
    assert rep["copied"] > 60
    x = _t.rand(1, 3, 4, 32, 32)
    with _t.no_grad():
        p264 = m(x, None, codec="h264")
        p265 = m(x, None, codec="h265")
    # different PREs -> different preprocessed outputs (both non-identity)
    assert not _t.allclose(p264, p265, atol=1e-6)
    assert abs(m.pre_264.dec_strength.item() + 0.62) < 1e-6
    assert abs(m.pre_265.dec_strength.item() + 0.34) < 1e-6
    # POST gates copied to both heads
    assert abs(m.post_strength_264.item() - 0.05) < 1e-6
    assert abs(m.post_strength_265.item() - 0.05) < 1e-6


def test_warmstart_with_trained_film_is_not_exact():
    """Audit 2026-09-11 #5: the OLD 'exact warm-start' test used a fresh v8
    (FiLM zero) and proved nothing about trained checkpoints. With a trained
    FiLM (non-zero final layer), load_v8_sandwich DROPS v8's learned affine —
    diff ~1e-3, NOT exact. This test pins that truth so the docs can't drift
    back to the overclaim."""
    import torch as _t
    from src.models.sandwich import SandwichPreprocessor
    from src.models.percodec_sandwich import PerCodecPostSandwich
    _t.manual_seed(0)
    v8 = SandwichPreprocessor()
    with _t.no_grad():
        v8.post_net.out_conv.weight.normal_(0, 0.05)
        v8.post_strength.fill_(0.3)
        v8.post_net.film.net[2].weight.normal_(0, 0.05)  # trained FiLM
        v8.post_net.film.net[2].bias.normal_(0, 0.05)
    v9 = PerCodecPostSandwich()
    v9.load_v8_sandwich(v8.state_dict())
    x = _t.rand(1, 3, 4, 32, 32)
    cond = _t.full((1, 1), 0.5)
    with _t.no_grad():
        d = (v8.post_restore(x, cond)
             - v9.post_restore(x, cond, codec="h264")).abs().max().item()
    assert d > 1e-5, "if this ever becomes exact, update RESULTS_percodec.md"
    assert d < 0.01, "unexpectedly large — investigate the loader"
