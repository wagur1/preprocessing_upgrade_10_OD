"""Checks for the tracking-eval sandwich arm + per-codec routing (v9 port).

Ported from v8 commit 1869d48, extended for DualCodecSandwich: _evaluate_tracking
now runs 3 arms (anchor / prep+codec / sandwich+codec) and _pre_chunked /
_post_chunked pass codec= to models whose forward/post_restore accept it
(DualCodecSandwich), while codec-agnostic models (sandwich/upvcm) keep the
old call signature.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.engine import _post_chunked, _pre_chunked
from src.models.dualpost_sandwich import DualCodecSandwich
from src.models.sandwich import SandwichPreprocessor
from src.models.upvcm import UPVCMPreprocessor


class _CodecStub(torch.nn.Module):
    """forward/post return x + offset(codec); records the codec it received."""

    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, x, cond=None, mask=None, codec="h264"):
        self.seen.append(("pre", codec))
        return x + (1.0 if codec == "h264" else 2.0)

    def post_restore(self, x_hat, cond=None, codec="h264"):
        self.seen.append(("post", codec))
        return x_hat + (10.0 if codec == "h264" else 20.0)


def _clip(t=7):
    return torch.rand(1, 3, t, 16, 16)


def test_post_chunked_preserves_length_and_order() -> None:
    pre = SandwichPreprocessor(s_ch=4, editor_ch=4, cond_dim=1, post_base=4)
    clip = _clip(12)
    out = _post_chunked(pre, clip, chunk=5, cond=None)
    assert out.shape == clip.shape
    assert torch.allclose(out, clip, atol=1e-6), "identity POST must be seamless across chunks"


def test_codec_is_forwarded_when_the_model_accepts_it() -> None:
    stub = _CodecStub()
    out = _pre_chunked(stub, _clip(), chunk=3, cond=None, codec="h265")
    assert stub.seen == [("pre", "h265")] * 3
    assert torch.allclose(out, _clip() + 2.0, atol=1e-6) or out.min() > 1.9
    stub.seen.clear()
    _post_chunked(stub, _clip(4), chunk=2, cond=None, codec="h264")
    assert stub.seen == [("post", "h264")] * 2


def test_codec_omitted_keeps_the_old_signature() -> None:
    pre = SandwichPreprocessor(s_ch=4, editor_ch=4, cond_dim=1, post_base=4)
    assert _pre_chunked(pre, _clip(6), chunk=4, cond=None, codec=None).shape == _clip(6).shape


def test_dualcodec_signatures_take_codec() -> None:
    m = DualCodecSandwich(s_ch=4, editor_ch=4, post_base=4)
    assert "codec" in inspect.signature(m.forward).parameters
    assert "codec" in inspect.signature(m.post_restore).parameters
    # and the codec-agnostic archs must NOT (the engine's detection relies on this)
    s = SandwichPreprocessor(s_ch=4, editor_ch=4, cond_dim=1, post_base=4)
    assert "codec" not in inspect.signature(s.forward).parameters
    assert "codec" not in inspect.signature(UPVCMPreprocessor(s_ch=4, editor_ch=4).forward).parameters


def test_dualcodec_zero_init_is_identity_both_codecs() -> None:
    m = DualCodecSandwich(s_ch=4, editor_ch=4, post_base=4)
    clip = _clip(9)
    for codec in ("h264", "h265"):
        out = _post_chunked(m, clip, chunk=4, cond=None, codec=codec)
        assert torch.allclose(out, clip, atol=1e-6), codec


if __name__ == "__main__":
    test_post_chunked_preserves_length_and_order()
    test_codec_is_forwarded_when_the_model_accepts_it()
    test_codec_omitted_keeps_the_old_signature()
    test_dualcodec_signatures_take_codec()
    test_dualcodec_zero_init_is_identity_both_codecs()
    print("v9 tracking port self-checks passed")
