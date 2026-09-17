"""Tests for the C1 yuv420 colourspace proxy (upgrade-5.1).

Covers ``src/models/color.py`` (BT.601 matrix, 4:2:0 subsample geometry) and
the ``VirtualCodec(colorspace="yuv420")`` integration: plane-split coding,
chroma quant offset, rate behaviour vs the legacy rgb path, and gradient flow
through the whole path.
"""

from __future__ import annotations

import pytest
import torch

from src.models.color import (
    rgb_to_ycbcr,
    ycbcr_to_rgb,
    rgb_to_yuv420_planes,
    yuv420_planes_to_rgb,
    yuv420_roundtrip,
)
from src.models.virtual_codec import VirtualCodec


# --------------------------------------------------------------------------
# colour matrix
# --------------------------------------------------------------------------
class TestColorMatrix:
    def test_roundtrip_near_identity(self):
        torch.manual_seed(0)
        x = torch.rand(2, 3, 4, 32, 32)
        err = (ycbcr_to_rgb(rgb_to_ycbcr(x)) - x).abs().max()
        assert err < 1e-5

    def test_4d_and_5d_supported(self):
        x4 = torch.rand(2, 3, 16, 16)
        x5 = torch.rand(2, 3, 4, 16, 16)
        assert rgb_to_ycbcr(x4).shape == x4.shape
        assert rgb_to_ycbcr(x5).shape == x5.shape

    def test_5d_equals_per_frame_4d(self):
        # regression: a naive reshape(-1, 3, H, W) on [B,3,T,H,W] interleaves
        # channels of different frames; the 5D path must match frame-by-frame
        torch.manual_seed(15)
        x5 = torch.rand(2, 3, 4, 16, 16)
        out5 = rgb_to_ycbcr(x5)
        for b in range(2):
            for t in range(4):
                per_frame = rgb_to_ycbcr(x5[b, :, t].unsqueeze(0))
                assert torch.allclose(out5[b, :, t].unsqueeze(0), per_frame, atol=1e-6)

    def test_luma_of_grey(self):
        # grey -> Y equals the grey level, chroma centred at 0.5
        g = torch.full((1, 3, 8, 8), 0.25)
        ycc = rgb_to_ycbcr(g)
        assert torch.allclose(ycc[:, 0], torch.full_like(ycc[:, 0], 0.25), atol=1e-6)
        assert torch.allclose(ycc[:, 1:], torch.full_like(ycc[:, 1:], 0.5), atol=1e-6)

    def test_range(self):
        torch.manual_seed(1)
        ycc = rgb_to_ycbcr(torch.rand(1, 3, 16, 16))
        assert ycc.min() >= 0.0 and ycc.max() <= 1.0


# --------------------------------------------------------------------------
# 4:2:0 geometry
# --------------------------------------------------------------------------
class TestSubsample:
    def test_plane_shapes(self):
        y, c = rgb_to_yuv420_planes(torch.rand(2, 3, 112, 128))
        assert y.shape == (2, 1, 112, 128)
        assert c.shape == (2, 2, 56, 64)

    def test_odd_sizes(self):
        y, c = rgb_to_yuv420_planes(torch.rand(1, 3, 31, 23))
        assert y.shape == (1, 1, 31, 23)
        assert c.shape == (1, 2, 16, 12)
        out = yuv420_planes_to_rgb(y, c)
        assert out.shape == (1, 3, 31, 23)

    def test_roundtrip_is_lossy_on_chroma(self):
        torch.manual_seed(2)
        x = torch.rand(1, 3, 16, 16)
        cost = (yuv420_roundtrip(x) - x).abs().mean()
        assert cost > 1e-4, "chroma subsampling must cost something"

    def test_roundtrip_preserves_achromatic(self):
        torch.manual_seed(3)
        # identical noise on all channels -> chroma exactly flat, luma-only wobble
        g = 0.5 + 0.01 * torch.rand(1, 1, 16, 16)
        grey = g.expand(1, 3, 16, 16)
        cost = (yuv420_roundtrip(grey) - grey).abs().mean()
        assert cost < 1e-3, f"achromatic damage should be tiny, got {cost}"

    def test_luma_passthrough(self):
        # luma survives the 4:2:0 round trip exactly, as long as the RGB clamp
        # never bites -- probe with low-saturation content (real decoders clamp)
        torch.manual_seed(4)
        low_sat = 0.5 + 0.1 * (torch.rand(1, 3, 2, 16, 16) - 0.5)
        y_src = rgb_to_ycbcr(low_sat)[:, 0:1]
        y_rt = rgb_to_ycbcr(yuv420_roundtrip(low_sat))[:, 0:1]
        assert torch.allclose(y_src, y_rt, atol=1e-6)

    def test_gradients_flow(self):
        x = torch.rand(1, 3, 2, 16, 16, requires_grad=True)
        yuv420_roundtrip(x).mean().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()


# --------------------------------------------------------------------------
# VirtualCodec yuv420 integration
# --------------------------------------------------------------------------
class TestVirtualCodecYuv420:
    def _codec(self, **kw) -> VirtualCodec:
        # default steps (fine=0.03) so quality 8 reconstructs near-losslessly;
        # the Kaggle-calibrated 3.0/1.0 steps are for rate-monotonicity checks
        return VirtualCodec(
            qualities=(1, 2, 3, 5, 8), block=8, step_coarse=0.25, step_fine=0.03, **kw
        )

    def _calibrated(self, **kw) -> VirtualCodec:
        return VirtualCodec(
            qualities=(1, 2, 3, 5, 8), block=8, step_coarse=3.0, step_fine=1.0, **kw
        )

    def test_default_is_yuv420(self):
        assert self._codec().colorspace == "yuv420"

    def test_invalid_colorspace_rejected(self):
        with pytest.raises(ValueError):
            VirtualCodec(colorspace="yuv422")

    def test_invalid_chroma_scale_rejected(self):
        with pytest.raises(ValueError):
            VirtualCodec(chroma_step_scale=0.0)

    def test_rate_monotone_across_qualities(self):
        torch.manual_seed(5)
        cod = self._codec()
        x = torch.rand(2, 3, 4, 32, 32)
        bpps = [cod.compress_decompress(x, q)[1] for q in (1, 2, 3, 5, 8)]
        assert bpps == sorted(bpps), bpps

    def test_distortion_monotone_endpoints(self):
        torch.manual_seed(6)
        cod = self._codec()
        x = torch.rand(2, 3, 4, 32, 32)
        mse = lambda q: (cod.compress_decompress(x, q)[0] - x).square().mean()
        assert mse(1) > mse(8)

    def test_fine_step_reconstructs_luma(self):
        torch.manual_seed(7)
        cod = self._codec()
        x = torch.rand(2, 3, 4, 32, 32)
        x_hat, _ = cod.compress_decompress(x, 8)
        y_src = rgb_to_ycbcr(x)[:, 0:1]
        y_rec = rgb_to_ycbcr(x_hat)[:, 0:1]
        assert (y_rec - y_src).abs().mean() < 0.05

    def test_chroma_damage_present_even_at_fine_step(self):
        # the 4:2:0 geometry is rate-INdependent, so even the finest step must
        # hurt chroma high-frequency -- the exact damage the rgb proxy missed
        torch.manual_seed(8)
        cod = self._codec()
        base = torch.rand(1, 3, 1, 32, 32)
        # paint high-frequency colour: same luma, alternating chroma
        chroma = torch.zeros(1, 3, 1, 32, 32)
        chroma[:, 0] = 0.5
        chroma[:, 1, :, :, 0::2] = 0.8
        chroma[:, 1, :, :, 1::2] = 0.2
        x = (base * 0.5 + chroma * 0.5).clamp(0, 1)
        x_hat, _ = cod.compress_decompress(x, 8)
        c_src = rgb_to_ycbcr(x)[:, 1:]
        c_rec = rgb_to_ycbcr(x_hat)[:, 1:]
        assert (c_rec - c_src).abs().mean() > 0.01

    def test_yuv420_cheaper_than_rgb(self):
        # chroma carries 1/4 the samples at a coarser step -> lower bpp
        torch.manual_seed(9)
        x = torch.rand(2, 3, 4, 32, 32)
        b_yuv = self._codec(colorspace="yuv420").compress_decompress(x, 8)[1]
        b_rgb = self._codec(colorspace="rgb").compress_decompress(x, 8)[1]
        assert b_yuv < b_rgb, (b_yuv, b_rgb)

    def test_chroma_scale_raises_chroma_distortion(self):
        torch.manual_seed(10)
        x = torch.rand(1, 3, 4, 32, 32)
        coarse = self._codec(chroma_step_scale=4.0)
        fine = self._codec(chroma_step_scale=1.0)
        x_c, _ = coarse.compress_decompress(x, 3)
        x_f, _ = fine.compress_decompress(x, 3)
        c_src = rgb_to_ycbcr(x)[:, 1:]
        d_c = (rgb_to_ycbcr(x_c)[:, 1:] - c_src).abs().mean()
        d_f = (rgb_to_ycbcr(x_f)[:, 1:] - c_src).abs().mean()
        assert d_c > d_f

    def test_gradients_flow_through_training_path(self):
        torch.manual_seed(11)
        cod = self._codec()
        x = torch.rand(2, 3, 4, 32, 32, requires_grad=True)
        x_hat, bpp = cod(x, 3)
        (x_hat.mean() + bpp).backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        assert torch.isfinite(bpp)

    def test_inter_and_intra_paths(self):
        torch.manual_seed(12)
        x = torch.rand(2, 3, 4, 32, 32)
        for inter in (True, False):
            cod = self._codec(inter=inter)
            x_hat, bpp = cod.compress_decompress(x, 3)
            assert x_hat.shape == x.shape and torch.isfinite(x_hat).all()
            assert 0.0 < bpp < 100.0

    def test_anneal_extremes(self):
        torch.manual_seed(13)
        cod = self._codec()
        for a in (0.0, 1.0):
            cod.set_anneal(a)
            x = torch.rand(1, 3, 2, 16, 16, requires_grad=True)
            x_hat, bpp = cod(x, 3)
            (x_hat.mean() + bpp).backward()
            assert x.grad is not None and torch.isfinite(bpp)
        cod.set_anneal(0.0)

    def test_achromatic_reconstructs(self):
        # chroma flat -> 4:2:0 damage vanishes; only quantisation remains.
        # Identical noise across channels keeps chroma exactly flat.
        torch.manual_seed(14)
        cod = self._codec()
        g = torch.rand(1, 1, 4, 32, 32).mean(dim=2, keepdim=True)  # smooth luma
        grey = (0.5 + 0.02 * g).expand(1, 3, 4, 32, 32).clamp(0, 1)
        x_hat, _ = cod.compress_decompress(grey, 8)
        assert (x_hat - grey).abs().mean() < 0.05
