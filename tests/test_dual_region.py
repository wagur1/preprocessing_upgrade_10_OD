"""Tests for the dual-region probe and the constant fill it introduces.

The four new arms are taken from published baselines, so what these tests pin is
that each arm changes exactly the pixels it claims to change. If `mask` touched a
protected pixel, or `roiblur` left the background un-filled, the probe would
attribute a rate change to the wrong mechanism.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ops"))

from src.models.mask_suppress import fill_outside, protect_mask, suppress  # noqa: E402


def _spec(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"ops/probe_dual_region.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def dual():
    return _spec("dual_region_test")


def _frame():
    """[1,3,1,8,8] frame with a strong vertical gradient and a sharp block."""
    base = torch.linspace(0, 1, 64).reshape(1, 1, 8, 8).repeat(1, 3, 1, 1)
    base[:, :, 2:6, 2:6] = 0.9
    return base.unsqueeze(2)


def _mask():
    m = torch.zeros(1, 1, 8, 8)
    m[:, :, 2:6, 2:6] = 1.0
    return m


def test_fill_outside_is_bit_exact_inside_and_constant_outside():
    x = _frame()
    out = fill_outside(x, _mask(), 0.5)
    assert torch.equal(out[:, :, :, 2:6, 2:6], x[:, :, :, 2:6, 2:6]), "protected pixels must not move"
    assert torch.allclose(out[:, :, :, :2, :], torch.full_like(out[:, :, :, :2, :], 0.5))
    assert torch.allclose(out[:, :, :, 6:, :], torch.full_like(out[:, :, :, 6:, :], 0.5))
    assert torch.isfinite(out).all()


def test_fill_outside_rejects_out_of_range_value_and_bad_mask():
    x, m = _frame(), _mask()
    for value in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError, match="value"):
            fill_outside(x, m, value)
    with pytest.raises(ValueError, match="mask"):
        fill_outside(x, torch.full_like(m, 1.5), 0.5)
    with pytest.raises(ValueError, match="mask"):
        fill_outside(x, m[:, :, :4, :4], 0.5)  # wrong spatial size


def test_arm_transforms_change_exactly_their_region(dual):
    x, m, p = _frame(), _mask(), dict(r0_sigma=4.0, roi_sigma=1.0, fill=0.5)

    r0 = dual.arm_r0blur(x, m, p)
    assert torch.equal(r0[:, :, :, 2:6, 2:6], x[:, :, :, 2:6, 2:6]), "r0blur must not touch ROI"
    assert not torch.equal(r0[:, :, :, :2, :], x[:, :, :, :2, :]), "r0blur must blur background"

    masked = dual.arm_mask(x, m, p)
    assert torch.equal(masked[:, :, :, 2:6, 2:6], x[:, :, :, 2:6, 2:6])
    assert torch.all(masked[:, :, :, :2, :] == 0.5), "mask must fill background with the constant"

    gb = dual.arm_globalblur(x, m, p)
    assert not torch.equal(gb, x), "globalblur must change the whole frame"
    assert not torch.equal(gb[:, :, :, 2:6, 2:6], x[:, :, :, 2:6, 2:6]), "globalblur has no mask"

    rb = dual.arm_roiblur(x, m, p)
    assert torch.all(rb[:, :, :, :2, :] == 0.5), "roiblur must fill background"
    assert not torch.equal(rb[:, :, :, 2:6, 2:6], x[:, :, :, 2:6, 2:6]), "roiblur must blur ROI"
    fill_const = torch.full_like(rb[:, :, :, 2:6, 2:6], 0.5)
    assert not torch.allclose(rb[:, :, :, 2:6, 2:6], fill_const), \
        "ROI must be blurred, not overwritten by the fill constant"


def test_probe_arg_validation(dual, tmp_path):
    base = dict(images="x", ann="y", n_images=1, image_ids=[], size=8, score=0.5,
                dilate=0.15, r0_sigma=4.0, roi_sigma=1.0, fill=0.5, device="cpu",
                out=str(tmp_path))
    def run_with(**over):
        dual.run(argparse.Namespace(**{**base, **over}))
    with pytest.raises(ValueError, match="QPs"):
        run_with(qps="30,30")
    with pytest.raises(ValueError, match="fill"):
        run_with(qps="30,40", fill=1.5)
    with pytest.raises(ValueError, match="r0-sigma"):
        run_with(qps="30,40", r0_sigma=-1)
    with pytest.raises(RuntimeError, match="ffmpeg"):
        run_with(qps="30,40")
