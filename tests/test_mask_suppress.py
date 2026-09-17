"""Tests for the R0 preprocessor (src/models/mask_suppress.py).

These pin the two properties the whole measurement rests on: the mask really
covers the detector's boxes (and refuses low-confidence ones), and suppression
touches nothing inside the mask. If either breaks, the probe would attribute a
detector-accuracy change to the codec instead of to the mask.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.mask_suppress import (  # noqa: E402
    mask_from_detections,
    protect_mask,
    suppress,
)


def test_mask_covers_boxes_and_ignores_low_scores():
    boxes = torch.tensor([[10.0, 12.0, 30.0, 36.0]])
    m = protect_mask(boxes, torch.tensor([0.9]), torch.tensor([1]), 64,
                     score_thresh=0.5, dilate=0.15)
    assert m.shape == (1, 1, 64, 64)
    assert m[:, :, 12:36, 10:30].min() == 1.0, "box interior not protected"
    assert m[:, :, 0, 0] == 0.0 and m[:, :, -1, -1] == 0.0, "border not suppressed"
    # confidence gate
    assert protect_mask(boxes, torch.tensor([0.1]), torch.tensor([1]), 64, 0.5, 0.15).sum() == 0
    # box at the frame edge must not raise or wrap
    edge = protect_mask(torch.tensor([[0.0, 0.0, 8.0, 8.0]]), torch.tensor([0.9]),
                        torch.tensor([1]), 64, 0.5, 0.5)
    assert edge[:, :, 0, 0] == 1.0


def test_suppress_is_identity_inside_and_blurs_outside():
    x = torch.rand(1, 3, 1, 64, 64)
    m = protect_mask(torch.tensor([[10.0, 12.0, 30.0, 36.0]]), torch.tensor([0.9]),
                     torch.tensor([1]), 64, 0.5, 0.15)
    out = suppress(x, m, sigma=6.0)
    inside = (slice(None), slice(None), slice(None), slice(12, 36), slice(10, 30))
    assert torch.equal(out[inside], x[inside]), "protected region was modified"
    assert not torch.allclose(out[:, :, :, :6, :6], x[:, :, :, :6, :6], atol=1e-4), \
        "outside was not blurred"
    assert torch.equal(suppress(x, m, 0.0), x), "sigma=0 must be a no-op"


def test_suppress_handles_clips_and_batches():
    """The transform is used on [B,3,T,H,W]; T>1 must not scramble the layout."""
    x = torch.rand(2, 3, 3, 32, 32)
    m = torch.zeros(1, 1, 32, 32)
    m[:, :, :8, :8] = 1.0
    out = suppress(x, m, 4.0)
    assert out.shape == x.shape
    assert torch.equal(out[:, :, :, :8, :8], x[:, :, :, :8, :8])


def test_mask_follows_the_boxes_device():
    """The detector's boxes decide the mask's device.

    A CPU mask against a CUDA frame raises at `x * m` — a RuntimeError a CPU-only
    local suite cannot reproduce, so the contract is pinned here instead: whatever
    device the boxes live on, the mask (and therefore `suppress`) must follow.
    """
    boxes = torch.tensor([[1.0, 2.0, 8.0, 9.0]])
    m = protect_mask(boxes, torch.tensor([0.9]), torch.tensor([1]), 32, 0.5, 0.1)
    assert m.device == boxes.device
    x = torch.rand(1, 3, 1, 32, 32, device=boxes.device)
    out = suppress(x, m, 4.0)
    assert out.device == x.device


def test_blur_matches_torchvision_gaussian_blur_reference():
    """The separable blur must equal torchvision's gaussian_blur when both use
    kernel 2*round(2*sigma)+1 and the same sigma (infinite-boundary agreement)."""
    from torchvision.transforms.functional import gaussian_blur as tv_blur

    x = torch.rand(1, 3, 1, 32, 48)
    m = torch.zeros(1, 1, 32, 48)
    for sigma in (1.5, 4.0):
        out = suppress(x, m, sigma)
        k = int(2 * round(2 * sigma) + 1)
        ref = tv_blur(x[:, :, 0], kernel_size=[k, k], sigma=[sigma, sigma])
        assert torch.allclose(out[:, :, 0], ref, atol=1e-5), \
            f"blur disagrees with torchvision at sigma={sigma}"
        assert out.shape == x.shape


def test_suppress_constant_and_impulse_invariance():
    """A constant image must stay constant under suppression; a protected pixel
    must pass through bit-exactly even where the blur support overlaps it."""
    x = torch.full((1, 3, 1, 16, 16), 0.5)
    m = torch.zeros(1, 1, 16, 16)
    out = suppress(x, m, 2.0)
    assert torch.allclose(out, x), "constant image changed under blur"

    impulse = torch.zeros(1, 1, 1, 16, 16)
    impulse[:, :, :, 5, 5] = 1.0
    m1 = torch.zeros(1, 1, 16, 16)
    m1[:, :, 4:7, 4:7] = 1.0
    out_i = suppress(impulse, m1, 2.0)
    assert out_i[:, :, :, 5, 5] == 1.0, "protected pixel was modified"


def test_suppress_handles_small_images_and_replicate_padding():
    """Reflection padding is illegal when the radius reaches the image size;
    suppression must fall back safely instead of raising."""
    x = torch.rand(1, 3, 1, 3, 3)
    m = torch.zeros(1, 1, 3, 3)
    out = suppress(x, m, 2.0)          # radius 4 > 3-1
    assert out.shape == x.shape and torch.isfinite(out).all()
    one = suppress(torch.rand(1, 3, 1, 1, 1), torch.zeros(1, 1, 1, 1), 2.0)
    assert one.shape == (1, 3, 1, 1, 1) and torch.isfinite(one).all()


def test_suppress_validates_dtype_device_and_mask_layout():
    """Device/dtype mismatches must be handled, not raised; bad layouts must be."""
    x = torch.rand(1, 3, 1, 16, 16, dtype=torch.float64)
    m64 = torch.zeros(1, 1, 16, 16, dtype=torch.float64)
    out = suppress(x, m64, 1.0)
    assert out.dtype == torch.float64 and torch.isfinite(out).all()
    cpu = suppress(x, torch.zeros(1, 1, 16, 16, dtype=torch.float16), 1.0)
    assert cpu.dtype == torch.float64
    for bad in (torch.zeros(1, 3, 16, 16), torch.zeros(2, 1, 16, 16),
                torch.zeros(1, 1, 16, 15)):
        try:
            suppress(x, bad, 1.0)
        except ValueError:
            continue
        raise AssertionError(f"mask {tuple(bad.shape)} should be rejected")
    try:
        suppress(torch.full((1, 3, 1, 16, 16), float("nan")),
                 torch.zeros(1, 1, 16, 16), 1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("non-finite input should be rejected")


def test_suppress_broadcasts_batch_and_time_masks():
    """A [B,1,H,W] mask must apply per-batch (differing B rows) and a
    [B,1,T,H,W] mask per frame; B=1 or T=1 must broadcast."""
    x = torch.rand(2, 3, 2, 32, 32)
    mb = torch.zeros(2, 1, 32, 32)
    mb[0, :, :8, :] = 1.0
    mb[1, :, 16:, :] = 1.0
    out = suppress(x, mb, 4.0)
    assert torch.equal(out[:, :, :, :8, :][:1], x[:, :, :, :8, :][:1])
    assert not torch.equal(out[:, :, :, :8, :][1], x[:, :, :, :8, :][1])
    assert torch.equal(out[:, :, :, 16:, :][1], x[:, :, :, 16:, :][1])
    # singleton B and T masks broadcast over the input's batch/time axes
    x1 = torch.rand(2, 3, 2, 32, 32)
    m1 = torch.zeros(1, 1, 32, 32)
    m1[:, :, :8, :8] = 1.0
    out1 = suppress(x1, m1, 4.0)
    assert torch.equal(out1[:, :, :, :8, :8], x1[:, :, :, :8, :8])
    m2 = torch.zeros(1, 1, 1, 32, 32)
    m2[:, :, :, :8, :8] = 1.0
    out2 = suppress(x1, m2, 4.0)
    assert torch.equal(out2[:, :, :, :8, :8], x1[:, :, :, :8, :8])
    mt = torch.zeros(2, 1, 2, 32, 32)
    mt[:, :, 1, :8, :8] = 1.0
    out_t = suppress(x, mt, 4.0)
    assert torch.equal(out_t[:, :, 1, :8, :8], x[:, :, 1, :8, :8])
    assert not torch.equal(out_t[:, :, 0, :8, :8], x[:, :, 0, :8, :8])


def test_protect_mask_requires_finite_matching_inputs():
    with pytest.raises(ValueError):
        protect_mask(torch.tensor([[float("nan"), 1, 5, 5]]),
                     torch.tensor([0.9]), torch.tensor([1]), 16)
    with pytest.raises(ValueError):
        protect_mask(torch.tensor([[0.0, 0.0, 5.0, 5.0]]),
                     torch.tensor([0.9, 0.9]), torch.tensor([1]), 16)
    with pytest.raises(ValueError):
        protect_mask(torch.tensor([[0.0, 0.0, 1.0]]),
                     torch.tensor([0.9]), torch.tensor([1]), 16)


def test_mask_from_detections_matches_manual_call():
    det = {"boxes": torch.tensor([[5.0, 5.0, 25.0, 25.0]]),
           "scores": torch.tensor([0.8]), "labels": torch.tensor([3])}
    a = mask_from_detections(det, 64, 0.5, 0.15)
    b = protect_mask(det["boxes"], det["scores"], det["labels"], 64, 0.5, 0.15)
    assert torch.equal(a, b)
