"""Tests for the R0 preprocessor (src/models/mask_suppress.py).

These pin the two properties the whole measurement rests on: the mask really
covers the detector's boxes (and refuses low-confidence ones), and suppression
touches nothing inside the mask. If either breaks, the probe would attribute a
detector-accuracy change to the codec instead of to the mask.
"""

import sys
from pathlib import Path

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


def test_mask_from_detections_matches_manual_call():
    det = {"boxes": torch.tensor([[5.0, 5.0, 25.0, 25.0]]),
           "scores": torch.tensor([0.8]), "labels": torch.tensor([3])}
    a = mask_from_detections(det, 64, 0.5, 0.15)
    b = protect_mask(det["boxes"], det["scores"], det["labels"], 64, 0.5, 0.15)
    assert torch.equal(a, b)
