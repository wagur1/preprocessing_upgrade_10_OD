"""Tests for ops/probe_detection.py — the metric path that decides the probe.

The expensive part of the probe (detector passes) is validated by inspection on
real COCO images; what these tests pin is the cheap, silent part: the box format
handed to COCOeval and the arithmetic around it.
"""

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "ops"))

from probe_detection import _coco_box, coco_map  # noqa: E402

ANN = {"categories": [{"id": 1, "name": "thing"}]}
GT_BOX = [10.0, 20.0, 100.0, 50.0]           # x, y, w, h
GT = {7: [{"id": 1, "image_id": 7, "category_id": 1, "iscrowd": 0,
           "area": 100.0 * 50.0, "bbox": GT_BOX}]}


def test_coco_box_converts_xyxy_to_xywh():
    """torchvision returns xyxy; COCO's bbox field is xywh."""
    xyxy = torch.tensor([10.0, 20.0, 110.0, 70.0])
    assert _coco_box(xyxy) == [10.0, 20.0, 100.0, 50.0]
    assert _coco_box([10.0, 20.0, 110.0, 70.0]) == [10.0, 20.0, 100.0, 50.0]
    # degenerate boxes must not produce zero/negative extents
    w, h = _coco_box([5.0, 5.0, 5.0, 5.0])[2:]
    assert w > 0 and h > 0


def test_perfect_detection_scores_one():
    """A detector returning the ground-truth box exactly must score mAP 1.0.

    Regression for the xyxy-as-xywh bug: submitting torchvision's format into
    COCO's xywh field made every IoU wrong, so mAP collapsed to ~0.02 at every
    resolution — which reads as a broken detector or a gt-alignment problem
    rather than a one-line format bug in the metric path.
    """
    preds = [{"image_id": 7, "category_id": 1,
              "bbox": _coco_box(torch.tensor([10.0, 20.0, 110.0, 70.0])),
              "score": 0.9}]
    ap, ap50 = coco_map(preds, GT, [7], ANN)
    assert ap > 0.99, f"perfect detection scored mAP={ap}"
    assert ap50 > 0.99


def test_xyxy_submitted_as_xywh_scores_low():
    """The old buggy behaviour must NOT look like a pass (guards the guard)."""
    bad = [{"image_id": 7, "category_id": 1,
            "bbox": [10.0, 20.0, 110.0, 70.0], "score": 0.9}]
    ap, _ = coco_map(bad, GT, [7], ANN)
    assert ap < 0.5, f"xyxy-as-xywh scored mAP={ap}, the test would not catch it"


def test_coco_map_handles_no_results():
    ap, ap50 = coco_map([], GT, [7], ANN)
    assert ap == 0.0 and ap50 == 0.0


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all probe-detection tests passed")
