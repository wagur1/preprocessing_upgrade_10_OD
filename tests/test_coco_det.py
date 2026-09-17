"""Tests for the COCO detection data path (src/data/coco_det.py).

The index builder is where a one-key mistake already cost a 12 h run: the val
split came out empty, which silently disabled validation and early stopping. The
empty-split assert and the box-scaling convention are pinned here.
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.coco_det import CocoDetDataset, build_coco_index, collate_coco_det  # noqa: E402

FIX = Path(r"C:\tmp\coco_fixture")


def _index(tmp_path: Path) -> Path:
    return build_coco_index(FIX / "val2017",
                            FIX / "annotations" / "instances_val2017.json",
                            tmp_path / "idx.json", n_train=6, n_val=4)


def test_index_splits_are_non_empty(tmp_path):
    """A non-empty val split is what keeps model selection alive."""
    idx = _index(tmp_path)
    d = json.loads(Path(idx).read_text())
    assert len(d["train"]) == 6 and len(d["val"]) == 4
    assert all(d[k] for k in ("train", "val")), "a split came out empty"


def test_item_shapes_and_box_scaling(tmp_path):
    ds = CocoDetDataset(str(_index(tmp_path)), split="train", frame_size=320)
    clip, tgt = ds[0]
    assert clip.shape == (3, 1, 320, 320)
    assert clip.min() >= 0.0 and clip.max() <= 1.0
    assert tgt["boxes"].shape[1] == 4 and tgt["labels"].ndim == 1
    # boxes must live inside the resized frame
    if len(tgt["boxes"]):
        assert tgt["boxes"].min() >= 0.0
        assert tgt["boxes"].max() <= 320.0 + 1e-3


def test_collate_returns_list_of_targets(tmp_path):
    ds = CocoDetDataset(str(_index(tmp_path)), split="train", frame_size=64)
    clips, targets = collate_coco_det([ds[0], ds[1]])
    assert clips.shape == (2, 3, 1, 64, 64)
    assert isinstance(targets, list) and len(targets) == 2
    assert set(targets[0]) == {"boxes", "labels"}
