"""COCO-style image detection data path (single-frame, T=1).

The project's other data paths are video (Kinetics clips, GOT-10k sequences);
detection on images needs the same index JSON convention but one frame per item
and per-image boxes. Images are squashed to a square ``frame_size`` and the gt
boxes are scaled with them, so the detector, the preprocessor and the metric all
live in the same coordinate frame as ops/probe_detection.py (which evaluates
this task).

Index format (one JSON, mirrors the other datasets):
    {"train": [{"path": ..., "boxes": [[x, y, w, h], ...], "labels": [...]}],
     "val":   [...], "test":  [...]}
Boxes are absolute pixels in the ORIGINAL image frame; the dataset scales them.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# ------------------------------------------------------------------ index ---
def build_coco_index(
    images_dir: str | Path,
    ann_file: str | Path,
    out_json: str | Path,
    n_train: int = 20000,
    n_val: int = 2000,
    seed: int = 0,
) -> Path:
    """Write an index JSON from a COCO instances file (train/val split by id)."""
    images_dir, ann_file = Path(images_dir), Path(ann_file)
    ann = json.loads(ann_file.read_text())
    by_img: Dict[int, List[dict]] = {}
    for a in ann["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    keep = [im for im in ann["images"] if im["id"] in by_img]
    rng = random.Random(seed)
    rng.shuffle(keep)

    def _rec(im):
        return {
            "path": str(images_dir / im["file_name"]),
            "boxes": [a["bbox"] for a in by_img[im["id"]]],
            "labels": [a["category_id"] for a in by_img[im["id"]]],
            "width": im["width"], "height": im["height"],
        }

    splits = {
        "train": [_rec(im) for im in keep[:n_train]],
        "val": [_rec(im) for im in keep[n_train:n_train + n_val]],
        "test": [_rec(im) for im in keep[n_train + n_val:]],
    }
    out = Path(out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": {"source": str(ann_file)}, **splits}))
    print(f"[coco_det] index -> {out}  "
          f"train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")
    return out


# ---------------------------------------------------------------- dataset ---
class CocoDetDataset(Dataset):
    """Single-frame images + detection targets.

    Each item is ``(image [3,1,S,S] in [0,1], targets {"boxes": [N,4] xyxy,
    "labels": [N]})`` with boxes in the resized frame — the same frame the
    detector and the codec see.
    """

    def __init__(self, index_json: str, split: str = "train", frame_size: int = 320,
                 train: bool = True, max_items: int | None = None):
        with open(index_json, "r", encoding="utf-8") as f:
            index = json.load(f)
        self.records = index[split]
        if max_items:
            self.records = self.records[:max_items]
        self.frame_size = int(frame_size)
        self.train = train

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        from PIL import Image

        rec = self.records[i]
        img = Image.open(rec["path"]).convert("RGB")
        w0, h0 = img.size
        s = self.frame_size
        img = img.resize((s, s), Image.BILINEAR)
        arr = torch.from_numpy(np.array(img, copy=True)).float().div_(255.0)
        clip = arr.permute(2, 0, 1).unsqueeze(1)              # [3,1,S,S]

        sx, sy = s / w0, s / h0
        boxes = []
        labels = []
        for (x, y, w, h), lab in zip(rec["boxes"], rec["labels"]):
            x1, y1 = x * sx, y * sy
            x2, y2 = (x + w) * sx, (y + h) * sy
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue                                       # drop degenerate boxes
            boxes.append([x1, y1, x2, y2])
            labels.append(int(lab))
        return clip, {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }


def collate_coco_det(batch):
    """[B,3,1,S,S] clip batch + a LIST of target dicts (torchvision's format)."""
    clips = torch.stack([b[0] for b in batch], dim=0)
    return clips, [b[1] for b in batch]
