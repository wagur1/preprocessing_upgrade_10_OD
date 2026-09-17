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

import hashlib
import json
import random
import tempfile
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
            "image_id": im["id"],
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


OD_SPLIT_POLICY = "train2017-heldout-val2017-test-v1"


def validate_od_index(index: dict, n_train: int | None = None,
                      n_val: int | None = None) -> str:
    """Reject legacy/leaky/empty OD splits; return a mount-independent identity."""
    meta = index.get("meta", {})
    if meta.get("split_policy") != OD_SPLIT_POLICY:
        raise ValueError("Incompatible OD split policy: rebuild the index and start a "
                         "fresh run; old val2017-selected checkpoints cannot resume.")
    sets = {}
    canonical = {}
    for name, expected in (("train", n_train), ("val", n_val), ("test", None)):
        records = index.get(name, [])
        if not records:
            raise ValueError(f"OD split '{name}' is empty")
        if expected is not None and len(records) != expected:
            raise ValueError(f"OD split '{name}' has {len(records)} records, expected {expected}")
        ids = [r["image_id"] for r in records]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate image IDs in OD split '{name}'")
        sets[name] = set(ids)
        canonical[name] = [{k: v for k, v in r.items() if k != "path"} for r in records]
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if sets[a] & sets[b]:
            raise ValueError(f"Overlapping image IDs in OD splits '{a}' and '{b}'")
    return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()


def prepare_od_index(train_dir, train_ann, test_dir, test_ann, out_json,
                     n_train=20000, n_val=2000, seed=0) -> str:
    """Train + selection on disjoint train2017 IDs, all annotated val2017 for test.

    Validate cached indexes too; never silently reuse the former leaky policy.
    The returned fingerprint is persisted in checkpoint cfg by the OD trainer.
    """
    if n_train <= 0 or n_val <= 0:
        raise ValueError("n_train and n_val must both be positive")
    out = Path(out_json)
    metadata = {"split_policy": OD_SPLIT_POLICY, "n_train": n_train,
                "n_val": n_val, "seed": seed}
    if out.exists():
        index = json.loads(out.read_text())
        if any(index.get("meta", {}).get(k) != v for k, v in metadata.items()):
            raise ValueError("Cached OD index is incompatible with requested split policy/counts/seed; "
                             "remove it and start a fresh run")
    else:
        with tempfile.TemporaryDirectory() as tmp:
            tr = build_coco_index(train_dir, train_ann, Path(tmp) / "train.json",
                                  n_train=n_train, n_val=n_val, seed=seed)
            te = build_coco_index(test_dir, test_ann, Path(tmp) / "test.json",
                                  n_train=0, n_val=0, seed=seed)
            a, b = json.loads(tr.read_text()), json.loads(te.read_text())
        index = {"meta": metadata, "train": a["train"], "val": a["val"], "test": b["test"]}
    fingerprint = validate_od_index(index, n_train, n_val)
    # Verify provenance against the actual annotation files, not just a cache's
    # self-reported policy. Also reject stale paths from a different mount.
    for names, directory, annotation in ((("train", "val"), train_dir, train_ann),
                                          (("test",), test_dir, test_ann)):
        images = {im["id"]: im["file_name"] for im in json.loads(Path(annotation).read_text())["images"]}
        for name in names:
            for rec in index[name]:
                iid = rec["image_id"]
                if iid not in images or Path(rec["path"]) != Path(directory) / images[iid]:
                    raise ValueError(f"OD split '{name}' has wrong annotation source or stale path: {iid}")
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(index))
    print(f"[dettrain] validated index train={len(index['train'])} val={len(index['val'])} "
          f"test={len(index['test'])} fingerprint={fingerprint}")
    return fingerprint


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
