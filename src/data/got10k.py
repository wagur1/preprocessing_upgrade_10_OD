"""GOT-10k dataset layer (tracking task).

GOT-10k on disk::

    <root>/<split>/<sequence>/
        00000001.jpg 00000002.jpg ...
        groundtruth.txt      # one "x,y,w,h" line per frame (pixels, top-left)
        absence.label        # optional: 1 = target absent this frame
        cover.label          # optional
        meta_info.ini        # optional

This module provides:

  * ``load_sequence``      -- read a full sequence as a clip tensor + boxes
                             (used at evaluation, where we track every frame),
  * ``GOT10kClipDataset``  -- short clips + per-frame boxes for training the
                             preprocessor,
  * ``iter_sequences``     -- iterate whole sequences listed in an index JSON.

Boxes are returned as **normalised xyxy in [0, 1]** w.r.t. each frame's original
size, so they stay valid after the frame is resized to a square working
resolution (image and box are distorted together).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

cv2.setNumThreads(0)

_IMG_EXTS = (".jpg", ".jpeg", ".png")


# --------------------------------------------------------------------------
# low-level readers
# --------------------------------------------------------------------------
def _list_frames(seq_dir: Path) -> List[Path]:
    frames = [p for p in seq_dir.iterdir() if p.suffix.lower() in _IMG_EXTS]
    return sorted(frames)


def _read_groundtruth(path: Path) -> np.ndarray:
    """Return [T,4] xyxy pixel boxes from a GOT-10k groundtruth.txt."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().replace("\t", ",").replace(" ", ",")
            if not line:
                continue
            parts = [p for p in line.split(",") if p != ""]
            x, y, w, h = (float(v) for v in parts[:4])
            rows.append([x, y, x + w, y + h])
    return np.asarray(rows, dtype=np.float64).reshape(-1, 4)


def _read_absence(seq_dir: Path, n: int) -> np.ndarray:
    """valid[t] = target present (True). Missing file -> all valid."""
    f = seq_dir / "absence.label"
    if not f.exists():
        return np.ones(n, dtype=bool)
    vals = []
    with open(f, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                vals.append(int(float(line)))
    absent = np.asarray(vals[:n], dtype=bool)
    if absent.size < n:
        absent = np.concatenate([absent, np.zeros(n - absent.size, dtype=bool)])
    return ~absent


def is_sequence_dir(d: Path) -> bool:
    return d.is_dir() and (d / "groundtruth.txt").exists()


def find_sequences(root: Path) -> List[Path]:
    """All GOT-10k sequence dirs under root (recursive)."""
    if is_sequence_dir(root):
        return [root]
    seqs = []
    for p in sorted(root.rglob("groundtruth.txt")):
        seqs.append(p.parent)
    return seqs


def _load_frame(path: Path, size: int) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return np.zeros((size, size, 3), np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------
# full-sequence loading (evaluation)
# --------------------------------------------------------------------------
def load_sequence(
    seq_dir: str | Path,
    frame_size: int = 256,
    max_frames: Optional[int] = None,
    stride: int = 1,
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, Tuple[int, int]]:
    """Read a whole sequence.

    Returns:
        clip:  [1, C, T, H, W] float in [0,1] (batch dim of 1),
        boxes: [T,4] normalised xyxy in [0,1],
        valid: [T] bool (target present),
        orig_wh: original (W, H) of the first frame.
    """
    seq_dir = Path(seq_dir)
    frame_paths = _list_frames(seq_dir)
    gt = _read_groundtruth(seq_dir / "groundtruth.txt")
    n = min(len(frame_paths), len(gt))
    frame_paths, gt = frame_paths[:n], gt[:n]
    valid = _read_absence(seq_dir, n)

    idxs = list(range(0, n, stride))
    if max_frames is not None:
        idxs = idxs[:max_frames]

    first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    oh, ow = (first.shape[0], first.shape[1]) if first is not None else (frame_size, frame_size)

    frames = [_load_frame(frame_paths[i], frame_size) for i in idxs]
    clip = np.stack(frames, axis=0)  # [T,H,W,C]
    t = torch.from_numpy(clip).float().div_(255.0).permute(3, 0, 1, 2).unsqueeze(0)

    boxes = gt[idxs].copy()
    boxes[:, [0, 2]] /= max(ow, 1)
    boxes[:, [1, 3]] /= max(oh, 1)
    boxes = np.clip(boxes, 0.0, 1.0)
    return t, boxes.astype(np.float32), valid[idxs], (ow, oh)


def iter_sequences(
    index_json: str,
    split: str = "val",
    frame_size: int = 256,
    max_frames: Optional[int] = None,
    max_seqs: Optional[int] = None,
    stride: int = 1,
) -> Iterator[Tuple[str, torch.Tensor, np.ndarray, np.ndarray]]:
    """Yield (name, clip, boxes, valid) for sequences in an index JSON."""
    with open(index_json, "r", encoding="utf-8") as f:
        index = json.load(f)
    records = index[split]
    if max_seqs is not None:
        records = records[:max_seqs]
    for rec in records:
        seq_dir = rec["dir"]
        clip, boxes, valid, _wh = load_sequence(seq_dir, frame_size, max_frames, stride)
        yield Path(seq_dir).name, clip, boxes, valid


# --------------------------------------------------------------------------
# clip dataset (training)
# --------------------------------------------------------------------------
class GOT10kClipDataset(Dataset):
    """Short clips + per-frame boxes from a GOT-10k index JSON.

    Each item is (clip [C,T,H,W] in [0,1], boxes [T,4] normalised xyxy).
    """

    def __init__(
        self,
        index_json: str,
        split: str = "train",
        num_frames: int = 8,
        frame_size: int = 256,
        temporal_stride: int = 3,
        train: bool = True,
    ):
        super().__init__()
        with open(index_json, "r", encoding="utf-8") as f:
            index = json.load(f)
        self.records: List[Dict] = index[split]
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.temporal_stride = temporal_stride
        self.train = train

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        import random

        rec = self.records[i]
        seq_dir = Path(rec["dir"])
        frame_paths = _list_frames(seq_dir)
        gt = _read_groundtruth(seq_dir / "groundtruth.txt")
        n = min(len(frame_paths), len(gt))
        span = self.num_frames * self.temporal_stride
        start_max = max(0, n - span)
        start = random.randint(0, start_max) if (self.train and start_max > 0) else start_max // 2
        idxs = [min(n - 1, start + k * self.temporal_stride) for k in range(self.num_frames)]

        first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
        oh, ow = (first.shape[0], first.shape[1]) if first is not None else (self.frame_size,) * 2

        frames = [_load_frame(frame_paths[j], self.frame_size) for j in idxs]
        clip = np.stack(frames, axis=0)  # [T,H,W,C]
        boxes = gt[idxs].astype(np.float64).copy()
        boxes[:, [0, 2]] /= max(ow, 1)
        boxes[:, [1, 3]] /= max(oh, 1)
        boxes = np.clip(boxes, 0.0, 1.0)

        t = torch.from_numpy(clip).float().div_(255.0).permute(3, 0, 1, 2).contiguous()
        return t, torch.from_numpy(boxes).float()


def collate_got10k(batch) -> Tuple[torch.Tensor, torch.Tensor]:
    clips = torch.stack([b[0] for b in batch], dim=0)   # [B,C,T,H,W]
    boxes = torch.stack([b[1] for b in batch], dim=0)   # [B,T,4]
    return clips, boxes
