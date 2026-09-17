"""Video clip dataset.

Reads short RGB clips from an *index JSON* produced by ``prepare_3gb.py``.
The index is a list of ``{"path", "label", "class"}`` records per split, so the
dataset itself is decoupled from any particular directory layout and from the
3 GB capping logic.

A clip is ``num_frames`` frames sampled with a temporal stride, decoded with
OpenCV (reliable on Kaggle), resized to ``frame_size`` and returned as a float
tensor ``[C, T, H, W]`` in ``[0, 1]``. Everything downstream (preprocessor,
codec, analyzer) speaks this ``[B, C, T, H, W]`` layout.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

cv2.setNumThreads(0)  # avoid oversubscription inside DataLoader workers


class VideoClipDataset(Dataset):
    """Clips + labels from a prepared index JSON.

    Args:
        index_json: path to the JSON emitted by ``prepare_3gb.py``.
        split: which split to read ("train", "val", or "test").
        num_frames: frames per clip (T).
        frame_size: spatial size the clip is resized to (square).
        temporal_stride: gap between sampled frames.
        train: random temporal crop + flip if True, else centred + deterministic.
    """

    def __init__(
        self,
        index_json: str,
        split: str = "train",
        num_frames: int = 16,
        frame_size: int = 128,
        temporal_stride: int = 2,
        train: bool = True,
        return_metadata: bool = False,
    ):
        super().__init__()
        with open(index_json, "r", encoding="utf-8") as f:
            index = json.load(f)
        if split not in index:
            raise KeyError(f"split '{split}' not in index (have {list(index)})")
        self.samples: List[Dict] = index[split]
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.temporal_stride = temporal_stride
        self.train = train
        self.return_metadata = return_metadata

    def __len__(self) -> int:
        return len(self.samples)

    # -- frame decoding ----------------------------------------------------
    def _read_clip(self, path: str) -> np.ndarray:
        """Return ``[T, H, W, C]`` uint8 RGB, T == num_frames (padded if short).

        Frames are decoded **sequentially** with ``read()``/``grab()``. Seeking
        per frame with ``CAP_PROP_POS_FRAMES`` would force ffmpeg to jump to the
        nearest keyframe and re-decode forward for *every* frame -- pathological
        on network-backed storage like ``/kaggle/input`` (it turns one clip into
        ``num_frames`` random-read storms). One seek to the clip start is enough;
        intermediate frames are skipped with ``grab()``, which decodes but skips
        the colour conversion and copy.
        """
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        stride = max(1, self.temporal_stride)
        span = self.num_frames * stride

        start = 0
        if total > 0:
            start_max = max(0, total - span)
            start = random.randint(0, start_max) if self.train else start_max // 2
        if start > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)  # a single seek, not one per frame

        picked: List[np.ndarray] = []
        while len(picked) < self.num_frames:
            ok, fr = cap.read()
            if not ok:
                break
            picked.append(fr)
            for _ in range(stride - 1):  # cheap skip: no BGR->RGB, no copy
                if not cap.grab():
                    break
        cap.release()

        if not picked:
            return np.zeros(
                (self.num_frames, self.frame_size, self.frame_size, 3), np.uint8
            )
        while len(picked) < self.num_frames:  # short clip -> repeat last frame
            picked.append(picked[-1])

        out = []
        for fr in picked:
            fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            fr = cv2.resize(
                fr, (self.frame_size, self.frame_size), interpolation=cv2.INTER_AREA
            )
            out.append(fr)
        return np.stack(out, axis=0)

    # -- item --------------------------------------------------------------
    def __getitem__(self, i: int):
        rec = self.samples[i]
        clip = self._read_clip(rec["path"])  # [T,H,W,C] uint8
        if self.train and random.random() < 0.5:
            clip = clip[:, :, ::-1, :]  # horizontal flip
        t = torch.from_numpy(np.ascontiguousarray(clip)).float().div_(255.0)
        t = t.permute(3, 0, 1, 2).contiguous()  # [C,T,H,W]
        label = int(rec["label"])
        if not self.return_metadata:
            return t, label
        # Stable mount-independent clip id: the last two path components
        # ('<class>/<file>.mp4'), matching build_train_index.clip_key —
        # the merge across eval shards keys on this string.
        seq = str(rec.get("sequence_id", rec.get("video_id", rec.get("id", ""))))
        if not seq:
            parts = rec["path"].replace("\\", "/").rstrip("/").split("/")
            seq = "/".join(parts[-2:])
        metadata = {
            "sequence_id": seq,
            "path": str(rec["path"]),
            "class": str(rec.get("class", "")),
            "label": label,
        }
        return t, label, metadata


def collate_clips(batch):
    clips = torch.stack([b[0] for b in batch], dim=0)  # [B,C,T,H,W]
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    if len(batch[0]) >= 3:
        return clips, labels, [b[2] for b in batch]
    return clips, labels
