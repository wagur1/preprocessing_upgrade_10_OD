"""Optional adapter for the paper's exact trackers via *pytracking*.

The paper evaluates GOT-10k with KYS / DiMP / ATOM / PrDiMP from the
`pytracking <https://github.com/visionml/pytracking>`_ library. Those trackers
are strong but heavy to install, so they are **optional**: the default analyzer
is the self-contained :class:`~src.tasks.siamfc.SiamFCNet`. When pytracking is
installed and its network weights are available, this adapter exposes the same
``track_sequence(clip, init_box_norm) -> [T,4] normalised xyxy`` interface the
evaluation loop uses, so the paper's trackers drop straight in.

Install (see also ``scripts/install_pytracking.sh``)::

    git clone https://github.com/visionml/pytracking.git
    # follow its install: ninja-build, precise-roi-pooling, jpeg4py, etc.
    # download network weights into pytracking/networks/

Usage::

    from src.tasks.pytracking_adapter import PyTrackingTracker
    trk = PyTrackingTracker("dimp", "dimp50")     # or ("atom","default"), ("kys","default")
    boxes = trk.track_sequence(clip, init_box_norm)

then pass ``trk`` to the tracking evaluation in place of the SiamFC analyzer.
"""

from __future__ import annotations

import importlib
from typing import Optional

import numpy as np
import torch


def pytracking_available() -> bool:
    return importlib.util.find_spec("pytracking") is not None


def _clip_to_frames(clip: torch.Tensor) -> np.ndarray:
    """[1,C,T,H,W] in [0,1] -> [T,H,W,C] uint8 RGB."""
    arr = (clip.clamp(0, 1) * 255).round().byte().cpu().numpy()[0]  # [C,T,H,W]
    return np.transpose(arr, (1, 2, 3, 0))  # [T,H,W,C]


class PyTrackingTracker:
    """Thin wrapper around a pytracking tracker matching the SiamFC interface."""

    def __init__(self, name: str = "dimp", parameter: str = "dimp50"):
        if not pytracking_available():
            raise ImportError(
                "pytracking is not installed. Run scripts/install_pytracking.sh "
                "or see https://github.com/visionml/pytracking. The default "
                "self-contained SiamFC tracker needs no extra install."
            )
        from pytracking.evaluation import Tracker  # type: ignore

        self.name = name
        self.parameter = parameter
        self._tracker_def = Tracker(name, parameter)
        self._params = self._tracker_def.get_parameters()

    def track_sequence(self, clip: torch.Tensor, init_box_norm: np.ndarray) -> np.ndarray:
        frames = _clip_to_frames(clip)  # [T,H,W,C] uint8
        t, h, w, _ = frames.shape
        scale = np.array([w, h, w, h], dtype=np.float64)
        b = np.asarray(init_box_norm, dtype=np.float64) * scale  # xyxy px
        init_xywh = [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]

        tracker = self._tracker_def.create_tracker(self._params)
        boxes = np.zeros((t, 4), dtype=np.float64)
        boxes[0] = init_box_norm

        tracker.initialize(frames[0], {"init_bbox": init_xywh})
        for i in range(1, t):
            out = tracker.track(frames[i])
            x, y, bw, bh = out["target_bbox"]
            boxes[i] = [x / w, y / h, (x + bw) / w, (y + bh) / h]
        return np.clip(boxes, 0.0, 1.0)


def build_tracker(name: str = "dimp", parameter: Optional[str] = None) -> PyTrackingTracker:
    defaults = {"dimp": "dimp50", "atom": "default", "kys": "default", "prdimp": "prdimp50"}
    return PyTrackingTracker(name, parameter or defaults.get(name, "default"))
