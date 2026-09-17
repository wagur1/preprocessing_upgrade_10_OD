"""Object-tracking analyzer (paper's second task, GOT-10k).

Wraps the self-contained :class:`SiamFCNet` (a real Siamese tracker on a frozen
ImageNet backbone) as a :class:`TaskAnalyzer`, so tracking plugs into the exact
same framework as action recognition:

  * training : ``accuracy_loss`` = SiamFC balanced logistic loss on the
    cross-correlation response, computed on the *compressed* clip -- the
    preprocessor learns to keep the target localisable after coding.
  * evaluation: ``track`` runs full multi-scale SiamFC inference over a
    sequence to produce per-frame boxes, from which GOT-10k AUC / AO are
    computed (see ``src/metrics/tracking_auc.py``).

The analyzer is frozen; only the preprocessor is trained. For the paper's exact
KYS/DiMP/ATOM/PrDiMP numbers, use ``src/tasks/pytracking_adapter.py`` in place
of this SiamFC tracker at evaluation.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch

from .base import TaskAnalyzer
from .siamfc import SiamFCNet


class TrackingAnalyzer(TaskAnalyzer):
    """Frozen SiamFC tracker used as the machine-vision analyzer for tracking."""

    def __init__(
        self,
        backbone: str = "resnet18",
        exemplar_size: int = 127,
        search_size: int = 255,
        pos_radius: float = 2.0,
    ):
        super().__init__()
        self.task_name = "tracking"
        self.net = SiamFCNet(
            backbone=backbone,
            exemplar_size=exemplar_size,
            search_size=search_size,
            pos_radius=pos_radius,
        )

    # -- training ----------------------------------------------------------
    def accuracy_loss(
        self, x_hat: torch.Tensor, target: Any
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """target: {"boxes": [B,T,4] normalised xyxy} (or the boxes tensor)."""
        boxes = target["boxes"] if isinstance(target, dict) else target
        loss = self.net.training_loss(x_hat, boxes)
        return loss, {}

    # -- feature distillation ---------------------------------------------
    def features(self, x: torch.Tensor) -> list:
        """Frozen SiamFC backbone features on full frames (for distillation)."""
        b, c, t, h, w = x.shape
        frames = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        return [self.net.embed(frames)]

    # -- evaluation --------------------------------------------------------
    @torch.no_grad()
    def track(self, clip: torch.Tensor, init_box_norm: np.ndarray) -> np.ndarray:
        """Run inference over a clip [1,C,T,H,W]; return [T,4] normalised boxes."""
        return self.net.track_sequence(clip, init_box_norm)

    @torch.no_grad()
    def predict(self, x_hat: torch.Tensor) -> torch.Tensor:
        """Return backbone features of each frame (generic hook)."""
        b, c, t, h, w = x_hat.shape
        frames = x_hat.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        return self.net.embed(frames)
