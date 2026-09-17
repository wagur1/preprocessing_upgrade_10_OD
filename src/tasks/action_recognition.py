"""Action-recognition analyzer (paper's primary task, Kinetics-400).

Uses a frozen torchvision video model pretrained on Kinetics-400 (default
``r3d_18``). The analyzer is never trained -- it only scores clips:

  * training : accuracy loss L_Acc = cross-entropy(logits, label).
  * eval     : top-k predictions for top-1..top-5 accuracy.

The pretrained weights carry the canonical 400-class ordering
(``weights.meta['categories']``). ``kinetics_category_index`` exposes a
name -> class-index map so the data layer can convert dataset folder names
(e.g. ``abseiling``) into the exact label index this frozen model expects --
which is what makes zero-shot accuracy on the frozen analyzer meaningful.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torchvision.models.video import (
    mc3_18,
    MC3_18_Weights,
    r2plus1d_18,
    R2Plus1D_18_Weights,
    r3d_18,
    R3D_18_Weights,
)

from .base import TaskAnalyzer

_BACKBONES = {
    "r3d_18": (r3d_18, R3D_18_Weights.KINETICS400_V1),
    "mc3_18": (mc3_18, MC3_18_Weights.KINETICS400_V1),
    "r2plus1d_18": (r2plus1d_18, R2Plus1D_18_Weights.KINETICS400_V1),
}

# Kinetics normalisation used by torchvision video weights.
_MEAN = (0.43216, 0.394666, 0.37645)
_STD = (0.22803, 0.22145, 0.216989)


def _canon(name: str) -> str:
    """Normalise a class name for matching (lowercase, alnum-joined)."""
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


@lru_cache(maxsize=None)
def kinetics_category_index(backbone: str = "r3d_18") -> Dict[str, int]:
    """Map canonicalised Kinetics-400 class name -> label index."""
    _, weights = _BACKBONES[backbone]
    cats: List[str] = weights.meta["categories"]
    return {_canon(c): i for i, c in enumerate(cats)}


def kinetics_categories(backbone: str = "r3d_18") -> List[str]:
    _, weights = _BACKBONES[backbone]
    return list(weights.meta["categories"])


class ActionRecognitionAnalyzer(TaskAnalyzer):
    """Frozen video classifier used as the machine-vision analyzer."""

    def __init__(self, backbone: str = "r3d_18", clip_size: int = 112):
        super().__init__()
        if backbone not in _BACKBONES:
            raise ValueError(f"unknown backbone '{backbone}'")
        ctor, weights = _BACKBONES[backbone]
        self.task_name = "action_recognition"
        self.backbone_name = backbone
        self.clip_size = clip_size
        self.net = ctor(weights=weights)
        self.register_buffer("mean", torch.tensor(_MEAN).view(1, 3, 1, 1, 1))
        self.register_buffer("std", torch.tensor(_STD).view(1, 3, 1, 1, 1))

    # -- input prep --------------------------------------------------------
    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        """[B,C,T,H,W] in [0,1] -> resized to clip_size, Kinetics-normalised."""
        b, c, t, h, w = x.shape
        if (h, w) != (self.clip_size, self.clip_size):
            x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            x = F.interpolate(
                x,
                size=(self.clip_size, self.clip_size),
                mode="bilinear",
                align_corners=False,
            )
            x = x.reshape(b, t, c, self.clip_size, self.clip_size)
            x = x.permute(0, 2, 1, 3, 4)
        return (x - self.mean) / self.std

    # -- training ----------------------------------------------------------
    def accuracy_loss(
        self, x_hat: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        logits = self.net(self._prep(x_hat))
        loss = F.cross_entropy(logits, target)
        return loss, {"logits": logits.detach()}

    # -- feature distillation ---------------------------------------------
    def features(self, x: torch.Tensor) -> list:
        """Frozen r3d_18 semantic features (stem, layer1, layer2) for distill."""
        h = self._prep(x)
        net = self.net
        feats = []
        h = net.stem(h)
        feats.append(h)
        h = net.layer1(h)
        feats.append(h)
        h = net.layer2(h)
        feats.append(h)
        return feats

    # -- evaluation --------------------------------------------------------
    @torch.no_grad()
    def predict(self, x_hat: torch.Tensor) -> torch.Tensor:
        return self.net(self._prep(x_hat))  # logits [B, 400]
