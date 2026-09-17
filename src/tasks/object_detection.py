"""Object detection analyzer: a frozen COCO detector as the machine task.

Third task of the project alongside action recognition and tracking. The
analyzer is a pretrained Faster R-CNN (torchvision COCO weights by default);
the CTC evaluates VCM proposals with Detectron2's Faster R-CNN X101-FPN, so
``backbone`` selects the torchvision variant available without a Detectron2
install and this stands in for it.

Two subtleties worth stating:

* ``accuracy_loss`` needs the detector's *training-mode* loss, which requires
  targets. That would normally put the network in train mode and let its
  BatchNorm running statistics drift — silently changing the "frozen" analyzer
  over a long run. The model is therefore kept in train mode with every
  BatchNorm forced back to eval.
* The loss is differentiable w.r.t. the input image, which is what makes it
  usable as ``L_Acc`` for the preprocessor; the parameters themselves are
  frozen by ``TaskAnalyzer.freeze``.

Input is [B,C,T,H,W] like every other analyzer; detection is single-frame, so
the temporal axis is squeezed and must be 1.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from .base import TaskAnalyzer

# COCO's 80 categories keep their original ids (torchvision's label space).
_CATEGORIES = [c for c in
               __import__("torchvision.models.detection",
                          fromlist=["FasterRCNN_ResNet50_FPN_Weights"])
               .FasterRCNN_ResNet50_FPN_Weights.COCO_V1.meta["categories"]
               if c != "N/A"]


class ObjectDetectionAnalyzer(TaskAnalyzer):
    """Frozen COCO detector; L_Acc is its own detection loss on the coded image."""

    task_name = "object_detection"

    def __init__(self, backbone: str = "fasterrcnn_resnet50_fpn",
                 score_thresh: float = 0.05, size: int = 320):
        super().__init__()
        self.backbone_name = backbone
        self.score_thresh = float(score_thresh)
        self.size = int(size)
        self.detector = self._build(backbone)
        self.detector.eval()
        self.freeze()
        self._lock_bn()

    @staticmethod
    def _build(name: str) -> nn.Module:
        from torchvision.models import detection as D
        table = {
            "fasterrcnn_resnet50_fpn": (D.fasterrcnn_resnet50_fpn,
                                        D.FasterRCNN_ResNet50_FPN_Weights.COCO_V1),
            "fasterrcnn_mobilenet_v3_large_fpn": (
                D.fasterrcnn_mobilenet_v3_large_fpn,
                D.FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1),
        }
        if name not in table:
            raise ValueError(f"unsupported detector '{name}'; have {list(table)}")
        fn, weights = table[name]
        return fn(weights=weights)

    def _lock_bn(self) -> None:
        """Keep BN statistics frozen even while the loss needs train mode."""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad_(False)

    @staticmethod
    def _frames(x_hat: torch.Tensor) -> torch.Tensor:
        if x_hat.ndim == 5:
            if x_hat.shape[2] != 1:
                raise ValueError(
                    "object detection is single-frame; got a clip with "
                    f"T={x_hat.shape[2]}")
            x_hat = x_hat[:, :, 0]
        return x_hat

    def accuracy_loss(self, x_hat: torch.Tensor,
                      target: Any) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Sum of the detector's losses on the coded frames vs the gt boxes."""
        frames = self._frames(x_hat)
        targets = target if isinstance(target, list) else [target] * frames.shape[0]
        self.detector.train()
        self._lock_bn()
        losses = self.detector(frames, targets)
        total = sum(v for v in losses.values())
        self.detector.eval()
        return total, {"det": {k: float(v.detach()) for k, v in losses.items()}}

    @torch.no_grad()
    def predict(self, x_hat: torch.Tensor):
        frames = self._frames(x_hat)
        self.detector.eval()
        return self.detector(frames, None)

    def features(self, x: torch.Tensor) -> list:
        """Backbone feature maps (kept empty: distillation is unused here)."""
        return []
