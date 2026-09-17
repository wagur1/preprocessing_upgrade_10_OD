"""Top-k classification accuracy."""

from __future__ import annotations

from typing import Sequence

import torch


@torch.no_grad()
def topk_accuracy(
    logits: torch.Tensor, targets: torch.Tensor, ks: Sequence[int] = (1, 5)
) -> dict:
    """Return {f"top{k}": fraction correct} for each k."""
    maxk = max(ks)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)  # [B, maxk]
    correct = pred.eq(targets.view(-1, 1))
    out = {}
    for k in ks:
        out[f"top{k}"] = correct[:, :k].any(dim=1).float().mean().item()
    return out
