"""Checks for the upgrade2 loss terms (needs torch; no analyzer download)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.losses import (
    LossWeights,
    feature_distillation,
    preprocessing_loss,
    temporal_consistency,
    total_variation,
)


class _FakeAnalyzer:
    """Minimal analyzer stub: linear "features" + squared-error "task" loss."""

    def features(self, x):
        return [x.mean(dim=2)]  # collapse T -> [B,C,H,W]

    def accuracy_loss(self, x_hat, target):
        return ((x_hat - target) ** 2).mean(), {}


def main() -> None:
    a = _FakeAnalyzer()
    src = torch.rand(2, 3, 4, 8, 8)
    same = src.clone()
    other = torch.rand(2, 3, 4, 8, 8)

    # feature distillation: 0 when recon == source, positive otherwise.
    assert feature_distillation(a, src, same).item() < 1e-6
    assert feature_distillation(a, src, other).item() > 0

    # empty-feature analyzer -> distillation disabled (0), not a crash.
    class _NoFeat:
        def features(self, x):
            return []
    assert feature_distillation(_NoFeat(), src, other).item() == 0.0

    # temporal consistency: 0 when inter-frame deltas match; positive otherwise.
    assert temporal_consistency(src, same).item() < 1e-6
    assert temporal_consistency(src, other).item() > 0
    # single-frame clip -> defined as 0 (no temporal delta).
    assert temporal_consistency(src[:, :, :1], other[:, :, :1]).item() == 0.0

    # combined loss: no source-fidelity (MSE-to-source) term, weights applied.
    w = LossWeights(lam_task=1.0, omega=0.5, beta=0.1, tau=0.1)
    bpp = torch.tensor(0.3)
    parts = preprocessing_loss(a, src, other, bpp, target=other, w=w)
    # task term is 0 here (x_hat == target), so total = omega*dist + beta*bpp + tau*temp
    expect = (w.omega * feature_distillation(a, src, other)
              + w.beta * bpp + w.tau * temporal_consistency(src, other))
    assert torch.allclose(parts["loss"], expect, atol=1e-5)

    # delta term: 0 unless weighted AND x_pre given; then adds w.delta*|x_pre-x|.
    assert parts["loss_delta"].item() == 0.0
    wd = LossWeights(lam_task=1.0, omega=0.5, beta=0.1, tau=0.1, delta=2.0)
    x_pre = src + 0.25                       # constant 0.25 edit everywhere
    pd = preprocessing_loss(a, src, other, bpp, target=other, w=wd, x_pre=x_pre)
    assert abs(pd["loss_delta"].item() - 0.25) < 1e-6
    assert torch.allclose(pd["loss"], expect + wd.delta * 0.25, atol=1e-5)
    # delta weight 0 -> ignored even if x_pre passed.
    p0 = preprocessing_loss(a, src, other, bpp, target=other, w=w, x_pre=x_pre)
    assert p0["loss_delta"].item() == 0.0

    # tv term: 0 for a constant frame, positive for a varying one.
    assert total_variation(torch.ones(2, 3, 4, 8, 8)).item() == 0.0
    assert total_variation(other).item() > 0
    # gamma wiring: 0 unless weighted AND x_pre given; then adds w.gamma*TV(x_pre).
    assert parts["loss_tv"].item() == 0.0
    wg = LossWeights(lam_task=1.0, omega=0.5, beta=0.1, tau=0.1, gamma=3.0)
    pg = preprocessing_loss(a, src, other, bpp, target=other, w=wg, x_pre=other)
    tv = total_variation(other)
    assert abs(pg["loss_tv"].item() - tv.item()) < 1e-6
    assert torch.allclose(pg["loss"], expect + wg.gamma * tv, atol=1e-5)
    wm = LossWeights(lam_task=1.0, omega=0.5, beta=0.1, tau=0.1, mu=10.0)
    pm = preprocessing_loss(a, src, other, bpp, target=other, w=wm)
    ld = torch.nn.functional.mse_loss(other, src)
    assert abs(pm["loss_d"].item() - ld.item()) < 1e-6
    assert torch.allclose(pm["loss"], expect + wm.mu * ld, atol=1e-5)
    print("loss self-check passed")


if __name__ == "__main__":
    main()
