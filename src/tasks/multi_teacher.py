"""Multi-teacher analyzer ensemble (upgrade3, contribution A1).

The single most-cited weakness of standard-codec preprocessing for machines is
that the learned edit *overfits the one analyzer it was trained against*: works
such as "Preprocessing Enhanced Image Compression for Machine Vision"
(Lu et al., arXiv:2206.05650 / TCSVT 2024) only demonstrate NARROW held-out
transfer (across backbones of the *same* task family). A preprocessor that is
proven **analyzer-agnostic** -- trained against a panel of frozen teachers and
validated on a *held-out* analyzer it never saw -- is an open niche for the
standard-codec setup (learned-codec works like UG-ICM arXiv:2501.04579 and
All-in-One-Transfer arXiv:2504.12997 explore it, but they retrain the codec).

This module wraps several frozen :class:`TaskAnalyzer` backbones as ONE analyzer
so the existing training loop is unchanged:

    * ``accuracy_loss`` : aggregates the task loss across teachers. Two modes --
        - ``mean``   : average the loss over ALL teachers every step (stable,
                       Nx analyzer cost).
        - ``sample`` : draw ONE teacher per step (stochastic multi-teacher, ~1x
                       cost, acts as a regulariser so the edit cannot specialise
                       to a single network -- cf. multi-teacher distillation,
                       arXiv:2510.18680).
    * ``features``      : semantic features of the *active* teacher (the one
                       sampled in ``accuracy_loss``), so distillation stays
                       coherent within a step.

Only the preprocessor trains; every teacher is frozen. Generalisation is then
measured at eval against a backbone that is deliberately NOT in this panel
(``eval.held_out_backbone``), which is the A1 claim.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from .base import TaskAnalyzer


class MultiTeacherAnalyzer(TaskAnalyzer):
    """Panel of frozen analyzers presented through the single-analyzer API.

    Args:
        teachers: list of already-built, frozen :class:`TaskAnalyzer` instances
            (all sharing the same label space / task).
        sampling: ``"sample"`` (one teacher per step) or ``"mean"`` (all).
        weights: optional per-teacher loss weights (defaults to uniform).
    """

    def __init__(
        self,
        teachers: List[TaskAnalyzer],
        sampling: str = "sample",
        weights: List[float] | None = None,
    ):
        super().__init__()
        if not teachers:
            raise ValueError("MultiTeacherAnalyzer needs at least one teacher")
        if sampling not in ("sample", "mean"):
            raise ValueError("sampling must be 'sample' or 'mean'")
        self.teachers = nn.ModuleList(teachers)
        self.sampling = sampling
        self.task_name = teachers[0].task_name
        n = len(teachers)
        w = weights if weights is not None else [1.0] * n
        if len(w) != n:
            raise ValueError("weights must match number of teachers")
        s = float(sum(w)) or 1.0
        self.register_buffer("_w", torch.tensor([x / s for x in w]), persistent=False)
        self._active = 0  # teacher used for the current step's feature distillation
        self._pinned = False  # when True, accuracy_loss reuses _active (no re-sample)

    def freeze(self) -> "MultiTeacherAnalyzer":
        for t in self.teachers:
            t.freeze()
        self.eval()
        return self

    # -- per-step teacher locking -----------------------------------------
    def pin_active(self) -> int:
        """Sample & LOCK one teacher for the whole step (``sample`` mode).

        Call once at the top of a training step so that the task-saliency mask
        (A2), the task loss and the feature distillation all use the *same*
        teacher -- otherwise ``accuracy_loss`` re-samples on every call and the
        mask ends up computed on a different teacher than the loss it steers.
        No-op for ``mean`` (all teachers are used anyway)."""
        if self.sampling == "sample":
            self._active = int(torch.multinomial(self._w, 1).item())
        self._pinned = True
        return self._active

    def unpin_active(self) -> None:
        """Release the per-step lock; ``accuracy_loss`` samples freely again."""
        self._pinned = False

    # -- training ----------------------------------------------------------
    def accuracy_loss(
        self, x_hat: torch.Tensor, target: Any
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if self.sampling == "sample":
            # weighted-random teacher for this step; features() reuses _active.
            # If pinned (pin_active), reuse the locked teacher instead of drawing
            # a fresh one -- keeps saliency/loss/distill coherent within a step.
            if not self._pinned:
                self._active = int(torch.multinomial(self._w, 1).item())
            return self.teachers[self._active].accuracy_loss(x_hat, target)
        # mean: weighted average over every teacher (features use teacher 0).
        self._active = 0
        total = x_hat.new_zeros(())
        for wt, t in zip(self._w.tolist(), self.teachers):
            l, _ = t.accuracy_loss(x_hat, target)
            total = total + wt * l
        return total, {}

    # -- feature distillation ---------------------------------------------
    def features(self, x: torch.Tensor) -> list:
        """Features of the teacher that scored this step (coherent distillation)."""
        return self.teachers[self._active].features(x)

    # -- evaluation --------------------------------------------------------
    @torch.no_grad()
    def predict(self, x_hat: torch.Tensor) -> Any:
        return self.teachers[0].predict(x_hat)


def _demo() -> None:
    # lightweight stand-in teachers so the self-check needs no torchvision weights
    class _Toy(TaskAnalyzer):
        def __init__(self, k):
            super().__init__()
            self.task_name = "toy"
            self.lin = nn.Conv3d(3, 4, 1)
            self.k = k

        def accuracy_loss(self, x_hat, target):
            f = self.lin(x_hat).mean() * self.k
            return (f - target).pow(2), {}

        @torch.no_grad()
        def predict(self, x_hat):
            return self.lin(x_hat).mean()

        def features(self, x):
            return [self.lin(x)]

    torch.manual_seed(0)
    panel = MultiTeacherAnalyzer([_Toy(1.0), _Toy(2.0), _Toy(3.0)], sampling="sample")
    x = torch.rand(2, 3, 4, 16, 16, requires_grad=True)
    loss, _ = panel.accuracy_loss(x, torch.zeros(()))
    feats = panel.features(x)                       # must use the sampled teacher
    assert feats and feats[0].requires_grad
    loss.backward()
    assert x.grad is not None
    panel_mean = MultiTeacherAnalyzer([_Toy(1.0), _Toy(3.0)], sampling="mean")
    lm, _ = panel_mean.accuracy_loss(x.detach().requires_grad_(True), torch.zeros(()))
    assert torch.isfinite(lm)
    # pin_active locks one teacher across repeated accuracy_loss calls (A2 needs
    # saliency + loss on the SAME teacher within a step).
    big = MultiTeacherAnalyzer([_Toy(1.0), _Toy(2.0), _Toy(3.0), _Toy(4.0)], sampling="sample")
    big.pin_active()
    a0 = big._active
    for _ in range(20):
        big.accuracy_loss(x, torch.zeros(()))
        assert big._active == a0, "pinned teacher must not change across calls"
    big.unpin_active()
    changed = False
    for _ in range(50):
        big.accuracy_loss(x, torch.zeros(()))
        changed |= big._active != a0
    assert changed, "after unpin, teacher should vary again"
    print("multi_teacher self-check passed (sample + mean, coherent features, pin/unpin)")


if __name__ == "__main__":
    _demo()
