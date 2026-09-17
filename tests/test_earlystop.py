"""Self-check for the early-stop decision (imports engine, so needs torch)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.engine import _earlystop_update


def main() -> None:
    inf = float("inf")

    # first observation always improves (best starts at +inf), resets counter.
    improved, best, ni, stop = _earlystop_update(1.0, inf, 1e-4, 0, 3)
    assert improved and best == 1.0 and ni == 0 and not stop

    # no improvement grows no_improve; no stop until it reaches patience.
    improved, best, ni, stop = _earlystop_update(1.0, 0.9, 1e-4, 0, 3)
    assert (not improved) and best == 0.9 and ni == 1 and not stop
    improved, best, ni, stop = _earlystop_update(1.0, 0.9, 1e-4, 2, 3)
    assert ni == 3 and stop, "must stop once no_improve reaches patience"

    # patience=0 disables early stopping.
    _, _, _, stop = _earlystop_update(1.0, 0.9, 1e-4, 99, 0)
    assert not stop

    # a drop smaller than min_delta does NOT count as improvement.
    improved, _, ni, _ = _earlystop_update(0.9 - 1e-6, 0.9, 1e-4, 0, 3)
    assert (not improved) and ni == 1

    print("early-stop self-check passed")


def test_earlystop_decisions():
    main()


import copy

import pytest
import torch

from src import engine


class TinyPre(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, clips, cond, mask=None):
        return clips * self.weight


class TinyCodec:
    qualities = [1]

    def __call__(self, clips, quality):
        return clips, clips.mean()


class NullTracker:
    def log_params(self, *args): pass
    def log_step(self, *args): pass
    def log_epoch(self, *args): pass
    def finish(self, *args): pass


@pytest.fixture
def fit_run(tmp_path, monkeypatch):
    def loss(analyzer, clips, x_hat, *args, **kwargs):
        value = x_hat.square().mean()
        return {key: value for key in (
            "loss", "loss_task", "loss_dist", "loss_rate", "loss_temp",
            "loss_delta", "loss_tv", "loss_tv_res", "loss_dct", "loss_dct3d")}

    monkeypatch.setattr(engine, "preprocessing_loss", loss)
    monkeypatch.setattr(engine, "make_tracker", lambda *a, **kw: NullTracker())
    base = {"out_dir": str(tmp_path), "model": {}, "codec": {},
            "train": {"epochs": 4, "cosine": True, "patience": 2,
                      "reg_warmup_frac": 0, "resume": True}}

    def run(values, **train):
        cfg = copy.deepcopy(base)
        cfg["train"].update(train)
        observations = iter(values) if values is not None else None
        monkeypatch.setattr(engine, "_val_loss", lambda *a: next(observations))
        pre = TinyPre()
        result = engine._fit(cfg, pre, TinyCodec(), None,
                             [torch.ones(1, 1, 1, 1, 1)],
                             [1] if values is not None else None,
                             lambda batch: (batch, None), "stub", 1)
        best = torch.load(result, weights_only=True)
        last = torch.load(tmp_path / "checkpoints/preprocessor_last.pth", weights_only=True)
        return best, last

    return run, tmp_path


def test_fit_saves_updated_best_and_patience_before_early_stop(fit_run):
    run, _ = fit_run
    best, last = run([1.0, 1.1, 1.2])
    assert (best["epoch"], best["best_val"], best["no_improve"]) == (1, 1.0, 0)
    assert (last["epoch"], last["best_val"], last["no_improve"]) == (3, 1.0, 2)
    assert last["global_step"] == 3
    assert last["sched"]["last_epoch"] == 3
    assert not torch.equal(best["model"]["weight"], last["model"]["weight"])


def test_fit_resume_preserves_historical_best_and_patience(fit_run):
    run, _ = fit_run
    best, last = run([1.0, 1.1], max_steps=2)
    assert (last["best_val"], last["no_improve"]) == (1.0, 1)
    resumed_best, resumed_last = run([1.2])
    assert resumed_last["epoch"] == 3  # restored patience stops after one epoch
    assert resumed_last["no_improve"] == 2
    assert resumed_best["epoch"] == 1
    assert torch.equal(resumed_best["model"]["weight"], best["model"]["weight"])


def test_fit_max_steps_still_validates_and_saves_current_state(fit_run):
    run, _ = fit_run
    best, last = run([0.75], max_steps=1)
    assert (last["epoch"], last["best_val"], last["no_improve"]) == (1, 0.75, 0)
    assert torch.equal(best["model"]["weight"], last["model"]["weight"])


def test_fit_no_validation_keeps_best_and_last(fit_run):
    run, _ = fit_run
    best, last = run(None, epochs=1)
    assert best["epoch"] == last["epoch"] == 1
    assert torch.equal(best["model"]["weight"], last["model"]["weight"])


@pytest.mark.parametrize("target_epochs", [1, 4])
def test_fit_resume_missing_best_fails_before_training_or_return(fit_run, target_epochs):
    run, root = fit_run
    run([1.0], max_steps=1)
    (root / "checkpoints/preprocessor.pth").unlink()
    with pytest.raises(ValueError, match="historical best checkpoint is missing"):
        run([], epochs=target_epochs)
    assert not (root / "checkpoints/preprocessor.pth").exists()


@pytest.mark.parametrize("saved_identity", [None, "other-split"])
def test_detection_resume_rejects_incompatible_split_identity(fit_run, saved_identity):
    run, root = fit_run
    _, last = run([1.0], max_steps=1)
    cfg = copy.deepcopy(last["cfg"])
    cfg["data"] = {"od_split_fingerprint": "current-split"}
    last["cfg"]["data"] = {"od_split_fingerprint": saved_identity}
    torch.save(last, root / "checkpoints/preprocessor_last.pth")
    with pytest.raises(ValueError, match="Incompatible OD checkpoint split identity"):
        engine._fit(cfg, TinyPre(), TinyCodec(), None, [1], [1],
                    lambda batch: (batch, None), "object_detection", 1)


if __name__ == "__main__":
    main()
