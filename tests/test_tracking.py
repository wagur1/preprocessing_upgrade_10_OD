"""Checks for the optional Comet ML tracker (no comet_ml install needed)."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tracking import KEY_FILE, _Comet, _Noop, _default_name, _flatten, make_tracker


class _FakeExp:
    """Records metric/curve calls without touching the network."""

    def __init__(self):
        self.id = "fake-key-123"
        self.metrics = []
        self.curves = []
        self.ended = False

    def log_metric(self, name, value, step=None, epoch=None):
        self.metrics.append((name, value, step))

    def log_parameters(self, params):
        self.params = params

    def log_curve(self, name, xs, ys):
        self.curves.append((name, xs, ys))

    def end(self):
        self.ended = True


def test_tracking_defaults_noop_and_wrapper() -> None:
    # default experiment name: everything after "outputs", joined by "-"
    assert _default_name(Path("outputs/sweep_51/seed_0/mu3")) == "sweep_51-seed_0-mu3"
    assert _default_name(Path("/kaggle/working/repo/outputs/ste/seed_1")) == "ste-seed_1"
    # no "outputs" component -> last 3 parts
    assert _default_name(Path("/a/b/c/d/run")) == "c-d-run"
    assert _default_name(Path("outputs")) == "run"

    # flatten: nested dicts, lists, scalars
    flat = {}
    _flatten({"loss": {"mu": 10.0, "qp_list": [30, 35]}, "ok": True, "n": None}, "", flat)
    assert flat == {"loss.mu": 10.0, "loss.qp_list": "30,35", "ok": True, "n": None}

    # no API key -> silent no-op tracker
    env = os.environ
    saved = {k: env.get(k) for k in ("COMET_API_KEY", "COMET_MODE",
                                     "COMET_EXPERIMENT_NAME", "COMET_PROJECT_NAME")}
    try:
        env.pop("COMET_API_KEY", None)
        env.pop("COMET_MODE", None)
        t = make_tracker(Path("outputs/x"))
        assert isinstance(t, _Noop)
        t.log_params({"a": 1}); t.log_step(1, {"loss": 1.0}); t.log_epoch(1, {"v": 2.0})
        t.log_eval({"bd": -1.0}); t.log_curve("c", [1], [2]); t.finish()

        # disabled mode wins even with a key
        env["COMET_API_KEY"] = "dummy"
        env["COMET_MODE"] = "disabled"
        assert isinstance(make_tracker(Path("outputs/x")), _Noop)
    finally:
        for k, v in saved.items():
            if v is None:
                env.pop(k, None)
            else:
                env[k] = v

    # wrapper behaviour against a fake experiment: throttle + key file + skip None
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        exp = _FakeExp()
        tr = _Comet(exp, out, "unit-test")
        assert (out / KEY_FILE).read_text(encoding="utf-8") == "fake-key-123"
        tr.log_step(5, {"loss": 1.0})            # not a multiple of 10 -> dropped
        tr.log_step(10, {"loss": 2.0, "qp": 35})  # kept
        tr.log_step(20, {"loss": 3.0}, force=True)
        assert exp.metrics == [("loss", 2.0, 10), ("qp", 35, 10), ("loss", 3.0, 20)]
        tr.log_eval({"bd_rate": -5.0, "bd_acc": None})   # None skipped
        assert ("eval/bd_rate", -5.0, None) in exp.metrics
        assert not any(n == "eval/bd_acc" for n, _, _ in exp.metrics)
        tr.log_curve("ok", [0.1, 0.2], [0.5, 0.4])
        tr.log_curve("bad", [0.1], [0.5, 0.4])   # length mismatch -> dropped
        assert exp.curves == [("ok", [0.1, 0.2], [0.5, 0.4])]
        tr.finish({"best_val": 0.25})
        assert exp.ended
        assert ("eval/best_val", 0.25, None) in exp.metrics
    print("tracking self-check passed")


if __name__ == "__main__":
    test_tracking_defaults_noop_and_wrapper()
