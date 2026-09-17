"""Probe restoration and diagnostic gating, with no real inference or codecs."""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "ops"))
import probe_detection as probe
from src import engine


class FakePre(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.conditions = []

    def forward(self, x, cond):
        self.conditions.append(float(cond.item()))
        return torch.zeros_like(x)

    def post_restore(self, x, cond):
        return x


@pytest.fixture
def harness(monkeypatch, tmp_path):
    pre = FakePre()
    stored = {"arch": "sandwich", "post_base": 8, "s_ch": 4, "editor_ch": 6,
              "cond_dim": 1, "qp_ref": [30, 50], "post_temporal": True,
              "dino_weight": 0.0, "motion_tau": 0.3, "w_budget": 0.2,
              "strength": 0.25, "custom_setting": {"enabled": False}}
    cfg = {"model": {k: None for k in stored}, "device": "cpu"}
    state = {"model": pre.state_dict(), "cfg": {"model": deepcopy(stored)}}
    built = []
    monkeypatch.setattr(probe, "load_config", lambda _: deepcopy(cfg))
    monkeypatch.setattr(torch, "load", lambda *a, **kw: state)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def build(config, device, role):
        built.append(deepcopy(config))
        assert role == "eval"
        return pre, None, None

    monkeypatch.setattr(probe, "_build_models", build)
    gt = {7: [{"id": 1, "image_id": 7, "category_id": 1, "iscrowd": 0,
               "area": 16, "bbox": [0, 0, 4, 4]}]}
    meta = {"categories": [{"id": 1, "name": "thing"}]}
    items = [(7, torch.ones(1, 3, 1, 8, 8), (8, 8), gt[7])]
    monkeypatch.setattr(probe, "_load_at", lambda *a: (meta, items, gt, [7]))

    class Detector:
        score_thresh = 0.05

        def __init__(self, device):
            pass

        def predict(self, x):
            return [{"boxes": torch.tensor([[0., 0., 4., 4.]]),
                     "scores": torch.tensor([float(x.max())]),
                     "labels": torch.tensor([1])}]

    coded = []

    class Codec:
        def __init__(self, codec, qp, preset):
            coded.append((codec, qp))

        def compress_decompress_items(self, x):
            return x, [0.1]

    monkeypatch.setattr(probe, "Detector", Detector)
    monkeypatch.setattr(probe, "StandardCodec", Codec)
    monkeypatch.setattr(probe, "ffmpeg_available", lambda: True)
    args = SimpleNamespace(images="unused", ann=None, config="unused", ckpt="unused",
                           n_images=1, size=8, stage_a_sizes=None, seed=19,
                           min_anchor_boxes=1, stage_a_threshold=None, stage="both",
                           qps="30,40,50", bootstrap=2, records=False, out=str(tmp_path))
    return args, pre, cfg, stored, state, built, coded


def test_full_checkpoint_settings_and_qp_ref_restored_before_build(harness):
    args, pre, cfg, stored, state, built, coded = harness
    result = probe.run(args)
    assert built[0]["model"] == stored
    assert state["cfg"]["model"] == stored
    assert pre.conditions == pytest.approx([0.5, 0, 0.5, 1, 0, 0.5, 1])
    assert not pre.training
    # Complete requested sweep despite zero stage-A PRE AP.
    assert result["stages"]["A"]["8"]["mAP_pre"] == 0
    assert result["verdict"] == "ran"
    assert coded == [(codec, qp) for codec in ("h264", "h265") for qp in (30, 40, 50)]
    for entry in result["stages"]["B"].values():
        assert set(entry["ci"]) == {"prep", "sandwich"}
        assert all(ci["seed"] == 19 and ci["n_invalid"] == 2
                   for ci in entry["ci"].values())


@pytest.mark.parametrize("bad", ["missing", "unexpected"])
def test_probe_strict_loading_rejects_incompatible_state(harness, bad):
    args, pre, cfg, stored, state, built, coded = harness
    if bad == "missing":
        state["model"] = {}
    else:
        state["model"]["unexpected"] = torch.ones(1)
    with pytest.raises(RuntimeError):
        probe.run(args)
    assert not coded


def test_legacy_checkpoint_uses_supplied_model_settings(harness):
    args, pre, cfg, stored, state, built, coded = harness
    cfg["model"] = deepcopy(stored)
    del state["cfg"]
    args.stage = "a"
    probe.run(args)
    assert built[0]["model"] == stored


def test_explicit_stage_a_threshold_remains_opt_in(harness):
    args, pre, cfg, stored, state, built, coded = harness
    args.stage_a_threshold = 0.85
    assert probe.run(args)["verdict"] == "stage_A_threshold_stop"
    assert not coded


def test_cli_threshold_default_disabled(monkeypatch, tmp_path):
    captured = []
    monkeypatch.setattr(sys, "argv", ["probe_detection", "--images", "unused",
                                     "--ckpt", "unused", "--out", str(tmp_path)])
    monkeypatch.setattr(probe, "run", lambda args: captured.append(args) or {})
    probe.main()
    assert captured[0].stage_a_threshold is None
    assert captured[0].stage == "both"


def test_engine_eval_restores_checkpoint_qp_ref(monkeypatch, tmp_path):
    pre = FakePre()
    state = {"model": pre.state_dict(), "cfg": {"model": {"qp_ref": [30, 50]}}}
    cfg = {"model": {"qp_ref": [20, 51]}, "device": "cpu",
           "task": {"name": "action_recognition"}}
    monkeypatch.setattr(torch, "load", lambda *a, **kw: state)

    def build(config, device, role):
        assert config["model"]["qp_ref"] == [30, 50]
        assert engine._qp_norm(40, config) == 0.5
        return pre, None, None

    monkeypatch.setattr(engine, "_build_models", build)
    monkeypatch.setattr(engine, "_evaluate_classification", lambda *a: {"ok": True})
    assert engine.evaluate(cfg, "unused", str(tmp_path)) == {"ok": True}
