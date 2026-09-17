"""Regression tests for exact selection, durable output, and offline pushing."""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ops"))
import probe_background_suppression as probe


def test_exact_selection(tmp_path):
    for i in (1, 2):
        Image.new("RGB", (12, 8), (30, 80, 120)).save(tmp_path / f"{i}.jpg")
    ann = tmp_path / "ann.json"
    ann.write_text(json.dumps(dict(images=[dict(id=i, file_name=f"{i}.jpg") for i in (1, 2, 3)], annotations=[])))
    _, items = probe.load_coco_ids(tmp_path, ann, [2, 1], 16)
    assert [v[0] for v in items] == [2, 1]
    assert items[0][1].shape == (1, 3, 1, 16, 16)
    assert items[0][2:] == ((8, 12), [])
    for ids in ([3], [999], [1, 3]):
        with pytest.raises(ValueError, match="missing"):
            probe.load_coco_ids(tmp_path, ann, ids, 16)
    assert probe.parse_image_ids(["2,1", "3"]) == [2, 1, 3]
    with pytest.raises(ValueError):
        probe.parse_image_ids(["1,1"])


def test_durable_records_before_aggregation(tmp_path, monkeypatch):
    class Detector:
        score_thresh = .05
        def __init__(self, device):
            pass
        def predict(self, x):
            return [dict(boxes=torch.empty(0, 4), scores=torch.empty(0), labels=torch.empty(0, dtype=torch.long))]
    class Codec:
        def __init__(self, **kwargs):
            pass
        def compress_decompress_items(self, x):
            return x, [.5]
    item = (1, torch.rand(1, 3, 1, 16, 16), (16, 16), [])
    monkeypatch.setattr(probe, "ffmpeg_available", lambda: True)
    monkeypatch.setattr(probe, "Detector", Detector)
    monkeypatch.setattr(probe, "StandardCodec", Codec)
    monkeypatch.setattr(probe, "load_coco_ids", lambda *a: ({}, [item]))
    def aggregate(*args):
        with np.load(tmp_path / "per_image_records.npz") as z:
            assert len(z.files) == 40
            assert z['anchor_h264_30_offsets'].tolist() == [0, 0]
            assert z['anchor_h264_30_boxes'].shape == (0, 5)
            assert z['anchor_h264_30_bpp'].tolist() == [.5]
        lines = (tmp_path / "records.jsonl").read_text().splitlines()
        assert len(lines) == 8 and json.loads(lines[0])["predictions"] == []
        assert len(json.loads((tmp_path / "progress.json").read_text())["completed_cells"]) == 8
        raise RuntimeError("aggregation interrupted")
    monkeypatch.setattr(probe, "coco_map", aggregate)
    a = argparse.Namespace(qps="30,40", sigmas="4", image_ids=["1"], size=16,
        n_images=1, score=.5, dilate=.15, device="cpu", out=str(tmp_path), images="unused", ann="unused")
    with pytest.raises(RuntimeError, match="aggregation interrupted"):
        probe.run(a)
    assert json.loads((tmp_path / "progress.json").read_text())["status"] == "failed"
    with pytest.raises(FileExistsError):
        probe.run(a)


@pytest.mark.parametrize("model", [False, True])
def test_pusher_dry_run(tmp_path, monkeypatch, model):
    spec = importlib.util.spec_from_file_location("push_bg_test", ROOT / "ops/push_detection_probe.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    script = "ops/probe_detection.py" if model else "ops/probe_background_suppression.py"
    monkeypatch.setattr(sys, "argv", ["push", "--commit", "abc123", "--script", script,
        "--dry-run", "--output-dir", str(tmp_path), "--extra-args", "--sigmas 4 --image-ids 139,285,632"])
    original = subprocess.run
    calls = []
    def local_only(cmd, **kwargs):
        assert cmd == ["bash", "-n"], "dry run attempted network or Kaggle"
        calls.append(cmd)
        return original(cmd, **kwargs)
    monkeypatch.setattr(mod.subprocess, "run", local_only)
    mod.main()
    assert len(calls) == 1
    nb = json.loads((tmp_path / "notebook.ipynb").read_text())
    source = "".join(nb["cells"][0]["source"])
    assert not re.search(r"__[A-Z][A-Z0-9_]*__", source)
    assert f"https://github.com/wagur1/{ROOT.name}.git" in source
    assert f"NEEDS_CKPT={int(model)}" in source
    meta = json.loads((tmp_path / "kernel-metadata.json").read_text())
    expected = ["awsaf49/coco-2017-dataset"]
    if model:
        expected += ["hieusunday0412/u8-bigpost-s1-ckpt"]
    assert meta["dataset_sources"] == expected
