"""Contract tests for the R0 Gaussian-POST A/B probe.

The experiment's validity rests on one invariant: the PRE-only and PRE+POST
arms must be scored on the byte-identical decoded tensor with identical bpp,
so any mAP difference is attributable to the POST filter alone. These tests
pin that invariant with a deterministic fake codec and a spy detector.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ops"))
import probe_gaussian_post as probe  # noqa: E402


class SpyDetector:
    score_thresh = .05

    def __init__(self, device=None):
        self.seen = []

    def predict(self, x):
        self.seen.append(x.detach().clone())
        return [dict(boxes=torch.empty(0, 4), scores=torch.empty(0),
                     labels=torch.empty(0, dtype=torch.long))]


class FakeCodec:
    """Deterministic lossy stand-in: decoded pixels visibly differ from input."""

    def __init__(self, **kwargs):
        self.calls = 0
        self.inputs = []

    def compress_decompress_items(self, x):
        self.calls += 1
        self.inputs.append(x.detach().clone())
        return (x + .25).clamp(0, 1), [.4]


class CountingCodec(FakeCodec):
    total = 0

    def compress_decompress_items(self, x):
        CountingCodec.total += 1
        return super().compress_decompress_items(x)


def test_paired_arms_share_decoded_source_and_bits():
    torch.manual_seed(0)
    x = torch.rand(1, 3, 1, 16, 16)
    mask = torch.zeros(1, 1, 16, 16)
    mask[0, 0, :8, :] = 1
    codec, det = FakeCodec(), SpyDetector()

    arms = probe.paired_predictions(x, mask, 1, 4.0, 1.0, codec, det)

    assert codec.calls == 2
    assert arms["prep"]["bpp"] == arms["post"]["bpp"] == .4
    assert arms["prep"]["decoded_source_sha256"] == arms["post"]["decoded_source_sha256"]
    assert arms["anchor"]["decoded_source_sha256"] != arms["prep"]["decoded_source_sha256"]
    assert arms["anchor"]["bpp"] == .4
    assert all(arms[a]["predictions"] == [] for a in arms)
    assert len(det.seen) == 3

    from src.models.mask_suppress import suppress
    from torchvision.transforms.functional import gaussian_blur as tv_blur
    # Observation order must be: anchor decode, R0 decode, Gaussian(R0 decode).
    assert torch.equal(det.seen[0], (x + .25).clamp(0, 1))
    assert torch.equal(det.seen[1], suppress(x, mask, 4.0).add(.25).clamp(0, 1))
    expected = tv_blur(det.seen[1][:, :, 0], [5, 5], [1., 1.]).unsqueeze(2)
    assert torch.allclose(det.seen[2], expected, atol=1e-5)
    assert not torch.allclose(det.seen[2], det.seen[1])
    # Encoded inputs were exactly the anchor source and the PRE tensor, unmutated.
    assert torch.equal(codec.inputs[0], x)
    assert torch.equal(codec.inputs[1], suppress(x, mask, 4.0))
    sha = hashlib.sha256(det.seen[1].contiguous().numpy().tobytes()).hexdigest()
    assert arms["prep"]["decoded_source_sha256"] == sha


def test_full_frame_gaussian_matches_torchvision():
    from torchvision.transforms.functional import gaussian_blur as tv_blur
    torch.manual_seed(1)
    x = torch.rand(2, 3, 1, 24, 20)
    ref = tv_blur(x[:, :, 0], [5, 5], [1., 1.]).unsqueeze(2)
    assert torch.allclose(probe.full_frame_gaussian(x, 1.0), ref, atol=1e-5)
    assert probe.full_frame_gaussian(x, 0.0).equal(x)


@pytest.mark.parametrize("bad", ["nan", "shape", "zero_bpp", "inf_bpp"])
def test_invalid_codec_output_rejected_before_detector(bad):
    torch.manual_seed(2)
    x = torch.rand(1, 3, 1, 16, 16)
    det = SpyDetector()

    class Bad(FakeCodec):
        def compress_decompress_items(self, x):
            dec, bpp = super().compress_decompress_items(x)
            if bad == "nan":
                dec = dec * float("nan")
            elif bad == "shape":
                dec = dec[:, :, :, :8, :]
            elif bad == "zero_bpp":
                bpp = [0.0]
            else:
                bpp = [float("inf")]
            return dec, bpp

    with pytest.raises(ValueError, match="Invalid codec output"):
        probe.paired_predictions(x, torch.zeros(1, 1, 16, 16), 1, 4.0, 1.0, Bad(), det)
    assert det.seen == []


@pytest.mark.parametrize("prep,post", [(0.0, 1.0), (float("nan"), 1.0),
                                       (4.0, -1.0), (4.0, float("nan"))])
def test_invalid_sigmas_rejected(prep, post):
    x = torch.rand(1, 3, 1, 16, 16)
    with pytest.raises(ValueError):
        probe.paired_predictions(x, torch.zeros(1, 1, 16, 16), 1, prep, post,
                                 FakeCodec(), SpyDetector())


def _item():
    torch.manual_seed(3)
    return (1, torch.rand(1, 3, 1, 16, 16), (16, 16), [])


def _args(tmp_path, qps="30,40"):
    return argparse.Namespace(images="unused", ann="unused", image_ids=["1"],
                              n_images=1, size=16, qps=qps, prep_sigma=4.0,
                              post_sigma=1.0, score=.5, dilate=.15, device="cpu",
                              out=str(tmp_path))


def test_cli_records_paired_cells(tmp_path, monkeypatch):
    CountingCodec.total = 0
    monkeypatch.setattr(probe, "ffmpeg_available", lambda: True)
    monkeypatch.setattr(probe, "Detector", SpyDetector)
    monkeypatch.setattr(probe, "StandardCodec", CountingCodec)
    monkeypatch.setattr(probe, "load_coco_ids", lambda *a: ({}, [_item()]))
    monkeypatch.setattr(probe, "coco_map", lambda *a, **k: (0.25, 0.5))

    probe.run(_args(tmp_path))

    with np.load(tmp_path / "per_image_records.npz") as z:
        assert len(z.files) == 60  # 2 codecs x 2 QPs x 3 arms x 5 arrays
        assert z["anchor_h264_30_img"].tolist() == [1]
        assert z["prep_h265_40_bpp"].tolist() == z["post_h265_40_bpp"].tolist()
        assert z["prep_h265_40_bpp"].item() == pytest.approx(0.4)
        assert z["anchor_h264_30_boxes"].shape == (0, 5)
    lines = (tmp_path / "records.jsonl").read_text().splitlines()
    assert len(lines) == 12
    assert {json.loads(l)["tag"] for l in lines} == {
        f"{arm}_{c}_{q}" for arm in ("anchor", "prep", "post")
        for c in ("h264", "h265") for q in (30, 40)}
    pairing = [json.loads(l) for l in (tmp_path / "pairing.jsonl").read_text().splitlines()]
    assert len(pairing) == 4
    assert all(p["prep_post_match"] and p["bpp_prep"] == p["bpp_post"] for p in pairing)
    assert json.loads((tmp_path / "progress.json").read_text())["status"] == "complete"
    result = json.loads((tmp_path / "probe_gaussian_post.json").read_text())
    assert result["mode"] == "smoke" and result["cover"] == 0.0
    assert CountingCodec.total == 8  # exactly two codec round trips per image/QP


def test_cli_bd_wiring_and_per_qp_delta(tmp_path, monkeypatch):
    CountingCodec.total = 0
    monkeypatch.setattr(probe, "ffmpeg_available", lambda: True)
    monkeypatch.setattr(probe, "Detector", SpyDetector)
    monkeypatch.setattr(probe, "StandardCodec", CountingCodec)
    monkeypatch.setattr(probe, "load_coco_ids", lambda *a: ({}, [_item()]))
    monkeypatch.setattr(probe, "coco_map", lambda *a, **k: (0.25, 0.5))

    bd_calls = []

    def fake_bd(rate_anchor, m_anchor, rate_test, m_test):
        bd_calls.append((list(rate_anchor), list(m_anchor), list(rate_test), list(m_test)))
        return -5.0

    monkeypatch.setattr(probe, "bd_rate", fake_bd)

    probe.run(_args(tmp_path, qps="30,35,40,45,50"))

    # Per codec: prep vs anchor, post vs anchor, post vs prep.
    assert len(bd_calls) == 6
    for rate_a, m_a, rate_t, m_t in bd_calls:
        assert rate_a == rate_t == [.4] * 5
        assert m_a == m_t == [.25] * 5
    result = json.loads((tmp_path / "probe_gaussian_post.json").read_text())
    for codec in ("h264", "h265"):
        curve = result["curves"][codec]
        assert curve["prep"]["bd_vs_anchor"] == -5.0
        assert curve["post"]["bd_vs_anchor"] == -5.0
        assert curve["post"]["bd_vs_prep"] == -5.0
        assert curve["post"]["mAP_minus_prep_per_qp"] == [0.0] * 5
        assert "bd_vs_anchor" not in curve["anchor"]


def test_cli_failure_is_durable_and_rerun_refused(tmp_path, monkeypatch):
    CountingCodec.total = 0

    class Boom(CountingCodec):
        def compress_decompress_items(self, x):
            if CountingCodec.total >= 2:
                raise RuntimeError("codec died")
            return super().compress_decompress_items(x)

    monkeypatch.setattr(probe, "ffmpeg_available", lambda: True)
    monkeypatch.setattr(probe, "Detector", SpyDetector)
    monkeypatch.setattr(probe, "StandardCodec", Boom)
    monkeypatch.setattr(probe, "load_coco_ids", lambda *a: ({}, [_item()]))
    monkeypatch.setattr(probe, "coco_map", lambda *a, **k: ([0.25], [0.5]))

    with pytest.raises(RuntimeError, match="codec died"):
        probe.run(_args(tmp_path))
    assert json.loads((tmp_path / "progress.json").read_text())["status"] == "failed"
    with pytest.raises(FileExistsError):
        probe.run(_args(tmp_path))


@pytest.mark.parametrize("field,value", [("size", 15), ("n_images", 0),
                                         ("prep_sigma", 0.0), ("post_sigma", -1.0)])
def test_cli_invalid_options(tmp_path, monkeypatch, field, value):
    kwargs = dict(images="unused", ann="unused", image_ids=["1"], n_images=1,
                  size=16, qps="30,40", prep_sigma=4.0, post_sigma=1.0,
                  score=.5, dilate=.15, device="cpu", out=str(tmp_path))
    kwargs[field] = value
    monkeypatch.setattr(probe, "ffmpeg_available", lambda: True)
    with pytest.raises(ValueError):
        probe.run(argparse.Namespace(**kwargs))


def test_cli_requires_ffmpeg(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "ffmpeg_available", lambda: False)
    with pytest.raises(RuntimeError, match="ffmpeg"):
        probe.run(_args(tmp_path))
