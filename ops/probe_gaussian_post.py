#!/usr/bin/env python
"""Paired A/B: does full-frame Gaussian POST help after R0 background suppression?

The codec round trips are shared exactly, per image/codec/QP:
  anchor  codec(x)                                -> detector
  prep    codec(PRE(x))   (mask blur, prep sigma) -> detector
  post    Gaussian(prep decode, post sigma)       -> detector   (no re-encode)

prep and post are scored on the byte-identical decoded tensor with identical
bpp (checked via recorded SHA-256 + pairing.jsonl), so a per-QP mAP difference
isolates the POST filter. Two-QP runs are smoke checks with no BD-rate claim.
Exploratory and not CTC-conformant; inspired by Otsuki & Nitta (IEVC 2026),
not a reproduction. Records are durable (journal + atomic snapshots); use a
fresh output directory, automatic resume is deliberately unsupported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.codecs.standard import StandardCodec, ffmpeg_available  # noqa: E402
from src.metrics.bd_rate import bd_rate  # noqa: E402
from src.models.mask_suppress import protect_mask, suppress  # noqa: E402
from probe_background_suppression import (  # noqa: E402
    Progress as BaseProgress, atomic_json, finite, load_coco, load_coco_ids,
    parse_image_ids,
)
from probe_detection import Detector, _coco_box, coco_map, scaled_gt  # noqa: E402

ARMS = ("anchor", "prep", "post")


def full_frame_gaussian(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Full-frame Gaussian blur (sigma 0 returns x), same kernel as the PRE."""
    return suppress(x, torch.zeros(x.shape[-2:], device=x.device), sigma)


def _decoded_source_sha(dec: torch.Tensor) -> str:
    return hashlib.sha256(
        dec.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def _check_decoded(dec: torch.Tensor, bpp: float, x: torch.Tensor) -> None:
    if (not math.isfinite(bpp) or bpp <= 0
            or tuple(dec.shape) != tuple(x.shape)
            or not torch.isfinite(dec).all()):
        raise ValueError("Invalid codec output")


def paired_predictions(x, mask, image_id, prep_sigma, post_sigma, codec, detector):
    """Run the three arms on shared codec round trips; see module docstring."""
    if x.ndim != 5 or not x.is_floating_point() or any(d == 0 for d in x.shape):
        raise ValueError("x must be nonempty floating [B,C,T,H,W]")
    if not torch.isfinite(x).all():
        raise ValueError("x must be finite")
    if (not math.isfinite(prep_sigma) or prep_sigma <= 0
            or not math.isfinite(post_sigma) or post_sigma <= 0):
        raise ValueError("prep/post sigma must be finite and positive")

    precoded = suppress(x, mask, prep_sigma)
    dec_anchor, bpp_anchor = codec.compress_decompress_items(x)
    _check_decoded(dec_anchor, float(bpp_anchor[0]), x)
    dec_prep, bpp_prep = codec.compress_decompress_items(precoded)
    _check_decoded(dec_prep, float(bpp_prep[0]), x)
    post_in = full_frame_gaussian(dec_prep, post_sigma)

    out = {}
    for arm, source, bpp, det_in in (
            ("anchor", dec_anchor, bpp_anchor[0], dec_anchor),
            ("prep", dec_prep, bpp_prep[0], dec_prep),
            ("post", dec_prep, bpp_prep[0], post_in)):
        d = detector.predict(det_in)[0]
        keep = d["scores"] >= detector.score_thresh
        preds = [dict(image_id=image_id, category_id=int(l), bbox=_coco_box(b), score=float(s))
                 for b, s, l in zip(d["boxes"][keep], d["scores"][keep], d["labels"][keep])]
        out[arm] = dict(bpp=float(bpp), predictions=preds,
                        decoded_source_sha256=_decoded_source_sha(source),
                        detector_input_sha256=_decoded_source_sha(det_in))
    return out


class Progress(BaseProgress):
    """Base durable records plus a per-image pairing journal."""

    def __init__(self, out_dir, metadata):
        super().__init__(out_dir, metadata)
        self.pairing = (out_dir / "pairing.jsonl").open("x", encoding="utf-8")

    def pair(self, entry):
        json.dump(finite(entry), self.pairing, allow_nan=False)
        self.pairing.write("\n")
        self.pairing.flush()
        os.fsync(self.pairing.fileno())


def run(a):
    qps = [int(q) for q in a.qps.split(",")]
    req_ids = parse_image_ids(a.image_ids)
    if len(qps) < 2 or len(set(qps)) != len(qps) or any(q < 0 or q > 51 for q in qps):
        raise ValueError("Use at least two distinct QPs in [0,51]")
    if (not math.isfinite(a.prep_sigma) or a.prep_sigma <= 0
            or not math.isfinite(a.post_sigma) or a.post_sigma <= 0):
        raise ValueError("prep/post sigma must be finite and positive")
    if a.size <= 0 or a.size % 2 or a.n_images <= 0:
        raise ValueError("size must be positive/even for yuv420p; n-images positive")
    if not math.isfinite(a.score) or not 0 <= a.score <= 1 or not math.isfinite(a.dilate) or a.dilate < 0:
        raise ValueError("score must be in [0,1]; dilation finite/nonnegative")
    if not ffmpeg_available():
        raise RuntimeError("ffmpeg/ffprobe not found on PATH")
    device = torch.device(a.device if a.device != "auto" else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(a.out)
    metadata = dict(size=a.size, qps=qps, prep_sigma=a.prep_sigma,
                    post_sigma=a.post_sigma, score=a.score, dilate=a.dilate,
                    device=str(device),
                    images=str(Path(a.images).resolve()), ann=str(Path(a.ann).resolve()),
                    image_selection="explicit_ids" if req_ids else "shuffled_seed_0",
                    requested_image_ids=req_ids,
                    mode="smoke" if len(qps) < 4 else "exploratory_rd",
                    held_out=False, ctc_conformant=False,
                    detector="torchvision FasterRCNN ResNet50 FPN COCO_V1",
                    codec_preset="medium",
                    protocol="single-frame RGB -> yuv420p; square resize; "
                             "PRE=detector-mask background blur before codec, "
                             "POST=full-frame Gaussian after decode (no re-encode)")
    prog = Progress(out_dir, metadata)
    try:
        prog.mark("loading_images")
        if req_ids:
            ann_meta, items = load_coco_ids(Path(a.images), Path(a.ann), req_ids, a.size)
        else:
            ann_meta, items = load_coco(Path(a.images), Path(a.ann), a.n_images, a.size, 0)
        if not items:
            raise ValueError("No images loaded; use --image-ids for a local subset")
        ids = [i for i, _, _, _ in items]
        gt = {i: scaled_gt(an, a.size, hw) for i, _, hw, an in items}
        metadata.update(image_ids=ids, n_images=len(items))
        prog.mark("loading_detector", **metadata)
        print(f"[post] {len(items)} images {ids}, {device}, QPs={qps}, "
              f"PRE sigma={a.prep_sigma}, POST sigma={a.post_sigma}", flush=True)
        det = Detector(device)
        masks = {}
        for i, t, _, _ in items:
            d = det.predict(t)[0]
            masks[i] = protect_mask(d["boxes"], d["scores"], d["labels"],
                                    a.size, a.score, a.dilate)
            prog.mark("masking", last_mask_image_id=i)
        cover = float(np.mean([m.mean().item() for m in masks.values()]))
        result = dict(metadata, cover=cover, curves={})
        records = {}
        # Complete each codec/QP cell for all arms before any COCO aggregation.
        for codec_name in ("h264", "h265"):
            for qp in qps:
                sc = StandardCodec(codec=codec_name, qp=qp, preset="medium")
                slots = {arm: {} for arm in ARMS}
                for i, t, _, _ in items:
                    t = t.to(device)
                    arms = paired_predictions(t, masks[i], i, a.prep_sigma,
                                              a.post_sigma, sc, det)
                    for arm in ARMS:
                        tag = f"{arm}_{codec_name}_{qp}"
                        slots[arm][i] = (arms[arm]["bpp"], arms[arm]["predictions"])
                        prog.record(tag, i, arms[arm]["bpp"], arms[arm]["predictions"])
                    prog.pair(dict(codec=codec_name, qp=qp, image_id=i,
                                   bpp_anchor=arms["anchor"]["bpp"],
                                   bpp_prep=arms["prep"]["bpp"],
                                   bpp_post=arms["post"]["bpp"],
                                   decoded_source_sha256_anchor=arms["anchor"]["decoded_source_sha256"],
                                   decoded_source_sha256_prep=arms["prep"]["decoded_source_sha256"],
                                   decoded_source_sha256_post=arms["post"]["decoded_source_sha256"],
                                   detector_input_sha256_post=arms["post"]["detector_input_sha256"],
                                   prep_post_match=(arms["prep"]["decoded_source_sha256"]
                                                    == arms["post"]["decoded_source_sha256"])))
                for arm in ARMS:
                    tag = f"{arm}_{codec_name}_{qp}"
                    records[tag] = slots[arm]
                    prog.cell(tag, slots[arm])
                    print(f"[post] saved {tag}: {len(slots[arm])} images", flush=True)
        # Records are already durable if aggregation is slow or fails.
        for codec_name in ("h264", "h265"):
            curves = result["curves"][codec_name] = {}
            for arm in ARMS:
                rates, aps = [], []
                for qp in qps:
                    tag = f"{arm}_{codec_name}_{qp}"
                    prog.mark("aggregating", current_cell=tag)
                    slot = records[tag]
                    rate = float(np.mean([v[0] for v in slot.values()]))
                    ap = coco_map([p for v in slot.values() for p in v[1]],
                                  gt, ids, ann_meta)[0]
                    rates.append(rate)
                    aps.append(ap)
                    atomic_json(out_dir / f"{tag}.json", dict(tag=tag, image_ids=ids,
                                rate=rate, mAP=ap, aggregation="complete"))
                curves[arm] = dict(rate=rates, mAP=aps)
                atomic_json(out_dir / "probe_gaussian_post.json", result)
            smoke = len(qps) < 4
            for arm in ("prep", "post"):
                bd = None
                reason = "requires at least four QPs; smoke only"
                if not smoke:
                    bd = bd_rate(curves["anchor"]["rate"], curves["anchor"]["mAP"],
                                 curves[arm]["rate"], curves[arm]["mAP"])
                    reason = ("exploratory" if math.isfinite(bd)
                              else "invalid or non-overlapping curves")
                curves[arm].update(bd_vs_anchor=bd, bd_status=reason)
            bd_pp = None
            pp_reason = "requires at least four QPs; smoke only"
            if not smoke:
                bd_pp = bd_rate(curves["prep"]["rate"], curves["prep"]["mAP"],
                                curves["post"]["rate"], curves["post"]["mAP"])
                pp_reason = ("exploratory" if math.isfinite(bd_pp)
                             else "invalid or non-overlapping curves")
            curves["post"].update(
                bd_vs_prep=bd_pp, bd_vs_prep_status=pp_reason,
                mAP_minus_prep_per_qp=[p - q for p, q in zip(curves["post"]["mAP"],
                                                             curves["prep"]["mAP"])])
            atomic_json(out_dir / "probe_gaussian_post.json", result)
        prog.mark("complete")
        print(f"[post] wrote {out_dir / 'probe_gaussian_post.json'}", flush=True)
        return result
    except BaseException as exc:
        prog.mark("failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        prog.journal.close()
        prog.pairing.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", required=True)
    ap.add_argument("--ann", required=True)
    ap.add_argument("--n-images", type=int, default=500)
    ap.add_argument("--image-ids", action="append", default=[],
                    help="Exact comma-separated COCO IDs; repeatable")
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--qps", default="30,35,40,45,50",
                    help="Two QPs allowed for smoke; BD requires four")
    ap.add_argument("--prep-sigma", type=float, default=4.0)
    ap.add_argument("--post-sigma", type=float, default=1.0)
    ap.add_argument("--score", type=float, default=0.5)
    ap.add_argument("--dilate", type=float, default=0.15)
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--out", default="outputs/probe_gaussian_post")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
