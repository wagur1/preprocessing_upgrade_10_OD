#!/usr/bin/env python
"""Dual-region probe: what is the mask worth, and what is the ROI worth?

Four single-variable arms against the plain-codec anchor, all taken from the
published baselines rather than invented here:

  anchor      codec(x)                          the anchor every paper uses
  r0blur      blur the background (sigma 4)     R0, re-measured as a cross-check
  mask        constant 0.5 fill in background   Li & Rhee's "NROI MASK"
  globalblur  blur the whole frame (sigma 4)    Shirahase/Otsuki low-pass floor
  roiblur     fill background + blur ROI (s 1)  Li & Rhee's winning variant

`globalblur` is the ablation no paper in this family reports: it separates the
contribution of *the mask* from the contribution of *a low-pass*. `mask` and
`roiblur` change the RATE side (applied before encoding), unlike the 0-bit POST
probe (ops/probe_gaussian_post.py).

The mask is built from the frozen detector on the original frame, exactly as in
probe_background_suppression. Same records/progress format, so the offline
analysis carries over. Exploratory, not CTC-conformant. Records are durable
(journal + atomic snapshots); use a fresh output directory, automatic resume is
deliberately unsupported.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.codecs.standard import StandardCodec, ffmpeg_available  # noqa: E402
from src.metrics.bd_rate import bd_rate  # noqa: E402
from src.models.mask_suppress import fill_outside, protect_mask, suppress  # noqa: E402
from probe_background_suppression import (  # noqa: E402
    Progress, atomic_json, load_coco, load_coco_ids, parse_image_ids,
)
from probe_detection import Detector, _coco_box, coco_map, scaled_gt  # noqa: E402

ARMS = ("anchor", "r0blur", "mask", "globalblur", "roiblur")
FIELDS = ("img", "bpp", "boxes", "labels", "offsets")


def _zero_mask(mask):
    return torch.zeros_like(mask)


def arm_r0blur(x, mask, p):
    return suppress(x, mask, p["r0_sigma"])


def arm_mask(x, mask, p):
    return fill_outside(x, mask, p["fill"])


def arm_globalblur(x, mask, p):
    return suppress(x, _zero_mask(mask), p["r0_sigma"])


def arm_roiblur(x, mask, p):
    blurred = suppress(x, _zero_mask(mask), p["roi_sigma"])
    return fill_outside(blurred, mask, p["fill"])


ARM_FNS = {"r0blur": arm_r0blur, "mask": arm_mask,
           "globalblur": arm_globalblur, "roiblur": arm_roiblur}


def run(a):
    qps = [int(q) for q in a.qps.split(",")]
    req_ids = parse_image_ids(a.image_ids)
    if len(qps) < 2 or len(set(qps)) != len(qps) or any(q < 0 or q > 51 for q in qps):
        raise ValueError("Use at least two distinct QPs in [0,51]")
    if a.size <= 0 or a.size % 2 or a.n_images <= 0:
        raise ValueError("size must be positive/even for yuv420p; n-images positive")
    for name, value in (("score", a.score), ("dilate", a.dilate),
                        ("r0-sigma", a.r0_sigma), ("roi-sigma", a.roi_sigma)):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if not 0 <= a.score <= 1:
        raise ValueError("score must be in [0,1]")
    if not math.isfinite(a.fill) or not 0 <= a.fill <= 1:
        raise ValueError("fill must be finite and in [0,1]")
    if not ffmpeg_available():
        raise RuntimeError("ffmpeg/ffprobe not found on PATH")
    device = torch.device(a.device if a.device != "auto" else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(a.out)
    params = dict(r0_sigma=a.r0_sigma, roi_sigma=a.roi_sigma, fill=a.fill)
    metadata = dict(size=a.size, qps=qps, score=a.score, dilate=a.dilate,
                    device=str(device),
                    images=str(Path(a.images).resolve()), ann=str(Path(a.ann).resolve()),
                    image_selection="explicit_ids" if req_ids else "shuffled_seed_0",
                    requested_image_ids=req_ids, mode="smoke" if len(qps) < 4 else "exploratory_rd",
                    held_out=False, ctc_conformant=False,
                    detector="torchvision FasterRCNN ResNet50 FPN COCO_V1",
                    codec_preset="medium",
                    arm_params=params,
                    protocol="single-frame RGB -> yuv420p; square resize; "
                             "r0blur=background blur; mask=constant-fill background; "
                             "globalblur=full-frame blur, no mask; "
                             "roiblur=constant-fill background + ROI blur BEFORE encode")
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
        print(f"[dual] {len(items)} images {ids[:3]}..., {device}, QPs={qps}, "
              f"arms={ARMS}, params={params}", flush=True)
        det = Detector(device)
        masks = {}
        for i, t, _, _ in items:
            d = det.predict(t)[0]
            masks[i] = protect_mask(d["boxes"], d["scores"], d["labels"], a.size, a.score, a.dilate)
            prog.mark("masking", last_mask_image_id=i)
        cover = float(np.mean([m.mean().item() for m in masks.values()]))
        result = dict(metadata, cover=cover, curves={})
        records = {}
        # Complete each codec/QP/arm cell, saving it before any COCO aggregation.
        for codec in ("h264", "h265"):
            for qp in qps:
                sc = StandardCodec(codec=codec, qp=qp, preset="medium")
                for arm in ARMS:
                    tag = f"{arm}_{codec}_{qp}"
                    slot = records[tag] = {}
                    for i, t, _, _ in items:
                        t = t.to(device)
                        if arm == "anchor":
                            xv = t
                        else:
                            xv = ARM_FNS[arm](t, masks[i], params)
                            if tuple(xv.shape) != tuple(t.shape) or not torch.isfinite(xv).all():
                                raise ValueError(f"{arm}: invalid transform output")
                        decoded, bpps = sc.compress_decompress_items(xv)
                        if not math.isfinite(bpps[0]) or bpps[0] <= 0 or not torch.isfinite(decoded).all():
                            raise ValueError("Invalid codec output")
                        d = det.predict(decoded)[0]
                        keep = d["scores"] >= det.score_thresh
                        preds = [dict(image_id=i, category_id=int(l), bbox=_coco_box(b), score=float(s))
                                 for b, s, l in zip(d["boxes"][keep], d["scores"][keep], d["labels"][keep])]
                        slot[i] = (float(bpps[0]), preds)
                        prog.record(tag, i, float(bpps[0]), preds)
                    prog.cell(tag, slot)
                    print(f"[dual] saved {tag}: {len(slot)} images", flush=True)
        # Records are already durable if aggregation is slow or fails;
        # Progress.cell keeps per_image_records.npz current after every cell.
        for codec in ("h264", "h265"):
            curves = result["curves"][codec] = {}
            for arm in ARMS:
                rates, aps = [], []
                for qp in qps:
                    tag = f"{arm}_{codec}_{qp}"
                    prog.mark("aggregating", current_cell=tag)
                    slot = records[tag]
                    rate = float(np.mean([v[0] for v in slot.values()]))
                    ap = coco_map([p for v in slot.values() for p in v[1]], gt, ids, ann_meta)[0]
                    rates.append(rate)
                    aps.append(ap)
                    atomic_json(out_dir / f"{tag}.json", dict(tag=tag, image_ids=ids,
                                rate=rate, mAP=ap, aggregation="complete"))
                curves[arm] = dict(rate=rates, mAP=aps)
                atomic_json(out_dir / "probe_dual_region.json", result)
            for arm in ARMS[1:]:
                bd = None
                reason = "requires at least four QPs; smoke only"
                if len(qps) >= 4:
                    bd = bd_rate(curves["anchor"]["rate"], curves["anchor"]["mAP"],
                                 curves[arm]["rate"], curves[arm]["mAP"])
                    reason = "exploratory" if math.isfinite(bd) else "invalid or non-overlapping curves"
                curves[arm].update(bd_vs_anchor=bd, bd_status=reason)
            atomic_json(out_dir / "probe_dual_region.json", result)
        prog.mark("complete")
        print(f"[dual] wrote {out_dir / 'probe_dual_region.json'}", flush=True)
        return result
    except BaseException as exc:
        prog.mark("failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        prog.journal.close()


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
    ap.add_argument("--score", type=float, default=0.5)
    ap.add_argument("--dilate", type=float, default=0.15)
    ap.add_argument("--r0-sigma", type=float, default=4.0)
    ap.add_argument("--roi-sigma", type=float, default=1.0)
    ap.add_argument("--fill", type=float, default=0.5)
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--out", default="outputs/probe_dual_region")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
