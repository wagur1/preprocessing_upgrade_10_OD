#!/usr/bin/env python
"""Exploratory detector-mask suppression probe using real x264/x265 codecs.

This is not a held-out or CTC-conformant evaluation: the detector is torchvision
Faster R-CNN, frames are square-resized, and mask/sigma choices can be tuned.
Two-QP runs are smoke checks, with no BD-rate claim. Records are flushed before
aggregation, with progress.json tracking the last completed operation. Use a
fresh output directory for each run; automatic resume is deliberately unsupported.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.codecs.standard import StandardCodec, ffmpeg_available  # noqa: E402
from src.metrics.bd_rate import bd_rate  # noqa: E402
from src.models.mask_suppress import protect_mask, suppress  # noqa: E402
from probe_detection import Detector, _coco_box, coco_map, load_coco, scaled_gt  # noqa: E402


def finite(value):
    if isinstance(value, dict):
        return {k: finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path, value):
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(finite(value), f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def parse_image_ids(specs):
    ids = [int(v.strip()) for spec in (specs or []) for v in spec.split(",")]
    if len(ids) != len(set(ids)) or any(i < 0 for i in ids):
        raise ValueError("image IDs must be unique nonnegative integers")
    return ids


def load_coco_ids(images_dir: Path, ann_file: Path, ids, size: int):
    """Strict exact selection, in requested order, including negative images."""
    from PIL import Image

    ann = json.loads(ann_file.read_text(encoding="utf-8"))
    by_id = {im["id"]: im for im in ann["images"]}
    id_to_ann = {}
    for a in ann["annotations"]:
        id_to_ann.setdefault(a["image_id"], []).append(a)
    missing = [i for i in ids if i not in by_id or
               not (images_dir / by_id[i]["file_name"]).is_file()]
    if missing:
        raise ValueError(f"Requested image IDs missing metadata or photos: {missing}")
    items = []
    for i in ids:
        with Image.open(images_dir / by_id[i]["file_name"]) as source:
            img = source.convert("RGB")
            w0, h0 = img.size
            img = img.resize((size, size), Image.Resampling.BILINEAR)
            t = torch.from_numpy(np.array(img, copy=True)).float().div_(255)
        t = t.permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
        items.append((i, t, (h0, w0), id_to_ann.get(i, [])))
    return ann, items


def cell_arrays(tag, slot):
    img = sorted(slot)
    boxes, labels, offsets = [], [], [0]
    for i in img:
        for p in slot[i][1]:
            boxes.append(p["bbox"] + [p["score"]])
            labels.append(p["category_id"])
        offsets.append(len(boxes))
    return {f"{tag}_img": np.asarray(img, dtype=np.int64),
            f"{tag}_bpp": np.asarray([slot[i][0] for i in img], dtype=np.float32),
            f"{tag}_boxes": np.asarray(boxes, dtype=np.float32).reshape(-1, 5),
            f"{tag}_labels": np.asarray(labels, dtype=np.int32),
            f"{tag}_offsets": np.asarray(offsets, dtype=np.int64)}


class Progress:
    """Append full predictions per image, plus atomic progress/NPZ snapshots."""
    def __init__(self, out_dir, metadata):
        self.out_dir = out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents mixing runs (including concurrent launches).
        self.journal = (out_dir / "records.jsonl").open("x", encoding="utf-8")
        self.state = dict(metadata, status="starting", completed_records=0,
                          completed_cells=[], updated_at=time.time())
        self.flat = {}
        self.mark("starting")

    def mark(self, status, **fields):
        self.state.update(fields, status=status, updated_at=time.time())
        atomic_json(self.out_dir / "progress.json", self.state)

    def record(self, tag, image_id, bpp, predictions):
        json.dump(dict(tag=tag, image_id=image_id, bpp=bpp, predictions=predictions),
                  self.journal, allow_nan=False)
        self.journal.write("\n")
        self.journal.flush()
        os.fsync(self.journal.fileno())
        self.mark("coding", completed_records=self.state["completed_records"] + 1,
                  last_record=dict(tag=tag, image_id=image_id))

    def cell(self, tag, slot):
        self.flat.update(cell_arrays(tag, slot))
        path = self.out_dir / "per_image_records.npz"
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("wb") as f:
            np.savez_compressed(f, **self.flat)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        atomic_json(self.out_dir / f"{tag}.json",
                    dict(tag=tag, image_ids=sorted(slot),
                         rate=float(np.mean([v[0] for v in slot.values()])),
                         aggregation="pending"))
        self.state["completed_cells"].append(tag)
        self.mark("coding")


def run(a):
    qps = [int(q) for q in a.qps.split(",")]
    sigmas = [float(s) for s in a.sigmas.split(",")]
    req_ids = parse_image_ids(a.image_ids)
    if len(qps) < 2 or len(set(qps)) != len(qps) or any(q < 0 or q > 51 for q in qps):
        raise ValueError("Use at least two distinct QPs in [0,51]")
    if (not sigmas or len(set(sigmas)) != len(sigmas) or
            any(not math.isfinite(s) or s <= 0 for s in sigmas)):
        raise ValueError("sigmas must be distinct, finite and positive")
    if a.size <= 0 or a.size % 2 or a.n_images <= 0:
        raise ValueError("size must be positive/even for yuv420p; n-images positive")
    if not math.isfinite(a.score) or not 0 <= a.score <= 1 or not math.isfinite(a.dilate) or a.dilate < 0:
        raise ValueError("score must be in [0,1]; dilation finite/nonnegative")
    if not ffmpeg_available():
        raise RuntimeError("ffmpeg/ffprobe not found on PATH")
    device = torch.device(a.device if a.device != "auto" else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(a.out)
    metadata = dict(size=a.size, qps=qps, sigmas=sigmas, score=a.score,
                    dilate=a.dilate, device=str(device),
                    images=str(Path(a.images).resolve()), ann=str(Path(a.ann).resolve()),
                    image_selection="explicit_ids" if req_ids else "shuffled_seed_0",
                    requested_image_ids=req_ids, mode="smoke" if len(qps) < 4 else "exploratory_rd",
                    held_out=False, ctc_conformant=False,
                    detector="torchvision FasterRCNN ResNet50 FPN COCO_V1",
                    codec_preset="medium", protocol="single-frame RGB -> yuv420p; square resize")
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
        print(f"[bg] {len(items)} images {ids}, {device}, QPs={qps}, sigmas={sigmas}", flush=True)
        det = Detector(device)
        masks = {}
        for i, t, _, _ in items:
            d = det.predict(t)[0]
            masks[i] = protect_mask(d["boxes"], d["scores"], d["labels"], a.size, a.score, a.dilate)
            prog.mark("masking", last_mask_image_id=i)
        cover = float(np.mean([m.mean().item() for m in masks.values()]))
        result = dict(metadata, cover=cover, curves={})
        arms = [("anchor", 0)] + [(f"blur{s:g}", s) for s in sigmas]
        records = {}
        # Complete each codec/QP/arm cell, saving it before any COCO aggregation.
        for codec in ("h264", "h265"):
            for qp in qps:
                sc = StandardCodec(codec=codec, qp=qp, preset="medium")
                for arm, sigma in arms:
                    tag = f"{arm}_{codec}_{qp}"
                    slot = records[tag] = {}
                    for i, t, _, _ in items:
                        t = t.to(device)
                        xv = t if arm == "anchor" else suppress(t, masks[i], sigma)
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
                    print(f"[bg] saved {tag}: {len(slot)} images", flush=True)
        # Records are already durable if aggregation is slow or fails.
        for codec in ("h264", "h265"):
            curves = result["curves"][codec] = {}
            for arm, _ in arms:
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
                atomic_json(out_dir / "probe_bgsuppress.json", result)
            for arm, _ in arms[1:]:
                bd = None
                reason = "requires at least four QPs; smoke only"
                if len(qps) >= 4:
                    bd = bd_rate(curves["anchor"]["rate"], curves["anchor"]["mAP"],
                                 curves[arm]["rate"], curves[arm]["mAP"])
                    reason = "exploratory" if math.isfinite(bd) else "invalid or non-overlapping curves"
                curves[arm].update(bd_vs_anchor=bd, bd_status=reason)
            atomic_json(out_dir / "probe_bgsuppress.json", result)
        prog.mark("complete")
        print(f"[bg] wrote {out_dir / 'probe_bgsuppress.json'}", flush=True)
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
    ap.add_argument("--image-ids", action="append", default=[], help="Exact comma-separated COCO IDs; repeatable")
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--qps", default="30,35,40,45,50", help="Two QPs allowed for smoke; BD requires four")
    ap.add_argument("--sigmas", default="4,8,16")
    ap.add_argument("--score", type=float, default=0.5)
    ap.add_argument("--dilate", type=float, default=0.15)
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--out", default="outputs/probe_bgsuppress")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
