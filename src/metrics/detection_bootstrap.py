"""Paired image bootstrap of COCO AP/BD-rate from cached xywh detections."""

from __future__ import annotations

import contextlib
from copy import deepcopy
import io
import random

import numpy as np

from .bd_rate import bd_rate


def _resampled_coco(results, gt_by_id, image_ids, ann_meta):
    """Give every sampled occurrence its own image and annotation IDs.

    ``results`` contains each source image's predictions once, already in COCO
    xywh format. Repeated source IDs in ``image_ids`` replicate both GT and DT;
    merely passing repeated IDs to COCOeval would silently deduplicate them.
    """
    by_image = {}
    for result in results:
        by_image.setdefault(result["image_id"], []).append(result)
    images, annotations, detections = [], [], []
    for new_id, source_id in enumerate(image_ids, 1):
        images.append({"id": new_id})
        for annotation in gt_by_id.get(source_id, []):
            annotations.append(dict(deepcopy(annotation), image_id=new_id,
                                    id=len(annotations) + 1))
        for result in by_image.get(source_id, []):
            detections.append(dict(deepcopy(result), image_id=new_id,
                                   id=len(detections) + 1))
    return {"info": {}, "images": images, "annotations": annotations,
            "categories": deepcopy(ann_meta["categories"])}, detections


def coco_map(results, gt_by_id, image_ids, ann_meta):
    """Recompute mAP@[.5:.95] and AP50, preserving sample multiplicities.

    Undefined AP (no evaluable ground truth) is NaN; missing predictions on
    evaluable ground truth score zero. Inputs are never modified by loadRes.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    dataset, detections = _resampled_coco(results, gt_by_id, image_ids, ann_meta)
    if not dataset["annotations"]:
        # no evaluable ground truth: AP is undefined, and bd_rate must see the
        # difference from "detector found nothing", which is a real 0.0
        return float("nan"), float("nan")
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = dataset
        gt.createIndex()
        if detections:
            dt = gt.loadRes(detections)
        else:
            # a detector that returns nothing on evaluable gt scores 0.0
            dt = COCO()
            dt.dataset = {**dataset, "annotations": []}
            dt.createIndex()
        ev = COCOeval(gt, dt, "bbox")
        ev.params.imgIds = [im["id"] for im in dataset["images"]]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return tuple(float(v) if v >= 0 else float("nan") for v in ev.stats[:2])


def detection_curve(points, qps, gt_by_id, image_ids, ann_meta):
    """Recompute a curve from qp -> image_id -> (bpp, cached detections)."""
    rates, aps = [], []
    for qp in qps:
        slot = points[qp]
        rates.append(float(np.mean([slot[i][0] for i in image_ids])))
        # coco_map replicates occurrences; do not replicate predictions twice.
        preds = [p for i in dict.fromkeys(image_ids) for p in slot[i][1]]
        aps.append(coco_map(preds, gt_by_id, image_ids, ann_meta)[0])
    return {"rate": rates, "mAP": aps}


def paired_detection_bootstrap(records, qps, gt_by_id, image_ids, ann_meta,
                               n_draws=200, seed=0):
    """N-image with-replacement draws shared across every arm and QP.

    ``records`` is arm -> qp -> image_id -> (bpp, detections). Return separate
    BD-rate draws and percentile CIs against anchor for each other arm. Invalid
    draws are retained as None and counted, never pooled across arms or retried.
    """
    image_ids = list(image_ids)
    qps = list(qps)
    if not image_ids or len(set(image_ids)) != len(image_ids):
        raise ValueError("bootstrap requires a nonempty set of unique source image IDs")
    if not qps or n_draws < 0:
        raise ValueError("bootstrap requires QPs and a nonnegative draw count")
    if "anchor" not in records:
        raise ValueError("bootstrap requires an anchor arm")
    for points in records.values():
        for qp in qps:
            if any(i not in points[qp] for i in image_ids):
                raise ValueError("all arms and QPs must contain every sampled image")
    draws = {arm: [] for arm in records if arm != "anchor"}
    rng = random.Random(seed)
    for _ in range(n_draws):
        sample = [rng.choice(image_ids) for _ in image_ids]
        curves = {arm: detection_curve(points, qps, gt_by_id, sample, ann_meta)
                  for arm, points in records.items()}
        anchor = curves["anchor"]
        for arm, values in draws.items():
            test = curves[arm]
            valid = all(np.isfinite(c["mAP"]).all() and
                        np.isfinite(c["rate"]).all() and
                        (np.asarray(c["rate"]) > 0).all()
                        for c in (anchor, test))
            value = bd_rate(anchor["rate"], anchor["mAP"], test["rate"],
                            test["mAP"]) if valid else float("nan")
            values.append(float(value) if np.isfinite(value) else None)
    summary = {}
    for arm, values in draws.items():
        valid = [v for v in values if v is not None]
        summary[arm] = {
            "draws": values, "seed": seed, "n_requested": n_draws,
            "n_draws": len(valid), "n_invalid": n_draws - len(valid),
            "lo": float(np.percentile(valid, 2.5)) if valid else None,
            "hi": float(np.percentile(valid, 97.5)) if valid else None,
        }
    return summary
