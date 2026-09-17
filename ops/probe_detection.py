#!/usr/bin/env python
"""Zero-shot detection probe: does the VCM preprocessor transfer to object detection?

Two stages in one run; stage A is a codec-free diagnostic, not an RD test:

  Stage A (no codec): mAP of a frozen COCO detector on x vs pre(x).
      A loss here does not establish rate-distortion failure. By default the
      requested codec sweep still runs; an explicit threshold can stop early.

  Stage B (codec sweep): the canonical 3-arm protocol on the mAP axis --
      anchor   codec(x)                 -> detector
      prep     codec(pre(x))            -> detector
      sandwich codec(pre(x)) -> post    -> detector
      BD-rate(mAP) per codec with a bootstrap CI over images.

Everything else mirrors the group's protocol: real x264/x265 preset medium,
QP 30-50, frozen checkpoints, and cached per-image xywh detections for paired
N-image with-replacement bootstrap CIs, separately for each arm.

Usage (Kaggle):
  python ops/probe_detection.py --images /kaggle/input/coco-2017-dataset/coco2017 \\
      --ckpt <preprocessor.pth> --config configs/sandwich_ar.yaml \\
      --n-images 500 --size 320 --stage both --out outputs/probe_detection
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.codecs.standard import StandardCodec, ffmpeg_available  # noqa: E402
from src.config import apply_overrides, load_config  # noqa: E402
from src.engine import _build_models, _qp_norm, _rate_cond  # noqa: E402
from src.metrics.bd_rate import bd_rate  # noqa: E402
from src.metrics.detection_bootstrap import (  # noqa: E402
    coco_map, paired_detection_bootstrap,
)


# ---------------------------------------------------------------- data -----
def load_coco(images_dir: Path, ann_file: Path, n: int, size: int, seed: int = 0):
    """Return [(image_id, tensor[1,3,1,H,W], orig_hw)] for n val images."""
    from PIL import Image

    ann = json.loads(Path(ann_file).read_text())
    id_to_ann = {}
    for a in ann["annotations"]:
        id_to_ann.setdefault(a["image_id"], []).append(a)
    imgs = [im for im in ann["images"] if im["id"] in id_to_ann]
    rng = random.Random(seed)
    rng.shuffle(imgs)
    out = []
    for im in imgs[:n]:
        p = Path(images_dir) / im["file_name"]
        if not p.is_file():
            continue
        img = Image.open(p).convert("RGB")
        w0, h0 = img.size
        img = img.resize((size, size), Image.BILINEAR)
        arr = torch.from_numpy(np.array(img, copy=True)).float().div_(255.0)     # [H,W,3]
        t = arr.permute(2, 0, 1).unsqueeze(0).unsqueeze(2)              # [1,3,1,H,W]
        out.append((im["id"], t, (h0, w0), id_to_ann[im["id"]]))
    return ann, out


def scaled_gt(anns, size: int, orig_hw):
    """COCO gt boxes rescaled from the original image to the probe resolution."""
    h0, w0 = orig_hw
    out = []
    for a in anns:
        x, y, w, h = a["bbox"]
        out.append({
            "id": a["id"], "image_id": a["image_id"],
            "category_id": a["category_id"], "iscrowd": a.get("iscrowd", 0),
            "area": (w * size / w0) * (h * size / h0),
            "bbox": [x * size / w0, y * size / h0, w * size / w0, h * size / h0],
        })
    return out


# ------------------------------------------------------------- detector ----
class Detector:
    """Frozen COCO Faster R-CNN. CTC uses Detectron2 X101-FPN; this is the
    drop-in stand-in available without a Detectron2 install (torchvision)."""

    def __init__(self, device, score_thresh: float = 0.05):
        from torchvision.models.detection import (
            FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn)
        self.device = device
        self.score_thresh = score_thresh
        self.model = fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1).to(device).eval()

    @torch.no_grad()
    def predict(self, imgs: torch.Tensor):
        """imgs [B,3,H,W] or [B,3,T,H,W] -> list of dicts (boxes/scores/labels).

        The probe feeds single-frame items, so a 5-D clip tensor is squeezed to
        its first frame; torchvision wants a list of [3,H,W]."""
        x = imgs.to(self.device)
        if x.ndim == 5:
            x = x[:, :, 0]
        return self.model(list(x))


# ---------------------------------------------------------------- probe ----
def _coco_box(b) -> list:
    """torchvision returns boxes as xyxy; COCO's bbox field is xywh.

    Submitting xyxy as xywh silently destroys every IoU match — the "width"
    becomes x2 and the "height" becomes y2 — which reads as a near-zero mAP at
    every resolution and looks like a detector or alignment problem instead.
    """
    vals = b.tolist() if hasattr(b, "tolist") else list(b)
    x1, y1, x2, y2 = (float(v) for v in vals[:4])
    return [x1, y1, max(x2 - x1, 1e-3), max(y2 - y1, 1e-3)]


def _load_at(images_dir: Path, ann_file: Path, n: int, size: int, seed: int,
             device: torch.device | None = None):
    """Load the fixture/COCO subset at one resolution, with rescaled gt boxes.

    Tensors are moved to ``device`` here rather than at each use site: the
    detector moves its own input internally but the preprocessor does not, so a
    CPU tensor against CUDA weights raises only on the PRE pass — a mismatch a
    CPU-only local pre-flight cannot surface.
    """
    ann_meta, items = load_coco(images_dir, ann_file, n, size, seed)
    if device is not None:
        items = [(i, t.to(device), hw, a) for i, t, hw, a in items]
    gt_by_id = {i: scaled_gt(a, size, hw) for i, _, hw, a in items}
    image_ids = [i for i, _, _, _ in items]
    return ann_meta, items, gt_by_id, image_ids


def run(args) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images_dir = Path(args.images)
    ann_file = Path(args.ann) if args.ann else images_dir.parent / "annotations" / "instances_val2017.json"
    print(f"[probe] images={images_dir} ann={ann_file} device={device}")

    cfg = apply_overrides(load_config(args.config),
                          [f"device={device.type}"])
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ck_cfg = state.get("cfg") if isinstance(state, dict) else None
    if isinstance(ck_cfg, dict) and isinstance(ck_cfg.get("model"), dict):
        # Restore all stored model settings, including non-state_dict behavior
        # such as qp_ref, strength and temporal flags. YAML only fills absent
        # keys for legacy checkpoints; never manufacture checkpoint defaults.
        cfg.setdefault("model", {}).update(deepcopy(ck_cfg["model"]))
    pre, codec, _ = _build_models(cfg, device, role="eval")
    pre.load_state_dict(state["model"] if "model" in state else state, strict=True)
    pre.eval()
    print(f"[probe] PRE+POST loaded from {args.ckpt}")

    det = Detector(device)
    sizes = [int(s) for s in (args.stage_a_sizes or str(args.size)).split(",")]
    results: dict = {"n_images": args.n_images, "size": args.size,
                     "stage_a_sizes": sizes, "stages": {"A": {}}}
    primary = None
    primary_size = args.size

    def det_pre(t):
        with torch.no_grad():
            xp = pre(t, _rate_cond(_qp_norm(40, cfg), 1, t.device, t.dtype))
        return det.predict(xp)

    # ------- stage A: no codec, does the edit preserve detection content? ----
    # Swept over sizes because the PRE was trained at 128: a size where pre(x)
    # keeps the anchor's mAP and a larger one where it does not separates
    # resolution transfer from task transfer. (GOT-10k already showed the edit
    # transfers across CONTENT domains at 128, so content is not the question.)
    for sz in sizes:
        ann_meta, items, gt_by_id, image_ids = _load_at(images_dir, ann_file, args.n_images, sz,
                                        args.seed, device)
        print(f"[probe] stage A at {sz}px ({len(items)} images)")

        def mAP_of(fn, tag, subset=None, _gt=gt_by_id, _ids=image_ids, _meta=ann_meta):
            ids = _ids if subset is None else subset
            preds = []
            for i, t, hw, _ in items:
                if i not in ids:
                    continue
                d = fn(t)[0]
                keep = d["scores"] >= det.score_thresh
                for b, s, l in zip(d["boxes"][keep], d["scores"][keep], d["labels"][keep]):
                    preds.append({"image_id": i, "category_id": int(l),
                                  "bbox": _coco_box(b), "score": float(s)})
            coco_ap, ap50 = coco_map(preds, _gt, ids, _meta)
            print(f"[probe] {tag}: mAP={coco_ap:.4f} mAP@.5={ap50:.4f} "
                  f"({len(preds)} boxes >= {det.score_thresh} over {len(ids)} images)")
            return coco_ap, len(preds)

        ap_x, n_boxes = mAP_of(lambda t: det.predict(t), f"stageA[{sz}] anchor (x)")
        ap_p, _ = mAP_of(det_pre, f"stageA[{sz}] prep (pre(x))")
        results["stages"]["A"][str(sz)] = {
            "mAP_x": ap_x, "mAP_pre": ap_p, "boxes_anchor": n_boxes,
            "ratio": (ap_p / ap_x) if ap_x > 0 else None}
        if sz == args.size:
            primary = (ap_x, ap_p, n_boxes, ann_meta, items, gt_by_id, image_ids)

    if primary is None:      # --size not listed: use the first swept size
        sz = sizes[0]
        ann_meta, items, gt_by_id, image_ids = _load_at(images_dir, ann_file, args.n_images, sz,
                                        args.seed, device)
        primary = (results["stages"]["A"][str(sz)]["mAP_x"],
                   results["stages"]["A"][str(sz)]["mAP_pre"],
                   results["stages"]["A"][str(sz)]["boxes_anchor"],
                   ann_meta, items, gt_by_id, image_ids)
        primary_size = sz
        print(f"[probe] note: --size {args.size} was not swept; stage B will use {sz}")
    ap_x, ap_p, n_boxes, ann_meta, items, gt_by_id, image_ids = primary
    results["size"] = primary_size

    if n_boxes < args.min_anchor_boxes:
        print(f"[probe] SETUP PROBLEM: the anchor produced only {n_boxes} boxes "
              f"(< {args.min_anchor_boxes}) — the detector/resolution pairing is "
              f"wrong (not a PRE effect). Aborting so quota is not spent on a "
              f"degenerate anchor curve.")
        results["verdict"] = "degenerate_anchor"
        return results
    if (args.stage_a_threshold is not None and ap_x > 0
            and ap_p / ap_x < args.stage_a_threshold):
        print(f"[probe] STAGE A threshold reached at {results['size']}px: pre(x) keeps only "
              f"{ap_p / ap_x:.2f} of the anchor mAP (< {args.stage_a_threshold}) "
              f"-> stopping at the requested diagnostic threshold, not an RD verdict.")
        results["verdict"] = "stage_A_threshold_stop"
        return results

    if args.stage == "a":
        results["verdict"] = "stage_A_only"
        return results

    # ---------------- stage B: rate-accuracy sweep on the mAP axis ----------
    if not ffmpeg_available():
        print("[probe] ffmpeg missing -> stage B skipped")
        results["verdict"] = "no_ffmpeg"
        return results

    qps = [int(q) for q in args.qps.split(",")]
    arms = ["anchor", "prep", "sandwich"]
    per_image = {arm: {} for arm in arms}       # arm -> qp -> {image_id: (bpp, preds)}
    for codec_name in ("h264", "h265"):
        for qp in qps:
            sc = StandardCodec(codec=codec_name, qp=qp, preset="medium")
            for arm in arms:
                per_image[arm].setdefault((codec_name, qp), {})
            for i, t, hw, _ in items:
                with torch.no_grad():
                    xp = pre(t, _rate_cond(_qp_norm(qp, cfg), 1, t.device, t.dtype))
                rec_a, bpp_a = sc.compress_decompress_items(t)
                rec_p, bpp_p = sc.compress_decompress_items(xp)
                rec_s = pre.post_restore(rec_p, _rate_cond(_qp_norm(qp, cfg), 1,
                                                           rec_p.device, rec_p.dtype))
                for arm, rec, bpp in (("anchor", rec_a, bpp_a), ("prep", rec_p, bpp_p),
                                      ("sandwich", rec_s, bpp_p)):
                    d = det.predict(rec[:, :, 0])[0]
                    keep = d["scores"] >= det.score_thresh
                    preds = [{"image_id": i, "category_id": int(l),
                              "bbox": _coco_box(b), "score": float(s)}
                             for b, s, l in zip(d["boxes"][keep], d["scores"][keep],
                                                d["labels"][keep])]
                    per_image[arm][(codec_name, qp)][i] = (float(bpp[0]), preds)
            print(f"[probe] coded {codec_name} qp{qp} for all arms")

    results["stages"]["B"] = {}
    for codec_name in ("h264", "h265"):
        curves = {}
        for arm in arms:
            rates, aps = [], []
            for qp in qps:
                slot = per_image[arm][(codec_name, qp)]
                rates.append(float(np.mean([v[0] for v in slot.values()])))
                preds = [p for v in slot.values() for p in v[1]]
                aps.append(coco_map(preds, gt_by_id, image_ids, ann_meta)[0])
            curves[arm] = {"rate": rates, "mAP": aps}
            print(f"[probe] {codec_name} {arm}: bpp={['%.4f' % r for r in rates]} "
                  f"mAP={['%.4f' % a for a in aps]}")
        entry = {"curves": curves}
        for arm in ("prep", "sandwich"):
            entry[f"bd_{arm}"] = bd_rate(curves["anchor"]["rate"], curves["anchor"]["mAP"],
                                         curves[arm]["rate"], curves[arm]["mAP"])
        if args.bootstrap:
            records = {arm: {qp: per_image[arm][(codec_name, qp)] for qp in qps}
                       for arm in arms}
            entry["ci"] = paired_detection_bootstrap(
                records, qps, gt_by_id, image_ids, ann_meta,
                n_draws=args.bootstrap, seed=args.seed)
            for arm, ci in entry["ci"].items():
                print(f"[probe] {codec_name} {arm}: {ci['n_draws']} valid / "
                      f"{ci['n_requested']} bootstrap draws; "
                      f"{ci['n_invalid']} invalid", flush=True)
        results["stages"]["B"][codec_name] = entry

    # persist the per-image records: mAP/BD/CI can then be recomputed offline,
    # which is what makes a 5000-image run possible at all — the probe shards by
    # image across kernels (the detector passes dominate: 3 per image per cell)
    # and the merge re-uses the bootstrap instead of re-running the detector.
    if args.records:
        rec_path = Path(args.out) / "per_image_records.npz"
        rec_path.parent.mkdir(parents=True, exist_ok=True)
        flat = {}
        for arm in arms:
            for (codec_name, qp), slot in per_image[arm].items():
                img = sorted(slot)
                boxes, offs, labs = [], [0], []
                for i in img:
                    for p in slot[i][1]:
                        boxes.append(p["bbox"] + [p["score"]])
                        labs.append(p["category_id"])
                    offs.append(len(boxes))
                tag = f"{arm}_{codec_name}_{qp}"
                flat[f"{tag}_img"] = np.asarray(img, dtype=np.int64)
                flat[f"{tag}_bpp"] = np.asarray([slot[i][0] for i in img], dtype=np.float32)
                flat[f"{tag}_boxes"] = np.asarray(boxes, dtype=np.float32).reshape(-1, 5)
                flat[f"{tag}_labels"] = np.asarray(labs, dtype=np.int32)
                flat[f"{tag}_offsets"] = np.asarray(offs, dtype=np.int64)
        np.savez_compressed(rec_path, **flat)
        print(f"[probe] per-image records -> {rec_path} ({rec_path.stat().st_size / 1e6:.1f} MB)",
              flush=True)

    results["verdict"] = "ran"
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--ann", default=None)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="configs/sandwich_ar.yaml")
    ap.add_argument("--n-images", type=int, default=500)
    ap.add_argument("--size", type=int, default=320,
                    help="probe resolution used by stage B")
    ap.add_argument("--stage-a-sizes", default=None,
                    help="comma list for the cheap stage-A sweep, e.g. 128,224,320. "
                         "128 is the PRE's training resolution (in-distribution), "
                         "so a size where pre(x) keeps its mAP but a larger one "
                         "does not isolates resolution transfer from task transfer.")
    ap.add_argument("--qps", default="30,35,40,45,50")
    ap.add_argument("--stage", choices=["a", "both"], default="both")
    ap.add_argument("--stage-a-threshold", type=float, default=None,
                    help="optional diagnostic early stop if pre(x) / anchor mAP "
                         "falls below this; disabled by default, not an RD test")
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--min-anchor-boxes", type=int, default=1,
                    help="abort if the anchor yields fewer detections: a broken "
                         "detector/resolution setup, not a PRE effect")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--records", action="store_true",
                    help="persist per-image (bpp, detections) so mAP/BD/CI can be "
                         "recomputed offline and image shards can be merged")
    ap.add_argument("--out", default="outputs/probe_detection")
    a = ap.parse_args()
    res = run(a)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    def _finite(o):
        """NaN/inf are not valid JSON and break downstream merges."""
        if isinstance(o, dict):
            return {k: _finite(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_finite(v) for v in o]
        if isinstance(o, float) and not np.isfinite(o):
            return None
        return o

    (out / "probe_detection.json").write_text(json.dumps(_finite(res), indent=2))
    print(f"\n[probe] verdict={res.get('verdict')}")
    print(json.dumps(_finite(res.get("stages", {}).get("B", {})), indent=2)[:1500])
    print(f"[probe] wrote {out / 'probe_detection.json'}")


if __name__ == "__main__":
    main()
