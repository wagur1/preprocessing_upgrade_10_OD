#!/usr/bin/env python
"""Open Images V6 background-suppression probe: mounted OID CSVs -> COCO json -> probe.

The Kaggle dataset programmerrdai/open-images-v6 ships the Open Images V6
annotation CSVs only — no photos. Inside a kernel with internet this script:

  1. maps OID label mids onto the COCO-80 classes the frozen detector knows
     (display-name match, case-insensitive, small alias table);
  2. deterministically samples --n-images validation images carrying at least
     --min-boxes mapped boxes (sorted ids, seeded shuffle; IsGroupOf rows are
     dropped — group boxes are not instance annotations);
  3. downloads them from the public open-images-dataset S3 bucket;
  4. writes a COCO-format annotation file whose image list holds exactly the
     downloaded photos, so the probe's seeded shuffle over an exact-size list
     is a no-op and the sampled set IS the evaluated set;
  5. invokes ops/probe_background_suppression.py unchanged as a subprocess.

Image IDs are zlib.crc32(ImageID) — a 32-bit int pycocotools can hold (the raw
hex is 64 bits and overflows its C long); the hex stays in file_name, and a
collision raises instead of silently merging two photos. GT boxes are the normalized CSV coordinates times the
pixel size of the downloaded photo. Pixel sizes are read from the raw grid —
no EXIF transpose — to match the probe's own loader, so GT scaling and the
detector input always share one orientation. The GT file lists only the
mapped categories present in the sampled images, and COCOeval drops
detections outside the GT category set, so mAP is averaged over the mapped
subset consistently for every arm.

Exploratory instrument: held_out False, ctc_conformant False — the same
caveats as probe_background_suppression. Determinism: the image set depends
only on the mounted CSV contents and this file; the same inputs give the same
set (same seed, same mapping, same min-box filter).
"""
from __future__ import annotations

import argparse
import csv
import zlib
import json
import random
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
S3_URL = "https://open-images-dataset.s3.amazonaws.com/validation/{iid}.jpg"

# torchvision fasterrcnn_resnet50_fpn COCO_V1 label order: id = index + 1.
COCO80 = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)
COCO_IDS = {name: i + 1 for i, name in enumerate(COCO80)}
# OID display names that mean a COCO class but spell it differently.
ALIAS = {
    "sofa": "couch", "sofa bed": "couch", "houseplant": "potted plant",
    "plant": "potted plant", "table": "dining table", "television": "tv",
    "mobile phone": "cell phone", "hotdog": "hot dog", "hair dryer": "hair drier",
    "motorbike": "motorcycle", "aeroplane": "airplane",
}


def build_mid_to_coco(class_csv: Path) -> dict:
    """Map OID LabelName mids -> COCO category id via display-name match."""
    mid_to_coco = {}
    with class_csv.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("DisplayName") or "").strip().lower()
            name = ALIAS.get(name, name)
            if name in COCO_IDS:
                mid_to_coco[row["LabelName"]] = COCO_IDS[name]
    return mid_to_coco


def sample_ids(bbox_csv: Path, mid_to_coco: dict, n_images: int, min_boxes: int,
               seed: int = 0):
    """Deterministically pick n_images validation ids with >= min_boxes mapped boxes."""
    import pandas as pd

    bbox = pd.read_csv(bbox_csv)
    bbox = bbox[(bbox["IsGroupOf"] == 0) & (bbox["LabelName"].isin(mid_to_coco))]
    counts = bbox.groupby("ImageID").size()
    eligible = sorted(counts[counts >= min_boxes].index.tolist())
    if len(eligible) < n_images:
        raise ValueError(f"only {len(eligible)} images with >= {min_boxes} mapped "
                         f"boxes; need {n_images}")
    rng = random.Random(seed)
    rng.shuffle(eligible)
    return eligible[:n_images], int(len(bbox))


def download_images(ids, out_dir: Path, tries: int = 3):
    """Download the validation photos; return {int(hex id): (w, h, file_name)}."""
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {}
    for iid in ids:
        dest = out_dir / f"{iid}.jpg"
        if not dest.is_file():
            last = None
            for attempt in range(tries):
                try:
                    urllib.request.urlretrieve(S3_URL.format(iid=iid), dest)
                    last = None
                    break
                except Exception as exc:  # noqa: BLE001 — retry any transport error
                    last = exc
                    time.sleep(1 + attempt)
            if last is not None:
                raise RuntimeError(f"download failed for {iid}: {last}")
        with Image.open(dest) as source:
            key = zlib.crc32(iid.encode("ascii"))
        if key in meta:
            raise ValueError(f"crc32 id collision between two sampled images ({iid})")
        meta[key] = (source.size[0], source.size[1], f"{iid}.jpg")
        print(f"[oid] downloaded {iid} ({len(meta)}/{len(ids)})", flush=True)
    return meta


def to_coco_json(bbox_csv: Path, mid_to_coco: dict, meta: dict):
    """COCO dict: images = downloaded photos; GT restricted to mapped classes."""
    import pandas as pd

    bbox = pd.read_csv(bbox_csv)
    bbox = bbox[(bbox["IsGroupOf"] == 0) & (bbox["LabelName"].isin(mid_to_coco))].copy()
    bbox["image_key"] = bbox["ImageID"].map(lambda s: zlib.crc32(s.encode("ascii")))
    bbox = bbox[bbox["image_key"].isin(meta)]

    used = sorted({mid_to_coco[mid] for mid in bbox["LabelName"]})
    names = {i + 1: name for i, name in enumerate(COCO80)}
    images, annotations = [], []
    ann_id = 0
    for iid, (w, h, file_name) in sorted(meta.items(), key=lambda kv: kv[1][2]):
        images.append({"id": iid, "file_name": file_name, "width": w, "height": h})
        rows = bbox[bbox["image_key"] == iid]
        for _, row in rows.iterrows():
            x = max(0.0, float(row["XMin"]) * w)
            y = max(0.0, float(row["YMin"]) * h)
            bw = min(w - x, (float(row["XMax"]) - float(row["XMin"])) * w)
            bh = min(h - y, (float(row["YMax"]) - float(row["YMin"])) * h)
            if bw <= 0 or bh <= 0:
                continue
            ann_id += 1
            annotations.append({
                "id": ann_id, "image_id": iid, "category_id": mid_to_coco[row["LabelName"]],
                "bbox": [x, y, bw, bh], "area": bw * bh, "iscrowd": 0,
            })
    categories = [{"id": cid, "name": names[cid]} for cid in used]
    return {"images": images, "annotations": annotations, "categories": categories}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv-dir", required=True,
                    help="directory holding the mounted OID CSVs (searched recursively)")
    ap.add_argument("--n-images", type=int, default=500)
    ap.add_argument("--min-boxes", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--download-dir", default="outputs/openimages/images")
    ap.add_argument("--ann-json", default="outputs/openimages/instances_openimages.json")
    ap.add_argument("--ids-json", default=None,
                    help="also write the sampled hex ids here (provenance)")
    # passthrough to probe_background_suppression
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--qps", default="30,35,40,45,50")
    ap.add_argument("--sigmas", default="4,8,16")
    ap.add_argument("--score", type=float, default=0.5)
    ap.add_argument("--dilate", type=float, default=0.15)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="outputs/probe_openimages")
    ap.add_argument("--probe-script", default="ops/probe_background_suppression.py")
    a = ap.parse_args()

    csv_dir = Path(a.csv_dir)
    bbox_csv = sorted(csv_dir.rglob("validation-annotations-bbox.csv"))[0]
    class_csv = sorted(csv_dir.rglob("*class-descriptions*.csv"))[0]
    print(f"[oid] bbox={bbox_csv} classes={class_csv}", flush=True)

    mid_to_coco = build_mid_to_coco(class_csv)
    if len(mid_to_coco) < 50:
        raise ValueError(f"class mapping suspiciously small: {len(mid_to_coco)}")
    print(f"[oid] mapped OID classes -> COCO: {len(mid_to_coco)}", flush=True)

    ids, kept_rows = sample_ids(bbox_csv, mid_to_coco, a.n_images, a.min_boxes, a.seed)
    print(f"[oid] sampled {len(ids)} images (kept {kept_rows} mapped rows)", flush=True)
    if a.ids_json:
        Path(a.ids_json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.ids_json).write_text(json.dumps(ids), encoding="utf-8")

    meta = download_images(ids, Path(a.download_dir))
    coco = to_coco_json(bbox_csv, mid_to_coco, meta)
    Path(a.ann_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.ann_json).write_text(json.dumps(coco), encoding="utf-8")
    print(f"[oid] wrote {a.ann_json}: {len(coco['images'])} images, "
          f"{len(coco['annotations'])} boxes, {len(coco['categories'])} categories",
          flush=True)

    cmd = [sys.executable, str(REPO / a.probe_script),
           "--images", a.download_dir, "--ann", a.ann_json,
           "--n-images", str(len(ids)), "--size", str(a.size),
           "--qps", a.qps, "--sigmas", a.sigmas,
           "--score", str(a.score), "--dilate", str(a.dilate),
           "--device", a.device, "--out", a.out]
    print(f"[oid] probe: {' '.join(cmd)}", flush=True)
    raise SystemExit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
