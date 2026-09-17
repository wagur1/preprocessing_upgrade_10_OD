#!/usr/bin/env python
"""Push the COCO-detection *training* kernel.

Separate from ops/push_kernel.py because the generic train template builds the
Kinetics clip index and asserts its fingerprint — irrelevant here. This one
builds a COCO detection index (train from train2017 annotations, val from
val2017) and runs the engine's detection path with resume across the 12 h cap.

Usage:
  KAGGLE_ACCOUNT=... KAGGLE_API_TOKEN=... python ops/push_detection_train.py \\
      --commit <sha> --account <acct> --n-train 20000 --n-val 2000 --epochs 8
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
COCO_DATASET = "awsaf49/coco-2017-dataset"

BASH = r"""%%bash
set -euo pipefail
export PYTHONUNBUFFERED=1

cd /kaggle/working
REPO=/kaggle/working/preprocessing_upgrade_10_OD
if [ -d "$REPO/.git" ]; then
  git -C "$REPO" fetch --all -q
  git -C "$REPO" checkout -q __COMMIT__
else
  git clone -q https://github.com/wagur1/preprocessing_upgrade_10_OD.git "$REPO"
  git -C "$REPO" checkout -q __COMMIT__
fi
cd "$REPO"

pip install -q pycocotools 2>&1 | tail -1 || true

# ---- locate COCO (mount layout: /kaggle/input/datasets/<owner>/<slug>/...) ---
TR=$(find /kaggle/input -maxdepth 8 -type d -name train2017 | head -1 || true)
VA=$(find /kaggle/input -maxdepth 8 -type d -name val2017 | head -1 || true)
ANN_TR=$(find /kaggle/input -maxdepth 8 -name 'instances_train2017.json' | head -1 || true)
ANN_VA=$(find /kaggle/input -maxdepth 8 -name 'instances_val2017.json' | head -1 || true)
echo "[dettrain] train=$TR"
echo "[dettrain] val=$VA"
echo "[dettrain] ann_train=$ANN_TR"
echo "[dettrain] ann_val=$ANN_VA"
if [ -z "$TR" ] || [ -z "$ANN_TR" ] || [ -z "$ANN_VA" ]; then
  echo "[dettrain] ERROR: COCO inputs missing" >&2
  exit 1
fi

# ---- index: train from train2017 annotations, val from val2017 -------------
INDEX=data/index/coco_det.json
mkdir -p data/index
if [ ! -f "$INDEX" ]; then
  TRAIN_DIR="$TR" TRAIN_ANN="$ANN_TR" VAL_DIR="$VA" VAL_ANN="$ANN_VA" \
  N_TRAIN=__N_TRAIN__ N_VAL=__N_VAL__ python - <<'PY'
import json, os
from src.data import build_coco_index
tr = build_coco_index(os.environ["TRAIN_DIR"], os.environ["TRAIN_ANN"],
                      "/tmp/idx_tr.json", n_train=int(os.environ["N_TRAIN"]),
                      n_val=0)
va = build_coco_index(os.environ["VAL_DIR"], os.environ["VAL_ANN"],
                      "/tmp/idx_va.json", n_train=0,
                      n_val=int(os.environ["N_VAL"]))
a = json.load(open("/tmp/idx_tr.json")); b = json.load(open("/tmp/idx_va.json"))
# b was built with n_train=0, so its images live in b["val"]/b["test"] — reading
# b["train"] here produced an EMPTY val split, which silently disabled both
# validation and early stopping (best_val stayed inf, the checkpoint was never
# selected) for a 12 h run. Assert the sizes instead of trusting the naming.
splits = {"train": a["train"], "val": b["val"], "test": b["test"]}
for name, recs in splits.items():
    if not recs:
        raise SystemExit(f"[dettrain] FATAL: split '{name}' is empty — check the "
                         f"index construction before spending GPU hours")
json.dump({"meta": {"train_ann": os.environ["TRAIN_ANN"]}, **splits},
          open("data/index/coco_det.json", "w"))
print(f"[dettrain] index train={len(splits['train'])} val={len(splits['val'])} "
      f"test={len(splits['test'])}")
PY
fi
python -c "import json;d=json.load(open('$INDEX'));print('[dettrain] index sizes', {k: len(v) for k,v in d.items() if isinstance(v, list)})"

# ---- resume across the 12 h cap (same pattern as the AR train kernel) ------
OUT_DIR=__OUT_DIR__
CKPT_DIR=$OUT_DIR/checkpoints
mkdir -p "$CKPT_DIR"
if [ "${FRESH:-0}" != "1" ]; then
  python - <<'PY'
import glob, os, shutil
cands = []
for pat in ("**/preprocessor_last.pth", "**/preprocessor.pth"):
    cands += glob.glob(os.path.join("/kaggle/input", pat), recursive=True)
if cands:
    def rank(p): return (os.path.getmtime(p), 1 if p.endswith("_last.pth") else 0)
    src = max(cands, key=rank)
    dst = "__OUT_DIR__/checkpoints/preprocessor_last.pth"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    print(f"[resume] restored {src} -> {dst}")
else:
    print("[resume] no prior checkpoint found; starting fresh")
PY
fi

python train.py --config __CONFIG__ \
    data.index="$INDEX" \
    train.resume=true \
    __OVERRIDES__

echo "[dettrain] session done; artifacts under $OUT_DIR"
ls -la "$CKPT_DIR" || true
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", required=True)
    ap.add_argument("--account", default=os.environ.get("KAGGLE_ACCOUNT", "dngbolm"))
    ap.add_argument("--config", default="configs/sandwich_coco_det.yaml")
    ap.add_argument("--dataset", default=COCO_DATASET)
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--overrides", default="")
    ap.add_argument("--slug", default="u9-train-cocodet")
    ap.add_argument("--accelerator", default="NvidiaTeslaT4")
    a = ap.parse_args()

    m = re.search(r"^out_dir:\s*(\S+)", (REPO / a.config).read_text(), re.M)
    out_dir = m.group(1) if m else "outputs/sandwich_coco_det"

    src = BASH.replace("__COMMIT__", a.commit).replace("__CONFIG__", a.config)
    src = src.replace("__OUT_DIR__", out_dir)
    src = src.replace("__N_TRAIN__", str(a.n_train)).replace("__N_VAL__", str(a.n_val))
    src = src.replace("__OVERRIDES__", a.overrides or f"train.epochs={a.epochs}")

    nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {},
                     "outputs": [], "source": src.splitlines(keepends=True)}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    push_dir = REPO / "ops" / "_push" / a.slug
    push_dir.mkdir(parents=True, exist_ok=True)
    (push_dir / "notebook.ipynb").write_text(json.dumps(nb))
    meta = {"id": f"{a.account}/{a.slug}", "title": a.slug,
            "code_file": "notebook.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
            "enable_internet": True, "dataset_sources": [d for d in a.dataset.split(",") if d],
            "kernel_sources": [], "competition_sources": [], "model_sources": []}
    (push_dir / "kernel-metadata.json").write_text(json.dumps(meta))

    cmd = ["kaggle", "kernels", "push", "-p", str(push_dir)]
    if a.accelerator:
        cmd += ["--accelerator", a.accelerator]
    print(f"[push] {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout.strip())
    if r.returncode:
        print(r.stderr.strip(), file=sys.stderr)
        sys.exit(r.returncode)


if __name__ == "__main__":
    main()
