#!/usr/bin/env python
"""Kaggle kernel source: zero-shot v9-b DualCodec eval on GOT-10k tracking.

Notebook payload: %%bash cell that clones the v9 repo at a pinned commit,
builds a GOT-10k index from the dataset's **val** split (full per-frame GT),
and runs the 3-arm protocol eval (anchor / prep+codec / sandwich+codec, PRE
and POST routed per codec) of the v9-b checkpoint that was assembled from
v8's Kinetics-trained record holders and has NEVER seen tracking data.

Frozen SiamFC analyzer, real x264+x265 medium QP 30-50, AUC quality axis,
frame_size 128. Result lands in outputs/eval_v9b_tracking.
"""

TRACKING_EVAL_BASH = r"""%%bash
set -euo pipefail
export PYTHONUNBUFFERED=1

cd /kaggle/working
REPO=/kaggle/working/pre_processing_upgrade_9

if [ -d "$REPO/.git" ]; then
  git -C "$REPO" fetch --all -q
  git -C "$REPO" checkout -q __COMMIT__
else
  git clone -q https://github.com/wagur1/pre_processing_upgrade_9.git "$REPO"
  git -C "$REPO" checkout -q __COMMIT__
fi
cd "$REPO"

pip install -q opencv-python-headless pyyaml tqdm scipy matplotlib pandas 2>/dev/null | tail -1 || true

# ---- GOT-10k val split (full ground truth; test split is init-box only) ----
VAL_DIR=$(dirname "$(find /kaggle/input -maxdepth 6 -type d -name 'GOT-10k_Val_000001' -print -quit)")
if [ -z "$VAL_DIR" ] || [ ! -d "$VAL_DIR" ]; then
  echo "ERROR: GOT-10k val dir not found under /kaggle/input" >&2
  exit 1
fi
echo "[eval] GOT-10k val dir: $VAL_DIR"
echo "[eval] val sequences visible: $(find "$VAL_DIR" -maxdepth 1 -type d -name 'GOT-10k_Val_*' | wc -l)"

INDEX=data/index/got10k_val.json
python -m src.data.prepare_got10k \
    --root "$VAL_DIR" \
    --out "$INDEX" \
    --cap-gb 5 \
    --val-frac 0.0 \
    --test-frac 0.99

# ---- v9-b checkpoint from the attached ckpt dataset ----
CKPT_SRC=$(find /kaggle/input -name 'v9b_dualcodec.pth' -not -path '*/pre_processing_upgrade_9/*' 2>/dev/null | head -1 || true)
if [ -z "$CKPT_SRC" ]; then
  CKPT_SRC=$(find /kaggle/input -name 'preprocessor.pth' -path '*outputs*' -not -path '*/pre_processing_upgrade_9/*' 2>/dev/null | head -1 || true)
fi
if [ -z "$CKPT_SRC" ]; then
  echo "ERROR: no v9b checkpoint in /kaggle/input (attach the u9b-ckpt dataset)" >&2
  exit 1
fi
echo "[eval] checkpoint: $CKPT_SRC"

OUT=outputs/eval_v9b_tracking
mkdir -p "$OUT"

python evaluate.py --config configs/v9b_tracking.yaml \
    --ckpt "$CKPT_SRC" \
    --out "$OUT" \
    data.index="$INDEX"

echo "[eval] done"
ls -la "$OUT"
"""

import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--commit", required=True)
    a = p.parse_args()
    print(TRACKING_EVAL_BASH.replace("__COMMIT__", a.commit))


if __name__ == "__main__":
    main()
