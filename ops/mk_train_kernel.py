#!/usr/bin/env python
"""Kaggle kernel source: train Round (b) — QP-conditioned additive edit.

Notebook payload: a single %%bash cell that clones the v7 repo at a pinned
commit, builds the canonical hash-split index, and runs training with
resume across 12h session boundaries (the engine checkpoints
preprocessor_last.pth every epoch; a relaunched kernel re-clones the repo,
re-downloads the latest checkpoint via kagglehub, and continues).

Design (docs/RUN_DESIGN_qpc.md, guard (iv)): the run config diff vs the
kappa=10 lineage is EXACTLY {model.arch: additive_cond, out_dir} — nothing
else. 16 epochs, batch 8, 128px, seed 0, teachers [r3d_18, mc3_18]
sampled, mu=10, kappa=10, beta=0.001.
"""

TRAIN_BASH = r"""%%bash
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

# ---- dataset + canonical index ----
KINETICS_ROOT=""
for c in /kaggle/input/kinetics-train-5per/train /kaggle/input/datasets/rohanmallick/kinetics-train-5per/kinetics400_5per; do
  [ -d "$c" ] && KINETICS_ROOT="$c" && break
done
if [ -z "$KINETICS_ROOT" ]; then
  sample=$(find /kaggle/input -maxdepth 8 -type f \( -iname '*.mp4' -o -iname '*.avi' -o -iname '*.mkv' -o -iname '*.mov' -o -iname '*.webm' -o -iname '*.m4v' \) -print -quit)
  KINETICS_ROOT=$(dirname "$(dirname "$sample")")
fi
echo "[train] Kinetics root: $KINETICS_ROOT"

INDEX=data/index/kinetics_hash_split.json
if [ ! -f "$INDEX" ] || ! python -c 'import json,sys; sys.exit("test" not in json.load(open(sys.argv[1])))' "$INDEX"; then
  python scripts/build_train_index.py --root "$KINETICS_ROOT" --out "$INDEX" --assert-fingerprint 30f083f8520a
fi

# ---- resume: pull the newest checkpoint from this kernel's own output ----
OUT_DIR=__OUT_DIR__
CKPT_DIR=$OUT_DIR/checkpoints
mkdir -p "$CKPT_DIR"
if [ "${FRESH:-0}" != "1" ] && [ -n "$(ls -A /kaggle/input 2>/dev/null)" ]; then
  python - <<'PY'
# The kernel's PREVIOUS version output is attached as a data source
# (kernel_sources: this kernel). kagglehub-style mount: /kaggle/input/<slug>/...
# Copy the last checkpoint if present so training resumes across sessions.
import os, shutil, glob
src_root = "/kaggle/input"
cands = []
for pat in ("**/preprocessor_last.pth", "**/preprocessor.pth"):
    cands += glob.glob(os.path.join(src_root, pat), recursive=True)
if cands:
    # newest mtime wins; prefer *_last for resume
    def rank(p):
        return (os.path.getmtime(p), 1 if p.endswith("_last.pth") else 0)
    src = max(cands, key=rank)
    dst_dir = "__OUT_DIR__/checkpoints"
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, "preprocessor_last.pth")
    shutil.copy2(src, dst)
    print(f"[resume] restored {src} -> {dst}")
else:
    print("[resume] no prior checkpoint found; starting fresh")
PY
fi

__EXTRA_BASH__

# ---- train (resume=true survives session death) ----
python train.py --config __CONFIG__ \
    data.index="$INDEX" \
    train.resume=true \
    __OVERRIDES__

echo "[train] session done; artifacts under $OUT_DIR"
ls -la "$CKPT_DIR" || true
"""

import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--commit", required=True, help="repo commit to pin")
    p.add_argument("--config", default="configs/upvcm_ar.yaml")
    p.add_argument("--out-dir", default=None,
                   help="outputs subdir (default: config's out_dir)")
    p.add_argument("--overrides", default="", help="extra dotted config overrides")
    p.add_argument("--fresh", action="store_true", help="ignore prior checkpoints")
    a = p.parse_args()

    out_dir = a.out_dir
    if out_dir is None:
        import re
        m = re.search(r"^out_dir:\s*(\S+)", open(a.config).read(), re.M)
        out_dir = m.group(1) if m else "outputs/train"

    bash = TRAIN_BASH.replace("__COMMIT__", a.commit)
    bash = bash.replace("__CONFIG__", a.config)
    bash = bash.replace("__OUT_DIR__", out_dir)
    bash = bash.replace("__OVERRIDES__", a.overrides or "train.epochs=16")
    if a.fresh:
        bash = bash.replace("${FRESH:-0}", "1")
    print(bash)


if __name__ == "__main__":
    main()
