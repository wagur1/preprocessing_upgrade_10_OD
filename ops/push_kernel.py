#!/usr/bin/env python
"""Push a v7 ops kernel to Kaggle (train or eval), with metadata.

Wraps `kaggle kernels push`: builds the notebook from ops/mk_{train,eval}_kernel.py,
writes kernel-metadata.json, pushes, and prints the resulting web URL.

Requires: KAGGLE_API_TOKEN env (source ops/kaggle_env.sh first).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

KINETICS_DATASET = "rohanmallick/kinetics-train-5per"


def make_notebook(bash_src: str) -> dict:
    cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": bash_src.splitlines(keepends=True),
    }
    return {
        "cells": [cell],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("kind", choices=["train", "eval", "probe"])
    p.add_argument("--commit", default=None, help="repo commit to pin (default: HEAD)")
    p.add_argument("--config", default="configs/sandwich_ar.yaml",
                   help="(train) config to run")
    p.add_argument("--overrides", default=None,
                   help="(train) dotted config overrides (default: train.epochs=16)")
    p.add_argument("--init-from", default=None,
                   help="(train) kernel slug whose output holds the warm-start checkpoint")
    p.add_argument("--extra-bash", default="", help="(train) extra bash inserted before train")
    p.add_argument("--dataset-extra", default=None,
                   help="(train) extra dataset slug to attach (e.g. warm-start ckpt dataset)")
    p.add_argument("--slug-suffix", default="", help="appended to kernel slug")
    p.add_argument("--shard-idx", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=3)
    p.add_argument("--train-kernel", default=None,
                   help="(eval) train kernel slug whose output is the checkpoint source")
    p.add_argument("--confirmatory", action="store_true",
                   help="(eval) use the fresh never-used-sibling holdout (audit #4)")
    p.add_argument("--ckpt-dataset", default=None,
                   help="(eval) dataset slug holding the checkpoint (e.g. frankenstein)")
    p.add_argument("--no-gpu", action="store_true")
    p.add_argument("--analyzer", choices=["heldout", "teacher"], default="heldout",
                   help="(eval) heldout: r2plus1d_18 canonical; teacher: "
                        "eval.held_out_backbone=null -> task.backbone (on-teacher arm; "
                        "the YAML value must be nulled, omitting the override is not enough)")
    p.add_argument("--accelerator", default=None,
                   help="e.g. NvidiaTeslaT4 (P100 sm_60 is INCOMPATIBLE with "
                        "Kaggle's preinstalled torch: no kernel image)")
    a = p.parse_args()

    commit = a.commit or subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True).stdout.strip()[:12]

    import importlib
    if a.kind == "train":
        mod = importlib.import_module("ops.mk_train_kernel")
        src = mod.TRAIN_BASH.replace("__COMMIT__", commit)
        src = src.replace("__CONFIG__", a.config)
        import re
        m = re.search(r"^out_dir:\s*(\S+)", open(str(REPO / a.config)).read(), re.M)
        src = src.replace("__OUT_DIR__", m.group(1) if m else "outputs/train")
        src = src.replace("__OVERRIDES__", a.overrides or "train.epochs=16")
        src = src.replace("__EXTRA_BASH__", a.extra_bash or "true")
    elif a.kind == "eval":
        mod = importlib.import_module("ops.mk_eval_kernel")
        src = mod.EVAL_BASH.replace("__COMMIT__", commit)
        if a.confirmatory:
            # must go INSIDE the bash cell (after %%bash) — prefixing before
            # the %%bash magic makes papermill run bash as python: SyntaxError
            src = src.replace("%%bash\nset -euo pipefail",
                              "%%bash\nset -euo pipefail\nexport CONFIRMATORY=1", 1)
        src = src.replace("__CONFIG__", a.config)
        src = src.replace("__SHARD_ARGS__",
                          f"eval.shard_idx={a.shard_idx} eval.num_shards={a.num_shards}")
        src = src.replace("__SUFFIX__", f"shard{a.shard_idx}")
        src = src.replace("__HELD_OUT__", mod.held_out_override(a.analyzer))
    else:
        src = PROBE_BASH.replace("__COMMIT__", commit)

    slug = f"u9-{a.kind}{a.slug_suffix or ('-shard%d' % a.shard_idx if a.kind == 'eval' else '')}"
    push_dir = REPO / "ops" / "_push" / slug
    push_dir.mkdir(parents=True, exist_ok=True)

    nb = make_notebook(src)
    (push_dir / "notebook.ipynb").write_text(json.dumps(nb))

    meta = {
        "id": f"__ACCOUNT__/{slug}",
        "title": slug,
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": not a.no_gpu,
        "enable_internet": True,
        "dataset_sources": [KINETICS_DATASET],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
    }
    if a.kind == "eval" and a.train_kernel:
        # Attach the train kernel's output: newest preprocessor.pth wins.
        meta["kernel_sources"] = [a.train_kernel]
    if a.kind == "eval" and a.ckpt_dataset:
        meta["dataset_sources"] = list(meta.get("dataset_sources", [])) + [a.ckpt_dataset]
    if a.kind == "train" and a.init_from:
        # Warm start: attach the source kernel's output (its checkpoints).
        meta["kernel_sources"] = [a.init_from]
    if a.dataset_extra:
        meta["dataset_sources"] = list(meta.get("dataset_sources", [])) + [a.dataset_extra]
    account = __import__("os").environ.get("KAGGLE_ACCOUNT", "")
    meta["id"] = meta["id"].replace("__ACCOUNT__", account or "wagur124705")

    (push_dir / "kernel-metadata.json").write_text(json.dumps(meta))

    cmd = ["kaggle", "kernels", "push", "-p", str(push_dir)]
    if a.accelerator:
        cmd += ["--accelerator", a.accelerator]
    print(f"[push] {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout.strip())
    if r.returncode != 0:
        print(r.stderr.strip(), file=sys.stderr)
        sys.exit(r.returncode)


PROBE_BASH = r"""%%bash
set -euo pipefail
export PYTHONUNBUFFERED=1
cd /kaggle/working
git clone -q https://github.com/wagur1/pre_processing_upgrade_9.git repo 2>/dev/null || (cd repo && git pull -q)
cd repo
git checkout -q __COMMIT__
pip install -q opencv-python-headless pyyaml tqdm scipy matplotlib pandas 2>/dev/null | tail -1 || true
python - <<'PY'
import subprocess, sys, torch
print(f"[probe] python {sys.version.split()[0]}")
print(f"[probe] torch {torch.__version__} cuda {torch.cuda.is_available()} "
      f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")
print("[probe] internet OK")
PY
KINETICS_ROOT=""
for c in /kaggle/input/kinetics-train-5per/train /kaggle/input/datasets/rohanmallick/kinetics-train-5per/kinetics400_5per; do
  [ -d "$c" ] && KINETICS_ROOT="$c" && break
done
if [ -z "$KINETICS_ROOT" ]; then
  sample=$(find /kaggle/input -maxdepth 8 -type f \( -iname '*.mp4' -o -iname '*.avi' -o -iname '*.mkv' -o -iname '*.mov' -o -iname '*.webm' -o -iname '*.m4v' \) -print -quit)
  KINETICS_ROOT=$(dirname "$(dirname "$sample")")
fi
echo "[probe] Kinetics root: $KINETICS_ROOT"
echo "[probe] mp4 visible: $(find "$KINETICS_ROOT" -type f -iname '*.mp4' | wc -l)"
python scripts/build_train_index.py --root "$KINETICS_ROOT" --out data/index/kinetics_hash_split.json --assert-fingerprint 30f083f8520a
python -m src.models.virtual_codec
python -m src.models.ste_codec
echo "[probe] ALL DONE"
"""


if __name__ == "__main__":
    main()
