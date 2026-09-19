#!/usr/bin/env python
"""Push the zero-shot detection probe as a Kaggle kernel.

Self-contained (does not touch push_kernel.py's train/eval/probe kinds): one
bash cell that clones the repo at a pinned commit, installs pycocotools, locates
the COCO val2017 subset + the frozen checkpoint in /kaggle/input, and runs
ops/probe_detection.py.

The probe is eval-only: no training, so it cannot burn quota on a bad run, and
stage A aborts before the codec sweep if the edit destroys detection content.

Usage:
  KAGGLE_ACCOUNT=... KAGGLE_API_TOKEN=... python ops/push_detection_probe.py \\
      --commit <sha> --account hieusunday0412 --n-images 500 --size 320 \\
      --stage-a-sizes 128,224,320
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

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

pip install -q pycocotools 2>/dev/null | tail -1 || true
python - <<'PY'
import torch, torchvision
print(f"[detprobe] torch {torch.__version__} cuda {torch.cuda.is_available()} "
      f"torchvision {torchvision.__version__}")
PY

# ---- locate inputs ------------------------------------------------------
# Kaggle mounts datasets at /kaggle/input/datasets/<owner>/<slug>/..., so the
# annotation file sits 6 levels down: maxdepth must clear that (a tighter bound
# silently returned an empty path and cost a session).
VAL=$(find /kaggle/input -maxdepth 8 -type d -name val2017 | head -1 || true)
ANN=$(find /kaggle/input -maxdepth 8 -name 'instances_val2017.json' | head -1 || true)
CKPT=$(find /kaggle/input -maxdepth 8 -name 'preprocessor.pth' | head -1 || true)
echo "[detprobe] val=$VAL"
echo "[detprobe] ann=$ANN"
echo "[detprobe] ckpt=$CKPT"
# A parameter-free probe (background suppression) needs no checkpoint — demanding
# one made the guard abort the whole run after the COCO mount, which is how the
# first R0 attempt and its v10 twin both died at startup.
# Self-discovering scripts (--no-coco-inputs, e.g. probe_openimages.py) locate
# their own inputs; the guard then only checks the checkpoint.
NEEDS_CKPT=__NEEDS_CKPT__
NEEDS_COCO=__NEEDS_COCO__
if [ "$NEEDS_COCO" = "1" ] && { [ -z "$VAL" ] || [ -z "$ANN" ]; }; then
  echo "[detprobe] ERROR: missing COCO inputs (val=${VAL:-none} ann=${ANN:-none}; " \
       "pass --no-coco-inputs for self-discovering scripts)" >&2
  exit 1
fi
if [ "$NEEDS_CKPT" = "1" ] && [ -z "$CKPT" ]; then
  echo "[detprobe] ERROR: missing an input (ckp=${CKPT:-none} needs_ckpt=$NEEDS_CKPT)" >&2
  exit 1
fi

__INVOKE__

echo "[detprobe] done"
ls -la outputs/__OUT_NAME__ || true
"""


MODEL_CALL = """python ops/probe_detection.py \\
    --images "$VAL" --ann "$ANN" --ckpt "$CKPT" \\
    --config configs/sandwich_ar.yaml \\
    --n-images __N_IMAGES__ --size __SIZE__ --stage-a-sizes __STAGE_A_SIZES__ \\
    --qps __QPS__ --bootstrap __BOOTSTRAP__ --stage both --records \\
    --out outputs/__OUT_NAME__"""

SIMPLE_CALL = """python __SCRIPT__ \\
    __INPUT_ARGS__ \\
    --n-images __N_IMAGES__ --size __SIZE__ --qps __QPS__ __EXTRA_ARGS__ \\
    --out outputs/__OUT_NAME__"""


def quoted_ids(path):
    """Local JSON list of ints -> one `--image-ids a,b,c` argument.

    A 500-ID list is ~3.9 kB, which is fine on one line; the earlier failure
    came from a hand-built multi-line invocation, not from the length.
    """
    ids = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(ids, list) or not ids:
        raise ValueError("--ids-file must hold a nonempty JSON list of image IDs")
    if any(type(i) is not int or i < 0 for i in ids):
        raise ValueError("--ids-file IDs must be nonnegative integers")
    if len(ids) != len(set(ids)):
        raise ValueError("--ids-file IDs must be unique")
    if len(ids) > 1000:
        raise ValueError("--ids-file holds more IDs than this probe supports")
    return "--image-ids " + ",".join(str(i) for i in ids)


def single_line(extra_args):
    """Reject anything that would fold into stray `n` tokens inside the cell.

    `__EXTRA_ARGS__` is substituted into one continued shell line. A literal
    `\\n` (double-escaped newline) survives as backslash+n, bash then reads it
    as the argument `n`, and the probe dies with "unrecognized arguments: n n n".
    """
    if "\n" in extra_args or "\r" in extra_args or "\\n" in extra_args:
        raise ValueError("--extra-args must be a single line with no \\n escapes; "
                         "pass long ID lists with --ids-file instead")
    return extra_args


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", required=True)
    ap.add_argument("--account", default=os.environ.get("KAGGLE_ACCOUNT", "hieusunday0412"))
    ap.add_argument("--datasets", default="awsaf49/coco-2017-dataset",
                    help="comma list; the COCO dataset provides val2017 + annotations")
    ap.add_argument("--ckpt-dataset", default="hieusunday0412/u8-bigpost-s1-ckpt",
                    help="private dataset holding the frozen checkpoint (owner-scoped!)")
    ap.add_argument("--n-images", type=int, default=500)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--stage-a-sizes", default="128,224,320")
    ap.add_argument("--qps", default="30,35,40,45,50")
    ap.add_argument("--bootstrap", type=int, default=0,
                    help="inline bootstrap draws; 0 (default) skips it — the CI is "
                         "recomputed offline from --records (CPU seconds, not GPU hours)")
    ap.add_argument("--script", default="ops/probe_detection.py",
                    help="which probe to run; anything other than the checkpoint "
                         "probe uses the simple invocation (no --stage/--records)")
    ap.add_argument("--extra-args", default="")
    ap.add_argument("--ids-file", default=None,
                    help="Local JSON list of COCO image IDs, appended as one "
                         "--image-ids argument (the 500-ID R0 set lives in configs/)")
    ap.add_argument("--out-name", default=None,
                    help="output directory name under outputs/ (default: probe_detection "
                         "for the checkpoint probe, probe_bgsuppress otherwise)")
    ap.add_argument("--slug", default="u10-probe-detection")
    ap.add_argument("--no-coco-inputs", action="store_true",
                    help="the script locates its own inputs (e.g. probe_openimages.py): "
                         "the guard skips val/ann and the invocation omits --images/--ann")
    ap.add_argument("--accelerator", default="NvidiaTeslaT4")
    ap.add_argument("--dry-run", action="store_true", help="Generate and syntax-check only; no network or Kaggle")
    ap.add_argument("--output-dir", type=Path, default=None)
    a = ap.parse_args()

    src = BASH.replace("__COMMIT__", shlex.quote(a.commit))
    is_model_probe = Path(a.script).name == "probe_detection.py"
    out_name = a.out_name or ("probe_detection" if is_model_probe else "probe_bgsuppress")
    extra_args = single_line(a.extra_args)
    if a.ids_file:
        extra_args = " ".join(x for x in (extra_args, quoted_ids(a.ids_file)) if x)
    src = src.replace("__INVOKE__", MODEL_CALL if is_model_probe else SIMPLE_CALL)
    src = src.replace("__SCRIPT__", shlex.quote(a.script)).replace("__EXTRA_ARGS__", extra_args)
    src = src.replace("__OUT_NAME__", shlex.quote(out_name))
    src = src.replace("__NEEDS_CKPT__", "1" if is_model_probe else "0")
    src = src.replace("__NEEDS_COCO__", "0" if a.no_coco_inputs else "1")
    input_args = "" if a.no_coco_inputs else '--images "$VAL" --ann "$ANN" \\'
    src = src.replace("__INPUT_ARGS__", input_args)
    src = src.replace("__N_IMAGES__", str(a.n_images))
    src = src.replace("__SIZE__", str(a.size))
    src = src.replace("__STAGE_A_SIZES__", shlex.quote(a.stage_a_sizes))
    src = src.replace("__QPS__", shlex.quote(a.qps))
    src = src.replace("__BOOTSTRAP__", str(a.bootstrap))

    if re.search(r"__[A-Z][A-Z0-9_]*__", src):
        raise ValueError("Unresolved notebook template placeholder")
    subprocess.run(["bash", "-n"], input="".join(src.splitlines(keepends=True)[1:]), text=True, check=True)

    nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {},
                     "outputs": [], "source": src.splitlines(keepends=True)}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}

    push_dir = a.output_dir or REPO / "ops" / "_push" / a.slug
    push_dir.mkdir(parents=True, exist_ok=True)
    (push_dir / "notebook.ipynb").write_text(json.dumps(nb))
    meta = {"id": f"{a.account}/{a.slug}", "title": a.slug,
            "code_file": "notebook.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
            "enable_internet": True,
            "dataset_sources": [d for d in a.datasets.split(",") if d]
                               + ([a.ckpt_dataset] if is_model_probe and a.ckpt_dataset else []),
            "kernel_sources": [], "competition_sources": [], "model_sources": []}
    (push_dir / "kernel-metadata.json").write_text(json.dumps(meta))

    if a.dry_run:
        print(f"[dry-run] validated bash; wrote notebook and metadata to {push_dir}")
        return

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
