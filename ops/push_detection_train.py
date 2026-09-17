#!/usr/bin/env python
"""Push COCO OD training with held-out train2017 validation and explicit resume.

val2017 is reserved for probes/test. To resume, mount a dataset using --dataset
and pass its exact checkpoint directory via --resume-checkpoint-dir. Both best
and last from the same run are required; no global search or mtime guessing.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
COCO_DATASET = "awsaf49/coco-2017-dataset"

# Executed inside the generated notebook; tests execute this exact code locally.
RESTORE = r'''
import copy, json, math, os, shutil
from pathlib import Path
import torch
from src.config import apply_overrides, load_config
from src.data.coco_det import validate_od_index

cfg = apply_overrides(load_config(os.environ["OD_CONFIG"]),
                      json.loads(os.environ["OD_OVERRIDES"]))
cfg["data"]["od_split_fingerprint"] = validate_od_index(
    json.loads(Path(cfg["data"]["index"]).read_text()))
dst = Path(cfg["out_dir"]) / "checkpoints"
source = os.environ.get("OD_RESUME_DIR", "")
names = ("preprocessor.pth", "preprocessor_last.pth")

def identity(config):
    result = copy.deepcopy(config)
    # Location/runtime and eval settings do not change training identity. Target
    # epochs may be extended; the engine keeps the saved optimizer/LR schedule.
    for key in ("out_dir", "device", "eval"):
        result.pop(key, None)
    result.get("data", {}).pop("index", None)
    for key in ("resume", "epochs"):
        result.get("train", {}).pop(key, None)
    return result

if not source:
    if any((dst / name).exists() for name in names):
        raise ValueError("Existing local checkpoints: choose an explicit --resume-checkpoint-dir "
                         "or a new out_dir for a fresh run")
    print("[resume] fresh run (no explicit checkpoint directory)")
else:
    src = Path(source)
    # An exact directory only: a dataset root with nested runs is ambiguous.
    if not src.is_dir() or not all((src / name).is_file() for name in names):
        raise ValueError("Resume requires an exact checkpoint directory containing paired "
                         "preprocessor.pth (historical best) and preprocessor_last.pth; "
                         "no recursive search or best-from-last fallback")
    states = [torch.load(src / name, map_location="cpu", weights_only=True) for name in names]
    for name, state in zip(names, states):
        saved = state.get("cfg", {})
        if not saved.get("data", {}).get("od_split_fingerprint"):
            raise ValueError("Incompatible legacy checkpoint: missing OD split identity; start fresh")
        if identity(saved) != identity(cfg):
            raise ValueError(f"Incompatible run/config/split identity in {name}; start fresh or "
                             "use the original run-id, config and split")
        if not all(k in state for k in ("model", "epoch", "global_step", "best_val", "no_improve")):
            raise ValueError(f"Incomplete training checkpoint: {name}")
    best, last = states
    if (not math.isfinite(best["best_val"]) or best["best_val"] != last["best_val"]
            or best["epoch"] > last["epoch"] or best["global_step"] > last["global_step"]
            or best["no_improve"] != 0):
        raise ValueError("Best/last are not a consistent historical checkpoint pair")
    if src.resolve() != dst.resolve():
        if any((dst / name).exists() for name in names):
            raise ValueError("Destination already has checkpoints; refusing to overwrite another pair")
        dst.mkdir(parents=True, exist_ok=True)
        for name in names:
            shutil.copy2(src / name, dst / name)
    print(f"[resume] preserved paired best and last from {src}")
'''

BASH = r'''%%bash
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
pip install -q pycocotools

TR=$(find /kaggle/input -maxdepth 8 -type d -name train2017 | head -1 || true)
VA=$(find /kaggle/input -maxdepth 8 -type d -name val2017 | head -1 || true)
ANN_TR=$(find /kaggle/input -maxdepth 8 -name 'instances_train2017.json' | head -1 || true)
ANN_VA=$(find /kaggle/input -maxdepth 8 -name 'instances_val2017.json' | head -1 || true)
if [ -z "$TR" ] || [ -z "$VA" ] || [ -z "$ANN_TR" ] || [ -z "$ANN_VA" ]; then
  echo "[dettrain] ERROR: COCO inputs missing" >&2
  exit 1
fi

# Always validate, including an index left by a previous notebook session.
TRAIN_DIR="$TR" TRAIN_ANN="$ANN_TR" TEST_DIR="$VA" TEST_ANN="$ANN_VA" \
INDEX=__INDEX__ N_TRAIN=__N_TRAIN__ N_VAL=__N_VAL__ SEED=__SEED__ python - <<'PY'
import os
from src.data.coco_det import prepare_od_index
prepare_od_index(os.environ["TRAIN_DIR"], os.environ["TRAIN_ANN"],
                 os.environ["TEST_DIR"], os.environ["TEST_ANN"], os.environ["INDEX"],
                 int(os.environ["N_TRAIN"]), int(os.environ["N_VAL"]), int(os.environ["SEED"]))
PY

export OD_CONFIG=__CONFIG__
export OD_OVERRIDES=__OVERRIDES_JSON__
export OD_RESUME_DIR=__RESUME_DIR__
python - <<'PY'
__RESTORE__
PY

OUT_DIR=__OUT_DIR__
python train.py --config __CONFIG__ __OVERRIDES__
echo "[dettrain] session done; artifacts under $OUT_DIR"
ls -la "$OUT_DIR/checkpoints"
'''


def render_notebook(a):
    # Use the same parser/coercion as train.py, and round-trip each token through
    # shlex.join rather than injecting raw overrides into the notebook shell.
    from src.config import apply_overrides, load_config

    if a.n_train <= 0 or a.n_val <= 0 or a.epochs <= 0:
        raise ValueError("n-train, n-val and epochs must be positive")
    user_tokens = shlex.split(a.overrides)
    tokens = [f"train.epochs={a.epochs}", *user_tokens]
    cfg = apply_overrides(load_config(str(REPO / a.config)), tokens)
    if cfg.get("task", {}).get("name") != "object_detection":
        raise ValueError("Detection pusher requires task.name=object_detection")
    if cfg.get("train", {}).get("finetune"):
        raise ValueError("This pusher supports fresh training or exact resume, not finetune")
    # Split sizes are controlled by the named CLI options, not stale YAML caps.
    for key, count in (("max_train_items", a.n_train), ("max_val_items", a.n_val)):
        if any(t.split("=", 1)[0] == f"data.{key}" for t in user_tokens):
            if cfg["data"][key] != count:
                raise ValueError(f"data.{key} must match the requested split count")
        tokens.append(f"data.{key}={count}")
    if any(t.split("=", 1)[0] in ("train.resume", "train.od_run_id", "data.od_split_fingerprint")
           for t in user_tokens):
        raise ValueError("Use --resume-checkpoint-dir/--run-id; split identity is computed, not overridden")
    tokens.extend([f"train.resume={str(bool(a.resume_checkpoint_dir)).lower()}",
                   f"train.od_run_id={a.run_id or a.slug}"])
    cfg = apply_overrides(cfg, tokens)
    if not isinstance(cfg.get("out_dir"), str) or not cfg["out_dir"]:
        raise ValueError("out_dir must be a nonempty path")
    if int(cfg["train"]["epochs"]) <= 0:
        raise ValueError("train.epochs must be positive")
    replacements = {
        "COMMIT": shlex.quote(a.commit), "CONFIG": shlex.quote(a.config),
        "OUT_DIR": shlex.quote(cfg["out_dir"]), "INDEX": shlex.quote(cfg["data"]["index"]),
        "N_TRAIN": str(a.n_train), "N_VAL": str(a.n_val), "SEED": str(int(cfg.get("seed", 0))),
        "OVERRIDES": shlex.join(tokens), "OVERRIDES_JSON": shlex.quote(json.dumps(tokens)),
        "RESUME_DIR": shlex.quote(a.resume_checkpoint_dir or ""), "RESTORE": RESTORE,
    }
    src = BASH
    for key, value in replacements.items():
        src = src.replace(f"__{key}__", value)
    return {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {},
                       "outputs": [], "source": src.splitlines(keepends=True)}],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--account", default=os.environ.get("KAGGLE_ACCOUNT", "dngbolm"))
    ap.add_argument("--config", default="configs/sandwich_coco_det.yaml")
    ap.add_argument("--dataset", default=COCO_DATASET, help="Comma-separated COCO and optional resume datasets")
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--overrides", default="", help="Shell-quoted key=value tokens; explicit train.epochs wins")
    ap.add_argument("--slug", default="u9-train-cocodet")
    ap.add_argument("--run-id", help="Stable training identity across continuation kernel slugs (default: slug)")
    ap.add_argument("--resume-checkpoint-dir", help="Exact mounted directory containing paired best and last")
    ap.add_argument("--accelerator", default="NvidiaTeslaT4")
    a = ap.parse_args()
    # Script execution puts ops/, not the repository root, on sys.path.
    sys.path.insert(0, str(REPO))
    nb = render_notebook(a)
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
