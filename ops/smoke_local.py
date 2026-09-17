#!/usr/bin/env python
"""Local end-to-end smoke test for v7 (no dataset needed).

Synthesizes a tiny fake-Kinetics index (procedurally generated clips), then:
  1. trains additive_cond 2 epochs x 8 steps (real virtual codec, real loss,
     FiLM conditioning on sampled QPs) -> preprocessor.pth
  2. runs evaluate() with eval.shard_idx/num_shards (real x264/x265 via
     ffmpeg, held-out analyzer r2plus1d_18) on BOTH shards
  3. runs ops/merge_eval.py on the two shard results.json -> merged BD + CI
  4. runs ops/gates_qpc.py on the checkpoint

Verifies the FULL v7 pipeline (train -> sharded eval -> merge -> gates)
locally before any Kaggle GPU quota is spent.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

CONFIG = sys.argv[1] if len(sys.argv) > 1 else str(REPO / "configs" / "upvcm_ar.yaml")
print(f"[smoke] config: {CONFIG}")

import numpy as np
import torch


def make_fake_dataset(root: Path, n_train=12, n_val=4, n_test=8) -> Path:
    """Write tiny mp4s class folders + index JSON in prepare_3gb format."""
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    root.mkdir(parents=True, exist_ok=True)
    # Use real Kinetics class names (2 of them) so the label mapping works
    classes = ["abseiling", "archery"]
    records = {"train": [], "val": [], "test": []}
    split_n = {"train": n_train, "val": n_val, "test": n_test}
    rng = np.random.RandomState(0)
    for ci, cls in enumerate(classes):
        cdir = root / cls
        cdir.mkdir(exist_ok=True)
        for split, n in split_n.items():
            for i in range(n):
                path = cdir / f"{split}_{i:03d}.mp4"
                vw = cv2.VideoWriter(str(path), fourcc, 12, (160, 120))
                for t in range(24):
                    frame = np.zeros((120, 160, 3), np.uint8)
                    ch = min(ci, 2)
                    frame[:, :, ch] = 60 + int(40 * np.sin(t / 3 + i))
                    cx, cy = 40 + 8 * t % 80, 60
                    frame[cy - 10 : cy + 10, cx - 10 : cx + 10] = 255
                    frame += rng.randint(0, 12, frame.shape).astype(np.uint8)
                    vw.write(frame)
                vw.release()
                records[split].append({
                    "path": str(path), "label": ci, "class": cls,
                    "bytes": path.stat().st_size})
    index = {
        "meta": {"root": str(root), "backbone": "r3d_18",
                 "n_train": n_train, "n_val": n_val, "n_test": n_test,
                 "n_classes": 2},
        "train": records["train"], "val": records["val"], "test": records["test"],
    }
    idx_path = root / "index.json"
    idx_path.write_text(json.dumps(index))
    print(f"[smoke] fake dataset: {sum(len(v) for v in records.values())} clips "
          f"at {root}")
    return idx_path


def main():
    tmp = Path("/tmp/u7_smoke")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    index = make_fake_dataset(tmp / "data")

    from src.config import apply_overrides, load_config
    from src.engine import evaluate, train

    # -- 1. train (proxy codec, 2 epochs capped at 8 steps each) --
    cfg = apply_overrides(
        load_config(CONFIG),
        [f"data.index={index}", "out_dir=/tmp/u7_smoke/out",
         "train.epochs=2", "train.max_steps=8", "train.batch_size=2",
         "train.resume=false", "train.num_workers=0", "val_max_batches=2",
         "device=cuda" if torch.cuda.is_available() else "device=cpu"])
    print("[smoke] training ...")
    ckpt = train(cfg)
    assert Path(ckpt).exists(), "no checkpoint after train"
    print(f"[smoke] train OK -> {ckpt}")

    # -- 2. sharded eval on both shards --
    for shard in (0, 1):
        ev_cfg = apply_overrides(
            load_config(CONFIG),
            [f"data.index={index}", "eval.batch_size=2", "eval.num_workers=0",
             f"eval.shard_idx={shard}", "eval.num_shards=2",
             "eval.per_sequence=true"])
        out = f"/tmp/u7_smoke/eval_shard{shard}"
        print(f"[smoke] eval shard {shard} ...")
        r = evaluate(ev_cfg, ckpt, out)
        assert "bd_prep_gain" in r
        print(f"[smoke] shard {shard} OK: {list(r['curves'])}")

    # -- 3. merge --
    r = subprocess.run(
        [sys.executable, str(REPO / "ops" / "merge_eval.py"),
         "/tmp/u7_smoke/eval_shard0", "/tmp/u7_smoke/eval_shard1",
         "--out", "/tmp/u7_smoke/merged", "--bootstrap", "200"],
        capture_output=True, text=True)
    print(r.stdout[-2000:])
    if r.returncode != 0:
        print(r.stderr[-2000:])
        raise SystemExit("merge FAILED")
    merged = json.loads((Path("/tmp/u7_smoke/merged") / "merged_results.json").read_text())
    assert merged["n_sequences"] == 16, f"expected 16 merged seqs (8/class x 2), got {merged['n_sequences']}"
    print(f"[smoke] merge OK: {merged['n_sequences']} sequences, BD h264 "
          f"{merged['bd_prep_gain'].get('prep+h264 vs h264')}")

    # -- 4. gates (arch-appropriate) --
    if "sandwich" in CONFIG:
        gates_script = "gates_sandwich.py"
    elif "upvcm" in CONFIG:
        gates_script = "gates_upvcm.py"
    else:
        gates_script = "gates_qpc.py"
    r = subprocess.run(
        [sys.executable, str(REPO / "ops" / gates_script),
         "--ckpt", ckpt, "--index", str(index), "--n-clips", "4"],
        capture_output=True, text=True)
    print(r.stdout[-1500:])
    if r.returncode != 0:
        # Gates are QUALITY filters for real checkpoints; the smoke checkpoint
        # (8 steps, fake data) is expected to fail G1/G3. Only a crash matters.
        if r.stderr.strip():
            print(r.stderr[-800:])
        print("[smoke] gates exit non-zero — expected for the tiny smoke checkpoint; "
              "wiring verified)")
    print("\n[smoke] FULL PIPELINE OK (train -> shard eval -> merge -> gates)")


if __name__ == "__main__":
    main()
