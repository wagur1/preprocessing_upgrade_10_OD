"""Build a <= 3 GB GOT-10k index for the tracking task.

GOT-10k's train split is ~66 GB, far over the 3 GB Kaggle budget, so we work
from the **val split** (which has ground-truth boxes on every frame, needed for
AUC). This script:

  1. finds every GOT-10k sequence dir (contains ``groundtruth.txt``),
  2. shuffles and greedily adds whole sequences until the cumulative image-byte
     size reaches the cap (default 3 GB) -- sequences are kept whole so tracking
     stays meaningful,
  3. splits the selected sequences into train / val / test, and writes an
     index JSON. Only the test split is used for final AUC/BD-Rate reporting.

Nothing is copied; the index just references the original sequence dirs.

Usage
-----
    python -m src.data.prepare_got10k \
        --root /kaggle/input/got10k/val \
        --out data/index/got10k_3gb.json --cap-gb 3 --val-frac 0.2 --test-frac 0.2
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List

from .got10k import _list_frames, find_sequences

_IMG_EXTS = (".jpg", ".jpeg", ".png")


def _seq_bytes(seq_dir: Path) -> int:
    total = 0
    for p in _list_frames(seq_dir):
        try:
            total += p.stat().st_size
        except OSError:
            pass
    return total


def build_index(
    root: str,
    out: str,
    cap_gb: float = 3.0,
    val_frac: float = 0.3,
    seed: int = 0,
    test_frac: float = 0.2,
) -> dict:
    if not 0.0 <= val_frac < 1.0 or not 0.0 <= test_frac < 1.0:
        raise ValueError("val_frac and test_frac must be in [0, 1)")
    if val_frac + test_frac >= 1.0:
        raise ValueError("val_frac + test_frac must be less than 1")
    random.seed(seed)
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"GOT-10k root not found: {root}")

    seqs = find_sequences(root_path)
    if not seqs:
        raise RuntimeError(
            f"no GOT-10k sequences (dirs with groundtruth.txt) under {root}"
        )
    random.shuffle(seqs)

    cap_bytes = int(cap_gb * (1024 ** 3))
    selected: List[dict] = []
    total = 0
    for seq in seqs:
        size = _seq_bytes(seq)
        if size <= 0:
            continue
        if total + size > cap_bytes and selected:
            break
        n_frames = len(_list_frames(seq))
        selected.append({"dir": str(seq), "bytes": size, "n_frames": n_frames})
        total += size

    random.shuffle(selected)
    n_test = max(1, int(len(selected) * test_frac)) if len(selected) > 2 else 0
    n_val = max(1, int(len(selected) * val_frac)) if len(selected) - n_test > 1 else 0
    test = selected[:n_test]
    val = selected[n_test:n_test + n_val]
    train = selected[n_test + n_val:]

    index = {
        "meta": {
            "root": str(root_path),
            "cap_gb": cap_gb,
            "selected_bytes": total,
            "selected_gb": round(total / 1024 ** 3, 3),
            "n_total": len(selected),
            "n_train": len(train),
            "n_val": len(val),
            "n_test": len(test),
        },
        "train": train,
        "val": val,
        "test": test,
    }
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(index, f)

    m = index["meta"]
    print(f"[prepare_got10k] wrote {out_path}")
    print(
        f"  selected {m['n_total']} sequences ({m['selected_gb']} GB) "
        f"-> train {m['n_train']} / val {m['n_val']} / test {m['n_test']}"
    )
    return index


def main() -> None:
    p = argparse.ArgumentParser(description="Build a <=3GB GOT-10k index.")
    p.add_argument("--root", required=True, help="GOT-10k split dir (e.g. .../val)")
    p.add_argument("--out", default="data/index/got10k_3gb.json")
    p.add_argument("--cap-gb", type=float, default=3.0)
    p.add_argument("--val-frac", type=float, default=0.3)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    build_index(a.root, a.out, a.cap_gb, a.val_frac, a.seed, a.test_frac)


if __name__ == "__main__":
    main()
