"""Build a <= 3 GB balanced clip index from a Kinetics-style dataset.

Kaggle's ``rohanmallick/kinetics-train-5per`` (and Kinetics generally) is laid
out as ``<root>/<split>/<class_name>/<video>.mp4``. This script:

  1. scans every video under the root and groups it by class folder,
  2. maps each class folder name to the frozen analyzer's Kinetics-400 label
     index (via ``kinetics_category_index``); unmapped classes are reported and
     skipped so training labels always match the pretrained model,
  3. greedily selects videos **round-robin across classes** until the total
     byte size reaches the cap (default 3 GB) -- balanced, not front-loaded,
  4. splits the selection into train/val/test and writes an index JSON.

Nothing is copied or transcoded: the index just points at the original files,
so the 3 GB cap bounds what we *read*, and the on-disk dataset is untouched.

Usage
-----
    python -m src.data.prepare_3gb \
        --root /kaggle/input/kinetics-train-5per \
    --out data/index/kinetics_3gb.json \
        --cap-gb 3 --val-frac 0.1 --test-frac 0.1
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from ..tasks.action_recognition import _canon, kinetics_category_index

_VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".m4v"}


def _find_videos(root: Path) -> Dict[str, List[Path]]:
    """Group video files by their immediate parent folder name (the class)."""
    by_class: Dict[str, List[Path]] = defaultdict(list)
    for dirpath, _dirs, files in os.walk(root):
        cls = Path(dirpath).name
        for fn in files:
            if Path(fn).suffix.lower() in _VIDEO_EXTS:
                by_class[cls].append(Path(dirpath) / fn)
    return by_class


def build_index(
    root: str,
    out: str,
    cap_gb: float = 3.0,
    val_frac: float = 0.1,
    backbone: str = "r3d_18",
    seed: int = 0,
    test_frac: float = 0.1,
) -> dict:
    if not 0.0 <= val_frac < 1.0 or not 0.0 <= test_frac < 1.0:
        raise ValueError("val_frac and test_frac must be in [0, 1)")
    if val_frac + test_frac >= 1.0:
        raise ValueError("val_frac + test_frac must be less than 1")
    random.seed(seed)
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"dataset root not found: {root}")

    name_to_idx = kinetics_category_index(backbone)
    by_class = _find_videos(root_path)
    if not by_class:
        raise RuntimeError(f"no videos found under {root}")

    # Resolve class folders -> Kinetics label indices; drop the unmappable.
    mapped: Dict[str, int] = {}
    unmapped: List[str] = []
    for cls in by_class:
        idx = name_to_idx.get(_canon(cls))
        if idx is None:
            unmapped.append(cls)
        else:
            mapped[cls] = idx
    if not mapped:
        raise RuntimeError(
            "no dataset class folder matched a Kinetics-400 label; check --root "
            "points at the split dir containing class subfolders"
        )

    # Shuffle within each class for an unbiased balanced pick.
    for cls in mapped:
        random.shuffle(by_class[cls])

    cap_bytes = int(cap_gb * (1024 ** 3))
    cursor = {cls: 0 for cls in mapped}
    classes = sorted(mapped)
    selected: List[dict] = []
    total = 0
    progress = True
    while total < cap_bytes and progress:
        progress = False
        for cls in classes:
            vids = by_class[cls]
            j = cursor[cls]
            if j >= len(vids):
                continue
            path = vids[j]
            cursor[cls] += 1
            progress = True
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size <= 0:
                continue
            if total + size > cap_bytes and selected:
                # Cap reached; stop adding (keep at least something per pass).
                total = cap_bytes
                progress = False
                break
            selected.append(
                {"path": str(path), "label": mapped[cls], "class": cls, "bytes": size}
            )
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
            "backbone": backbone,
            "selected_bytes": sum(s["bytes"] for s in selected),
            "selected_gb": round(sum(s["bytes"] for s in selected) / 1024 ** 3, 3),
            "n_total": len(selected),
            "n_train": len(train),
            "n_val": len(val),
            "n_test": len(test),
            "n_classes": len({s["class"] for s in selected}),
            "unmapped_classes": sorted(unmapped),
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
    print(f"[prepare_3gb] wrote {out_path}")
    print(
        f"  selected {m['n_total']} clips ({m['selected_gb']} GB) "
        f"across {m['n_classes']} classes -> train {m['n_train']} / "
        f"val {m['n_val']} / test {m['n_test']}"
    )
    if unmapped:
        print(f"  skipped {len(unmapped)} non-Kinetics class folders")
    return index


def main() -> None:
    p = argparse.ArgumentParser(description="Build a <=3GB balanced clip index.")
    p.add_argument("--root", required=True, help="dataset split dir with class subfolders")
    p.add_argument("--out", default="data/index/kinetics_3gb.json")
    p.add_argument("--cap-gb", type=float, default=3.0)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--backbone", default="r3d_18")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    build_index(a.root, a.out, a.cap_gb, a.val_frac, a.backbone, a.seed, a.test_frac)


if __name__ == "__main__":
    main()
