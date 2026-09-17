"""Build the CONFIRMATORY holdout index (audit 2026-09-11 finding #4).

The canonical hash-split index covers only the dataset's ``train`` subdir
(10,805 mapped clips: 8636 train / 1010 val / 1159 test). The Kaggle dataset
ships sibling split dirs (~18k more clips) that NO experiment ever touched.

This script builds an eval-only index whose ``test`` split is drawn from
those never-used siblings:

    test_cf = mapped clips from sibling dirs whose clip key is NOT in the
              canonical (train ∪ val ∪ test) key set

…so it is disjoint from everything any model ever trained/selected on.
``train``/``val`` are left EMPTY (eval-only; training reproduction uses the
canonical index).

Usage (inside a kernel, after the canonical root is known):
    python ops/build_confirmatory_index.py \
        --canonical-root <.../kinetics400_5per/train> \
        --dataset-root  <parent of the canonical root> \
        --out data/index/confirmatory.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_train_index import build as build_canonical, clip_key
from src.data.prepare_3gb import _find_videos
from src.tasks.action_recognition import _canon, kinetics_category_index


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--canonical-root", required=True,
                   help="the dir the canonical index was built from (…/train)")
    p.add_argument("--dataset-root", required=True,
                   help="its parent — siblings are searched here")
    p.add_argument("--out", default="data/index/confirmatory.json")
    p.add_argument("--max-test", type=int, default=0,
                   help="0 = keep every never-used clip")
    a = p.parse_args()

    canon = build_canonical(a.canonical_root, "/tmp/_canon_index.json")
    used = {clip_key(r["path"]) for r in canon["train"]}
    used |= {clip_key(r["path"]) for r in canon["val"]}
    used |= {clip_key(r["path"]) for r in canon["test"]}
    print(f"[confirm] canonical used keys: {len(used)}")

    name_to_idx = kinetics_category_index()
    ds_root = Path(a.dataset_root)
    test_cf, roots_used = [], []
    for sib in sorted(ds_root.iterdir()):
        if not sib.is_dir() or sib.resolve() == Path(a.canonical_root).resolve():
            continue
        by_class = _find_videos(sib)
        mapped = {c: v for c, v in by_class.items() if _canon(c) in name_to_idx}
        if not mapped:
            continue
        roots_used.append(sib.name)
        for cls, vids in mapped.items():
            label = name_to_idx[_canon(cls)]
            for v in vids:
                try:
                    size = v.stat().st_size
                except OSError:
                    continue
                if size <= 0:
                    continue
                k = clip_key(str(v))
                if k in used:
                    continue
                test_cf.append({"path": str(v), "label": label,
                                "class": cls, "bytes": size})
    print(f"[confirm] sibling roots used: {roots_used} "
          f"| never-used clips: {len(test_cf)}")
    if len(test_cf) < 500:
        raise SystemExit("[confirm] FATAL: fewer than 500 never-used clips — "
                         "the sibling-dir assumption is wrong; inspect the "
                         "dataset layout before proceeding")
    if a.max_test and len(test_cf) > a.max_test:
        # deterministic subsample by key hash (stable across machines)
        test_cf = sorted(test_cf, key=lambda r: clip_key(r["path"]))[:a.max_test]
    test_cf.sort(key=lambda r: clip_key(r["path"]))
    fingerprint = hashlib.md5(
        ",".join(clip_key(r["path"]) for r in test_cf).encode()).hexdigest()[:12]
    index = {
        "meta": {
            "root": str(ds_root),
            "split_rule": "confirmatory holdout: never-indexed sibling clips, "
                          "disjoint from canonical train/val/test by clip key",
            "n_train": 0, "n_val": 0, "n_test": len(test_cf),
            "n_classes": len({r["label"] for r in test_cf}),
            "test_fingerprint": fingerprint,
            "canonical_test_fingerprint":
                canon["meta"]["test_fingerprint"],
        },
        "train": [], "val": [], "test": test_cf,
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(index), encoding="utf-8")
    print(f"[confirm] wrote {out}: test={len(test_cf)} "
          f"classes={index['meta']['n_classes']} fingerprint={fingerprint}")
    # hard disjointness check
    assert not (used & {clip_key(r['path']) for r in test_cf}), "LEAKAGE"


if __name__ == "__main__":
    main()
