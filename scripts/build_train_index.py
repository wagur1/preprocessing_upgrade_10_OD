"""Build the canonical hash-split index (train/val/test) for the additive round.

The u6_big4 eval pipeline re-derives its test set from a MOUNT-INDEPENDENT
clip id — ``<class>/<file>.mp4`` — rather than list position (``os.walk`` order
is unstable across machines/mounts):

    test = clips with md5(key)[0:8]  % 10 == 0   -> the canonical 1159
           (test-set fingerprint 30f083f8520a)
    val  = clips with md5(key)[0:8]  % 10 == 1   -> ~1k, deterministic
    train = every other mapped clip               -> ~8.6k

This script produces the TRAINING-side index in the harness's
``prepare_3gb`` format ({"meta", "train", "val", "test"} lists of
{"path", "label", "class", "bytes"}) so training can never touch the canonical
test clips, and the val split is reproducible without shipping files around.

Class mapping matches ``prepare_3gb`` exactly (folder name -> Kinetics-400
label index via ``kinetics_category_index``; unmappable classes dropped),
because the canonical test set is defined over the MAPPED clips only.

Usage
-----
    python scripts/build_train_index.py \
        --root /kaggle/input/kinetics-train-5per \
        --out data/index/kinetics_hash_split.json \
        --assert-fingerprint 30f083f8520a
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.prepare_3gb import _find_videos
from src.tasks.action_recognition import _canon, kinetics_category_index

CANONICAL_TEST_FINGERPRINT = "30f083f8520a"


def clip_key(path: str) -> str:
    """Mount-independent clip id: '<class>/<file>.mp4' (u6_big4 cell.txt rule)."""
    parts = path.replace("\\", "/").rstrip("/").split("/")
    return "/".join(parts[-2:])


def _md5_int(key: str, lo: int = 0, hi: int = 8) -> int:
    return int(hashlib.md5(key.encode("utf-8")).hexdigest()[lo:hi], 16)


def build(root: str, out: str, backbone: str = "r3d_18") -> dict:
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"dataset root not found: {root}")

    name_to_idx = kinetics_category_index(backbone)
    by_class = _find_videos(root_path)
    mapped = {cls: vids for cls, vids in by_class.items()
              if _canon(cls) in name_to_idx}
    if not mapped:
        raise RuntimeError("no dataset class folder matched a Kinetics-400 label")

    train, val, test = [], [], []
    for cls, vids in mapped.items():
        label = name_to_idx[_canon(cls)]
        for v in vids:
            try:
                size = v.stat().st_size
            except OSError:
                continue
            if size <= 0:
                continue
            rec = {"path": str(v), "label": label, "class": cls, "bytes": size}
            r = _md5_int(clip_key(rec["path"])) % 10
            (test if r == 0 else val if r == 1 else train).append(rec)

    # The canonical fingerprint is md5 over the comma-joined SORTED keys of the
    # full test set (cell.txt: sorted by key, joined with ","). Only asserted
    # when the whole dataset is present -- a partial mount legitimately has a
    # different (subset) fingerprint.
    test_sorted = sorted(test, key=lambda r: clip_key(r["path"]))
    fingerprint = hashlib.md5(
        ",".join(clip_key(r["path"]) for r in test_sorted).encode()).hexdigest()[:12]

    index = {
        "meta": {
            "root": str(root_path),
            "split_rule": "md5('<class>/<file>.mp4')[0:8] % 10: 0=test 1=val rest=train",
            "backbone": backbone,
            "n_train": len(train), "n_val": len(val), "n_test": len(test),
            "n_classes": len(mapped),
            "test_fingerprint": fingerprint,
            "unmapped_classes": sorted(set(by_class) - set(mapped)),
        },
        "train": train, "val": val, "test": test,
    }
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(index), encoding="utf-8")
    m = index["meta"]
    print(f"[hash-split] wrote {out}")
    print(f"  train {m['n_train']} / val {m['n_val']} / test {m['n_test']} "
          f"across {m['n_classes']} classes | test fingerprint {fingerprint}")
    if m["unmapped_classes"]:
        print(f"  skipped {len(m['unmapped_classes'])} non-Kinetics class folders")
    return index


def main() -> None:
    p = argparse.ArgumentParser(description="Build the canonical hash-split index.")
    p.add_argument("--root", required=True, help="dataset split dir with class subfolders")
    p.add_argument("--out", default="data/index/kinetics_hash_split.json")
    p.add_argument("--backbone", default="r3d_18")
    p.add_argument("--assert-fingerprint", default=None, metavar="FP",
                   help="hard-fail unless the test-set fingerprint matches "
                        "(use only when the FULL dataset is mounted)")
    a = p.parse_args()
    idx = build(a.root, a.out, a.backbone)
    if a.assert_fingerprint:
        got = idx["meta"]["test_fingerprint"]
        if got != a.assert_fingerprint:
            raise SystemExit(f"[hash-split] FINGERPRINT MISMATCH: got {got}, "
                             f"expected {a.assert_fingerprint} -- the canonical "
                             f"test set is NOT this one; do not train on it")
        print(f"[hash-split] fingerprint OK ({got})")


if __name__ == "__main__":
    main()
