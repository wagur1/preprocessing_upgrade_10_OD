#!/usr/bin/env python
"""Merge sharded eval outputs -> canonical BD-Rate + bootstrap CI.

Reads the per-sequence record JSONs emitted by the eval kernels
(``sequence_bd_rate.json`` contains only summaries; the real payload is the
``sequence_points.csv`` rows / the per-sequence ``codecs`` store). This
script consumes the ``results.json`` of each shard, whose ``extra`` field
carries the full per-sequence store, re-keyed by the mount-independent
sequence_id, plus each shard's own aggregate curve.

Protocol (canonical, unchanged from the v6 lineage):
  * quality axis = dataset top-1 on the merged test set
  * BD-Rate over the QP30-50 overlap window (5 points, cubic fit on
    log-rate, linear accuracy — src/metrics/bd_rate.py)
  * bootstrap CI at the CLIP level (10k resamples, percentile 2.5/97.5)
  * gap rule: prep - anchor top-1 >= -0.05 at EVERY QP, both codecs

Usage:
    python ops/merge_eval.py shard0/eval_qpcshard0 shard1/... [shard2/...] \
        [--out outputs/eval_qpc_merged] [--bootstrap 10000] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics.bd_rate import bd_metric, bd_rate  # noqa: E402

QPS = [30, 35, 40, 45, 50]
CODECS = ("h264", "h265")


def load_sequences(shard_dirs: list[Path]) -> dict:
    """Merge per-sequence point stores keyed by sequence_id (clip key).

    The raw per-QP per-codec points (bpp, top1, target_prob) live in each
    shard's ``sequence_points.csv`` — results.json only carries BD summaries.
    """
    import csv

    seqs: dict = {}
    n_dup = 0
    for d in shard_dirs:
        pts = Path(d) / "sequence_points.csv"
        if not pts.exists():
            raise SystemExit(f"ERROR: {pts} missing; run eval with eval.per_sequence=true")
        with open(pts, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                sid = row["sequence_id"]
                rec = seqs.setdefault(sid, {
                    "sequence_id": sid, "path": row["path"],
                    "class": row["class"], "codecs": {},
                })
                if rec["path"] != row["path"]:
                    n_dup += 1
                codec_rec = rec["codecs"].setdefault(row["codec"], {})
                codec_rec[row["qp"]] = {
                    "bpp": float(row["bpp"]),
                    "top1": int(row["top1"]),
                    "target_prob": float(row["target_prob"]),
                }
    if n_dup:
        print(f"[merge] WARNING: {n_dup} path mismatches across shards")
    return seqs


def dataset_curves(seqs: dict) -> dict:
    """Aggregate per-sequence points into dataset-level (bpp, top1) curves."""
    curves = {}
    for codec in CODECS:
        for method in (codec, f"prep+{codec}", f"sandwich+{codec}"):
            bpp_sum, correct, n = {}, {}, {}
            for rec in (seqs.values() if isinstance(seqs, dict) else seqs):
                points = rec["codecs"].get(method)
                if not points:
                    continue
                for qp in QPS:
                    key = str(qp)
                    if key not in points:
                        continue
                    slot = bpp_sum.setdefault(qp, 0.0)
                    slot2 = correct.setdefault(qp, 0)
                    slot3 = n.setdefault(qp, 0)
                    bpp_sum[qp] = slot + points[key]["bpp"]
                    correct[qp] = slot2 + points[key]["top1"]
                    n[qp] = slot3 + 1
            qps = [q for q in QPS if n.get(q)]
            if len(qps) < 2:
                continue
            curves[method] = {
                "keys": [str(q) for q in qps],
                "bpp": [bpp_sum[q] / n[q] for q in qps],
                "accuracy": [correct[q] / n[q] for q in qps],
                "n": [n[q] for q in qps],
            }
    return curves


def gap_rule(curves: dict) -> dict:
    out = {}
    for codec in CODECS:
        a, p_ = curves.get(codec), curves.get(f"sandwich+{codec}") or curves.get(f"prep+{codec}")
        if not a or not p_:
            continue
        gaps = {q: p_["accuracy"][i] - a["accuracy"][i]
                for i, q in enumerate(a["keys"])}
        out[codec] = {
            "gaps": gaps,
            "min_gap": min(gaps.values()),
            "pass": min(gaps.values()) >= -0.05,
        }
    return out


def bd(curves: dict) -> dict:
    out = {}
    for codec in CODECS:
        for arm in ("prep", "sandwich"):
            a, p_ = curves.get(codec), curves.get(f"{arm}+{codec}")
            if not a or not p_:
                continue
            out[f"{arm}+{codec} vs {codec}"] = {
                "bd_rate_pct": bd_rate(a["bpp"], a["accuracy"], p_["bpp"], p_["accuracy"]),
                "bd_accuracy": bd_metric(a["bpp"], a["accuracy"], p_["bpp"], p_["accuracy"]),
            }
    return out


def bootstrap(seqs: dict, n_boot: int, seed: int) -> dict:
    """Clip-level bootstrap of the same-codec BD-Rate (the headline CI)."""
    rng = random.Random(seed)
    ids = sorted(seqs)
    if len(ids) < 20:
        return {"note": "too few sequences for bootstrap"}
    out = {}
    for codec in CODECS:
        for arm in ("prep", "sandwich"):
            samples = []
            for _ in range(n_boot):
                pick = [rng.choice(ids) for _ in ids]
                sub = [seqs[sid] for sid in pick]  # keep multiplicity
                c = dataset_curves(sub)
                a, p_ = c.get(codec), c.get(f"{arm}+{codec}")
                if not a or not p_ or len(a["bpp"]) < 4:
                    continue
                try:
                    samples.append(bd_rate(a["bpp"], a["accuracy"], p_["bpp"], p_["accuracy"]))
                except Exception:
                    continue
            if not samples:
                out[f"{arm}+{codec}"] = None
                continue
            arr = np.array(samples, dtype=float)
            arr = arr[np.isfinite(arr)]
            out[f"{arm}+{codec}"] = {
                "mean": float(arr.mean()),
                "p2.5": float(np.percentile(arr, 2.5)),
                "p97.5": float(np.percentile(arr, 97.5)),
                "p_bt_less_0": float((arr < 0).mean()),
                "n_boot": len(arr),
            }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("shards", nargs="+", help="per-shard eval output dirs (containing sequence_points.csv)")
    p.add_argument("--out", default="outputs/eval_qpc_merged")
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    dirs = [Path(s) for s in a.shards]
    missing = [str(d) for d in dirs if not (d / "sequence_points.csv").exists()]
    if missing:
        raise SystemExit(f"ERROR: missing shard sequence_points.csv in: {missing}")
    seqs = load_sequences(dirs)
    print(f"[merge] {len(seqs)} unique sequences across {len(dirs)} shards")

    curves = dataset_curves(seqs)
    result = {
        "n_sequences": len(seqs),
        "curves": curves,
        "gap_rule": gap_rule(curves),
        "bd_prep_gain": bd(curves),
    }
    if a.bootstrap > 0:
        print(f"[merge] bootstrapping {a.bootstrap} resamples ...")
        result["bootstrap_ci"] = bootstrap(seqs, a.bootstrap, a.seed)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "merged_results.json").write_text(json.dumps(result, indent=2))

    print("\n=== MERGED CANONICAL RESULT (negative BD-Rate = savings) ===")
    for k, v in result["bd_prep_gain"].items():
        r = v["bd_rate_pct"]
        acc = v["bd_accuracy"]
        print(f"  {k:28s} BD-Rate {r:+.2f}%   BD-Acc {acc:+.4f}")
    for codec, g in result["gap_rule"].items():
        print(f"  gap rule [{codec}]: min gap {g['min_gap']:+.4f} -> {'PASS' if g['pass'] else 'FAIL'}")
    if result.get("bootstrap_ci"):
        for codec, b in result["bootstrap_ci"].items():
            if codec == "note" or not isinstance(b, dict):
                print(f"  CI: {b if codec == 'note' else 'undefined'}")
            else:
                print(f"  CI [{codec}]: {b['p2.5']:+.2f}% .. {b['p97.5']:+.2f}%  "
                      f"(mean {b['mean']:+.2f}%, P(BD<0)={b['p_bt_less_0']:.3f})")
    print(f"\n[merge] wrote {out/'merged_results.json'}")


if __name__ == "__main__":
    main()
