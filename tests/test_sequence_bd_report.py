"""Checks sequence-level BD output uses a continuous classification metric."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.engine import _sequence_reports


def _point(bpp, prob, top1):
    return {"bpp": bpp, "target_prob": prob, "top1": top1}


def main() -> None:
    # Five continuous target probabilities make per-video BD fitting possible;
    # ordinary per-video top-1 is deliberately retained only as a diagnostic.
    store = {
        "0": {
            "sequence_id": 0,
            "path": "class_a/clip.mp4",
            "class": "class_a",
            "codecs": {
                "h264": {
                    "30": _point(0.25, 0.80, 1), "35": _point(0.15, 0.70, 1),
                    "40": _point(0.09, 0.50, 1), "45": _point(0.06, 0.30, 0),
                    "50": _point(0.04, 0.15, 0),
                },
                "prep+h264": {
                    "30": _point(0.22, 0.81, 1), "35": _point(0.13, 0.72, 1),
                    "40": _point(0.08, 0.53, 1), "45": _point(0.05, 0.32, 0),
                    "50": _point(0.035, 0.16, 0),
                },
            },
        },
    }
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        extra = _sequence_reports(out, store, [30, 35, 40, 45, 50])
        assert (out / "sequence_points.csv").exists()
        assert (out / "sequence_bd_rate.csv").exists()
        assert (out / "sequence_bd_rate.json").exists()
        value = extra["per_sequence"]["0"]["bd_prep_gain"]["prep+h264 vs h264"]
        assert value["metric"] == "target_prob"
        assert value["bd_rate_pct"] is not None and value["bd_rate_pct"] < 0
        with open(out / "sequence_points.csv", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 10 and "top1" in rows[0] and "target_prob" in rows[0]
        report = json.loads((out / "sequence_bd_rate.json").read_text(encoding="utf-8"))
        assert report["n_sequences"] == 1 and report["sequences_with_bd"] == 1
    print("sequence BD report self-check passed")


if __name__ == "__main__":
    main()
