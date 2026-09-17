#!/usr/bin/env python
"""Evaluate prep+CompressAI vs bare H.264 / H.265 and report BD-Rate.

    python evaluate.py --config configs/action_recognition.yaml \
        --ckpt outputs/checkpoints/preprocessor.pth \
        data.index=data/index/kinetics_3gb.json

Writes results.json, curves.csv and rate_accuracy.png to the eval output dir.
"""

from __future__ import annotations

import argparse

from src.config import apply_overrides, load_config
from src.engine import evaluate


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("overrides", nargs="*", help="dotted key=value config overrides")
    a = p.parse_args()

    cfg = apply_overrides(load_config(a.config), a.overrides)
    evaluate(cfg, a.ckpt, a.out)


if __name__ == "__main__":
    main()
