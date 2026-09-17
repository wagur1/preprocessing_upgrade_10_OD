#!/usr/bin/env python
"""Train the video preprocessor.

    python train.py --config configs/action_recognition.yaml \
        data.index=data/index/kinetics_3gb.json train.epochs=5

Any ``a.b.c=value`` argument overrides the corresponding config key.
"""

from __future__ import annotations

import argparse

from src.config import apply_overrides, load_config
from src.engine import train


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("overrides", nargs="*", help="dotted key=value config overrides")
    a = p.parse_args()

    cfg = apply_overrides(load_config(a.config), a.overrides)
    train(cfg)


if __name__ == "__main__":
    main()
