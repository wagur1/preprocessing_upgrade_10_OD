"""Self-check for the early-stop decision (imports engine, so needs torch)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.engine import _earlystop_update


def main() -> None:
    inf = float("inf")

    # first observation always improves (best starts at +inf), resets counter.
    improved, best, ni, stop = _earlystop_update(1.0, inf, 1e-4, 0, 3)
    assert improved and best == 1.0 and ni == 0 and not stop

    # no improvement grows no_improve; no stop until it reaches patience.
    improved, best, ni, stop = _earlystop_update(1.0, 0.9, 1e-4, 0, 3)
    assert (not improved) and best == 0.9 and ni == 1 and not stop
    improved, best, ni, stop = _earlystop_update(1.0, 0.9, 1e-4, 2, 3)
    assert ni == 3 and stop, "must stop once no_improve reaches patience"

    # patience=0 disables early stopping.
    _, _, _, stop = _earlystop_update(1.0, 0.9, 1e-4, 99, 0)
    assert not stop

    # a drop smaller than min_delta does NOT count as improvement.
    improved, _, ni, _ = _earlystop_update(0.9 - 1e-6, 0.9, 1e-4, 0, 3)
    assert (not improved) and ni == 1

    print("early-stop self-check passed")


if __name__ == "__main__":
    main()
