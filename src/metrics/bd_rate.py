"""Bjontegaard-Delta metrics adapted to a rate-vs-accuracy curve.

Classic BD-Rate compares two codecs by the average bitrate difference at equal
*quality*, where quality is usually PSNR. For machine vision the "quality" axis
is the **task accuracy** instead (higher = better, exactly like PSNR), so the
same piecewise/polynomial integration applies unchanged:

  * ``bd_rate``  : average % bitrate change of *test* vs *anchor* at equal
                   accuracy. Negative => test needs fewer bits for the same
                   accuracy => test is better.
  * ``bd_metric``: average accuracy change at equal bitrate.

Rates are integrated in the log domain (standard). We fit a cubic when there
are >=4 points, else fall back to a lower-order fit, and integrate over the
overlapping accuracy (resp. rate) range.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _prep(rate: Sequence[float], metric: Sequence[float]):
    r = np.asarray(rate, dtype=np.float64)
    m = np.asarray(metric, dtype=np.float64)
    # rate must be > 0 (we integrate log(rate)); drop non-finite/degenerate points
    # so one bad point (e.g. a quality id whose bpp collapsed to 0) can't poison
    # the whole fit with -inf and blow up polyfit's SVD.
    keep = np.isfinite(r) & np.isfinite(m) & (r > 0)
    r, m = r[keep], m[keep]
    order = np.argsort(m)
    return r[order], m[order]


def _fit_order(n: int) -> int:
    if n >= 4:
        return 3
    if n == 3:
        return 2
    return 1


def _safe_polyfit(x: np.ndarray, y: np.ndarray):
    """polyfit that degrades gracefully. Caps the order at (#distinct x - 1) so a
    degenerate axis (all-equal accuracy => rank-deficient Vandermonde) can't raise
    LinAlgError; returns None when a fit is impossible."""
    ndist = len(np.unique(x))
    if ndist < 2:
        return None  # can't fit a curve through a single distinct x
    order = min(_fit_order(len(x)), ndist - 1)
    try:
        return np.polyfit(x, y, order)
    except np.linalg.LinAlgError:
        return None


def bd_rate(
    rate_anchor: Sequence[float],
    metric_anchor: Sequence[float],
    rate_test: Sequence[float],
    metric_test: Sequence[float],
) -> float:
    """Average % bitrate difference (test - anchor) at equal accuracy.

    Returns a percentage; negative means the test pipeline saves bits.
    """
    r1, m1 = _prep(rate_anchor, metric_anchor)
    r2, m2 = _prep(rate_test, metric_test)
    if len(m1) < 2 or len(m2) < 2:
        return float("nan")
    lr1, lr2 = np.log(r1), np.log(r2)

    p1 = _safe_polyfit(m1, lr1)
    p2 = _safe_polyfit(m2, lr2)
    if p1 is None or p2 is None:
        return float("nan")

    lo = max(min(m1), min(m2))
    hi = min(max(m1), max(m2))
    if hi <= lo:
        return float("nan")  # no overlap in accuracy -> BD-Rate undefined

    P1 = np.polyint(p1)
    P2 = np.polyint(p2)
    int1 = np.polyval(P1, hi) - np.polyval(P1, lo)
    int2 = np.polyval(P2, hi) - np.polyval(P2, lo)
    avg_diff = (int2 - int1) / (hi - lo)
    return float((np.exp(avg_diff) - 1.0) * 100.0)


def bd_metric(
    rate_anchor: Sequence[float],
    metric_anchor: Sequence[float],
    rate_test: Sequence[float],
    metric_test: Sequence[float],
) -> float:
    """Average accuracy difference (test - anchor) at equal bitrate."""
    r1, m1 = _prep(rate_anchor, metric_anchor)
    r2, m2 = _prep(rate_test, metric_test)
    if len(m1) < 2 or len(m2) < 2:
        return float("nan")
    lr1, lr2 = np.log(r1), np.log(r2)

    p1 = _safe_polyfit(lr1, m1)
    p2 = _safe_polyfit(lr2, m2)
    if p1 is None or p2 is None:
        return float("nan")

    lo = max(min(lr1), min(lr2))
    hi = min(max(lr1), max(lr2))
    if hi <= lo:
        return float("nan")

    P1 = np.polyint(p1)
    P2 = np.polyint(p2)
    int1 = np.polyval(P1, hi) - np.polyval(P1, lo)
    int2 = np.polyval(P2, hi) - np.polyval(P2, lo)
    return float((int2 - int1) / (hi - lo))


if __name__ == "__main__":
    import math
    # normal: test needs fewer bits at equal accuracy -> negative BD-Rate
    acc = [0.60, 0.68, 0.73, 0.76]
    r_a = [0.10, 0.20, 0.35, 0.60]
    r_t = [0.05, 0.11, 0.20, 0.36]
    assert bd_rate(r_a, acc, r_t, acc) < 0
    # degenerate inputs must NOT raise (the SVD crash we fixed):
    #  - a single collapsed (rate=0) point is dropped; the rest still fit -> finite
    assert math.isfinite(bd_rate(r_a, acc, [0.0, 0.11, 0.20, 0.36], acc))
    #  - unrecoverable cases return nan instead of blowing up
    assert math.isnan(bd_rate(r_a, acc, r_t, [0.7, 0.7, 0.7, 0.7]))       # flat accuracy axis
    assert math.isnan(bd_rate(r_a, acc, [0.1], [0.7]))                    # single point
    assert math.isnan(bd_rate(r_a, acc, [0.0, 0.0, 0.0, 0.0], acc))       # all rates 0
    # bd_metric fits accuracy over log-rate: a flat accuracy curve is a valid fit
    # (finite), an all-collapsed rate axis is not (nan).
    assert math.isfinite(bd_metric(r_a, acc, r_t, [0.7, 0.7, 0.7, 0.7]))
    assert math.isnan(bd_metric(r_a, acc, [0.0, 0.0, 0.0, 0.0], acc))
    print("bd_rate self-check passed (normal fit + degenerate cases return nan)")
