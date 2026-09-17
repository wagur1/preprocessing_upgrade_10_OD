"""Metric va tinh BD-rate khong phu thuoc numpy/scipy."""

from __future__ import annotations
import math
from typing import Iterable, Sequence, Tuple


def _clean(points: Iterable[Tuple[float, float]]) -> list[Tuple[float, float]]:
    out = [(float(r), float(q)) for r, q in points if r > 0 and math.isfinite(r) and math.isfinite(q)]
    if len(out) < 2:
        raise ValueError("Can it nhat 2 diem (rate, quality)")
    return sorted(out, key=lambda x: x[1])


def _polyfit2(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float, float]:
    """Least squares bac 2 bang Gaussian elimination, co fallback bac 1."""
    n = len(x)
    sx = sum(x); sx2 = sum(v*v for v in x); sx3 = sum(v**3 for v in x); sx4 = sum(v**4 for v in x)
    sy = sum(y); sxy = sum(a*b for a, b in zip(x, y)); sx2y = sum(a*a*b for a, b in zip(x, y))
    a = [[sx4, sx3, sx2, sx2y], [sx3, sx2, sx, sxy], [sx2, sx, n, sy]]
    for i in range(3):
        pivot = max(range(i, 3), key=lambda j: abs(a[j][i]))
        if abs(a[pivot][i]) < 1e-12:
            # Bac 1 neu du lieu khong duoc phong phu.
            den = n * sx2 - sx * sx
            m = (n * sxy - sx * sy) / den if abs(den) > 1e-12 else 0.0
            return 0.0, m, (sy - m * sx) / n
        a[i], a[pivot] = a[pivot], a[i]
        for j in range(i + 1, 3):
            f = a[j][i] / a[i][i]
            for k in range(i, 4): a[j][k] -= f * a[i][k]
    z = [0.0, 0.0, 0.0]
    for i in range(2, -1, -1): z[i] = (a[i][3] - sum(a[i][k] * z[k] for k in range(i + 1, 3))) / a[i][i]
    return z[0], z[1], z[2]


def _integral_poly(coef: Tuple[float, float, float], lo: float, hi: float) -> float:
    a, b, c = coef
    return a * (hi**3 - lo**3) / 3 + b * (hi**2 - lo**2) / 2 + c * (hi - lo)


def bd_rate(reference: Iterable[Tuple[float, float]], test: Iterable[Tuple[float, float]]) -> float:
    """BD-rate (%): test rate thay doi so voi reference tai cung quality.

    Quality co the la mAP, mIoU, Recall, VMAF... Ket qua am la tiet kiem bitrate.
    """
    ref, tst = _clean(reference), _clean(test)
    lo, hi = max(ref[0][1], tst[0][1]), min(ref[-1][1], tst[-1][1])
    if hi <= lo: raise ValueError("Khoang quality giao nhau rong")
    cr = _polyfit2([q for r, q in ref], [math.log(r) for r, q in ref])
    ct = _polyfit2([q for r, q in tst], [math.log(r) for r, q in tst])
    avg = (_integral_poly(ct, lo, hi) - _integral_poly(cr, lo, hi)) / (hi - lo)
    return (math.exp(avg) - 1.0) * 100.0


def bd_quality(reference: Iterable[Tuple[float, float]], test: Iterable[Tuple[float, float]]) -> float:
    """BD-quality (% points): quality thay doi tai cung bitrate (xap xi)."""
    ref, tst = _clean(reference), _clean(test)
    lr = [(math.log(r), q) for r, q in ref]; lt = [(math.log(r), q) for r, q in tst]
    lo, hi = max(lr[0][0], lt[0][0]), min(lr[-1][0], lt[-1][0])
    if hi <= lo: raise ValueError("Khoang bitrate giao nhau rong")
    qr = _polyfit2([x for x, q in lr], [q for x, q in lr])
    qt = _polyfit2([x for x, q in lt], [q for x, q in lt])
    return (_integral_poly(qt, lo, hi) - _integral_poly(qr, lo, hi)) / (hi - lo)


def pareto_score(task_quality: float, bitrate_kbps: float, weights=(1.0, 0.15)) -> float:
    """Ham muc tieu lon hon la tot: quality - weight*log(1+bitrate)."""
    if bitrate_kbps < 0: raise ValueError("bitrate phai khong am")
    return float(weights[0]) * task_quality - float(weights[1]) * math.log1p(bitrate_kbps)
