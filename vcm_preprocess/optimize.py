"""Black-box optimizer cho pipeline + codec/task evaluator."""

from __future__ import annotations
from dataclasses import dataclass
import random
from typing import Callable, Dict, Iterable, Optional, Tuple
from .pipeline import PreprocessConfig
from .metrics import bd_rate

Evaluator = Callable[[PreprocessConfig], Dict[str, float]]
RDEvaluator = Callable[[PreprocessConfig], Iterable[Tuple[float, float]]]

@dataclass
class OptimizationResult:
    config: PreprocessConfig
    score: float
    metrics: Dict[str, float]
    evaluations: int

class RandomSearch:
    """Random search co local refinement, reproducible va khong can ML framework."""
    def __init__(self, evaluator: Evaluator, iterations=40, seed=7, weights=(1.0, .15)):
        self.evaluator, self.iterations, self.rng, self.weights = evaluator, iterations, random.Random(seed), weights

    def _score(self, m):
        quality = m.get("task_quality", m.get("map", m.get("miou", 0.0)))
        bitrate = m.get("bitrate_kbps", m.get("bitrate", 0.0))
        return self.weights[0] * quality - self.weights[1] * __import__('math').log1p(max(0, bitrate))

    def run(self, initial: Optional[PreprocessConfig] = None) -> OptimizationResult:
        best_cfg = initial or PreprocessConfig(); best_m = self.evaluator(best_cfg); best = self._score(best_m); n = 1
        for _ in range(max(0, self.iterations - 1)):
            cfg = PreprocessConfig(self.rng.random(), self.rng.random(), self.rng.uniform(.85, 1.2), self.rng.uniform(.8, 1.2), self.rng.uniform(.9, 1.1), False)
            m = self.evaluator(cfg); n += 1; s = self._score(m)
            if s > best: best, best_cfg, best_m = s, cfg, m
        return OptimizationResult(best_cfg, best, best_m, n)


class BDRateSearch:
    """Toi uu truc tiep BD-rate tren ca duong cong RD.

    `evaluator(config)` phai tra ve cac diem (bitrate_kbps, task_quality).
    Score nho hon la tot; `best_bd_rate` am nghia tiet kiem bitrate so voi reference.
    """
    def __init__(self, reference_points: Iterable[Tuple[float, float]], evaluator: RDEvaluator,
                 iterations=40, seed=7, quality_floor: Optional[float] = None,
                 latency_penalty: float = 0.0):
        self.reference = list(reference_points); self.evaluator = evaluator
        self.iterations = iterations; self.rng = random.Random(seed)
        self.quality_floor = quality_floor; self.latency_penalty = latency_penalty

    @staticmethod
    def _sample(rng: random.Random) -> PreprocessConfig:
        return PreprocessConfig(rng.random(), rng.random(), rng.uniform(.85, 1.2),
                                rng.uniform(.8, 1.2), rng.uniform(.9, 1.1), False)

    def run(self, initial: Optional[PreprocessConfig] = None) -> OptimizationResult:
        best_cfg = initial or PreprocessConfig(); best_curve = list(self.evaluator(best_cfg)); n = 1
        best = bd_rate(self.reference, best_curve)
        for _ in range(max(0, self.iterations - 1)):
            cfg = self._sample(self.rng); curve = list(self.evaluator(cfg)); n += 1
            score = bd_rate(self.reference, curve)
            if self.quality_floor is not None:
                score += max(0.0, self.quality_floor - max(q for _, q in curve)) * 100.0
            if score < best:
                best, best_cfg, best_curve = score, cfg, curve
        return OptimizationResult(best_cfg, -best, {"bd_rate_percent": best, "rd_points": best_curve}, n)
