"""Small, dependency-free statistics for sandboxes: Beta posteriors, exact McNemar, start-state bins."""

from __future__ import annotations

from math import comb

import numpy as np

from ..sim.so101 import REGION_JITTER, REGION_XY


def beta_samples(successes: int, n: int, size: int = 20_000, seed: int = 0) -> np.ndarray:
    """Posterior samples of a success rate under a uniform Beta(1, 1) prior."""
    return np.random.default_rng(seed).beta(1 + successes, 1 + n - successes, size)


def interval(samples: np.ndarray, level: float = 0.9) -> tuple[float, float]:
    a = (1 - level) / 2
    lo, hi = np.quantile(samples, [a, 1 - a])
    return round(float(lo), 4), round(float(hi), 4)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value. b = pairs only the first arm solved, c = only the second."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return float(min(1.0, 2 * p))


def start_xy(cube_region: str, cube_offset: tuple[float, float], seed: int) -> tuple[float, float]:
    """The object's start position for a seed, reproducing PickPlaceEnv.reset's draws (first two uniforms)."""
    rng = np.random.default_rng(seed)
    x, y = REGION_XY[cube_region]
    return (x + cube_offset[0] + rng.uniform(-REGION_JITTER, REGION_JITTER),
            y + cube_offset[1] + rng.uniform(-REGION_JITTER, REGION_JITTER))


def start_bin(x: float, y: float, cell: float = 0.01) -> str:
    """A 1 cm grid cell label for an initial object position."""
    return f"{round(x / cell) * cell:.2f},{round(y / cell) * cell:.2f}"
