"""Cross-entropy method over skill corrections (episodic RL, common random numbers per iteration).

Each iteration samples a population around the current mean, evaluates every candidate on
the same seeds, refits mean/std to the elite fraction (with smoothing), and reports measured
stats. Pure orchestration: ``evaluate(params, seeds) -> list[(return, success)]`` does the sim.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from ..services.simulation_runner import wilson
from .types import IterationStats, SkillParam

Evaluate = Callable[[dict[str, float], list[int]], list[tuple[float, bool]]]


def cem(params: list[SkillParam], evaluate: Evaluate, iterations: int, population: int, episodes: int,
        seed_base: int, on_iteration: Callable[[IterationStats], None] | None = None,
        elite_frac: float = 0.25, smoothing: float = 0.7, should_stop: Callable[[], bool] | None = None,
        rng_seed: int = 0) -> tuple[dict[str, float], list[IterationStats]]:
    rng = np.random.default_rng(rng_seed)
    lo = np.array([p.lo for p in params])
    hi = np.array([p.hi for p in params])
    mean = np.array([np.clip(p.init, p.lo, p.hi) for p in params], dtype=float)
    std = (hi - lo) / 4
    names = [p.name for p in params]
    n_elite = max(1, int(round(population * elite_frac)))
    best_ret, best = -np.inf, dict(zip(names, mean.tolist()))
    history: list[IterationStats] = []
    t0 = time.time()
    for it in range(iterations):
        if should_stop and should_stop():
            break
        seeds = [seed_base + it * 1000 + k for k in range(episodes)]  # same scenes for every candidate
        cands = np.clip(rng.normal(mean, std, size=(population, len(params))), lo, hi)
        cands[0] = mean  # always re-measure the current mean
        scores, succ, n = [], 0, 0
        for c in cands:
            res = evaluate(dict(zip(names, c.tolist())), seeds)
            scores.append(float(np.mean([r for r, _ in res])))
            succ += sum(s for _, s in res)
            n += len(res)
        order = np.argsort(scores)[::-1]
        elite = cands[order[:n_elite]]
        mean = smoothing * elite.mean(axis=0) + (1 - smoothing) * mean
        std = np.maximum(smoothing * elite.std(axis=0) + (1 - smoothing) * std, (hi - lo) * 0.01)
        if scores[order[0]] > best_ret:
            best_ret, best = scores[order[0]], dict(zip(names, cands[order[0]].tolist()))
        st = IterationStats(iteration=it, episodes=n, mean_return=round(float(np.mean(scores)), 4),
                            success_rate=round(succ / n, 4), ci95=wilson(succ, n), best_return=round(best_ret, 4),
                            best_params={k: round(v, 5) for k, v in best.items()}, elapsed_s=round(time.time() - t0, 1))
        history.append(st)
        if on_iteration:
            on_iteration(st)
    return best, history
