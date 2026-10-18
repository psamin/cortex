"""Cross-entropy method over skill corrections (episodic RL, common random numbers per iteration).

Defaults follow docs/RL_DESIGN.md: population 32, 20% elites, at most 15 iterations, smoothing
plus a std floor so the distribution doesn't collapse early. Stopping rules:
  * "target"   the elite candidates' success lower Wilson bound reaches the target
  * "plateau"  over 3 iterations, elite success gained < 2 pp AND elite mean return gained < 0.01
               (so a search still climbing towards its first successes keeps going)
  * "converged" stalled with the elites already at or above the target rate
  * "budget"   iterations exhausted
  * "stopped"  should_stop() returned True
``evaluate(params, seeds) -> list[(return, success)]`` does the simulation.
``after_iteration(stats) -> bool`` may widen the domain (ADR); return True if it did.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from ..services.simulation_runner import wilson
from .types import IterationStats, SkillParam

Evaluate = Callable[[dict[str, float], list[int]], list[tuple[float, bool]]]
PLATEAU_PP, PLATEAU_RETURN, PLATEAU_ITERS = 0.02, 0.01, 3


def cem(params: list[SkillParam], evaluate: Evaluate, iterations: int, population: int, episodes: int,
        seed_base: int, target: float = 0.8, on_iteration: Callable[[IterationStats], None] | None = None,
        after_iteration: Callable[[IterationStats], bool] | None = None, elite_frac: float = 0.2,
        smoothing: float = 0.7, should_stop: Callable[[], bool] | None = None, rng_seed: int = 0,
        init: dict[str, float] | None = None) -> tuple[dict[str, float], list[IterationStats], str]:
    rng = np.random.default_rng(rng_seed)
    names = [p.name for p in params]
    lo = np.array([p.lo for p in params])
    hi = np.array([p.hi for p in params])
    mean = np.array([np.clip((init or {}).get(p.name, p.init), p.lo, p.hi) for p in params], dtype=float)
    std = (hi - lo) / 4
    floor = (hi - lo) * 0.02
    n_elite = max(2, int(round(population * elite_frac)))
    best_ret, best = -np.inf, dict(zip(names, mean.tolist()))
    history: list[IterationStats] = []
    elite_rates: list[float] = []
    elite_returns: list[float] = []
    t0, reason = time.time(), "budget"
    for it in range(iterations):
        if should_stop and should_stop():
            reason = "stopped"
            break
        seeds = [seed_base + it * 1000 + k for k in range(episodes)]  # same scenes for every candidate
        cands = np.clip(rng.normal(mean, std, size=(population, len(params))), lo, hi)
        cands[0] = mean  # always re-measure the current mean
        scores, wins = [], []
        for c in cands:
            res = evaluate(dict(zip(names, c.tolist())), seeds)
            scores.append(float(np.mean([r for r, _ in res])))
            wins.append(sum(s for _, s in res))
        order = np.argsort(scores)[::-1]
        elite_idx = order[:n_elite]
        elite = cands[elite_idx]
        mean = smoothing * elite.mean(axis=0) + (1 - smoothing) * mean
        std = np.maximum(smoothing * elite.std(axis=0) + (1 - smoothing) * std, floor)
        if scores[order[0]] > best_ret:
            best_ret, best = scores[order[0]], dict(zip(names, cands[order[0]].tolist()))
        n_all = population * episodes
        e_wins, e_n = int(sum(wins[i] for i in elite_idx)), n_elite * episodes
        e_ci = wilson(e_wins, e_n)
        st = IterationStats(iteration=it, episodes=n_all, mean_return=round(float(np.mean(scores)), 4),
                            success_rate=round(sum(wins) / n_all, 4), ci95=wilson(int(sum(wins)), n_all),
                            elite_success_rate=round(e_wins / e_n, 4),
                            best_return=round(best_ret, 4), best_params={k: round(v, 5) for k, v in best.items()},
                            elapsed_s=round(time.time() - t0, 1))
        history.append(st)
        elite_rates.append(e_wins / e_n)
        elite_returns.append(float(np.mean([scores[i] for i in elite_idx])))
        if on_iteration:
            on_iteration(st)
        widened = after_iteration(st) if after_iteration else False
        if e_ci[0] >= target and not widened:
            reason = "target"
            break
        if (not widened and len(elite_rates) > PLATEAU_ITERS
                and elite_rates[-1] - elite_rates[-1 - PLATEAU_ITERS] < PLATEAU_PP
                and elite_returns[-1] - elite_returns[-1 - PLATEAU_ITERS] < PLATEAU_RETURN):
            # Stalled: "converged" if the elites already succeed at the target rate (too few elite episodes for
            # the lower bound to clear it), else a genuine plateau below target.
            reason = "converged" if elite_rates[-1] >= target else "plateau"
            break
    return best, history, reason
