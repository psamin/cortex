"""ORACLE reference values Q_ref(s, a). NOT deployable.

Q_ref(s, a) is the mean environment return after taking `a` in the true current
state and then following the privileged SimBFS planner until termination or
`horizon` steps. It is estimated from `n` independent stochastic rollouts, so the
labels are approximate. This computation is excluded from deployment cost.
"""
from __future__ import annotations

import numpy as np

from dreamcache.environments.cloning import clone_env
from dreamcache.speculation.evaluator import EvalConfig
from dreamcache.speculation.executor import run_branch

REF_EVAL = EvalConfig(kind="env", success_bonus=0.0, failure_penalty=0.0, step_cost=0.0)


def q_ref(env, legal, history, n=16, horizon=100, seed=0) -> dict[int, float]:
    rng = np.random.default_rng(10_000_019 + seed)
    seeds = rng.integers(0, 2**31 - 1, size=n)  # common random numbers across actions
    return {a: float(np.mean([run_branch(clone_env(env), a, history, horizon, legal, "sim_bfs",
                                         int(s), REF_EVAL)["utility"] for s in seeds]))
            for a in legal}


def regret(q: dict[int, float], action: int) -> float:
    return max(q.values()) - q[action]
