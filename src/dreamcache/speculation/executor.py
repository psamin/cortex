"""Speculative executor: clone -> act -> roll out H steps -> score -> select.

RNG handling. Each sandbox is RESEEDED with a fresh stream, so sandboxes never
replay the real environment's future obstacle motion: the real env's Generator
is never copied into a branch that is used for decisions. With common random
numbers (crn=True), sample j uses the same seed for every candidate, so all
candidates face the same sampled obstacle futures. This reduces variance in the
comparison and avoids giving one branch luckier randomness than another.
"""
from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from dreamcache.environments.cloning import clone_env, reseed
from dreamcache.environments.minigrid_adapter import is_failure, is_success
from dreamcache.speculation.evaluator import EvalConfig, snapshot, subgoal_distance, utility
from dreamcache.speculation.rollouts import RolloutPolicy
from dreamcache.types import Cost


@dataclass
class SpecConfig:
    k: int = 3
    horizon: int = 5
    samples: int = 4          # stochastic rollouts per candidate
    rollout_policy: str = "sim_bfs"
    candidates: str = "llm_rank"
    crn: bool = True
    risk_lambda: float = 0.0  # conservative score = mean - lambda * std
    parallel: int = 0         # 0 = sequential; N = process pool with N workers


def run_branch(sim, action, history, horizon, legal, roll_kind, seed, ecfg, fast=None):
    """One sandbox rollout. `sim` must already be an independent clone."""
    t0 = time.process_time()
    reseed(sim, seed)
    start = snapshot(sim)
    d0 = subgoal_distance(sim) if ecfg.kind == "shaped" else None
    pol = RolloutPolicy(roll_kind, legal, seed, fast)
    pol.cost = Cost()
    hist = list(history)
    obs, r, term, trunc, _ = sim.step(action)
    total, steps = float(r), 1
    succ, fail = is_success(r, term), is_failure(r, term)
    hist.append(action)
    while not (term or trunc) and steps <= horizon:
        a = pol.act(sim, obs, hist)
        obs, r, term, trunc, _ = sim.step(a)
        total += float(r)
        steps += 1
        hist.append(a)
        succ, fail = is_success(r, term), is_failure(r, term)
    u = utility(ecfg, start, sim, total, steps, succ, fail, d0)
    return {"utility": u, "success": succ, "failure": fail, "steps": steps,
            "cpu": time.process_time() - t0, "llm_cost": pol.cost}


def _branch_job(args):
    return run_branch(*args)


_POOLS: dict[int, ProcessPoolExecutor] = {}


def speculate(env, candidates, history, legal, scfg: SpecConfig, ecfg: EvalConfig, seed: int,
              fast_policy=None):
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 2**31 - 1, size=(len(candidates), scfg.samples))
    if scfg.crn:
        base[:] = base[0]
    fp = fast_policy if scfg.rollout_policy == "fast" else None  # not picklable; only needed for LLM rollouts
    jobs = [(clone_env(env), a, history, scfg.horizon, legal, scfg.rollout_policy,
             int(base[i, j]), ecfg, fp)
            for i, a in enumerate(candidates) for j in range(scfg.samples)]
    if scfg.parallel and scfg.rollout_policy != "fast":
        pool = _POOLS.setdefault(scfg.parallel, ProcessPoolExecutor(scfg.parallel))
        outs = list(pool.map(_branch_job, jobs))
    else:
        outs = [run_branch(*j) for j in jobs]
    cost = Cost(branches=len(jobs))
    results = []
    for i, a in enumerate(candidates):
        o = outs[i * scfg.samples:(i + 1) * scfg.samples]
        us = np.array([x["utility"] for x in o])
        score = float(us.mean() - scfg.risk_lambda * us.std())
        results.append({"action": int(a), "score": score, "mean": float(us.mean()),
                        "std": float(us.std()), "p_success": float(np.mean([x["success"] for x in o])),
                        "p_failure": float(np.mean([x["failure"] for x in o]))})
        for x in o:
            cost.sim_steps += x["steps"]
            cost.sim_cpu_seconds += x["cpu"]
            cost += x["llm_cost"]
    best = max(range(len(results)), key=lambda i: (results[i]["score"], -i))
    cost.wall_seconds = time.perf_counter() - t0
    return {"selected_action": results[best]["action"], "candidate_results": results,
            "selected_score": results[best]["score"], "cost": cost,
            "rollout_latency_ms": cost.wall_seconds * 1e3,
            "rollout_compute_seconds": cost.sim_cpu_seconds}
