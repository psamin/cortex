"""UCT Monte Carlo Tree Search over cloned MiniGrid simulators. SIMULATOR-ASSISTED.

Uses the open-source `mcts` package (pbsinclair42/MCTS, MIT) unchanged; this file only
adapts a MiniGrid clone to its state interface (getPossibleActions / takeAction /
isTerminal / getReward). Each tree node holds one sampled successor (determinized
open-loop expansion), so stochastic obstacle motion is sampled once per node; this is
a known limitation of vanilla UCT on stochastic domains and is stated in the report.
Rewards are the same utility as the rollout evaluator, squashed to [0, 1] for UCB.
"""
from __future__ import annotations

import random
import time

import numpy as np
from mcts import mcts as UCT

from dreamcache.environments.cloning import clone_env, reseed
from dreamcache.environments.minigrid_adapter import is_failure, is_success
from dreamcache.speculation.evaluator import EvalConfig, snapshot, subgoal_distance, utility
from dreamcache.speculation.rollouts import RolloutPolicy
from dreamcache.types import Cost


class _Counter:
    steps = 0


class SimState:
    def __init__(self, sim, legal, depth, max_depth, ecfg, start, d0, total_r=0.0, done=False,
                 succ=False, fail=False, rng=None, counter=None):
        self.sim, self.legal, self.depth, self.max_depth = sim, legal, depth, max_depth
        self.ecfg, self.start, self.d0 = ecfg, start, d0
        self.total_r, self.done, self.succ, self.fail = total_r, done, succ, fail
        self.rng, self.counter = rng, counter

    def getPossibleActions(self):
        return list(self.legal)

    def takeAction(self, a):
        s = clone_env(self.sim)
        reseed(s, int(self.rng.integers(0, 2**31 - 1)))
        _, r, term, trunc, _ = s.step(a)
        self.counter.steps += 1
        return SimState(s, self.legal, self.depth + 1, self.max_depth, self.ecfg, self.start, self.d0,
                        self.total_r + float(r), term or trunc, is_success(r, term), is_failure(r, term),
                        self.rng, self.counter)

    def isTerminal(self):
        return self.done or self.depth >= self.max_depth

    def getReward(self):
        u = utility(self.ecfg, self.start, self.sim, self.total_r, self.depth, self.succ, self.fail, self.d0)
        span = self.ecfg.success_bonus + self.ecfg.failure_penalty + 2.0
        return float(np.clip((u + self.ecfg.failure_penalty + 1.0) / span, 0.0, 1.0))


def mcts_decide(env, legal, iterations: int, max_depth: int, ecfg: EvalConfig, seed: int,
                rollout_kind: str = "sim_bfs"):
    t0 = time.perf_counter()
    c0 = time.process_time()
    random.seed(seed)  # the package uses the global `random` module
    rng = np.random.default_rng(seed)
    counter = _Counter()
    root_sim = clone_env(env)
    root = SimState(root_sim, legal, 0, max_depth, ecfg, snapshot(root_sim),
                    subgoal_distance(root_sim) if ecfg.kind == "shaped" else None, rng=rng, counter=counter)
    pol = RolloutPolicy(rollout_kind, legal, seed)

    def rollout(state):
        while not state.isTerminal():
            state = state.takeAction(pol.act(state.sim, state.sim.gen_obs(), []))
        return state.getReward()

    tree = UCT(iterationLimit=iterations, rolloutPolicy=rollout)
    action = tree.search(initialState=root)
    stats = {int(a): {"visits": n.numVisits, "value": n.totalReward / max(1, n.numVisits)}
             for a, n in tree.root.children.items()}
    cost = Cost(sim_steps=counter.steps, sim_cpu_seconds=time.process_time() - c0, branches=iterations,
                wall_seconds=time.perf_counter() - t0)
    return int(action), stats, cost
