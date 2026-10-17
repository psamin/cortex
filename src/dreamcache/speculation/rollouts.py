"""Rollout policies used INSIDE sandboxes (distinct from the fast LLM proposer).

sim_bfs  SIMULATOR-ASSISTED shortest-path heuristic on the sandbox's full state.
obs_bfs  observation-only heuristic applied to the sandbox's own observations.
random   uniform over legal actions.
fast     the deployable fast policy itself (expensive: one model call per step).
"""
from __future__ import annotations

import numpy as np

from dreamcache.policies.heuristic import ObsBFSPolicy, SimBFSPolicy


class RolloutPolicy:
    def __init__(self, kind: str, legal, seed: int, fast_policy=None):
        self.kind, self.legal = kind, legal
        self.rng = np.random.default_rng(seed)
        self.sim_bfs = SimBFSPolicy()
        self.obs_bfs = ObsBFSPolicy(seed=seed)
        self.fast = fast_policy
        self.cost = None

    def act(self, sim, obs, history) -> int:
        if self.kind == "sim_bfs":
            return self.sim_bfs.act_sim(sim)
        if self.kind == "obs_bfs":
            return self.obs_bfs.act(obs)
        if self.kind == "random":
            return int(self.rng.choice(self.legal))
        if self.kind == "fast":
            d = self.fast.decide(obs, history)
            if self.cost is not None:
                self.cost += d.cost
            return d.action
        raise ValueError(self.kind)
