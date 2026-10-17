"""CPU stand-in for the LLM fast policy (tests and pipeline debugging only).

Observation-only: wraps ObsBFSPolicy and emits noisy, sometimes confidently wrong
scores. Results produced with it are labeled `surrogate` and are not LLM results.
"""
from __future__ import annotations

import time

import numpy as np

from dreamcache.policies.heuristic import ObsBFSPolicy
from dreamcache.types import Cost, Decision


class SurrogatePolicy:
    privileged = False
    name = "surrogate"

    def __init__(self, legal, seed=0, eps=0.25):
        self.legal, self.eps = legal, eps
        self.base = ObsBFSPolicy(seed=seed)
        self.rng = np.random.default_rng(seed + 7)

    def _scores(self, obs):
        a = self.base.act(obs)
        if self.rng.random() < self.eps:
            a = int(self.rng.choice(self.legal))
        w = self.rng.dirichlet(np.ones(len(self.legal)))
        conf = self.rng.uniform(0.3, 0.9)
        s = {b: (1 - conf) * w[i] for i, b in enumerate(self.legal)}
        s[a] += conf
        return s, a

    def decide(self, obs, history) -> Decision:
        t0 = time.perf_counter()
        s, a = self._scores(obs)
        return Decision(a, s, Cost(llm_calls=1, wall_seconds=time.perf_counter() - t0))

    def propose(self, obs, history, n, seed):
        acts = []
        for _ in range(n):
            s, _ = self._scores(obs)
            p = np.array([s[b] for b in self.legal])
            acts.append(int(self.rng.choice(self.legal, p=p / p.sum())))
        return acts, Cost(llm_calls=n)

    # CPU stand-ins for the draft/verify interface (tests only)
    def verify(self, obs, history, proposed):
        a = self.base.act(obs)
        return a == proposed, a, Cost(llm_calls=1)

    def draft_plan(self, obs, history, length):
        return [self._scores(obs)[1]] * length, Cost(llm_calls=1)

    def verify_plan(self, obs, history, plan):
        a = self.base.act(obs)
        return (1 if plan and plan[0] == a else 0), a, Cost(llm_calls=1)
