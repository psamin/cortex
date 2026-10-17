"""Speculation schedulers: given P(next op), choose which future actor calls to pre-execute.

U(o) = p(o) * T_saved - lambda * C_spec - kappa * n_inflight
  T_saved  : expected critical-path time a hit removes (~ one actor call latency, EMA-measured)
  C_spec   : resource cost of one speculative actor call (same EMA latency, as GPU-seconds)
  kappa    : contention penalty per already-inflight call (throughput is sub-linear, see
             configs/gpus.md); this is a scheduling heuristic, outcomes are measured separately.

Policies: none | random | top1 | topk | p_latency | utility | utility_adaptive
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SchedConfig:
    policy: str = "top1"
    k: int = 2
    max_inflight: int = 3       # max concurrent speculative workers
    budget_ratio: float = 1.0   # max speculative calls per real step (episode-level)
    p_min: float = 0.35         # p_latency: prefetch if p * T_saved >= p_min * T
    lam: float = 0.3
    kappa: float = 0.15


class Scheduler:
    def __init__(self, cfg: SchedConfig, ops, seed=0):
        self.cfg, self.ops = cfg, list(ops)
        self.rng = np.random.default_rng(seed)
        self.T = 1.0          # EMA of actor-call latency (s)
        self.spent = 0
        self.steps = 0
        self.hit_ema = 0.5    # for utility_adaptive

    def observe_latency(self, seconds):
        self.T = 0.8 * self.T + 0.2 * seconds

    def observe_hit(self, hit):
        self.hit_ema = 0.9 * self.hit_ema + 0.1 * float(hit)

    def select(self, p: dict) -> list:
        c = self.cfg
        self.steps += 1
        budget_left = c.budget_ratio * self.steps - self.spent
        cap = int(max(0, min(c.max_inflight, np.floor(budget_left))))
        if c.policy == "none" or cap == 0:
            return []
        ranked = sorted(p, key=lambda o: -p[o])
        if c.policy == "random":
            out = [self.ops[int(self.rng.integers(len(self.ops)))]]
        elif c.policy == "top1":
            out = ranked[:1]
        elif c.policy == "topk":
            out = ranked[:c.k]
        elif c.policy == "p_latency":
            out = [o for o in ranked if p[o] * self.T >= c.p_min * self.T][: c.k]
        elif c.policy in ("utility", "utility_adaptive"):
            # adaptive: scale probabilities by observed calibration (hit rate vs predicted)
            scale = 1.0
            if c.policy == "utility_adaptive":
                scale = float(np.clip(self.hit_ema / max(1e-3, p[ranked[0]]), 0.5, 1.5))
            out = []
            for o in ranked:
                u = scale * p[o] * self.T - c.lam * self.T - c.kappa * self.T * len(out)
                if u <= 0:
                    break
                out.append(o)
        else:
            raise ValueError(c.policy)
        out = out[:cap]
        self.spent += len(out)
        return out
