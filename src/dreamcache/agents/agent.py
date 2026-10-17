"""One agent class; the experimental conditions differ only in the trigger and in
how a triggered decision is resolved. Every condition uses the same fast policy.

  A greedy        trigger=never
  B always        trigger=always,    resolve=rollout
  C selective     trigger=threshold, resolve=rollout   (detector: entropy/disagreement/risk)
  D oracle        trigger=oracle,    resolve=rollout   (ORACLE: Q_ref regret > delta; not deployable)
  E random        trigger=random(rate)
  F periodic      trigger=periodic(N)
  G bestofk       trigger=always|threshold, resolve=norollout (observable 1-step evaluator)
  H think         greedy with a thinking-enabled fast policy (set in runner)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from dreamcache.agents.oracle import q_ref, regret
from dreamcache.detection.difficulty import DetectorConfig, difficulty, signals
from dreamcache.detection.risk import observable_risk
from dreamcache.policies.candidate_generator import generate
from dreamcache.speculation.evaluator import EvalConfig
from dreamcache.speculation.executor import SpecConfig, speculate
from dreamcache.types import Cost


@dataclass
class AgentConfig:
    trigger: str = "never"         # never|always|threshold|random|periodic|oracle
    threshold: float = 0.5
    rate: float = 0.2              # random trigger
    period: int = 5                # periodic trigger
    oracle_delta: float = 0.1
    oracle_samples: int = 8
    resolve: str = "rollout"       # rollout|norollout
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    spec: SpecConfig = field(default_factory=SpecConfig)
    evaluator: EvalConfig = field(default_factory=EvalConfig)
    compute_signals: bool = True   # log detector signals even when not used to trigger


class Agent:
    def __init__(self, fast, legal, family, cfg: AgentConfig, seed: int):
        self.fast, self.legal, self.family, self.cfg = fast, legal, family, cfg
        self.rng = np.random.default_rng(seed + 99991)
        self.seed = seed

    def _need_proposals(self):
        c = self.cfg
        return c.trigger == "threshold" and c.detector.needs_proposals or (
            c.spec.candidates == "llm_sample" and c.trigger != "never")

    def act(self, env, obs, history, t):
        """Returns (action, step record). `env` is used ONLY for sandboxes / oracle."""
        c = self.cfg
        t0 = time.perf_counter()
        dec = self.fast.decide(obs, history)
        cost = Cost()
        cost += dec.cost
        proposals = None
        if self._need_proposals():
            proposals, pc = self.fast.propose(obs, history, c.detector.n_proposals, self.seed * 7919 + t)
            cost += pc
        sig = signals(dec, obs, self.legal, self.family, proposals)
        score = difficulty(sig, c.detector)
        rec = {"t": t, "fast_action": dec.action, "probs": {str(k): v for k, v in dec.probs.items()},
               "signals": sig, "difficulty": score, "proposals": proposals}

        if c.trigger == "never":
            trig = False
        elif c.trigger == "always":
            trig = True
        elif c.trigger == "threshold":
            trig = score >= c.threshold
        elif c.trigger == "random":
            trig = bool(self.rng.random() < c.rate)
        elif c.trigger == "periodic":
            trig = t % c.period == 0
        elif c.trigger == "oracle":
            q = q_ref(env, self.legal, history, n=c.oracle_samples, seed=self.seed * 1000 + t)
            rec["oracle_q"] = {str(k): v for k, v in q.items()}
            rec["oracle_regret"] = regret(q, dec.action)
            trig = rec["oracle_regret"] > c.oracle_delta
        else:
            raise ValueError(c.trigger)

        action = dec.action
        if trig:
            cands = generate(c.spec.candidates, c.spec.k, dec, self.legal, proposals, obs)
            rec["candidates"] = cands
            if c.resolve == "rollout":
                out = speculate(env, cands, history, self.legal, c.spec, c.evaluator,
                                seed=self.seed * 100_003 + t, fast_policy=self.fast)
                action = out["selected_action"]
                rec["candidate_results"] = out["candidate_results"]
                cost += out["cost"]
            else:  # best-of-K without rollouts: fast-policy score minus observable risk
                action = max(cands, key=lambda a: dec.probs.get(a, 0)
                             - observable_risk(obs, a, self.family))
        cost.wall_seconds = time.perf_counter() - t0
        rec.update(triggered=bool(trig), action=int(action), changed=bool(action != dec.action),
                   cost=cost.as_dict())
        return action, rec
