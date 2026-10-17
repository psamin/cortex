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
  J verify        small model proposes, BIG model verifies/overrides (always or on trigger)
  O cascade       on trigger, the BIG model decides from scratch (FrugalGPT-style routing)
  M vote          on trigger, self-consistency majority vote over n small-model samples
  N mcts          on trigger, UCT search over cloned simulators (SIMULATOR-ASSISTED)
  K spec_draft    speculative action drafting: small drafts L actions in one call, BIG
                  verifies the plan in one call; the accepted prefix runs without more
                  BIG calls (speculative decoding analogue). Mode, not a trigger. NOT
                  lossless: trusts the big model's self-reported accepted-prefix count.
  K* spec_draft_lossless  faithful greedy speculative decoding over actions, after Dynamic
                  Speculative Planning (arXiv 2509.01920) / ISP (arXiv 2410.00079): the small
                  model drafts L actions; the drafted states are produced in simulator clones
                  sharing the real RNG; the BIG model decides on every drafted state IN
                  PARALLEL; accept the longest prefix where draft == big argmax, then append
                  the big model's own action at the first mismatch (the "bonus token").
                  Executed actions equal the big model's greedy choices (lossless w.r.t.
                  L_greedy). SIMULATOR-ASSISTED (clones produce drafted states).
  S swiftsage     trigger rule ported from SwiftSage (arXiv 2305.17390, eval_agent_fast_slow.py):
                  stay fast if step < 10, or < 5 steps since last slow call, or the fast
                  action is not among the last 3 actions; otherwise call the slow (BIG) model.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from collections import Counter

from dreamcache.agents.oracle import q_ref, regret
from dreamcache.environments.observations import features
from dreamcache.detection.difficulty import DetectorConfig, difficulty, signals
from dreamcache.detection.risk import observable_risk
from dreamcache.policies.candidate_generator import generate
from dreamcache.speculation.evaluator import EvalConfig
from dreamcache.speculation.executor import SpecConfig, speculate
from dreamcache.speculation.mcts_search import mcts_decide
from dreamcache.types import Cost


@dataclass
class AgentConfig:
    trigger: str = "never"         # never|always|threshold|random|periodic|oracle
    threshold: float = 0.5
    rate: float = 0.2              # random trigger
    period: int = 5                # periodic trigger
    oracle_delta: float = 0.1
    oracle_samples: int = 8
    resolve: str = "rollout"       # rollout|norollout|verify|cascade|vote|mcts
    mode: str = "step"             # step | spec_draft
    plan_len: int = 4              # spec_draft draft length
    vote_n: int = 5
    mcts_iterations: int = 60
    mcts_depth: int = 8
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    spec: SpecConfig = field(default_factory=SpecConfig)
    evaluator: EvalConfig = field(default_factory=EvalConfig)
    compute_signals: bool = True   # log detector signals even when not used to trigger


class Agent:
    def __init__(self, fast, legal, family, cfg: AgentConfig, seed: int, big=None, graph=None):
        self.fast, self.legal, self.family, self.cfg = fast, legal, family, cfg
        self.big = big
        self.graph = graph   # B7: callable(obs, history) -> P(next executed action), learned from history
        self.queue: list[int] = []
        self.rng = np.random.default_rng(seed + 99991)
        self.seed = seed

    def _need_proposals(self):
        c = self.cfg
        return c.trigger == "threshold" and c.detector.needs_proposals or (
            c.spec.candidates == "llm_sample" and c.trigger != "never")

    def act(self, env, obs, history, t):
        """Returns (action, step record). `env` is used ONLY for sandboxes / oracle."""
        c = self.cfg
        if c.mode == "spec_draft":
            return self._spec_draft(obs, history, t)
        if c.mode == "spec_draft_lossless":
            return self._spec_draft_lossless(env, obs, history, t)
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
        elif c.trigger == "swiftsage":
            since = t - getattr(self, "_last_slow", -100)
            trig = not (t < 10 or since < 5 or dec.action not in history[-3:])
            if trig:
                self._last_slow = t
        elif c.trigger == "oracle":
            q = q_ref(env, self.legal, history, n=c.oracle_samples, seed=self.seed * 1000 + t)
            rec["oracle_q"] = {str(k): v for k, v in q.items()}
            rec["oracle_regret"] = regret(q, dec.action)
            trig = rec["oracle_regret"] > c.oracle_delta
        else:
            raise ValueError(c.trigger)

        action = dec.action
        if trig:
            if c.spec.candidates == "graph" and self.graph is not None:
                # B7: rank rollout candidates by the LLM scores blended 50/50 with the
                # execution-graph predictor learned from this agent's past trajectories
                g = self.graph(obs, history)
                blended = {a: 0.5 * dec.probs.get(a, 0.0) + 0.5 * g.get(a, 0.0) for a in self.legal}
                rec["graph_probs"] = {str(a): v for a, v in g.items()}
                from dreamcache.types import Decision as _D
                cands = generate("llm_rank", c.spec.k, _D(dec.action, blended), self.legal, proposals, obs)
            else:
                cands = generate(c.spec.candidates, c.spec.k, dec, self.legal, proposals, obs)
            rec["candidates"] = cands
            if c.resolve == "rollout":
                out = speculate(env, cands, history, self.legal, c.spec, c.evaluator,
                                seed=self.seed * 100_003 + t, fast_policy=self.fast)
                action = out["selected_action"]
                rec["candidate_results"] = out["candidate_results"]
                cost += out["cost"]
            elif c.resolve == "norollout":  # best-of-K: fast-policy score minus observable risk
                action = max(cands, key=lambda a: dec.probs.get(a, 0)
                             - observable_risk(obs, a, self.family))
            elif c.resolve == "verify":
                acc, action, vc = self.big.verify(obs, history, dec.action)
                rec["verify_accept"] = acc
                cost += vc
            elif c.resolve == "cascade":
                bd = self.big.decide(obs, history)
                action = bd.action
                cost += bd.cost
            elif c.resolve == "vote":
                votes, vc = self.fast.propose(obs, history, c.vote_n, self.seed * 31 + t)
                cost += vc
                cnt = Counter([dec.action] + votes)
                top = max(cnt.values())
                action = dec.action if cnt[dec.action] == top else next(a for a, n in cnt.most_common() if n == top)
                rec["votes"] = votes
            elif c.resolve == "mcts":
                action, stats, mc = mcts_decide(env, self.legal, c.mcts_iterations, c.mcts_depth, c.evaluator,
                                                seed=self.seed * 100_003 + t, rollout_kind=c.spec.rollout_policy)
                rec["mcts"] = {str(k): v for k, v in stats.items()}
                cost += mc
            else:
                raise ValueError(c.resolve)
        cost.wall_seconds = time.perf_counter() - t0
        rec.update(triggered=bool(trig), action=int(action), changed=bool(action != dec.action),
                   cost=cost.as_dict())
        return action, rec

    def _spec_draft(self, obs, history, t):
        """Speculative action drafting (see module docstring)."""
        c = self.cfg
        t0 = time.perf_counter()
        cost = Cost()
        rec = {"t": t, "mode": "spec_draft"}
        if self.queue and self.queue[0] == 2 and features(obs).front_blocked:
            rec["queue_flushed"] = len(self.queue)   # observable hazard: drop stale draft
            self.queue = []
        if not self.queue:
            plan, dc = self.fast.draft_plan(obs, history, c.plan_len)
            n, corr, vc = self.big.verify_plan(obs, history, plan)
            cost += dc
            cost += vc
            rec.update(draft=plan, accepted=n, correction=corr)
            self.queue = plan[:n] if n > 0 else [corr]
            rec["triggered"] = True
        else:
            rec["triggered"] = False
        action = self.queue.pop(0)
        cost.wall_seconds = time.perf_counter() - t0
        rec.update(action=int(action), changed=False, cost=cost.as_dict())
        return action, rec

    def _spec_draft_lossless(self, env, obs, history, t):
        from concurrent.futures import ThreadPoolExecutor

        from dreamcache.environments.cloning import clone_env
        c = self.cfg
        t0 = time.perf_counter()
        cost = Cost()
        rec = {"t": t, "mode": "spec_draft_lossless"}
        if self.queue:
            action = self.queue.pop(0)
            rec.update(triggered=False, action=int(action), changed=False)
            cost.wall_seconds = time.perf_counter() - t0
            rec["cost"] = cost.as_dict()
            return action, rec
        plan, dc = self.fast.draft_plan(obs, history, c.plan_len)
        cost += dc
        # drafted states: follow the draft in a clone that shares the real RNG
        states, sim, h, o = [(obs, list(history))], clone_env(env), list(history), obs
        for a in plan[:-1]:
            o, _, term, trunc, _ = sim.step(a)
            h = h + [a]
            if term or trunc:
                break
            states.append((o, h))
        if not hasattr(self, "_pool"):
            self._pool = ThreadPoolExecutor(max(2, c.plan_len))
        spec = getattr(self, "big_spec", None) or self.big   # drafted future states: speculative capacity
        decs = list(self._pool.map(lambda ist: (self.big if ist[0] == 0 else spec).decide(ist[1][0], ist[1][1]),
                                   enumerate(states)))
        accepted, n_acc = [], 0
        for d in decs:
            cost += d.cost  # all parallel big calls are paid for, used or not
        for a, d in zip(plan, decs):
            if d.action == a:
                accepted.append(int(a))
                n_acc += 1
            else:
                accepted.append(int(d.action))  # bonus: the big model's own action
                break
        rec.update(draft=plan, accepted=n_acc, big_calls=len(decs), triggered=True)
        self.queue = accepted[1:]
        action = accepted[0]
        cost.wall_seconds = time.perf_counter() - t0
        rec.update(action=int(action), changed=False, cost=cost.as_dict())
        return action, rec
