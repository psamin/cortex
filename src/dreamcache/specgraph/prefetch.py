"""Predictive execution layer: pre-execute the actor's NEXT-step model call.

PrefetchingPolicy wraps any fast policy (`decide(obs, history) -> Decision`). Before the
agent decides at step t, `speculate(env, obs, history)` asks a predictor for the
distribution of the action the agent will execute, lets a scheduler pick guesses, rolls
each guess one step in a simulator clone that shares the real RNG (so a correct guess
reproduces the exact next observation), and submits `decide(next_obs, history+[g])` to a
worker pool. When the agent later calls `decide` on the real next state, the prefetched
result is reused ONLY after validation:

  1. required    : the agent actually requests decide() for this state
  2. inputs match: sha256 of the exact prompt text equals the speculative input hash
  3. environment : same model id / prompt version fingerprint
  4. equivalence : decide() is a pure function of the prompt at temperature 0 (up to
                   server nondeterminism, measured and reported separately)
Anything that fails validation is discarded and executed normally.

Every speculative record keeps: parent step, assumed action, input hash, fingerprint,
start/finish timestamps, cost, and its fate (reused / invalidated / unused / late).
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from dreamcache.environments.cloning import clone_env
from dreamcache.environments.observations import describe
from dreamcache.specgraph.graph import context_key, input_hash


class PrefetchingPolicy:
    def __init__(self, inner, legal, predictor, scheduler, env_name, fingerprint, max_workers=4,
                 heuristic=None):
        self.inner, self.legal = inner, legal
        self.predictor, self.scheduler = predictor, scheduler
        self.env_name, self.fp = env_name, fingerprint
        self.heuristic = heuristic
        self.pool = ThreadPoolExecutor(max_workers)
        self.spec: dict[str, dict] = {}   # input_hash -> speculative record
        self.log: list[dict] = []
        self.predict_overhead = 0.0
        self.cur_t = 0
        self.name = getattr(inner, "name", "prefetch")
        self.privileged = False

    # pass-throughs used by agents
    def propose(self, *a, **k):
        return self.inner.propose(*a, **k)

    def __getattr__(self, item):
        return getattr(self.inner, item)

    def _hash(self, obs, history):
        return input_hash(self.fp + "\n" + describe(obs, history, self.legal))

    def speculate(self, env, obs, history, t):
        """Call at the START of step t (before the agent decides): launches work for step t+1."""
        self.cur_t = t
        t0 = time.perf_counter()
        row = {"env": self.env_name, "context": context_key(obs, history),
               "prev_op": history[-1] if history else -1,
               "heuristic_op": self.heuristic.act(obs) if self.heuristic else None}
        p = self.predictor.predict(row)
        guesses = self.scheduler.select(p)
        self.predict_overhead += time.perf_counter() - t0
        for g in guesses:
            sim = clone_env(env)
            nobs, _, term, trunc, _ = sim.step(int(g))
            if term or trunc:
                continue
            h2 = list(history) + [int(g)]
            key = self._hash(nobs, h2)
            if key in self.spec:
                continue  # never execute equivalent work twice
            rec = {"parent_t": t, "assumed_action": int(g), "input_hash": key, "fingerprint": self.fp,
                   "p": float(p[g]), "start": time.perf_counter(), "finish": None, "fate": "pending"}

            def run(nobs=nobs, h2=h2, rec=rec):
                d = self.inner.decide(nobs, h2)
                rec["finish"] = time.perf_counter()
                return d

            rec["future"] = self.pool.submit(run)
            self.spec[key] = rec
        return guesses

    def decide(self, obs, history):
        key = self._hash(obs, history)
        rec = self.spec.pop(key, None)
        if rec is not None and rec["parent_t"] != self.cur_t - 1:
            self.spec[key], rec = rec, None   # launched for a different step: not valid here
        # speculations that targeted THIS step but were not reached are now stale;
        # those launched at this step (for t+1) stay pending
        for k in [k for k, r in self.spec.items() if r["parent_t"] < self.cur_t]:
            r = self.spec.pop(k)
            r["fate"] = "unused"
            self.log.append(r)
        if rec is not None and rec["fingerprint"] == self.fp:
            t0 = time.perf_counter()
            done_before = rec["future"].done()
            d = rec["future"].result()
            waited = time.perf_counter() - t0
            rec["fate"] = "reused" if done_before else "reused_late"
            rec["waited_s"] = waited
            self.log.append(rec)
            self.scheduler.observe_hit(True)
            d.cost.wall_seconds = waited  # critical-path time actually spent on this call
            d.cost.cache_hits += 1         # (not charged twice; compute counted in finalize)
            d.prefetched = True
            return d
        self.scheduler.observe_hit(False)
        d = self.inner.decide(obs, history)
        self.scheduler.observe_latency(d.cost.wall_seconds)
        return d

    def finalize(self):
        """Wait for outstanding work; return speculative accounting (all compute counted)."""
        for rec in self.spec.values():
            rec["fate"] = "unused"
            self.log.append(rec)
        self.spec = {}
        stats = {"launched": len(self.log), "reused": 0, "reused_late": 0, "unused": 0,
                 "spec_llm_seconds": 0.0, "spec_llm_calls": 0, "wasted_llm_seconds": 0.0,
                 "predict_overhead_s": self.predict_overhead}
        for rec in self.log:
            stats[rec["fate"]] = stats.get(rec["fate"], 0) + 1
            try:
                d = rec["future"].result()
            except Exception:  # noqa: BLE001
                continue
            s = d.cost.llm_seconds
            stats["spec_llm_seconds"] += s
            stats["spec_llm_calls"] += 1
            if rec["fate"] == "unused":
                stats["wasted_llm_seconds"] += s
        self.pool.shutdown(wait=True)
        return stats
