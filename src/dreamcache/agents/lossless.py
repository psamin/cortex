"""Lossless speculative actions (after "Speculative Actions: A Lossless Framework for
Faster Agentic Systems", 2025). Targets LATENCY, not compute.

The actor is the BIG model; every executed action is the big model's decision on the
true (observation, history) prompt, so behaviour equals L_greedy (big model, no search)
up to server nondeterminism. A fast guesser predicts the actor's action; each guess is
executed in a simulator clone that shares the real RNG state (SIMULATOR-ASSISTED: it
reproduces the exact next state if the real action equals the guess), and the actor's
next-step call is launched immediately on the predicted next observation. If the real
next prompt matches a prefetched one, that call is already in flight or finished.

Guessers: "obs_heuristic" (deployable observation-only BFS, ~0 ms), "small_llm" (the
small model's greedy action), or both (top-2 speculation). Wasted prefetches are
counted as compute. The LLM response cache must be DISABLED for latency runs.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from dreamcache.environments.cloning import clone_env
from dreamcache.environments.observations import describe
from dreamcache.policies.heuristic import ObsBFSPolicy
from dreamcache.types import Cost


class LosslessSpecAgent:
    def __init__(self, big, small, legal, guessers=("obs_heuristic",), seed=0, max_workers=6, spec_big=None):
        self.big, self.small, self.legal = big, small, legal
        self.spec_big = spec_big or big   # speculative calls may go to a separate server/GPU
        self.guessers = list(guessers)
        self.heur = ObsBFSPolicy(seed=seed)
        self.pool = ThreadPoolExecutor(max_workers)
        self.pending: dict[str, object] = {}
        self.all_futs = []
        self.hits = self.misses = 0

    def _key(self, obs, history):
        return describe(obs, history, self.legal)

    def _submit(self, obs, history, speculative=False):
        actor = self.spec_big if speculative else self.big
        f = self.pool.submit(actor.decide, obs, list(history))
        self.all_futs.append(f)
        return f

    def act(self, env, obs, history, t):
        t0 = time.perf_counter()
        key = self._key(obs, history)
        fut = self.pending.pop(key, None)
        hit = fut is not None
        if hit:
            self.hits += 1
        else:
            self.misses += 1
            fut = self._submit(obs, history)
        self.pending = {}  # every other prefetch is now stale (wasted compute, still counted)
        guesses, gcost = [], Cost()
        for g in self.guessers:
            if g == "obs_heuristic":
                guesses.append(self.heur.act(obs))
            elif g == "small_llm":
                d = self.small.decide(obs, history)
                gcost += d.cost
                guesses.append(d.action)
        for g in dict.fromkeys(guesses):
            sim = clone_env(env)  # shares the real RNG: exact next state if the actor picks g
            nobs, _, term, trunc, _ = sim.step(g)
            if not (term or trunc):
                h2 = list(history) + [int(g)]
                self.pending[self._key(nobs, h2)] = self._submit(nobs, h2, speculative=True)
        dec = fut.result()
        action = dec.action
        cost = Cost()
        cost += gcost
        cost.wall_seconds = time.perf_counter() - t0
        rec = {"t": t, "action": int(action), "fast_action": int(action), "triggered": False,
               "spec_hit": hit, "guesses": [int(g) for g in guesses],
               "guess_correct": int(action) in [int(g) for g in guesses], "cost": cost.as_dict()}
        return action, rec

    def finalize(self) -> dict:
        """Wait for all actor calls (incl. wasted prefetches) and total their compute."""
        total = Cost()
        for f in self.all_futs:
            try:
                total += f.result().cost
            except Exception:  # noqa: BLE001
                pass
        self.pool.shutdown(wait=True)
        return {"actor_calls": len(self.all_futs), "actor_cost": total.as_dict(),
                "hits": self.hits, "misses": self.misses}
