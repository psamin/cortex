"""Next-operation predictors P(op_{t+1} | context, history).

All are count-based with additive (Laplace) smoothing, fit only on training
trajectories, and frozen before evaluation unless explicitly run online.

  FrequencyPredictor     global marginal P(op)                      (baseline)
  TransitionPredictor    P(op | prev_op)  (level-1 workflow-graph edges)
  ContextPredictor       P(op | context key)  backed off to P(op | prev_op), then P(op)
  PersonalizedPredictor  shrinkage: (1 - a) P_global(op | ctx) + a P_local(op | ctx, env),
                         a = n_local / (n_local + tau)  (more local evidence -> more local weight)
  HeuristicPredictor     the observation-only BFS heuristic as a point predictor (no learning)
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np


class _Counts:
    def __init__(self, ops, smoothing):
        self.ops, self.s = list(ops), smoothing
        self.c = defaultdict(Counter)

    def add(self, key, op):
        self.c[key][op] += 1

    def n(self, key):
        return sum(self.c[key].values()) if key in self.c else 0

    def dist(self, key, prior=None):
        cnt = self.c.get(key, Counter())
        tot = sum(cnt.values())
        if prior is None:
            prior = {o: 1 / len(self.ops) for o in self.ops}
        k = self.s * len(self.ops)
        return {o: (cnt[o] + k * prior[o]) / (tot + k) for o in self.ops}


class FrequencyPredictor:
    name = "frequency"

    def __init__(self, ops, smoothing=1.0):
        self.m = _Counts(ops, smoothing)

    def fit(self, rows):
        for r in rows:
            self.m.add("*", r["op"])
        return self

    def predict(self, r):
        return self.m.dist("*")


class TransitionPredictor:
    name = "transition(prev_op)"

    def __init__(self, ops, smoothing=1.0):
        self.ops = ops
        self.g = FrequencyPredictor(ops, smoothing)
        self.m = _Counts(ops, smoothing)

    def fit(self, rows):
        self.g.fit(rows)
        for r in rows:
            self.m.add(r["prev_op"], r["op"])
        return self

    def predict(self, r):
        return self.m.dist(r["prev_op"], self.g.predict(r))


class ContextPredictor:
    name = "context"

    def __init__(self, ops, smoothing=0.5):
        self.ops = ops
        self.t = TransitionPredictor(ops, smoothing)
        self.m = _Counts(ops, smoothing)

    def fit(self, rows):
        self.t.fit(rows)
        for r in rows:
            self.m.add(r["context"], r["op"])
        return self

    def predict(self, r):
        return self.m.dist(r["context"], self.t.predict(r))


class PersonalizedPredictor:
    name = "personalized(shrinkage)"

    def __init__(self, ops, tau=20.0, smoothing=0.5):
        self.ops, self.tau = ops, tau
        self.glob = ContextPredictor(ops, smoothing)
        self.local = _Counts(ops, smoothing)

    def fit(self, rows):
        self.glob.fit(rows)
        for r in rows:
            self.local.add((r["env"], r["context"]), r["op"])
        return self

    def predict(self, r):
        pg = self.glob.predict(r)
        key = (r["env"], r["context"])
        n = self.local.n(key)
        a = n / (n + self.tau)
        pl = self.local.dist(key, pg)
        return {o: (1 - a) * pg[o] + a * pl[o] for o in self.ops}


class HeuristicPredictor:
    name = "obs_heuristic(point)"

    def __init__(self, ops, eps=0.05):
        self.ops, self.eps = ops, eps

    def fit(self, rows):
        return self

    def predict(self, r):
        h = r.get("heuristic_op")
        if h not in self.ops:
            return {o: 1 / len(self.ops) for o in self.ops}
        return {o: (1 - self.eps) if o == h else self.eps / (len(self.ops) - 1) for o in self.ops}


def evaluate(pred, rows, n_bins=10):
    """Top-1/top-3 accuracy, NLL, multiclass Brier, ECE (top-1 confidence)."""
    if not rows:
        return {}
    top1 = top3 = nll = brier = 0.0
    confs, hits = [], []
    for r in rows:
        p = pred.predict(r)
        ranked = sorted(p, key=lambda o: -p[o])
        y = r["op"]
        top1 += ranked[0] == y
        top3 += y in ranked[:3]
        nll += -np.log(max(p.get(y, 0.0), 1e-12))
        brier += sum((p[o] - (o == y)) ** 2 for o in p)
        confs.append(p[ranked[0]])
        hits.append(ranked[0] == y)
    confs, hits = np.array(confs), np.array(hits, float)
    bins = np.minimum((confs * n_bins).astype(int), n_bins - 1)
    ece = sum(abs(hits[bins == b].mean() - confs[bins == b].mean()) * (bins == b).mean()
              for b in range(n_bins) if (bins == b).any())
    n = len(rows)
    return {"n": n, "top1": top1 / n, "top3": top3 / n, "nll": nll / n, "brier": brier / n, "ece": float(ece)}
