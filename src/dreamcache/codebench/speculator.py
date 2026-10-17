"""Predictive execution for coding workflows: pre-run predicted read-only tools on the
CURRENT immutable snapshot while the agent's LLM is still deciding.

Never pre-runs anything against a snapshot that does not exist yet (e.g. after an edit
the agent has not produced); results are keyed by (op, snapshot hash, fingerprint) and
validated on reuse. Policies:
  none        no speculation (baseline)
  memo        plain memoization of identical (op, snapshot) requests, no prediction
  all         ordinary parallelism: run every read-only op for every new snapshot
              (no prediction; isolates the value of *prediction* vs. parallelism)
  random      one random read-only op per new snapshot
  top1/topk   learned predictor's most likely read-only op(s)
  utility     include op if p(op) * T_op - lam * cpu_op > 0
"""
from __future__ import annotations

import shutil
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from dreamcache.codebench.tools import READ_ONLY, execute, isolated_copy

OPS = ["read_file", "edit", "run_tests", "lint", "deps_check", "finish"]


class OpPredictor:
    """P(next op | prev op, last outcome, family) with shrinkage toward the global model.

    context = (prev_op, last_rc_bucket); personalization key = task family (template).
    a = n_local / (n_local + tau).  mode: global | personalized | frequency
    """

    def __init__(self, mode="personalized", tau=10.0, s=0.5):
        self.mode, self.tau, self.s = mode, tau, s
        self.g = defaultdict(Counter)
        self.loc = defaultdict(Counter)
        self.marg = Counter()

    def fit(self, rows):
        for r in rows:
            self.observe(r)
        return self

    def observe(self, r):
        ctx = (r["prev_op"], r["last_ok"])
        self.g[ctx][r["op"]] += 1
        self.loc[(r["family"], ctx)][r["op"]] += 1
        self.marg[r["op"]] += 1

    def _dist(self, cnt, prior):
        tot = sum(cnt.values())
        k = self.s * len(OPS)
        return {o: (cnt[o] + k * prior[o]) / (tot + k) for o in OPS}

    def predict(self, prev_op, last_ok, family):
        uni = {o: 1 / len(OPS) for o in OPS}
        pm = self._dist(self.marg, uni)
        if self.mode == "frequency":
            return pm
        pg = self._dist(self.g[(prev_op, last_ok)], pm)
        if self.mode == "global":
            return pg
        cl = self.loc[(family, (prev_op, last_ok))]
        n = sum(cl.values())
        a = n / (n + self.tau)
        pl = self._dist(cl, pg)
        return {o: (1 - a) * pg[o] + a * pl[o] for o in OPS}


class Speculator:
    def __init__(self, policy, predictor=None, fp="", k=2, lam=0.3, max_workers=3, seed=0):
        self.policy, self.pred, self.fp, self.k, self.lam = policy, predictor, fp, k, lam
        self.pool = ThreadPoolExecutor(max_workers)
        self.results: dict[tuple, dict] = {}   # (op, snap, fp) -> record
        self.rng = np.random.default_rng(seed)
        self.T = {o: 0.5 for o in READ_ONLY}   # EMA of op wall time
        self.C = {o: 0.5 for o in READ_ONLY}   # EMA of op CPU time
        self.log = []

    def on_new_snapshot(self, root, snap, prev_op, last_ok, family):
        """Called when the agent starts thinking about the next op for snapshot `snap`."""
        if self.policy in ("none", "memo"):
            return
        if self.policy == "all":
            ops = list(READ_ONLY)
        elif self.policy == "random":
            ops = [READ_ONLY[int(self.rng.integers(len(READ_ONLY)))]]
        else:
            p = self.pred.predict(prev_op, last_ok, family)
            ranked = [o for o in sorted(p, key=lambda o: -p[o]) if o in READ_ONLY]
            if self.policy == "top1":
                ops = ranked[:1]
            elif self.policy == "topk":
                ops = ranked[: self.k]
            elif self.policy == "utility":
                ops = [o for o in ranked if p[o] * self.T[o] - self.lam * self.C[o] > 0][: self.k]
            else:
                raise ValueError(self.policy)
        for op in ops:
            key = (op, snap, self.fp)
            if key in self.results:
                continue
            t_launch = time.perf_counter()
            copy = isolated_copy(root)   # isolation: speculative work never touches the real workspace
            rec = {"op": op, "snapshot": snap, "fp": self.fp, "launch": t_launch, "fate": "pending",
                   "copy_overhead_s": time.perf_counter() - t_launch}

            def job(copy=copy, op=op, rec=rec):
                rec["start"] = time.perf_counter()
                try:
                    out = execute(op, copy)
                finally:
                    shutil.rmtree(copy.rsplit("/ws", 1)[0], ignore_errors=True)
                rec["finish"] = time.perf_counter()
                return out

            rec["future"] = self.pool.submit(job)
            self.results[key] = rec

    def request(self, op, root, snap):
        """The agent asks for `op` on the REAL current snapshot. Returns (result, info)."""
        key = (op, snap, self.fp)
        rec = self.results.get(key)
        t0 = time.perf_counter()
        if rec is not None:
            done = rec["future"].done()
            out = rec["future"].result()
            waited = time.perf_counter() - t0
            if rec["snapshot"] == snap and rec["fp"] == self.fp:   # validation
                fate = "reused" if done else "reused_late"
                if rec["fate"] == "pending":
                    rec["fate"] = fate
                    self.log.append(rec)
                return out, {"critical_s": waited, "reused": True, "fate": fate}
        out = execute(op, root)
        crit = time.perf_counter() - t0
        self.T[op] = 0.8 * self.T[op] + 0.2 * out["seconds"]
        self.C[op] = 0.8 * self.C[op] + 0.2 * max(out["cpu"], 1e-3)
        if self.policy != "none":   # memoize ordinary results of pure ops
            from concurrent.futures import Future
            f = Future()
            f.set_result(out)
            self.results[key] = {"op": op, "snapshot": snap, "fp": self.fp, "fate": "memo", "future": f}
        return out, {"critical_s": crit, "reused": False, "fate": "executed"}

    def finalize(self):
        st = Counter()
        spec_cpu = wasted_cpu = spec_wall = copy_s = 0.0
        for rec in self.results.values():
            if rec["fate"] == "memo":
                continue
            out = rec["future"].result()
            fate = rec["fate"] if rec["fate"] != "pending" else "unused"
            st[fate] += 1
            spec_cpu += out["cpu"]
            spec_wall += out["seconds"]
            copy_s += rec.get("copy_overhead_s", 0.0)
            if fate == "unused":
                wasted_cpu += out["cpu"]
        self.pool.shutdown(wait=True)
        return {"launched": sum(st.values()), **dict(st), "spec_cpu_s": spec_cpu, "wasted_cpu_s": wasted_cpu,
                "spec_wall_s": spec_wall, "copy_overhead_s": copy_s}
