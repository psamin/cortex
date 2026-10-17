"""Per-episode metric extraction from runner records."""
from __future__ import annotations

import json

import numpy as np

COST_KEYS = ["llm_calls", "prompt_tokens", "gen_tokens", "llm_seconds", "cache_hits", "malformed",
             "sim_steps", "sim_cpu_seconds", "branches"]


def load(path):
    eps, errs = [], []
    for line in open(path):
        r = json.loads(line)
        (errs if "error" in r else eps).append(r)
    return eps, errs


def episode_metrics(ep: dict) -> dict:
    recs = ep["records"]
    costs = [r.get("cost", {}) for r in recs]
    lat = [c.get("wall_seconds", 0.0) for c in costs]
    m = {"env": ep["env"], "seed": ep["seed"], "method": ep["method"],
         "success": float(ep["success"]), "failure": float(ep["failure"]),
         "return": ep["return"], "steps": ep["steps"], "episode_seconds": ep["episode_seconds"],
         "trigger_rate": float(np.mean([r.get("triggered", False) for r in recs])) if recs else 0.0,
         "n_triggers": int(sum(r.get("triggered", False) for r in recs)),
         "changed_rate": float(np.mean([r.get("changed", False) for r in recs])) if recs else 0.0,
         "decision_latency_mean": float(np.mean(lat)) if lat else 0.0}
    for k in COST_KEYS:
        m[k] = float(sum(c.get(k, 0) for c in costs))
    # Total compute proxy: model-server seconds + simulator CPU seconds (reported separately too).
    m["compute_seconds"] = m["llm_seconds"] + m["sim_cpu_seconds"]
    m["_latencies"] = lat
    return m
