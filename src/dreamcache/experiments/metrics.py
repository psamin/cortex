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
    if ep.get("extra") and "prefetch" in ep["extra"]:  # predictive execution (specgraph)
        ex = ep["extra"]["prefetch"]
        m["llm_seconds"] += ex["wasted_llm_seconds"]   # reused calls are already in step costs
        m["wasted_llm_seconds"] = ex["wasted_llm_seconds"]
        n_l = max(1, ex["launched"])
        m["spec_launched"] = ex["launched"]
        m["spec_reuse_rate"] = (ex.get("reused", 0) + ex.get("reused_late", 0)) / n_l
        m["spec_late_rate"] = ex.get("reused_late", 0) / n_l
        m["spec_unused_rate"] = ex.get("unused", 0) / n_l
        m["prefetch_hit_rate"] = (ex.get("reused", 0) + ex.get("reused_late", 0)) / max(1, len(recs))
        m["predict_overhead_s"] = ex["predict_overhead_s"]
    elif ep.get("extra"):  # lossless speculation: actor calls incl. wasted prefetches
        ex = ep["extra"]
        for k in COST_KEYS:
            m[k] += float(ex["actor_cost"].get(k, 0))
        m["prefetch_hit_rate"] = ex["hits"] / max(1, ex["hits"] + ex["misses"])
        m["actor_calls_per_step"] = ex["actor_calls"] / max(1, len(recs))
    # Total compute proxy: model-server seconds + simulator CPU seconds (reported separately too).
    m["compute_seconds"] = m["llm_seconds"] + m["sim_cpu_seconds"]
    acc = [r["verify_accept"] for r in recs if "verify_accept" in r]
    m["verify_accept_rate"] = float(np.mean(acc)) if acc else float("nan")
    drafts = [r for r in recs if "draft" in r]
    m["draft_accepted_mean"] = float(np.mean([r["accepted"] for r in drafts])) if drafts else float("nan")
    m["draft_calls_per_step"] = len(drafts) / len(recs) if drafts and recs else float("nan")
    m["_latencies"] = lat
    return m
