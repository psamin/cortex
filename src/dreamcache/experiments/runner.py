"""Experiment runner: one config file -> many (method x seed) episodes -> run dir.

Usage:
  python -m dreamcache.experiments.runner configs/smoke.yaml [--out results] [--workers 8]
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import fields
from importlib import metadata

import yaml

from dreamcache.agents.agent import Agent, AgentConfig
from dreamcache.detection.difficulty import DetectorConfig
from dreamcache.environments.minigrid_adapter import (env_family, is_failure, is_success,
                                                      legal_actions, make_env)
from dreamcache.policies.heuristic import ObsBFSPolicy, RandomPolicy, SimBFSPolicy
from dreamcache.speculation.evaluator import EvalConfig
from dreamcache.speculation.executor import SpecConfig

SPLITS = {"dev": range(0, 20), "val": range(1000, 1050), "test": range(10000, 10100),
          "test300": range(10000, 10300)}


def build(dc, d: dict | None):
    d = dict(d or {})
    kw = {}
    for f in fields(dc):
        if f.name in d:
            v = d[f.name]
            sub = {"detector": DetectorConfig, "spec": SpecConfig, "evaluator": EvalConfig}.get(f.name)
            kw[f.name] = build(sub, v) if sub and isinstance(v, dict) else v
    return dc(**kw)


def merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        out[k] = merge(out.get(k, {}), v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def make_fast(spec: dict, legal, seed, family="dynamic"):
    kind = spec.get("kind", "llm")
    if kind == "surrogate":
        from dreamcache.policies.surrogate import SurrogatePolicy
        return SurrogatePolicy(legal, seed=seed, eps=spec.get("eps", 0.25))
    from dreamcache.policies.ollama_policy import LLMPolicy, OllamaClient
    client = OllamaClient(model=spec.get("model", "qwen3:4b"), think=spec.get("think", False),
                          cache_path="" if spec.get("cache") is False else None,
                          num_predict=spec.get("num_predict", 1024 if spec.get("think") else
                                               160 if spec.get("prompt") == "v3" else 80))
    return LLMPolicy(client, legal, proposal_temperature=spec.get("proposal_temperature", 1.0),
                     prompt=spec.get("prompt", "v1"), family=family)


_PRED_CACHE = {}


def make_prefetcher(fast, pf: dict, legal, env_id, fast_spec, seed):
    """Wrap the actor policy with predictive execution (experiment 2, specgraph)."""
    from dreamcache.policies.heuristic import ObsBFSPolicy
    from dreamcache.specgraph import predictors as P
    from dreamcache.specgraph.prefetch import PrefetchingPolicy
    from dreamcache.specgraph.scheduler import SchedConfig, Scheduler
    name = pf.get("predictor", "context")
    key = (name, pf.get("rows"), tuple(legal))
    if key not in _PRED_CACHE:
        cls = {"frequency": P.FrequencyPredictor, "transition": P.TransitionPredictor,
               "context": P.ContextPredictor, "personalized": P.PersonalizedPredictor,
               "heuristic": P.HeuristicPredictor}[name]
        rows = [json.loads(line) for line in open(pf["rows"])] if pf.get("rows") else []
        wf = pf.get("workflow")
        rows = [{**r, "env": f"{r['env']}|{r['workflow']}"} for r in rows]
        _PRED_CACHE[key] = cls(legal).fit(rows)
    pred = _PRED_CACHE[key]
    wf = pf.get("workflow", "")

    class _Keyed:  # personalization key = env x workflow, matching training rows
        def predict(self, row):
            return pred.predict({**row, "env": f"{row['env']}|{wf}"})

    fp = f"{fast_spec.get('model')}|{fast_spec.get('prompt', 'v1')}"
    return PrefetchingPolicy(fast, legal, _Keyed(), Scheduler(build(SchedConfig, pf.get("sched")), legal, seed),
                             env_id, fp, max_workers=pf.get("workers", 4), heuristic=ObsBFSPolicy(seed=seed))


def run_episode(env_id, seed, method: dict, fast_spec: dict, max_steps=None):
    legal, family = legal_actions(env_id), env_family(env_id)
    env, obs = make_env(env_id, seed)
    kind = method.get("kind", "agent")
    t_start = time.perf_counter()
    records, history = [], []
    total_r, success, failure = 0.0, False, False
    if kind == "agent":
        fast = make_fast(fast_spec, legal, seed, family)
        big = (make_fast({**fast_spec, "model": fast_spec["big_model"]}, legal, seed, family)
               if fast_spec.get("big_model") else None)
        pf_cfg = method.get("prefetch")
        if pf_cfg:
            fast = make_prefetcher(fast, pf_cfg, legal, env_id, fast_spec, seed)
        agent = Agent(fast, legal, family, build(AgentConfig, method.get("agent")), seed, big=big)
    elif kind == "lossless":
        from dreamcache.agents.lossless import LosslessSpecAgent
        small = make_fast(fast_spec, legal, seed, family)
        big = make_fast({**fast_spec, "model": fast_spec["big_model"]}, legal, seed, family)
        agent = LosslessSpecAgent(big, small, legal, guessers=method.get("guessers", ["obs_heuristic"]), seed=seed)
    elif kind == "planner_sim":
        planner = SimBFSPolicy()
    elif kind == "planner_obs":
        planner = ObsBFSPolicy(seed=seed)
    elif kind == "random":
        planner = RandomPolicy(legal, seed)
    limit = max_steps or env.unwrapped.max_steps
    for t in range(limit):
        if kind in ("agent", "lossless"):
            if kind == "agent" and method.get("prefetch"):
                agent.fast.speculate(env, obs, history, t)
            a, rec = agent.act(env, obs, history, t)
        else:
            t0 = time.perf_counter()
            a = (planner.act_sim(env) if kind in ("planner_sim", "random") else planner.act(obs))
            rec = {"t": t, "action": int(a), "triggered": False,
                   "cost": {"wall_seconds": time.perf_counter() - t0}}
        obs, r, term, trunc, _ = env.step(a)
        history.append(int(a))
        total_r += float(r)
        rec["reward"] = float(r)
        records.append(rec)
        if term or trunc:
            success, failure = is_success(r, term), is_failure(r, term)
            break
    extra = agent.finalize() if kind == "lossless" else None
    if kind == "agent" and method.get("prefetch"):
        extra = {"prefetch": agent.fast.finalize()}
    return {"env": env_id, "seed": seed, "method": method["name"], "success": success, "extra": extra,
            "failure": failure, "return": total_r, "steps": len(records),
            "episode_seconds": time.perf_counter() - t_start, "records": records}


def _job(args, attempts=3):
    """Retry a failed episode from scratch (same seed) so transient server errors do not
    silently drop (and bias) episodes; persistent failures are logged, never dropped."""
    err = None
    for i in range(attempts):
        try:
            r = run_episode(*args)
            r["attempts"] = i + 1
            return r
        except Exception:  # noqa: BLE001
            err = traceback.format_exc()
    return {"error": err, "env": args[0], "seed": args[1], "method": args[2]["name"]}


def manifest(cfg, fast_spec):
    def sh(*c):
        try:
            return subprocess.run(c, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            return None
    gc = open("GIT_COMMIT").read().strip() if os.path.exists("GIT_COMMIT") else sh("git", "rev-parse", "HEAD")
    m = {"timestamp": dt.datetime.now().isoformat(), "config": cfg,
         "git_commit": gc, "git_dirty": bool(sh("git", "status", "--porcelain")),
         "python": sys.version, "platform": platform.platform(), "hostname": platform.node(),
         "slurm_job": os.environ.get("SLURM_JOB_ID"), "gpu": sh("nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"),
         "packages": {p: _ver(p) for p in ("minigrid", "gymnasium", "numpy", "pydantic", "httpx")}}
    if fast_spec.get("kind", "llm") == "llm":
        from dreamcache.policies.ollama_policy import OllamaClient
        c = OllamaClient(model=fast_spec.get("model", "qwen3:4b"))
        try:
            m["ollama"] = {"model": c.model, "digest": c.digest(),
                           "version": c.http.get("/api/version").json().get("version"),
                           "num_parallel": os.environ.get("OLLAMA_NUM_PARALLEL"),
                           "temperature_decide": 0.0,
                           "proposal_temperature": fast_spec.get("proposal_temperature", 1.0),
                           "think": fast_spec.get("think", False)}
        except Exception as e:
            m["ollama"] = {"error": repr(e)}
    return m


def _ver(p):
    try:
        return metadata.version(p)
    except metadata.PackageNotFoundError:
        return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--out", default=os.environ.get("DC_RESULTS", "results"))
    ap.add_argument("--workers", type=int, default=int(os.environ.get("DC_WORKERS", "4")))
    ap.add_argument("--only", nargs="*", help="run only these method names")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(open(a.config))
    defaults = cfg.get("defaults", {})
    seeds = list(SPLITS[cfg["split"]])[: cfg.get("n_seeds")]
    fast_spec = cfg.get("fast", {"kind": "llm"})
    name = os.path.splitext(os.path.basename(a.config))[0]
    run_dir = os.path.join(a.out, f"{dt.datetime.now():%Y%m%d-%H%M%S}-{name}")
    os.makedirs(run_dir, exist_ok=True)
    json.dump(manifest(cfg, fast_spec), open(os.path.join(run_dir, "manifest.json"), "w"),
              indent=2, default=str)
    methods = [m if m.get("kind", "agent") != "agent" else
               {**m, "agent": merge(defaults, m.get("agent", {}))} for m in cfg["methods"]]
    if a.only:
        methods = [m for m in methods if m["name"] in a.only]
    jobs = []
    for env_id in cfg["envs"]:
        for m in methods:
            fs = merge(fast_spec, m.get("fast", {}))
            jobs += [(env_id, s, m, fs, cfg.get("max_steps")) for s in seeds]
    print(f"[runner] {len(jobs)} episodes -> {run_dir}", flush=True)
    t0 = time.time()
    out = open(os.path.join(run_dir, "episodes.jsonl"), "w")
    done = 0
    with ProcessPoolExecutor(a.workers) as ex:
        for fut in as_completed([ex.submit(_job, j) for j in jobs]):
            r = fut.result()
            out.write(json.dumps(r, default=float) + "\n")
            out.flush()
            done += 1
            if "error" in r:
                print(f"[runner] ERROR {r['method']} seed={r['seed']}\n{r['error']}", flush=True)
            elif done % 10 == 0 or done == len(jobs):
                print(f"[runner] {done}/{len(jobs)} {time.time()-t0:.0f}s", flush=True)
    out.close()
    print(f"[runner] done {run_dir}", flush=True)
    return run_dir


if __name__ == "__main__":
    main()
