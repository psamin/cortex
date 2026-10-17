"""Benchmark the fast policy's Ollama latency/throughput at several concurrencies.

Uses real MiniGrid observations as prompts. Writes JSON to --out.
"""
import argparse
import json
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from dreamcache.environments.minigrid_adapter import legal_actions, make_env
from dreamcache.policies.ollama_policy import LLMPolicy, OllamaClient


def states(env_id, n):
    out, rng = [], np.random.default_rng(0)
    for s in range(n):
        env, obs = make_env(env_id, s)
        for _ in range(int(rng.integers(0, 4))):
            obs, _, term, trunc, _ = env.step(int(rng.integers(0, 2)))
        out.append(obs)
    return out


def gpu_mem():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3:4b")
    ap.add_argument("--env", default="MiniGrid-Dynamic-Obstacles-8x8-v0")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    legal = legal_actions(a.env)
    client = OllamaClient(model=a.model, cache_path="")
    t0 = time.perf_counter(); client.warmup(); warm = time.perf_counter() - t0
    pol = LLMPolicy(client, legal)
    obs = states(a.env, a.n)
    res = {"model": a.model, "digest": client.digest(), "warmup_s": warm, "gpu_mem_after_load": gpu_mem(), "conc": {}}
    for c in [1, 4, 8, 16]:
        def one(o):
            t = time.perf_counter(); d = pol.decide(o, []); return time.perf_counter() - t, d
        t = time.perf_counter()
        with ThreadPoolExecutor(c) as ex:
            outs = list(ex.map(one, obs))
        wall = time.perf_counter() - t
        lat = sorted(x[0] for x in outs)
        res["conc"][c] = {"p50_s": statistics.median(lat), "p95_s": lat[int(0.95 * len(lat)) - 1],
                          "throughput_calls_per_s": len(obs) / wall,
                          "mean_prompt_tokens": statistics.mean(x[1].cost.prompt_tokens for x in outs),
                          "mean_gen_tokens": statistics.mean(x[1].cost.gen_tokens for x in outs),
                          "malformed": sum(x[1].cost.malformed for x in outs),
                          "action_hist": {str(k): sum(x[1].action == k for x in outs) for k in legal},
                          "gpu_mem": gpu_mem()}
        print(a.model, c, res["conc"][c], flush=True)
    sample = pol.client.chat(pol._messages(obs[0], []), pol.schema, 0.0, 0, use_cache=False)[0]
    res["sample_reply"] = sample
    json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
