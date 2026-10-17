"""Dev-split prompt/model probe (development only; never run on val/test seeds).

For states sampled from DEV seeds (visited by the obs-only heuristic with random
perturbations), compares each (model, prompt) fast-policy choice with the privileged
SimBFS action and records fatal choices (forward into a visible wall/obstacle in
Dynamic-Obstacles). The privileged label is used only to *diagnose* prompt quality.
"""
import argparse
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from dreamcache.environments.minigrid_adapter import env_family, legal_actions, make_env
from dreamcache.environments.observations import features
from dreamcache.policies.heuristic import ObsBFSPolicy, SimBFSPolicy
from dreamcache.policies.ollama_policy import LLMPolicy, OllamaClient


def sample_states(env_id, n, seed=0):
    rng, out, s = np.random.default_rng(seed), [], 0
    legal = legal_actions(env_id)
    while len(out) < n:
        env, obs = make_env(env_id, s % 20)  # dev seeds 0-19
        pol = ObsBFSPolicy(seed=s)
        for _ in range(int(rng.integers(0, 12))):
            a = pol.act(obs) if rng.random() < 0.7 else int(rng.choice(legal))
            obs, _, term, trunc, _ = env.step(a)
            if term or trunc:
                break
        if not (term or trunc):
            out.append((obs, SimBFSPolicy().act_sim(env)))
        s += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["qwen3:4b"])
    ap.add_argument("--prompts", nargs="+", default=["v1", "v2", "v3"])
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = []
    for env_id in ["MiniGrid-Dynamic-Obstacles-8x8-v0", "MiniGrid-DoorKey-8x8-v0"]:
        legal, fam = legal_actions(env_id), env_family(env_id)
        states = sample_states(env_id, a.n)
        for m in a.models:
            for pr in a.prompts:
                pol = LLMPolicy(OllamaClient(model=m, cache_path="", num_predict=160 if pr == "v3" else 80),
                                legal, prompt=pr, family=fam)
                t = time.perf_counter()
                with ThreadPoolExecutor(8) as ex:
                    decs = list(ex.map(lambda st: pol.decide(st[0], []), states))
                wall = time.perf_counter() - t
                agree = np.mean([d.action == st[1] for d, st in zip(decs, states)])
                fatal = np.mean([d.action == 2 and features(st[0]).front_blocked for d, st in zip(decs, states)])
                r = {"env": env_id, "model": m, "prompt": pr, "agree_with_planner": float(agree),
                     "fatal_rate": float(fatal), "mean_latency_s": float(np.mean([d.cost.wall_seconds for d in decs])),
                     "throughput_8way": len(states) / wall,
                     "mean_gen_tokens": float(np.mean([d.cost.gen_tokens for d in decs])),
                     "malformed": int(sum(d.cost.malformed for d in decs)),
                     "action_hist": dict(Counter(int(d.action) for d in decs)),
                     "planner_hist": dict(Counter(int(st[1]) for st in states)),
                     "mean_top_prob": float(np.mean([max(d.probs.values()) for d in decs]))}
                print(json.dumps(r), flush=True)
                res.append(r)
    json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
