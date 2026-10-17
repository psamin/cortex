"""Offline ORACLE criticality labels for detector evaluation (Section 10).

Replays logged trajectories from reset(seed) (the env is deterministic given the
seed and action sequence) to recover each encountered state, then estimates
Q_ref(s, a) for every legal action with the privileged reference evaluator.
Decision regret = max_a Q_ref - Q_ref(fast_action). The detector signals that were
logged online are joined to the labels, so any signal weighting can be evaluated
offline without new model calls.

Usage: python -m dreamcache.experiments.label_states RUN_DIR [RUN_DIR ...] --out labels.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

from dreamcache.agents.oracle import q_ref
from dreamcache.environments.minigrid_adapter import legal_actions, make_env


def label_episode(ep, n, horizon, max_states):
    env, _ = make_env(ep["env"], ep["seed"])
    legal = legal_actions(ep["env"])
    out, history = [], []
    for rec in ep["records"][:max_states]:
        if "signals" in rec:
            q = q_ref(env, legal, history, n=n, horizon=horizon, seed=ep["seed"] * 1000 + rec["t"])
            fa = rec["fast_action"]
            out.append({"env": ep["env"], "seed": ep["seed"], "method": ep["method"], "t": rec["t"],
                        "fast_action": fa, "signals": rec["signals"], "difficulty": rec["difficulty"],
                        "q": {str(k): v for k, v in q.items()}, "regret": max(q.values()) - q[fa],
                        "best_action": max(q, key=q.get), "fast_conf": max(map(float, rec["probs"].values()))})
        _, _, term, trunc, _ = env.step(rec["action"])
        history.append(rec["action"])
        if term or trunc:
            break
    return out


def _job(a):
    return label_episode(*a)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--methods", nargs="*")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--max-states", type=int, default=60)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args(argv)
    eps = []
    for r in a.runs:
        for line in open(os.path.join(r, "episodes.jsonl")):
            e = json.loads(line)
            if "error" not in e and (not a.methods or e["method"] in a.methods):
                eps.append(e)
    with ProcessPoolExecutor(a.workers) as ex, open(a.out, "w") as f:
        for rows in ex.map(_job, [(e, a.n, a.horizon, a.max_states) for e in eps]):
            for row in rows:
                f.write(json.dumps(row) + "\n")
    print(f"labeled states from {len(eps)} episodes -> {a.out}")


if __name__ == "__main__":
    main()
