"""Build transition rows (and the SQLite execution graph) from logged trajectories.

Episodes are replayed from reset(seed) with their logged actions (MiniGrid is
deterministic given seed + actions), so each row's context is recomputed from the
observation the agent actually saw. Row = (workflow, env, episode, t, context,
prev_op, op=executed action, heuristic_op). The label `op` is the action executed AT
that step; the context uses only the observation and history BEFORE it.

Usage: python -m dreamcache.specgraph.dataset RUN_DIR... --methods A B --workflow NAME --out rows.jsonl [--db graph.sqlite]
"""
from __future__ import annotations

import argparse
import json
import os

from dreamcache.environments.minigrid_adapter import make_env
from dreamcache.policies.heuristic import ObsBFSPolicy
from dreamcache.specgraph.graph import ExecGraph, Transition, context_key


def episode_rows(ep, workflow):
    env, obs = make_env(ep["env"], ep["seed"])
    heur = ObsBFSPolicy(seed=ep["seed"])
    rows, hist = [], []
    for rec in ep["records"]:
        a = int(rec["action"])
        rows.append({"workflow": workflow, "env": ep["env"], "episode": f"{ep['method']}:{ep['seed']}",
                     "seed": ep["seed"], "t": rec["t"], "context": context_key(obs, hist),
                     "prev_op": hist[-1] if hist else -1, "op": a, "heuristic_op": heur.act(obs),
                     "latency_s": rec.get("cost", {}).get("wall_seconds", 0.0)})
        obs, _, term, trunc, _ = env.step(a)
        hist.append(a)
        if term or trunc:
            break
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--workflow", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--db")
    a = ap.parse_args(argv)
    g = ExecGraph(a.db) if a.db else None
    n = 0
    with open(a.out, "a") as f:
        for r in a.runs:
            for line in open(os.path.join(r, "episodes.jsonl")):
                ep = json.loads(line)
                if "error" in ep or ep["method"] not in a.methods:
                    continue
                for row in episode_rows(ep, a.workflow):
                    f.write(json.dumps(row) + "\n")
                    n += 1
                    if g:
                        g.add(Transition(row["workflow"], row["env"], row["episode"], row["t"], row["context"],
                                         str(row["prev_op"]), str(row["op"])), duration=row["latency_s"])
    if g:
        g.commit()
    print(f"{n} rows -> {a.out}")


if __name__ == "__main__":
    main()
