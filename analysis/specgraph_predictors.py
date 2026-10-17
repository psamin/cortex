"""H1/H4 offline: predictor quality vs history size; global vs personalized at matched budget.

Splits are by SEED SPLIT (train = val-split seeds 1000-1049, eval = test-split seeds
10000+), so no evaluation episode contributes to training. Within train, history size
n = number of training trajectories (episodes), sampled in a fixed order.

Usage: python analysis/specgraph_predictors.py --train rows_train.jsonl --eval rows_eval.jsonl --out DIR
"""
import argparse
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from dreamcache.specgraph.predictors import (ContextPredictor, FrequencyPredictor, HeuristicPredictor,  # noqa: E402
                                             PersonalizedPredictor, TransitionPredictor, evaluate)

SIZES = [0, 10, 25, 50, 100, 250, 500]


def load(p):
    return [json.loads(line) for line in open(p)]


def by_episode(rows):
    d = defaultdict(list)
    for r in rows:
        d[(r["workflow"], r["env"], r["episode"])].append(r)
    return d


def keyed(rows):
    """Personalization key: environment x workflow (actor/method)."""
    return [{**r, "env": f"{r['env']}|{r['workflow']}"} for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    train, ev = keyed(load(a.train)), keyed(load(a.eval))
    ops = sorted({r["op"] for r in train + ev})
    eps = list(by_episode(train).values())
    res = {"n_train_rows": len(train), "n_eval_rows": len(ev), "n_train_episodes": len(eps),
           "envs": sorted({r["env"] for r in ev}), "curve": {}, "per_env": {}}
    models = {"uniform": None, "frequency": FrequencyPredictor, "transition": TransitionPredictor,
              "context(global)": ContextPredictor, "personalized": PersonalizedPredictor,
              "heuristic": HeuristicPredictor}
    for name, M in models.items():
        res["curve"][name] = {}
        for n in [s for s in SIZES if s <= len(eps)] + ([len(eps)] if len(eps) not in SIZES else []):
            vals = []
            for rep in range(a.reps if 0 < n < len(eps) else 1):
                rng = np.random.default_rng(rep)
                idx = rng.permutation(len(eps))[:n]
                tr = [r for i in idx for r in eps[i]]
                if M is None:
                    m = evaluate(FrequencyPredictor(ops, smoothing=1e9).fit([]), ev)
                else:
                    m = evaluate(M(ops).fit(tr), ev)
                vals.append(m)
            res["curve"][name][n] = {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}
    # per-environment breakdown at full data, global vs personalized (H4)
    for env in res["envs"]:
        sub = [r for r in ev if r["env"] == env]
        res["per_env"][env] = {nm: evaluate(M(ops).fit(train), sub) for nm, M in
                               [("frequency", FrequencyPredictor), ("context(global)", ContextPredictor),
                                ("personalized", PersonalizedPredictor), ("heuristic", HeuristicPredictor)]}
    # matched data budget: personalized on k local episodes vs global on k random episodes
    res["matched_budget"] = {}
    for env in res["envs"]:
        local = [e for e in eps if e[0]["env"] == env]
        sub = [r for r in ev if r["env"] == env]
        for k in [s for s in (10, 25, 50) if s <= len(local)]:
            rng = np.random.default_rng(0)
            loc = [r for i in rng.permutation(len(local))[:k] for r in local[i]]
            glo = [r for i in rng.permutation(len(eps))[:k] for r in eps[i]]
            res["matched_budget"][f"{env}|k={k}"] = {
                "global_random_k": evaluate(ContextPredictor(ops).fit(glo), sub)["top1"],
                "local_k": evaluate(ContextPredictor(ops).fit(loc), sub)["top1"],
                "personalized(all global + local)": evaluate(PersonalizedPredictor(ops).fit(train), sub)["top1"]}
    json.dump(res, open(os.path.join(a.out, "predictors.json"), "w"), indent=2)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name, c in res["curve"].items():
        xs = sorted(c, key=int)
        ax[0].plot([int(x) for x in xs], [c[x]["top1"] for x in xs], marker="o", label=name)
        ax[1].plot([int(x) for x in xs], [c[x]["nll"] for x in xs], marker="o", label=name)
    for x, yl in zip(ax, ("top-1 accuracy (held-out)", "NLL (held-out)")):
        x.set_xscale("symlog", linthresh=10)
        x.set_xlabel("training trajectories")
        x.set_ylabel(yl)
        x.grid(alpha=0.3)
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "predictor_vs_history.png"), dpi=130)
    print(json.dumps({k: v for k, v in res.items() if k != "curve"}, indent=1)[:3000])
    for name, c in res["curve"].items():
        last = c[max(c, key=int)]
        print(f"{name:18s} top1={last['top1']:.3f} top3={last['top3']:.3f} nll={last['nll']:.3f} "
              f"brier={last['brier']:.3f} ece={last['ece']:.3f}")


if __name__ == "__main__":
    main()
