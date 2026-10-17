"""Track B runner. Phase 1 (reference, policy=none) on TRAIN tasks builds history and the
LLM response cache; predictors are fit on train trajectories and FROZEN; then every policy
runs on EVAL tasks in replay mode (identical decisions + think times), `--repeats` times.

Usage: python -m dreamcache.codebench.run --out results/codebench --model qwen3:4b [--repeats 2]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import tempfile

from dreamcache.codebench.speculator import OpPredictor, Speculator
from dreamcache.codebench.tasks import make_task, task_ids
from dreamcache.codebench.tools import fingerprint
from dreamcache.policies.ollama_policy import OllamaClient

POLICIES = [("none", None), ("memo", None), ("all", None), ("random", None),
            ("top1_frequency", "frequency"), ("top1_global", "global"), ("top1_personalized", "personalized"),
            ("topk2_personalized", "personalized"), ("utility_personalized", "personalized")]


def rows_of(results):
    return [{"prev_op": s["prev_op"], "last_ok": s["last_ok"], "family": s["family"], "op": s["op"]}
            for r in results for s in r["steps"]]


def run_split(split, client, policy, predictor, fp, replay, out_f, rep=0, run_tag=""):
    from dreamcache.codebench.agent import run_task
    res = []
    for t, m in task_ids():
        d = tempfile.mkdtemp(prefix="cb-")
        ws = os.path.join(d, "ws")
        meta = make_task(ws, t, m, split)
        pol = policy.split("_")[0].replace("topk2", "topk")
        sp = Speculator(pol, predictor, fp=fp, k=2)
        r = run_task(client, ws, meta, sp, replay=replay)
        r["spec"] = sp.finalize()
        r.update(policy=policy, split=split, rep=rep, run=run_tag)
        out_f.write(json.dumps(r) + "\n")
        out_f.flush()
        res.append(r)
        shutil.rmtree(d, ignore_errors=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="qwen3:4b")
    ap.add_argument("--repeats", type=int, default=2)
    a = ap.parse_args()
    run_dir = os.path.join(a.out, f"{dt.datetime.now():%Y%m%d-%H%M%S}-codebench")
    os.makedirs(run_dir, exist_ok=True)
    cache = os.path.join(os.environ.get("SCR", tempfile.gettempdir()), "cb_cache.sqlite")
    client = OllamaClient(model=a.model, num_predict=400, cache_path=cache)
    fp = fingerprint()
    json.dump({"model": a.model, "digest": client.digest(), "fingerprint": fp, "timestamp": dt.datetime.now().isoformat(),
               "git_commit": open("GIT_COMMIT").read().strip() if os.path.exists("GIT_COMMIT") else None,
               "policies": POLICIES, "repeats": a.repeats}, open(os.path.join(run_dir, "manifest.json"), "w"), indent=2)
    f = open(os.path.join(run_dir, "tasks.jsonl"), "w")
    train = run_split("train", client, "none", None, fp, False, f, run_tag="history")
    print(f"[codebench] train history: {len(train)} tasks, success={sum(r['success'] for r in train)}", flush=True)
    rows = rows_of(train)
    # reference eval run (real LLM calls) populates the cache; then replayed variants
    run_split("eval", client, "none", None, fp, False, f, run_tag="reference")
    for rep in range(a.repeats):
        for name, mode in POLICIES:
            pred = OpPredictor(mode).fit(rows) if mode else None
            rs = run_split("eval", client, name, pred, fp, True, f, rep=rep, run_tag="replay")
            print(f"[codebench] rep={rep} {name}: wall={sum(r['wall_s'] for r in rs):.1f}s "
                  f"success={sum(r['success'] for r in rs)}/{len(rs)}", flush=True)
    f.close()
    print(f"[codebench] done {run_dir}", flush=True)


if __name__ == "__main__":
    main()
