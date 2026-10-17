"""A/B: scripted IK oracle vs GPT-6 Astra demonstrations on matched scenes and budgets.

    # 1. collect both arms on identical scenes/seeds and report collection metrics
    uv --project backend run python scripts/ab_oracle_vs_astra.py collect --n 10 --diversity medium
    # 2. build one dataset per arm and propose ACT training for each (approval needed to run)
    uv --project backend run python scripts/ab_oracle_vs_astra.py train --project prj_...
    # 3. once both checkpoints exist, evaluate both on the same held-out seeds and compare
    uv --project backend run python scripts/ab_oracle_vs_astra.py evaluate --project prj_... --oracle-ck ck_... --astra-ck ck_...

Everything printed comes from the backend's stored measurements; arms that were not
run are reported as "not run". Reports are written to .cortex-data/reports/.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import httpx

REPORTS = Path(__file__).resolve().parents[1] / ".cortex-data" / "reports"


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    z, p = 1.96, k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def wait_collection(c: httpx.Client, cid: str) -> dict:
    while True:
        r = c.get(f"/agent-collections/{cid}").json()
        if r["collection"]["status"] not in ("queued", "running"):
            return r
        t = r["collection"]["totals"]
        print(f"  {cid}: {t.get('attempts', 0)} attempts, {t.get('successes', 0)} ok", end="\r", flush=True)
        time.sleep(3)


def wait_job(c: httpx.Client, jid: str) -> dict:
    while (j := c.get(f"/jobs/{jid}").json())["status"] in ("queued", "running"):
        time.sleep(2)
    return j


def arm_metrics(r: dict) -> dict:
    att = r["attempts"]
    n = len(att)
    ok = sum(a["status"] == "success" for a in att)
    return {"collection_id": r["collection"]["id"], "provider": r["collection"]["provider"],
            "model": r["collection"]["model"], "mode": r["collection"]["mode"], "status": r["collection"]["status"],
            "attempts": n, "successes": ok, "success_rate": ok / n if n else None, "ci95": wilson(ok, n),
            "recoveries": sum(a["recoveries"] for a in att),
            "mean_demo_sim_s": round(sum(a["num_frames"] for a in att) / max(n, 1) / 30, 2),
            "mean_wall_s": round(sum(a["wall_s"] for a in att) / max(n, 1), 2),
            "model_calls": sum(int(a["usage"].get("calls", 0)) for a in att),
            "cost_usd": round(sum(float(a["usage"].get("cost_usd", 0)) for a in att), 4),
            "cells_covered": len({(a["scenario"]["cube_region"], a["scenario"]["lighting"]) for a in att
                                  if a["status"] == "success"}),
            "episode_ids": [a["episode_id"] for a in att if a["episode_id"]],
            "seeds": [a["seed"] for a in att]}


def save(name: str, data: dict) -> Path:
    REPORTS.mkdir(parents=True, exist_ok=True)
    p = REPORTS / f"{name}_{int(time.time())}.json"
    p.write_text(json.dumps(data, indent=1))
    return p


def cmd_collect(c: httpx.Client, a) -> None:
    if not c.get("/planner").json().get("available"):
        sys.exit("GPT-6 Astra is not configured; the A/B needs it (no other model is substituted).")
    pid = a.project or c.post("/projects", json={"task_prompt": "Pick up the cube and place it in the bowl",
                                                 "demo_budget": a.n}).json()["id"]
    base = {"num_demos": a.n, "diversity": a.diversity, "base_seed": a.base_seed}
    arms = {}
    for coll in ("oracle", "astra"):
        body = {**base, "collector": coll}
        if coll == "astra":
            body["max_total_usd"] = a.max_usd
        r = c.post(f"/projects/{pid}/agent-collections", json=body)
        r.raise_for_status()
        print(f"{coll}: collection {r.json()['id']}")
        arms[coll] = arm_metrics(wait_collection(c, r.json()["id"]))
    assert arms["oracle"]["seeds"] == arms["astra"]["seeds"], "scenes are not matched"
    print("\nmatched seeds:", arms["astra"]["seeds"])
    keys = ["attempts", "successes", "success_rate", "ci95", "recoveries", "mean_demo_sim_s", "mean_wall_s",
            "model_calls", "cost_usd", "cells_covered"]
    print(f"\n{'metric':<18}{'scripted oracle':>22}{'GPT-6 Astra':>22}")
    for k in keys:
        fmt = (lambda v: f"({v[0]:.2f}, {v[1]:.2f})") if k == "ci95" else (
            lambda v: "n/a" if v is None else f"{v:.2f}" if isinstance(v, float) else str(v))
        print(f"{k:<18}{fmt(arms['oracle'][k]):>22}{fmt(arms['astra'][k]):>22}")
    print("\nreport:", save("ab_collect", {"project_id": pid, "arms": arms}))
    print(f"next: scripts/ab_oracle_vs_astra.py train --project {pid}")


def cmd_train(c: httpx.Client, a) -> None:
    out = {}
    for coll, src in (("oracle", "sim_scripted"), ("astra", "sim_agent")):
        j = c.post(f"/projects/{a.project}/datasets", json={"name": f"ab-{coll}", "sources": [src]}).json()
        ds = wait_job(c, j["id"])["result_ref"]
        tj = c.post(f"/projects/{a.project}/training/proposals", json={"dataset_version_id": ds}).json()
        out[coll] = {"dataset_version_id": ds, "training_job_id": tj["id"], "status": tj["status"]}
        print(f"{coll}: dataset {ds} -> training {tj['id']} ({tj['status']}; approve it in the studio)")
    print("report:", save("ab_train", {"project_id": a.project, **out}))


def cmd_evaluate(c: httpx.Client, a) -> None:
    runs = {}
    for coll, ck in (("oracle", a.oracle_ck), ("astra", a.astra_ck)):
        r = c.post(f"/projects/{a.project}/evaluations", json={
            "policy": {"kind": "checkpoint", "checkpoint_id": ck}, "episodes_per_branch": a.episodes,
            "base_seed": 1000, "name": f"ab-{coll}"}).json()
        runs[coll] = r["run"]["id"]
    for coll, ev in runs.items():
        while (r := c.get(f"/evaluations/{ev}").json())["run"]["status"] not in ("succeeded", "failed"):
            time.sleep(3)
    cmp = c.post(f"/projects/{a.project}/comparisons", json={"evaluation_ids": list(runs.values())}).json()
    for row in cmp["rows"]:
        print(f"{row['evaluation']:<12} {row['branch']:<17} {row['successes']}/{row['n']}")
    print(cmp["summary"])
    print("report:", save("ab_evaluate", {"project_id": a.project, "evaluations": runs, "comparison": cmp}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["collect", "train", "evaluate"])
    ap.add_argument("--base", default="http://127.0.0.1:8000/api")
    ap.add_argument("--project")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--diversity", default="medium", choices=["plan", "medium", "high"])
    ap.add_argument("--base-seed", type=int, default=31_000)
    ap.add_argument("--max-usd", type=float, default=8.0)
    ap.add_argument("--oracle-ck")
    ap.add_argument("--astra-ck")
    ap.add_argument("--episodes", type=int, default=10)
    a = ap.parse_args()
    c = httpx.Client(base_url=a.base, timeout=120, headers={"X-Cortex-Actor": "gui"})
    {"collect": cmd_collect, "train": cmd_train, "evaluate": cmd_evaluate}[a.cmd](c, a)


if __name__ == "__main__":
    main()
