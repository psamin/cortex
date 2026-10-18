"""Hackathon demo driver: autonomous agent demonstrations end to end, against the live backend.

    uv --project backend run python scripts/demo_agent.py --demos 3
    uv --project backend run python scripts/demo_agent.py --demos 3 --collector oracle   # no model key needed
    uv --project backend run python scripts/demo_agent.py --project prj_... --improve   # targeted round

Prints the agent's executed actions live (from the backend event stream), the
measured outcomes, provenance and spend, the LeRobot dataset version, and, if a
trained checkpoint exists, its evaluation and the targeted improvement round.
Open the studio's Collect tab alongside to watch the same execution.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time

import httpx


def stream(base: str, pid: str, stop: threading.Event) -> None:
    with httpx.Client(timeout=None) as c, c.stream("GET", f"{base}/projects/{pid}/events") as r:
        ev = None
        for line in r.iter_lines():
            if stop.is_set():
                return
            if line.startswith("event: "):
                ev = line[7:]
            elif line.startswith("data: ") and ev in ("agent_step", "agent_attempt"):
                d = json.loads(line[6:])
                if ev == "agent_attempt":
                    print(f"  ── attempt {d['index'] + 1}: {d['status'].upper()}  seed={d['seed']}  "
                          f"calls={d['usage'].get('calls', 0)}  ${d['usage'].get('cost_usd', 0):.3f}  "
                          f"recoveries={d['recoveries']}", flush=True)
                elif d.get("type") == "attempt_start":
                    s = d["scenario"]
                    print(f"\n▶ attempt {d['index'] + 1}/{d['total']}  cube={s['cube_region']} light={s['lighting']} "
                          f"seed={d['seed']}", flush=True)
                elif d.get("type") == "agent_step":
                    err = "" if d["ok"] else f"  ✗ {d['result']['error']['code']}: {d['result']['error']['message']}"
                    print(f"    {d['phase']:<26} {d['tool']}{err}", flush=True)
                elif d.get("type") == "episode_start":
                    print(f"\n▶ oracle demo {d['index'] + 1}/{d['total']}  {d['condition']}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000/api")
    ap.add_argument("--project")
    ap.add_argument("--demos", type=int, default=3)
    ap.add_argument("--collector", default="astra", choices=["astra", "oracle"])
    ap.add_argument("--improve", action="store_true", help="run improve_policy on an existing project")
    a = ap.parse_args()
    c = httpx.Client(base_url=a.base, timeout=120, headers={"X-Cortex-Actor": "gui"})

    pl = c.get("/planner").json()
    print("planner:", pl.get("route") or pl.get("reason"))
    if a.collector == "astra" and not pl["available"]:
        sys.exit(f"The agent planner is not available ({pl.get('reason')}). Start Claude Code and the MCP "
                 "server, or rerun with --collector oracle.")

    pid = a.project or c.post("/projects", json={"task_prompt": "Train my SO-101 to pick up a cube and place it "
                                                                "in a bowl."}).json()["id"]
    print("project:", pid, "  studio: http://127.0.0.1:5173/p/" + pid + "/collect")
    stop = threading.Event()
    threading.Thread(target=stream, args=(a.base, pid, stop), daemon=True).start()
    time.sleep(0.5)

    if a.improve:
        r = c.post("/tools/improve_policy", json={"project_id": pid, "max_new_demos": a.demos}).json()
        print("\nimprove_policy:", r["summary"])
        cid = r.get("collection_id")
    else:
        r = c.post(f"/projects/{pid}/agent-collections", json={"num_demos": a.demos, "collector": a.collector,
                                                               "diversity": "medium"})
        if r.status_code >= 400:
            sys.exit(r.json()["detail"])
        cid = r.json()["id"]
    if cid:
        while (col := c.get(f"/agent-collections/{cid}").json())["collection"]["status"] in ("queued", "running"):
            time.sleep(1)
        t = col["collection"]["totals"]
        print(f"\ncollection {cid}: {t['successes']}/{t['attempts']} simulator-verified demos, "
              f"{t['model_calls']} model calls, ${t['cost_usd']:.2f}, {t['recoveries']} recoveries "
              f"[{col['collection']['provider'] or 'scripted IK oracle'} · {col['collection']['model']}]")
    stop.set()

    eps = [e for e in c.get(f"/projects/{pid}/episodes").json() if e["status"] == "accepted"]
    if eps and not a.improve:
        j = c.post(f"/projects/{pid}/datasets", json={"name": "demo", "eval_fraction": 0.0}).json()
        while (jj := c.get(f"/jobs/{j['id']}").json())["status"] in ("queued", "running"):
            time.sleep(1)
        ds = c.get(f"/projects/{pid}/datasets").json()[-1]
        print(f"dataset {ds['id']}: LeRobotDataset v3, {len(ds['links'])} episodes, stored via "
              f"{c.get('/health').json()['storage']['provider']}")
    cks = c.get(f"/projects/{pid}/checkpoints").json()
    if cks:
        print(f"latest checkpoint {cks[-1]['id']}: evaluate with cortex.evaluate_learned_policy, then --improve")
    else:
        print("no trained checkpoint yet: propose training from the Improve tab (approval required)")
    print(f"\nClaude Code:  claude -p \"Use the cortex MCP tools to show demonstration results for {pid}\"")


if __name__ == "__main__":
    main()
