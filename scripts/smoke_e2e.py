"""End-to-end smoke test against a running backend (real sim, no mocks).

    uv --project backend run python scripts/smoke_e2e.py [--base http://127.0.0.1:8000/api] [--small]

Walks the hackathon acceptance sequence: create project -> record genuine sim
demos -> quality/VLM analysis -> evidence-backed recommendation -> dataset
version (LeRobot v3) -> training proposal -> parallel 3-branch evaluation ->
comparison -> the same project through the MCP tool endpoint -> audit log.
"""

from __future__ import annotations

import argparse
import sys
import time

import httpx

ok_count = 0


def check(cond: bool, msg: str) -> None:
    global ok_count
    print(("  ok  " if cond else "  FAIL ") + msg, flush=True)
    if not cond:
        sys.exit(1)
    ok_count += 1


def wait_job(c: httpx.Client, job_id: str, timeout: float = 600) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = c.get(f"/jobs/{job_id}").json()
        if j["status"] in ("succeeded", "failed", "cancelled"):
            return j
        time.sleep(1)
    raise TimeoutError(job_id)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000/api")
    ap.add_argument("--small", action="store_true", help="fewer demos/episodes for a quick run")
    a = ap.parse_args()
    c = httpx.Client(base_url=a.base, timeout=120, headers={"X-Cortex-Actor": "gui"})
    t0 = time.time()

    h = c.get("/health").json()
    print("providers:", {k: v["provider"] for k, v in h.items() if isinstance(v, dict)})

    print("1. create project")
    p = c.post("/projects", json={"task_prompt": "Teach a robot to pick and place a cube.", "mode": "sim",
                                  "demo_budget": 12}).json()
    pid = p["id"]
    check(p["spec"]["objects"] == ["cube", "bowl"] and len(p["plan"]["cells"]) == 5, f"project {pid} with plan")

    print("2. record genuine sim demos")
    conds = [{"cube_region": "center", "lighting": "normal"}, {"cube_region": "right", "lighting": "normal"},
             {"cube_region": "center", "lighting": "dim"}]
    per = 1 if a.small else 2
    j = c.post(f"/projects/{pid}/recordings/batch", json={"conditions": conds, "per_condition": per,
                                                          "auto_accept": True}).json()
    j = wait_job(c, j["id"])
    check(j["status"] == "succeeded", f"record job {j['status']} {j.get('error') or ''}")
    eps = c.get(f"/projects/{pid}/episodes").json()
    check(len(eps) == 3 * per and all(e["num_frames"] > 30 for e in eps), f"{len(eps)} episodes with frames")
    check(all(e["quality"] is not None for e in eps), "quality reports attached")
    dim = [e for e in eps if e["condition"]["lighting"] == "dim"]
    print("     dim-light flags:", [f["code"] for f in dim[0]["quality"]["flags"]])
    m = c.get(f"/episodes/{eps[0]['id']}/media").json()
    check(c.get(m["video_url"].replace("http://localhost:8000", a.base.rsplit("/api", 1)[0])).status_code == 200,
          "episode video served")

    print("3. analyze")
    j = wait_job(c, c.post(f"/projects/{pid}/analyses", json={"use_vlm": True}).json()["id"])
    check(j["status"] == "succeeded", f"analysis {j['status']}")
    an = c.get(f"/projects/{pid}/analyses/latest").json()
    print("    ", an["summary"])

    print("4. recommend")
    recs = c.post(f"/projects/{pid}/recommendations", json={}).json()
    check(len(recs) > 0 and recs[0]["evidence"], f"{len(recs)} recommendations")
    for r in recs[:3]:
        print(f"     [{r['support']}] {r['title']}")
    check(any("left" in r["title"] for r in recs), "uncovered left region is recommended")

    print("5. dataset version with augmentation")
    j = wait_job(c, c.post(f"/projects/{pid}/datasets", json={
        "augmentations": [{"type": "brightness", "params": {"factor": 0.6}}], "eval_fraction": 0.0}).json()["id"])
    check(j["status"] == "succeeded", f"dataset job {j['status']} {j.get('error') or ''}")
    ds = c.get(f"/projects/{pid}/datasets").json()[-1]
    n_aug = sum(1 for ln in ds["links"] if ln["augmentation"])
    check(n_aug == len(eps) and len({ln["episode_id"] for ln in ds["links"]}) == len(eps),
          f"{len(ds['links'])} LeRobot episodes = {len(eps)} sources + {n_aug} augmented (linked)")

    print("6. training proposal + approval gate")
    tj = c.post(f"/projects/{pid}/training/proposals", json={"dataset_version_id": ds["id"]}).json()
    check(tj["status"] == "proposed" and tj["estimate"]["gpu_hours"] > 0, f"{tj['id']} proposed on {tj['backend']}")
    r = c.post(f"/training/{tj['id']}/launch")
    check(r.status_code == 403, "launch refused before approval")
    r = c.post("/tools/launch_training_job", json={"training_job_id": tj["id"]}, headers={"X-Cortex-Actor": "mcp"})
    check(r.status_code == 403, "MCP cannot self-approve training")

    print("7. parallel 3-branch evaluation (scripted expert reference)")
    n = 2 if a.small else 4
    ev = c.post(f"/projects/{pid}/evaluations", json={"policy": {"kind": "scripted_expert"},
                                                      "episodes_per_branch": n, "executor": "local"}).json()
    evid = ev["run"]["id"]
    for _ in range(600):
        ev = c.get(f"/evaluations/{evid}").json()
        if ev["run"]["status"] in ("succeeded", "failed"):
            break
        time.sleep(1)
    check(ev["run"]["status"] == "succeeded", f"evaluation {ev['run']['status']}")
    for b in ev["branches"]:
        r = b["result"]
        print(f"     {b['spec']['name']:<17} {r['successes']}/{r['n']} seeds={b['seeds']} "
              f"CI95=({r['ci95'][0]:.2f},{r['ci95'][1]:.2f}) wall={r['wall_s']:.0f}s")
    check(len(ev["branches"]) == 3 and all(b["result"]["n"] == n for b in ev["branches"]), "3 branches measured")

    print("8. compare + next actions")
    cmp = c.post(f"/projects/{pid}/comparisons", json={}).json()
    check(len(cmp["rows"]) == 3, cmp["summary"])
    nxt = c.get(f"/projects/{pid}/next-actions").json()
    print("     next:", [x["action"] for x in nxt])

    print("9. same project via MCP tool endpoint")
    st = c.post("/tools/get_project_state", json={"project_id": pid}, headers={"X-Cortex-Actor": "mcp"}).json()
    check(st["state"]["project"]["id"] == pid, st["summary"])
    audit = c.get(f"/projects/{pid}/audit").json()
    check(any(x["actor"] == "mcp" for x in audit), f"{len(audit)} audit rows incl. mcp")
    print(f"\nALL {ok_count} CHECKS PASSED in {time.time() - t0:.0f}s  (project {pid})")


if __name__ == "__main__":
    main()
