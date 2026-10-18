"""E2E: a new prompt starts a new simulation, never another task's episodes or frames.

Drives the same REST/SSE calls the studio makes (Home -> POST /projects, the chat -> POST /projects/{pid}/agent,
New chat -> POST /projects/{pid}/agent/threads, the viewer -> /rollouts, /episodes, /events, /sim/live) against a
running backend, real MuJoCo sim, no mocks. Run it against an isolated backend (own port, own data dir, no .env):

    CORTEX_DOTENV=0 DATA_DIR=/tmp/cortex-e2e AGENT_ENGINE=rules \\
      uv --project backend run uvicorn cortex.api.app:app --port 8020   (from backend/)
    uv --project backend run python scripts/e2e_new_simulation.py --base http://127.0.0.1:8020/api

The water-bottle demo is slow on a laptop (~2.5 min each): --bottle-demos 0 skips that recording.
--restart "<shell command>" also runs the backend-restart-mid-collection scenario (the command must restart the
backend at --base, e.g. "scripts/stop.sh && scripts/start.sh").
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from collections.abc import Callable

import httpx

CUBE = "pick up the cube"
BOTTLE = "pick up a water bottle and put it in the bin"
RESULTS: list[tuple[str, str, str]] = []


class Api:
    def __init__(self, base: str):
        self.base = base
        self.c = httpx.Client(base_url=base, timeout=900, headers={"X-Cortex-Actor": "gui"})

    def get(self, path: str, **kw):
        r = self.c.get(path, **kw)
        r.raise_for_status()
        return r.json()

    def post(self, path: str, body: dict | None = None):
        r = self.c.post(path, json=body or {})
        r.raise_for_status()
        return r.json()

    def project(self, prompt: str) -> dict:  # Home's submit
        return self.post("/projects", {"task_prompt": prompt})

    def turn(self, pid: str, message: str, thread: str | None = None, stop_after: Callable[[str], bool] | None = None
             ) -> list[tuple[str, dict]]:
        """One chat turn over SSE, parsed like web/src/api/client.ts streamAgent. `stop_after(event)` -> True
        closes the stream there (the studio's Stop / switching chats aborts the fetch)."""
        body = {"message": message, **({"thread": thread} if thread else {})}
        out: list[tuple[str, dict]] = []
        with self.c.stream("POST", f"/projects/{pid}/agent", json=body, headers={"Accept": "text/event-stream"}) as r:
            r.raise_for_status()
            buf = ""
            for chunk in r.iter_text():
                buf += chunk
                while "\n\n" in buf:
                    frame, buf = buf.split("\n\n", 1)
                    ev, data = None, {}
                    for ln in frame.splitlines():
                        if ln.startswith("event: "):
                            ev = ln[7:]
                        elif ln.startswith("data: "):
                            data = json.loads(ln[6:])
                    if ev:
                        out.append((ev, data))
                        if ev == "done" or (stop_after and stop_after(ev)):
                            return out
        return out

    def wait_job(self, job_id: str, timeout: float = 1200) -> dict:
        t0 = time.time()
        while time.time() - t0 < timeout:
            j = self.get(f"/jobs/{job_id}")
            if j["status"] in ("succeeded", "failed", "cancelled"):
                return j
            time.sleep(1)
        raise TimeoutError(job_id)

    def record(self, pid: str, n: int) -> dict:  # Collect page: one demo per condition, auto-accepted
        conds = [{"cube_region": r, "lighting": "normal"} for r in ("center", "left", "right")][:n]
        return self.post(f"/projects/{pid}/recordings/batch", {"conditions": conds, "per_condition": 1,
                                                              "auto_accept": True})

    def dataset(self, pid: str) -> tuple[dict, dict | None]:
        j = self.wait_job(self.post(f"/projects/{pid}/datasets", {})["id"])
        ds = next((d for d in self.get(f"/projects/{pid}/datasets") if d["id"] == j.get("result_ref")), None)
        return j, ds


def scenario(name: str):
    def deco(fn):
        def run(*a, **kw):
            t0 = time.time()
            print(f"\n== {name}", flush=True)
            try:
                note = fn(*a, **kw) or ""
                RESULTS.append((name, "PASS", f"{note} ({time.time() - t0:.0f}s)".strip()))
            except Skip as e:
                RESULTS.append((name, "SKIP", str(e)))
            except Exception as e:  # noqa: BLE001 - every scenario reports, the run goes on
                RESULTS.append((name, "FAIL", f"{type(e).__name__}: {e}"[:300]))
                print(f"   FAIL {e}", flush=True)
        return run
    return deco


class Skip(Exception):
    pass


def check(cond: bool, msg: str) -> None:
    print(("   ok   " if cond else "   FAIL ") + msg, flush=True)
    if not cond:
        raise AssertionError(msg)


def tools(evs) -> list[str]:
    return [d["tool"] for e, d in evs if e == "tool_call"]


def moved(evs) -> dict | None:
    return next((d for e, d in evs if e == "project"), None)


def episode_ids(api: Api, pid: str) -> set[str]:
    return {e["id"] for e in api.get(f"/projects/{pid}/episodes")}


def rollout_ids(api: Api, pid: str) -> set[str]:
    return {r["id"] for r in api.get(f"/projects/{pid}/rollouts")}


# ------------------------------------------------------------------ scenarios

@scenario("1. Home 'pick up the cube' -> built-in scene -> demos -> dataset")
def s_cube(api: Api, st: dict, demos: int):
    p = api.project(CUBE)
    evs = api.turn(p["id"], CUBE)
    check(evs[-1][0] == "done" and moved(evs) is None and "start_intake" not in tools(evs), "cube turn stays put")
    proj = api.get(f"/projects/{p['id']}")
    check(proj["scene"] is None and proj["intake"] is None, "built-in cube tabletop (no intake scene)")
    j = api.wait_job(api.record(p["id"], demos)["id"])
    check(j["status"] == "succeeded", f"{demos} cube demos recorded ({j['status']} {j.get('error') or ''})")
    eps = api.get(f"/projects/{p['id']}/episodes")
    check(len(eps) == demos and all(e["project_id"] == p["id"] for e in eps), f"{len(eps)} episodes, all this project's")
    check(rollout_ids(api, p["id"]) == {e["id"] for e in eps}, "viewer rollouts = this project's demos")
    j, ds = api.dataset(p["id"])
    check(j["status"] == "succeeded" and ds is not None, f"dataset built ({j['status']} {j.get('error') or ''})")
    check({ln["episode_id"] for ln in ds["links"]} <= {e["id"] for e in eps}, "dataset links only this project's demos")
    st["cube"] = p["id"]
    return f"{p['id']}: {len(eps)} demos, dataset {ds['id']}"


@scenario("2. Home 'water bottle in the bin' -> new project, own scene, 0 episodes -> demos -> dataset")
def s_bottle(api: Api, st: dict, demos: int):
    p = api.project(BOTTLE)
    check(p["id"] != st.get("cube"), "a new project")
    evs = api.turn(p["id"], BOTTLE)
    check(evs[-1][0] == "done" and moved(evs) is None, "turn ends in this project")
    check({"start_intake", "build_simulation"} <= set(tools(evs)), "scene built from the prompt")
    built = next(d for e, d in evs if e == "tool_result" and d["tool"] == "build_simulation")
    check(built.get("ok", True) and "built" in built["summary"], built["summary"])
    proj = api.get(f"/projects/{p['id']}")
    check(set(proj["scene"]["objects"]) == {"water_bottle"} and proj["intake"]["task"]["place"] == ["bin"],
          "water bottle scene with a 'bin' zone")
    check(episode_ids(api, p["id"]) == set() and rollout_ids(api, p["id"]) == set(), "0 episodes, 0 rollouts")
    st["bottle"] = p["id"]
    if not demos:
        raise Skip("scene OK, 0 episodes; recording skipped (--bottle-demos 0)")
    j = api.wait_job(api.record(p["id"], demos)["id"])
    check(j["status"] == "succeeded", f"{demos} bottle demo(s) recorded ({j['status']} {j.get('error') or ''})")
    eps = api.get(f"/projects/{p['id']}/episodes")
    check(len(eps) == demos, f"{len(eps)} episodes: success={[e['success'] for e in eps]}")
    j, ds = api.dataset(p["id"])
    check(j["status"] == "succeeded" and ds is not None, f"dataset built ({j['status']} {j.get('error') or ''})")
    check({ln["episode_id"] for ln in ds["links"]} <= {e["id"] for e in eps}, "dataset links only bottle demos")
    return f"{p['id']}: {len(eps)} demos, dataset {ds['id']}"


@scenario("3. New chat in the cube project + 'water bottle' -> its own new project (the 27-episodes bug)")
def s_new_chat(api: Api, st: dict):
    cube = st["cube"]
    before = episode_ids(api, cube)
    t = api.post(f"/projects/{cube}/agent/threads")  # New chat
    evs = api.turn(cube, BOTTLE, t["id"])
    m = moved(evs)
    check(m is not None and m["id"] != cube, f"a new project for the new task ({m and m['id']})")
    note = next((d for e, d in evs if e == "notice"), {})
    check(m["id"] in note.get("ref_ids", []), f"notice: {note.get('text', '')[:110]}")
    proj = api.get(f"/projects/{cube}")
    check(proj["scene"] is None and proj["intake"] is None and episode_ids(api, cube) == before,
          "cube project untouched (no scene swap, same demos)")
    # the studio opens the new project and sends the prompt there (ProjectChatPage autoSend)
    evs2 = api.turn(m["id"], m["prompt"])
    new = api.get(f"/projects/{m['id']}")
    check(moved(evs2) is None and new["scene"] and set(new["scene"]["objects"]) == {"water_bottle"},
          "the new project builds its own bottle scene")
    check(episode_ids(api, m["id"]) == set() and rollout_ids(api, m["id"]) == set(), "new project: 0 episodes/rollouts")
    # the agent's train path there has nothing of the cube's to take
    r = api.c.post("/tools/train_from_agent_demonstrations", json={"project_id": m["id"]},
                   headers={"X-Cortex-Actor": "agent"})
    check(r.status_code == 400 and "no accepted" in r.json()["detail"], f"train refused: {r.json().get('detail')}")
    st["forked"], st["cube_thread"] = m["id"], t["id"]
    return f"{cube} -> {m['id']}"


@scenario("4. Switch back to the first project: its own episodes, rollouts and chats")
def s_switch_back(api: Api, st: dict):
    cube, forked = st["cube"], st["forked"]
    a, b = episode_ids(api, cube), episode_ids(api, forked)
    check(a and not (a & b), f"cube {len(a)} episodes, new project {len(b)}, none shared")
    check(rollout_ids(api, cube) == a, "cube viewer rollouts = cube demos")
    threads = api.get(f"/projects/{cube}/agent/threads")
    check(len(threads) == 2, f"cube project has its 2 chats ({[t['title'] for t in threads]})")
    rows = api.get(f"/projects/{cube}/agent/history", params={"thread": st["cube_thread"]})
    check(any(r["kind"] == "notice" and forked in r.get("ref_ids", []) for r in rows), "that chat links the new project")
    check(api.get(f"/projects/{forked}/agent/threads")[0]["title"].lower().startswith("pick up a water bottle"),
          "the new project's chat holds the prompt")
    # project-scoped live stream: the bottle project's MJPEG never serves the cube project's frame
    with api.c.stream("GET", f"/projects/{forked}/sim/live", timeout=3) as r:
        got = b""
        try:
            for chunk in r.iter_bytes():
                got += chunk
                break
        except httpx.ReadTimeout:
            pass
    check(got == b"", "the new project's live view has no frame (nothing ran there)")


@scenario("5. The cube demos' dataset path: a sim_agent-only snapshot says why, in the call")
def s_dataset_msg(api: Api, st: dict):
    r = api.c.post("/tools/train_from_agent_demonstrations", json={"project_id": st["cube"]},
                   headers={"X-Cortex-Actor": "agent"})
    d = r.json().get("detail", "")
    check(r.status_code == 400 and "sim_agent" in d and "sim_scripted" in d, d)
    check(not [j for j in api.get(f"/projects/{st['cube']}/jobs") if j["kind"] == "train_from_agent"],
          "no background job that fails later")
    r = api.c.post("/tools/train_from_agent_demonstrations", json={"project_id": st["cube"], "sources": ["sim_scripted"]},
                   headers={"X-Cortex-Actor": "agent"})
    j = api.wait_job(r.json()["job_id"])
    check(j["status"] == "succeeded", f"with the scripted sources it builds and proposes ({j['status']} {j.get('error')})")
    tjs = api.get(f"/projects/{st['cube']}/training")
    check(len(tjs) == 1 and tjs[0]["status"] in ("proposed", "approved", "queued"), f"training {tjs[0]['status']}")


@scenario("6. Rapid back-to-back prompts (Home x3, and two at once in one project)")
def s_rapid(api: Api, st: dict):
    out: dict[str, list] = {}

    def home(prompt):
        p = api.project(prompt)
        out[p["id"]] = api.turn(p["id"], prompt)

    ths = [threading.Thread(target=home, args=(x,)) for x in
           (CUBE, "put the tennis ball in the bowl", "put the cup on the plate")]
    [t.start() for t in ths]
    [t.join() for t in ths]
    check(len(out) == 3 and all(e[-1][0] == "done" for e in out.values()), "3 projects, 3 finished turns")
    scenes = {pid: (api.get(f"/projects/{pid}")["scene"] or {}).get("objects") for pid in out}
    check(sorted(str(sorted(s)) if s else "cube" for s in scenes.values()) == ["['cup']", "['tennis_ball']", "cube"],
          f"each its own scene: {scenes}")
    both: list = []
    ths = [threading.Thread(target=lambda m=m: both.append(api.turn(st["cube"], m)))
           for m in ("what should I do next?", "status")]
    [t.start() for t in ths]
    [t.join() for t in ths]
    check(len(both) == 2 and all(e[-1][0] == "done" for e in both), "two prompts at once in one project both finish")


@scenario("7. A prompt naming an unknown object")
def s_unknown(api: Api, st: dict):
    prompt = "pick up the zorblax and put it in the bowl"
    p = api.project(prompt)
    evs = api.turn(p["id"], prompt)
    check(evs[-1][0] == "done" and not any(e == "error" for e, _ in evs), "the turn finishes without an error")
    proj = api.get(f"/projects/{p['id']}")
    check(proj["scene"] is None and moved(evs) is None, "no scene invented for it; the studio shows the 'Asked' line")
    evs = api.turn(st["cube"], prompt)  # in a project with demos: stays (nothing to build), never swaps the scene
    check(moved(evs) is None and api.get(f"/projects/{st['cube']}")["scene"] is None, "cube project unchanged")


@scenario("8. An empty prompt")
def s_empty(api: Api, st: dict):
    p = api.project("Teach a robot to pick and place a cube.")  # what Home sends for an empty box
    evs = api.turn(p["id"], "")
    check(evs[-1][0] == "done", "an empty message ends cleanly")
    r = api.c.post(f"/projects/{p['id']}/agent", json={})
    check(r.status_code == 422, f"no message at all is a 422 ({r.status_code})")


@scenario("9. A very long prompt")
def s_long(api: Api, st: dict):
    prompt = BOTTLE + ". " + "The bottle is on the left side of the table near the edge. " * 120
    p = api.project(prompt)
    check(len(p["name"]) <= 60, f"name stays short ({p['name']!r})")
    evs = api.turn(p["id"], prompt)
    check(evs[-1][0] == "done", "turn finishes")
    proj = api.get(f"/projects/{p['id']}")
    check(proj["scene"] and set(proj["scene"]["objects"]) == {"water_bottle"}, "bottle scene built")
    title = api.get(f"/projects/{p['id']}/agent/threads")[0]["title"]
    check(len(title) <= 60, f"chat title truncated ({len(title)} chars)")
    evs = api.turn(st["cube"], prompt)
    m = moved(evs)
    note = next((d for e, d in evs if e == "notice"), {})
    check(m is not None and len(note.get("text", "")) < 600, "in a project with demos: new project, short notice")


@scenario("10. Two tabs on different projects: each event stream and live view is its own")
def s_two_tabs(api: Api, st: dict):
    a, b = st["cube"], st.get("bottle") or st["forked"]
    seen: dict[str, list[str]] = {a: [], b: []}
    stop = threading.Event()

    def listen(pid):
        try:
            with httpx.Client(base_url=api.base, timeout=None).stream("GET", f"/projects/{pid}/events") as r:
                for line in r.iter_lines():
                    if stop.is_set():
                        return
                    if line.startswith("data: ") and "job_" in line:
                        seen[pid].append(line)
        except Exception:  # noqa: BLE001
            pass

    ths = [threading.Thread(target=listen, args=(x,), daemon=True) for x in (a, b)]
    [t.start() for t in ths]
    time.sleep(1)
    j = api.wait_job(api.record(a, 1)["id"])
    time.sleep(1)
    stop.set()
    check(j["status"] == "succeeded", "a demo recorded in tab A's project")
    check(any(j["id"] in x for x in seen[a]) and not any(j["id"] in x for x in seen[b]),
          f"tab A saw its job ({len(seen[a])} events), tab B saw none of it ({len(seen[b])})")
    check(not (episode_ids(api, a) & episode_ids(api, b)), "episodes never cross projects")


@scenario("11. Autonomous vs manual mode carries to a new task's project")
def s_modes(api: Api, st: dict):
    for mode in ("manual", "autonomous"):
        p = api.project(CUBE)
        api.c.put(f"/projects/{p['id']}/autopilot/policy", json={"autonomy": mode}).raise_for_status()
        api.turn(p["id"], CUBE)
        j = api.wait_job(api.record(p["id"], 1)["id"])
        check(j["status"] == "succeeded", f"{mode}: a demo")
        m = moved(api.turn(p["id"], "now put the cup on the plate"))
        check(m is not None and api.get(f"/projects/{m['id']}")["autopilot"]["autonomy"] == mode,
              f"{mode}: the new project keeps the mode")


@scenario("12. Stop pressed mid-turn (stream aborted + /agent/stop)")
def s_stop(api: Api, st: dict):
    p = api.project("put the tennis ball in the bowl")
    evs = api.turn(p["id"], "put the tennis ball in the bowl", stop_after=lambda e: e == "tool_result")
    r = api.post(f"/projects/{p['id']}/agent/stop")
    check(evs[-1][0] == "tool_result" and "stopped" in r, f"aborted after {len(evs)} events; stop -> {r}")
    time.sleep(2)
    evs = api.turn(p["id"], "put the tennis ball in the bowl")  # the turn lock was released: the next turn runs
    proj = api.get(f"/projects/{p['id']}")
    check(evs[-1][0] == "done" and proj["scene"] and set(proj["scene"]["objects"]) == {"tennis_ball"},
          "the next turn runs and the scene ends up built")


@scenario("13. Empty-project grace period (#172) while a new task's project is set up")
def s_grace(api: Api, st: dict):
    p = api.project(CUBE)
    api.turn(p["id"], CUBE)
    api.wait_job(api.record(p["id"], 1)["id"])
    m = moved(api.turn(p["id"], "put the cup on the plate"))
    listed = {x["id"] for x in api.get("/projects", params={"history": "true"})}
    check(m["id"] not in listed and api.get(f"/projects/{m['id']}")["id"] == m["id"],
          "the new project is hidden until it has history, but not deleted")
    api.turn(m["id"], m["prompt"])
    listed = {x["id"] for x in api.get("/projects", params={"history": "true"})}
    check(m["id"] in listed and p["id"] in listed, "after its first turn Home lists it (and the old one)")


@scenario("14. Switching chat threads while a turn runs")
def s_threads(api: Api, st: dict):
    p = api.project("put the cup on the plate")
    t1 = api.post(f"/projects/{p['id']}/agent/threads")
    res: list = []
    th = threading.Thread(target=lambda: res.append(api.turn(p["id"], "put the cup on the plate", t1["id"])))
    th.start()
    time.sleep(0.3)
    t2 = api.post(f"/projects/{p['id']}/agent/threads")  # History panel: New chat / open another thread
    mid = api.get(f"/projects/{p['id']}/agent/history", params={"thread": t2["id"]})
    th.join()
    h1 = api.get(f"/projects/{p['id']}/agent/history", params={"thread": t1["id"]})
    h2 = api.get(f"/projects/{p['id']}/agent/history", params={"thread": t2["id"]})
    ids = {t["id"] for t in api.get(f"/projects/{p['id']}/agent/threads")}
    check(mid == [] and h2 == [], "the other thread stays empty during and after the turn")
    check(any(r["kind"] == "tool_call" for r in h1) and h1[-1]["kind"] == "done", "the running turn lands in its thread")
    check({t1["id"], t2["id"]} <= ids, "both threads listed")


@scenario("15. Backend restart mid-collection: the job is marked interrupted, the studio recovers")
def s_restart(api: Api, st: dict, cmd: str | None):
    if not cmd:
        raise Skip("pass --restart '<command>' to run it")
    pid = st["cube"]
    job = api.record(pid, 3)
    time.sleep(6)
    subprocess.run(cmd, shell=True, check=True)
    for _ in range(60):
        try:
            api.get("/health")
            break
        except Exception:  # noqa: BLE001
            time.sleep(1)
    j = api.get(f"/jobs/{job['id']}")
    check(j["status"] == "failed" and "restart" in f"{j.get('error')} {j.get('error_detail')}",
          f"job {j['status']}: {j.get('error')}")
    check(not [x for x in api.get(f"/projects/{pid}/jobs") if x["status"] in ("queued", "running")],
          "nothing shows as running (the Live tab turns off)")
    evs = api.turn(pid, "status")
    check(evs[-1][0] == "done", "the chat works after the restart")
    jj = api.wait_job(api.record(pid, 1)["id"])
    check(jj["status"] == "succeeded", "recording works again")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8020/api")
    ap.add_argument("--cube-demos", type=int, default=2)
    ap.add_argument("--bottle-demos", type=int, default=1)
    ap.add_argument("--restart", default=None, help="shell command that restarts the backend at --base")
    ap.add_argument("--only", default=None, help="comma-separated scenario numbers")
    a = ap.parse_args()
    api = Api(a.base)
    h = api.get("/health")
    print("providers:", {k: v["provider"] for k, v in h.items() if isinstance(v, dict)})
    if "local" not in h["db"]["provider"] and "sqlite" not in h["db"]["provider"]:
        sys.exit("refusing to run against a shared database: point --base at an isolated backend")
    st: dict = {}
    plan = [(1, lambda: s_cube(api, st, a.cube_demos)), (2, lambda: s_bottle(api, st, a.bottle_demos)),
            (3, lambda: s_new_chat(api, st)), (4, lambda: s_switch_back(api, st)), (5, lambda: s_dataset_msg(api, st)),
            (6, lambda: s_rapid(api, st)), (7, lambda: s_unknown(api, st)), (8, lambda: s_empty(api, st)),
            (9, lambda: s_long(api, st)), (10, lambda: s_two_tabs(api, st)), (11, lambda: s_modes(api, st)),
            (12, lambda: s_stop(api, st)), (13, lambda: s_grace(api, st)), (14, lambda: s_threads(api, st)),
            (15, lambda: s_restart(api, st, a.restart))]
    only = {int(x) for x in a.only.split(",")} if a.only else None
    for n, fn in plan:
        if only is None or n in only or n in (1, 3):  # 1 and 3 set up the projects the others use
            fn()
    print("\n| scenario | result | notes |\n|---|---|---|")
    for name, res, note in RESULTS:
        print(f"| {name} | {res} | {note} |")
    sys.exit(1 if any(r == "FAIL" for _, r, _ in RESULTS) else 0)


if __name__ == "__main__":
    main()
