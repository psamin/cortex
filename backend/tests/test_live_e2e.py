"""The studio's Inference tab end to end, against a fake Futurama (fake_futurama.py).

A real uvicorn server runs the app; `ssh`, `squeue`, `sbatch` and `scancel` are the fakes on PATH, so
live_policy_service runs its real commands, its real tunnel subprocess and its real stream proxy.
Covers: requesting a GPU and narrating each Slurm step, attaching to a running job, the tunnel, streaming
and recording, and the edge cases a demo hits (no VPN, queue waits, a job that dies, stop, two viewers,
a backend restart, a stream that drops, long episodes, projects matched by name).
"""

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest
import uvicorn

import fake_futurama
from fake_futurama import FakeServe, free_port
from cortex.services import live_policy_service as lp


def _reset():
    """What a fresh backend process knows. Its tunnel subprocess belongs to a server loop that may be gone."""
    t = lp.STATE.get("tunnel")
    if t is not None:
        try:
            os.kill(t.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    lp.STATE.update(job=None, tunnel=None, phase="stopped", step=None, detail=None, error=None, failed_at=0.0)


@pytest.fixture
def futurama(tmp_path, monkeypatch):
    serve = FakeServe()
    cluster = fake_futurama.install(tmp_path / "futurama", serve.port)
    monkeypatch.setenv("PATH", f"{tmp_path / 'futurama' / 'bin'}:{os.environ['PATH']}")
    yield cluster, serve
    serve.close()


@pytest.fixture
def api(tmp_path, monkeypatch, futurama):
    """The backend on a free port, with the cluster configured and a fast poll."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("LIVE_POLICY_URL", f"http://127.0.0.1:{free_port()}")
    monkeypatch.setenv("FUTURAMA_SSH_HOST", "futurama")
    monkeypatch.setenv("FUTURAMA_PROJECTS_FILE", str(tmp_path / "projects.json"))
    for k in ("INSFORGE_BASE_URL", "INSFORGE_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    (tmp_path / "projects.json").write_text("{}")
    monkeypatch.setattr(lp, "POLL_S", 0.05)
    from cortex.config import settings

    settings.cache_clear()
    from cortex.api.app import app

    _reset()
    port = free_port()
    run = {}

    def up():
        run["server"] = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
        run["thread"] = threading.Thread(target=run["server"].run, daemon=True)
        run["thread"].start()
        for _ in range(200):
            if run["server"].started:
                break
            time.sleep(0.02)

    def down():
        run["server"].should_exit = True
        run["thread"].join(10)

    def restart():
        """A backend restart: a graceful shutdown, then a process that knows nothing (no state, no tasks)."""
        tunnel = lp.STATE.get("tunnel")
        down()
        _reset()
        up()
        return tunnel

    up()
    with httpx.Client(base_url=f"http://127.0.0.1:{port}/api", timeout=30) as c:
        c.restart = restart
        yield c
    down()
    _reset()
    settings.cache_clear()


def _until(api, done, timeout=15.0):
    """Poll GET /live-policy like the studio does; returns the last status and every distinct line shown."""
    shown, st = [], {}
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = api.get("/live-policy").json()
        line = (st.get("step"), st.get("reason"))
        if not shown or shown[-1] != line:
            shown.append(line)
        if done(st):
            return st, shown
        time.sleep(0.03)
    raise AssertionError(f"timed out; last status {st}, shown {shown}")


def _ready(st):
    return st["available"]


def _project(api, prompt, name=None):
    return api.post("/projects", json={"task_prompt": prompt, **({"name": name} if name else {})}).json()["id"]


def _replays(api, pid, n=1, timeout=15.0):
    rows = []
    t0 = time.time()
    while time.time() - t0 < timeout:
        rows = api.get(f"/projects/{pid}/replays").json()
        if len(rows) >= n:
            return rows
        time.sleep(0.1)
    return rows


# ------------------------------------------------------------------ the main path
def test_opening_the_tab_requests_a_gpu_narrates_it_and_streams(api, futurama, tmp_path):
    cluster, serve = futurama
    assert api.get("/live-policy").json()["phase"] == "stopped"
    st = api.post("/live-policy/start").json()
    job = st["job"]
    assert job and "Submitted Slurm job" in st["reason"]
    st, shown = _until(api, _ready)
    assert not any("isn't reachable" in (r or "") for _, r in shown)  # the tunnel opening is still "Connecting"
    at = [list(lp.STEPS).index(s) for s, _ in shown if s]
    assert at == sorted(at)  # the steps only move forward
    steps = {s for s, _ in shown}
    assert {"queued", "ready"} <= steps and len(steps & {"install", "prepare", "load"}) >= 2
    assert any("queued: Resources" in (r or "") for _, r in shown)
    assert st["gpu"] == "NVIDIA L40S" and st["policies"]["jenga"]["held_out"] == [58, 59]
    sbatch = cluster.read()["jobs"][job]["argv"]
    assert "--gres=gpu:l40s:1" in sbatch and sbatch[-1].endswith("/cortex-jenga")
    tunnel = next(c for c in cluster.calls() if "-N" in c)
    assert tunnel[tunnel.index("-J") + 1] == "futurama" and tunnel[-1] == "pi34@zapp.cc.gatech.edu"
    assert tunnel[tunnel.index("-L") + 1].endswith(f":127.0.0.1:{serve.port}")

    # a real-data replay and a sim run, each recorded into its project
    real, sim = _project(api, "Pull a block out of a Jenga tower"), _project(api, "Jenga sim", name="Jenga (sim)")
    (tmp_path / "projects.json").write_text(json.dumps({"jenga": real, "sim_jenga": sim}))
    assert api.get("/live-policy").json()["projects"] == {"jenga": real, "sim_jenga": sim}
    r = api.get("/live-policy/stream/jenga/58")
    assert r.status_code == 200 and r.content.count(b"--frame") == 40
    r = api.get("/live-policy/sim/jenga/123456")
    assert r.status_code == 200 and r.content.count(b"--frame") == 45
    rp = _replays(api, real)[0]
    assert rp["kind"] == "live_inference" and rp["status"] == "complete" and rp["source"].endswith(f"Slurm job {job}")
    rp = _replays(api, sim)[0]
    assert rp["kind"] == "live_sim" and rp["status"] == "complete" and rp["episodes"][0]["success"] is True
    rows = [x for x in api.get(f"/projects/{sim}/rollouts").json() if x["kind"] == "replay"]
    assert rows[0]["label"] == "Live sim · seed 123456" and rows[0]["video_url"]


def test_it_attaches_to_a_gpu_job_that_is_already_running(api, futurama):
    cluster, serve = futurama
    job = cluster.add_job(age=5)  # past READY
    st = api.post("/live-policy/start").json()
    assert st["job"] == job and "already allocated" in st["reason"]
    _until(api, _ready)
    assert len(cluster.read()["jobs"]) == 1  # nothing new submitted


# ------------------------------------------------------------------ cluster edge cases
def test_without_the_vpn_start_fails_fast_with_a_clear_message(api, futurama):
    cluster, _ = futurama
    cluster.set(unreachable=True)
    t0 = time.time()
    r = api.post("/live-policy/start")
    assert r.status_code == 400 and "VPN" in r.json()["detail"] and time.time() - t0 < 10
    st = api.get("/live-policy").json()
    assert st["phase"] == "stopped" and "VPN" in st["error"]
    cluster.set(unreachable=False)  # VPN on: Try again works at once (the cluster was never asked, so no cooldown)
    assert api.post("/live-policy/start").status_code == 200
    _until(api, _ready)


def test_a_hung_ssh_does_not_leave_the_tab_spinning(api, futurama, monkeypatch):
    cluster, _ = futurama
    monkeypatch.setattr(lp, "SSH_TIMEOUT_S", 1.0)
    cluster.set(hang_s=5)
    r = api.post("/live-policy/start")
    assert r.status_code == 400 and "Futurama" in r.json()["detail"]
    st = api.get("/live-policy").json()
    assert st["phase"] == "stopped" and st["error"]  # the studio offers "Try again", not an endless spinner


def test_a_tunnel_that_fails_says_why(api, futurama):
    cluster, _ = futurama
    cluster.set(tunnel_fails=True)
    cluster.add_job(age=5)
    api.post("/live-policy/start")
    st, shown = _until(api, lambda s: s.get("error"), timeout=10)
    assert "Couldn't open the tunnel to zapp" in st["error"] and "No route to host" in st["error"]
    assert all(s != "ready" for s, _ in shown)


def test_a_queue_wait_shows_slurms_reason(api, futurama):
    cluster, _ = futurama
    cluster.set(timeline=[[0.0, "PENDING (QOSMaxGRESPerUser)", None]])
    api.post("/live-policy/start")
    st, _ = _until(api, lambda s: "QOSMaxGRESPerUser" in (s.get("reason") or ""))
    assert st["step"] == "queued" and not st["available"]


def test_a_job_that_dies_before_ready_says_so(api, futurama):
    cluster, _ = futurama
    cluster.set(timeline=[[0.0, "RUNNING zapp", "[live] Installing LeRobot on zapp (NVIDIA L40S)"],
                          [0.3, None, None]])  # e.g. pip failed: the job left the queue
    api.post("/live-policy/start")
    st, shown = _until(api, lambda s: s.get("error"), timeout=10)
    assert "ended before the server was ready" in st["error"] and st["phase"] == "stopped"


def test_losing_the_vpn_while_starting_is_not_narrated_as_progress(api, futurama):
    cluster, _ = futurama
    cluster.set(timeline=[[0.0, "PENDING (Resources)", None]])
    api.post("/live-policy/start")
    _until(api, lambda s: "Resources" in (s.get("reason") or ""))
    cluster.set(unreachable=True)
    st, _ = _until(api, lambda s: "reach Futurama" in (s.get("reason") or ""), timeout=10)
    assert "GPU allocated" not in st["reason"]
    cluster.set(unreachable=False)  # back on the VPN: it carries on
    _until(api, lambda s: "Resources" in (s.get("reason") or ""))


def test_stop_cancels_the_job_and_no_stale_error_follows(api, futurama):
    cluster, _ = futurama
    cluster.set(timeline=[[0.0, "PENDING (Resources)", None]])
    job = api.post("/live-policy/start").json()["job"]
    _until(api, lambda s: s.get("step") == "queued" and "Resources" in (s.get("reason") or ""))
    st = api.post("/live-policy/stop").json()
    assert st["phase"] == "stopped" and cluster.read()["jobs"][job]["gone"]
    time.sleep(0.5)  # the background poll must not report the cancelled job as "ended before ready"
    st = api.get("/live-policy").json()
    assert st["phase"] == "stopped" and not st["error"]


def test_restarting_right_after_stop_is_not_clobbered_by_the_old_job(api, futurama):
    cluster, _ = futurama
    cluster.set(timeline=[[0.0, "PENDING (Resources)", None], [0.4, "RUNNING zapp", None]])
    api.post("/live-policy/start")
    api.post("/live-policy/stop")
    cluster.set(timeline=fake_futurama.timeline(cluster.read()["serve_port"]))
    second = api.post("/live-policy/start").json()["job"]
    st, _ = _until(api, _ready)
    assert st["available"] and lp.STATE["job"] == second and not st.get("error")


def test_stop_without_a_known_job_cancels_the_serve_job_by_name(api, futurama):
    """After a backend restart the studio no longer knows the job id, but Stop GPU must still stop it."""
    cluster, _ = futurama
    job = cluster.add_job(age=5)
    api.post("/live-policy/start")
    _until(api, _ready)
    lp.STATE["job"] = None  # what a restarted backend knows
    api.post("/live-policy/stop")
    assert cluster.read()["jobs"][job]["gone"]


def _gone(pid, wait=3.0):
    t0 = time.time()
    while time.time() - t0 < wait:
        try:
            if os.waitpid(pid, os.WNOHANG)[0]:
                return True
        except ChildProcessError:  # already reaped
            return True
        time.sleep(0.05)
    return False


def test_a_backend_restart_while_starting_reattaches(api, futurama):
    cluster, _ = futurama
    job = api.post("/live-policy/start").json()["job"]
    api.restart()
    st = api.get("/live-policy").json()
    assert st["phase"] == "stopped" and "Click Request a GPU" in st["reason"]
    st = api.post("/live-policy/start").json()  # one click attaches to the job, no second job
    assert st["job"] == job
    _until(api, _ready)
    assert len(cluster.read()["jobs"]) == 1


def _dev_sh_uvicorn_options(port: int) -> list[str]:
    """scripts/dev.sh's uvicorn options, on another port: the backend runs the way the studio runs it."""
    sh = (Path(__file__).resolve().parents[2] / "scripts" / "dev.sh").read_text().replace("\\\n", " ")
    opts = sh.split("uvicorn cortex.api.app:app", 1)[1].split(")", 1)[0].split()
    return [*opts[:opts.index("--port") + 1], str(port), *opts[opts.index("--port") + 2:]]


def test_a_stopped_backend_leaves_no_tunnel_behind(tmp_path, futurama):
    """uvicorn stopped with SIGTERM (scripts/dev.sh's cleanup, a restart) while a studio tab holds its event
    stream: it must exit and take its `ssh -N` with it, or the tunnel keeps the local port and the next GPU's
    tunnel can't open."""
    cluster, _ = futurama
    cluster.add_job(age=5)
    port, backend = free_port(), Path(__file__).resolve().parents[1]
    env = {k: v for k, v in os.environ.items() if not k.startswith(("INSFORGE_", "OPENROUTER_"))}
    env.update(CORTEX_DOTENV="0", DATA_DIR=str(tmp_path / "data"), LIVE_POLICY_URL=f"http://127.0.0.1:{free_port()}",
               FUTURAMA_SSH_HOST="futurama", FUTURAMA_PROJECTS_FILE=str(tmp_path / "projects.json"),
               PYTHONPATH=str(backend))
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "cortex.api.app:app", *_dev_sh_uvicorn_options(port)],
                            cwd=backend, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port}/api", timeout=30) as c:
            for _ in range(150):
                try:
                    c.get("/health")
                    break
                except httpx.HTTPError:
                    time.sleep(0.1)
            pid_ = _project(c, "Jenga sim", name="Jenga (sim)")
            tab = threading.Thread(target=_hold_events, args=(port, pid_), daemon=True)  # an open studio tab
            tab.start()
            c.post("/live-policy/start")
            _until(c, _ready)
        pid = cluster.tunnels()[-1]["pid"]
        proc.terminate()
        proc.wait(20)  # uvicorn waits for open connections; dev.sh caps that
        t0 = time.time()
        while time.time() - t0 < 5 and _alive(pid):
            time.sleep(0.05)
        assert not _alive(pid)
    finally:
        proc.kill()
        for t in cluster.tunnels():
            if _alive(t["pid"]):
                os.kill(t["pid"], signal.SIGTERM)


def _hold_events(port: int, pid: str) -> None:
    try:
        with httpx.stream("GET", f"http://127.0.0.1:{port}/api/projects/{pid}/events", timeout=None) as r:
            for _ in r.iter_raw():
                pass
    except httpx.HTTPError:
        pass


def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def test_a_backend_restart_closes_its_tunnel(api, futurama):
    """After a restart the job is cancelled by name and a new GPU's tunnel opens on the same local port."""
    cluster, _ = futurama
    job = cluster.add_job(age=5)
    api.post("/live-policy/start")
    _until(api, _ready)
    tunnel = api.restart()
    assert tunnel is not None and _gone(tunnel.pid)
    api.post("/live-policy/stop")  # the job id is unknown now: cancelled by name
    assert cluster.read()["jobs"][job]["gone"]
    cluster.set(timeline=fake_futurama.timeline(cluster.read()["serve_port"]))
    api.post("/live-policy/start")  # a new job, a new tunnel on the same local port
    st, _ = _until(api, _ready)
    assert st["available"] and lp.STATE["job"] != job


# ------------------------------------------------------------------ the server's contents
def test_a_server_without_sim_policies_yet(api, futurama):
    cluster, serve = futurama
    serve.info["sim"] = {}
    cluster.add_job(age=5)
    api.post("/live-policy/start")
    st, _ = _until(api, _ready)
    assert st["sim"] == {}
    r = api.get("/live-policy/sim/jenga/7")
    assert r.status_code == 404 and "No sim-trained policy" in r.json()["detail"]


def test_a_viewer_who_leaves_mid_run_is_saved_as_partial_and_frees_the_gpu(api, futurama, tmp_path):
    cluster, serve = futurama
    cluster.add_job(age=5)
    api.post("/live-policy/start")
    _until(api, _ready)
    sim = _project(api, "Jenga sim", name="Jenga (sim)")
    (tmp_path / "projects.json").write_text(json.dumps({"sim_jenga": sim}))
    serve.frames["sim"], serve.fps, serve.succeed_at = 600, 60, 500  # left while still running
    with api.stream("GET", "/live-policy/sim/jenga/11") as r:
        n = 0
        for chunk in r.iter_raw():
            n += chunk.count(b"--frame")
            if n >= 40:
                break
    rp = _replays(api, sim)[0]
    assert rp["status"] == "partial"
    for _ in range(100):
        if serve.ended:
            break
        time.sleep(0.05)
    assert serve.ended and serve.ended[0][2] == "viewer left"  # the GPU server is free for the next run



def test_a_stream_that_drops_mid_run_is_saved_as_partial(api, futurama, tmp_path):
    cluster, serve = futurama
    cluster.add_job(age=5)
    api.post("/live-policy/start")
    _until(api, _ready)
    sim = _project(api, "Jenga sim", name="Jenga (sim)")
    (tmp_path / "projects.json").write_text(json.dumps({"sim_jenga": sim}))
    serve.frames["sim"], serve.drop_after, serve.succeed_at = 200, 60, 150  # cut while still running
    with api.stream("GET", "/live-policy/sim/jenga/9") as r:
        got = b""
        try:
            for chunk in r.iter_raw():
                got += chunk
        except httpx.HTTPError:
            pass
    rp = _replays(api, sim)[0]
    assert rp["status"] == "partial" and rp["episodes"][0]["frames"] == 60


def test_stop_while_streaming_ends_the_stream_and_keeps_the_partial_run(api, futurama, tmp_path):
    cluster, serve = futurama
    job = cluster.add_job(age=5)
    api.post("/live-policy/start")
    _until(api, _ready)
    sim = _project(api, "Jenga sim", name="Jenga (sim)")
    (tmp_path / "projects.json").write_text(json.dumps({"sim_jenga": sim}))
    serve.frames["sim"], serve.fps, serve.succeed_at = 600, 60, 500  # left while still running
    out = {}

    def watch():
        n = 0
        try:
            with api.stream("GET", "/live-policy/sim/jenga/12") as r:
                for chunk in r.iter_raw():
                    n += chunk.count(b"--frame")
        except httpx.HTTPError as e:
            out["error"] = type(e).__name__
        out["frames"] = n

    t = threading.Thread(target=watch)
    t.start()
    time.sleep(1.0)
    st = api.post("/live-policy/stop").json()
    t.join(10)
    assert not t.is_alive() and 0 < out["frames"] < 600
    assert st["phase"] == "stopped" and cluster.read()["jobs"][job]["gone"]
    assert _replays(api, sim)[0]["status"] == "partial"


def test_two_viewers_the_newest_run_is_the_one_that_plays(api, futurama, tmp_path):
    """The GPU server runs one stream at a time. A second viewer (or "Run on a new layout" while a run
    plays) must not sit on a blank image until the first run ends: the newest request takes over."""
    cluster, serve = futurama
    cluster.add_job(age=5)
    api.post("/live-policy/start")
    _until(api, _ready)
    serve.frames["sim"], serve.fps = 1200, 60  # a 20 s run
    first = {}

    def watch():
        n = 0
        with api.stream("GET", "/live-policy/sim/jenga/1") as r:
            for chunk in r.iter_raw():
                n += chunk.count(b"--frame")
        first["frames"] = n

    t = threading.Thread(target=watch)
    t.start()
    time.sleep(1.0)
    t0 = time.time()
    with api.stream("GET", "/live-policy/sim/jenga/2") as r:
        for chunk in r.iter_raw():
            if b"--frame" in chunk:
                break
    waited = time.time() - t0
    t.join(30)
    assert waited < 3, f"the second viewer waited {waited:.1f} s for its first frame"
    assert first["frames"] < 1200  # the first run gave way


# ------------------------------------------------------------------ long runs, projects by name
def test_a_long_palletizing_run_is_recorded_whole(api, futurama, tmp_path):
    cluster, serve = futurama
    serve.info["sim"]["palletize"] = {**serve.info["sim"]["jenga"], "instruction": "unload the boxes"}
    cluster.add_job(age=5)
    api.post("/live-policy/start")
    _until(api, _ready)
    pid = _project(api, "Move boxes off a forklift onto a pallet", name="Palletizing (sim)")
    (tmp_path / "projects.json").write_text(json.dumps({"sim_palletize": pid}))
    serve.frames["sim"], serve.max_steps, serve.succeed_at = 1300, 1300, 1250
    r = api.get("/live-policy/sim/palletize/5")
    assert r.content.count(b"--frame") == 1300
    ep = _replays(api, pid, timeout=60)[0]["episodes"][0]
    assert ep["frames"] == 1300 and ep["steps"] == 1300 and ep["success"] is True and ep["video_key"]


def test_a_sim_run_is_saved_in_the_project_named_for_its_task(api, futurama):
    """projects.json has no sim_<task> entry (the import ran elsewhere): the project is found by its name."""
    cluster, serve = futurama
    cluster.add_job(age=5)
    api.post("/live-policy/start")
    _until(api, _ready)
    pid = _project(api, "Push a block out of a Jenga tower (MuJoCo sim)", name="Jenga (sim)")
    assert api.get("/live-policy").json()["projects"] == {"sim_jenga": pid}  # the studio opens it on Jenga
    api.get("/live-policy/sim/jenga/77")
    rows = _replays(api, pid, timeout=10)
    assert rows and rows[0]["kind"] == "live_sim"
