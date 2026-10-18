"""Live GPU inference from the studio (#157): status of the tunneled server, start/stop, tunnel command."""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from cortex.services import live_policy_service as lp

POLICIES = {"gpu": "NVIDIA H200", "policies": {"jenga": {"checkpoint": "jenga_010000", "stage": "jenga",
                                                         "held_out": [58, 59, 60]}}}


@pytest.fixture(autouse=True)
def _reset_state():
    lp.STATE.update(job=None, tunnel=None, phase="stopped", step=None, detail=None, error=None, failed_at=0.0)
    yield
    lp.STATE.update(job=None, tunnel=None, phase="stopped", step=None, detail=None, error=None, failed_at=0.0)


def _ctx(url: str, host: str | None = None):
    return SimpleNamespace(settings=SimpleNamespace(
        live_policy_url=url, futurama_ssh_host=host, futurama_policy_dir="~/cortex-jenga",
        futurama_node_domain="cc.gatech.edu", live_policy_gres="gpu:l40s:1",
        futurama_projects_file=Path("/nonexistent/projects.json")))


@pytest.fixture
def server():
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            body = json.dumps(POLICIES).encode()
            self.send_response(200 if self.path == "/policies.json" else 404)
            self.end_headers()
            self.wfile.write(body)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


async def test_status_reports_the_served_policies(server):
    st = await lp.status(_ctx(server))
    assert st["available"] and st["gpu"] == "NVIDIA H200" and st["policies"]["jenga"]["held_out"] == [58, 59, 60]


async def test_status_explains_when_nothing_is_running():
    st = await lp.status(_ctx("http://127.0.0.1:9"))
    assert not st["available"] and "Click Request a GPU" in st["reason"]


async def test_start_needs_the_cluster_configured():
    with pytest.raises(ValueError, match="FUTURAMA_SSH_HOST"):
        await lp.start(_ctx("http://127.0.0.1:9"))


class _Tunnel:
    """A running `ssh -N` (the server fixture stands for the far end)."""
    returncode, stderr = None, None

    async def wait(self):
        await asyncio.sleep(3600)

    def terminate(self):
        pass


async def test_tunnel_goes_through_the_gpu_node(monkeypatch, server):
    calls = []

    async def fake_exec(*args, **kw):
        calls.append(args)
        return _Tunnel()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    await lp.open_tunnel(_ctx(server, "pi34@login.example"), "angleyne", 30568)
    args = calls[0]
    assert args[args.index("-J") + 1] == "pi34@login.example"
    assert args[args.index("-L") + 1] == f"{server.rsplit(':', 1)[1]}:127.0.0.1:30568"
    assert args[-1] == "pi34@angleyne.cc.gatech.edu"
    assert lp.STATE["phase"] == "ready"  # only once the server answered through the tunnel


async def test_stop_cancels_the_job(monkeypatch):
    sent = []

    async def fake_sh(*args, timeout=60):
        sent.append(args)
        return 0, ""

    monkeypatch.setattr(lp, "_sh", fake_sh)
    lp.STATE.update(job="10568", phase="ready")
    st = await lp.stop(_ctx("http://127.0.0.1:9", "pi34@login.example"))
    assert sent and sent[0][-1] == "scancel 10568"
    assert lp.STATE["job"] is None and not st["available"]


async def test_tunnel_user_comes_from_the_ssh_config_for_an_alias(monkeypatch, server):
    calls = []

    async def fake_sh(*args, timeout=60):
        return 0, "hostname planetexpress-login.cc.gatech.edu\nuser pi34\n"

    async def fake_exec(*args, **kw):
        calls.append(args)
        return _Tunnel()

    monkeypatch.setattr(lp, "_sh", fake_sh)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    await lp.open_tunnel(_ctx(server, "futurama"), "angleyne", 30579)
    assert calls[0][-1] == "pi34@angleyne.cc.gatech.edu" and calls[0][calls[0].index("-J") + 1] == "futurama"


def test_narration_follows_the_job():
    assert lp.narrate("1", "PENDING (Resources)", "")[1] == "Waiting for a free GPU (Slurm job 1 queued: Resources)"
    assert lp.narrate("1", "RUNNING angleyne", "[live] Installing LeRobot on zapp (NVIDIA L40S)") == (
        "install", "Installing LeRobot on zapp (NVIDIA L40S)")
    assert lp.narrate("1", "RUNNING angleyne", "[live] Preparing held-out episodes: jenga")[0] == "prepare"
    assert lp.narrate("1", "RUNNING angleyne", "[serve] READY port=30579 policies=['jenga']")[0] == "connect"


async def _no_connect(ctx, job):  # the background connect is tested on its own below
    return None


async def test_start_attaches_to_an_allocated_gpu(monkeypatch):
    sent = []

    async def fake_sh(*args, timeout=60):
        sent.append(args[-1])
        return 0, "ATTACH 10579\n"

    monkeypatch.setattr(lp, "_sh", fake_sh)
    monkeypatch.setattr(lp, "_connect_when_ready", _no_connect)
    st = await lp.start(_ctx("http://127.0.0.1:9", "futurama"))
    assert lp.STATE["job"] == "10579" and "already allocated" in st["reason"]
    assert "squeue -u $USER -n cortex-serve" in sent[0]


async def test_start_requests_the_configured_gpu(monkeypatch):
    sent = []

    async def fake_sh(*args, timeout=60):
        sent.append(args[-1])
        return 0, "10580\n"

    monkeypatch.setattr(lp, "_sh", fake_sh)
    monkeypatch.setattr(lp, "_connect_when_ready", _no_connect)
    await lp.start(_ctx("http://127.0.0.1:9", "futurama"))
    assert lp.STATE["job"] == "10580" and "--gres=gpu:l40s:1" in sent[0]


async def test_connects_once_the_server_is_ready(monkeypatch):
    polls = iter([("PENDING (Resources)", "", None),
                  ("RUNNING zapp", "[live] Preparing held-out episodes: jenga", None),
                  ("RUNNING zapp", "[serve] READY port=30580", {"node": "zapp", "port": 30580, "job": 10580})])
    seen, opened = [], []

    async def fake_poll(ctx, job):
        return (0, *next(polls))

    async def fake_open(ctx, node, port):
        opened.append((node, port))

    async def no_sleep(_):
        seen.append(lp.STATE["step"])

    monkeypatch.setattr(lp, "_poll", fake_poll)
    monkeypatch.setattr(lp, "open_tunnel", fake_open)
    monkeypatch.setattr(lp.asyncio, "sleep", no_sleep)
    lp.STATE["job"] = "10580"
    await lp._connect_when_ready(_ctx("http://127.0.0.1:9", "futurama"), "10580")
    assert seen == ["queued", "prepare"] and opened == [("zapp", 30580)]


def test_a_job_that_left_the_queue_has_ended():
    """Futurama's Slurm drops a finished job from squeue within seconds; `squeue -j` then prints an error."""
    assert lp.job_ended("slurm_load_jobs error: Invalid job id specified")
    assert lp.job_ended("") and lp.job_ended("FAILED NonZeroExitCode") and lp.job_ended("CANCELLED None")
    assert not lp.job_ended("PENDING (Resources)") and not lp.job_ended("RUNNING zapp")


async def test_a_failed_request_cools_down_before_asking_the_cluster_again(monkeypatch):
    """#221: never hammer the shared cluster: after a failure, start() submits nothing for a while."""
    sent = []

    async def fake_sh(*args, timeout=60):
        sent.append(args[-1])
        return (1, "sbatch: error: Batch job submission failed") if len(sent) == 1 else (0, "NONE\n")

    monkeypatch.setattr(lp, "_sh", fake_sh)
    with pytest.raises(RuntimeError):
        await lp.start(_ctx("http://127.0.0.1:9", "futurama"))
    with pytest.raises(RuntimeError, match="protects your Futurama access"):
        await lp.start(_ctx("http://127.0.0.1:9", "futurama"))
    assert "sbatch" in sent[0] and "sbatch" not in sent[1]  # during the cooldown: look for a server, submit nothing


async def test_during_the_cooldown_an_already_running_server_is_still_attached(monkeypatch):
    """A failed attach (e.g. to a job that was just ending) must not lock the studio out of a live server."""
    sent = []

    async def fake_sh(*args, timeout=60):
        sent.append(args[-1])
        return 0, "ATTACH 10639\n"

    monkeypatch.setattr(lp, "_sh", fake_sh)
    monkeypatch.setattr(lp, "_connect_when_ready", _no_connect)
    lp.STATE["failed_at"] = lp.time.time()  # a request failed a moment ago
    await lp.start(_ctx("http://127.0.0.1:9", "futurama"))
    assert lp.STATE["job"] == "10639" and "sbatch" not in sent[0] and "-t PENDING,CONFIGURING,RUNNING" in sent[0]
