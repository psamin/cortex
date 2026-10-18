"""Live GPU inference from the studio (#157): status of the tunneled server, start/stop, tunnel command."""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from cortex.services import live_policy_service as lp

POLICIES = {"gpu": "NVIDIA H200", "policies": {"jenga": {"checkpoint": "jenga_010000", "stage": "jenga",
                                                         "held_out": [58, 59, 60]}}}


@pytest.fixture(autouse=True)
def _reset_state():
    lp.STATE.update(job=None, tunnel=None, phase="stopped", error=None)
    yield
    lp.STATE.update(job=None, tunnel=None, phase="stopped", error=None)


def _ctx(url: str, host: str | None = None):
    return SimpleNamespace(settings=SimpleNamespace(
        live_policy_url=url, futurama_ssh_host=host, futurama_policy_dir="~/cortex-jenga",
        futurama_node_domain="cc.gatech.edu"))


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
    assert not st["available"] and "Start one" in st["reason"]


async def test_start_needs_the_cluster_configured():
    with pytest.raises(ValueError, match="FUTURAMA_SSH_HOST"):
        await lp.start(_ctx("http://127.0.0.1:9"))


async def test_tunnel_goes_through_the_gpu_node(monkeypatch):
    calls = []

    async def fake_exec(*args, **kw):
        calls.append(args)
        return SimpleNamespace(returncode=None, terminate=lambda: None)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    await lp.open_tunnel(_ctx("http://127.0.0.1:8765", "pi34@login.example"), "angleyne", 30568)
    args = calls[0]
    assert args[args.index("-J") + 1] == "pi34@login.example"
    assert args[args.index("-L") + 1] == "8765:127.0.0.1:30568"
    assert args[-1] == "pi34@angleyne.cc.gatech.edu"
    assert lp.STATE["phase"] == "ready"


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
