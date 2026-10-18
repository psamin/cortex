"""The control agent can request, watch and release a Futurama GPU through the MCP (cortex.request_gpu etc.)."""

import pytest
from fastapi.testclient import TestClient

from cortex.services import live_policy_service as lp

MCP = {"X-Cortex-Actor": "mcp"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LIVE_POLICY_URL", "http://127.0.0.1:9")  # nothing listens: no GPU server
    monkeypatch.delenv("FUTURAMA_SSH_HOST", raising=False)
    from cortex.config import settings

    settings.cache_clear()
    lp.STATE.update(job=None, tunnel=None, phase="stopped", step=None, detail=None, error=None, failed_at=0.0)
    from cortex.api.app import app

    with TestClient(app) as c:
        yield c
    lp.STATE.update(job=None, tunnel=None, phase="stopped", step=None, detail=None, error=None, failed_at=0.0)
    settings.cache_clear()


def test_status_says_no_gpu_is_running(client):
    r = client.post("/api/tools/get_gpu_status", json={}, headers=MCP)
    assert r.status_code == 200, r.text
    assert r.json()["available"] is False and r.json()["summary"]


def test_request_explains_when_the_cluster_is_not_configured(client):
    r = client.post("/api/tools/request_gpu", json={}, headers=MCP)
    assert r.status_code == 400 and "FUTURAMA_SSH_HOST" in r.text


def test_request_submits_a_slurm_job(client, monkeypatch):
    from cortex.api.app import ctx

    monkeypatch.setattr(ctx().settings, "futurama_ssh_host", "login.example")  # the app read settings at startup

    async def fake_sh(*args, timeout=60):
        return 0, "12345\n"

    async def no_connect(ctx, job):  # don't start the real poller/tunnel in a test
        return None

    monkeypatch.setattr(lp, "_sh", fake_sh)
    monkeypatch.setattr(lp, "_connect_when_ready", no_connect)
    pid = client.post("/api/tools/start_intake", json={"prompt": "pick up the cube"}, headers=MCP).json()["project_id"]
    r = client.post("/api/tools/request_gpu", json={"project_id": pid}, headers=MCP)
    assert r.status_code == 200, r.text
    assert r.json()["phase"] == "starting" and r.json()["ref_ids"] == ["12345"]
    assert "get_gpu_status" in r.json()["summary"]
