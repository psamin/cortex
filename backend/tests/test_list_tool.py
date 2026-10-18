"""cortex.list: one compact read-only query over everything; cortex.cancel_job stops agent work."""

import asyncio

import pytest
from fastapi.testclient import TestClient

MCP = {"X-Cortex-Actor": "mcp"}
KINDS = ["projects", "sim_tasks", "skills", "jobs", "datasets", "checkpoints", "training", "evaluations", "replays",
         "rl_sandboxes", "agent_log"]


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from cortex.config import settings

    settings.cache_clear()
    from cortex.api.app import app

    with TestClient(app) as c:
        yield c
    settings.cache_clear()


def call(c, name, **args):
    return c.post(f"/api/tools/{name}", json=args, headers=MCP)


def test_every_kind_answers_compactly(client):
    pid = call(client, "start_intake", prompt="pick up the cube").json()["project_id"]
    for kind in KINDS:
        args = {"kind": kind} if kind in ("projects", "sim_tasks", "skills") else {"kind": kind, "project_id": pid}
        r = call(client, "list", **args)
        assert r.status_code == 200, (kind, r.text)
        body = r.json()
        assert body["kind"] == kind and isinstance(body["items"], list) and body["summary"]
    projects = call(client, "list", kind="projects").json()["items"]
    assert any(p["id"] == pid for p in projects)
    tasks = {t["task"]: t for t in call(client, "list", kind="sim_tasks").json()["items"]}
    assert "jenga" in tasks and "conveyor" in tasks and tasks["jenga"]["instruction"]
    log = call(client, "list", kind="agent_log", project_id=pid).json()["items"]
    assert any(e["tool"] == "start_intake" for e in log)


def test_project_kinds_need_a_project(client):
    r = call(client, "list", kind="jobs")
    assert r.status_code == 400 and "project_id" in r.text


def test_cancel_job_stops_a_running_job(client):
    from cortex.api.app import ctx

    pid = call(client, "start_intake", prompt="pick up the cube").json()["project_id"]

    async def forever(job):
        await asyncio.sleep(60)

    async def submit():
        return ctx().jobs.submit("record", pid, forever)

    j = client.portal.call(submit)  # submit on the app's own event loop
    r = call(client, "cancel_job", job_id=j.id)
    assert r.status_code == 200 and "Stopped" in r.json()["summary"], r.text
    again = call(client, "cancel_job", job_id=j.id).json()
    assert "nothing to stop" in again["summary"] or "Stopped" in again["summary"]
    assert call(client, "cancel_job", job_id="job_missing").status_code == 404
