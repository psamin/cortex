"""#356: a robot session whose worker misses a reply is closed, never answered with the previous call's reply."""

import asyncio
import sys

import pytest

from cortex.services import robot_session_service as rss

# A worker that answers each request with its own number; the first answer comes late.
WORKER = r"""
import json, sys, time
n = 0
for line in sys.stdin:
    n += 1
    if n == 1:
        time.sleep(1.0)
    print(json.dumps({"ok": True, "n": n}), flush=True)
"""


async def _session(monkeypatch):
    monkeypatch.setattr(rss, "RPC_TIMEOUT_S", 0.3)
    proc = await asyncio.create_subprocess_exec(sys.executable, "-c", WORKER, stdin=asyncio.subprocess.PIPE,
                                                stdout=asyncio.subprocess.PIPE)
    s = rss.RobotSession(id="rs_test", project_id="prj_1", collection=None, attempt=None, proc=proc)  # type: ignore[arg-type]
    rss.SESSIONS[s.id] = s
    return s, proc


async def test_a_late_reply_closes_the_session_instead_of_answering_the_next_call(monkeypatch):
    s, proc = await _session(monkeypatch)
    with pytest.raises(rss.SessionBroken, match="start a new episode"):
        await rss._rpc(s, {"op": "call"})
    assert "rs_test" not in rss.SESSIONS
    assert await asyncio.wait_for(proc.wait(), 5) is not None  # the worker was killed, not left running
    with pytest.raises(rss.SessionBroken):
        await rss._rpc(s, {"op": "call"})  # never the late reply to the first request


async def test_replies_in_step_are_returned(monkeypatch):
    monkeypatch.setattr(rss, "RPC_TIMEOUT_S", 5)
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-c", WORKER.replace("time.sleep(1.0)", "pass"),
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE)
    s = rss.RobotSession(id="rs_ok", project_id="prj_1", collection=None, attempt=None, proc=proc)  # type: ignore[arg-type]
    assert (await rss._rpc(s, {"op": "a"}))["n"] == 1
    assert (await rss._rpc(s, {"op": "b"}))["n"] == 2
    proc.kill()
    await proc.wait()


def test_a_broken_session_reaches_the_agent_as_its_own_message():
    """It must come back as a 400 with the text ("start a new episode"), not a generic 500."""
    from fastapi.testclient import TestClient

    from cortex.api.app import app

    @app.get("/__test/broken")
    async def broken():
        raise rss.SessionBroken("the robot session didn't answer 'call' within 180 s, so it was closed; "
                                "start a new episode")
    try:
        r = TestClient(app, raise_server_exceptions=False).get("/__test/broken")
    finally:
        app.router.routes.pop()
    assert r.status_code == 400 and "start a new episode" in r.json()["detail"]


# ---------------------------------------- #362: a crashed worker is one failed attempt, with its reason

DIES = r"""
import sys
sys.stdin.readline()
print("GLFWError: failed to create the OpenGL context", file=sys.stderr, flush=True)
sys.exit(3)
"""


async def test_a_crashed_worker_says_why_and_its_attempt_is_recorded_failed(tmp_path, monkeypatch):
    from cortex.config import Settings
    from cortex.context import build_ctx
    from cortex.models import AgentAttempt, AgentCollection, CreateProjectRequest, Scenario, StartAgentCollectionRequest
    from cortex.services import agent_collection_service as acs
    from cortex.services import agent_orchestrator as orch

    ctx = await build_ctx(Settings(data_dir=tmp_path, _env_file=None))
    p = await orch.create_project(ctx, CreateProjectRequest(task_prompt="pick up the cube and place it in the bowl"))
    col = AgentCollection(project_id=p.id, collector="agent", status="running", mode="camera",
                          request=StartAgentCollectionRequest(num_demos=1, collector="agent"),
                          totals=acs._empty_totals(0))
    await ctx.repo.save("agent_collections", col)
    att = AgentAttempt(project_id=p.id, collection_id=col.id, index=0, collector="agent", seed=1,
                       scenario=Scenario(), status="failed", mode="camera")
    proc = await asyncio.create_subprocess_exec(sys.executable, "-c", DIES, stdin=asyncio.subprocess.PIPE,
                                                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    s = rss.RobotSession(id="rs_dies", project_id=p.id, collection=col, attempt=att, proc=proc)
    s.err_task = asyncio.create_task(rss._drain_err(s))
    rss.SESSIONS[s.id] = s
    with pytest.raises(rss.SessionBroken, match="OpenGL context"):
        await rss._rpc(s, {"op": "call"})
    out = await rss.record_broken(ctx, "rs_dies")
    assert out and out["attempt_id"] == att.id
    saved = await ctx.repo.get("agent_attempts", att.id, AgentAttempt)
    assert saved.status == "failed" and saved.stop_reason == "session worker crashed" and "OpenGL" in saved.error
    assert await rss.record_broken(ctx, "rs_dies") is None  # recorded once


async def test_a_stale_scene_spawns_no_worker(tmp_path, monkeypatch):
    from cortex.config import Settings
    from cortex.context import build_ctx
    from cortex.models import CreateProjectRequest
    from cortex.services import agent_orchestrator as orch
    from cortex.services import intake_service

    ctx = await build_ctx(Settings(data_dir=tmp_path, _env_file=None))
    p = await orch.create_project(ctx, CreateProjectRequest(task_prompt="pick up the cube and place it in the bowl"))

    def stale(ctx, project):
        raise ValueError("the sim scene is out of date with the intake; call build_simulation first")

    async def no_spawn(*a, **kw):
        raise AssertionError("spawned a worker for a stale scene")
    monkeypatch.setattr(intake_service, "scene_payload", stale)
    monkeypatch.setattr(rss.asyncio, "create_subprocess_exec", no_spawn)
    with pytest.raises(ValueError, match="build_simulation"):
        await rss.start(ctx, p)
