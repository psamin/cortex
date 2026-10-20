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
