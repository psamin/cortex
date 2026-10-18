"""#192: no studio engine can hang a turn. Claude Code and model failures fall back to the next engine with a plain
notice, and the SSE stream always ends with `done`."""

import asyncio
import json

import httpx

from cortex.config import Settings
from cortex.context import ModelRoute, build_ctx
from cortex.models import CreateProjectRequest
from cortex.services import agent, chat_store
from cortex.services import agent_orchestrator as orch
from cortex.services import claude_code_agent as cc


async def _setup(tmp_path, **settings):
    ctx = await build_ctx(Settings(data_dir=tmp_path, _env_file=None, **settings))
    p = await orch.create_project(ctx, CreateProjectRequest(task_prompt="pick up the cube"))
    return ctx, p


async def _turn(ctx, pid, message="hello"):
    async def collect():
        out = []
        async for chunk in agent.chat(ctx, pid, message):
            ev, _, data = chunk.partition("\ndata: ")
            out.append((ev.removeprefix("event: "), json.loads(data)))
        return out
    return await asyncio.wait_for(collect(), 20)  # the test itself must not hang


def _claude_code_is_up(monkeypatch):
    async def up():
        return True
    monkeypatch.setattr(cc, "available", lambda: None)
    monkeypatch.setattr(cc, "mcp_up", up)


class FakeClaude:
    """A `claude -p` process: streams the given events, or nothing at all (hung) when `events` is None."""

    def __init__(self, events=None):
        self.pid = 0
        self.returncode = None
        self.stderr = None
        self.killed = False
        self._events = events

        async def lines():
            if self._events is None:
                await asyncio.Event().wait()
            for ev in self._events:
                yield (json.dumps(ev) + "\n").encode()
        self.stdout = lines()

    def kill(self):
        self.killed, self.returncode = True, -9

    terminate = kill

    async def wait(self):
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


def _fake_exec(monkeypatch, proc):
    async def fake(*a, **k):
        return proc
    monkeypatch.setattr(cc.asyncio, "create_subprocess_exec", fake)


def test_a_hung_claude_code_is_stopped_and_the_built_in_planner_answers(tmp_path, monkeypatch):
    _claude_code_is_up(monkeypatch)
    proc = FakeClaude(events=None)
    _fake_exec(monkeypatch, proc)
    monkeypatch.setattr(cc, "START_TIMEOUT_S", 0.2)

    async def go():
        ctx, p = await _setup(tmp_path)
        return await _turn(ctx, p.id)

    events = asyncio.run(go())
    assert proc.killed
    notices = [d["text"] for ev, d in events if ev == "notice"]
    assert any("Claude Code did not start" in n and "built-in planner" in n for n in notices), notices
    assert ("tool_call", {"tool": "get_next_actions", "args": {}}) in events  # the fallback answered
    assert events[-1][0] == "done"


def test_claude_code_that_is_not_logged_in_falls_back_with_a_plain_reason(tmp_path, monkeypatch):
    _claude_code_is_up(monkeypatch)
    _fake_exec(monkeypatch, FakeClaude(events=[
        {"type": "system", "subtype": "init", "session_id": "s1", "model": "claude"},
        {"type": "assistant", "message": {"model": "<synthetic>", "content": [
            {"type": "text", "text": "Not logged in · Please run /login"}]}},
        {"type": "result", "subtype": "success", "is_error": True, "num_turns": 1,
         "result": "Not logged in · Please run /login", "session_id": "s1"},
    ]))

    async def go():
        ctx, p = await _setup(tmp_path)
        return await _turn(ctx, p.id)

    events = asyncio.run(go())
    texts = "".join(d["delta"] for ev, d in events if ev == "text")
    assert "Not logged in" not in texts  # Claude Code's own error line is not shown as the agent's answer
    notices = [d["text"] for ev, d in events if ev == "notice"]
    assert any("not logged in" in n and "built-in planner" in n for n in notices), notices
    assert events[-1][0] == "done"


def test_a_hung_model_times_out_and_the_built_in_planner_answers(tmp_path, monkeypatch):
    async def hang(request):
        await asyncio.sleep(30)
        return httpx.Response(200, json={})

    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(hang), **kw))
    monkeypatch.setattr(agent, "MODEL_TIMEOUT_S", 0.2)

    async def go():
        ctx, p = await _setup(tmp_path, agent_engine="model")
        ctx.model = ModelRoute(provider="selfhosted", label="self-hosted test model", base_url="http://model.test/v1",
                               api_key="none", planner_model="m", vlm_model="m", agent_model="m")
        return await _turn(ctx, p.id)

    events = asyncio.run(go())
    notices = [d["text"] for ev, d in events if ev == "notice"]
    assert any("did not answer" in n and "built-in planner" in n for n in notices), notices
    assert ("tool_call", {"tool": "get_next_actions", "args": {}}) in events
    assert events[-1][0] == "done"


def test_the_stream_ends_with_done_when_a_turn_fails(tmp_path, monkeypatch):
    async def boom(*a, **k):
        raise RuntimeError("planner exploded")
        yield  # an async generator, like the real one

    monkeypatch.setattr(agent, "_rules", boom)

    async def go():
        ctx, p = await _setup(tmp_path, agent_engine="rules")
        events = await _turn(ctx, p.id)
        t = await chat_store.current(ctx, p.id)
        return events, await chat_store.load(ctx, p.id, t["id"])

    events, log = asyncio.run(go())
    assert events[-1] == ("done", {"is_error": True})
    assert "planner exploded" in "".join(d["delta"] for ev, d in events if ev == "text")
    assert log[-1]["kind"] == "done"


def test_the_engine_says_what_is_missing_in_plain_words(tmp_path, monkeypatch):
    monkeypatch.setattr(cc, "available", lambda: "Claude Code not found")

    async def go():
        ctx, p = await _setup(tmp_path)
        info = await agent.engine_info(ctx)
        events = await _turn(ctx, p.id)
        events2 = await _turn(ctx, p.id)
        return info, events, events2

    info, events, events2 = asyncio.run(go())
    assert info["engine"] == "rules" and info["label"] == "built-in planner"
    assert info["note"] == "Claude Code not found and no model key is set: using the built-in planner"
    assert ("notice", {"kind": "notice", "tone": "engine", "text": info["note"], "ref_ids": []}) in events
    assert not [d for ev, d in events2 if ev == "notice"]  # said once per chat, not on every turn
    reply = "".join(d["delta"] for ev, d in events if ev == "text")
    assert "configure a model endpoint" not in reply
