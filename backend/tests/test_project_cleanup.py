"""Home lists projects with history; empty ones are deleted after a grace period, never one with history."""

import asyncio
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from cortex.config import Settings
from cortex.context import build_ctx
from cortex.models import CreateProjectRequest, now
from cortex.services import agent, chat_store, project_cleanup, project_service


def run(coro):
    return asyncio.run(coro)


async def make(ctx, name: str, age_min: float):
    p = project_service.create_project(CreateProjectRequest(task_prompt=f"pick up the cube ({name})", name=name))
    p.created_at = now() - timedelta(minutes=age_min)
    await ctx.repo.save("projects", p, status="active")
    return p


async def status(ctx, pid: str):
    return (await ctx.store.get("projects", pid))["status"]


@pytest.fixture(autouse=True)
def fresh_cache():
    project_cleanup._KNOWN.clear()
    yield
    project_cleanup._KNOWN.clear()


@pytest.mark.parametrize("table", project_cleanup.HISTORY_TABLES)
def test_a_project_with_history_is_never_deleted(tmp_path, table):
    async def go():
        ctx = await build_ctx(Settings(data_dir=tmp_path, _env_file=None))
        old = await make(ctx, "Old with history", age_min=600)
        await ctx.store.put(table, {"id": f"x_{table}", "project_id": old.id, "status": "succeeded",
                                    "data": {"id": f"x_{table}", "project_id": old.id}, "created_at": now().isoformat()})
        shown = await project_cleanup.with_history(ctx, await project_cleanup.listed(ctx))
        assert [p.id for p in shown] == [old.id]
        assert await status(ctx, old.id) == "active"
    run(go())


def test_a_chat_or_a_job_counts_as_history(tmp_path):
    async def go():
        ctx = await build_ctx(Settings(data_dir=tmp_path, _env_file=None))
        chatted = await make(ctx, "Chatted", age_min=60)
        t = await chat_store.current(ctx, chatted.id)
        await chat_store.append(ctx, chatted.id, t["id"], {"kind": "user", "text": "collect 6 demos"})
        working = await make(ctx, "Job running", age_min=60)

        async def slow(job):
            await asyncio.sleep(0.5)

        ctx.jobs.submit("record", working.id, slow)
        shown = await project_cleanup.with_history(ctx, await project_cleanup.listed(ctx))
        assert {p.id for p in shown} == {chatted.id, working.id}
        assert await status(ctx, chatted.id) == await status(ctx, working.id) == "active"
        await ctx.jobs.shutdown()
    run(go())


def test_empty_projects_are_hidden_and_deleted_only_after_the_grace_period(tmp_path):
    async def go():
        ctx = await build_ctx(Settings(data_dir=tmp_path, _env_file=None))
        stale = await make(ctx, "Pick cube", age_min=11)
        fresh = await make(ctx, "Being set up", age_min=2)
        await chat_store.new_thread(ctx, stale.id)  # an empty "New chat" is not history
        assert await project_cleanup.with_history(ctx, await project_cleanup.listed(ctx)) == []
        assert await status(ctx, stale.id) == "deleted"
        assert await status(ctx, fresh.id) == "active"  # too young to delete, just not on Home
        assert [p.id for p in await project_cleanup.listed(ctx)] == [fresh.id]
        # the deleted project's id still resolves, and it comes back once it gets history
        from cortex.services.agent_orchestrator import get_project
        assert (await get_project(ctx, stale.id)).name == "Pick cube"
        await project_cleanup.restore(ctx, stale.id)
        assert await status(ctx, stale.id) == "active"
    run(go())


def test_chatting_in_a_deleted_project_restores_it(tmp_path):
    async def go():
        ctx = await build_ctx(Settings(data_dir=tmp_path, agent_engine="rules", _env_file=None))
        p = await make(ctx, "Came back", age_min=30)
        await project_cleanup.with_history(ctx, [p])
        assert await status(ctx, p.id) == "deleted"
        async for _ in agent.chat(ctx, p.id, "status"):  # the rule-based planner: no model, no Claude Code
            pass
        assert await status(ctx, p.id) == "active"
        assert [x.id for x in await project_cleanup.with_history(ctx, await project_cleanup.listed(ctx))] == [p.id]
    run(go())


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    for k in ("INSFORGE_BASE_URL", "INSFORGE_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    from cortex.config import settings

    settings.cache_clear()
    from cortex.api.app import app

    with TestClient(app) as c:
        yield c
    settings.cache_clear()


def test_api_lists_all_live_projects_and_home_gets_only_those_with_history(client):
    from cortex.api import app as app_mod

    ctx = app_mod.CTX
    empty = client.portal.call(make, ctx, "Empty", 1)
    used = client.portal.call(make, ctx, "Used", 1)
    t = client.portal.call(chat_store.current, ctx, used.id)
    client.portal.call(chat_store.append, ctx, used.id, t["id"], {"kind": "user", "text": "train a policy"})
    assert [p["id"] for p in client.get("/api/projects").json()] == [used.id, empty.id]
    assert [p["id"] for p in client.get("/api/projects", params={"history": "true"}).json()] == [used.id]


def test_an_unreadable_history_check_keeps_the_project(tmp_path, monkeypatch):
    async def go():
        ctx = await build_ctx(Settings(data_dir=tmp_path, _env_file=None))
        old = await make(ctx, "Unreadable", age_min=600)
        empty = await make(ctx, "Empty index", age_min=600)
        ctx.local(chat_store._key(empty.id, "index.json")).parent.mkdir(parents=True, exist_ok=True)
        ctx.local(chat_store._key(empty.id, "index.json")).write_text("")  # an interrupted write
        assert await chat_store.threads(ctx, empty.id) == []

        real = project_cleanup.has_history

        async def broken(c, pid):
            if pid == old.id:
                raise ValueError("Expecting value: line 1 column 1 (char 0)")
            return await real(c, pid)

        monkeypatch.setattr(project_cleanup, "has_history", broken)
        shown = await project_cleanup.with_history(ctx, await project_cleanup.listed(ctx))
        assert [p.id for p in shown] == [old.id] and await status(ctx, old.id) == "active"
    run(go())
