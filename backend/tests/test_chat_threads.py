import asyncio
import json

from cortex.config import Settings
from cortex.context import build_ctx
from cortex.services import agent, chat_store


def run(coro):
    return asyncio.run(coro)


def test_new_chat_keeps_old_threads_and_each_has_its_own_session(tmp_path):
    async def go():
        ctx = await build_ctx(Settings(data_dir=tmp_path, _env_file=None))
        assert await agent.load_log(ctx, "prj_1") == []  # no chat yet, nothing created
        t1 = await chat_store.current(ctx, "prj_1")
        await agent._append_log(ctx, "prj_1", {"kind": "user", "text": "collect 3 demos with your own agent"}, t1["id"])
        await chat_store.set_session(ctx, "prj_1", t1["id"], "sess-1")
        t2 = await chat_store.new_thread(ctx, "prj_1")
        await agent._append_log(ctx, "prj_1", {"kind": "user", "text": "train ACT on the new dataset"}, t2["id"])
        items = await chat_store.threads(ctx, "prj_1")
        assert [t["title"] for t in items] == ["train ACT on the new dataset", "collect 3 demos with your own agent"]
        assert [t["session_id"] for t in items] == [None, "sess-1"]
        assert (await agent.load_log(ctx, "prj_1"))[0]["text"] == "train ACT on the new dataset"  # latest by default
        old = await agent.load_log(ctx, "prj_1", t1["id"])
        assert [e["text"] for e in old] == ["collect 3 demos with your own agent"]
        # job lines land in the latest thread
        await agent._append_log(ctx, "prj_1", {"kind": "job_done", "status": "succeeded", "text": "Dataset build finished"})
        assert (await agent.load_log(ctx, "prj_1"))[-1]["kind"] == "job_done"
    run(go())


def test_a_project_from_before_threads_keeps_its_chat(tmp_path):
    legacy = tmp_path / "agent_sessions" / "prj_2"
    legacy.mkdir(parents=True)
    (legacy / "chat.jsonl").write_text("\n".join(json.dumps(e) for e in [
        {"kind": "user", "text": "pick up the cube", "at": "2026-10-17T10:00:00+00:00"},
        {"kind": "assistant", "text": "on it", "at": "2026-10-17T10:00:05+00:00"}]) + "\n")
    (tmp_path / "agent_sessions" / "prj_2.json").write_text(json.dumps({"session_id": "old-sess"}))

    async def go():
        ctx = await build_ctx(Settings(data_dir=tmp_path, _env_file=None))
        items = await chat_store.threads(ctx, "prj_2")
        assert len(items) == 1 and items[0]["title"] == "pick up the cube" and items[0]["session_id"] == "old-sess"
        assert [e["kind"] for e in await agent.load_log(ctx, "prj_2")] == ["user", "assistant"]
    run(go())


def test_turn_changes_are_measured_from_the_project():
    before = {"episodes": 10, "dataset_versions": 1, "checkpoints": 0, "training_jobs": 0, "evaluation_runs": 2,
              "agent_collections": 1}
    after = {**before, "episodes": 12, "checkpoints": 1, "agent_collections": 2}
    assert [c["text"] for c in agent._diff(before, after)] == ["+2 demos", "+1 policy", "+1 collection"]
    assert agent._diff(before, before) == []
