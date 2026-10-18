"""The default autonomous flow never stalls (#341): a turn a restart cut off resumes on the next start, a job the
restart ended still continues the loop, training approved outside the chat still continues it, a step that keeps
failing retries a bounded number of times and then pauses for a Continue click, and the user's turn outlives the
studio's request. Claude Code is faked; nothing here talks to a live backend or MCP server."""

import asyncio
import contextlib
import json

import pytest

from cortex.config import Settings
from cortex.context import build_ctx
from cortex.models import CreateProjectRequest, JobStatus, ResourceEstimate, TrainingJob
from cortex.services import agent, autonomy, chat_store, project_service
from cortex.services import agent_orchestrator as orch
from cortex.services import claude_code_agent

PROMPT = "teach a robot how to pick and place a cube"


def forget_process_state() -> None:
    """What a backend restart loses: everything autonomy keeps in memory."""
    for d in (autonomy._STATES, autonomy._RUNS, autonomy._LIVE, autonomy._TASKS, autonomy._TURNS, autonomy._LOCKS,
              autonomy._AUTO_TURNS):
        d.clear()
    autonomy._NOTIFIED.clear()
    autonomy._STOPPED.clear()


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    monkeypatch.setattr(autonomy, "WAKE_DELAY_S", 0.0)
    monkeypatch.setattr(autonomy, "RECOVERY_BACKOFF_S", (0.0, 0.0, 0.0))
    forget_process_state()
    yield
    forget_process_state()


async def setup(tmp_path, cluster: bool = True):
    ctx = await build_ctx(Settings(data_dir=tmp_path, cortex_autonomy_cluster=cluster, _env_file=None))
    p = project_service.create_project(CreateProjectRequest(task_prompt=PROMPT))
    await ctx.repo.save("projects", p, status="active")
    return ctx, p


def fake_agent(monkeypatch, fn) -> None:
    async def engine_info(_ctx):
        return {"engine": "claude_code", "label": "Claude Code (your agent)", "note": None}
    monkeypatch.setattr(agent, "engine_info", engine_info)
    monkeypatch.setattr(claude_code_agent, "chat", fn)


async def until(cond, timeout: float = 5.0) -> None:
    for _ in range(int(timeout / 0.02)):
        if cond():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("timed out waiting")


def idle(ctx, pid) -> bool:
    st = autonomy.status(ctx, pid)
    return not st["running"] and not st["resuming"]


async def test_a_turn_a_restart_cut_off_resumes_on_the_next_start(tmp_path, monkeypatch):
    """prj_d36f5dfbb38e: build_simulation returned, the backend restarted, and the chat said "Deciding the next
    step…" forever. Now the next start closes the cut-off turn and the agent picks the workflow up on its own."""
    ctx, p = await setup(tmp_path)
    sent: list[str] = []

    async def fake_claude(ctx_, pid, message, sse, tid=None):
        sent.append(message)
        yield sse("tool_call", {"tool": "build_simulation", "args": {}})
        yield sse("tool_result", {"tool": "build_simulation", "summary": "Scene built; feasible", "ref_ids": []})
        if len(sent) == 1:
            yield sse("status", {"state": "thinking", "text": "Deciding the next step…"})
            await asyncio.Event().wait()  # the backend goes down here
        yield sse("text", {"delta": "Collecting demos next."})
        yield sse("done", {})

    fake_agent(monkeypatch, fake_claude)
    stream = agent.chat(ctx, p.id, PROMPT)
    async for chunk in stream:
        if chunk.startswith("event: status") and "Deciding" in chunk:
            break
    assert autonomy.status(ctx, p.id)["running"] and autonomy.status(ctx, p.id)["status"] == "Deciding the next step…"
    task = autonomy._TURNS[p.id]  # the backend stops: whatever the path, the loop cancels the turn's task
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    await stream.aclose()
    tid = json.loads((tmp_path / "agent_runs.json").read_text())[p.id]["turn"]["thread"]  # also what SIGKILL leaves

    forget_process_state()  # the restart
    ctx2 = await build_ctx(Settings(data_dir=tmp_path, _env_file=None))
    assert await autonomy.resume_after_restart(ctx2) == 1
    await until(lambda: len(sent) == 2 and idle(ctx2, p.id))

    assert sent[1].startswith("[Cortex, autonomous mode] Picking the workflow up again")
    assert "the backend restarted during the last step" in sent[1]
    rows = await chat_store.load(ctx2, p.id, tid)
    kinds = [r["kind"] for r in rows]
    assert kinds.count("user") == 1 and kinds.count("done") == 2
    cut = kinds.index("done")  # the cut-off turn is closed (no step left spinning), then picked up
    assert kinds[cut - 1] == "tool_result"
    assert rows[cut + 1]["kind"] == "notice" and rows[cut + 1]["text"].startswith("Continuing on my own")
    assert rows[-2]["text"] == "Collecting demos next." and rows[-1]["kind"] == "done"
    runs = json.loads((tmp_path / "agent_runs.json").read_text())
    assert "turn" not in runs[p.id] and runs[p.id]["attempts"] == 0  # finished cleanly: retries reset
    assert autonomy.status(ctx2, p.id)["paused"] is None


async def test_a_job_the_restart_ended_still_continues_the_loop(tmp_path, monkeypatch):
    ctx, p = await setup(tmp_path)
    sent: list[str] = []

    async def fake_claude(ctx_, pid, message, sse, tid=None):
        sent.append(message)
        if len(sent) == 1:  # starts a collection and ends its turn, as the system prompt asks
            async def work(job):
                await asyncio.Event().wait()
            job = ctx_.jobs.submit("agent_collect", pid, work)
            yield sse("tool_result", {"tool": "start_agent_collection", "summary": "started", "ref_ids": [job.id]})
        yield sse("done", {})

    fake_agent(monkeypatch, fake_claude)
    async for _ in agent.chat(ctx, p.id, PROMPT):
        pass
    tid = autonomy.armed_thread(ctx, p.id)
    assert tid  # waiting on the collection

    forget_process_state()  # the backend dies with the collection running
    ctx2 = await build_ctx(Settings(data_dir=tmp_path, _env_file=None))
    assert [j.kind for j in ctx2.jobs.interrupted] == ["agent_collect"]
    agent.install_job_listener(ctx2)
    await autonomy.resume_after_restart(ctx2)
    await until(lambda: len(sent) == 2 and idle(ctx2, p.id))

    assert sent[1].startswith("[Cortex, autonomous mode] Background work finished: Demonstration collection failed")
    assert "restarted" in sent[1]
    rows = await chat_store.load(ctx2, p.id, tid)
    assert any(r["kind"] == "job_done" and r["status"] == "failed" for r in rows)
    await ctx.jobs.shutdown()


async def test_training_approved_outside_the_chat_still_continues_the_loop(tmp_path, monkeypatch):
    """Cluster training waits for a click (#221). Approving it on the Training page sends no chat message: the chain
    stays armed, so the training job's end wakes the agent to evaluate."""
    ctx, p = await setup(tmp_path, cluster=False)
    agent.install_job_listener(ctx)
    sent: list[str] = []
    tj = TrainingJob(project_id=p.id, dataset_version_id="ds_1", policy_type="act", backend="futurama_slurm",
                     config={}, rationale="", estimate=ResourceEstimate(gpu="1x L40S", gpu_hours=0.6, est_cost_usd=0.0,
                                                                        paid=False, notes=""))

    async def fake_claude(ctx_, pid, message, sse, tid=None):
        sent.append(message)
        if len(sent) == 1:
            await ctx_.repo.save("training_jobs", tj)
            yield sse("tool_result", {"tool": "train_from_agent_demonstrations", "summary": "proposed",
                                      "ref_ids": [tj.id]})
            yield sse("text", {"delta": "Training needs your approval on the card."})
        else:
            yield sse("text", {"delta": "Evaluating the new policy."})
        yield sse("done", {})

    fake_agent(monkeypatch, fake_claude)
    async for _ in agent.chat(ctx, p.id, PROMPT):
        pass
    tid = autonomy.armed_thread(ctx, p.id)
    assert tid  # armed while the training waits for the click, instead of ending the chain
    rows = await chat_store.load(ctx, p.id, tid)
    assert any(r["kind"] == "notice" and r["tone"] == "needs_you" and "cluster" in r["text"] for r in rows)

    tj.status = JobStatus.queued  # a human approved and launched it on the Training page
    await ctx.repo.save("training_jobs", tj)

    async def train(job):
        await asyncio.sleep(0.05)
        return "ck_1"
    ctx.jobs.submit("train", p.id, train)
    await until(lambda: len(sent) == 2 and idle(ctx, p.id))
    assert sent[1].startswith("[Cortex, autonomous mode] Background work finished: Training finished")
    await ctx.jobs.shutdown()


async def test_a_step_that_keeps_failing_retries_a_few_times_then_pauses_for_continue(tmp_path, monkeypatch):
    ctx, p = await setup(tmp_path)
    sent: list[str] = []
    crash = {"on": True}

    async def fake_claude(ctx_, pid, message, sse, tid=None):
        sent.append(message)
        yield sse("text", {"delta": "Working on it."})
        yield sse("done", {"is_error": crash["on"]})  # Claude Code crashed or timed out mid-turn

    fake_agent(monkeypatch, fake_claude)
    q = ctx.bus.subscribe(p.id)
    async for _ in agent.chat(ctx, p.id, PROMPT):
        pass
    await until(lambda: autonomy.status(ctx, p.id)["paused"] is not None and idle(ctx, p.id))
    await asyncio.sleep(0.1)
    assert len(sent) == 1 + autonomy.MAX_RECOVERIES  # bounded: no endless retry loop
    st = autonomy.status(ctx, p.id)
    assert not st["waiting"] and "times in a row" in st["paused"]["reason"]
    tid = st["paused"]["thread"]
    rows = await chat_store.load(ctx, p.id, tid)
    assert rows[-1]["kind"] == "notice" and rows[-1]["tone"] == "needs_you"
    assert rows[-1]["text"].startswith("Paused:") and "Continue" in rows[-1]["text"]
    published = []
    while not q.empty():
        published.append(q.get_nowait()[1])
    assert any(e.get("kind") == "paused" for e in published)

    crash["on"] = False  # the human presses Continue
    assert await autonomy.resume(ctx, p.id)
    await until(lambda: len(sent) == 2 + autonomy.MAX_RECOVERIES and idle(ctx, p.id))
    assert "Continue was pressed" in sent[-1]
    assert autonomy.status(ctx, p.id)["paused"] is None


async def test_an_automatic_turn_never_hands_over_to_the_rule_planner(tmp_path, monkeypatch):
    """Claude Code unavailable on a wake-up (usage limit, logged out): the built-in planner used to answer, match
    "demo" in the wake-up text and record demos again on every wake-up. Now the step fails, retries, then pauses."""
    ctx, p = await setup(tmp_path)
    ctx.model = None  # no model route either
    calls: list[str] = []

    async def fake_claude(ctx_, pid, message, sse, tid=None):
        calls.append(message)
        raise claude_code_agent.Unavailable("Claude Code is not logged in")
        yield  # an async generator, like the real one

    fake_agent(monkeypatch, fake_claude)
    tid = (await chat_store.current(ctx, p.id))["id"]
    assert await autonomy.resume(ctx, p.id)
    await until(lambda: autonomy.status(ctx, p.id)["paused"] is not None and idle(ctx, p.id))
    assert len(calls) == 1 + autonomy.MAX_RECOVERIES
    rows = await chat_store.load(ctx, p.id, tid)
    assert not any(r["kind"] in ("tool_call", "tool_result") for r in rows)


async def test_the_user_turn_keeps_going_when_the_studio_goes_away(tmp_path, monkeypatch):
    """A reload or another page used to cancel the request and kill Claude Code mid-turn."""
    ctx, p = await setup(tmp_path)
    go_on = asyncio.Event()

    async def fake_claude(ctx_, pid, message, sse, tid=None):
        yield sse("tool_call", {"tool": "start_agent_collection", "args": {}})
        await go_on.wait()
        yield sse("tool_result", {"tool": "start_agent_collection", "summary": "done", "ref_ids": []})
        yield sse("text", {"delta": "Collected."})
        yield sse("done", {})

    fake_agent(monkeypatch, fake_claude)
    q = ctx.bus.subscribe(p.id)
    stream = agent.chat(ctx, p.id, PROMPT)
    async for chunk in stream:
        if chunk.startswith("event: tool_call"):
            break
    await stream.aclose()  # the tab goes away
    assert autonomy.status(ctx, p.id)["running"]
    go_on.set()
    await until(lambda: idle(ctx, p.id))
    tid = (await chat_store.threads(ctx, p.id))[0]["id"]
    rows = await chat_store.load(ctx, p.id, tid)
    assert [r["kind"] for r in rows][-3:] == ["tool_result", "assistant", "done"]
    live = []
    while not q.empty():
        ev, data = q.get_nowait()
        if data.get("kind") == "live":
            live.append((data["event"], data["thread"]))
    assert ("tool_result", tid) in live and live[-1] == ("done", tid)  # the rest streams to whoever reopens it


async def test_stop_ends_the_turn_for_good_and_always_ends_the_live_line(tmp_path, monkeypatch):
    ctx, p = await setup(tmp_path)

    async def fake_claude(ctx_, pid, message, sse, tid=None):
        yield sse("tool_call", {"tool": "start_agent_collection", "args": {}})
        await asyncio.Event().wait()

    fake_agent(monkeypatch, fake_claude)
    q = ctx.bus.subscribe(p.id)
    async for chunk in agent.chat(ctx, p.id, PROMPT):
        if chunk.startswith("event: tool_call"):
            assert autonomy.stop(ctx, p.id) is True  # the user's Stop: not a cut-off, nothing resumes it
    assert "turn" not in json.loads((tmp_path / "agent_runs.json").read_text())[p.id]
    assert idle(ctx, p.id) and autonomy.armed_thread(ctx, p.id) is None
    assert autonomy.stop(ctx, p.id) is False  # nothing left to stop (e.g. a restart had ended it)...
    events = []
    while not q.empty():
        events.append(q.get_nowait()[1])
    assert [e["event"] for e in events if e.get("kind") == "live"][-1] == "done"  # ...the live line still ends


async def test_get_intake_status_without_an_intake_is_not_an_error(tmp_path):
    ctx = await build_ctx(Settings(data_dir=tmp_path, _env_file=None))
    p = await orch.create_project(ctx, CreateProjectRequest(task_prompt=PROMPT))
    out = await orch.call_tool(ctx, "get_intake_status", {"project_id": p.id}, actor="mcp")
    assert out["has_intake"] is False and out["questions"] == [] and out["summary"].startswith("No intake yet")


def test_the_studio_reads_the_agent_state_and_probes_intake_without_errors(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    for k in ("INSFORGE_BASE_URL", "INSFORGE_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    from cortex.config import settings

    settings.cache_clear()
    from cortex.api.app import app

    try:
        with TestClient(app) as c:  # the lifespan runs resume_after_restart on an empty data dir
            pid = c.post("/api/projects", json={"task_prompt": PROMPT}).json()["id"]
            r = c.post("/api/tools/get_intake_status", json={"project_id": pid}, headers={"X-Cortex-Actor": "mcp"})
            assert r.status_code == 200 and r.json()["has_intake"] is False
            st = c.get(f"/api/projects/{pid}/agent/state").json()
            assert st == {"running": False, "thread": None, "status": None, "resuming": False, "waiting": False,
                          "paused": None}
    finally:
        settings.cache_clear()


async def test_a_hung_collection_attempt_times_out_instead_of_hanging_the_job(tmp_path, monkeypatch):
    from cortex.models import AgentCollection, Scenario, StartAgentCollectionRequest
    from cortex.services import claude_code_collector as ccc
    from cortex.services import robot_session_service as rs

    ctx, p = await setup(tmp_path)
    monkeypatch.setattr(ccc, "ATTEMPT_IDLE_S", 0.2)
    monkeypatch.setitem(rs.SESSIONS, "rs_hung", object())
    finished: list[str] = []

    async def start(*a, **k):
        return {"session_id": "rs_hung", "attempt_id": "aa_none", "task": None}

    async def finish(ctx_, sid, believes_success, summary, actor="mcp"):
        finished.append(summary)
        rs.SESSIONS.pop(sid, None)

    class Hung:
        pid, returncode, killed = 0, None, False

        def __init__(self):
            async def lines():
                await asyncio.Event().wait()
                yield b""
            self.stdout = lines()

        def kill(self):
            self.killed, self.returncode = True, -9

        async def wait(self):
            return self.returncode

    proc = Hung()

    async def fake_exec(*a, **k):
        return proc

    monkeypatch.setattr(rs, "start", start)
    monkeypatch.setattr(rs, "finish", finish)
    monkeypatch.setattr(ccc.asyncio, "create_subprocess_exec", fake_exec)
    col = AgentCollection(project_id=p.id, collector="agent", request=StartAgentCollectionRequest(num_demos=1))
    out = await asyncio.wait_for(ccc.run_attempt(ctx, p, col, Scenario(), 1, 0), 5)
    assert proc.killed and out["finished_by_agent"] is False and "timed out" in finished[0]
