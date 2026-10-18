"""Autonomous mode (#170): the studio agent keeps going on its own; training inside the standing approval
starts without a click; spend outside it still waits for a human; manual mode keeps today's behaviour."""

import asyncio

import pytest

from cortex.config import Settings
from cortex.context import build_ctx
from cortex.models import CreateProjectRequest, JobStatus, ResourceEstimate, TrainingJob
from cortex.services import agent, autonomy, chat_store, project_service, training_service


def run(coro):
    return asyncio.run(coro)


async def setup(tmp_path, mode: str = "autonomous"):
    ctx = await build_ctx(Settings(data_dir=tmp_path, _env_file=None))
    p = project_service.create_project(CreateProjectRequest(task_prompt="pick up the cube"))
    p.autopilot.autonomy = mode  # type: ignore[assignment]
    await ctx.repo.save("projects", p, status="active")
    return ctx, p


def proposed(p, gpu_hours=0.6, paid=False, backend="futurama_slurm") -> TrainingJob:
    return TrainingJob(project_id=p.id, dataset_version_id="ds_1", policy_type="act", backend=backend, config={}, rationale="",
                       estimate=ResourceEstimate(gpu="1x L40S", gpu_hours=gpu_hours, est_cost_usd=2.0 if paid else 0.0,
                                                 paid=paid, notes=""))


@pytest.fixture(autouse=True)
def no_cluster(monkeypatch):
    """launch() would ssh to Futurama: record the launch instead."""
    async def launch(ctx, tj_id, actor="gui"):
        tj = await ctx.repo.get("training_jobs", tj_id, TrainingJob)
        assert tj.status == JobStatus.approved, "launched without approval"
        tj.status = JobStatus.queued
        await ctx.repo.save("training_jobs", tj)
        return tj
    monkeypatch.setattr(training_service, "launch", launch)
    monkeypatch.setattr(autonomy, "WAKE_DELAY_S", 0.0)
    autonomy._NOTIFIED.clear()
    autonomy._AUTO_TURNS.clear()


def test_autonomous_is_the_default_and_manual_is_a_choice():
    p = project_service.create_project(CreateProjectRequest(task_prompt="pick up the cube"))
    assert p.autopilot.autonomy == "autonomous" and autonomy.is_autonomous(p)
    assert "AUTONOMOUS MODE" in autonomy.mode_rules(p) and "real-robot motion need the human" in autonomy.mode_rules(p)
    p.autopilot.autonomy = "manual"
    assert "MANUAL MODE" in autonomy.mode_rules(p) and "approval" in autonomy.mode_rules(p)


def test_standing_approval_covers_only_free_training_under_the_cap():
    p = project_service.create_project(CreateProjectRequest(task_prompt="pick up the cube"))
    assert autonomy.standing_approval(p, proposed(p))[0]
    assert not autonomy.standing_approval(p, proposed(p, gpu_hours=5))[0]  # over max_gpu_hours_per_job
    assert not autonomy.standing_approval(p, proposed(p, paid=True, backend="hf_jobs"))[0]  # paid, not allowed
    assert not autonomy.standing_approval(p, proposed(p, backend="manual"))[0]  # nothing to run it on
    p.autopilot.auto_approve_free_training = False
    assert not autonomy.standing_approval(p, proposed(p))[0]


def test_autonomous_mode_approves_and_starts_covered_training(tmp_path):
    async def go():
        ctx, p = await setup(tmp_path)
        tj = proposed(p)
        await ctx.repo.save("training_jobs", tj)
        row = await autonomy.approve_if_covered(ctx, p.id, tj.id)
        cur = await ctx.repo.get("training_jobs", tj.id, TrainingJob)
        assert cur.status == JobStatus.queued and cur.approved_by == "standing approval (autonomous mode)"
        assert row["kind"] == "notice" and row["tone"] == "auto" and tj.id in row["ref_ids"]
        assert await autonomy.approve_if_covered(ctx, p.id, tj.id) is None  # already running: nothing to do
    run(go())


def test_spend_outside_the_standing_approval_waits_for_a_human(tmp_path):
    async def go():
        ctx, p = await setup(tmp_path)
        tj = proposed(p, gpu_hours=6)
        await ctx.repo.save("training_jobs", tj)
        row = await autonomy.approve_if_covered(ctx, p.id, tj.id)
        assert row["tone"] == "needs_you" and "GPU-hour cap" in row["text"]
        assert (await ctx.repo.get("training_jobs", tj.id, TrainingJob)).status == JobStatus.proposed
        assert await autonomy.approve_if_covered(ctx, p.id, tj.id) is None  # said once, not on every mention
    run(go())


def test_manual_mode_never_approves(tmp_path):
    async def go():
        ctx, p = await setup(tmp_path, mode="manual")
        tj = proposed(p)
        await ctx.repo.save("training_jobs", tj)
        assert await autonomy.approve_if_covered(ctx, p.id, tj.id) is None
        assert (await ctx.repo.get("training_jobs", tj.id, TrainingJob)).status == JobStatus.proposed
    run(go())


async def chain(tmp_path, monkeypatch, mode: str):
    """A user turn starts a background job and ends; the job then finishes. Returns what the agent was sent,
    the thread's rows and the live events published for the studio."""
    ctx, p = await setup(tmp_path, mode)
    agent.install_job_listener(ctx)
    release = asyncio.Event()
    sent: list[str] = []

    async def fake_engine(_ctx):
        return "claude_code"

    async def fake_claude(ctx_, pid, message, sse, tid=None):
        sent.append(message)
        if len(sent) == 1:  # the user's turn starts a collection and ends, as the system prompt asks
            async def work(job):
                await release.wait()
                return None
            job = ctx_.jobs.submit("record", pid, work)
            yield sse("tool_call", {"tool": "record_demonstrations", "args": {}})
            yield sse("tool_result", {"tool": "record_demonstrations", "summary": "started", "ref_ids": [job.id]})
        yield sse("text", {"delta": "Recording is running." if len(sent) == 1 else "Next: building the dataset."})
        yield sse("done", {})

    from cortex.services import claude_code_agent
    monkeypatch.setattr(agent, "engine", fake_engine)
    monkeypatch.setattr(claude_code_agent, "chat", fake_claude)
    q = ctx.bus.subscribe(p.id)
    async for _ in agent.chat(ctx, p.id, "teach it to pick up the cube"):
        pass
    tid = (await chat_store.threads(ctx, p.id))[0]["id"]
    assert autonomy.armed_thread(ctx, p.id) == (tid if mode == "autonomous" else None)
    release.set()
    for _ in range(100):  # the job finishes, its line lands, and (autonomous) the agent is woken
        await asyncio.sleep(0.02)
        if len(sent) == 2 and autonomy.armed_thread(ctx, p.id) is None:
            break
    await asyncio.sleep(0.1)
    live = []
    while not q.empty():
        ev, data = q.get_nowait()
        if ev == "chat" and data.get("kind") == "live":
            live.append(data["event"])
    rows = await chat_store.load(ctx, p.id, tid)
    await ctx.jobs.shutdown()
    return sent, rows, live


def test_a_finished_job_wakes_the_agent_in_autonomous_mode(tmp_path, monkeypatch):
    sent, rows, live = run(chain(tmp_path, monkeypatch, "autonomous"))
    assert len(sent) == 2 and sent[1].startswith("[Cortex, autonomous mode]")
    assert "Recording demonstrations finished" in sent[1]
    kinds = [r["kind"] for r in rows]
    assert kinds.count("user") == 1  # the wake-up is a notice, not a message from the user
    assert any(r["kind"] == "notice" and r["text"].startswith("Continuing on my own") for r in rows)
    assert rows[-2]["text"] == "Next: building the dataset."
    assert "text" in live and live[-1] == "done"  # the studio sees the automatic turn live


def test_manual_mode_waits_for_the_user(tmp_path, monkeypatch):
    sent, rows, live = run(chain(tmp_path, monkeypatch, "manual"))
    assert len(sent) == 1 and live == []
    assert rows[-1]["kind"] == "job_done"


def test_stop_ends_the_chain(tmp_path):
    async def go():
        ctx, p = await setup(tmp_path)
        autonomy._state(ctx)[p.id] = {"thread": "th_1", "pending": []}
        assert autonomy.armed_thread(ctx, p.id) == "th_1"
        autonomy.stop(ctx, p.id)
        assert autonomy.armed_thread(ctx, p.id) is None
    run(go())
