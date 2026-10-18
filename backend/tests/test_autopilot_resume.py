import asyncio

import pytest

from cortex.config import Settings
from cortex.context import build_ctx
from cortex.models import (
    AgentCollection, AutopilotPolicy, AutopilotRun, CreateProjectRequest, StartAgentCollectionRequest,
)
from cortex.services import agent_orchestrator as orch
from cortex.services import autopilot
from cortex.services.agent_collection_service import _empty_totals
from cortex.services.jobs import Job, JobManager


@pytest.mark.parametrize("persisted", [False, True], ids=["job_never_persisted", "job_restored_as_interrupted"])
async def test_autopilot_resumes_an_interrupted_collection(tmp_path, persisted):
    ctx = await build_ctx(Settings(data_dir=tmp_path, _env_file=None))
    p = await orch.create_project(ctx, CreateProjectRequest(
        task_prompt="pick up the cube and place it in the bowl", demo_budget=3,
        autopilot=AutopilotPolicy(collector="oracle", max_iterations=1, demos_per_round=2)))
    job_id = "job_from_dead_process"
    if persisted:  # the job was on disk as running; a new JobManager (the restart) restores it as failed
        job = Job(kind="agent_collection", project_id=p.id, status="running")
        ctx.jobs.jobs[job.id] = job
        ctx.jobs._save()
        ctx.jobs = JobManager(ctx.bus, persist=tmp_path / "jobs.json")
        job_id = job.id
        assert ctx.jobs.get(job_id).status == "failed"
    # State as a crashed backend left it: a collection 'running' under a job that will never finish.
    col = AgentCollection(project_id=p.id, collector="oracle", status="running", job_id=job_id,
                          request=StartAgentCollectionRequest(num_demos=2, collector="oracle"), totals=_empty_totals(0))
    await ctx.repo.save("agent_collections", col)
    run = AutopilotRun(project_id=p.id, collector="oracle", state="AGENT_COLLECTING", iteration=1,
                       context={"collection_id": col.id, "collection_job_id": job_id})
    await ctx.repo.save("autopilot_runs", run)

    assert await autopilot.resume_all(ctx) == 1
    for _ in range(120):  # it should mark the dead collection, validate (0 demos), and start a top-up collection
        cur = await autopilot.current(ctx, p.id)
        if cur.context.get("collection_id") not in (None, col.id):
            break
        await asyncio.sleep(0.5)
    cur = await autopilot.current(ctx, p.id)
    old = await ctx.repo.get("agent_collections", col.id, AgentCollection)
    assert old.status == "failed" and "restart" in (old.error or "")
    assert cur.state == "AGENT_COLLECTING" and cur.context["collection_id"] != col.id
    assert any("Resumed after backend restart" in s.message for s in cur.steps)
    assert any("collecting more" in s.message for s in cur.steps)
    stopped = await autopilot.stop(ctx, p.id)
    assert stopped.status == "stopped"


@pytest.mark.parametrize("state", ["POLICY_EVALUATED", "NEXT_EXPERIMENT_SELECTED"])
async def test_autopilot_stops_cleanly_when_every_evaluation_branch_failed(tmp_path, state):
    from cortex.models import EvaluationRun, JobStatus, PolicyRef, SimulationBranch
    from cortex.services.simulation_runner import DEFAULT_BRANCHES

    ctx = await build_ctx(Settings(data_dir=tmp_path, _env_file=None))
    p = await orch.create_project(ctx, CreateProjectRequest(
        task_prompt="pick up the cube and place it in the bowl",
        autopilot=AutopilotPolicy(collector="oracle", max_iterations=3)))
    ev = EvaluationRun(project_id=p.id, policy=PolicyRef(kind="checkpoint", checkpoint_id="ck_1"),
                       policy_label="act checkpoint ck_1", executor="local", name="it1", status=JobStatus.failed)
    branches = [SimulationBranch(evaluation_id=ev.id, project_id=p.id, spec=s, executor="local", seeds=[1000],
                                 status=JobStatus.failed, error="RuntimeError: checkpoint files missing")
                for s in DEFAULT_BRANCHES]
    ev.branch_ids = [b.id for b in branches]
    await ctx.repo.save("evaluation_runs", ev)
    for b in branches:
        await ctx.repo.save("simulation_branches", b, project_id=p.id)
    run = AutopilotRun(project_id=p.id, collector="oracle", state=state, iteration=1,
                       context={"evaluation_id": ev.id, "checkpoint_id": "ck_1"})
    await ctx.repo.save("autopilot_runs", run)

    assert await autopilot.resume_all(ctx) == 1
    for _ in range(100):
        cur = await autopilot.current(ctx, p.id)
        if cur.status != "running":
            break
        await asyncio.sleep(0.1)
    assert cur.status == "failed"
    assert "produced no results" in cur.error and "checkpoint files missing" in cur.error
    assert "ValueError" not in cur.error
