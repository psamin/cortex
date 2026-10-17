import asyncio

from cortex.config import Settings
from cortex.context import build_ctx
from cortex.models import (
    AgentCollection, AutopilotPolicy, AutopilotRun, CreateProjectRequest, StartAgentCollectionRequest,
)
from cortex.services import agent_orchestrator as orch
from cortex.services import autopilot
from cortex.services.agent_collection_service import _empty_totals


async def test_autopilot_resumes_an_interrupted_collection(tmp_path):
    ctx = await build_ctx(Settings(data_dir=tmp_path, _env_file=None))
    p = await orch.create_project(ctx, CreateProjectRequest(
        task_prompt="pick up the cube and place it in the bowl", demo_budget=3,
        autopilot=AutopilotPolicy(collector="oracle", max_iterations=1, demos_per_round=2)))
    # State as a crashed backend left it: a collection 'running' under a job this process never saw.
    col = AgentCollection(project_id=p.id, collector="oracle", status="running", job_id="job_from_dead_process",
                          request=StartAgentCollectionRequest(num_demos=2, collector="oracle"), totals=_empty_totals(0))
    await ctx.repo.save("agent_collections", col)
    run = AutopilotRun(project_id=p.id, collector="oracle", state="AGENT_COLLECTING", iteration=1,
                       context={"collection_id": col.id, "collection_job_id": "job_from_dead_process"})
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
