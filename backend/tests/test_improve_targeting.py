"""#366: improve_policy reproduces the weak branch's shifted start and fine-tunes the evaluated policy."""

import asyncio
from types import SimpleNamespace

from cortex.config import Settings
from cortex.context import build_ctx
from cortex.models import (
    BranchResult,
    BranchSpec,
    Checkpoint,
    Condition,
    CreateProjectRequest,
    JobStatus,
    StartAgentCollectionRequest,
)
from cortex.services import agent_collection_service as acs
from cortex.services import agent_orchestrator as orch
from cortex.services import dataset_service, simulation_runner, training_service


def _eval(status, ck_id, offset):
    weak = SimpleNamespace(spec=BranchSpec(name="shifted_object", condition=Condition(), cube_offset=offset),
                           result=BranchResult(n=10, successes=2, success_rate=0.2, ci95=(0.05, 0.5),
                                               mean_reward=0.2, episodes=[]))
    ok = SimpleNamespace(spec=BranchSpec(name="baseline", condition=Condition()),
                         result=BranchResult(n=10, successes=9, success_rate=0.9, ci95=(0.6, 1.0),
                                             mean_reward=0.9, episodes=[]))
    run = SimpleNamespace(id=f"ev_{status}", status=status, policy=SimpleNamespace(kind="checkpoint", checkpoint_id=ck_id))
    return {"run": run, "branches": [ok, weak]}


async def _improve(tmp_path, monkeypatch, intake: bool):
    ctx = await build_ctx(Settings(data_dir=tmp_path, _env_file=None))
    p = await orch.create_project(ctx, CreateProjectRequest(task_prompt="pick up the cube and place it in the bowl"))
    if intake:  # an intake scene: SceneEnv ignores cube_region and uses the offset
        p.intake, p.scene = SimpleNamespace(), {"built": True}  # type: ignore[assignment]
    ck = Checkpoint(project_id=p.id, training_job_id=None, policy_type="smolvla", step=1000, path="ck")
    await ctx.repo.save("checkpoints", ck)

    async def evals(ctx, pid):  # the finished one first, then one still running (must be ignored)
        return [_eval(JobStatus.succeeded, ck.id, (0.04, 0.06)), _eval(JobStatus.running, "ck_other", (0.0, 0.0))]

    seen = {}

    async def start(ctx, project, req, actor="gui", cycles=0):
        seen["req"] = req
        seen["scenarios"] = acs.plan_scenarios(project, req, 20_000)
        return SimpleNamespace(id="ac_1", job_id=None)

    async def dataset(ctx, project, req, actor="gui"):
        return ctx.jobs.submit("dataset", project.id, lambda j: asyncio.sleep(0, result="ds_new"))

    async def propose(ctx, project, ds_id, policy_type="act", steps=None, actor="gui", base_checkpoint_id=None):
        seen["propose"] = (ds_id, policy_type, base_checkpoint_id)
        return SimpleNamespace(id="tj_1")

    async def await_job(ctx, job_id):
        if job_id is None:
            return None
        await asyncio.shield(ctx.jobs.tasks[job_id])
        return ctx.jobs.get(job_id)

    monkeypatch.setattr(orch, "get_project", lambda ctx, pid: asyncio.sleep(0, result=p))
    monkeypatch.setattr(simulation_runner, "list_evaluations", evals)
    monkeypatch.setattr(acs, "start_collection", start)
    monkeypatch.setattr(dataset_service, "create_version", dataset)
    monkeypatch.setattr(training_service, "propose", propose)
    monkeypatch.setattr(orch, "_await_job", await_job)
    out = await orch.t_improve(ctx, orch.AgentArgs.Improve(project_id=p.id, max_new_demos=4), "mcp")
    await asyncio.shield(ctx.jobs.tasks[out["job_id"]])
    return seen, ck


async def test_an_intake_scene_collects_at_the_weak_branchs_offset(tmp_path, monkeypatch):
    seen, ck = await _improve(tmp_path, monkeypatch, intake=True)
    req: StartAgentCollectionRequest = seen["req"]
    assert req.cube_offset == (0.04, 0.06)
    for sc in seen["scenarios"]:  # the offset plus at most the diversity jitter (no cube-task clamp on y)
        assert abs(sc.cube_offset[0] - 0.04) <= 0.013 and abs(sc.cube_offset[1] - 0.06) <= 0.013
    assert seen["propose"] == ("ds_new", "smolvla", ck.id)  # fine-tunes the evaluated policy


async def test_the_cube_task_reaches_the_same_position_from_the_nearest_region(tmp_path, monkeypatch):
    from cortex.sim.so101 import REGION_XY

    seen, _ = await _improve(tmp_path, monkeypatch, intake=False)
    req = seen["req"]
    region = req.conditions[0].cube_region
    want = (REGION_XY["center"][0] + 0.04, REGION_XY["center"][1] + 0.06)
    got = (REGION_XY[region][0] + req.cube_offset[0], REGION_XY[region][1] + req.cube_offset[1])
    assert abs(got[0] - want[0]) < 1e-3 and abs(got[1] - want[1]) < 1e-3
