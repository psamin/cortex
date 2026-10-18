"""#310: the control agent saves a trained checkpoint as a named Physical Skill and deploys it in the deployment
simulator (the GPU server's closed-loop MuJoCo), through cortex.list/save/deploy_physical_skill."""

import json

import pytest

from cortex.config import Settings
from cortex.context import build_ctx
from cortex.errors import NotFound
from cortex.models import Checkpoint, CreateProjectRequest, JobStatus, ResourceEstimate, TrainingJob
from cortex.services import live_policy_service as lp
from cortex.services import project_service, sim_import_service, skills_service
from cortex.services.agent_orchestrator import _skill_brief, call_tool

SIM_SRC = "futurama:~/cortex-sim/unscrew"
LIVE = {"available": True, "phase": "ready", "step": "ready", "gpu": "NVIDIA L40S", "policies": {}, "projects": {},
        "sim": {"unscrew": {"variant": "act", "step": 30000, "success_rate": 0.8, "eval_episodes": 50,
                            "eval_seeds": "100000..100049", "instruction": "unscrew the bolt"}}}
DOWN = {"available": False, "phase": "stopped", "step": None, "reason": "Not connected to a GPU server."}


def _write(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data))


def _mirror(root):
    """<job_dir>/unscrew as sim_task.sbatch leaves it: ACT step 30000, evaluated on 4 unseen seeds (3 succeed)."""
    _write(root / "dataset" / "meta" / "info.json", {"total_frames": 900})
    _write(root / "dataset" / "meta" / "cortex_sim.json", {"instruction": "unscrew the bolt", "kept_seeds": [0, 1],
                                                           "failed_seeds": [], "oracle_success_rate": 1.0})
    _write(root / "runs" / "act" / "checkpoints" / "030000" / "pretrained_model" / "config.json", {"type": "act"})
    res = [{"seed": 100000 + i, "success": i != 3, "steps": 200, "video": None} for i in range(4)]
    _write(root / "runs" / "act" / "eval" / "030000" / "metrics.json",
           {"policy": "act", "seeds": "100000..100003", "success_rate": 0.75, "results": res})
    return root


@pytest.fixture(autouse=True)
def _stopped():
    yield
    lp.STATE.update(job=None, phase="stopped", step=None, detail=None, error=None, failed_at=0.0)


@pytest.fixture
async def world(tmp_path, monkeypatch):
    ctx = await build_ctx(Settings(data_dir=tmp_path / "data", _env_file=None, futurama_projects_file=tmp_path / "x",
                                   live_policy_url="http://127.0.0.1:9", cortex_studio_url="http://studio.test"))
    gpu = {"status": DOWN, "start": DOWN, "started": 0}

    async def status(_ctx):
        return gpu["status"]

    async def start(_ctx):
        gpu["started"] += 1
        if isinstance(gpu["start"], Exception):
            raise gpu["start"]
        gpu["status"] = gpu["start"]
        return gpu["start"]

    monkeypatch.setattr(lp, "status", status)
    monkeypatch.setattr(lp, "start", start)
    sim = await sim_import_service.import_task(ctx, "unscrew", _mirror(tmp_path / "mirror"), SIM_SRC, [])
    return ctx, gpu, sim


async def _skill(ctx):
    """The imported unscrew skill (not a served policy the server runs that was never imported)."""
    [s] = [_skill_brief(s) for s in await skills_service.list_skills(ctx) if s["checkpoint_id"]]
    return s


async def test_save_names_the_skill_everywhere(world):
    ctx, _, sim = world
    s = await _skill(ctx)
    out = await call_tool(ctx, "save_physical_skill", {"project_id": sim["project_id"],
                                                       "checkpoint_id": s["checkpoint_id"], "name": " Unscrew  v1 "}, "mcp")
    assert out["summary"] == "Saved physical skill 'Unscrew v1' (75% success on 4 unseen sim layouts)."
    assert out["skill_id"] == s["skill_id"] and out["saved"] and not s["saved"]
    assert s["score"]["kind"] == "sim_success" and s["deployable"] and not s["live_now"]
    assert out["skill_url"] == f"http://studio.test/studio/skills/{s['skill_id']}"
    again = await _skill(ctx)  # the studio's Physical Skills list reads the same name
    assert again["name"] == "Unscrew v1" and again["saved"]


async def test_save_refuses_a_checkpoint_from_another_project_or_an_unfinished_job(world):
    ctx, _, sim = world
    s = await _skill(ctx)
    other = project_service.create_project(CreateProjectRequest(task_prompt="pick up the cube"))
    await ctx.repo.save("projects", other, status="active")
    with pytest.raises(NotFound):
        await call_tool(ctx, "save_physical_skill", {"project_id": other.id, "checkpoint_id": s["checkpoint_id"],
                                                     "name": "x"}, "mcp")
    tj = TrainingJob(project_id=other.id, dataset_version_id="ds_x", policy_type="act", backend="futurama_slurm",
                     config={}, status=JobStatus.running, rationale="",
                     estimate=ResourceEstimate(gpu="1x L40S", gpu_hours=0, est_cost_usd=0, paid=False, notes=""))
    await ctx.repo.save("training_jobs", tj)
    ck = Checkpoint(project_id=other.id, training_job_id=tj.id, policy_type="act", step=1000, path="runs/act_1")
    await ctx.repo.save("checkpoints", ck, status="ready")
    with pytest.raises(ValueError, match="save the skill once it has succeeded"):
        await call_tool(ctx, "save_physical_skill", {"project_id": other.id, "checkpoint_id": ck.id, "name": "x"},
                        "mcp")


async def test_deploy_goes_through_the_gpu_safeguards(world):
    ctx, gpu, _ = world
    s = await _skill(ctx)
    gpu["start"] = RuntimeError("The last GPU request failed, so Cortex waits 9 more min before asking the cluster")
    with pytest.raises(ValueError, match="waits 9 more min"):
        await call_tool(ctx, "deploy_physical_skill", {"skill_id": s["skill_id"]}, "mcp")
    gpu["start"] = {**DOWN, "phase": "starting", "step": "queued", "job": "10701",
                    "steps": {"queued": "Waiting for a free GPU"}}
    out = await call_tool(ctx, "deploy_physical_skill", {"skill_id": s["skill_id"]}, "mcp")
    assert gpu["started"] == 2 and out["deployed"] is False and out["run_id"] is None
    assert out["summary"].startswith("Deploying physical skill 'Unscrew a bolt with the gripper': Bringing up a GPU")
    assert out["deployment_url"] == f"http://studio.test/studio/skills/{s['skill_id']}"


async def test_deploy_starts_a_live_run_on_an_unseen_layout(world):
    ctx, gpu, sim = world
    gpu["status"] = LIVE
    s = await _skill(ctx)
    out = await call_tool(ctx, "deploy_physical_skill", {"skill_id": s["skill_id"]}, "mcp")
    assert gpu["started"] == 0  # a GPU already live is used as is
    assert out["deployed"] and out["seed"] >= 100_000 and out["project_id"] == sim["project_id"]
    assert out["run_id"] == f"live_sim/unscrew/{out['seed']}" and out["ref_ids"] == [out["run_id"]]
    assert out["deployment_url"] == f"http://studio.test/studio/skills/{s['skill_id']}?seed={out['seed']}"
    assert out["stream_path"] == f"/api/live-policy/sim/unscrew/{out['seed']}"
    assert out["summary"].startswith("Deploying physical skill 'Unscrew a bolt with the gripper' on unseen layout")


async def test_deploy_refuses_what_the_gpu_server_does_not_serve(world):
    ctx, gpu, _ = world
    gpu["status"] = {**LIVE, "sim": {"unscrew": {**LIVE["sim"]["unscrew"], "variant": "smolvla"}}}
    s = await _skill(ctx)
    with pytest.raises(ValueError, match="can't run live yet: The GPU server is running the SmolVLA policy"):
        await call_tool(ctx, "deploy_physical_skill", {"skill_id": s["skill_id"]}, "mcp")
    with pytest.raises(NotFound):
        await call_tool(ctx, "deploy_physical_skill", {"skill_id": "sk_nope"}, "mcp")
