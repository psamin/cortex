"""#353: with no GPU backend, launching training hands back the command instead of a 'queued' nobody runs."""

import json

from cortex.config import Settings
from cortex.context import build_ctx
from cortex.models import CreateProjectRequest, DatasetEpisodeLink, DatasetVersion, JobStatus, TrainingJob
from cortex.services import agent_orchestrator as orch
from cortex.services import training_service as ts


async def _launched(tmp_path):
    ctx = await build_ctx(Settings(data_dir=tmp_path, futurama_ssh_host=None, _env_file=None))
    p = await orch.create_project(ctx, CreateProjectRequest(task_prompt="pick up the cube and place it in the bowl"))
    ds = DatasetVersion(project_id=p.id, name="v1", lerobot_root=f"projects/{p.id}/datasets/v1/lerobot",
                        links=[DatasetEpisodeLink(episode_id="ep_0", split="train", lerobot_episode_index=0)])
    await ctx.repo.save("dataset_versions", ds, status="ready")
    tj = await ts.propose(ctx, p, ds.id)
    assert tj.backend == "manual"
    await ts.approve(ctx, tj.id, "tester", "gui", None)
    out = await orch.call_tool(ctx, "launch_training_job", {"training_job_id": tj.id}, "mcp")
    return ctx, p, tj, out


async def test_launch_returns_the_command_and_how_to_register(tmp_path):
    ctx, p, tj, out = await _launched(tmp_path)
    assert out["manual"] and out["requires_human"] and "NOT training" in out["summary"]
    assert "lerobot-train" in out["command"]
    assert out["register_with"]["tool"] == "register_training_checkpoint"
    assert out["register_with"]["args"]["training_job_id"] == tj.id


async def test_next_actions_say_to_run_and_register_until_a_checkpoint_exists(tmp_path):
    ctx, p, tj, _ = await _launched(tmp_path)
    acts = (await orch.project_state(ctx, p)).next_actions
    reg = [n for n in acts if n.action == "register_training_checkpoint"]
    assert reg and reg[0].args["training_job_id"] == tj.id and "run its training command" in reg[0].reason

    ck_dir = tmp_path / "trained" / "pretrained_model"
    ck_dir.mkdir(parents=True)
    (ck_dir / "config.json").write_text(json.dumps({"type": "act"}))
    (ck_dir / "model.safetensors").write_bytes(b"\0")
    out = await orch.call_tool(ctx, "register_training_checkpoint",
                               {"project_id": p.id, "training_job_id": tj.id, "path": str(ck_dir)}, "mcp")
    assert out["ref_ids"][0].startswith("ck_")
    assert (await ctx.repo.get("training_jobs", tj.id, TrainingJob)).status == JobStatus.succeeded
    acts = (await orch.project_state(ctx, p)).next_actions
    assert not [n for n in acts if n.action == "register_training_checkpoint"]
