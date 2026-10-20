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


def _write_checkpoint(d):
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"type": "act"}))
    (d / "model.safetensors").write_bytes(b"\0")


async def test_next_actions_say_to_run_and_register_until_a_checkpoint_exists(tmp_path):
    ctx, p, tj, out = await _launched(tmp_path)
    acts = (await orch.project_state(ctx, p)).next_actions
    reg = [n for n in acts if n.action == "register_training_checkpoint"]
    assert reg and reg[0].args["training_job_id"] == tj.id and "run its training command" in reg[0].reason

    # the path the handoff names is absolute, inside the data dir, and is where the command writes
    path = out["register_with"]["args"]["path"]
    assert path.startswith(str(tmp_path.resolve()).replace("\\", "/")) and f"--output_dir={path.rsplit('/checkpoints', 1)[0]}" in out["command"]
    _write_checkpoint(ts.manual_checkpoint_path(ctx, tj))
    got = await orch.call_tool(ctx, "register_training_checkpoint", out["register_with"]["args"], "mcp")
    assert got["ref_ids"][0].startswith("ck_")
    assert (await ctx.repo.get("training_jobs", tj.id, TrainingJob)).status == JobStatus.succeeded
    acts = (await orch.project_state(ctx, p)).next_actions
    assert not [n for n in acts if n.action == "register_training_checkpoint"]


async def test_registering_is_refused_outside_a_launched_manual_job_or_the_data_dir(tmp_path):
    import pytest

    ctx, p, tj, out = await _launched(tmp_path / "data")
    outside = tmp_path / "elsewhere" / "pretrained_model"
    _write_checkpoint(outside)
    with pytest.raises(ValueError, match="inside this job's output folder"):
        await orch.call_tool(ctx, "register_training_checkpoint",
                             {"project_id": p.id, "training_job_id": tj.id, "path": str(outside)}, "mcp")
    with pytest.raises(ValueError, match="inside this job's output folder"):
        await orch.call_tool(ctx, "register_training_checkpoint",
                             {"project_id": p.id, "training_job_id": tj.id, "path": "../elsewhere/pretrained_model"},
                             "mcp")
    proposed = await ts.propose(ctx, p, tj.dataset_version_id)  # never approved
    with pytest.raises(ValueError, match="only a launched manual job"):
        await orch.call_tool(ctx, "register_training_checkpoint",
                             {"project_id": p.id, "training_job_id": proposed.id,
                              "path": str(ts.manual_checkpoint_path(ctx, proposed))}, "mcp")
    other = await orch.create_project(ctx, CreateProjectRequest(task_prompt="stack the red block on the blue one"))
    with pytest.raises(ValueError, match="another project"):
        await orch.call_tool(ctx, "register_training_checkpoint",
                             {"project_id": other.id, "training_job_id": tj.id,
                              "path": out["register_with"]["args"]["path"]}, "mcp")
    assert (await ctx.repo.get("training_jobs", proposed.id, TrainingJob)).status == JobStatus.proposed


async def test_another_folder_in_the_data_dir_is_refused_too(tmp_path):
    import pytest

    ctx, p, tj, out = await _launched(tmp_path)
    other = tmp_path / "projects" / "prj_other" / "training" / "tj_x" / "pretrained_model"
    _write_checkpoint(other)
    with pytest.raises(ValueError, match="this job's output folder"):
        await orch.call_tool(ctx, "register_training_checkpoint",
                             {"project_id": p.id, "training_job_id": tj.id, "path": str(other)}, "mcp")
