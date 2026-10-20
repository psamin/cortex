"""A failed or stopped training job can be launched again without re-proposing it."""

import asyncio

import pytest

from cortex.config import Settings
from cortex.context import build_ctx
from cortex.models import CreateProjectRequest, DatasetEpisodeLink, DatasetVersion, JobStatus, TrainingJob
from cortex.services import agent_orchestrator as orch
from cortex.services import training_service as ts


async def _failed(tmp_path, monkeypatch, reached_cluster: bool):
    async def no_vpn(*a, **kw):
        return 255, "ssh: Could not resolve hostname futurama"
    monkeypatch.setattr(ts, "_sh", no_vpn)
    ctx = await build_ctx(Settings(data_dir=tmp_path, futurama_ssh_host="futurama", _env_file=None))
    p = await orch.create_project(ctx, CreateProjectRequest(task_prompt="pick up the cube and place it in the bowl"))
    ds = DatasetVersion(project_id=p.id, name="v1", lerobot_root=f"projects/{p.id}/datasets/v1/lerobot",
                        links=[DatasetEpisodeLink(episode_id="ep_0", split="train", lerobot_episode_index=0)])
    await ctx.repo.save("dataset_versions", ds, status="ready")
    tj = await ts.propose(ctx, p, ds.id)
    await ts.approve(ctx, tj.id, "tester", "gui", None)
    await ts.launch(ctx, tj.id)
    job = next(j for j in ctx.jobs.list(p.id) if j.kind == "train")
    await asyncio.wait_for(asyncio.shield(ctx.jobs.tasks[job.id]), 5)
    tj = await ctx.repo.get("training_jobs", tj.id, TrainingJob)
    assert tj.status == JobStatus.failed and "Could not resolve" in tj.error
    if reached_cluster:
        tj.external_id, tj.metrics = "12345", {"loss_curve": [[1, 0.5]]}
        await ctx.repo.save("training_jobs", tj)
    return ctx, p, tj


async def _hang(*a, **kw):
    await asyncio.sleep(3600)


async def test_a_job_that_failed_before_the_cluster_relaunches_for_anyone(tmp_path, monkeypatch):
    ctx, p, tj = await _failed(tmp_path, monkeypatch, reached_cluster=False)
    state = await orch.project_state(ctx, p)
    [row] = state.training_jobs
    assert "Could not resolve" in row["error"] and row["reached_cluster"] is False
    nxt = [n for n in state.next_actions if n.action == "launch_training_job"]
    assert nxt and not nxt[0].requires_approval and "retry" in nxt[0].reason

    monkeypatch.setattr(ts, "_sh", _hang)
    out = await ts.launch(ctx, tj.id, "mcp")
    assert out.status == JobStatus.queued and out.error is None and out.external_id is None
    await ctx.jobs.shutdown()


async def test_a_job_that_ran_on_the_cluster_needs_a_human_to_retry(tmp_path, monkeypatch):
    ctx, p, tj = await _failed(tmp_path, monkeypatch, reached_cluster=True)
    nxt = [n for n in (await orch.project_state(ctx, p)).next_actions if n.action == "launch_training_job"]
    assert nxt and nxt[0].requires_approval
    with pytest.raises(ts.ApprovalRequired, match="click Retry"):
        await ts.launch(ctx, tj.id, "mcp")
    assert (await ctx.repo.get("training_jobs", tj.id, TrainingJob)).status == JobStatus.failed

    monkeypatch.setattr(ts, "_sh", _hang)
    out = await ts.launch(ctx, tj.id, "gui")
    assert out.status == JobStatus.queued and out.external_id is None
    assert "loss_curve" not in out.metrics  # the old run's curve is not shown as the new one's
    await ctx.jobs.shutdown()


async def test_a_never_approved_job_is_not_retried(tmp_path, monkeypatch):
    ctx, p, tj = await _failed(tmp_path, monkeypatch, reached_cluster=False)
    tj.approved_by = None
    await ctx.repo.save("training_jobs", tj)
    with pytest.raises(ts.ApprovalRequired, match="never approved"):
        await ts.launch(ctx, tj.id, "gui")


async def test_a_newer_job_on_the_same_data_replaces_the_retry_hint(tmp_path, monkeypatch):
    ctx, p, tj = await _failed(tmp_path, monkeypatch, reached_cluster=False)
    await ts.propose(ctx, p, tj.dataset_version_id)
    acts = [n.action for n in (await orch.project_state(ctx, p)).next_actions]
    assert "launch_training_job" not in acts and "approve_training_job" in acts
