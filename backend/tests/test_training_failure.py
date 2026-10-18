import asyncio

from cortex.config import Settings
from cortex.context import build_ctx
from cortex.models import CreateProjectRequest, DatasetVersion, JobStatus, TrainingJob
from cortex.services import training_service
from cortex.services.project_service import create_project


async def _ctx_with_approved_job(tmp_path):
    ctx = await build_ctx(Settings(data_dir=tmp_path, futurama_ssh_host="futurama", _env_file=None))
    p = create_project(CreateProjectRequest(task_prompt="pick up the cube and place it in the bowl"))
    ds = DatasetVersion(project_id=p.id, name="v1", links=[], lerobot_root="projects/x/datasets/d/lerobot")
    await ctx.repo.save("dataset_versions", ds, status="ready")
    tj = await training_service.propose(ctx, p, ds.id)
    assert tj.backend == "futurama_slurm"
    await training_service.approve(ctx, tj.id, "tester", "gui", None)
    return ctx, tj


async def test_failed_submit_marks_the_training_job_failed(tmp_path, monkeypatch):
    async def no_vpn(*args, timeout=600):
        return 255, "ssh: Could not resolve hostname futurama"
    monkeypatch.setattr(training_service, "_sh", no_vpn)
    ctx, tj = await _ctx_with_approved_job(tmp_path)

    await training_service.launch(ctx, tj.id)
    job = next(j for j in ctx.jobs.list(tj.project_id) if j.kind == "train")
    await asyncio.wait_for(asyncio.shield(ctx.jobs.tasks[job.id]), 5)

    cur = await ctx.repo.get("training_jobs", tj.id, TrainingJob)
    assert job.status == "failed"
    assert cur.status == JobStatus.failed and "Could not resolve hostname" in cur.error


async def test_stopped_submit_marks_the_training_job_cancelled(tmp_path, monkeypatch):
    async def hang(*args, timeout=600):
        await asyncio.sleep(3600)
    monkeypatch.setattr(training_service, "_sh", hang)
    ctx, tj = await _ctx_with_approved_job(tmp_path)

    await training_service.launch(ctx, tj.id)
    job = next(j for j in ctx.jobs.list(tj.project_id) if j.kind == "train")
    await asyncio.sleep(0.05)
    await ctx.jobs.cancel(job.id)
    await asyncio.sleep(0.05)

    cur = await ctx.repo.get("training_jobs", tj.id, TrainingJob)
    assert cur.status == JobStatus.cancelled
