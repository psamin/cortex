"""#349: a training job whose backend died before Slurm accepted it doesn't stay 'queued' forever."""

from cortex.config import Settings
from cortex.context import build_ctx
from cortex.models import JobStatus, ResourceEstimate, TrainingJob
from cortex.services import training_service as ts


def _tj(backend="futurama_slurm", status=JobStatus.queued, external_id=None) -> TrainingJob:
    return TrainingJob(project_id="prj_1", dataset_version_id="ds_1", policy_type="act", backend=backend,
                       config={"steps": 10}, rationale="", status=status, external_id=external_id,
                       approved_by="tester",
                       estimate=ResourceEstimate(gpu="", gpu_hours=0, est_cost_usd=0, paid=False, notes=""))


async def _ctx(tmp_path, host="futurama"):
    return await build_ctx(Settings(data_dir=tmp_path, futurama_ssh_host=host, _env_file=None))


async def test_a_job_killed_mid_upload_is_failed_at_the_next_start(tmp_path, monkeypatch):
    async def never(*a, **kw):
        raise AssertionError("nothing should be polled")
    monkeypatch.setattr(ts, "_sh", never)
    ctx = await _ctx(tmp_path)
    uploading, starting, manual = _tj(), _tj(status=JobStatus.running), _tj(backend="manual")
    for t in (uploading, starting, manual):
        await ctx.repo.save("training_jobs", t)

    ctx2 = await _ctx(tmp_path)  # the hard-killed backend comes back
    await ts.resume_pollers(ctx2)
    for t in (uploading, starting):
        cur = await ctx2.repo.get("training_jobs", t.id, TrainingJob)
        assert cur.status == JobStatus.failed and cur.error == ts.INTERRUPTED_BEFORE_SUBMIT
    # a manual job is queued until a human runs its command: left alone
    assert (await ctx2.repo.get("training_jobs", manual.id, TrainingJob)).status == JobStatus.queued
    assert not [j for j in ctx2.jobs.list("prj_1") if j.kind == "train"]


async def test_it_is_swept_even_without_a_cluster_configured(tmp_path):
    ctx = await _ctx(tmp_path, host=None)
    t = _tj()
    await ctx.repo.save("training_jobs", t)
    await ts.resume_pollers(await _ctx(tmp_path, host=None))
    assert (await ctx.repo.get("training_jobs", t.id, TrainingJob)).status == JobStatus.failed


async def test_a_submitted_job_is_still_re_attached(tmp_path, monkeypatch):
    async def running(*a, **kw):
        return 0, "RUNNING\n"
    monkeypatch.setattr(ts, "_sh", running)
    ctx = await _ctx(tmp_path)
    t = _tj(status=JobStatus.running, external_id="12345")
    await ctx.repo.save("training_jobs", t)
    ctx2 = await _ctx(tmp_path)
    await ts.resume_pollers(ctx2)
    assert (await ctx2.repo.get("training_jobs", t.id, TrainingJob)).status == JobStatus.running
    assert [j.kind for j in ctx2.jobs.list("prj_1")] == ["train"]
    await ctx2.jobs.shutdown()
