"""#152 findings 8-11 (and #125): jobs end in the right state across stops and backend restarts.
ssh, squeue and the sim workers are faked; nothing leaves the machine."""

import asyncio

from cortex.config import Settings
from cortex.context import build_ctx
from cortex.errors import WorkerFailed
from cortex.models import (
    CreateProjectRequest,
    EvaluationRun,
    JobStatus,
    LaunchEvaluationRequest,
    PolicyRef,
    ResourceEstimate,
    SimulationBranch,
    TrainingJob,
)
from cortex.services import agent_orchestrator as orch
from cortex.services import simulation_runner as sr
from cortex.services import training_service as ts
from cortex.services.jobs import EventBus, JobManager


async def _ctx(tmp_path, **kw):
    return await build_ctx(Settings(data_dir=tmp_path, futurama_ssh_host="futurama", _env_file=None, **kw))


async def _slurm_job(ctx, status=JobStatus.running) -> TrainingJob:
    tj = TrainingJob(project_id="prj_1", dataset_version_id="ds_1", policy_type="act", backend="futurama_slurm",
                     config={"steps": 10}, rationale="", status=status, external_id="12345",
                     estimate=ResourceEstimate(gpu="", gpu_hours=0, est_cost_usd=0, paid=False, notes=""))
    await ctx.repo.save("training_jobs", tj)
    return tj


def _fake_sh(calls, squeue=(0, "RUNNING\n"), tail=(0, "")):
    async def sh(*args, timeout=600):
        cmd = " ".join(args)
        calls.append(cmd)
        if "squeue" in cmd:
            return squeue
        if "tail -n 5" in cmd:
            return tail
        return 0, ""
    return sh


# ---------------------------------------------------------------- 11: the stop race

class _Proc:
    """A worker process whose death the job notices before the cancel reaches its task."""

    def __init__(self):
        self.returncode, self.died = None, asyncio.Event()

    def terminate(self):
        self.returncode = -15
        self.died.set()

    kill = terminate

    async def wait(self):
        await asyncio.sleep(0.05)  # the job task runs first and raises WorkerFailed
        return self.returncode


async def test_a_user_stop_that_kills_the_worker_first_is_cancelled_not_failed():
    jm = JobManager(EventBus())

    async def work(job):
        proc = _Proc()
        jm.procs[job.id].append(proc)
        await proc.died.wait()
        raise WorkerFailed("worker exited -15")

    job = jm.submit("collect", "prj_1", work)
    await asyncio.sleep(0.01)
    await jm.cancel(job.id)
    await asyncio.sleep(0.1)
    assert job.status == "cancelled" and job.error is None
    assert not jm.stop_requested(job.id)  # cleared once the job ends


async def test_a_real_worker_failure_is_still_failed():
    jm = JobManager(EventBus())

    async def work(job):
        raise WorkerFailed("worker exited 1: boom")

    job = jm.submit("collect", "prj_1", work)
    await asyncio.sleep(0.05)
    assert job.status == "failed"


# ------------------------------------------------- 8: restarts don't orphan Slurm jobs

async def test_backend_shutdown_keeps_the_slurm_job_for_the_next_start(tmp_path, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(ts, "_sh", _fake_sh(calls))
    ctx = await _ctx(tmp_path)
    tj = await _slurm_job(ctx)
    await ts.resume_pollers(ctx)
    await asyncio.sleep(0.05)
    await ctx.jobs.shutdown()

    cur = await ctx.repo.get("training_jobs", tj.id, TrainingJob)
    assert cur.status == JobStatus.running and cur.external_id == "12345"
    assert not any("scancel" in c for c in calls)

    ctx2 = await _ctx(tmp_path)  # the next start re-attaches to it
    await ts.resume_pollers(ctx2)
    assert [j.kind for j in ctx2.jobs.list("prj_1") if j.status in ("queued", "running")] == ["train"]
    await ctx2.jobs.shutdown()


async def test_a_user_stop_cancels_the_slurm_job(tmp_path, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(ts, "_sh", _fake_sh(calls))
    ctx = await _ctx(tmp_path)
    tj = await _slurm_job(ctx)
    await ts.resume_pollers(ctx)
    job = next(j for j in ctx.jobs.list("prj_1") if j.kind == "train")
    await asyncio.sleep(0.05)
    await ctx.jobs.cancel(job.id)
    await asyncio.sleep(0.05)

    cur = await ctx.repo.get("training_jobs", tj.id, TrainingJob)
    assert cur.status == JobStatus.cancelled and job.status == "cancelled"
    assert any(c.endswith("scancel 12345") for c in calls)


# ------------------------------------------------------- 9: the poller always ends

async def test_a_purged_slurm_job_counts_as_left_the_queue(tmp_path, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(ts, "POLL_S", 0)
    monkeypatch.setattr(ts, "_sh", _fake_sh(calls, squeue=(1, "slurm_load_jobs error: Invalid job id specified"),
                                            tail=(0, "Traceback: CUDA out of memory")))
    ctx = await _ctx(tmp_path)
    tj = await _slurm_job(ctx)
    await ts.resume_pollers(ctx)
    job = next(j for j in ctx.jobs.list("prj_1") if j.kind == "train")
    await asyncio.wait_for(asyncio.shield(ctx.jobs.tasks[job.id]), 5)

    cur = await ctx.repo.get("training_jobs", tj.id, TrainingJob)
    assert cur.status == JobStatus.failed and "ended without completing" in cur.error
    assert sum("squeue" in c for c in calls) == 1


async def test_the_poller_gives_up_after_repeated_ssh_failures(tmp_path, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(ts, "POLL_S", 0)
    monkeypatch.setattr(ts, "MAX_SSH_FAILURES", 3)
    monkeypatch.setattr(ts, "_sh", _fake_sh(calls, squeue=(255, "ssh: connect to host futurama: timed out")))
    ctx = await _ctx(tmp_path)
    tj = await _slurm_job(ctx)
    await ts.resume_pollers(ctx)
    job = next(j for j in ctx.jobs.list("prj_1") if j.kind == "train")
    await asyncio.wait_for(asyncio.shield(ctx.jobs.tasks[job.id]), 5)

    cur = await ctx.repo.get("training_jobs", tj.id, TrainingJob)
    assert cur.status == JobStatus.failed and "lost contact" in cur.error
    assert sum("squeue" in c for c in calls) == 3


# ------------------------------------------- 10 (#125): evaluations never stay running

async def _evaluation(tmp_path, monkeypatch, worker):
    ctx = await _ctx(tmp_path)
    monkeypatch.setattr(ctx.jobs, "run_worker", worker)
    p = await orch.create_project(ctx, CreateProjectRequest(task_prompt="pick up the cube and place it in the bowl"))
    run = await sr.launch_evaluation(ctx, p, LaunchEvaluationRequest(policy=PolicyRef(kind="scripted_expert"),
                                                                     episodes_per_branch=1))
    await asyncio.sleep(0.05)
    return ctx, run


async def _stored(ctx, run):
    r = await ctx.repo.get("evaluation_runs", run.id, EvaluationRun)
    bs = [await ctx.repo.get("simulation_branches", b, SimulationBranch) for b in r.branch_ids]
    return r, bs


async def _hang(*args, **kw):
    await asyncio.sleep(3600)


async def test_a_stopped_evaluation_is_cancelled(tmp_path, monkeypatch):
    ctx, run = await _evaluation(tmp_path, monkeypatch, _hang)
    await ctx.jobs.cancel(run.job_id)
    await asyncio.sleep(0.05)
    r, bs = await _stored(ctx, run)
    assert r.status == JobStatus.cancelled
    assert {b.status for b in bs} == {JobStatus.cancelled}
    assert ctx.jobs.get(run.job_id).status == "cancelled"


async def test_a_stop_that_kills_the_eval_workers_first_is_cancelled(tmp_path, monkeypatch):
    stop = asyncio.Event()

    async def dies_on_stop(*args, **kw):
        await stop.wait()
        raise WorkerFailed("worker exited -15")

    ctx, run = await _evaluation(tmp_path, monkeypatch, dies_on_stop)
    ctx.jobs.stopping.add(run.job_id)  # what jobs.cancel records before it terminates the workers
    stop.set()
    await asyncio.sleep(0.05)
    r, bs = await _stored(ctx, run)
    assert r.status == JobStatus.cancelled and {b.status for b in bs} == {JobStatus.cancelled}
    assert ctx.jobs.get(run.job_id).status == "cancelled"


async def test_shutdown_and_the_startup_sweep_end_an_interrupted_evaluation(tmp_path, monkeypatch):
    ctx, run = await _evaluation(tmp_path, monkeypatch, _hang)
    await ctx.jobs.shutdown()
    r, bs = await _stored(ctx, run)
    assert r.status == JobStatus.failed and all(b.error == sr.INTERRUPTED for b in bs)

    # a crash leaves no handler to run: the next start sweeps whatever is still marked running
    r.status = JobStatus.running
    await ctx.repo.save("evaluation_runs", r)
    for b in bs:
        b.status, b.error = JobStatus.queued, None
        await ctx.repo.save("simulation_branches", b, project_id=b.project_id)
    ctx2 = await _ctx(tmp_path)
    assert await sr.sweep_interrupted(ctx2) == 1
    r, bs = await _stored(ctx2, run)
    assert r.status == JobStatus.failed and {b.status for b in bs} == {JobStatus.failed}


async def test_startup_sweep_skips_a_row_it_cannot_save(tmp_path, monkeypatch):
    """An orphan branch (its project is gone, so the DB rejects the update) must not stop the backend."""
    ctx, run = await _evaluation(tmp_path, monkeypatch, _hang)
    await ctx.jobs.shutdown()
    r, bs = await _stored(ctx, run)
    r.status = JobStatus.running
    await ctx.repo.save("evaluation_runs", r)
    for b in bs:
        b.status = JobStatus.queued
        await ctx.repo.save("simulation_branches", b, project_id=b.project_id)
    ctx2 = await _ctx(tmp_path)
    real = ctx2.repo.save

    async def fk_violation(table, row, **kw):
        if table == "simulation_branches":
            raise RuntimeError("409: Key (project_id)=() is not present in table projects")
        return await real(table, row, **kw)

    monkeypatch.setattr(ctx2.repo, "save", fk_violation)
    assert await sr.sweep_interrupted(ctx2) == 1  # the run is still swept
