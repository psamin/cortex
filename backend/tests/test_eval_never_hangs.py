"""#125: an evaluation never hangs with every branch queued. Sim workers are faked; each wait is bounded, so the
old behaviour (a job 'running' forever at progress 0) fails here instead of hanging the suite."""

import asyncio
import json
import threading
from pathlib import Path

from cortex.config import Settings
from cortex.context import build_ctx
from cortex.models import (
    CreateProjectRequest,
    EvaluationRun,
    JobStatus,
    LaunchEvaluationRequest,
    PolicyRef,
    SimulationBranch,
)
from cortex.services import agent_orchestrator as orch
from cortex.services import jobs
from cortex.services import simulation_runner as sr
from cortex.services.jobs import EventBus, Job, JobManager


async def _until(cond, timeout=3.0):
    async def poll():
        while not await cond():
            await asyncio.sleep(0.02)
    await asyncio.wait_for(poll(), timeout)


async def _worker(job, module, cfg, cfg_path, on_event=None, env=None):
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps({"policy": "fake", "episodes": [
        {"seed": s, "success": True, "reward": 1.0, "steps": 1, "max_cube_z": 0.1, "final_cube_box_dist": 0.0}
        for s in cfg["seeds"]]}))


async def _project(tmp_path):
    ctx = await build_ctx(Settings(data_dir=tmp_path, _env_file=None))
    ctx.jobs.run_worker = _worker
    return ctx, await orch.create_project(ctx, CreateProjectRequest(task_prompt="pick up the cube and place it in the bowl"))


def _launch(ctx, p):
    return sr.launch_evaluation(ctx, p, LaunchEvaluationRequest(policy=PolicyRef(kind="scripted_expert"),
                                                                episodes_per_branch=1))


async def _stored(ctx, run):
    r = await ctx.repo.get("evaluation_runs", run.id, EvaluationRun)
    return r, [await ctx.repo.get("simulation_branches", b, SimulationBranch) for b in r.branch_ids]


# ------------------------------------------------- the cause: the job's state file write at its start

async def test_a_job_whose_state_file_write_fails_at_start_still_runs(tmp_path, monkeypatch):
    """Windows refuses os.replace while another process holds jobs.json; that write is a snapshot, not the job."""
    jm = JobManager(EventBus(), persist=tmp_path / "jobs.json")
    real, failed = jobs.os.replace, []

    def replace(src, dst):
        if any(j.status == "running" for j in jm.jobs.values()) and not failed:  # the write as the job starts
            failed.append(dst)
            raise PermissionError(13, "Access is denied", str(dst))
        return real(src, dst)

    monkeypatch.setattr(jobs.os, "replace", replace)
    ran = []
    job = jm.submit("evaluate", "prj_1", lambda j: asyncio.sleep(0, result=ran.append(j.id)))

    async def done():
        return job.status in jobs.TERMINAL
    await _until(done)
    assert failed and ran == [job.id] and job.status == "succeeded"


async def test_any_other_error_as_a_job_starts_fails_the_job_instead_of_leaving_it_running(tmp_path, monkeypatch):
    jm = JobManager(EventBus(), persist=tmp_path / "jobs.json")
    real = jm._save

    def save():
        if any(j.status == "running" for j in jm.jobs.values()) and not hasattr(jm, "_broke"):
            jm._broke = True
            raise RuntimeError("state snapshot broke")
        real()

    monkeypatch.setattr(jm, "_save", save)
    job = jm.submit("evaluate", "prj_1", lambda j: asyncio.sleep(3600))

    async def done():
        return job.status in jobs.TERMINAL
    await _until(done)
    assert job.status == "failed" and "state snapshot broke" in job.error_detail


async def test_two_backends_on_one_data_dir_never_leave_an_evaluation_queued(tmp_path):
    """The trigger reproduced: another backend persisting the same jobs.json races this one's write."""
    ctx, p = await _project(tmp_path)
    other, stop = JobManager(EventBus(), persist=tmp_path / "jobs.json"), threading.Event()

    def other_backend():
        while not stop.is_set():
            try:
                other._save()
            except OSError:
                pass

    t = threading.Thread(target=other_backend, daemon=True)
    t.start()
    try:
        runs = []
        for _ in range(6):
            runs += await asyncio.gather(*(_launch(ctx, p) for _ in range(5)))
            await asyncio.sleep(0.02)

        async def all_done():
            return all(ctx.jobs.get(r.job_id).status in jobs.TERMINAL for r in runs)
        await _until(all_done, 10)
    finally:
        stop.set()
        t.join()
    for r in runs:
        stored, branches = await _stored(ctx, r)
        assert stored.status == JobStatus.succeeded and {b.status for b in branches} == {JobStatus.succeeded}


# ------------------------------------------------- the watchdog: a branch that never starts is failed in words

async def test_a_branch_whose_start_is_stuck_in_the_store_is_failed_and_the_others_run(tmp_path, monkeypatch):
    monkeypatch.setattr(sr, "START_TIMEOUT_S", 0.2, raising=False)
    ctx, p = await _project(tmp_path)
    real, stuck = ctx.repo.save, []

    async def save(table, obj, **kw):
        if table == "simulation_branches" and obj.status == JobStatus.running and not stuck:
            stuck.append(obj.id)
            await asyncio.Event().wait()  # a store call that never returns
        return await real(table, obj, **kw)

    monkeypatch.setattr(ctx.repo, "save", save)
    run = await _launch(ctx, p)

    async def done():
        return ctx.jobs.get(run.job_id).status in jobs.TERMINAL
    await _until(done)
    r, bs = await _stored(ctx, run)
    by_id = {b.id: b for b in bs}
    assert by_id[stuck[0]].status == JobStatus.failed and by_id[stuck[0]].error.startswith("Never started")
    assert [b.status for b in bs if b.id != stuck[0]] == [JobStatus.succeeded] * 2
    assert r.status == JobStatus.succeeded and ctx.jobs.get(run.job_id).status == "succeeded"


async def test_an_evaluation_whose_branches_all_never_start_fails_with_the_reason(tmp_path, monkeypatch):
    monkeypatch.setattr(sr, "START_TIMEOUT_S", 0.2, raising=False)
    ctx, p = await _project(tmp_path)
    real = ctx.repo.save

    async def save(table, obj, **kw):
        if table == "simulation_branches" and obj.status == JobStatus.running:
            await asyncio.Event().wait()
        return await real(table, obj, **kw)

    monkeypatch.setattr(ctx.repo, "save", save)
    run = await _launch(ctx, p)

    async def done():
        return ctx.jobs.get(run.job_id).status in jobs.TERMINAL
    await _until(done)
    monkeypatch.setattr(ctx.repo, "save", real)
    r, bs = await _stored(ctx, run)
    job = ctx.jobs.get(run.job_id)
    assert r.status == JobStatus.failed and {b.status for b in bs} == {JobStatus.failed}
    assert job.status == "failed" and job.error.startswith("Never started")  # words, not "internal error"


async def test_an_evaluation_whose_job_never_reaches_its_branches_is_failed(tmp_path, monkeypatch):
    """Whatever stops the job before its branches (the #125 hang), the watchdog beside it ends the evaluation."""
    monkeypatch.setattr(sr, "START_TIMEOUT_S", 0.2, raising=False)
    ctx, p = await _project(tmp_path)

    def submit_that_never_runs(kind, project_id, fn, idempotency_key=None):
        job = Job(kind=kind, project_id=project_id, status="running", idempotency_key=idempotency_key)
        ctx.jobs.jobs[job.id] = job
        return job

    monkeypatch.setattr(ctx.jobs, "submit", submit_that_never_runs)
    run = await _launch(ctx, p)

    async def ended():
        r, _ = await _stored(ctx, run)
        return r.status != JobStatus.running
    await _until(ended)
    r, bs = await _stored(ctx, run)
    job = ctx.jobs.get(run.job_id)
    assert r.status == JobStatus.failed and {b.status for b in bs} == {JobStatus.failed}
    assert all(b.error.startswith("Never started") for b in bs)
    assert job.status == "failed" and job.error.startswith("Never started")
