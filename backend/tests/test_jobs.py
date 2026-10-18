import asyncio
import json

from cortex.services.jobs import EventBus, JobManager


def test_jobs_persist_notify_and_survive_restart(tmp_path):
    path = tmp_path / "jobs.json"
    finished = []

    async def run():
        jm = JobManager(EventBus(), persist=path)
        jm.on_finished.append(lambda j: finished.append((j.kind, j.status)))
        jm.on_finished.append(lambda j: 1 / 0)  # a broken listener must not break the job
        ok = jm.submit("import", "prj_1", lambda job: asyncio.sleep(0, result="ds_1"))
        bad = jm.submit("train", "prj_1", lambda job: _boom())
        stuck = jm.submit("evaluate", "prj_1", lambda job: asyncio.sleep(3600))
        await asyncio.sleep(0.05)
        return ok, bad, stuck, path.read_text()  # snapshot while `stuck` is still running (the process "dies" here)

    async def _boom():
        raise RuntimeError("no GPU")

    ok, bad, stuck, snapshot = asyncio.run(run())
    assert ok.status == "succeeded" and ok.result_ref == "ds_1"
    assert bad.status == "failed" and "no GPU" in bad.error_detail  # people see a sentence, not the raw error
    assert ("import", "succeeded") in finished and ("train", "failed") in finished
    saved = {j["id"]: j["status"] for j in json.loads(snapshot)}
    assert saved[ok.id] == "succeeded" and saved[stuck.id] == "running"
    path.write_text(snapshot)

    # a new process (backend restart) reports the in-flight job as interrupted instead of 404
    jm2 = JobManager(EventBus(), persist=path)
    assert jm2.get(ok.id).status == "succeeded"
    assert jm2.get(stuck.id).status == "failed" and "restart" in jm2.get(stuck.id).error


def test_finished_jobs_past_the_limit_are_forgotten_in_memory_too(tmp_path, monkeypatch):
    from cortex.services import jobs

    monkeypatch.setattr(jobs, "MAX_KEPT", 3)

    async def run():
        jm = JobManager(EventBus(), persist=tmp_path / "jobs.json")
        stuck = jm.submit("evaluate", "prj_1", lambda job: asyncio.sleep(3600))
        done = [jm.submit("import", "prj_1", lambda job: asyncio.sleep(0)) for _ in range(5)]
        await asyncio.sleep(0.05)
        kept = set(jm.jobs)
        stuck_task = jm.tasks.get(stuck.id)
        await jm.shutdown()
        return stuck, done, kept, stuck_task, jm

    stuck, done, kept, stuck_task, jm = asyncio.run(run())
    assert stuck.id in kept  # unfinished jobs are never pruned, however old
    assert {d.id for d in done[-3:]} <= kept and not {d.id for d in done[:2]} & kept
    assert set(jm.tasks) <= set(jm.jobs)
    assert stuck_task is not None and stuck_task.cancelled()


def test_shutdown_ends_workers_and_reports_the_job_as_interrupted(tmp_path):
    import sys

    path = tmp_path / "jobs.json"

    async def run():
        jm = JobManager(EventBus(), persist=path)
        holder = {}

        async def work(job):
            proc = await asyncio.create_subprocess_exec(sys.executable, "-c", "import time; time.sleep(60)",
                                                        stdout=asyncio.subprocess.PIPE)
            jm.procs[job.id].append(proc)
            holder["proc"] = proc
            await proc.wait()

        job = jm.submit("collect", "prj_1", work)
        await asyncio.sleep(0.5)
        await jm.shutdown()
        return job, holder["proc"]

    job, proc = asyncio.run(run())
    assert proc.returncode is not None  # the worker did not outlive the backend
    assert job.status == "running"  # not "stopped by user", and not a worker failure
    assert JobManager(EventBus(), persist=path).get(job.id).error.startswith("It stopped because Cortex restarted")
