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
