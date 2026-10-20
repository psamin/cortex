"""#347: an event loop without subprocess support (uvicorn --reload on Windows) is reported, not silent."""
import asyncio

import pytest

from cortex import subprocs
from cortex.errors import WorkerFailed
from cortex.services import claude_code_agent as cca
from cortex.services.jobs import EventBus, JobManager


def test_only_windows_selector_loops_are_refused(monkeypatch):
    loop = asyncio.new_event_loop()
    try:
        monkeypatch.setattr(subprocs.sys, "platform", "linux")
        assert subprocs.unavailable_reason(loop) is None
        monkeypatch.setattr(subprocs.sys, "platform", "win32")
        assert subprocs.unavailable_reason(asyncio.SelectorEventLoop()) == subprocs.WHY

        class Proactor(asyncio.SelectorEventLoop):  # stands in for asyncio.ProactorEventLoop off Windows
            pass
        monkeypatch.setattr(subprocs.asyncio, "ProactorEventLoop", Proactor, raising=False)
        assert subprocs.unavailable_reason(Proactor()) is None
    finally:
        loop.close()


async def test_the_chat_engine_reports_it_instead_of_failing_every_turn(monkeypatch):
    monkeypatch.setattr(cca.shutil, "which", lambda name: "/usr/bin/claude")
    monkeypatch.setattr(subprocs, "unavailable_reason", lambda loop=None: subprocs.WHY)
    assert cca.available() == subprocs.WHY


async def test_a_worker_spawn_says_why(tmp_path, monkeypatch):
    async def no_subprocesses(*a, **kw):
        raise NotImplementedError
    monkeypatch.setattr(asyncio, "create_subprocess_exec", no_subprocesses)
    jm = JobManager(EventBus())
    job = jm.submit("noop", "prj_1", lambda j: asyncio.sleep(0))
    with pytest.raises(WorkerFailed) as e:
        await jm.run_worker(job, "cortex.workers.none", {}, tmp_path / "cfg.json")
    assert e.value.user_facing and "--reload" in str(e.value)


async def test_a_cluster_launch_is_refused_up_front(tmp_path, monkeypatch):
    from cortex.config import Settings
    from cortex.context import build_ctx
    from cortex.models import CreateProjectRequest, DatasetVersion
    from cortex.services import training_service as ts
    from cortex.services.project_service import create_project

    ctx = await build_ctx(Settings(data_dir=tmp_path, futurama_ssh_host="futurama", _env_file=None))
    p = create_project(CreateProjectRequest(task_prompt="pick up the cube and place it in the bowl"))
    ds = DatasetVersion(project_id=p.id, name="v1", links=[], lerobot_root="projects/x/d")
    await ctx.repo.save("dataset_versions", ds, status="ready")
    tj = await ts.propose(ctx, p, ds.id)
    await ts.approve(ctx, tj.id, "tester", "gui", None)
    monkeypatch.setattr(subprocs, "unavailable_reason", lambda loop=None: subprocs.WHY)
    with pytest.raises(ValueError, match="--reload"):
        await ts.launch(ctx, tj.id)
    assert not [j for j in ctx.jobs.list(p.id) if j.kind == "train"]
