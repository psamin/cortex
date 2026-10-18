"""prj_e5e9c205d5f3: 27 accepted scripted demos, and train_from_agent failed in a background job with "no accepted
episodes to snapshot (segments: demo, reset)". The snapshot now names the filter that left nothing, and the tool
says it in the call."""

import asyncio

import pytest

from cortex.config import Settings
from cortex.context import build_ctx
from cortex.models import (Condition, CreateDatasetVersionRequest, CreateProjectRequest, Episode, EpisodeSource,
                           EpisodeStatus)
from cortex.services import agent_orchestrator as orch
from cortex.services import dataset_service


def _ep(pid, i, source=EpisodeSource.sim_scripted, segment="demo", status=EpisodeStatus.accepted):
    return Episode(project_id=pid, index=i, source=source, condition=Condition(), seed=i, num_frames=120, fps=30,
                   success=True, status=status, segment=segment)


def _snap(eps, **req):
    return dataset_service.snapshot_episodes(eps, CreateDatasetVersionRequest(**req))


def test_the_error_names_the_source_filter_not_the_segments():
    eps = [_ep("p", i) for i in range(27)]
    with pytest.raises(ValueError) as e:
        _snap(eps, sources=["sim_agent"])
    msg = str(e.value)
    assert "sim_agent" in msg and "27 accepted are 27 sim_scripted" in msg and "segments" not in msg


def test_cycle_segments_demo_and_reset_by_default_recover_on_request():
    eps = [_ep("p", 0), _ep("p", 1, segment="reset"), _ep("p", 2, segment="recover")]
    assert [e.segment for e in _snap(eps)] == ["demo", "reset"]
    assert len(_snap(eps, segments=["demo", "reset", "recover"])) == 3
    with pytest.raises(ValueError, match="segments recover"):
        _snap(eps[:2], segments=["recover"])


def test_nothing_accepted_and_imported_only_say_so():
    with pytest.raises(ValueError, match="record demos"):
        _snap([_ep("p", 0, status=EpisodeStatus.recorded)])
    with pytest.raises(ValueError, match="1 imported"):
        _snap([_ep("p", 0, source=EpisodeSource.imported)])  # real demos never mix into sim datasets by default


def test_train_from_agent_demonstrations_fails_in_the_call_not_in_a_background_job(tmp_path):
    async def go():
        ctx = await build_ctx(Settings(data_dir=tmp_path, _env_file=None, agent_engine="rules"))
        p = await orch.create_project(ctx, CreateProjectRequest(task_prompt="pick up the cube"))
        for i in range(3):
            await ctx.repo.save("episodes", _ep(p.id, i))
        with pytest.raises(ValueError, match="sim_scripted"):
            await orch.call_tool(ctx, "train_from_agent_demonstrations", {"project_id": p.id}, actor="agent")
        jobs = ctx.jobs.list(p.id)
        ok = await orch.call_tool(ctx, "train_from_agent_demonstrations",
                                  {"project_id": p.id, "sources": ["sim_scripted"]}, actor="agent")
        await ctx.jobs.shutdown()
        return jobs, ok

    jobs, ok = asyncio.run(go())
    assert jobs == [] and ok["job_id"].startswith("job_")
