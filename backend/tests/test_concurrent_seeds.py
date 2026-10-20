"""#357: collections and robot sessions started at the same moment get disjoint seeds."""

import asyncio

from cortex.config import Settings
from cortex.context import build_ctx
from cortex.models import CreateProjectRequest, StartAgentCollectionRequest
from cortex.services import agent_collection_service as acs
from cortex.services import agent_orchestrator as orch


async def test_two_collections_started_together_get_disjoint_seeds(tmp_path, monkeypatch):
    ctx = await build_ctx(Settings(data_dir=tmp_path, _env_file=None))
    p = await orch.create_project(ctx, CreateProjectRequest(task_prompt="pick up the cube and place it in the bowl"))
    monkeypatch.setattr(ctx.jobs, "submit", lambda *a, **kw: type("J", (), {"id": "job_x"})())
    req = StartAgentCollectionRequest(num_demos=3, collector="oracle")
    a, b = await asyncio.gather(acs.start_collection(ctx, p, req), acs.start_collection(ctx, p, req))
    sa, sb = {s["seed"] for s in a.scenarios}, {s["seed"] for s in b.scenarios}
    assert len(sa) == len(sb) == 3 and not sa & sb
    # and the next one continues past both, even though no attempt has finished yet
    c = await acs.start_collection(ctx, p, req)
    assert min(s["seed"] for s in c.scenarios) > max(sa | sb)


async def test_next_seed_skips_reserved_session_seeds(tmp_path):
    ctx = await build_ctx(Settings(data_dir=tmp_path, _env_file=None))
    first = await acs.next_agent_seed(ctx, "prj_1")
    assert first == acs.AGENT_SEED_BASE
    assert await acs.next_agent_seed(ctx, "prj_1", reserved=[first]) == first + 1
