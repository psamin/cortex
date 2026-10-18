import asyncio

import pytest

from cortex.config import Settings
from cortex.context import build_ctx
from cortex.models import CreateProjectRequest, DatasetVersion, InfraChoice, StartAgentCollectionRequest
from cortex.services import agent_collection_service as acs
from cortex.services import agent_orchestrator as orch
from cortex.services import infra_service, training_service


def setup(tmp_path, **settings):
    async def go():
        ctx = await build_ctx(Settings(data_dir=tmp_path, _env_file=None, **settings))
        p = await orch.create_project(ctx, CreateProjectRequest(task_prompt="pick up the cube and place it in the bowl"))
        return ctx, p
    return asyncio.run(go())


def test_options_say_why_something_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(infra_service.cc, "available", lambda: "the `claude` CLI is not on PATH")
    ctx, p = setup(tmp_path)
    out = asyncio.run(infra_service.options(ctx, p))
    by = {o["id"]: o for o in out["planner"]}
    assert by["auto"]["available"] and not by["claude_code"]["available"]
    assert by["claude_code"]["reason"] == "the `claude` CLI is not on PATH"
    fut = next(o for o in out["training_backend"] if o["id"] == "futurama_slurm")
    assert not fut["available"] and "FUTURAMA_SSH_HOST" in fut["reason"]
    assert [g["id"] for g in out["gpus"]] == ["gpu:l40s:1", "gpu:l40:1", "gpu:l4:1", "gpu:h200:1"]
    real = next(o for o in out["robot"] if o["id"] == "real_so101")
    assert not real["available"] and "CORTEX_ENABLE_HARDWARE" in real["reason"]
    assert out["choice"] == InfraChoice().model_dump()
    with pytest.raises(ValueError):
        InfraChoice(training_walltime_min=3000)  # beyond the general partition's 1-day limit


def test_a_configured_model_route_reports_its_own_problem(tmp_path, monkeypatch):
    ctx, p = setup(tmp_path)
    monkeypatch.setattr(infra_service.acs, "resolve_planner", lambda ctx: {"label": "self-hosted Qwen (x)"})

    async def down(ctx):
        return "self-hosted endpoint x is not reachable"
    monkeypatch.setattr(infra_service.acs, "selfhosted_reachable", down)
    model = next(o for o in asyncio.run(infra_service.options(ctx, p))["planner"] if o["id"] == "model")
    assert model == {"id": "model", "label": "self-hosted Qwen (x)", "available": False,
                     "reason": "self-hosted endpoint x is not reachable"}


def test_no_paid_model_gateway_is_ever_routed(tmp_path):
    """InsForge credentials or a leftover OPENROUTER_API_KEY never create a model route: model work runs
    on the user's Claude Code, or the opt-in self-hosted endpoint."""
    from cortex.context import resolve_model_route

    s = Settings(data_dir=tmp_path, _env_file=None, insforge_base_url="https://x.insforge.app", insforge_api_key="k",
                 openrouter_api_key="sk-or-x")
    assert resolve_model_route(s) is None
    r = resolve_model_route(Settings(data_dir=tmp_path, _env_file=None, planner_base_url="http://127.0.0.1:8001/v1"))
    assert r is not None and r.provider == "selfhosted" and r.price_per_m is None


def test_training_uses_the_chosen_gpu_and_walltime(tmp_path, monkeypatch):
    ctx, p = setup(tmp_path, futurama_ssh_host="futurama")

    async def go():
        ds = DatasetVersion(project_id=p.id, name="v1", links=[], lerobot_root=str(tmp_path))
        await ctx.repo.save("dataset_versions", ds, status="ready")
        await infra_service.set_choice(ctx, p, InfraChoice(training_gres="gpu:h200:1", training_walltime_min=240))
        p2 = await orch.get_project(ctx, p.id)
        tj = await training_service.propose(ctx, p2, ds.id)
        assert tj.backend == "futurama_slurm"
        assert tj.config["slurm"] == {"partition": "general", "gres": "gpu:h200:1", "time_min": 240}
        assert tj.estimate.gpu.startswith("1x H200 141 GB")
        await infra_service.set_choice(ctx, p2, InfraChoice(training_backend="manual"))
        assert (await training_service.propose(ctx, await orch.get_project(ctx, p.id), ds.id)).backend == "manual"
    asyncio.run(go())


def test_futurama_chosen_without_a_host_is_a_clear_error(tmp_path):
    ctx, p = setup(tmp_path)

    async def go():
        ds = DatasetVersion(project_id=p.id, name="v1", links=[], lerobot_root=str(tmp_path))
        await ctx.repo.save("dataset_versions", ds, status="ready")
        await infra_service.set_choice(ctx, p, InfraChoice(training_backend="futurama_slurm"))
        with pytest.raises(ValueError, match="FUTURAMA_SSH_HOST"):
            await training_service.propose(ctx, await orch.get_project(ctx, p.id), ds.id)
    asyncio.run(go())


def test_collections_use_the_projects_planner_choice(tmp_path, monkeypatch):
    ctx, p = setup(tmp_path)
    seen = []

    async def spy(ctx, req):
        seen.append(req.planner)
        raise acs.PlannerUnavailable("stop here")
    monkeypatch.setattr(acs, "use_claude_code", spy)

    async def go():
        await infra_service.set_choice(ctx, p, InfraChoice(planner="model"))
        p2 = await orch.get_project(ctx, p.id)
        for planner in ("auto", "claude_code"):
            with pytest.raises(acs.PlannerUnavailable):
                await acs.start_collection(ctx, p2, StartAgentCollectionRequest(num_demos=1, planner=planner))
    asyncio.run(go())
    assert seen == ["model", "claude_code"]  # "auto" follows the project; an explicit request wins


def test_the_agent_reads_and_changes_the_same_choices(tmp_path, monkeypatch):
    monkeypatch.setattr(infra_service.cc, "available", lambda: "the `claude` CLI is not on PATH")
    ctx, p = setup(tmp_path)

    async def go():
        got = await orch.call_tool(ctx, "get_infrastructure", {"project_id": p.id}, "mcp")
        assert "planner / Claude Code (your agent): unavailable (the `claude` CLI is not on PATH)" in got["summary"]
        out = await orch.call_tool(ctx, "set_infrastructure", {"project_id": p.id, "training_gres": "gpu:h200:1",
                                                               "training_walltime_min": 300}, "mcp")
        assert out["summary"].startswith("Saved. Chosen: planner auto, training auto on gpu:h200:1 for 300 min.")
        assert (await orch.get_project(ctx, p.id)).infra.training_gres == "gpu:h200:1"
        with pytest.raises(ValueError):  # validated like the studio's form
            await orch.call_tool(ctx, "set_infrastructure", {"project_id": p.id, "training_walltime_min": 5}, "mcp")
    asyncio.run(go())
