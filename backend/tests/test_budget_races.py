"""#152 findings 13 and 14: in-flight planner budget, and a scene build that keeps a newer policy."""

import asyncio
from pathlib import Path

import pytest

from cortex.config import Settings
from cortex.context import build_ctx
from cortex.intake.service import start as intake_start
from cortex.intake.spec import IntakeStatus
from cortex.models import (
    AgentAttempt, AgentCollection, AutopilotPolicy, CreateProjectRequest, Project, Scenario,
    StartAgentCollectionRequest,
)
from cortex.services import agent_collection_service as acs
from cortex.services import agent_orchestrator as orch
from cortex.services import intake_service
from cortex.services.jobs import Job
from cortex.sim.scene import SceneInfo


async def _project(tmp_path, budget: float = 10.0):
    ctx = await build_ctx(Settings(data_dir=tmp_path, _env_file=None))
    p = await orch.create_project(ctx, CreateProjectRequest(
        task_prompt="pick up the cube and place it in the bowl",
        autopilot=AutopilotPolicy(model_budget_usd=budget, collector="agent")))
    return ctx, p


def _running(project_id: str, cap: float) -> AgentCollection:
    return AgentCollection(
        project_id=project_id, collector="agent", status="running",
        request=StartAgentCollectionRequest(max_total_usd=cap, collector="agent"),
        totals=acs._empty_totals(cap))


async def test_a_running_collection_reserves_its_unspent_cap(tmp_path):
    ctx, p = await _project(tmp_path)
    col = _running(p.id, 8)
    await ctx.repo.save("agent_collections", col)
    attempt = AgentAttempt(project_id=p.id, collection_id=col.id, index=0, collector="agent", seed=1,
                           scenario=Scenario(), status="failed", usage={"cost_usd": 2.0})
    await ctx.repo.save("agent_attempts", attempt)

    # 10 standing - 2 already spent - 6 still reserved of the $8 cap = $2 left.
    assert await acs.reserved_usd(ctx, p.id) == 6
    with pytest.raises(acs.PlannerBudgetExceeded):
        await acs.check_budget(ctx, p, StartAgentCollectionRequest(max_total_usd=3, collector="agent"))
    assert await acs.check_budget(ctx, p, StartAgentCollectionRequest(max_total_usd=2, collector="agent")) == 2

    col.status = "succeeded"
    await ctx.repo.save("agent_collections", col)
    assert await acs.reserved_usd(ctx, p.id) == 0
    assert await acs.check_budget(ctx, p, StartAgentCollectionRequest(max_total_usd=8, collector="agent")) == 8


async def test_two_starts_cannot_each_take_the_full_budget(tmp_path, monkeypatch):
    ctx, p = await _project(tmp_path, budget=10)

    async def not_claude(*_a, **_k):
        return False

    async def up(*_a, **_k):
        return None

    monkeypatch.setattr(acs, "use_claude_code", not_claude)
    monkeypatch.setattr(acs, "selfhosted_reachable", up)
    monkeypatch.setattr(acs, "openrouter_balance_problem", up)
    monkeypatch.setattr(acs, "resolve_planner", lambda _ctx: {
        "provider": "test", "model": "m", "label": "test", "api_key": "k", "endpoint": "http://planner",
        "point_format": "px", "openrouter_extras": False, "price_per_m": {"input": 1.0, "output": 1.0}})

    def no_worker(kind, project_id, run, idempotency_key=None):
        return Job(kind=kind, project_id=project_id)

    monkeypatch.setattr(ctx.jobs, "submit", no_worker)
    req = StartAgentCollectionRequest(num_demos=1, max_total_usd=10, collector="agent", planner="model")
    first, second = await asyncio.gather(
        acs.start_collection(ctx, p, req, actor="test"),
        acs.start_collection(ctx, p, req, actor="test"),
        return_exceptions=True)
    saved = [r for r in (first, second) if isinstance(r, AgentCollection)]
    refused = [r for r in (first, second) if isinstance(r, acs.PlannerBudgetExceeded)]
    assert len(saved) == 1 and len(refused) == 1
    assert saved[0].totals["budget_cap_usd"] == 10


async def test_a_policy_write_during_scene_build_is_kept(tmp_path, monkeypatch):
    ctx, p = await _project(tmp_path)
    p.intake = intake_start("pick up the cube")
    await ctx.repo.save("projects", p, status="active")

    def fake_status(spec, previous=None):
        return IntakeStatus(spec=spec, questions=[], complete=True)

    def fake_build(spec, out_dir):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "scene_test.xml"
        path.write_text("<mujoco/>", encoding="utf-8")
        return SceneInfo(path=path, objects={}, zones={}, starts={}, heights={}, pick=None, place=[])

    async def write_policy_then_report(_fn, *_args):
        stored = await ctx.repo.get("projects", p.id, Project)
        stored.autopilot.model_budget_usd = 42
        await ctx.repo.save("projects", stored, status="active")
        return {"summary": "ok"}

    async def no_frame(*_a, **_k):
        return {}

    monkeypatch.setattr(intake_service, "status", fake_status)
    monkeypatch.setattr(intake_service, "build_scene", fake_build)
    monkeypatch.setattr(intake_service, "_on_gl", write_policy_then_report)
    monkeypatch.setattr(intake_service, "capture_frame", no_frame)

    out = await intake_service.build_simulation(ctx, p, "test")
    fresh = await ctx.repo.get("projects", p.id, Project)
    assert fresh.autopilot.model_budget_usd == 42
    assert fresh.scene["digest"] == out["scene"]["digest"]
    assert out["feasibility"]["summary"] == "ok"
