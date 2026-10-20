"""Real trials and sim <-> real (docs/STACK_SPEC.md 2.2)."""

from types import SimpleNamespace

import pytest

from cortex.config import Settings
from cortex.context import build_ctx
from cortex.models import BranchResult, BranchSpec, Checkpoint, Condition, CreateProjectRequest, JobStatus
from cortex.services import agent_orchestrator as orch
from cortex.services import real_trials as rt
from cortex.services import simulation_runner


def _branch(name, k, n, cond=Condition(), offset=(0.0, 0.0)):
    return SimpleNamespace(spec=BranchSpec(name=name, condition=cond, cube_offset=offset),
                           result=BranchResult(n=n, successes=k, success_rate=k / n, ci95=(0, 1), mean_reward=0,
                                               episodes=[]))


def _eval(ck, branches, status=JobStatus.succeeded):
    return {"run": SimpleNamespace(id=f"ev_{ck}", status=status,
                                   policy=SimpleNamespace(kind="checkpoint", checkpoint_id=ck)),
            "branches": branches}


async def _setup(tmp_path, monkeypatch, sims):
    ctx = await build_ctx(Settings(data_dir=tmp_path, _env_file=None))
    p = await orch.create_project(ctx, CreateProjectRequest(task_prompt="pick up the cube and place it in the bowl"))
    cks = []
    for i in range(4):
        ck = Checkpoint(project_id=p.id, training_job_id=None, policy_type="act", step=i, path="x")
        await ctx.repo.save("checkpoints", ck)
        cks.append(ck.id)

    async def evals(ctx, pid):
        return [_eval(cks[i], b) for i, b in sims.items()]
    monkeypatch.setattr(simulation_runner, "list_evaluations", evals)
    monkeypatch.setattr(rt, "list_evaluations", evals)
    return ctx, p, cks


def trials(ck, k, n, cond=None):
    return [rt.TrialIn(checkpoint_id=ck, success=i < k, **({"condition": cond} if cond else {})) for i in range(n)]


async def test_real_next_to_sim_per_checkpoint_and_condition(tmp_path, monkeypatch):
    sims = {0: [_branch("baseline", 8, 10), _branch("dim", 3, 10, Condition(lighting="dim"))]}
    ctx, p, cks = await _setup(tmp_path, monkeypatch, sims)
    await rt.log_trials(ctx, p, trials(cks[0], 3, 5), "Ana", "studio", "gui")
    await rt.log_trials(ctx, p, trials(cks[0], 0, 4, Condition(lighting="dim")), "Ana", "studio", "gui")
    rep = await rt.report(ctx, p)
    rows = {r.condition: r for r in rep.rows}
    assert rows["center/normal"].real.rate == pytest.approx(0.6) and rows["center/normal"].sim.rate == pytest.approx(0.8)
    assert rows["center/normal"].gap == pytest.approx(-0.2)
    assert rows["center/dim"].real.successes == 0 and rows["center/dim"].sim.rate == pytest.approx(0.3)
    (all_,) = rep.checkpoints
    assert all_.real.n == 9 and all_.sim.n == 20
    assert rep.spearman is None and "needs 2 more checkpoints" in rep.summary


async def test_a_shifted_branch_counts_for_the_region_the_object_lands_in(tmp_path, monkeypatch):
    sims = {0: [_branch("shifted_object", 2, 10, offset=(0.04, 0.06))]}
    ctx, p, cks = await _setup(tmp_path, monkeypatch, sims)
    await rt.log_trials(ctx, p, trials(cks[0], 1, 4, Condition(cube_region="left")), "Ana", "studio", "gui")
    (row,) = (await rt.report(ctx, p)).rows
    assert row.condition == "left/normal" and row.sim.rate == pytest.approx(0.2)


async def test_rank_agreement_across_checkpoints(tmp_path, monkeypatch):
    sims = {i: [_branch("baseline", k, 10)] for i, k in enumerate([2, 5, 7, 9])}
    ctx, p, cks = await _setup(tmp_path, monkeypatch, sims)
    for ck, k in zip(cks, [1, 4, 6, 9]):  # the real arm ranks them the same way
        await rt.log_trials(ctx, p, trials(ck, k, 10), "Ana", "studio", "gui")
    rep = await rt.report(ctx, p)
    assert rep.checkpoints_compared == 4 and rep.spearman == pytest.approx(1.0) and "agree" in rep.summary


def test_spearman_handles_ties_and_flat_data():
    assert rt.spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert rt.spearman([1, 1, 2], [1, 2, 3]) == pytest.approx(0.866, abs=1e-3)
    assert rt.spearman([1, 1, 1], [1, 2, 3]) is None


async def test_an_agent_records_but_never_judges(tmp_path, monkeypatch):
    ctx, p, cks = await _setup(tmp_path, monkeypatch, {})
    from pydantic import ValidationError

    with pytest.raises(ValidationError):  # judged_by is required: a person watched the rollout
        await orch.call_tool(ctx, "log_real_trials", {"project_id": p.id, "trials": [
            {"checkpoint_id": cks[0], "success": True}]}, "mcp")
    out = await orch.call_tool(ctx, "log_real_trials", {"project_id": p.id, "judged_by": "Ana (operator)", "trials": [
        {"checkpoint_id": cks[0], "success": True}]}, "mcp")
    (t,) = await rt.list_trials(ctx, p.id)
    assert t.source == "agent" and t.judged_by == "Ana (operator)" and out["sim_real"]["checkpoints"]
    with pytest.raises(PermissionError):  # only the studio withdraws a judgement
        await rt.void_trial(ctx, t.id, "mcp")
    await rt.void_trial(ctx, t.id, "gui")
    assert await rt.list_trials(ctx, p.id) == []


async def test_trials_must_name_this_projects_checkpoints(tmp_path, monkeypatch):
    from cortex.errors import NotFound

    ctx, p, cks = await _setup(tmp_path, monkeypatch, {})
    with pytest.raises(NotFound):
        await rt.log_trials(ctx, p, trials(cks[0], 1, 1) + trials("ck_elsewhere", 1, 1), "Ana", "studio", "gui")
    assert await rt.list_trials(ctx, p.id) == []  # nothing saved when one is wrong


async def test_next_actions_suggest_real_trials_only_with_hardware(tmp_path, monkeypatch):
    sims = {0: [_branch("baseline", 8, 10)]}
    ctx, p, cks = await _setup(tmp_path, monkeypatch, sims)

    async def none(ctx, pid):
        return []
    monkeypatch.setattr(simulation_runner, "list_evaluations", none)  # state built bare, then given one evaluation
    state = await orch.project_state(ctx, p)
    state.evaluations = [{"id": "ev", "status": JobStatus.succeeded, "checkpoint_id": cks[0], "branches": {}}]
    state.episode_counts = {**state.episode_counts, "total": 12, "accepted": 12}  # a project that has demos
    assert "log_real_trials" not in [n.action for n in await orch.next_actions(ctx, p, state)]
    ctx.settings.cortex_enable_hardware = True
    assert "log_real_trials" in [n.action for n in await orch.next_actions(ctx, p, state)]
    await rt.log_trials(ctx, p, trials(cks[0], 1, 1), "Ana", "studio", "gui")
    assert "log_real_trials" not in [n.action for n in await orch.next_actions(ctx, p, state)]


def test_the_routes(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    for k in ("INSFORGE_BASE_URL", "INSFORGE_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    from cortex.config import settings

    settings.cache_clear()
    from cortex.api import app as api

    studio = {"Origin": "http://localhost:5173"}
    with TestClient(api.app) as c:
        async def seed():
            p = await orch.create_project(api.CTX, CreateProjectRequest(task_prompt="pick up the cube"))
            ck = Checkpoint(project_id=p.id, training_job_id=None, policy_type="act", step=1, path="x")
            await api.CTX.repo.save("checkpoints", ck)
            return p.id, ck.id
        pid, ck = c.portal.call(seed)
        r = c.post(f"/api/projects/{pid}/real-trials", headers=studio,
                   json={"judged_by": "Ana", "trials": [{"checkpoint_id": ck, "success": False,
                                                          "failure_mode": "missed_grasp"}]})
        assert r.status_code == 200 and r.json()[0]["source"] == "studio"
        rep = c.get(f"/api/projects/{pid}/sim-real").json()
        assert rep["checkpoints"][0]["real"]["n"] == 1 and rep["checkpoints"][0]["sim"] is None
        tid = r.json()[0]["id"]
        assert c.post(f"/api/real-trials/{tid}/void").status_code == 403  # no studio origin
        assert c.post(f"/api/real-trials/{tid}/void", headers=studio).status_code == 200
        assert c.get(f"/api/projects/{pid}/real-trials").json() == []
    settings.cache_clear()
