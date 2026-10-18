"""RL sandboxes over the tool registry (REST/MCP): weaknesses -> propose (real pilot) -> accept -> run -> status.

The full flow runs the real pinned model in worker subprocesses with a tiny budget (~3-8 minutes)."""

import time

import pytest
from fastapi.testclient import TestClient

from cortex.models import (
    BranchResult,
    BranchSpec,
    Condition,
    EpisodeResult,
    EvaluationRun,
    PolicyRef,
    SimulationBranch,
)
from cortex.rl import RLSandbox, SandboxBudget, propose
from cortex.rl.types import Cost
from cortex.services.simulation_runner import wilson
from cortex.sim.so101 import configure_gl

configure_gl()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    for k in ("INSFORGE_BASE_URL", "INSFORGE_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    from cortex.config import settings

    settings.cache_clear()
    from cortex.api.app import app

    with TestClient(app) as c:
        yield c
    settings.cache_clear()


def ctx():
    from cortex.api import app as appmod

    return appmod.CTX


def tool(c, name, status=200, **args):
    r = c.post(f"/api/tools/{name}", json=args, headers={"X-Cortex-Actor": "mcp"})
    assert r.status_code == status, r.text
    return r.json()


def project_with_weak_checkpoint(c) -> str:
    """A project whose learned checkpoint was evaluated in sim: weak on a shifted object (3/10, grasp failures)."""
    pid = tool(c, "create_project", task_prompt="Pick up the red cube and place it in the green box")["project"]["id"]
    run = EvaluationRun(project_id=pid, policy=PolicyRef(kind="checkpoint", checkpoint_id="ck_1"))
    branches = []
    for name, succ, offset in (("baseline", 9, (0.0, 0.0)), ("shifted_object", 3, (0.02, 0.0))):
        eps = [EpisodeResult(seed=i, success=True, reward=1.0, steps=200) for i in range(succ)]
        eps += [EpisodeResult(seed=100 + i, success=False, reward=0.2, steps=300, failure_mode="grasp_failed")
                for i in range(10 - succ)]
        res = BranchResult(n=10, successes=succ, success_rate=succ / 10, ci95=wilson(succ, 10), mean_reward=0.5,
                           episodes=eps)
        branches.append(SimulationBranch(evaluation_id=run.id, spec=BranchSpec(name=name, condition=Condition(),
                                                                                 cube_offset=offset),
                                         seeds=[e.seed for e in eps], result=res))
    run.branch_ids = [b.id for b in branches]

    async def save():
        await ctx().repo.save("evaluation_runs", run)
        for b in branches:
            await ctx().repo.save("simulation_branches", b, project_id=pid)
    c.portal.call(save)
    return pid


def save_sandbox(c, sb: RLSandbox) -> None:
    async def save():
        await ctx().repo.save("rl_sandboxes", sb)
    c.portal.call(save)


def test_weaknesses_and_approval_gates(client):
    pid = project_with_weak_checkpoint(client)
    ws = tool(client, "get_policy_weaknesses", project_id=pid)
    assert [w["key"] for w in ws["weaknesses"]] == ["shifted_object"]  # baseline 9/10 passes the 0.8 target
    w = ws["weaknesses"][0]
    assert w["dominant_failure"] == "grasp_failed" and w["n"] == 10 and w["ci95"][0] < 0.3 < w["ci95"][1]

    from cortex.rl.weakness import find_weaknesses
    from cortex.services.simulation_runner import list_evaluations

    weak = find_weaknesses(client.portal.call(list_evaluations, ctx(), pid), [])
    gpu, paid, idle = propose(pid, weak * 3, SandboxBudget(executor="futurama_gpu"))
    paid.budget.executor = "local_cpu"
    paid.estimate.cost = Cost(episodes=10, cpu_hours=0.1, wall_clock_est_min=3, usd=1.5, sim_steps_per_s=300)
    idle.budget.executor = "local_cpu"
    for sb in (gpu, paid, idle):
        sb.status = "ready"
        save_sandbox(client, sb)

    r = tool(client, "accept_rl_sandboxes", 403, project_id=pid, ids=[gpu.id], budget_cap=0)
    assert r["approval_required"] and "human" in r["detail"]  # GPU work is never agent-approved
    r = tool(client, "accept_rl_sandboxes", 403, project_id=pid, ids=[paid.id], budget_cap=1.0)
    assert r["code"] == "rl_budget"  # over the cap the agent gave
    tool(client, "accept_rl_sandboxes", 404, project_id=pid, ids=["rl_missing"], budget_cap=0)

    st = tool(client, "stop_rl_sandbox", sandbox_id=idle.id)
    assert st["sandbox"]["status"] == "dismissed"
    r = tool(client, "accept_rl_sandboxes", 400, project_id=pid, ids=[idle.id], budget_cap=0)
    assert "only sandboxes with a finished pilot" in r["detail"]


def test_propose_accept_run_status_end_to_end(client):
    pid = project_with_weak_checkpoint(client)
    events: list[dict] = []
    bus = ctx().bus
    orig = bus.publish
    bus.publish = lambda p, kind, data: (events.append({"kind": kind, **data}) if kind == "rl" else None,
                                         orig(p, kind, data))[1]

    budget = {"iterations": 2, "population": 4, "episodes_per_candidate": 1, "pilot_episodes": 4,
              "pilot_cem_iterations": 1, "heldout_episodes": 5, "demos": 0, "max_parallel": 1}
    out = tool(client, "propose_rl_sandboxes", project_id=pid, max_sandboxes=1, budget=budget, wait_s=300)
    sb = out["sandboxes"][0]
    assert sb["status"] == "ready", sb
    assert sb["weakness_id"] == "shifted_object" and sb["method"] == "cem_skill" and sb["kind"] == "estimate"
    assert sb["pilot"]["n"] == 4 and sb["pilot"]["cem_iterations"] == 1
    for k in ("predicted_sandbox_success", "predicted_policy_uplift", "predicted_regression_elsewhere"):
        assert set(sb[k]) == {"mean", "ci90", "source"}
    assert sb["predicted_sandbox_success"]["source"] == "measured" and sb["predicted_policy_uplift"]["source"] == "prior"
    assert sb["cost"]["usd"] == 0 and sb["cost"]["cpu_hours"] > 0 and sb["cost"]["sim_steps_per_s"] > 0
    assert 0 <= sb["p_improve_ge_delta"] <= 1 and sb["assumptions"]

    acc = tool(client, "accept_rl_sandboxes", project_id=pid, ids=[sb["id"]], budget_cap=0)
    assert acc["sandboxes"][0]["status"] == "running"
    deadline = time.time() + 1800  # minutes on an idle machine; much longer when other sims share the CPU
    while time.time() < deadline:
        st = tool(client, "get_rl_sandbox_status", sandbox_id=sb["id"])["sandboxes"][0]
        if st["status"] not in ("accepted", "running"):
            break
        time.sleep(3)
    assert st["status"] == "succeeded", st
    assert st["iteration"] == 2 and len(st["curve"]) == 2 and st["elite_params"] and st["episodes"] >= 18
    res = st["result"]
    assert res["kind"] == "measured" and len(res["heldout_seeds"]) == 5 and 0 <= res["p_value"] <= 1
    assert st["current_ranges"]["offset_jitter_m"] >= 0.02 and st["cost_so_far"]["usd"] == 0

    listed = tool(client, "get_rl_sandbox_status", project_id=pid)["sandboxes"]
    assert [s["id"] for s in listed] == [sb["id"]]

    from cortex.services.rl_service import RLRound

    rounds = client.portal.call(ctx().repo.list, "rl_rounds", RLRound, pid)
    assert len(rounds) == 1 and rounds[0].realised_uplift is None  # filled in once ACT is re-evaluated

    kinds = {e["type"] for e in events}
    assert {"rl_phase", "rl_episode", "rl_iteration", "rl_pilot", "rl_result", "sim_stats"} <= kinds
    assert all(e["sandbox_id"] == sb["id"] for e in events)
