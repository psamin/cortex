"""Who may call which registry tool: `agent_callable=False` tools are refused for the studio agent
(actor "agent") before they run, while the GUI and MCP clients still reach them."""

import asyncio
import dataclasses
import json

import pytest

from cortex.config import Settings
from cortex.context import build_ctx
from cortex.models import CreateProjectRequest
from cortex.services import agent
from cortex.services import agent_orchestrator as orch

HUMAN_ONLY = sorted(t.name for t in orch.TOOLS.values() if not t.agent_callable)


def setup(tmp_path):
    async def go():
        ctx = await build_ctx(Settings(data_dir=tmp_path, _env_file=None))
        p = await orch.create_project(ctx, CreateProjectRequest(task_prompt="pick up the cube and place it in the bowl"))
        return ctx, p
    return asyncio.run(go())


@pytest.fixture
def stubbed(monkeypatch):
    """Replace every human-only tool's function with a recorder, so nothing real runs."""
    ran: list[tuple[str, str]] = []

    def stub(name):
        async def fn(ctx, a, actor):
            ran.append((name, actor))
            return {"summary": f"{name} ran", "ref_ids": []}
        return fn
    for name in HUMAN_ONLY:
        monkeypatch.setitem(orch.TOOLS, name, dataclasses.replace(orch.TOOLS[name], fn=stub(name)))
    return ran


def _args(name, pid):
    return {"approve_training_from_view": {"project_id": pid, "training_job_id": "tj_x", "nonce": "n"},
            "launch_training_job": {"training_job_id": "tj_x"},
            "robot_start_episode": {"project_id": pid},
            "robot_act": {"session_id": "rs_x", "tool": "observe_scene"},
            "robot_finish_episode": {"session_id": "rs_x", "believes_success": True},
            "verify_agent_key": {},
            "get_diagnosis_inputs": {"project_id": pid},
            "submit_failure_hypotheses": {"project_id": pid, "analysis_id": "an_x", "route": "mcp_agent", "model": "m",
                                          "answers": [{"ref_id": "x", "error": "e"}]}}[name]


def test_the_human_only_set_is_what_the_policy_says():
    assert HUMAN_ONLY == ["approve_training_from_view", "get_diagnosis_inputs", "launch_training_job", "robot_act",
                          "robot_finish_episode", "robot_start_episode", "submit_failure_hypotheses",
                          "verify_agent_key"]
    assert not {s["function"]["name"] for s in agent.tool_specs()} & set(HUMAN_ONLY)


@pytest.mark.parametrize("name", HUMAN_ONLY)
def test_studio_agent_is_refused_before_the_tool_runs_and_the_refusal_is_audited(tmp_path, stubbed, name):
    ctx, p = setup(tmp_path)

    async def go():
        with pytest.raises(orch.NotAgentCallable, match=name) as e:
            await orch.call_tool(ctx, name, _args(name, p.id), "agent")
        assert e.value.code == "not_agent_callable" and e.value.approval_required is False
        return await ctx.store.list("tool_audit_logs", project_id=p.id)
    audit = asyncio.run(go())
    assert stubbed == []
    if name in ("robot_start_episode", "get_diagnosis_inputs"):  # these carry a project id
        row = audit[-1]
        assert (row["status"], row["data"]["actor"], row["data"]["tool"]) == ("refused", "agent", name)


def test_refusal_does_not_depend_on_valid_arguments(tmp_path, stubbed):
    ctx, _ = setup(tmp_path)
    with pytest.raises(orch.NotAgentCallable):
        asyncio.run(orch.call_tool(ctx, "launch_training_job", {}, "agent"))


@pytest.mark.parametrize("actor", ["gui", "mcp", "autopilot"])
@pytest.mark.parametrize("name", HUMAN_ONLY)
def test_other_actors_still_reach_human_only_tools(tmp_path, stubbed, actor, name):
    ctx, p = setup(tmp_path)
    out = asyncio.run(orch.call_tool(ctx, name, _args(name, p.id), actor))
    assert out["summary"] == f"{name} ran" and stubbed == [(name, actor)]


def test_studio_llm_cannot_reach_a_hidden_tool_by_naming_it(tmp_path, stubbed):
    """The model only sees agent-callable specs, but nothing stops it from emitting another name."""
    ctx, p = setup(tmp_path)
    with pytest.raises(orch.NotAgentCallable):
        asyncio.run(agent.run_tool(ctx, p.id, "launch_training_job", {"training_job_id": "tj_x"}))
    assert stubbed == []


def test_api_answers_403_without_an_approval_card():
    from cortex.api.app import app

    handler = app.exception_handlers[PermissionError]
    res = asyncio.run(handler(None, orch.NotAgentCallable("robot_act is not available to the studio agent")))
    body = json.loads(res.body)
    assert res.status_code == 403
    assert body["code"] == "not_agent_callable" and body["approval_required"] is False
