"""MCP Apps views (#119): read-only snapshots, and training approval that only a human click in the
view can give (a single-use nonce the model never sees)."""

import asyncio

import pytest

from cortex.config import Settings
from cortex.context import build_ctx
from cortex.models import CreateProjectRequest, JobStatus, ResourceEstimate, TrainingJob
from cortex.services import agent_orchestrator as orch
from cortex.services import mcp_views, training_service


def run(coro):
    return asyncio.run(coro)


async def _setup(tmp_path):
    ctx = await build_ctx(Settings(data_dir=tmp_path, _env_file=None))
    p = await orch.create_project(ctx, CreateProjectRequest(task_prompt="pick up the cube and place it in the bowl"))
    tj = TrainingJob(project_id=p.id, dataset_version_id="ds_missing", policy_type="act", backend="manual",
                     config={"steps": 100}, rationale="test",
                     estimate=ResourceEstimate(gpu="x", gpu_hours=0.1, est_cost_usd=0.0, paid=False, notes=""))
    await ctx.repo.save("training_jobs", tj)
    return ctx, p, tj


def test_review_issues_a_nonce_only_while_awaiting_approval(tmp_path):
    async def go():
        ctx, p, tj = await _setup(tmp_path)
        out = await orch.call_tool(ctx, "review_training_job", {"project_id": p.id}, "mcp")
        assert out["training_job_id"] == tj.id and out["awaiting_approval"]
        assert out["view_approval_nonce"] and out["dataset"] is None
        tj.status = JobStatus.approved
        await ctx.repo.save("training_jobs", tj)
        out = await orch.call_tool(ctx, "review_training_job", {"project_id": p.id, "training_job_id": tj.id}, "mcp")
        assert "view_approval_nonce" not in out and not out["awaiting_approval"]
    run(go())


def test_view_nonce_approves_once_and_wrong_or_reused_nonces_are_refused(tmp_path):
    async def go():
        ctx, p, tj = await _setup(tmp_path)
        nonce = (await orch.call_tool(ctx, "review_training_job", {"project_id": p.id}, "mcp"))["view_approval_nonce"]
        args = {"project_id": p.id, "training_job_id": tj.id}
        with pytest.raises(training_service.ApprovalRequired):
            await orch.call_tool(ctx, "approve_training_from_view", {**args, "nonce": "guessed"}, "mcp")
        out = await orch.call_tool(ctx, "approve_training_from_view", {**args, "nonce": nonce}, "mcp")
        assert out["status"] == "approved"
        saved = await training_service._get(ctx, tj.id)
        assert saved.status == JobStatus.approved and "MCP App view" in saved.approved_by
        with pytest.raises(training_service.ApprovalRequired):
            await orch.call_tool(ctx, "approve_training_from_view", {**args, "nonce": nonce}, "mcp")
    run(go())


def test_a_nonce_is_bound_to_its_training_job_and_expires(tmp_path, monkeypatch):
    nonce = mcp_views.issue_nonce("tj_a")
    assert not mcp_views.redeem_nonce(nonce, "tj_b")  # consumed by the wrong job too
    assert not mcp_views.redeem_nonce(nonce, "tj_a")
    nonce = mcp_views.issue_nonce("tj_a")
    monkeypatch.setattr(mcp_views.time, "monotonic", lambda: 10**12)
    assert not mcp_views.redeem_nonce(nonce, "tj_a")


def test_mcp_without_a_nonce_still_cannot_approve(tmp_path):
    async def go():
        ctx, p, tj = await _setup(tmp_path)
        with pytest.raises(training_service.ApprovalRequired):
            await orch.call_tool(ctx, "launch_training_job", {"training_job_id": tj.id}, "mcp")
        assert (await training_service._get(ctx, tj.id)).status == JobStatus.proposed
    run(go())


def test_weakest_condition_only_when_branches_differ():
    b = lambda name, rate: {"name": name, "success_rate": rate}  # noqa: E731
    assert mcp_views.weakest_branch([]) is None
    assert mcp_views.weakest_branch([b("baseline", 1.0), b("shifted", 1.0)]) is None
    assert mcp_views.weakest_branch([b("baseline", 1.0), b("shifted", 0.4), b("dim", 0.6)]) == "shifted"


def test_snapshots_say_clearly_when_there_is_nothing_yet(tmp_path):
    async def go():
        ctx, p, _ = await _setup(tmp_path)
        with pytest.raises(KeyError, match="no collection"):
            await orch.call_tool(ctx, "collection_snapshot", {"project_id": p.id}, "mcp")
        with pytest.raises(KeyError, match="no evaluation"):
            await orch.call_tool(ctx, "evaluation_report", {"project_id": p.id}, "mcp")
    run(go())
