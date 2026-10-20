"""#354: approving training needs the studio in a browser (or the MCP server's own channel for view approvals),
not just a request without agent headers."""

import pytest
from fastapi.testclient import TestClient

from cortex.api import app as api
from cortex.config import Settings
from cortex.context import build_ctx
from cortex.models import CreateProjectRequest, JobStatus, ResourceEstimate, TrainingJob
from cortex.services import agent_orchestrator as orch

STUDIO = {"Origin": "http://localhost:5173", "X-Cortex-Actor": "gui"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    for k in ("INSFORGE_BASE_URL", "INSFORGE_API_KEY", "OPENROUTER_API_KEY", "CORTEX_MCP_KEY"):
        monkeypatch.delenv(k, raising=False)
    from cortex.config import settings

    settings.cache_clear()
    with TestClient(api.app) as c:
        async def seed():
            p = await orch.create_project(api.CTX, CreateProjectRequest(task_prompt="pick up the cube and place it"))
            tj = TrainingJob(project_id=p.id, dataset_version_id="ds_x", policy_type="act", backend="manual",
                             config={"steps": 100}, rationale="test",
                             estimate=ResourceEstimate(gpu="x", gpu_hours=0.1, est_cost_usd=0, paid=False, notes=""))
            await api.CTX.repo.save("training_jobs", tj)
            return p, tj
        p, tj = c.portal.call(seed)
        c.status = lambda tid: c.portal.call(api.CTX.repo.get, "training_jobs", tid, TrainingJob).status
        yield c, settings(), p, tj
    settings.cache_clear()


def test_a_bare_request_cannot_approve(client):
    c, _, _, tj = client
    for headers in ({}, {"X-Cortex-Actor": "gui"}, {"Origin": "http://evil.example", "X-Cortex-Actor": "gui"}):
        r = c.post(f"/api/training/{tj.id}/approve", json={}, headers=headers)
        assert r.status_code == 403, headers
    assert c.status(tj.id) == JobStatus.proposed


def test_the_studio_in_a_browser_approves(client):
    c, _, _, tj = client
    r = c.post(f"/api/training/{tj.id}/approve", json={"approver": "ana"}, headers=STUDIO)
    assert r.status_code == 200 and r.json()["status"] == "approved"
    # the backend's own origin (a studio build it serves) counts too
    r = c.post(f"/api/training/{tj.id}/approve", json={}, headers={"Origin": "http://testserver"})
    assert r.status_code == 200


def test_rest_clients_never_get_the_view_nonce_and_cannot_redeem_one(client):
    c, _, p, tj = client
    r = c.post("/api/tools/review_training_job", json={"project_id": p.id}, headers={"X-Cortex-Actor": "mcp"})
    assert r.status_code == 200 and r.json()["awaiting_approval"] and "view_approval_nonce" not in r.json()
    r = c.post("/api/tools/approve_training_from_view",
               json={"project_id": p.id, "training_job_id": tj.id, "nonce": "x"}, headers={"X-Cortex-Actor": "mcp"})
    assert r.status_code == 403 and c.status(tj.id) == JobStatus.proposed


def test_the_mcp_server_channel_gets_and_redeems_the_nonce(client):
    c, _, p, tj = client
    mcp = {"X-Cortex-Actor": "mcp", "X-Cortex-View-Channel": "1"}
    nonce = c.post("/api/tools/review_training_job", json={"project_id": p.id}, headers=mcp).json()["view_approval_nonce"]
    r = c.post("/api/tools/approve_training_from_view",
               json={"project_id": p.id, "training_job_id": tj.id, "nonce": nonce}, headers=mcp)
    assert r.status_code == 200 and c.status(tj.id) != JobStatus.proposed


def test_with_an_mcp_key_the_channel_header_alone_is_not_enough(client):
    c, s, p, tj = client
    s.cortex_mcp_key = "k" * 24
    mcp = {"X-Cortex-Actor": "mcp", "X-Cortex-View-Channel": "1"}
    assert "view_approval_nonce" not in c.post("/api/tools/review_training_job", json={"project_id": p.id},
                                               headers=mcp).json()
    keyed = {**mcp, "Authorization": f"Bearer {s.cortex_mcp_key}"}
    assert c.post("/api/tools/review_training_job", json={"project_id": p.id}, headers=keyed).json()[
        "view_approval_nonce"]


async def test_the_studio_agent_engine_never_gets_a_nonce(tmp_path):
    ctx = await build_ctx(Settings(data_dir=tmp_path, _env_file=None))
    p = await orch.create_project(ctx, CreateProjectRequest(task_prompt="pick up the cube and place it"))
    tj = TrainingJob(project_id=p.id, dataset_version_id="ds_x", policy_type="act", backend="manual",
                     config={"steps": 100}, rationale="test",
                     estimate=ResourceEstimate(gpu="x", gpu_hours=0.1, est_cost_usd=0, paid=False, notes=""))
    await ctx.repo.save("training_jobs", tj)
    out = await orch.call_tool(ctx, "review_training_job", {"project_id": p.id}, "agent")
    assert out["awaiting_approval"] and "view_approval_nonce" not in out
