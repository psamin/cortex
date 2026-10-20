"""#359: what the MCP schemas and approval errors promise matches the backend."""

import re
from pathlib import Path

from fastapi.testclient import TestClient

from cortex.services import agent_orchestrator as orch

ROOT = Path(__file__).resolve().parents[2]


def _schema_keys(tool: str) -> set[str]:
    src = (ROOT / "mcp" / "src" / "tools.ts").read_text(encoding="utf-8")
    block = src[src.index(f'name: "{tool}"'):]
    block = block[block.index("inputSchema: z.object({"):block.index("annotations:")]
    return set(re.findall(r"^      ([a-z_]+): ", block, re.M))


def test_mcp_schemas_carry_the_backend_arguments_agents_need():
    assert "base_checkpoint_id" in _schema_keys("propose_training_job")
    assert set(orch.Args.ProposeTraining.model_fields) >= {"base_checkpoint_id"}
    assert "seed" in _schema_keys("robot_start_episode")
    assert "seed" in orch.RobotArgs.Start.model_fields


def test_a_launch_refused_for_approval_names_the_jobs_project(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    for k in ("INSFORGE_BASE_URL", "INSFORGE_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    from cortex.api import app as api
    from cortex.config import settings
    from cortex.models import CreateProjectRequest, ResourceEstimate, TrainingJob

    settings.cache_clear()
    with TestClient(api.app) as c:
        async def seed():
            p = await orch.create_project(api.CTX, CreateProjectRequest(task_prompt="pick up the cube and place it"))
            tj = TrainingJob(project_id=p.id, dataset_version_id="ds_x", policy_type="act", backend="manual",
                             config={}, rationale="",
                             estimate=ResourceEstimate(gpu="x", gpu_hours=0.1, est_cost_usd=0, paid=False, notes=""))
            await api.CTX.repo.save("training_jobs", tj)
            return p, tj
        p, tj = c.portal.call(seed)
        r = c.post("/api/tools/launch_training_job", json={"training_job_id": tj.id}, headers={"X-Cortex-Actor": "mcp"})
        assert r.status_code == 403 and r.json()["approval_required"] and r.json()["project_id"] == p.id
    settings.cache_clear()
