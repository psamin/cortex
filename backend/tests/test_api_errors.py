"""HTTP mapping of errors, and stopping collections that already finished."""

import pytest
from fastapi.testclient import TestClient

from cortex.models import AgentCollection, StartAgentCollectionRequest


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    for k in ("INSFORGE_BASE_URL", "INSFORGE_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    from cortex.config import settings

    settings.cache_clear()
    from cortex.api.app import app

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    settings.cache_clear()


def test_missing_entity_is_404_but_a_stray_key_error_is_a_500(client, monkeypatch):
    r = client.get("/api/projects/prj_nope")
    assert r.status_code == 404 and "prj_nope" in r.json()["detail"]

    from cortex.services import agent_orchestrator as orch

    async def buggy(ctx, pid):
        return {}["status"]  # a bug, not a missing project

    monkeypatch.setattr(orch, "get_project", buggy)
    r = client.get("/api/projects/prj_any")
    assert r.status_code == 500 and r.json()["code"] == "internal_error"


@pytest.mark.parametrize("status", ["succeeded", "failed", "budget_exhausted", "cancelled"])
def test_stop_keeps_the_outcome_of_a_finished_collection(client, status):
    from cortex.api import app as app_mod

    col = AgentCollection(project_id="prj_1", collector="agent", request=StartAgentCollectionRequest(),
                          status=status)
    client.portal.call(app_mod.CTX.repo.save, "agent_collections", col)
    r = client.post(f"/api/agent-collections/{col.id}/stop")
    assert r.status_code == 200 and r.json()["status"] == status


def test_stop_cancels_a_running_collection(client):
    from cortex.api import app as app_mod

    col = AgentCollection(project_id="prj_1", collector="agent", request=StartAgentCollectionRequest(),
                          status="running")
    client.portal.call(app_mod.CTX.repo.save, "agent_collections", col)
    assert client.post(f"/api/agent-collections/{col.id}/stop").json()["status"] == "cancelled"
