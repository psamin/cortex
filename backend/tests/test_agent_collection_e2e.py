"""Integration: REST/MCP tool -> collection service -> agent worker subprocess ->
planner loop (against the local FakePlanner test double on the self-hosted route) ->
MuJoCo + IK -> attempts/episodes in the store -> LeRobot dataset.
Takes ~1 minute (real physics, subprocesses)."""

import os
import time

import pytest
from fastapi.testclient import TestClient

from fake_planner import FakePlanner

MODEL = "test/fake-planner"  # FakePlanner's default model id


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    data = tmp_path_factory.mktemp("cortex-data")
    with FakePlanner(sabotage={1}) as fp:
        env = {"DATA_DIR": str(data), "PLANNER_BASE_URL": fp.base, "SELFHOSTED_MODEL": fp.model_id,
               "SELFHOSTED_POINT_FORMAT": "pixels", "CORTEX_PLANNER_BASE_URL": "",
               "INSFORGE_BASE_URL": "", "INSFORGE_API_KEY": "", "AGENT_ENGINE": "model"}
        old = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        from cortex.config import settings
        settings.cache_clear()
        from cortex.api.app import app
        with TestClient(app) as c:
            yield c
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        settings.cache_clear()


def wait(c, cid, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = c.get(f"/api/agent-collections/{cid}").json()
        if r["collection"]["status"] not in ("queued", "running"):
            return r
        time.sleep(1)
    raise TimeoutError(cid)


def test_agent_collection_end_to_end(client):
    c = client
    pl = c.get("/api/planner").json()
    assert pl["available"] and pl["model"] == MODEL and pl["provider"] == "selfhosted"

    p = c.post("/api/projects", json={"task_prompt": "Train my SO-101 to pick up a cube and put it in a bowl"}).json()
    col = c.post(f"/api/projects/{p['id']}/agent-collections",
                 json={"num_demos": 3, "diversity": "medium", "max_usd_per_episode": 1.0}).json()
    assert col["collector"] == "agent" and col["model"] == MODEL and len(col["scenarios"]) == 3
    r = wait(c, col["id"])
    att = r["attempts"]
    assert len(att) == 3 and r["collection"]["status"] == "succeeded"
    by_idx = {a["index"]: a for a in att}
    assert by_idx[1]["status"] == "failed"  # sabotaged attempt: model pointed 70 px off
    ok = [a for a in att if a["status"] == "success"]
    assert len(ok) == 2, [a.get("error") or a.get("stop_reason") for a in att]
    for a in att:
        assert a["provider"].startswith("self-hosted") and a["endpoint"].startswith("http://127.0.0.1")
        assert {"trace", "episode", "front"} <= set(a["artifacts"])
    # failures are diagnostics only: no Episode row
    eps = c.get(f"/api/projects/{p['id']}/episodes").json()
    assert {e["provenance"]["attempt_id"] for e in eps} == {a["id"] for a in ok}
    assert all(e["source"] == "sim_agent" and e["provenance"]["planner"] == MODEL for e in eps)
    # trace is inspectable and phases come from executed tools
    tr = c.get(f"/api/agent-attempts/{ok[0]['id']}").json()
    phases = [s["phase"] for s in tr["trace"]]
    assert "Selecting grasp" in phases and "Closing gripper" in phases and "Verifying task success" in phases

    # same service through the MCP tool endpoint (oracle collector on matched-style scenes)
    m = c.post("/api/tools/start_agent_collection", headers={"X-Cortex-Actor": "mcp"},
               json={"project_id": p["id"], "num_demos": 1, "collector": "oracle"}).json()
    assert m["job_id"]
    r2 = wait(c, m["collection_id"])
    assert r2["attempts"][0]["collector"] == "oracle"
    audit = c.get(f"/api/projects/{p['id']}/audit").json()
    assert any(x["actor"] == "mcp" and x["tool"] == "start_agent_collection" for x in audit)

    # dataset from agent demos only: valid LeRobot v3, failed attempt excluded, oracle excluded
    j = c.post(f"/api/projects/{p['id']}/datasets", json={"sources": ["sim_agent"], "eval_fraction": 0.0}).json()
    t0 = time.time()
    while c.get(f"/api/jobs/{j['id']}").json()["status"] in ("queued", "running") and time.time() - t0 < 300:
        time.sleep(1)
    ds = c.get(f"/api/projects/{p['id']}/datasets").json()[-1]
    assert {ln["episode_id"] for ln in ds["links"]} == {a["episode_id"] for a in ok}
    from lerobot.datasets import LeRobotDataset

    from cortex.config import settings
    d = LeRobotDataset(f"cortex/{p['id']}_{ds['id']}", root=settings().data_dir / ds["lerobot_root"])
    assert d.num_episodes == 2 and d[0]["action"].shape == (6,)

    comp = c.get(f"/api/projects/{p['id']}/collector-comparison").json()
    assert comp["agent"]["attempts"] == 3 and comp["oracle"]["attempts"] == 1


def test_astra_is_never_substituted(client, monkeypatch):
    from cortex.api import app as appmod
    ctx = appmod.ctx()
    saved = ctx.model
    ctx.model = None
    try:
        p = client.post("/api/projects", json={"task_prompt": "pick up the cube"}).json()
        r = client.post(f"/api/projects/{p['id']}/agent-collections", json={"num_demos": 1})
        assert r.status_code == 503 and "does not substitute" in r.json()["detail"]
    finally:
        ctx.model = saved


async def test_budget_gate_blocks_overspend(client):
    """A priced planner route is capped by the project's standing model budget."""
    from cortex.api import app as appmod
    from cortex.models import Project, StartAgentCollectionRequest
    from cortex.services import agent_collection_service as acs

    ctx = appmod.ctx()
    p = client.post("/api/projects", json={"task_prompt": "pick up the cube"}).json()
    project = await ctx.repo.get("projects", p["id"], Project)
    with pytest.raises(acs.PlannerBudgetExceeded, match="budget"):
        await acs.check_budget(ctx, project, StartAgentCollectionRequest(num_demos=1, max_total_usd=500))


def test_selfhosted_qwen_route_converts_norm1000_points(tmp_path):
    """Our own model behind an OpenAI-compatible endpoint (the vLLM route), answering in
    Qwen3-VL's relative 0-1000 coordinates: reachability check, point conversion, plain
    OpenAI-compatible request fields, zero per-token cost, and the real model id in provenance."""
    model = "Qwen/Qwen3-VL-8B-Instruct"
    with FakePlanner(point_format="norm1000", model_id=model) as fp:
        env = {"DATA_DIR": str(tmp_path), "PLANNER_BASE_URL": fp.base, "PLANNER_API_KEY": "local",
               "INSFORGE_BASE_URL": "", "INSFORGE_API_KEY": "",
               "CORTEX_PLANNER_BASE_URL": "", "AGENT_ENGINE": "model"}
        old = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        from cortex.config import settings
        settings.cache_clear()
        from cortex.api.app import app
        try:
            with TestClient(app) as c:
                pl = c.get("/api/planner").json()
                assert pl["available"] and pl["provider"] == "selfhosted" and pl["model"] == model
                assert pl["point_format"] == "norm1000"
                p = c.post("/api/projects", json={"task_prompt": "pick up the cube and put it in the bowl"}).json()
                col = c.post(f"/api/projects/{p['id']}/agent-collections", json={"num_demos": 1}).json()
                r = wait(c, col["id"])
                a = r["attempts"][0]
                assert a["status"] == "success", a.get("error") or a.get("stop_reason")
                assert a["usage"]["cost_usd"] == 0 and a["usage"]["cost_source"].startswith("self-hosted")
                assert a["model"] == model and a["endpoint"] == fp.url
                tr = c.get(f"/api/agent-attempts/{a['id']}").json()["trace"]
                g = next(s for s in tr if s["tool"] == "propose_grasp")
                assert g["args"]["image_point"]["u"] <= 1000 and "args_executed" in g  # model units -> pixels
                assert g["args_executed"]["image_point"]["u"] == pytest.approx(g["args"]["image_point"]["u"] * 0.48)
                ep = c.get(f"/api/projects/{p['id']}/episodes").json()[0]
                assert ep["provenance"]["planner"] == model
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            settings.cache_clear()
    assert all("reasoning" not in b and "usage" not in b and b.get("temperature") == 0.0 for b in fp.seen_bodies)
