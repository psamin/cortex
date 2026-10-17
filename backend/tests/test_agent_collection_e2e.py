"""Integration: REST/MCP tool -> collection service -> agent worker subprocess ->
planner loop (against the local FakePlanner test double, never GPT-6 Astra) ->
MuJoCo + IK -> attempts/episodes in the store -> LeRobot dataset.
Takes ~1 minute (real physics, subprocesses)."""

import os
import time

import pytest
from fastapi.testclient import TestClient

from fake_planner import FakePlanner


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    data = tmp_path_factory.mktemp("cortex-data")
    with FakePlanner(sabotage={1}) as fp:
        env = {"DATA_DIR": str(data), "OPENROUTER_API_KEY": "test-key", "CORTEX_PLANNER_BASE_URL": fp.url,
               "INSFORGE_BASE_URL": "", "INSFORGE_API_KEY": ""}
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
    assert pl["available"] and pl["model"] == "openai/gpt-6-astra" and pl["provider"] == "openrouter"

    p = c.post("/api/projects", json={"task_prompt": "Train my SO-101 to pick up a cube and put it in a bowl"}).json()
    col = c.post(f"/api/projects/{p['id']}/agent-collections",
                 json={"num_demos": 3, "diversity": "medium", "max_usd_per_episode": 1.0}).json()
    assert col["collector"] == "astra" and col["model"] == "openai/gpt-6-astra" and len(col["scenarios"]) == 3
    r = wait(c, col["id"])
    att = r["attempts"]
    assert len(att) == 3 and r["collection"]["status"] == "succeeded"
    by_idx = {a["index"]: a for a in att}
    assert by_idx[1]["status"] == "failed"  # sabotaged attempt: model pointed 70 px off
    ok = [a for a in att if a["status"] == "success"]
    assert len(ok) == 2, [a.get("error") or a.get("stop_reason") for a in att]
    for a in att:
        assert a["provider"] == "OpenRouter (OPENROUTER_API_KEY)" and a["endpoint"].startswith("http://127.0.0.1")
        assert {"trace", "episode", "front"} <= set(a["artifacts"])
    # failures are diagnostics only: no Episode row
    eps = c.get(f"/api/projects/{p['id']}/episodes").json()
    assert {e["provenance"]["attempt_id"] for e in eps} == {a["id"] for a in ok}
    assert all(e["source"] == "sim_agent" and e["provenance"]["planner"] == "gpt-6-astra" for e in eps)
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
    assert comp["astra"]["attempts"] == 3 and comp["oracle"]["attempts"] == 1


def test_astra_is_never_substituted(client, monkeypatch):
    from cortex.api import app as appmod
    ctx = appmod.ctx()
    saved = ctx.openrouter_key
    ctx.openrouter_key = None
    try:
        p = client.post("/api/projects", json={"task_prompt": "pick up the cube"}).json()
        r = client.post(f"/api/projects/{p['id']}/agent-collections", json={"num_demos": 1})
        assert r.status_code == 503 and "does not substitute" in r.json()["detail"]
    finally:
        ctx.openrouter_key = saved


def test_budget_gate_blocks_mcp_overspend(client):
    p = client.post("/api/projects", json={"task_prompt": "pick up the cube"}).json()
    r = client.post("/api/tools/start_agent_collection", headers={"X-Cortex-Actor": "mcp"},
                    json={"project_id": p["id"], "num_demos": 1, "max_total_usd": 500})
    assert r.status_code == 403 and "budget" in r.json()["detail"].lower()
