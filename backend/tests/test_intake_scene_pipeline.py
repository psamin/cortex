"""P5b: a project with a built intake scene records and evaluates in that scene, not the cube tabletop."""

import pytest
from fastapi.testclient import TestClient

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


def tool(c, name, **args):
    r = c.post(f"/api/tools/{name}", json=args, headers={"X-Cortex-Actor": "mcp"})
    assert r.status_code == 200, r.text
    return r.json()


def finished(c, out, deadline_s=1800):
    """Poll a job to a terminal state. Rendered side-grasp demos are long; a loaded CPU can take many minutes."""
    import time

    job = out.get("job") or out["evaluation"]["run"]
    end = time.monotonic() + deadline_s
    while job["status"] in ("queued", "running") and time.monotonic() < end:
        time.sleep(10)
        job = tool(c, "get_job", job_id=out["job"]["id"])["job"]
    return job


def bottle_project(c):
    pid = tool(c, "start_intake", prompt="pick up the small water bottle and put it on the mat")["project_id"]
    tool(c, "update_intake", project_id=pid, patch={
        "objects": [{"name": "small_water_bottle", "confirm": True}],
        "zones": [{"name": "mat", "center": [0.30, -0.13]}], "task": {"place": ["mat"]}})
    out = tool(c, "build_simulation", project_id=pid)
    assert out["feasibility"]["objects"][0]["grasp_mode"] == "side"
    return pid


def test_oracle_demo_and_evaluation_run_in_the_intake_scene(client):
    pid = bottle_project(client)
    rec = tool(client, "record_demonstrations", project_id=pid,
               conditions=[{"cube_region": "center", "lighting": "normal", "count": 1}], wait_s=600)
    assert finished(client, rec)["status"] == "succeeded", rec
    eps = tool(client, "list_episodes", project_id=pid)["episodes"]
    assert len(eps) == 1 and eps[0]["success"], eps
    assert eps[0]["frames"] > 300  # a side-grasp cycle, not the 240-step cube oracle

    ev = tool(client, "launch_simulation_evaluation", project_id=pid, policy="scripted_expert",
              episodes_per_branch=1, wait_s=900)
    assert finished(client, ev)["status"] == "succeeded", ev
    runs = client.get(f"/api/projects/{pid}/evaluations").json()
    assert runs, ev
    base = next(b for b in runs[-1]["branches"] if b["spec"]["name"] == "baseline")
    assert base["result"]["n"] == 1 and base["result"]["successes"] == 1, base


def test_stale_scene_is_refused_not_silently_used(client):
    pid = bottle_project(client)
    tool(client, "update_intake", project_id=pid, patch={"environment": {"lighting": "dim"}})
    r = client.post("/api/tools/record_demonstrations", headers={"X-Cortex-Actor": "mcp"},
                    json={"project_id": pid, "conditions": [{"cube_region": "center", "lighting": "normal",
                                                             "count": 1}], "wait_s": 30})
    body = r.json()
    assert r.status_code == 400 or "build_simulation" in str(body), body
