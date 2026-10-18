"""The intake flow an MCP agent runs: start -> answer questions -> build sim -> look at it."""

import base64

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


def test_agent_intake_to_simulation(client):
    st = tool(client, "start_intake", prompt="Pick up the small water bottle and put it on the coaster")
    pid = st["project_id"]
    assert [o["name"] for o in st["spec"]["objects"]] == ["small_water_bottle"] and not st["complete"]

    # An agent that just accepts every suggested patch must end with a complete intake.
    for _ in range(8):
        open_q = [q for q in st["questions"] if q["required"]]
        if not open_q:
            break
        st = tool(client, "update_intake", project_id=pid, patch=open_q[0]["patch"])
    assert st["complete"], [q["path"] for q in st["questions"] if q["required"]]

    # Describe the surroundings, then build.
    st = tool(client, "update_intake", project_id=pid, patch={"environment": {
        "lighting": "bright", "description": "white desk",
        "obstacles": [{"name": "lamp_base", "shape": "cylinder", "size": [0.08, 0.03], "center": [0.12, 0.2]}]}})
    assert "environment.lighting" in st["changed"]
    r = client.post("/api/tools/capture_frame", json={"project_id": pid}, headers={"X-Cortex-Actor": "mcp"})
    assert r.status_code == 409  # not built yet

    out = tool(client, "build_simulation", project_id=pid)
    f = out["feasibility"]
    assert f["objects"][0]["grasp_mode"] == "side" and f["objects"][0]["grasp"]["ok"]
    assert base64.b64decode(out["image_b64"])[:2] == b"\xff\xd8"

    fr = tool(client, "capture_frame", project_id=pid, camera="front")
    assert fr["source"] == "sim" and fr["key"].endswith(".jpg")
    r = client.get(f"/api/projects/{pid}/simulation/frame?camera=top")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"

    # A description edit does not change the scene; a lighting change makes it stale.
    st = tool(client, "update_intake", project_id=pid, patch={"environment": {"description": "oak desk"}})
    assert st["scene_built"] and not st["scene_stale"]
    st = tool(client, "update_intake", project_id=pid, patch={"environment": {"lighting": "dim"}})
    assert st["scene_stale"] and any("build_simulation" in n for n in st["notes"])
    s = tool(client, "get_intake_status", project_id=pid)
    assert s["scene_built"] and s["scene_stale"] and s["scene_digest"]


def test_intake_errors_are_400_with_reason(client):
    pid = tool(client, "start_intake", prompt="cube")["project_id"]
    r = client.post("/api/tools/update_intake", json={"project_id": pid, "patch": {"task": {"pik": "cube"}}},
                    headers={"X-Cortex-Actor": "mcp"})
    assert r.status_code == 400 and "pik" in r.text
    r = client.post("/api/tools/build_simulation", json={"project_id": pid}, headers={"X-Cortex-Actor": "mcp"})
    assert r.status_code == 400 and "incomplete" in r.text


def test_patch_during_build_is_not_lost(client):
    import threading

    pid = tool(client, "start_intake", prompt="pick up the cube")["project_id"]
    tool(client, "update_intake", project_id=pid, patch={
        "objects": [{"name": "cube", "confirm": True}], "zones": [{"name": "a", "center": [0.22, 0.12]}],
        "task": {"place": ["a"]}})
    t = threading.Thread(target=lambda: tool(client, "build_simulation", project_id=pid))
    t.start()
    tool(client, "update_intake", project_id=pid, patch={"objects": [{"name": "cube", "mass_kg": 0.02}]})
    t.join()
    s = tool(client, "get_intake_status", project_id=pid)
    assert s["spec"]["objects"][0]["mass_kg"] == 0.02  # the patch survived the build
    assert s["scene_built"]  # either order: the build saw the patch, or it is marked stale


def test_scene_env_cache_is_bounded(client):
    from cortex.services import intake_service

    pid = tool(client, "start_intake", prompt="pick up the cube")["project_id"]
    tool(client, "update_intake", project_id=pid, patch={
        "objects": [{"name": "cube", "confirm": True}], "zones": [{"name": "a", "center": [0.22, 0.12]}],
        "task": {"place": ["a"]}})
    for light in ("dim", "normal", "bright"):
        tool(client, "update_intake", project_id=pid, patch={"environment": {"lighting": light}})
        tool(client, "build_simulation", project_id=pid)
    assert len(intake_service._envs) <= intake_service.MAX_ENVS
    r = client.get(f"/api/projects/{pid}/sim/frame")  # legacy preview, now on the GL thread
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
