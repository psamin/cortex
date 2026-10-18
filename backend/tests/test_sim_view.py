"""The stylized viewer's routes: scene description, STL meshes, and body poses of a built scene."""

import struct

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


def built_project(c) -> str:
    st = tool(c, "start_intake", prompt="Pick up the small water bottle and put it on the coaster")
    pid = st["project_id"]
    for _ in range(8):
        open_q = [q for q in st["questions"] if q["required"]]
        if not open_q:
            break
        st = tool(c, "update_intake", project_id=pid, patch=open_q[0]["patch"])
    assert st["complete"]
    tool(c, "update_intake", project_id=pid, patch={"environment": {
        "obstacles": [{"name": "lamp_base", "shape": "cylinder", "size": [0.08, 0.03], "center": [0.12, 0.2]}]}})
    return pid


def test_upstream_model_and_meshes(client):
    r = client.get("/api/sim/model")
    assert r.status_code == 200, r.text
    m = r.json()
    assert m["scene"] == "so101_tabletop"
    assert [j["label"] for j in m["joints"]][:2] == ["J01 / SHOULDER_PAN", "J02 / SHOULDER_LIFT"]
    roles = {x["role"] for x in m["meshes"]}
    assert roles == {"arm", "motor"}
    bodies = {x["body"] for x in m["meshes"]}
    assert {"base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper", "moving_jaw_so101_v1"} <= bodies
    for x in m["meshes"]:
        assert len(x["pos"]) == 3 and len(x["quat"]) == 4
        assert abs(sum(v * v for v in x["quat"]) - 1) < 1e-4
    assert any(p["role"] == "table" and p["type"] == "box" for p in m["primitives"])
    assert m["tcp"]["body"] == "gripper"

    # Every mesh the model names is served, as a binary STL whose triangle count matches its size.
    names = {x["mesh"] for x in m["meshes"]}
    r = client.get(f"/api/sim/meshes/{sorted(names)[0]}.stl")
    assert r.status_code == 200 and r.headers["content-type"].startswith("model/stl")
    (n,) = struct.unpack("<I", r.content[80:84])
    assert n > 0 and len(r.content) == 84 + 50 * n
    assert "immutable" in r.headers["cache-control"]

    # Names outside the asset list are 404s, never a filesystem path.
    for bad in ("nope", "..%2F..%2Fso101_tabletop", "%2E%2E%2Fso101"):
        assert client.get(f"/api/sim/meshes/{bad}.stl").status_code == 404


def test_project_model_and_state(client):
    pid = built_project(client)
    assert client.get(f"/api/sim/model?pid={pid}").status_code == 409  # not built yet
    # Before a build, the state is the upstream tabletop under a plan condition (the legacy preview).
    up = client.get(f"/api/projects/{pid}/simulation/state?cube_region=left").json()
    assert up["scene"] == "so101_tabletop" and up["pick_body"] == "red_cube"
    assert up["bodies"]["red_cube"]["pos"][1] > 0.0  # "left" is +y in the robot frame
    assert client.get(f"/api/projects/{pid}/simulation/state?cube_region=moon").status_code == 400

    out = tool(client, "build_simulation", project_id=pid)
    digest = out["scene"]["digest"]

    m = client.get(f"/api/sim/model?pid={pid}").json()
    assert m["scene"] == digest and m["project_id"] == pid
    roles = {p["role"] for p in m["primitives"]}
    assert {"table", "object", "zone", "obstacle", "parked"} <= roles
    obj = next(p for p in m["primitives"] if p["role"] == "object")
    assert obj["body"] == "obj_small_water_bottle" and obj["type"] == "cylinder"
    zone = next(p for p in m["primitives"] if p["role"] == "zone")
    assert zone["name"].startswith("zone_") and zone["body"] == "world"

    s = client.get(f"/api/projects/{pid}/simulation/state").json()
    assert s["scene"] == digest and s["source"] == "reset" and s["pick_body"] == "obj_small_water_bottle"
    assert set(s["joints_deg"]) == {"shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll",
                                    "gripper"}
    assert all(abs(v) < 200 for v in s["joints_deg"].values())
    # Every body the model places has a pose; the object rests on the table near its nominal start.
    assert {x["body"] for x in m["meshes"]} <= set(s["bodies"])
    bottle = s["bodies"]["obj_small_water_bottle"]
    assert len(bottle["quat"]) == 4 and 0 < bottle["pos"][2] < 0.2
    start = out["scene"]["starts"]["small_water_bottle"]
    assert abs(bottle["pos"][0] - start[0]) < 0.03 and abs(bottle["pos"][1] - start[1]) < 0.03
    assert len(s["tcp"]) == 3 and s["tcp"][2] > 0

    # Same seed, same state (cached); another seed jitters the start.
    assert client.get(f"/api/projects/{pid}/simulation/state").json()["bodies"] == s["bodies"]
    s2 = client.get(f"/api/projects/{pid}/simulation/state?seed=3").json()
    assert s2["bodies"]["obj_small_water_bottle"]["pos"] != bottle["pos"]
