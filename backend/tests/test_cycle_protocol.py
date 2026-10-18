"""P6: the cycle protocol in sim (demo, reset, recover segments) on intake scenes, real pinned model.

The cube runs go through the services (recording -> tagged episodes -> dataset lineage); the small bottle,
whose side-grasp segments are slow, runs the worker loop directly."""

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from cortex import intake
from cortex.models import Condition, CreateDatasetVersionRequest, Episode, EpisodeSource
from cortex.sim.scene import build_scene
from cortex.sim.so101 import configure_gl
from cortex.workers import sim_worker

configure_gl()


def test_segment_defaults_keep_existing_data_as_demos():
    ep = Episode(project_id="p", index=0, source=EpisodeSource.sim_scripted, condition=Condition(), seed=1,
                 num_frames=1, fps=30, success=True)
    assert ep.segment == "demo"
    assert CreateDatasetVersionRequest().segments == ["demo", "reset"]  # recover is opt-in


def scene_payload(prompt: str, zone: tuple[float, float], policy: str, tmp: Path) -> tuple[dict, tuple]:
    spec = intake.start(prompt)
    name = spec.objects[0].name
    spec = intake.apply_patch(spec, {"objects": [{"name": name, "confirm": True}],
                                     "zones": [{"name": "spot_a", "center": list(zone)}],
                                     "task": {"place": ["spot_a"], "reset_policy": policy}})
    o = spec.objects[0]
    info = build_scene(spec, tmp / "scene")
    return ({"info": info.to_dict(tmp), "data_dir": str(tmp),
             "object": {"name": o.name, "height": o.height, "grasp_width": o.grasp_width}},
            info.starts[o.name])


def run_cycles(prompt, zone, policy, tmp, cycles=2, seed=3):
    scene, origin = scene_payload(prompt, zone, policy, tmp)
    res = sim_worker.cmd_cycles({"cycles": cycles, "reset_policy": policy, "seed": seed, "scene": scene,
                                 "live_dir": str(tmp / "live"), "out_root": str(tmp / "segs")})
    return res, np.asarray(origin)


def test_random_zone_moves_the_next_start(tmp_path):
    res, origin = run_cycles("pick up the cube", (0.22, 0.12), "random_zone", tmp_path)
    segs = res["segment_results"]
    assert [s["segment"] for s in segs] == ["demo", "reset", "demo", "reset"], segs
    assert all(s["success"] for s in segs), [(s["segment"], s["failure_reason"]) for s in segs]
    assert segs[1]["target"]["label"] == "random"
    d0, d1 = (np.asarray(segs[i]["start_pose"]["xyz"][:2]) for i in (0, 2))
    assert np.linalg.norm(d1 - origin) > 0.02  # the second demo starts where the reset put it, not the origin
    assert np.linalg.norm(d1 - np.asarray(segs[1]["target"]["xy"])) < 0.02
    assert np.linalg.norm(d1 - d0) > 0.015


def test_small_bottle_two_cycles_return_to_origin(tmp_path):
    res, origin = run_cycles("pick up the small water bottle", (0.30, -0.13), "return_to_origin", tmp_path, seed=1)
    segs = res["segment_results"]
    assert [s["segment"] for s in segs] == ["demo", "reset", "demo", "reset"], segs
    assert all(s["success"] for s in segs), [(s["segment"], s["failure_reason"]) for s in segs]
    end = np.asarray(segs[-1]["end_pose"]["xyz"][:2])
    assert np.linalg.norm(end - origin) < 0.03 and segs[-1]["end_pose"]["upright"]
    for s in segs:  # every successful segment is a LeRobot-ready episode
        with np.load(Path(s["out_dir"]) / "episode.npz") as z:
            assert len(z["action"]) == s["num_frames"] > 100


# ------------------------------------------------------------------ services: tagged episodes and datasets

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


def test_cube_cycles_record_tagged_segments_and_dataset_keeps_demo_and_reset(client):
    from cortex.api import app as api
    from cortex.models import EpisodeStatus
    from cortex.services import agent_orchestrator as orch
    from cortex.services import dataset_service, recording_service

    pid = tool(client, "start_intake", prompt="pick up the cube and put it on the spot")["project_id"]
    tool(client, "update_intake", project_id=pid, patch={
        "objects": [{"name": "cube", "confirm": True}], "zones": [{"name": "spot_a", "center": [0.22, 0.12]}],
        "task": {"place": ["spot_a"], "reset_policy": "return_to_origin"}})
    tool(client, "build_simulation", project_id=pid)
    ctx = api.ctx()

    async def record():
        p = await orch.get_project(ctx, pid)
        job = await recording_service.start_recording(ctx, p, [Condition()], cycles=2)
        return await orch._wait(ctx, job.id, 900)

    j = client.portal.call(record)
    assert j["status"] == "succeeded", j
    eps = client.portal.call(recording_service.list_episodes, ctx, pid)
    assert [e.segment for e in sorted(eps, key=lambda e: e.index)] == ["demo", "reset", "demo", "reset"]
    for e in eps:
        pv = e.provenance
        assert pv["planner"] == "scripted_ik_oracle" and pv["segment"] == e.segment
        assert pv["cycle_id"].endswith(f"/{pv['cycle']}") and "start_pose" in pv and "end_pose" in pv
    last = max(eps, key=lambda e: e.index)
    p = client.portal.call(orch.get_project, ctx, pid)
    origin = np.asarray(p.scene["starts"][p.intake.task.pick])
    assert np.linalg.norm(np.asarray(last.provenance["end_pose"]["xyz"][:2]) - origin) < 0.03

    async def dataset():
        for e in eps:
            await recording_service.set_status(ctx, e.id, EpisodeStatus.accepted)
        job = await dataset_service.create_version(ctx, p, CreateDatasetVersionRequest())
        return await orch._wait(ctx, job.id, 600)

    j = client.portal.call(dataset)
    assert j["status"] == "succeeded", j
    ds = client.portal.call(dataset_service.list_versions, ctx, pid)[-1]
    assert ds.num_source_episodes == 4
    lineage = json.loads((ctx.local(ds.lerobot_root) / "meta" / "cortex_lineage.json").read_text())
    assert lineage["segments"] == {"included": ["demo", "reset"], "counts": {"demo": 2, "reset": 2, "recover": 0}}
