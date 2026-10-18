"""Kitting benchmark stages 1-2 (issue #156): tray scene, seeded episodes, simulator success check, oracle demos.
Real pinned EnvHub model; oracle episodes run headless (no rendering)."""

import json

import numpy as np
import pytest

from cortex.sim import kitting as k
from cortex.sim.feasibility import check
from cortex.sim.kitting_oracle import PROVENANCE, measure, record
from cortex.sim.scene import build_scene
from cortex.sim.so101 import LIGHT_SCALE, Variation, configure_gl

configure_gl()
HOME_STEPS = 25


@pytest.fixture(scope="module")
def env2(tmp_path_factory):
    env = k.KittingEnv(2, tmp_path_factory.mktemp("kit"))
    yield env
    env.close()


def test_tray_scene_has_slots_sized_to_their_components(tmp_path):
    spec, task = k.sample_episode(2, 0)
    xml = build_scene(spec, tmp_path).path.read_text()
    for name in ("obs_tray_base", "obs_tray_rail_0", "obs_tray_wall_1", "zone_slot_1", "zone_slot_2",
                 "obj_gear_block", "obj_peg"):
        assert name in xml
    slots = {z.name: z for z in spec.zones}
    for kit, (comp, slot) in k.KIT_TYPES.items():
        o, (hx, hy) = k.COMPONENTS[comp], slots[slot].half_size
        footprint = np.hypot(*o.size[:2]) / 2 if o.shape == "box" else o.size[0] / 2  # any yaw
        assert footprint + 0.005 < min(hx, hy) < footprint + 0.02, (kit, comp, slot)  # fits, but snug
    walls = [ob for ob in spec.environment.obstacles if ob.name != "tray_base"]
    assert all(ob.z == k.BASE_T and ob.size[-1] == k.WALL_H for ob in walls)
    assert spec.task.pick == task.component and spec.task.place == [task.slot]


def test_episodes_are_seeded_valid_and_varied():
    (x0, x1), (y0, y1) = k.tray_footprint()
    for stage in (1, 2):
        tasks = [k.sample_task(stage, s) for s in range(50)]
        assert [t.model_dump() for t in tasks] == [k.sample_task(stage, s).model_dump() for s in range(50)]
        assert len({json.dumps(t.starts) for t in tasks}) == 50
        assert {t.lighting for t in tasks} == set(k.LIGHTINGS)
        for t in tasks:
            assert set(t.starts) == set(k.STAGE_COMPONENTS[stage])
            for x, y in t.starts.values():
                assert k.START_R[0] <= np.hypot(x, y) <= k.START_R[1]
                assert not (x0 - 0.03 < x < x1 + 0.03 and y0 - 0.03 < y < y1 + 0.03)  # clear of the tray
            if stage == 2:
                (ax, ay), (bx, by) = t.starts.values()
                assert np.hypot(ax - bx, ay - by) >= k.MIN_SEPARATION
        kits = {t.kit for t in tasks}
        assert kits == ({"B"} if stage == 1 else {"A", "B"})
    with pytest.raises(ValueError, match="not built"):
        k.sample_task(3, 0)


def test_reset_reproduces_the_sampled_episode(env2):
    env2.reset(1)
    q1 = env2.world.data.qpos.copy()
    env2.reset(5)
    env2.reset(1)
    assert np.allclose(q1, env2.world.data.qpos)
    t = env2.task
    assert env2.pick == t.component and np.allclose(env2.box_pos()[:2], dict(env2.info.zones)[t.slot][0])
    for name, xy in t.starts.items():
        assert np.allclose(env2.world.data.xpos[k._body(env2, name)][:2], xy, atol=1e-6)
    assert np.allclose(env2.world.model.vis.headlight.diffuse, np.clip(env2._light0[1] * LIGHT_SCALE[t.lighting], 0, 1))
    with pytest.raises(ValueError, match="fully determined"):
        env2.reset(1, Variation(lighting="dim"))


def test_feasibility_accepts_both_kits_at_the_edges_of_the_start_region(tmp_path):
    far = (round(k.START_R[1] * np.cos(-0.46), 3), round(k.START_R[1] * np.sin(-0.46), 3))  # inside the start box
    near, corner = (k.START_R[0], 0.0), (k.START_X[0], k.START_Y[0])
    for kit, starts in (("A", {"gear_block": far, "peg": near}), ("B", {"gear_block": corner, "peg": far})):
        spec = k.scene_spec(2, starts, kit=kit)
        r = check(spec, build_scene(spec, tmp_path / kit))
        assert r.ok, r.summary


def settle(env, steps=HOME_STEPS):
    for _ in range(steps):
        env.step(np.asarray(env.module.HOME, dtype=np.float32), render=False)


def place(env, name, xy, upright=True):
    d = env.world.data
    qa = env.world._free_qadr(k._body(env, name))
    o = k.COMPONENTS[name]
    z = k.BASE_T + (o.height if upright else o.size[0]) / 2 + 0.001
    d.qpos[qa:qa + 3] = (*xy, z)
    d.qpos[qa + 3:qa + 7] = (1, 0, 0, 0) if upright else (np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0)
    d.qvel[:] = 0


def test_success_check_names_the_failure(env2):
    slot = {z: c for z, (c, _) in env2.info.zones.items()}
    env2.reset(0)  # kit B: the gear block into slot 2
    t = env2.task
    assert t.component == "gear_block" and k.check_success(env2, t) == (False, "not picked")
    place(env2, "gear_block", slot["slot_1"])
    settle(env2)
    assert k.check_success(env2, t) == (False, "wrong slot")
    env2.max_cube_z = 0.1  # it was lifted, then let go on the table
    place(env2, "gear_block", (0.20, -0.05))
    settle(env2)
    assert k.check_success(env2, t) == (False, "dropped")
    place(env2, "gear_block", slot["slot_2"])
    settle(env2)
    assert k.check_success(env2, t) == (True, "ok") and env2.in_box()
    place(env2, "peg", (slot["slot_2"][0], slot["slot_2"][1] + 0.016))
    place(env2, "gear_block", (slot["slot_2"][0], slot["slot_2"][1] - 0.012))
    settle(env2)
    assert k.check_success(env2, t) == (False, "slot occupied")

    env2.reset(1)  # kit A: the peg into slot 1
    assert env2.task.component == "peg"
    place(env2, "peg", slot["slot_1"], upright=False)
    settle(env2)
    assert k.check_success(env2, env2.task) == (False, "not upright")


@pytest.mark.parametrize("stage,seeds", [(1, [0, 1, 2]), (2, [0, 1, 4])])
def test_oracle_succeeds_headless(stage, seeds, tmp_path):
    eps = measure(stage, seeds, tmp_path)
    assert all(e["success"] and e["reason"] == "ok" for e in eps), eps
    if stage == 2:
        assert {e["kit"] for e in eps} == {"A", "B"}


def test_reason_reads_as_progress_during_an_oracle_episode(env2):
    from cortex.sim.kitting_oracle import KittingOracle

    obs = env2.reset(1)
    pol = KittingOracle()
    pol.reset(env2, obs, 1)
    reasons = []
    while not pol.ex.done and len(reasons) < 1000:
        env2.step(pol.act(obs), render=False)
        r = k.check_success(env2, env2.task)[1]
        if not reasons or reasons[-1] != r:
            reasons.append(r)
    settle(env2, 10)
    assert reasons[0] == "not picked" and "not released" in reasons and k.check_success(env2, env2.task)[1] == "ok"


def test_record_writes_the_sim_worker_layout_with_provenance(tmp_path):
    res = record(2, [1], tmp_path, max_steps=30)
    out = tmp_path / "stage2_seed1"
    for name in ("episode.npz", "front.mp4", "thumb.jpg", "result.json"):
        assert (out / name).exists(), name
    with np.load(out / "episode.npz") as z:
        assert z["front"].shape[0] == z["action"].shape[0] == z["state"].shape[0] == 30
    saved = json.loads((out / "result.json").read_text())
    assert saved["provenance"]["planner"] == PROVENANCE == "scripted oracle (privileged state)"
    assert saved["provenance"]["kit"] == "A" and saved["num_frames"] == 30 and res[0]["seed"] == 1
    assert not saved["success"] and saved["reason"] != "ok"  # 30 steps is not enough to finish
