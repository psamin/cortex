"""The generic scripted skill completes home -> pick -> place -> home on intake scenes (real pinned model)."""

import numpy as np

from cortex import intake
from cortex.sim.pick_place_skill import PickPlaceSkill
from cortex.sim.scene import build_scene
from cortex.sim.scene_env import SceneEnv
from cortex.sim.so101 import configure_gl

configure_gl()


def run(prompt, zones, tmp_path, seed=1, max_steps=3000, start=None, jitter=0.015, render=True):
    spec = intake.start(prompt)
    name = spec.objects[0].name
    spec = intake.apply_patch(spec, {"objects": [{"name": name, "confirm": True}], "zones": zones,
                                     "task": {"place": [zones[0]["name"]]}})
    env = SceneEnv(build_scene(spec, tmp_path))
    obs = env.reset(seed, start=start, jitter=jitter)
    skill = PickPlaceSkill(env, height=spec.objects[0].height, grasp_width=spec.objects[0].grasp_width)
    skill.plan(obs["agent_pos"])
    phases = []
    for _ in range(max_steps):
        env.step(skill.act(), render=render)
        if not phases or phases[-1] != skill.phase:
            phases.append(skill.phase)
        if skill.done:
            break
    for _ in range(30):  # let the object settle
        env.step(np.asarray(env.world.joint_state(), dtype=np.float32), render=render)
    home_err = float(np.abs(env.world.joint_state()[:5] - np.asarray(env.module.HOME)[:5]).max())
    zone = np.asarray(zones[0]["center"], dtype=np.float64)
    place_err = float(np.linalg.norm(env.cube_pos()[:2] - zone))
    out = dict(success=env.in_box(), failed=skill.failed, phases=phases, home_err=home_err,
               place_err=place_err)
    env.close()
    return out


def test_cube_top_grasp_home_to_home(tmp_path):
    r = run("pick up the cube", [{"name": "spot_a", "center": [0.22, 0.12]}], tmp_path)
    assert r["failed"] is None, r
    assert r["success"], r
    assert r["phases"][0] == "hover" and r["phases"][-1] == "home" and r["home_err"] < 2.0


def test_small_bottle_side_grasp_home_to_home(tmp_path):
    r = run("pick up the small water bottle", [{"name": "spot_a", "center": [0.30, -0.13]}], tmp_path,
            render=False)
    assert r["failed"] is None, r
    assert r["success"], r
    assert r["place_err"] < 0.01, r
    assert "slide" in r["phases"] and r["phases"][-1] == "home" and r["home_err"] < 2.0


# Reachable side-grasp arc. (0.30, ±0.13) is where an off-axis start used to tip the bottle
# on every seed, and where a reset from the mat failed. Each start is placed at another
# point on the arc, so both the pick and the place are off the centre line.
ARC = ((0.32, 0.00), (0.32, -0.02), (0.31, 0.10), (0.31, -0.10), (0.30, 0.13), (0.30, -0.13))


def test_bottle_side_grasp_sweeps_the_reachable_arc(tmp_path):
    summary = []
    misses = []
    for i, start in enumerate(ARC):
        zone = ARC[(i + 3) % len(ARC)]
        ok = 0
        errs = []
        for seed in range(10):
            r = run("pick up the small water bottle", [{"name": "spot_a", "center": list(zone)}],
                    tmp_path / f"{i}_{seed}", seed=seed, start=start, jitter=0.0, render=False)
            good = r["success"] and r["failed"] is None and r["place_err"] < 0.01
            errs.append(r["place_err"] if good else None)
            if good:
                ok += 1
            else:
                misses.append((start, zone, seed, r["failed"], round(r["place_err"], 4), r["success"]))
        placed = [e for e in errs if e is not None]
        worst = max(placed) if placed else None
        summary.append((start, zone, ok, worst))
        print(f"{start} -> {zone}: {ok}/10  max placement {None if worst is None else round(worst * 100, 2)} cm",
              flush=True)
    short = [s for s in summary if s[2] < 9]
    assert not short, {"summary": summary, "misses": misses}
