"""Kitting benchmark (issue #156), stages 1-2: put the right component in the right slot of a kit tray.

Built on the intake scene stack (scene.py, scene_env.py), without changing it:

  * the kit tray is static obstacles: a thin base, low walls, and dividers into named slots. Each slot is a
    PlaceZone sized to its component;
  * two component types fit the gripper: ``gear_block`` (red box) and ``peg`` (blue cylinder);
  * stage 1: only the gear block is on the table, and it goes in its slot;
  * stage 2: both components are on the table, and the kit type says which one goes where
    (kit A: the peg in slot 1; kit B: the gear block in slot 2).

``(stage, seed)`` fully determines an episode: component starts and yaws (in reach, clear of the tray and of
each other), lighting, kit type, and the upstream env's own joint and camera noise.

Service API: ``sample_episode(stage, seed) -> (IntakeSpec, KitTask)``, ``KittingEnv(stage, scene_dir)``
(``reset(seed)`` reproduces that episode), and ``check_success(env, task) -> (ok, reason)``, judged from
simulator state only.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from pydantic import BaseModel

from ..intake.spec import Environment, IntakeSpec, ObjectSpec, Obstacle, PlaceZone, TaskV2
from .scene import body_name, build_scene
from .scene_env import FACE_DOWN_COS, SETTLED_SPEED, UPRIGHT_COS, SceneEnv
from .so101 import LIGHT_SCALE, Variation

COMPONENTS = {
    "gear_block": ObjectSpec(name="gear_block", shape="box", size=[0.026, 0.026, 0.026], mass_kg=0.02,
                             rgba=(0.85, 0.15, 0.12, 1.0), start_jitter_m=0.0),
    "peg": ObjectSpec(name="peg", shape="cylinder", size=[0.022, 0.045], mass_kg=0.015,
                      rgba=(0.15, 0.35, 0.90, 1.0), start_jitter_m=0.0),
}
# Tray: slots side by side along +y, each SLOT_DEPTH deep along x. Sized from the oracle's measured landing
# spread (200 episodes): the block lands within ~8 mm of centre (worst seen 14 mm) at any yaw, because the skill
# does not control release yaw, so its slot is 18.4 mm (half diagonal) + 8 mm + margin per side. In a 48 mm slot
# it leaned on a wall in 2 of 20 episodes. The peg lands within 5 mm.
TRAY_CENTER = (0.22, 0.10)
SLOTS = (("slot_1", 0.044), ("slot_2", 0.058))  # slot_1 holds the peg, slot_2 the gear block
SLOT_DEPTH = 0.058
BASE_T, WALL_T, WALL_H = 0.003, 0.006, 0.012  # base thickness; wall thickness and height above the base
TRAY_RGBA, WALL_RGBA = (0.30, 0.32, 0.36, 1.0), (0.22, 0.24, 0.27, 1.0)
KIT_TYPES = {"A": ("peg", "slot_1"), "B": ("gear_block", "slot_2")}
STAGES = {1: "one component (the gear block), randomized start, into its slot",
          2: "two component types and two slots; the kit type picks the component and the slot"}
STAGE_COMPONENTS = {1: ("gear_block",), 2: ("gear_block", "peg")}
STAGE_KITS = {1: ("B",), 2: ("A", "B")}
# Component starts: in front of and to the right of the tray, apart from each other, and within the radii where
# feasibility's reach check passes for both parts at grasp and carry height (measured: 0.16-0.26 m, bearings
# 0 to -50 degrees; carry poses fail beyond 0.27 m).
START_X, START_Y, START_R = (0.15, 0.28), (-0.16, 0.0), (0.16, 0.26)
MIN_SEPARATION = 0.09  # centre to centre: at 0.071 m the open jaws knocked the other part over (stage 2, seed 10)
START_YAW = 0.4
LIGHTINGS = ("dim", "normal", "bright")
REST_TOL = 0.006  # resting on the tray base, not perched on a wall or on another part
LIFTED = 0.02  # the target part rose this far above its resting height at some point


class KitTask(BaseModel):
    """One kitting episode, fully determined by (stage, seed)."""

    stage: int
    seed: int
    kit: str
    component: str
    slot: str
    instruction: str
    lighting: str
    starts: dict[str, tuple[float, float]]
    yaws: dict[str, float]


def tray() -> tuple[list[PlaceZone], list[Obstacle]]:
    """The kit tray: slots (PlaceZones, the inner area of each slot) and its obstacles."""
    cx, cy = TRAY_CENTER
    total_y = sum(w for _, w in SLOTS) + WALL_T * (len(SLOTS) + 1)
    total_x = SLOT_DEPTH + 2 * WALL_T
    obs = [Obstacle(name="tray_base", shape="box", size=[total_x, total_y, BASE_T], center=(cx, cy), rgba=TRAY_RGBA)]
    for i, sx in enumerate((-1, 1)):  # rails along y, in front of and behind the slots
        obs.append(Obstacle(name=f"tray_rail_{i}", shape="box", size=[WALL_T, total_y, WALL_H],
                            center=(round(cx + sx * (SLOT_DEPTH + WALL_T) / 2, 4), cy), z=BASE_T, rgba=WALL_RGBA))
    zones: list[PlaceZone] = []
    y = cy - total_y / 2
    for i in range(len(SLOTS) + 1):  # end walls and dividers across y
        obs.append(Obstacle(name=f"tray_wall_{i}", shape="box", size=[SLOT_DEPTH, WALL_T, WALL_H],
                            center=(cx, round(y + WALL_T / 2, 4)), z=BASE_T, rgba=WALL_RGBA))
        y += WALL_T
        if i < len(SLOTS):
            name, w = SLOTS[i]
            zones.append(PlaceZone(name=name, center=(cx, round(y + w / 2, 4)), half_size=(SLOT_DEPTH / 2, w / 2)))
            y += w
    return zones, obs


def tray_footprint() -> tuple[tuple[float, float], tuple[float, float]]:
    """(x range, y range) of the tray on the table."""
    (cx, cy), hx = TRAY_CENTER, SLOT_DEPTH / 2 + WALL_T
    hy = (sum(w for _, w in SLOTS) + WALL_T * (len(SLOTS) + 1)) / 2
    return (cx - hx, cx + hx), (cy - hy, cy + hy)


def _check_stage(stage: int) -> None:
    if stage not in STAGES:
        raise ValueError(f"kitting stage {stage} is not built yet (stages: {sorted(STAGES)})")


def instruction(kit: str, component: str, slot: str) -> str:
    return f"Kit {kit}: place the {component.replace('_', ' ')} in {slot.replace('_', ' ')}."


def scene_spec(stage: int, starts: dict[str, tuple[float, float]] | None = None, lighting: str = "normal",
               kit: str | None = None) -> IntakeSpec:
    """The stage's scene as an IntakeSpec. Without starts, components sit at fixed nominal spots (every
    KittingEnv reset moves them to the episode's sampled starts)."""
    _check_stage(stage)
    names = STAGE_COMPONENTS[stage]
    kit = kit or STAGE_KITS[stage][-1]
    component, slot = KIT_TYPES[kit]
    nominal = {"gear_block": (0.19, -0.02), "peg": (0.20, -0.13)}  # inside the start region, 0.11 m apart
    starts = starts or {n: nominal[n] for n in names}
    zones, obstacles = tray()
    objects = [COMPONENTS[n].model_copy(update={"start": tuple(starts[n])}) for n in names]
    return IntakeSpec(
        prompt=instruction(kit, component, slot), objects=objects, zones=zones,
        environment=Environment(lighting=lighting, obstacles=obstacles, assumed=set(),  # type: ignore[arg-type]
                                description="kitting: a kit tray with named slots"),
        task=TaskV2(pick=component, place=[slot]))


def sample_task(stage: int, seed: int) -> KitTask:
    _check_stage(stage)
    rng = np.random.default_rng(seed)
    kits = STAGE_KITS[stage]
    kit = kits[int(rng.integers(len(kits)))]
    lighting = LIGHTINGS[int(rng.integers(len(LIGHTINGS)))]
    names = STAGE_COMPONENTS[stage]
    for _ in range(1000):  # sample all starts together, so a central first part cannot crowd out the second
        pts = np.round(rng.uniform((START_X[0], START_Y[0]), (START_X[1], START_Y[1]), (len(names), 2)), 4)
        r = np.hypot(pts[:, 0], pts[:, 1])
        if np.all((START_R[0] <= r) & (r <= START_R[1])) and all(
                np.hypot(*(pts[i] - pts[j])) >= MIN_SEPARATION for i in range(len(pts)) for j in range(i)):
            break
    else:  # acceptance is tens of percent; reaching this means the region constants are broken
        raise RuntimeError(f"no collision-free starts (stage {stage}, seed {seed})")
    starts = {n: (float(x), float(y)) for n, (x, y) in zip(names, pts)}
    yaws = {n: round(float(rng.uniform(-START_YAW, START_YAW)), 4) for n in names}
    component, slot = KIT_TYPES[kit]
    return KitTask(stage=stage, seed=seed, kit=kit, component=component, slot=slot,
                   instruction=instruction(kit, component, slot), lighting=lighting, starts=starts, yaws=yaws)


def sample_episode(stage: int, seed: int) -> tuple[IntakeSpec, KitTask]:
    """The episode's scene (an IntakeSpec with the sampled starts, lighting and task) and its KitTask."""
    task = sample_task(stage, seed)
    return scene_spec(stage, task.starts, task.lighting, task.kit), task


class KittingEnv(SceneEnv):
    """A SceneEnv over the stage's tray scene whose ``reset(seed)`` sets up episode (stage, seed) and whose
    ``in_box`` (the success flag every worker reads) is ``check_success``."""

    def __init__(self, stage: int, scene_dir: Path, episode_steps: int = 1500, height: int = 96, width: int = 128):
        self.stage = stage
        self.task: KitTask | None = None
        super().__init__(build_scene(scene_spec(stage), scene_dir), height=height, width=width,
                         episode_steps=episode_steps)

    def reset(self, seed: int, var: Variation | None = None, **_) -> dict:  # type: ignore[override]
        if var is not None:  # no silent substitution: a condition's lighting or offsets would be dropped
            raise ValueError("a kitting episode is fully determined by (stage, seed); pass no Variation")
        task = sample_task(self.stage, seed)
        self.env.reset(seed=seed)
        if self.world is None:
            self._bind()
        self._set_pick(task.component)
        self.set_target(task.slot)
        for name, xy in task.starts.items():
            self.place_object(name, xy, yaw=task.yaws[name])
        d = self.world.data
        d.qvel[:] = 0
        self._apply_light(LIGHT_SCALE[task.lighting])
        mujoco.mj_forward(self.world.model, d)
        self.task = task
        self.max_cube_z = 0.0
        self.steps = 0
        return self._obs()

    def _set_pick(self, name: str) -> None:
        """Retarget the pick-object bindings (``cube_*``, ``obj_*``, ``others``) at another component."""
        w = self.world
        o = COMPONENTS[name]
        self.pick, self.obj_h, self.shape = name, o.height, o.shape
        self.cube = _body(self, name)
        self.cube_qadr = w._free_qadr(self.cube)
        self.obj_geoms = _geoms(self, name)
        self.others = {n: w._free_qadr(_body(self, n)) for n in self.info.objects if n != name}

    def in_box(self) -> bool:
        return self.task is not None and check_success(self, self.task)[0]


def _body(env: SceneEnv, name: str) -> int:
    return mujoco.mj_name2id(env.world.model, mujoco.mjtObj.mjOBJ_BODY, body_name(name))


def _geoms(env: SceneEnv, name: str) -> set[int]:
    m, b = env.world.model, _body(env, name)
    return {g for g in range(m.ngeom) if m.geom_bodyid[g] == b}


_SLOT_RECTS = {z.name: (z.center, z.half_size) for z in tray()[0]}


def _in_slot(xy: np.ndarray, slot: str) -> bool:
    (cx, cy), (hx, hy) = _SLOT_RECTS[slot]
    return bool(abs(xy[0] - cx) <= hx and abs(xy[1] - cy) <= hy)


def _upright(R: np.ndarray, shape: str) -> bool:
    if shape == "box":  # a block rests on any face
        return bool(np.max(np.abs(R[2, :])) > FACE_DOWN_COS)
    return bool(abs(R[2, 2]) > UPRIGHT_COS)  # a peg stands on an end


def _rest_z(R: np.ndarray, o: ObjectSpec) -> float:
    if o.shape == "box":
        return o.size[int(np.argmax(np.abs(R[2, :])))] / 2
    return o.height / 2


def check_success(env: SceneEnv, task: KitTask) -> tuple[bool, str]:
    """The simulator's verdict: the task's component rests in the task's slot, released, upright where it
    matters, and no other part is in that slot. Returns (True, "ok") or (False, reason), reason one of
    "not released", "slot occupied", "wrong slot", "dropped", "not picked", "not upright", "not settled".
    Mid-episode the reason reads as progress: "not picked", then "not released" while carried."""
    d, m = env.world.data, env.world.model
    parts = list(env.info.objects)
    pos = {n: d.xpos[_body(env, n)].copy() for n in parts}
    others_in = [n for n in parts if n != task.component and _in_slot(pos[n], task.slot)]
    b, p, o = _body(env, task.component), pos[task.component], COMPONENTS[task.component]
    geoms = _geoms(env, task.component)
    for g1, g2 in zip(d.contact.geom1[:d.ncon], d.contact.geom2[:d.ncon]):
        if (g1 in env.gripper_geoms and g2 in geoms) or (g2 in env.gripper_geoms and g1 in geoms):
            return False, "not released"
    if not _in_slot(p, task.slot):
        if others_in:
            return False, "slot occupied"
        if any(_in_slot(p, s) for s, _ in SLOTS if s != task.slot):
            return False, "wrong slot"
        lifted = env.pick != task.component or env.max_cube_z > o.height / 2 + LIFTED
        return False, "dropped" if lifted else "not picked"
    R = d.xmat[b].reshape(3, 3)
    if not _upright(R, o.shape):
        return False, "not upright"
    dof = m.jnt_dofadr[m.body_jntadr[b]]
    still = np.linalg.norm(d.qvel[dof:dof + 3]) < SETTLED_SPEED
    if not still or abs(p[2] - (BASE_T + _rest_z(R, o))) > REST_TOL:
        return False, "not settled"
    if others_in:
        return False, "slot occupied"
    return True, "ok"
