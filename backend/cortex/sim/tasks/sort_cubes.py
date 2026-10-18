"""Sort cubes by color (issue #334): put the red, green and blue cubes into the bins of the same color.

Scene, on the shared workcell (``workcell.py``: dark rubber mat, grey bench, its front camera), in metres in
the robot frame (+x away from the base, +y to its left). Everything sits on arcs around the shoulder-pan axis
inside the arm's top-down reach (0.14-0.22 m from the axis, measured with ``expert.solve_ik``):

* three bins on the arm's left: low-walled trays, 8.4 cm square inside with 2.2 cm walls, painted red, green
  and blue (a lighter floor of the same color, so a cube stands out in its bin), each turned to face the arm;
* three 3 cm wooden cubes on the arm's right: the upstream EnvHub red, green and blue cubes (yellow stays parked).

Per seed: which bin slot each color takes and which start slot each cube takes (both shuffled), each cube's
offset (+-1 cm) and yaw (+-0.4 rad), each bin's offset (+-4 mm), the bench tint, a small front-camera tilt
and the lighting.

Success: every cube rests in the bin of its own color, inside its walls, on its floor, at rest and not
touched by the arm. ``metrics()["sorted"]`` counts the cubes that do. The ``Oracle`` runs ``ScriptedExpert``'s
pick once per cube on privileged state, so it is a demonstrator, not a policy.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET  # nosec B405 - we build this XML ourselves
from typing import Any

import mujoco
import numpy as np

from ..expert import GRIP_CLOSED, ScriptedExpert, _line, closing_direction, solve_ik
from ..so101 import SETTLED_SPEED
from .base import rgba
from .workcell import WorkcellEnv, add_workcell, box, yaw_quat

COLORS = {"red": (0.95, 0.15, 0.12), "green": (0.20, 0.75, 0.25), "blue": (0.15, 0.35, 0.95)}  # the cubes' paint
PAN_AXIS = np.array([0.0388, 0.0])  # shoulder_pan axis on the table (Menagerie SO-101)
BIN_SLOTS = ((85.0, 0.185), (50.0, 0.19), (15.0, 0.20))  # (bearing deg, radius m) about the pan axis
CUBE_SLOTS = ((-15.0, 0.175), (-50.0, 0.175), (-85.0, 0.175))  # 10.5 cm apart: open jaws clear the neighbours
CUBE_JITTER = 0.01
CUBE_YAW = 0.4
BIN_JITTER = 0.004
CUBE_HALF = 0.015
BIN_INNER, BIN_WALL_T, BIN_WALL_H, BIN_FLOOR = 0.042, 0.004, 0.022, 0.003  # half width inside; wall; floor
REST_TOL = 0.006  # cube centre at most this far above lying flat on the bin floor
GRASP_SHIFT = 0.008  # oracle: open jaws' clearance moved from the moving jaw's side to the fixed jaw's (m)
CARRY_Z, RELEASE_Z, RETREAT_Z = 0.075, 0.017, 0.08  # oracle: gripper site over the bin; at 0.017 the cube is 2 mm up
GRIP_RELEASE = 30.0  # oracle: open this far in the bin (moving jaw 1.7 cm off the cube, clear of the walls)


def slot_xy(bearing_deg: float, radius: float) -> np.ndarray:
    b = np.deg2rad(bearing_deg)
    return PAN_AXIS + radius * np.array([np.cos(b), np.sin(b)])


class Env(WorkcellEnv):
    name = "sort_cubes"
    instruction = "Put each cube in the bin of the same color."
    max_steps = 900  # the oracle takes 650-680 steps
    view = {"lookat": (0.13, 0.07, 0.03), "distance": 0.62, "azimuth": 180, "elevation": -55}  # run.Presenter

    # ----------------------------------------------------------------- scene
    def build(self, mj: ET.Element, wb: ET.Element) -> None:
        add_workcell(mj, wb)
        asset = mj.find("asset")
        assert asset is not None
        i, t, h = BIN_INNER, BIN_WALL_T, BIN_WALL_H
        for k, (color, rgb) in enumerate(COLORS.items()):
            c = np.asarray(rgb)
            ET.SubElement(asset, "material", name=f"bin_{color}", rgba=rgba((*0.8 * c, 1)), specular="0.2",
                          shininess="0.3")
            ET.SubElement(asset, "material", name=f"bin_{color}_floor", rgba=rgba((*(0.55 * c + 0.4), 1)),
                          specular="0.1", shininess="0.2")
            b = ET.SubElement(wb, "body", name=f"bin_{color}", pos=f"0.6 {1.5 + 0.1 * k} 0")  # placed per seed
            box(b, f"bin_{color}_floor", (0, 0, BIN_FLOOR / 2), (i + t, i + t, BIN_FLOOR / 2), f"bin_{color}_floor")
            for s in (-1, 1):
                box(b, f"bin_{color}_wall_x{s:+d}", (s * (i + t / 2), 0, h / 2), (t / 2, i + t, h / 2), f"bin_{color}")
                box(b, f"bin_{color}_wall_y{s:+d}", (0, s * (i + t / 2), h / 2), (i, t / 2, h / 2), f"bin_{color}")

    def bind_task(self) -> None:
        super().bind_task()
        self.cubes = {c: self.body(f"{c}_cube") for c in COLORS}  # the upstream cubes; yellow stays parked
        self.bins = {c: self.body(f"bin_{c}") for c in COLORS}
        self.center = np.r_[np.mean([slot_xy(*s) for s in BIN_SLOTS + CUBE_SLOTS], axis=0), 0.0]
        self.layout: dict[str, Any] = {}

    def reset_task(self, rng: np.random.Generator) -> None:
        m = self.world.model
        bin_slot, cube_slot = rng.permutation(len(BIN_SLOTS)), rng.permutation(len(CUBE_SLOTS))
        pts = []
        for k, color in enumerate(COLORS):
            bearing, r = BIN_SLOTS[bin_slot[k]]
            xy = slot_xy(bearing, r) + rng.uniform(-BIN_JITTER, BIN_JITTER, 2)
            m.body_pos[self.bins[color]] = (*xy, 0.0)
            m.body_quat[self.bins[color]] = yaw_quat(np.deg2rad(bearing))
            pts.append(xy)
            xy = slot_xy(*CUBE_SLOTS[cube_slot[k]]) + rng.uniform(-CUBE_JITTER, CUBE_JITTER, 2)
            self.set_free(f"{color}_cube", (xy[0], xy[1], CUBE_HALF), yaw=rng.uniform(-CUBE_YAW, CUBE_YAW))
            pts.append(xy)
        self.center = np.r_[np.mean(pts, axis=0), 0.0]
        self.layout = {"bin_slots": dict(zip(COLORS, bin_slot.tolist())),
                       "cube_slots": dict(zip(COLORS, cube_slot.tolist()))}
        self.jitter_look(rng)

    # ------------------------------------------------------------ verifier
    def cube_xyz(self, color: str) -> np.ndarray:
        return self.world.data.xpos[self.cubes[color]].copy()

    def cube_yaw_of(self, color: str) -> float:
        q = self.world.data.xquat[self.cubes[color]]
        return float(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))

    def bin_xyz(self, color: str) -> np.ndarray:
        return self.world.data.xpos[self.bins[color]].copy()

    def in_bin(self, cube: str, bin_color: str) -> bool:
        """The cube lies inside that bin's walls, on its floor (not perched on a wall or held above it)."""
        d, b = self.world.data, self.bins[bin_color]
        local = (d.xpos[self.cubes[cube]] - d.xpos[b]) @ d.xmat[b].reshape(3, 3)
        return bool(np.all(np.abs(local[:2]) < BIN_INNER) and local[2] < BIN_FLOOR + CUBE_HALF + REST_TOL)

    def at_rest(self, color: str) -> bool:
        m, b = self.world.model, self.cubes[color]
        a = m.jnt_dofadr[m.body_jntadr[b]]
        return bool(np.linalg.norm(self.world.data.qvel[a:a + 3]) < SETTLED_SPEED)

    def sorted_cube(self, color: str) -> bool:
        return self.in_bin(color, color) and self.at_rest(color) and not self.robot_touches(self.cubes[color])

    def success(self) -> bool:
        return all(self.sorted_cube(c) for c in COLORS)

    def cube_pos(self) -> np.ndarray:
        """The middle of the layout (bins and cube starts), for the viewer's look-at: the whole task in frame."""
        return self.center.copy()

    def metrics(self) -> dict[str, Any]:
        wrong = sum(self.in_bin(c, other) for c in COLORS for other in COLORS if other != c)
        return {"sorted": sum(self.sorted_cube(c) for c in COLORS), "in_wrong_bin": wrong, **self.layout}


# ---------------------------------------------------------------------------------------------- oracle
class OneCube:
    """The env as ``ScriptedExpert`` reads it, for one color: that cube is "the cube", its bin is "the box".
    ``offset`` moves where the expert grasps (m, table plane)."""

    def __init__(self, env: Env, color: str):
        self.env, self.color = env, color
        self.world, self.site = env.world, env.site
        self.offset = np.zeros(3)

    def cube_pos(self) -> np.ndarray:
        return self.env.cube_xyz(self.color) + self.offset

    def cube_yaw(self) -> float:
        return self.env.cube_yaw_of(self.color)

    def box_pos(self) -> np.ndarray:
        return self.env.bin_xyz(self.color)


class Oracle:
    """``ScriptedExpert``'s pick once per cube, red, then green, then blue: hover, straight descent, close, lift,
    carry. Its place is replaced for the shallow bins: a straight descent until the cube is 2 mm above the bin
    floor, a partial release, and a straight retreat above the bin. Each cube's plan is solved when the previous one
    ends, from where that cube is then; after the last cube the arm returns to its home pose. Every command is
    an absolute joint target from the plan, so run.py's DART noise on the executed command does not stall it
    (the closed-loop ``PickPlacePolicy`` waits for the arm to arrive, and under noise it never does)."""

    name = "sort_cubes scripted expert (oracle, privileged state)"
    needs_pixels = False

    def reset(self, env: Env, obs: dict[str, Any], seed: int) -> None:
        self.env, self.rng = env, np.random.default_rng(seed)
        self.todo = list(COLORS)
        self.homing = False
        self.ex = ScriptedExpert(env, self.rng)  # no plan yet: act() plans the first cube
        self.ex.cmd = np.asarray(obs["agent_pos"], dtype=np.float64).copy()
        self.ex.waypoints, self.ex.phases, self.ex.i, self.ex.hold = [], [], 0, 0

    def _plan(self, color: str, cmd: np.ndarray) -> ScriptedExpert:
        one = OneCube(self.env, color)
        ex = ScriptedExpert(one, self.rng)  # type: ignore[arg-type]
        # The expert picks, of the four wrist rolls that put the jaws on the cube's faces, the one nearest its
        # seed's roll. Seeded from the last cube's roll, each pan swing pushed it further, up to the wrist's
        # -157 degree stop, and the jaws came down on the cube's edges (2 of 30 seeds failed even without noise).
        # Seed the plan at roll 0, mid-range; the motion still starts from where the arm was commanded.
        seed = cmd.copy()
        seed[4] = 0.0
        ex.plan(seed)
        # Open, the fixed jaw is 5 mm from the cube's face (the moving jaw 53 mm), so a few mm of DART noise put
        # it on the cube's edge. Grasp GRASP_SHIFT further from the fixed jaw; closing slides the cube onto it.
        one.offset = -GRASP_SHIFT * self._closing(ex.waypoints[ex.phases.index("grasp")][0])
        seed[4] = ex.waypoints[0][0][4]  # the same of the four rolls: near +-45 degrees the shift could flip it,
        ex.plan(seed)  # and the shift would then run along the jaws instead of across them
        # The expert releases 5.5 cm up with the jaws snapped wide open, made for the deep box. Into a shallow bin
        # the squeezed cube sprang up to 3 cm sideways as the jaw let go and leaned on a wall (still 2.5 cm from a
        # 3 cm release height). Lower it onto the floor, open only to GRIP_RELEASE, then retreat straight up:
        # it lands within 1 cm of the bin centre.
        k = ex.phases.index("carry") + 1
        del ex.waypoints[k:], ex.phases[k:]
        b = self.env.bin_xyz(color)[:2]
        top, low, up = np.r_[b, CARRY_Z], np.r_[b, RELEASE_Z], np.r_[b, RETREAT_Z]
        legs = [*_line(top, low, GRIP_CLOSED, "lower"), (low, GRIP_CLOSED, 4, False, "lower"),
                (low, GRIP_RELEASE, 10, False, "release"), *_line(low, up, GRIP_RELEASE, "retreat"),
                (up, GRIP_RELEASE, 4, False, "retreat")]
        q = ex.waypoints[-1][0]
        for target, grip, hold, _, phase in legs:
            arm, err = solve_ik(self.env, target, q)
            ex.max_ik_err = max(ex.max_ik_err, err)
            q = np.r_[arm, grip]
            ex.waypoints.append((q, hold))
            ex.phases.append(phase)
        ex.phases = [f"{color}: {p}" for p in ex.phases]
        ex.cmd = cmd.copy()
        return ex

    def _next(self) -> None:
        if self.todo:
            self.ex = self._plan(self.todo.pop(0), self.ex.cmd)
        else:
            self.homing = True
            self.ex.waypoints, self.ex.phases = [(np.asarray(self.env.module.HOME, dtype=np.float64), 8)], ["home"]
            self.ex.i, self.ex.hold = 0, 0

    def _closing(self, q: np.ndarray) -> np.ndarray:
        """The jaws' closing direction (fixed toward moving jaw) in the table plane, with the arm at q (deg)."""
        w = self.env.world
        d = mujoco.MjData(w.model)
        d.qpos[:] = w.data.qpos
        d.qpos[w.qadr[:5]] = np.deg2rad(q[:5])
        mujoco.mj_kinematics(w.model, d)
        c = closing_direction(w.model, d)
        c[2] = 0.0
        return c / (np.linalg.norm(c) or 1.0)

    @property
    def phase(self) -> str:
        return self.ex.phase if self.ex.waypoints else ""

    def act(self, obs: dict[str, Any]) -> np.ndarray:
        if self.ex.done and not self.homing:
            self._next()
        return self.ex.act()

    @property
    def done(self) -> bool:
        return self.homing and self.ex.done
