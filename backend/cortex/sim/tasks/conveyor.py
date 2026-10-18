"""Conveyor (issue #255): pick a part off a moving belt and place it in a bin.

Scene, on the shared workcell (``workcell.py``: rubber mat, materials, front camera), in metres in the
robot frame (+x away from the base, +y to its left):

* a belt (``sim/belt.py``) running right to left across the front of the arm, 0.20 m out, from
  y = +0.32 (upstream, out of reach) to y = -0.15. Its top is 3 cm above the table;
* a bin beside the downstream end, nearer the base, clear of where a missed part falls.

Per seed: one part (a box or a cylinder, finish and size drawn per seed) waits at the upstream end. The
belt starts after a delay (0-1 s) and runs at 3-8 cm/s, and the part has a lateral offset of up to
1.2 cm. The belt drives every free body resting on it (see ``Belt.drive``), so the part rides to the
end and falls off unless the arm takes it.

Success: the part is in the bin, released, at rest and not on the belt. A part that falls off the belt's
end is a failure (``metrics()["fell_off"]``). The ``Oracle`` reads privileged state (belt speed and the
part's pose) and is a demonstrator, not a policy.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET  # nosec B405 - we build this XML ourselves
from typing import Any

import mujoco
import numpy as np

from ..belt import Belt
from ..so101 import FPS
from .workcell import FACE, GRIP_OPEN, GRIP_SHUT, TopDownOracle, WorkcellEnv, add_workcell, box, rot

BELT = Belt("line", start=(0.20, 0.32), end=(0.20, -0.15), width=0.07, height=0.03)
SPEED = (0.03, 0.08)  # m/s
DELAY = (0.0, 1.0)  # s before the belt starts
LATERAL = 0.012  # part offset across the belt, +-
S0 = 0.03  # part centre this far downstream of the belt's start at reset
YAW_JITTER = np.deg2rad(5.0)

PART_HALF_W = (0.0105, 0.0120)  # across the belt (the jaws close on this)
PART_HALF_L = (0.0105, 0.0140)  # along the belt (boxes)
PART_HALF_H = (0.018, 0.024)
SHAPES = ("box", "cylinder")

BIN_XY = np.array([0.04, -0.20])
BIN_JITTER = 0.01
BIN_INNER = 0.04  # half width inside the walls
BIN_WALL_T, BIN_WALL_H, BIN_FLOOR = 0.004, 0.03, 0.003

MAX_SPEED = 0.05  # settled

# name -> (rgb, specular, shininess)
FINISHES = {
    "safety orange": ((0.90, 0.36, 0.06), 0.3, 0.4),
    "anodized blue": ((0.10, 0.24, 0.62), 0.6, 0.6),
    "zinc": ((0.70, 0.72, 0.76), 0.8, 0.7),
    "brass": ((0.74, 0.58, 0.26), 0.9, 0.8),
}


class Env(WorkcellEnv):
    name = "conveyor"
    instruction = "Pick the part off the moving conveyor and place it in the bin."
    max_steps = 900
    focus = "part"
    view = {"lookat": (0.15, 0.06, 0.03), "distance": 0.85, "azimuth": 200, "elevation": -32}  # run.Presenter
    belt = BELT

    # ----------------------------------------------------------------- scene
    def build(self, mj: ET.Element, wb: ET.Element) -> None:
        add_workcell(mj, wb)
        asset = mj.find("asset")
        assert asset is not None
        ET.SubElement(asset, "material", name="part_finish", rgba="0.6 0.6 0.6 1", specular="0.5", shininess="0.5")
        self.belt.add_bed(wb)
        b = ET.SubElement(wb, "body", name="bin", pos=f"{BIN_XY[0]} {BIN_XY[1]} 0")
        i, t, h = BIN_INNER, BIN_WALL_T, BIN_WALL_H
        box(b, "bin_floor", (0, 0, BIN_FLOOR / 2), (i + t, i + t, BIN_FLOOR / 2), "tray")
        for s in (-1, 1):
            box(b, f"bin_wall_x{s:+d}", (s * (i + t / 2), 0, h / 2), (t / 2, i + t, h / 2), "tray")
            box(b, f"bin_wall_y{s:+d}", (0, s * (i + t / 2), h / 2), (i, t / 2, h / 2), "tray")
        for shape in SHAPES:  # both part shapes exist; the unused one is parked under the table
            p = ET.SubElement(wb, "body", name=f"part_{shape}", pos=f"0.6 {1.2 + 0.1 * SHAPES.index(shape)} 0.1")
            ET.SubElement(p, "freejoint", name=f"part_{shape}_free")
            size = f"{PART_HALF_W[1]} {PART_HALF_H[1]}" if shape == "cylinder" else \
                f"{PART_HALF_L[1]} {PART_HALF_W[1]} {PART_HALF_H[1]}"
            ET.SubElement(p, "geom", name=f"part_{shape}_geom", type=shape, size=size, mass="0.03",
                          material="part_finish", condim="4", friction="1 0.01 0.001", solref="0.01 1")

    # ------------------------------------------------------------- per seed
    def bind_task(self) -> None:
        super().bind_task()
        m = self.world.model
        self.bed = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"belt_{self.belt.name}_bed")
        self.bin = self.body("bin")
        self.part_bodies = {s: self.body(f"part_{s}") for s in SHAPES}
        self.part_geoms = {s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"part_{s}_geom") for s in SHAPES}
        self.part_mat = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, "part_finish")
        self.shape, self.part = "box", self.part_bodies["box"]
        self.speed, self.start_step, self.lateral = SPEED[0], 0, 0.0
        self.half = np.array([PART_HALF_L[1], PART_HALF_W[1], PART_HALF_H[1]])
        self.finish = ""

    def reset_task(self, rng: np.random.Generator) -> None:
        m = self.world.model
        self.speed = float(rng.uniform(*SPEED))
        self.start_step = int(round(rng.uniform(*DELAY) * FPS))
        self.lateral = float(rng.uniform(-LATERAL, LATERAL))
        self.shape = SHAPES[int(rng.integers(len(SHAPES)))]
        self.part = self.part_bodies[self.shape]
        hw, hl, hh = rng.uniform(*PART_HALF_W), rng.uniform(*PART_HALF_L), rng.uniform(*PART_HALF_H)
        g = self.part_geoms[self.shape]
        if self.shape == "cylinder":
            self.half = np.array([hw, hw, hh])
            m.geom_size[g] = (hw, hh, 0)
        else:
            self.half = np.array([hl, hw, hh])  # local x along the belt
            m.geom_size[g] = self.half
        m.geom_aabb[g] = np.r_[0, 0, 0, self.half]
        m.geom_rbound[g] = np.linalg.norm(self.half)
        names = list(FINISHES)
        self.finish = names[int(rng.integers(len(names)))]
        rgb, spec, shine = FINISHES[self.finish]
        m.mat_rgba[self.part_mat] = (*rgb, 1)
        m.mat_specular[self.part_mat] = spec
        m.mat_shininess[self.part_mat] = shine
        m.body_pos[self.bin, :2] = BIN_XY + rng.uniform(-BIN_JITTER, BIN_JITTER, 2)
        xy = self.belt.point(S0, self.lateral)
        self.set_free(f"part_{self.shape}", (xy[0], xy[1], self.belt.height + hh),
                      yaw=self.belt.yaw + rng.uniform(-YAW_JITTER, YAW_JITTER))
        other = SHAPES[1 - SHAPES.index(self.shape)]
        self.set_free(f"part_{other}", (0.6, 1.3, -0.7))  # parked under the table
        self.jitter_look(rng)

    def belt_speed(self, step: int | None = None) -> float:
        """The belt speed during control step ``step`` (default: the next one)."""
        k = self.steps if step is None else step
        return self.speed if k >= self.start_step else 0.0

    def step(self, action: np.ndarray, render: bool = True):
        w = self.world
        self.belt.drive(w.model, w.data, self.bed, self.belt_speed(), exclude=self.robot_bodies)
        return super().step(action, render)

    # ------------------------------------------------------------ geometry
    def part_pos(self) -> np.ndarray:
        return self.world.data.xpos[self.part].copy()

    def along(self) -> float:
        """The part centre's distance downstream of the belt start."""
        return self.belt.coords(self.part_pos())[0]

    def on_belt(self) -> bool:
        d, m = self.world.data, self.world.model
        for c in d.contact[:d.ncon]:
            if self.bed in (c.geom1, c.geom2) and self.part in (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]):
                return True
        return False

    def fell_off(self) -> bool:
        """The part rode past the belt's downstream end and dropped below its top, beside the end."""
        s, lat = self.belt.coords(self.part_pos())
        return bool(s > self.belt.length and abs(lat) < self.belt.width / 2 + 0.04
                    and self.part_pos()[2] < self.belt.height)

    def in_bin(self) -> bool:
        local = self.part_pos() - self.world.data.xpos[self.bin]
        return bool(np.all(np.abs(local[:2]) < BIN_INNER) and local[2] < BIN_WALL_H)

    def settled_part(self) -> bool:
        m = self.world.model
        a = m.jnt_dofadr[m.body_jntadr[self.part]]
        return bool(np.linalg.norm(self.world.data.qvel[a:a + 3]) < MAX_SPEED)

    def success(self) -> bool:
        return (self.in_bin() and self.settled_part() and not self.on_belt()
                and not self.robot_touches(self.part))

    def cube_pos(self) -> np.ndarray:
        return self.part_pos()

    def metrics(self) -> dict[str, Any]:
        return {"shape": self.shape, "finish": self.finish, "speed_mps": round(self.speed, 4),
                "delay_s": round(self.start_step / FPS, 3), "lateral_m": round(self.lateral, 4),
                "in_bin": self.in_bin(), "fell_off": self.fell_off()}


# ---------------------------------------------------------------------------------------------- oracle
DT = 1.0 / FPS
GAP = 0.003  # fixed jaw to the part's near face while the part rides in
FINGER_CLEAR = 0.006  # fingertips at least this far above the belt
GRASP_DEPTH = 0.016  # fingertips this far below the part top, at most
HOVER = 0.04  # hover this far above the grasp height
ENTRY_CLEAR = 0.035  # the part's centre is at least this far upstream when the jaws are down
CLOSE_LEAD = 4  # steps before the part's centre reaches the jaws that the close starts
CLOSE_N = 8
REACH_TOL = 1e-3  # IK miss allowed for an intercept point
END_MARGIN = 0.05  # intercept at least this far before the belt's end
CARRY = 0.07  # part bottom above the table while carrying


class Oracle(TopDownOracle):
    """Scripted line picker (privileged state: belt speed and start, the part's pose). It predicts where
    the part will be from the belt speed and picks the earliest intercept point inside the arm's reach
    that the arm can get to first. It hovers there early with the jaws across the belt, lowers them so
    the part rides in between, closes as the part's centre reaches them, then lifts, places the part in
    the bin and goes home."""

    name = "conveyor oracle (privileged state)"

    def reset(self, env: Env, obs: dict[str, Any], seed: int) -> None:
        super().reset(env, obs, seed)
        self.intercept: float | None = None

    # ----------------------------------------------------------- prediction
    def _eta(self, s: float) -> int:
        """Control steps from now until the part's centre is ``s`` along the belt (belt speed known)."""
        env = self.env
        wait = max(0, env.start_step - env.steps)
        return wait + int(np.ceil(max(0.0, s - env.along()) / (env.speed * DT)))

    def _site_at(self, s: float) -> tuple[np.ndarray, float]:
        """Gripper site (xy) and jaw yaw to grasp the part when its centre is ``s`` along the belt."""
        env, belt = self.env, self.env.belt
        c = belt.point(s, env.lateral)
        u = -belt.lateral if -belt.lateral @ c >= 0 else belt.lateral  # closing axis, pointing outward
        site = c + (FACE - env.half[1] - GAP) * u
        return site, float(np.arctan2(u[1], u[0]))

    def _grasp_z(self) -> float:
        env = self.env
        top = env.belt.height + 2 * env.half[2]
        return top - min(GRASP_DEPTH, 2 * env.half[2] - FINGER_CLEAR)

    def window(self) -> list[float]:
        """Distances along the belt where the arm reaches the grasp and its hover (privileged, cached)."""
        env, z = self.env, self._grasp_z()
        out, q = [], self.cmd.copy()
        for s in np.arange(0.0, env.belt.length - END_MARGIN, 0.01):
            site, yaw = self._site_at(float(s))
            q1 = self._ik(np.r_[site, z + HOVER], yaw, q)
            q2 = self._ik(np.r_[site, z], yaw, q1)
            if (np.linalg.norm(self._fk(q1)[0] - np.r_[site, z + HOVER]) < REACH_TOL
                    and np.linalg.norm(self._fk(q2)[0] - np.r_[site, z]) < REACH_TOL):
                out.append(float(s))
                q = q1
        return out

    # ---------------------------------------------------------------- script
    def _script(self):
        env = self.env
        z = self._grasp_z()
        reach = self.window()
        if not reach:
            raise RuntimeError("the belt is out of the arm's reach")
        # The earliest reachable point the arm can be hovering at, jaws down, before the part arrives.
        plan = None
        for s in reach:
            site, yaw = self._site_at(s)
            hover = self._ik(np.r_[site, z + HOVER], yaw, self.cmd)
            n_move = len(self._joint(hover, GRIP_OPEN, settle=4)) + 30  # the move and the descent
            if self._eta(s - ENTRY_CLEAR) >= n_move and s - ENTRY_CLEAR > env.along():
                plan = (s, site, yaw, hover)
                break
        if plan is None:
            raise RuntimeError("the part reaches the end of the reach window before the arm can")
        s, site, yaw, hover = plan
        self.intercept = s
        self.phase = "hover"
        yield self._joint(hover, GRIP_OPEN, settle=4)
        self.phase = "lower"
        yield self._line(np.r_[site, z], yaw, GRIP_OPEN, 24, settle=2)
        self.phase = "wait"
        wait = self._eta(s) - CLOSE_LEAD
        if wait > 0:
            yield [self.cmd.copy()] * wait
        self.phase = "grasp"
        yield self._grip(GRIP_SHUT, CLOSE_N, settle=6)
        self.phase = "lift"
        held, _ = self._site()
        drop = held[2] - (env.part_pos()[2] - env.half[2])  # site above the part bottom
        yield self._line(np.r_[held[:2], CARRY + drop], yaw, GRIP_SHUT, 24, settle=2)
        self.phase = "carry"
        b = env.world.data.xpos[env.bin][:2]
        s_now, psi = self._site()
        held_xy = rot(-psi) @ (env.part_pos()[:2] - s_now[:2])  # part centre in the jaw frame
        radial = float(np.arctan2(b[1], b[0]))
        target = np.r_[b - rot(radial) @ held_xy, CARRY + drop]
        yield self._joint(self._ik(target, radial, self.cmd), GRIP_SHUT, settle=6)
        self.phase = "lower to bin"
        p0, psi0 = self._fk(self.cmd)
        yield self._line(np.r_[p0[:2], BIN_FLOOR + 0.01 + drop], psi0, GRIP_SHUT, 24, settle=3)
        self.phase = "release"
        yield self._grip(GRIP_OPEN, 10, settle=8)
        self.phase = "retreat"
        p0, psi0 = self._fk(self.cmd)
        yield self._line(np.r_[p0[:2], CARRY + drop + 0.02], psi0, GRIP_OPEN, 20, settle=2)
        self.phase = "home"
        home = np.asarray(env.module.HOME, dtype=np.float64)
        yield self._joint(home, home[5], settle=6, n=45)

