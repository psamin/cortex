"""A moving conveyor belt in MuJoCo (issue #255), shared by the ``conveyor`` task and intake scenes.

The belt is a static bed: a box whose top surface is the belt. At every control step, ``Belt.drive``
sets the horizontal velocity of each free body resting on that top (in contact with the bed, with its
centre inside the footprint) to the belt velocity along the belt's axis. Vertical and angular motion are
left to the physics. The bed's contacts are frictionless (``condim=1``, ``priority=1``), so a driven
part keeps the belt velocity for the whole control period instead of being braked by friction between
substeps. A part that passes the belt's downstream end stops being driven, tips over the edge and
falls. That's the standard MuJoCo conveyor trick: stable and deterministic.

Bodies touching the robot aren't driven, so a gripped part moves with the jaws, not the belt.
Units: metres, metres per second; xy in the world (robot) frame.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET  # nosec B405 - we build this XML ourselves
from collections.abc import Iterable
from dataclasses import dataclass

import mujoco
import numpy as np

BED_RGBA = "0.10 0.10 0.11 1"  # black rubber belt
FRAME_RGBA = "0.55 0.57 0.60 1"  # steel side rails
RAIL_W = 0.006  # side rails beside the belt (visual only)


@dataclass
class Belt:
    """A straight belt from ``start`` to ``end`` (xy, the direction of travel), ``width`` across and its
    top surface ``height`` above the table, moving at ``speed`` m/s."""

    name: str
    start: tuple[float, float]
    end: tuple[float, float]
    width: float
    height: float
    speed: float = 0.05

    @property
    def axis(self) -> np.ndarray:
        d = np.subtract(self.end, self.start)
        return d / np.linalg.norm(d)

    @property
    def lateral(self) -> np.ndarray:
        """Unit vector across the belt, to the left of the direction of travel."""
        a = self.axis
        return np.array([-a[1], a[0]])

    @property
    def length(self) -> float:
        return float(np.linalg.norm(np.subtract(self.end, self.start)))

    @property
    def yaw(self) -> float:
        a = self.axis
        return float(np.arctan2(a[1], a[0]))

    def point(self, s: float, lateral: float = 0.0) -> np.ndarray:
        """The xy point ``s`` metres downstream of the start, ``lateral`` metres to the left."""
        return np.asarray(self.start, dtype=np.float64) + s * self.axis + lateral * self.lateral

    def coords(self, xy: np.ndarray) -> tuple[float, float]:
        """(along, lateral) of an xy point in the belt frame."""
        r = np.asarray(xy[:2], dtype=np.float64) - np.asarray(self.start)
        return float(r @ self.axis), float(r @ self.lateral)

    def over(self, xy: np.ndarray) -> bool:
        s, lat = self.coords(xy)
        return 0.0 <= s <= self.length and abs(lat) <= self.width / 2

    # ---------------------------------------------------------------- scene
    def add_bed(self, wb: ET.Element, rgba: str = BED_RGBA, rails: bool = True) -> str:
        """Add the bed (and visual side rails) to a worldbody; returns the bed geom's name."""
        c = (np.asarray(self.start) + np.asarray(self.end)) / 2
        q = (np.cos(self.yaw / 2), 0.0, 0.0, np.sin(self.yaw / 2))
        quat = " ".join(f"{v:.6f}" for v in q)
        name = f"belt_{self.name}"
        body = ET.SubElement(wb, "body", name=name, pos=f"{c[0]:.5f} {c[1]:.5f} 0", quat=quat)
        ET.SubElement(body, "geom", name=f"{name}_bed", type="box", rgba=rgba, condim="1", priority="1",
                      pos=f"0 0 {self.height / 2:.5f}",
                      size=f"{self.length / 2:.5f} {self.width / 2:.5f} {self.height / 2:.5f}")
        if rails:
            for s in (-1, 1):
                ET.SubElement(body, "geom", name=f"{name}_rail{s:+d}", type="box", rgba=FRAME_RGBA,
                              contype="0", conaffinity="0",
                              pos=f"0 {s * (self.width / 2 + RAIL_W / 2):.5f} {self.height / 2 + 0.002:.5f}",
                              size=f"{self.length / 2:.5f} {RAIL_W / 2:.5f} {self.height / 2 + 0.002:.5f}")
        return f"{name}_bed"

    # -------------------------------------------------------------- physics
    def riders(self, model: mujoco.MjModel, data: mujoco.MjData, bed: int,
               exclude: Iterable[int] = ()) -> list[int]:
        """Free bodies resting on the belt: in contact with the bed geom, centre inside the footprint,
        and not touching any body in ``exclude`` (the robot)."""
        skip = set(exclude)
        on, touched = set(), set()
        for c in data.contact[:data.ncon]:
            b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
            if c.geom1 == bed:
                on.add(b2)
            elif c.geom2 == bed:
                on.add(b1)
            if b1 in skip:
                touched.add(b2)
            if b2 in skip:
                touched.add(b1)
        out = []
        for b in sorted(on - touched - skip):
            j = model.body_jntadr[b]
            if j >= 0 and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and self.over(data.xpos[b]):
                out.append(int(b))
        return out

    def drive(self, model: mujoco.MjModel, data: mujoco.MjData, bed: int, speed: float | None = None,
              exclude: Iterable[int] = ()) -> list[int]:
        """Set the horizontal velocity of every rider to the belt velocity; returns the riders."""
        v = (self.speed if speed is None else speed) * self.axis
        riders = self.riders(model, data, bed, exclude)
        for b in riders:
            dof = model.jnt_dofadr[model.body_jntadr[b]]
            data.qvel[dof:dof + 2] = v
        return riders
