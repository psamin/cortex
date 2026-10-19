"""An industrial arm for the hand-built sim tasks (issue #161): a UR10e with a suction cup or a gripper.

The arm is MuJoCo Menagerie's ``universal_robots_ur10e`` (BSD-3-Clause, ROS-Industrial), downloaded
pinned to a Menagerie commit on first use and cached (``CORTEX_CACHE``, else the temp directory).

``IndustrialEnv`` keeps the sim-task contract of ``base.TaskEnv`` (``reset(seed)``, ``step(action,
render)``, ``success``, ``metrics``, ``instruction``, ``max_steps``, ``world.model``/``world.data``,
obs ``{"pixels": {"front", "wrist"}, "agent_pos"}``), with its own units (``joint_names`` names them):

* action: six joint position targets in degrees, then the tool command 0-100;
* state: six joint positions in degrees, then the tool's reading.

Tools (``IndustrialEnv.effector``):

* "suction": on at >= 50; the reading is the vacuum gauge: 0 off, 50 on without a seal, 100 sealed on a
  box. The cup holds a box through an equality weld (one per box, off by default). It switches on when
  suction is commanded and the cup pad touches the top of a box, at the pose they are in, and off when
  suction is released.
* "gripper": a parallel two-finger gripper. Command and reading follow the SO-101's gripper: 0 closed,
  100 fully open (7 cm between the fingers). The fingers are position controlled with a force limit, so
  a closed command on an object squeezes it.

Arm links and the gripper are gravity compensated (``gravcomp``), as a real controller does.

``ArmOracle`` is the base for scripted oracles on this arm: 6-D damped-least-squares IK to a TCP pose
(``down(yaw)`` points the cup down at a heading), and min-jerk joint motions and Cartesian paths.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.request
import xml.etree.ElementTree as ET  # nosec B405 - our own pinned file and XML we build
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

MENAGERIE = "google-deepmind/mujoco_menagerie"
MENAGERIE_COMMIT = "62ab00f2bc44f6f20c1c9b7a13f3fb1dd41a4e35"
UR10E = "universal_robots_ur10e"

FPS = 30
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
SUCTION_ON = 50.0  # suction command at or above this is on
VAC_OFF, VAC_OPEN, VAC_SEALED = 0.0, 50.0, 100.0  # the vacuum gauge in the state
TOOL_LEN = 0.135  # flange to the cup's sealing face
CUP_R = 0.045
LIGHT_RANGE = (0.7, 1.3)
ELBOW_MIN = np.deg2rad(10.0)  # IK keeps the elbow up
# shoulder_lift + elbow + wrist_1 = 0: the wrist_2 axis points straight down, so the wrist rides above a
# horizontal tool and wrist_2 alone turns its heading (``ArmOracle.posture``)
WRIST_DOWN = np.array([0.0, 1.0, 1.0, 1.0, 0.0, 0.0])
# Parallel gripper (tool frame: z out of the flange, fingers close along y). Command 0-100 like the SO-101's
# gripper: 0 closed, 100 fully open (FINGER_OPEN per finger, 7 cm between them).
PALM_HALF = np.array([0.028, 0.045, 0.026])
PALM_Z = 0.064  # flange to the palm face the fingers stand on
FINGER_OPEN = 0.035
FINGER_HALF = np.array([0.005, 0.0025])  # finger plate half width (x) and half thickness (y)
FINGER_LEN = 0.06
GRIP_TIP = PALM_Z + FINGER_LEN  # flange to the fingertips
GRIP_TCP = GRIP_TIP - 0.012  # the grasp point, between the pads
GRIP_HALF = 30.0  # gripper command at or below this is closed (holding)


def cache_root() -> Path:
    """Where downloaded and generated assets live: ``CORTEX_CACHE``, else <tmp>/cortex_assets."""
    return Path(os.environ.get("CORTEX_CACHE") or Path(tempfile.gettempdir()) / "cortex_assets")


@lru_cache
def ur10e_dir() -> Path:
    """The pinned Menagerie UR10e (XML, meshes, license), downloaded once into the cache."""
    out = cache_root() / f"menagerie-{MENAGERIE_COMMIT[:12]}" / UR10E
    if (out / "ur10e.xml").exists():
        return out
    base = f"https://raw.githubusercontent.com/{MENAGERIE}/{MENAGERIE_COMMIT}/{UR10E}/"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="ur10e-", dir=out.parent))  # same filesystem, so the rename is atomic

    def fetch(rel: str) -> None:
        (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(base + rel, timeout=60) as r:  # nosec B310 - pinned https URL
            (tmp / rel).write_bytes(r.read())

    fetch("ur10e.xml")
    meshes = [m.get("file") for m in ET.parse(tmp / "ur10e.xml").getroot().iter("mesh")]  # nosec B314
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(fetch, ["LICENSE", *(f"assets/{f}" for f in meshes)]))
    try:
        tmp.rename(out)  # atomic: parallel recorders may race to fill the cache
    except OSError:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def quat_yaw(yaw: float) -> str:
    return f"{np.cos(yaw / 2):.6f} 0 0 {np.sin(yaw / 2):.6f}"


def look_at(pos, target, up=(0.0, 0.0, 1.0)) -> str:
    """MJCF ``xyaxes`` for a camera at ``pos`` looking at ``target``."""
    f = np.asarray(target, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    x = np.cross(f, up)
    x /= np.linalg.norm(x)
    y = np.cross(x, f)
    return " ".join(f"{v:.4f}" for v in (*x, *y))


def geom(parent: ET.Element, name: str | None, kind: str, size, pos=(0, 0, 0), material: str | None = None,
         collide: bool = True, **kw: str) -> ET.Element:
    attrs = {"type": kind, "size": " ".join(f"{v:.4f}" for v in np.atleast_1d(size)),
             "pos": " ".join(f"{v:.4f}" for v in pos), **kw}
    if name:
        attrs["name"] = name
    if material:
        attrs["material"] = material
    if not collide:
        attrs.update(contype="0", conaffinity="0", mass="0", group="1")
    return ET.SubElement(parent, "geom", **attrs)


def arm_scene(base_pos: tuple[float, float, float], base_yaw: float = 0.0,
              tool: str = "suction") -> tuple[ET.Element, ET.Element]:
    """The UR10e with a ``tool`` ("suction" or "gripper") and a wrist camera, its base at ``base_pos``;
    returns (mujoco, worldbody)."""
    d = ur10e_dir()
    mj = ET.parse(d / "ur10e.xml").getroot()  # nosec B314
    mj.set("model", "cortex_industrial")
    comp = mj.find("compiler")
    assert comp is not None
    comp.set("meshdir", (d / "assets").as_posix())
    for k in mj.findall("keyframe"):
        mj.remove(k)
    wb = mj.find("worldbody")
    assert wb is not None
    for light in wb.findall("light"):
        wb.remove(light)
    base = wb.find("body")
    assert base is not None and base.get("name") == "base"
    base.set("pos", " ".join(f"{v:.4f}" for v in base_pos))
    base.set("quat", quat_yaw(base_yaw))  # at shoulder pan 0 the arm reaches along the base's +x
    for b in base.iter("body"):
        if b is not base:
            b.set("gravcomp", "1")
    opt = mj.find("option")
    assert opt is not None
    opt.set("timestep", "0.002")
    asset = mj.find("asset")
    assert asset is not None
    for name, rgba in {"tool_steel": "0.62 0.64 0.67 1", "tool_dark": "0.10 0.10 0.11 1",
                       "cup_rubber": "0.05 0.05 0.05 1", "vac_blue": "0.10 0.32 0.62 1"}.items():
        ET.SubElement(asset, "material", name=name, rgba=rgba, specular="0.4", shininess="0.4")

    # The tool on the flange: +z of the attachment site points out of the flange.
    w3 = next(b for b in base.iter("body") if b.get("name") == "wrist_3_link")
    site = w3.find("site[@name='attachment_site']")
    assert site is not None
    body = ET.SubElement(w3, "body", name="tool", pos=site.get("pos", "0 0 0"), quat=site.get("quat", "1 0 0 0"))
    {"suction": _suction_tool, "gripper": _gripper_tool}[tool](mj, body)
    return mj, wb


def _suction_tool(mj: ET.Element, tool: ET.Element) -> None:
    ET.SubElement(tool, "inertial", pos="0 0 0.06", mass="0.9", diaginertia="0.002 0.002 0.001")
    geom(tool, None, "cylinder", (0.045, 0.008), (0, 0, 0.008), "tool_steel", collide=False)
    geom(tool, None, "box", (0.03, 0.03, 0.022), (0, 0, 0.038), "vac_blue", collide=False)  # ejector
    geom(tool, None, "cylinder", (0.012, 0.02), (0.042, 0, 0.038), "tool_dark", collide=False)  # hose fitting
    geom(tool, None, "cylinder", (0.018, 0.03), (0, 0, 0.09), "tool_steel", collide=False)
    for i, (r, z) in enumerate([(0.034, 0.112), (0.040, 0.118), (0.034, 0.124)]):  # bellows
        geom(tool, f"bellow{i}", "cylinder", (r, 0.004), (0, 0, z), "cup_rubber", collide=False)
    geom(tool, "cup", "cylinder", (CUP_R, 0.0035), (0, 0, TOOL_LEN - 0.0035), "cup_rubber",
         friction="1.2 0.02 0.002", mass="0.05")
    ET.SubElement(tool, "site", name="tcp", pos=f"0 0 {TOOL_LEN}", size="0.004", rgba="1 0 0 0", group="4")
    # Wrist camera on the ejector, beside the cup, looking along the tool with a slight inward tilt.
    tilt = np.deg2rad(15.0)
    geom(tool, None, "box", (0.045, 0.012, 0.006), (-0.075, 0, 0.012), "tool_dark", collide=False)  # bracket
    geom(tool, None, "box", (0.018, 0.022, 0.016), (-0.12, 0, 0.04), "tool_dark", collide=False)  # camera
    ET.SubElement(tool, "camera", name="wrist", pos="-0.12 0 0.057", fovy="75",
                  xyaxes=f"0 -1 0 {-np.cos(tilt):.4f} 0 {np.sin(tilt):.4f}")


def _gripper_tool(mj: ET.Element, tool: ET.Element) -> None:
    """A parallel two-finger gripper: two flat fingers on slides across the tool's y axis, coupled by an
    equality so they close symmetrically, driven by one position actuator with a force limit. The
    fingers are thin enough, closed, to push a Jenga block into its slot (1.0 x 1.0 cm tip)."""
    tool.set("gravcomp", "1")
    opt = mj.find("option")
    assert opt is not None
    # Elliptic friction cones with a high impratio keep a squeezed part from creeping in the fingers.
    opt.set("cone", "elliptic")
    opt.set("impratio", "10")
    ET.SubElement(tool, "inertial", pos="0 0 0.04", mass="0.8", diaginertia="0.0015 0.0012 0.001")
    geom(tool, None, "cylinder", (0.04, 0.006), (0, 0, 0.006), "tool_steel", collide=False)  # adapter plate
    geom(tool, "palm", "box", PALM_HALF, (0, 0, PALM_Z - PALM_HALF[2]), "tool_dark")
    geom(tool, None, "box", (0.008, PALM_HALF[1] + 0.002, 0.004), (0, 0, PALM_Z - 0.002), "tool_steel",
         collide=False)  # finger rail
    for side, sign in (("l", 1), ("r", -1)):
        f = ET.SubElement(tool, "body", name=f"finger_{side}", pos=f"0 0 {PALM_Z:.4f}", gravcomp="1")
        ET.SubElement(f, "inertial", pos=f"0 {sign * 0.006:.4f} 0.02", mass="0.05",
                      diaginertia="2e-5 2e-5 5e-6")
        ET.SubElement(f, "joint", name=f"finger_{side}", type="slide", axis=f"0 {sign} 0",
                      range=f"0 {FINGER_OPEN:.4f}", damping="4", armature="0.02")
        geom(f, None, "box", (0.012, 0.007, 0.006), (0, sign * 0.009, 0.002), "tool_steel",
             collide=False)  # carriage
        geom(f, f"finger_{side}", "box", (FINGER_HALF[0], FINGER_HALF[1], FINGER_LEN / 2),
             (0, sign * FINGER_HALF[1], FINGER_LEN / 2), "tool_dark", friction="1.2 0.02 0.002", condim="4",
             solref="0.004 1", solimp="0.95 0.99 0.001")
        geom(f, None, "box", (FINGER_HALF[0] + 0.0002, 0.0004, 0.011),
             (0, sign * 0.0001, FINGER_LEN - 0.012), "cup_rubber", collide=False)  # grip pad
    ET.SubElement(tool, "site", name="tcp", pos=f"0 0 {GRIP_TCP:.4f}", size="0.004", rgba="1 0 0 0", group="4")
    contact = mj.find("contact")
    if contact is None:
        contact = ET.SubElement(mj, "contact")
    ET.SubElement(contact, "exclude", body1="finger_l", body2="finger_r")
    eq = mj.find("equality")
    if eq is None:
        eq = ET.SubElement(mj, "equality")
    ET.SubElement(eq, "joint", joint1="finger_r", joint2="finger_l", polycoef="0 1 0 0 0",
                  solref="0.005 1")
    act = mj.find("actuator")
    assert act is not None
    ET.SubElement(act, "position", name="gripper", joint="finger_l", kp="2000", kv="40",
                  ctrlrange=f"0 {FINGER_OPEN:.4f}", forcerange="-40 40")
    # Wrist camera on top of the palm (tool +x), looking along the fingers, tilted down onto them.
    tilt = np.deg2rad(20.0)
    geom(tool, None, "box", (0.012, 0.02, 0.016), (PALM_HALF[0] + 0.012, 0, 0.03), "tool_dark", collide=False)
    ET.SubElement(tool, "camera", name="wrist", pos=f"{PALM_HALF[0] + 0.02:.4f} 0 0.05", fovy="80",
                  xyaxes=f"0 1 0 {np.cos(tilt):.4f} 0 {np.sin(tilt):.4f}")


class World:
    """Compiled model, data, observation renderer and the arm's indices."""

    def __init__(self, xml: str, height: int, width: int):
        self.model = mujoco.MjModel.from_xml_string(xml)
        m = self.model
        self.substeps = max(1, round(1 / (FPS * m.opt.timestep)))
        m.opt.timestep = 1 / (FPS * self.substeps)
        self.data = mujoco.MjData(m)
        self.renderer = mujoco.Renderer(m, height, width)
        self.joint_ids = [m.joint(f"{j}_joint").id for j in JOINTS]
        self.qadr = np.array([m.jnt_qposadr[j] for j in self.joint_ids])
        self.dofs = np.array([m.jnt_dofadr[j] for j in self.joint_ids])
        self.actuators = np.array([m.actuator(j).id for j in JOINTS])
        self.lo, self.hi = m.jnt_range[self.joint_ids].T.copy()

    def render(self, camera: str) -> np.ndarray:
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render().copy()


class IndustrialEnv:
    """Base for UR10e tasks. Subclasses define ``build`` (add the scene; for suction, the boxes it can
    hold named in ``self.boxes``), ``reset_task`` and ``success``; ``home`` is the arm's start pose
    (degrees), ``effector`` its tool: "suction" or "gripper" (``home_grip`` is the gripper's start)."""

    name = "industrial"
    instruction = ""
    max_steps = 1500
    focus = ""
    effector = "suction"
    base_pos = (0.0, 0.0, 0.0)
    home = np.zeros(6)
    home_grip = 0.0
    view: dict[str, Any] = {}

    @property
    def joint_names(self) -> list[str]:
        return [*JOINTS, self.effector]

    @property
    def robot_type(self) -> str:
        return f"ur10e_{self.effector}"

    def __init__(self, height: int = 96, width: int = 128):
        mj, wb = arm_scene(self.base_pos, tool=self.effector)
        self.boxes: list[str] = []
        self.build(mj, wb)
        eq = mj.find("equality")
        if eq is None:
            eq = ET.SubElement(mj, "equality")
        for b in self.boxes if self.effector == "suction" else []:  # the cup's grip on each box, off until it seals
            ET.SubElement(eq, "weld", name=f"suck_{b}", body1="tool", body2=b, active="false",
                          solref="0.004 1", solimp="0.95 0.99 0.001")
        xml = ET.tostring(mj, encoding="unicode")
        self.xml_hash = hashlib.sha256(xml.encode()).hexdigest()[:12]
        self.height, self.width = height, width
        self.world = World(xml, height, width)
        self._bind()

    # ------------------------------------------------------------ per task
    def build(self, mj: ET.Element, wb: ET.Element) -> None:
        raise NotImplementedError

    def bind_task(self) -> None:
        """Look up the task's ids once the model exists."""

    def reset_task(self, rng: np.random.Generator) -> None:
        raise NotImplementedError

    def success(self) -> bool:
        raise NotImplementedError

    def metrics(self) -> dict[str, Any]:
        return {}

    # -------------------------------------------------------------- binding
    def _bind(self) -> None:
        w = self.world
        m = w.model
        self.tool = m.body("tool").id
        self.tcp = m.site("tcp").id
        self.box_ids = [m.body(b).id for b in self.boxes]
        self.box_geom = {m.body(b).id: m.geom(b).id for b in self.boxes}
        if self.effector == "suction":
            self.cup = m.geom("cup").id
            self.welds = [m.equality(f"suck_{b}").id for b in self.boxes]
        else:
            self.welds = []
            self.finger_qadr = np.array([m.jnt_qposadr[m.joint(f"finger_{s}").id] for s in "lr"])
            self.finger_geoms = {m.geom(f"finger_{s}").id for s in "lr"}
            self.grip_act = m.actuator("gripper").id
        root = m.body_rootid[self.tool]
        self.robot_bodies = {b for b in range(m.nbody) if m.body_rootid[b] == root}
        self.light0 = (m.light_diffuse.copy(), m.vis.headlight.diffuse.copy(), m.vis.headlight.ambient.copy())
        self.body_pos0, self.body_quat0 = m.body_pos.copy(), m.body_quat.copy()
        self.cam_pos0, self.cam_quat0 = m.cam_pos.copy(), m.cam_quat.copy()
        self.geom_size0, self.geom_pos0 = m.geom_size.copy(), m.geom_pos.copy()
        self.attached: int | None = None  # index into self.boxes
        self.suction = False
        self.steps = 0
        self.bind_task()

    # ---------------------------------------------------------------- api
    def reset(self, seed: int, var: Any = None) -> dict[str, Any]:
        w = self.world
        m, d = w.model, w.data
        m.body_pos[:], m.body_quat[:] = self.body_pos0, self.body_quat0
        m.cam_pos[:], m.cam_quat[:] = self.cam_pos0, self.cam_quat0
        m.geom_size[:], m.geom_pos[:] = self.geom_size0, self.geom_pos0
        mujoco.mj_resetData(m, d)
        d.eq_active[self.welds] = 0
        self.attached, self.suction = None, False
        rng = np.random.default_rng(seed)
        q = np.deg2rad(self.home + rng.uniform(-1.0, 1.0, 6))
        d.qpos[w.qadr] = q
        d.ctrl[w.actuators] = q
        if self.effector == "gripper":
            d.qpos[self.finger_qadr] = d.ctrl[self.grip_act] = self.home_grip / 100 * FINGER_OPEN
        self.reset_task(rng)
        self.apply_light(rng.uniform(*LIGHT_RANGE), rng.uniform(-1, 1))
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        self.steps = 0
        return self.obs()

    def state(self) -> np.ndarray:
        """Joints (deg), then the vacuum gauge (suction) or the finger opening 0-100 (gripper)."""
        w = self.world
        if self.effector == "gripper":
            tool = w.data.qpos[self.finger_qadr[0]] / FINGER_OPEN * 100
        else:
            tool = VAC_SEALED if self.attached is not None else VAC_OPEN if self.suction else VAC_OFF
        return np.r_[np.rad2deg(w.data.qpos[w.qadr]), tool].astype(np.float32)

    def obs(self) -> dict[str, Any]:
        w = self.world
        return {"agent_pos": self.state(), "pixels": {"front": w.render("front"), "wrist": w.render("wrist")}}

    def holding(self, action: np.ndarray) -> bool:
        """Whether this command holds (suction on, gripper closed), for exploration noise in ``run.Nudge``."""
        return bool(action[-1] >= SUCTION_ON if self.effector == "suction" else action[-1] <= GRIP_HALF)

    def step(self, action: np.ndarray, render: bool = True):
        w = self.world
        a = np.asarray(action, dtype=np.float64)
        if a.shape != (7,):
            raise ValueError(f"expected 7 values ({', '.join(self.joint_names)}), got shape {a.shape}")
        if self.effector == "gripper":
            w.data.ctrl[self.grip_act] = np.clip(a[6], 0.0, 100.0) / 100 * FINGER_OPEN
        else:
            self._suck(bool(a[6] >= SUCTION_ON))
        w.data.ctrl[w.actuators] = np.clip(np.deg2rad(a[:6]), w.lo, w.hi)
        mujoco.mj_step(w.model, w.data, nstep=w.substeps)
        self.steps += 1
        ok = self.success()
        return (self.obs() if render else None), float(ok), ok, self.steps >= self.max_steps, {"is_success": ok}

    def close(self) -> None:
        self.world.renderer.close()

    # ------------------------------------------------------------ suction
    def _suck(self, on: bool) -> None:
        d = self.world.data
        self.suction = on
        if not on and self.attached is not None:
            d.eq_active[self.welds[self.attached]] = 0
            self.attached = None
        if on and self.attached is None:
            i = self.sealed_box()
            if i is not None:
                self._weld(i)

    def sealed_box(self) -> int | None:
        """The box whose top face the cup pad touches, if any."""
        m, d = self.world.model, self.world.data
        for c in d.contact[:d.ncon]:
            if self.cup not in (c.geom1, c.geom2):
                continue
            other = c.geom2 if c.geom1 == self.cup else c.geom1
            b = m.geom_bodyid[other]
            if b in self.box_ids and other == self.box_geom[b]:
                up = d.xmat[b].reshape(3, 3)[:, 2]
                n = c.frame[:3]
                if abs(n @ up) > 0.9 and d.site_xpos[self.tcp][2] > d.xpos[b][2]:
                    return self.box_ids.index(b)
        return None

    def _weld(self, i: int) -> None:
        """Hold box ``i`` at its current pose relative to the tool."""
        m, d = self.world.model, self.world.data
        b, e = self.box_ids[i], self.welds[i]
        R1 = d.xmat[self.tool].reshape(3, 3)
        rel = R1.T @ (d.xpos[b] - d.xpos[self.tool])
        q1inv, qrel = np.zeros(4), np.zeros(4)
        mujoco.mju_negQuat(q1inv, d.xquat[self.tool])
        mujoco.mju_mulQuat(qrel, q1inv, d.xquat[b])
        m.eq_data[e, :3] = 0  # anchor at body2's origin
        m.eq_data[e, 3:6] = rel
        m.eq_data[e, 6:10] = qrel
        m.eq_data[e, 10] = 1
        d.eq_active[e] = 1
        self.attached = i

    # -------------------------------------------------------------- looks
    def apply_light(self, scale: float, warmth: float = 0.0) -> None:
        """Scale every light; ``warmth`` in [-1, 1] shifts it from cool to warm white."""
        m = self.world.model
        d0, hd, ha = self.light0
        tint = np.array([1 + 0.06 * warmth, 1.0, 1 - 0.08 * warmth])
        m.light_diffuse[:] = np.clip(d0 * scale * tint, 0, 1)
        m.vis.headlight.diffuse[:] = np.clip(hd * scale, 0, 1)
        m.vis.headlight.ambient[:] = np.clip(ha * scale, 0, 1)

    def jitter_camera(self, name: str, rng: np.random.Generator, pos: float = 0.03, rot: float = 0.02) -> None:
        m = self.world.model
        c = m.camera(name).id
        m.cam_pos[c] = self.cam_pos0[c] + rng.uniform(-pos, pos, 3)
        tilt = rng.uniform(-rot, rot, 3)
        dq, q = np.zeros(4), np.zeros(4)
        mujoco.mju_axisAngle2Quat(dq, tilt / (np.linalg.norm(tilt) or 1.0), np.linalg.norm(tilt))
        mujoco.mju_mulQuat(q, self.cam_quat0[c], dq)
        m.cam_quat[c] = q

    # ------------------------------------------------------------ helpers
    def set_free(self, name: str, pos, quat) -> None:
        m, d = self.world.model, self.world.data
        a = m.jnt_qposadr[m.body_jntadr[m.body(name).id]]
        d.qpos[a:a + 3] = pos
        d.qpos[a + 3:a + 7] = quat

    def tcp_pos(self) -> np.ndarray:
        return self.world.data.site_xpos[self.tcp].copy()

    def cup_touches(self, i: int) -> bool:
        d = self.world.data
        g =self.box_geom[self.box_ids[i]]
        return any({c.geom1, c.geom2} == {self.cup, g} for c in d.contact[:d.ncon])

    def fingers_touching(self, body: int) -> int:
        """How many gripper fingers touch ``body``."""
        m, d = self.world.model, self.world.data
        hit = set()
        for c in d.contact[:d.ncon]:
            for a, b in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if a in self.finger_geoms and m.geom_bodyid[b] == body:
                    hit.add(a)
        return len(hit)


# -------------------------------------------------------------------------------------------- oracle
def min_jerk(n: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n + 1)[1:]
    return 10 * t**3 - 15 * t**4 + 6 * t**5


def wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def down(yaw: float) -> np.ndarray:
    """Tool orientation pointing straight down (suction cup) with the tool x axis at heading ``yaw``."""
    return np.array([[np.cos(yaw), np.sin(yaw), 0], [np.sin(yaw), -np.cos(yaw), 0], [0, 0, -1.0]])


def slerp(R0: np.ndarray, R1: np.ndarray, s: float) -> np.ndarray:
    """Rotation a fraction ``s`` of the way from R0 to R1 (about their relative axis)."""
    q0, q1, out = np.zeros(4), np.zeros(4), np.zeros(9)
    mujoco.mju_mat2Quat(q0, R0.ravel())
    mujoco.mju_mat2Quat(q1, R1.ravel())
    if q0 @ q1 < 0:
        q1 = -q1
    w = np.arccos(np.clip(q0 @ q1, -1.0, 1.0))
    q = q0 if w < 1e-9 else (np.sin((1 - s) * w) * q0 + np.sin(s * w) * q1) / np.sin(w)
    mujoco.mju_quat2Mat(out, q / np.linalg.norm(q))
    return out.reshape(3, 3)


class ArmOracle:
    """Base for scripted oracles on ``IndustrialEnv`` (privileged sim state, so a demonstrator, never a
    learned policy). ``_script`` yields lists of 7-D commands; each list plays out, one per step, before
    the script resumes, so every segment is planned from the state it starts in."""

    name = "oracle (privileged state)"
    needs_pixels = False
    deg_per_step = 2.0  # pace of joint moves (largest joint), before min-jerk shaping
    posture: np.ndarray | None = None  # c (6,): the IK also drives c . q to 0 (rad), e.g. WRIST_DOWN
    window: tuple[np.ndarray, np.ndarray] | None = None  # (lo, hi) degrees: joint ranges the IK keeps to

    def reset(self, env: IndustrialEnv, obs: dict[str, Any], seed: int) -> None:
        self.env = env
        self.d = mujoco.MjData(env.world.model)
        self.cmd = np.asarray(obs["agent_pos"], dtype=np.float64).copy()
        if env.effector == "suction":
            self.cmd[6] = 0.0
        self.queue: list[np.ndarray] = []
        self.script = self._script()
        self.done = False
        self.phase = "start"
        self.ik_error = 0.0  # worst TCP miss (m) of any waypoint planned so far: a plan check for tests

    def act(self, obs: dict[str, Any]) -> np.ndarray:
        while not self.queue and not self.done:
            try:
                self.queue = list(next(self.script))
            except StopIteration:
                self.done = True
        if self.queue:
            self.cmd = self.queue.pop(0)
        return self.cmd.astype(np.float32)

    def _script(self) -> Iterator[list[np.ndarray]]:
        raise NotImplementedError

    # ------------------------------------------------------------ kinematics
    def _fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """TCP position and orientation (3 x 3) at joints ``q`` (deg)."""
        w, d = self.env.world, self.d
        d.qpos[:] = w.data.qpos
        d.qpos[w.qadr] = np.deg2rad(q[:6])
        mujoco.mj_kinematics(w.model, d)
        return d.site_xpos[self.env.tcp].copy(), d.site_xmat[self.env.tcp].reshape(3, 3).copy()

    def _ik(self, xyz: np.ndarray, R_goal: np.ndarray, seed: np.ndarray, iters: int = 60) -> np.ndarray:
        """Joints (deg) putting the TCP at ``xyz`` with orientation ``R_goal`` (e.g. ``down(yaw)``).
        Damped least squares from ``seed`` (so consecutive targets stay on one branch), elbow up."""
        w, d, env = self.env.world, self.d, self.env
        m = w.model
        d.qpos[:] = w.data.qpos
        lo, hi = w.lo.copy(), w.hi.copy()
        lo[2] = ELBOW_MIN  # never through the straight-arm singularity onto the other elbow branch
        if self.window is not None:
            lo, hi = np.maximum(lo, np.deg2rad(self.window[0])), np.minimum(hi, np.deg2rad(self.window[1]))
        q = np.clip(np.deg2rad(np.asarray(seed[:6], dtype=np.float64)), lo, hi)
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(iters):
            d.qpos[w.qadr] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            R = d.site_xmat[env.tcp].reshape(3, 3)
            ep = np.asarray(xyz) - d.site_xpos[env.tcp]
            er = 0.5 * (np.cross(R[:, 0], R_goal[:, 0]) + np.cross(R[:, 1], R_goal[:, 1])
                        + np.cross(R[:, 2], R_goal[:, 2]))
            if np.linalg.norm(ep) < 2e-4 and np.linalg.norm(er) < 2e-3:
                break
            mujoco.mj_jacSite(m, d, jp, jr, env.tcp)
            J = np.vstack([jp[:, w.dofs], 0.5 * jr[:, w.dofs]])
            e = np.r_[ep, 0.5 * er]
            if self.posture is not None:  # a soft extra row: settles the redundancy at wrist singularities
                J = np.vstack([J, 0.05 * self.posture])
                e = np.r_[e, -0.05 * (self.posture @ q)]
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(len(e)), e)
            q = np.clip(q + np.clip(dq, -0.2, 0.2), lo, hi)
        return np.rad2deg(q)

    # ------------------------------------------------------------ primitives
    def _joint(self, q: np.ndarray, tool: float, settle: int = 2, n: int | None = None) -> list[np.ndarray]:
        """Min-jerk joint move to ``q`` (deg) with the tool command ``tool`` (suction or gripper)."""
        q1 = np.r_[q[:6], tool]
        n = n or int(np.clip(np.abs(q1[:6] - self.cmd[:6]).max() / self.deg_per_step, 15, 90))
        out = [np.r_[self.cmd[:6] + (q1[:6] - self.cmd[:6]) * s, tool] for s in min_jerk(n)]
        return out + [q1] * settle

    def _path(self, points: list[tuple[np.ndarray, np.ndarray]], tool: float, n: int | None = None,
              settle: int = 2) -> list[np.ndarray]:
        """Through (TCP xyz, orientation) waypoints from the current command, min-jerk in joint-space arc length."""
        qs = [self.cmd[:6]]
        for xyz, R in points:
            qs.append(self._ik(xyz, R, qs[-1]))
            self.ik_error = max(self.ik_error, float(np.linalg.norm(self._fk(qs[-1])[0] - xyz)))
        qs_arr = np.array(qs)
        seg = np.abs(np.diff(qs_arr, axis=0)).max(axis=1)
        s_cum = np.r_[0.0, np.cumsum(seg)]
        total = s_cum[-1]
        n = n or int(np.clip(total / self.deg_per_step, 12, 120))
        out = []
        for s in min_jerk(n):
            x = s * total
            i = int(np.clip(np.searchsorted(s_cum, x, side="right") - 1, 0, len(seg) - 1))
            f = (x - s_cum[i]) / seg[i] if seg[i] > 0 else 1.0
            out.append(np.r_[qs_arr[i] + (qs_arr[i + 1] - qs_arr[i]) * f, tool])
        return out + [out[-1]] * settle

    def _line(self, xyz: np.ndarray, R: np.ndarray, tool: float, n: int | None = None, settle: int = 2,
              step: float = 0.02) -> list[np.ndarray]:
        """Straight TCP line to ``xyz`` while turning to ``R``, IK every ``step`` metres."""
        p0, R0 = self._fk(self.cmd)
        xyz = np.asarray(xyz, dtype=np.float64)
        k = int(np.clip(np.linalg.norm(xyz - p0) / step, 1, 60))
        return self._path([(p0 + (xyz - p0) * s, slerp(R0, R, s)) for s in np.linspace(0, 1, k + 1)[1:]],
                          tool, n, settle)

    def _hold(self, tool: float, n: int) -> list[np.ndarray]:
        return [np.r_[self.cmd[:6], tool]] * n
