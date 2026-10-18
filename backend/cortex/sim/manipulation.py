"""Typed, validated manipulation API over PickPlaceEnv for planner agents.

A planner (GPT-6 Astra, or any other) chooses *semantic* actions and waypoints;
this module validates them and turns them into rate-limited joint trajectories
with the existing damped-least-squares IK. Every simulator step it executes is
recorded (synchronized camera frames, joint state, joint-target action), so a
successful episode is a genuine timestep-level demonstration in the same format
as the scripted oracle's.

Two planning modes:
  camera  (default) the planner sees rendered camera images and robot
          proprioception only. Object positions come from pixels it points at,
          back-projected through the calibrated overhead camera onto the table
          plane and refined locally in the image (color blob around the point).
  privileged  observations also include simulator ground-truth object poses.
          Always labelled "privileged-state planning".

The tools are generic over the task object: on the upstream tabletop it is the red cube and the
box; on an intake scene (``SceneEnv``) it is the scene's pick object and its named place zones.
Short objects are grasped from above, tall ones from the side (``sim/grasp.py``), reusing the
verified offsets and IK helpers of ``sim/pick_place_skill.py``.

Task success is always judged by the simulator (ground truth), never by the model.
"""

from __future__ import annotations

import base64
import io
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import mujoco
import numpy as np
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field, ValidationError

from .expert import ARM, approach_error_deg, solve_ik
from .grasp import approaches, grasp_height, grasp_mode
from .pick_place_skill import LIFT, PickPlaceSkill
from .so101 import PickPlaceEnv
from .viewer import CAM_EVERY

PLAN_CAM, PLAN_PX = "top", 480
OBSERVE_POSE = np.array([0.0, -100.0, 0.0, -90.0, 0.0, 30.0], np.float32)  # arm vertical: clear overhead view
GRIP_OPEN, GRIP_CLOSED = 60.0, 0.0
HOLDING_MIN = 5.0  # gripper stalls above this when closed on an object
WORKSPACE = {"x": (0.08, 0.36), "y": (-0.22, 0.24), "z": (0.006, 0.20)}
CUBE_Z, CARRY_Z, RELEASE_Z, RETREAT_Z = 0.015, 0.075, 0.055, 0.072
MAX_REACH_ERR = 0.012
MOTION_TIMEOUT_STEPS = 150
EPISODE_MAX_STEPS = 900  # 30 s of sim time
SIDE_EPISODE_MAX_STEPS = 2400  # 80 s: side grasps move slowly along straight lines near the object
ARM_DEG_PER_STEP, GRIP_PER_STEP = 4.0, 8.0
CONTACT_DEG_PER_STEP = 2.0  # straight-line moves with an approach axis (side grasps), as in the skill
LINE_STEP_M = 0.01  # waypoint spacing of straight-line moves
VERTICAL_XY_M = 0.02  # top grasps: a move that shifts the gripper less than this sideways is a straight line
LINE_AXIS_TOL_DEG = 5.0  # straight lines only when the gripper already points along the axis; else a joint move
MAX_AXIS_ERR_DEG = 10.0  # an approach-axis IK solution must point within this of the wanted axis
HOME_TOL_DEG = 2.0
# Side grasps, in the physical gripper-site frame. Measured on the scripted skill's verified small-bottle runs
# (#94, seeds 1 and 4): with the object centred in the open jaws (OPEN_GAP_OFFSET) the site sits 2.0 cm past the
# object's centre along the approach and in line with it; the slide to the fixed pad (SLIDE_DEPTH, pad_z) moves
# it 8 mm sideways along the jaws' closing axis.
SIDE_TIP_PAST_CENTRE, SIDE_SLIDE_LATERAL = 0.020, 0.008
SIDE_XY_MARGIN = 0.04
HUE_TOL_DEG, MIN_SAT, MIN_VAL = 35.0, 20, 20  # object-colour segmentation (OpenCV HSV: S, V in 0-255)


@dataclass(frozen=True)
class TaskObject:
    """The object the tools pick: its name, size and colour come from the scene (or the upstream red cube)."""

    name: str
    height: float
    shape: str
    width: float
    rgba: tuple[float, ...] | None  # None: the upstream cube, segmented by saturation

    @property
    def mode(self) -> str:
        return grasp_mode(self.height)

    @property
    def grasp_z(self) -> float:
        return grasp_height(self.height)

    @property
    def label(self) -> str:
        return self.name.replace("_", " ")

    def describe(self) -> str:
        dims = (f"{self.width * 100:.1f} cm wide" if self.shape == "sphere"
                else f"{self.height * 100:.1f} cm tall, {self.width * 100:.1f} cm wide")
        return f"the {self.label} ({self.shape}, {dims}; {self.mode} grasp)"


def task_object(env: PickPlaceEnv) -> TaskObject:
    info = getattr(env, "info", None)
    if info is None:
        return TaskObject("red_cube", 0.03, "box", 0.03, None)
    m = env.world.model
    g = min(env.obj_geoms)  # type: ignore[attr-defined]
    size = m.geom_size[g]
    shape = env.shape  # type: ignore[attr-defined]
    width = float(2 * min(size[:2]) if shape == "box" else 2 * size[0])
    return TaskObject(env.pick, float(env.obj_h), shape, width,  # type: ignore[attr-defined]
                      tuple(float(c) for c in m.geom_rgba[g]))


def color_mask(hsv: np.ndarray, rgba: tuple[float, ...]) -> np.ndarray:
    """Pixels whose hue is near the object's colour. Hue survives the lighting scale (dim/normal), and
    a minimum saturation keeps out the near-white table. Achromatic objects fall back to brightness."""
    import cv2

    ref = cv2.cvtColor(np.uint8([[[round(c * 255) for c in rgba[:3]]]]), cv2.COLOR_RGB2HSV)[0, 0].astype(int)
    if ref[1] < 40:  # grey/white/black object: no usable hue; match value instead
        return ((np.abs(hsv[..., 2].astype(int) - ref[2]) < 40) & (hsv[..., 1] < 60)).astype(np.uint8)
    dh = np.abs(hsv[..., 0].astype(int) - ref[0])
    dh = np.minimum(dh, 180 - dh) * 2.0  # OpenCV hue is 0-179 (2 degrees per unit)
    return ((dh < HUE_TOL_DEG) & (hsv[..., 1] > MIN_SAT) & (hsv[..., 2] > MIN_VAL)).astype(np.uint8)


def _norm_name(s: str) -> str:
    return s.strip().lower().replace(" ", "_").replace("-", "_")


def _key(xyz: np.ndarray) -> tuple[float, ...]:
    return tuple(round(float(v), 3) for v in np.round(np.asarray(xyz, dtype=np.float64), 4))


def _unit(v: list[float] | np.ndarray) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(a))
    if n < 0.5 or n > 1.5:
        raise ToolError("bad_approach", "approach must be a unit vector (use the one propose_grasp returns)",
                        approach=[round(float(x), 4) for x in a])
    return a / n


class ToolError(Exception):
    def __init__(self, code: str, message: str, **details: Any):
        super().__init__(message)
        self.code, self.message, self.details = code, message, details


# ------------------------------------------------------------- tool schemas

class ImagePoint(BaseModel):
    u: float = Field(..., ge=0, le=PLAN_PX - 1, description="pixel column in the overhead image (0 = left)")
    v: float = Field(..., ge=0, le=PLAN_PX - 1, description="pixel row in the overhead image (0 = top)")


class Pose(BaseModel):
    x: float = Field(..., description="metres, robot base frame, +x forward")
    y: float = Field(..., description="metres, +y to the robot's left")
    z: float = Field(..., description="metres above the table surface")
    yaw_deg: float | None = Field(None, ge=-180, le=180, description="gripper yaw about vertical; null = keep")
    approach: list[float] | None = Field(None, min_length=3, max_length=3,
                                         description="gripper approach axis (unit vector, base frame) for side "
                                                     "grasps; null = straight down (top grasp). Use the value "
                                                     "propose_grasp returns.")


class ObserveArgs(BaseModel):
    camera: Literal["overhead", "front", "wrist"] = "overhead"


class ProposeGraspArgs(BaseModel):
    object_id: str | None = Field(None, max_length=64, description="the task object (default: the object to pick)")
    image_point: ImagePoint | None = Field(None, description="where the object is in the latest overhead image "
                                                             "(required in camera mode)")


class MoveArgs(BaseModel):
    target: Pose
    reason: str = Field("", max_length=200)


class GripperArgs(BaseModel):
    state: Literal["open", "closed"]


class PlaceArgs(BaseModel):
    zone: str | None = Field(None, max_length=64, description="a named place zone from the task")
    image_point: ImagePoint | None = Field(None, description="center of the placement target in the overhead image")
    target: Pose | None = Field(None, description="explicit placement point instead of an image point")


class FinishArgs(BaseModel):
    believes_success: bool
    summary: str = Field("", max_length=400)


class Empty(BaseModel):
    pass


# ------------------------------------------------------------- session

@dataclass
class Recorder:
    front: list[np.ndarray] = field(default_factory=list)
    wrist: list[np.ndarray] = field(default_factory=list)
    state: list[np.ndarray] = field(default_factory=list)
    action: list[np.ndarray] = field(default_factory=list)
    reward: list[float] = field(default_factory=list)
    video: list[np.ndarray] = field(default_factory=list)
    hd: dict[str, list[np.ndarray]] = field(default_factory=dict)  # display copies of front/wrist, every CAM_EVERY

    def __len__(self) -> int:
        return len(self.action)


class ManipulationSession:
    def __init__(self, env: PickPlaceEnv, mode: Literal["camera", "privileged"] = "camera",
                 on_frame: Any = None, video_renderer: Any = None, cam_renderer: Any = None):
        self.env, self.mode = env, mode
        self.on_frame = on_frame  # called with each preview frame (live view)
        self.video_renderer = video_renderer
        self.cam_renderer = cam_renderer  # cortex.sim.viewer.PolicyCameras: sharp copies for the viewer only
        self.rec = Recorder()
        self.obs: dict[str, Any] | None = None
        self.cmd = np.zeros(6)
        self._plan_renderer: mujoco.Renderer | None = None
        self.last_overhead: np.ndarray | None = None
        self.holding = False
        self.counters = {"tool_calls": 0, "tool_errors": 0, "grasp_failures": 0}
        self.obj: TaskObject | None = None
        self.max_steps = EPISODE_MAX_STEPS
        self._q_hint: dict[tuple[float, ...], np.ndarray] = {}  # IK solutions of the poses propose_grasp returned
        self._skill: PickPlaceSkill | None = None
        self._last_ik_err = 0.0
        self._last_axis: tuple[np.ndarray, np.ndarray] | None = None  # (arm joints, approach axis) of the last move
        if env.world is not None:
            self._bind_object()

    def _bind_object(self) -> None:
        self.obj = task_object(self.env)
        self.max_steps = EPISODE_MAX_STEPS if self.obj.mode == "top" else SIDE_EPISODE_MAX_STEPS
        self._skill = PickPlaceSkill(self.env, height=self.obj.height, grasp_width=self.obj.width)

    # ---------------- task description (object and zones), for prompts and tool descriptions
    def zones(self) -> dict[str, tuple[tuple[float, float], tuple[float, float]] | None]:
        info = getattr(self.env, "info", None)
        if info is None:
            return {"box": None}  # the upstream box: its position is only known from the image or ground truth
        return dict(info.zones)

    def target_zone(self) -> str:
        return getattr(self.env, "target_name", None) or next(iter(self.zones()), "box")

    def task_context(self) -> dict[str, Any]:
        assert self.obj is not None
        return {"object": self.obj.name, "object_text": self.obj.describe(), "grasp_mode": self.obj.mode,
                "zones": list(self.zones()), "target_zone": self.target_zone(),
                "scene": "intake" if getattr(self.env, "info", None) is not None else "tabletop"}

    # ---------------- episode boundary (owned by the collector, not the planner)
    def reset_environment(self, seed: int, variation) -> dict[str, Any]:
        self.obs = self.env.reset(seed, variation)
        self.cmd = self.obs["agent_pos"].astype(np.float64).copy()
        self.rec = Recorder()
        self.holding = False
        self.last_overhead = None
        self._q_hint = {}
        self._last_axis = None
        self._bind_object()
        return {"seed": seed}

    # ---------------- low level: drive joint targets, record every step
    def _step(self, target: np.ndarray) -> None:
        assert self.obs is not None
        if len(self.rec) >= self.max_steps:
            raise ToolError("episode_time_limit", f"episode exceeded {self.max_steps} steps "
                                                  f"({self.max_steps // 30} s)")
        action = target.astype(np.float32)
        self.rec.front.append(self.obs["pixels"]["front"])
        self.rec.wrist.append(self.obs["pixels"]["wrist"])
        self.rec.state.append(self.obs["agent_pos"])
        self.rec.action.append(action)
        self.obs, reward, _, _, _ = self.env.step(action)
        self.rec.reward.append(float(reward))
        if self.video_renderer is not None:  # cortex.sim.viewer.DemoCamera (presentation only)
            frame = self.video_renderer.render(self.env.world.data)
            self.rec.video.append(frame)
            if self.on_frame:
                self.on_frame(frame)
        if self.cam_renderer is not None and (len(self.rec) - 1) % CAM_EVERY == 0:
            for k, f in self.cam_renderer.render(self.env.world.data).items():
                self.rec.hd.setdefault(k, []).append(f)

    def _drive(self, q_target: np.ndarray, timeout: int = MOTION_TIMEOUT_STEPS, settle: int = 3,
               rate: float = ARM_DEG_PER_STEP, measured: bool = False) -> int:
        steps, held = 0, 0
        while steps < timeout:
            delta = q_target - self.cmd
            step = np.clip(delta, -rate, rate)
            step[5] = np.clip(delta[5], -GRIP_PER_STEP, GRIP_PER_STEP)
            self.cmd = self.cmd + step
            self._step(self.cmd)
            steps += 1
            arrived = not measured or np.abs(np.asarray(self.env.world.joint_state())[:ARM]
                                             - q_target[:ARM]).max() < 0.7
            if np.abs(q_target - self.cmd).max() < 0.5 and arrived:
                held += 1
                if held >= settle:
                    return steps
        raise ToolError("motion_timeout", f"motion did not converge within {timeout} steps", steps=steps)

    # ---------------- geometry helpers
    def ee_pos(self) -> np.ndarray:
        return self.env.world.data.site_xpos[self.env.site].copy()

    def gripper_opening(self) -> float:
        return float(self.obs["agent_pos"][5])  # type: ignore[index]

    def _camera(self, name: str) -> tuple[np.ndarray, np.ndarray, float]:
        m, d = self.env.world.model, self.env.world.data
        cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, name)
        f = (PLAN_PX / 2) / np.tan(np.deg2rad(m.cam_fovy[cam]) / 2)
        return d.cam_xpos[cam].copy(), d.cam_xmat[cam].reshape(3, 3).copy(), f

    def pixel_to_table(self, u: float, v: float, z: float | None = None) -> np.ndarray:
        """Back-project an overhead pixel onto the horizontal plane at height z (default: the task object's
        grasp height, which is CUBE_Z for the cube)."""
        if z is None:
            z = self.obj.grasp_z if self.obj is not None else CUBE_Z
        p, R, f = self._camera(PLAN_CAM)
        d = R @ np.array([(u - PLAN_PX / 2) / f, -(v - PLAN_PX / 2) / f, -1.0])
        if abs(d[2]) < 1e-6:
            raise ToolError("bad_pixel", "ray parallel to table")
        t = (z - p[2]) / d[2]
        return p + t * d

    def table_to_pixel(self, xyz: np.ndarray) -> tuple[float, float]:
        p, R, f = self._camera(PLAN_CAM)
        c = R.T @ (np.asarray(xyz) - p)
        return PLAN_PX / 2 + f * c[0] / -c[2], PLAN_PX / 2 - f * c[1] / -c[2]

    def _render(self, camera: str) -> np.ndarray:
        if self._plan_renderer is None:
            self._plan_renderer = mujoco.Renderer(self.env.world.model, PLAN_PX, PLAN_PX)
        cam = {"overhead": PLAN_CAM, "front": "front", "wrist": "wrist_cam"}[camera]
        self._plan_renderer.update_scene(self.env.world.data, camera=cam)
        return self._plan_renderer.render().copy()

    def refine_blob(self, img: np.ndarray, u: float, v: float, radius: int = 45,
                    rgba: tuple[float, ...] | None = None) -> dict[str, Any]:
        """Camera-only refinement: segment the color region under the point and
        return its centroid and in-plane orientation (image evidence only).

        With ``rgba`` (the object's colour from the scene) pixels are kept by hue distance to that colour;
        without it (the upstream red cube) by high saturation."""
        import cv2

        h, w = img.shape[:2]
        u0, v0 = int(round(u)), int(round(v))
        x0, x1, y0, y1 = max(0, u0 - radius), min(w, u0 + radius), max(0, v0 - radius), min(h, v0 + radius)
        hsv = cv2.cvtColor(np.ascontiguousarray(img[y0:y1, x0:x1]), cv2.COLOR_RGB2HSV)
        if rgba is None:
            # Saturated (colored) regions against the low-saturation tabletop; saturation is
            # invariant to the lighting scale, so this works in dim and bright scenes alike.
            mask = ((hsv[..., 1] > 115) & (hsv[..., 2] > 20)).astype(np.uint8)
        else:
            mask = color_mask(hsv, rgba)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        best, best_d = 0, np.inf
        for lab in range(1, n):
            if stats[lab, cv2.CC_STAT_AREA] < 30:
                continue
            ys, xs = np.nonzero(labels == lab)
            dist = np.min(np.hypot(xs - (u0 - x0), ys - (v0 - y0)))
            if dist < best_d:
                best, best_d = lab, dist
        if best == 0 or best_d > 20:
            return {"refined": False, "u": u, "v": v, "pixels": 0}
        ys, xs = np.nonzero(labels == best)
        pts = np.stack([xs + x0, ys + y0], axis=1).astype(np.float32)
        (cx, cy), (rw, rh), ang = cv2.minAreaRect(pts)
        return {"refined": True, "u": float(cx), "v": float(cy), "pixels": int(len(xs)),
                "rect_deg": float(ang), "rect_px": [float(rw), float(rh)]}

    # ---------------- validation
    def _validate_pose(self, p: Pose, allow_low: bool = False) -> None:
        side = self.obj is not None and self.obj.mode == "side"
        for axis in ("x", "y", "z"):
            lo, hi = WORKSPACE[axis]
            if axis == "z":
                hi = self._z_max()
            elif side:  # side grasps hold the gripper tip past the object's centre, along the approach
                lo, hi = lo - SIDE_XY_MARGIN, hi + SIDE_XY_MARGIN
            val = getattr(p, axis)
            if not lo <= val <= hi:
                raise ToolError("outside_workspace", f"{axis}={val:.3f} m outside [{lo}, {hi}]",
                                axis=axis, value=val, limits=[lo, hi])
        if p.z < 0.008 and not allow_low:
            raise ToolError("table_collision", "target is below the 8 mm table clearance", z=p.z)

    def _z_max(self) -> float:
        """Tall objects need room above them (the side-grasp transit waypoint); the cube keeps 0.20 m."""
        return max(WORKSPACE["z"][1], (self.obj.height if self.obj else 0.0) + 0.12)

    def _ik(self, p: Pose) -> np.ndarray:
        if p.approach is not None:
            return self._ik_axis(np.array([p.x, p.y, p.z]), _unit(p.approach))
        yaw = np.deg2rad(p.yaw_deg) if p.yaw_deg is not None else None
        target = np.array([p.x, p.y, p.z])
        q, err = solve_ik(self.env, target, self.cmd, yaw)
        if err > MAX_REACH_ERR:
            # DLS IK is local: a large base rotation (cube on the right -> box on the left) can stall in a
            # local minimum. Retry from base yaw aimed at the target and from home; among solutions within
            # tolerance prefer the one nearest the current joints (smallest, safest motion).
            az = float(np.rad2deg(np.arctan2(p.y, p.x)))
            seeds = []
            for pan in (az, -az):
                s = self.cmd.copy()
                s[0] = pan
                seeds.append(s)
            seeds.append(np.array([0.0, -60.0, 60.0, 35.0, 0.0, self.cmd[5]]))
            sols = [solve_ik(self.env, target, s, yaw) for s in seeds]
            ok = [(qq, ee) for qq, ee in sols if ee <= MAX_REACH_ERR]
            if ok:
                q, err = min(ok, key=lambda r: float(np.abs(r[0] - self.cmd[:ARM]).sum()))
        if err > MAX_REACH_ERR:
            raise ToolError("unreachable", f"IK residual {err * 1000:.0f} mm exceeds {MAX_REACH_ERR * 1000:.0f} mm",
                            residual_m=round(err, 4))
        return self._check_limits(q)

    def _check_limits(self, q: np.ndarray) -> np.ndarray:
        w = self.env.world
        lo, hi = np.rad2deg(w.lo[:ARM]), np.rad2deg(w.hi[:ARM])
        if np.any(q < lo - 1e-3) or np.any(q > hi + 1e-3):
            raise ToolError("joint_limit", "solution violates joint limits", q=q.round(1).tolist())
        return q

    def _check_path_collisions(self, q_target_deg: np.ndarray, start_deg: np.ndarray | None = None) -> None:
        """Kinematic sweep: arm geoms must not penetrate the table or box walls."""
        w = self.env.world
        m = w.model
        d = mujoco.MjData(m)
        d.qpos[:] = w.data.qpos
        free = {int(m.jnt_bodyid[j]) for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE}
        arm_bodies = set(range(m.nbody)) - {0, w.box_body, *w.cube_bodies} - free  # objects are not the arm
        statics = {g for g in range(m.ngeom) if m.geom_bodyid[g] in (0, w.box_body)}
        start = (self.cmd[:ARM] if start_deg is None else np.asarray(start_deg)[:ARM]).copy()
        for s in np.linspace(0, 1, 6)[1:]:
            q = start + s * (q_target_deg[:ARM] - start)
            d.qpos[w.qadr[:ARM]] = np.deg2rad(q)
            mujoco.mj_forward(m, d)
            for i in range(d.ncon):
                c = d.contact[i]
                pair = {c.geom1, c.geom2}
                if c.dist < -0.003 and pair & statics and any(m.geom_bodyid[g] in arm_bodies for g in pair):
                    other = next(g for g in pair if g in statics)
                    raise ToolError("collision", f"path collides with {m.geom(other).name or 'box/table'} "
                                    f"at {s:.0%} of the motion", fraction=round(float(s), 2))

    def _fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        w = self.env.world
        d = mujoco.MjData(w.model)
        d.qpos[:] = w.data.qpos
        d.qpos[w.qadr[:ARM]] = np.deg2rad(q[:ARM])
        mujoco.mj_kinematics(w.model, d)
        return d.site_xpos[self.env.site].copy(), d.site_xmat[self.env.site].reshape(3, 3).copy()

    def _ik_axis(self, xyz: np.ndarray, axis: np.ndarray, seed: np.ndarray | None = None,
                 tol: float = MAX_REACH_ERR, use_hint: bool = True) -> np.ndarray:
        """IK with the gripper's approach axis along ``axis`` (side grasps) and the deterministic jaw roll.

        ``solve_ik`` measures its residual before it sets the jaw roll, and the roll moves the gripper site
        sideways by up to ~2.5 cm, so the target is corrected (fixed-point, as the skill's ``_ik_obj``) until
        the site lands on ``xyz`` *with* that roll. The residual checked is the physical one.
        Seeds: a solution propose_grasp already computed for this pose, the given seed, the current joints,
        then the skill's elbow-up/down seeds."""
        assert self._skill is not None
        hint = self._q_hint.get(_key(xyz)) if use_hint else None
        for s, only in ((hint, True), (seed, True), (self.cmd, True), (None, False)):
            if only and s is None:
                continue
            tgt, q = np.asarray(xyz, dtype=np.float64).copy(), None if s is None else np.asarray(s, np.float64)
            for i in range(5):
                _, q = self._skill._ik(tgt, axis, q, only_seed=only or i > 0)
                if q is None:
                    break
                miss = xyz - self._fk(q)[0]
                if np.linalg.norm(miss) < 1e-3:
                    break
                tgt = tgt + miss
            if q is None:
                continue
            e = float(np.linalg.norm(xyz - self._fk(q)[0]))
            if e <= tol and approach_error_deg(self.env, q, axis) <= MAX_AXIS_ERR_DEG:
                self._last_ik_err = e
                return self._check_limits(q)
        raise ToolError("unreachable", f"no IK solution within {tol * 1000:.0f} mm with the gripper along the "
                        f"approach axis {np.round(axis, 3).tolist()}", approach=np.round(axis, 3).tolist())

    def _line_plan(self, goal: np.ndarray, axis: np.ndarray) -> list[np.ndarray]:
        """Joint waypoints for a straight-line move of the gripper tip, with the approach axis held. Every
        waypoint is solved and every segment swept for collisions before the arm moves."""
        start = self.ee_pos()
        n = max(2, int(np.ceil(np.linalg.norm(goal - start) / LINE_STEP_M)))
        qs, prev = [], self.cmd[:ARM].copy()
        for t in np.linspace(0, 1, n + 1)[1:]:
            q = self._ik_axis(start + (goal - start) * t, axis, seed=prev, tol=MAX_REACH_ERR if t == 1 else 0.02,
                              use_hint=False)  # stay on the continuous IK branch along the line
            self._check_path_collisions(q, start_deg=prev)
            qs.append(q)
            prev = q
        return qs

    def _move_axis(self, goal: np.ndarray, axis: np.ndarray, grip: float | None = None) -> int:
        """Side-grasp motion: a straight line when the gripper already points along the axis (approach,
        slide, lift, lower, retreat), otherwise a validated joint move (transit, carry). Slow, like the skill."""
        grip = self.cmd[5] if grip is None else grip
        last = self._last_axis  # the axis of the previous side-grasp move, if the arm has not moved since
        aligned = (last is not None and np.allclose(last[0], self.cmd[:ARM], atol=0.5)
                   and float(np.rad2deg(np.arccos(np.clip(last[1] @ axis, -1, 1)))) < LINE_AXIS_TOL_DEG)
        if aligned and np.linalg.norm(goal - self.ee_pos()) < 0.15:
            qs = self._line_plan(goal, axis)
            steps = 0
            for i, q in enumerate(qs):
                steps += self._drive(np.r_[q, grip], rate=CONTACT_DEG_PER_STEP,
                                     settle=3 if i == len(qs) - 1 else 1, measured=True)
        else:
            q = self._ik_axis(goal, axis)
            target = np.r_[q, grip]
            self._check_path_collisions(target)
            steps = self._drive(target, rate=CONTACT_DEG_PER_STEP if self.holding else ARM_DEG_PER_STEP, timeout=240)
        self._last_axis = (self.cmd[:ARM].copy(), axis.copy())
        return steps

    def _move_vertical(self, p: Pose) -> int:
        """Top grasps: a straight line to ``p`` (descend to the grasp, lift, lower, retreat). One joint move turns
        each joint at its own rate limit, so the fingertips dip to the table short of the object and sweep it away
        before the jaws close (#225). Every waypoint is solved and swept for collisions before the arm moves."""
        start, goal = self.ee_pos(), np.array([p.x, p.y, p.z])
        yaw = np.deg2rad(p.yaw_deg) if p.yaw_deg is not None else None
        n = max(1, int(np.ceil(np.linalg.norm(goal - start) / LINE_STEP_M)))
        qs, prev = [], self.cmd[:ARM].copy()
        for t in np.linspace(0, 1, n + 1)[1:]:
            q, err = solve_ik(self.env, start + (goal - start) * t, prev, yaw)  # stay on one IK branch
            if t == 1 and err > MAX_REACH_ERR:
                q = self._ik(p)  # the end pose: the usual solve from several seeds (raises if unreachable)
            elif err > 0.02:
                raise ToolError("unreachable", f"no IK on the straight path ({err * 1000:.0f} mm off)",
                                residual_m=round(err, 4))
            q = self._check_limits(q)
            self._check_path_collisions(q, start_deg=prev)
            qs.append(q)
            prev = q
        steps = 0
        for i, q in enumerate(qs):
            last = i == len(qs) - 1
            steps += self._drive(np.r_[q, self.cmd[5]], rate=CONTACT_DEG_PER_STEP, settle=3 if last else 1,
                                 measured=last)
        return steps

    def _resolve_object(self, object_id: str | None) -> dict[str, Any]:
        assert self.obj is not None
        if object_id is None or _norm_name(object_id) == self.obj.name:
            return {}
        if _norm_name(object_id) == "red_cube":  # the legacy default some clients still send; said so, not hidden
            return {"note": f"object_id 'red_cube' is the legacy default; the task object is '{self.obj.name}'"}
        raise ToolError("unknown_object", f"no graspable object '{object_id}'; the task object is '{self.obj.name}'",
                        objects=[self.obj.name])

    # ================================================================= TOOLS
    def observe_scene(self, a: ObserveArgs) -> tuple[dict[str, Any], np.ndarray]:
        if a.camera == "overhead" and np.abs(self.cmd[:ARM] - OBSERVE_POSE[:ARM]).max() > 5 and not self.holding:
            # Clear the overhead view first: a real, recorded motion to the observe pose.
            self._drive(np.r_[OBSERVE_POSE[:ARM], self.cmd[5]])
        img = self._render(a.camera)
        if a.camera == "overhead":
            self.last_overhead = img
        out: dict[str, Any] = {"camera": a.camera, "image_px": [PLAN_PX, PLAN_PX],
                               "note": "overhead image: +x (forward) points up the image, +y (robot left) points left"}
        if self.mode == "privileged":
            c, b = self.env.cube_pos(), self.env.box_pos()
            if getattr(self.env, "info", None) is None:
                out["privileged_ground_truth"] = {"red_cube_xyz": c.round(4).tolist(), "box_xyz": b.round(4).tolist(),
                                                  "cube_yaw_deg": round(float(np.rad2deg(self.env.cube_yaw())), 1)}
            else:
                assert self.obj is not None
                zones = {n: [round(float(v), 4) for v in (self._zone_xy(n, privileged=True))] for n in self.zones()}
                out["privileged_ground_truth"] = {f"{self.obj.name}_xyz": c.round(4).tolist(),
                                                  "zones_xy": zones, "target_zone": self.target_zone()}
        return out, img

    def get_robot_state(self, a: Empty) -> dict[str, Any]:
        ee = self.ee_pos()
        return {"joints_deg": np.round(self.obs["agent_pos"], 1).tolist(),  # type: ignore[index]
                "gripper_opening_0_100": round(self.gripper_opening(), 1),
                "end_effector_xyz": ee.round(4).tolist(), "holding_object": self.holding,
                "steps_used": len(self.rec), "steps_limit": self.max_steps}

    def propose_grasp(self, a: ProposeGraspArgs) -> dict[str, Any]:
        assert self.obj is not None
        note = self._resolve_object(a.object_id)
        if self.mode == "privileged" and a.image_point is None:
            c = self.env.cube_pos()
            yaw = float(np.rad2deg(self.env.cube_yaw()))
            source = "privileged_ground_truth"
            evidence: dict[str, Any] = {}
        else:
            if a.image_point is None:
                raise ToolError("image_point_required", "camera mode: point at the object in the overhead image")
            if self.last_overhead is None:
                raise ToolError("no_observation", "call observe_scene(camera='overhead') first")
            blob = self.refine_blob(self.last_overhead, a.image_point.u, a.image_point.v, rgba=self.obj.rgba)
            c = self.pixel_to_table(blob["u"], blob["v"], self.obj.grasp_z)
            # minAreaRect angle is in image axes; image right = -y, image up = +x in the robot frame.
            yaw = -float(blob.get("rect_deg", 0.0)) % 90.0
            source = "camera_backprojection"
            evidence = {"model_point": [a.image_point.u, a.image_point.v], "refined_point": [blob["u"], blob["v"]],
                        "refined": blob["refined"], "blob_pixels": blob["pixels"],
                        "segmentation": "object colour (hue)" if self.obj.rgba is not None else "saturation"}
        head = {"object_id": self.obj.name, "source": source, "estimated_xyz": c.round(4).tolist(),
                "grasp_mode": self.obj.mode, **note}
        tail = {"evidence": evidence} if evidence else {}
        if self.obj.mode == "side":
            return {**head, **self._side_grasp(np.r_[c[:2], self.obj.grasp_z]), **tail}
        gz = self.obj.grasp_z - 0.003  # 0.012 for the cube
        grasp = Pose(x=float(c[0]), y=float(c[1]), z=gz, yaw_deg=yaw)
        pre = Pose(x=float(c[0]), y=float(c[1]), z=gz + 0.043, yaw_deg=yaw)
        reachable = True
        detail = None
        try:
            self._validate_pose(grasp)
            self._ik(grasp)
            self._ik(pre)
        except ToolError as e:
            reachable, detail = False, e.message
        return {**head, "pregrasp": pre.model_dump(exclude_none=True), "grasp": grasp.model_dump(exclude_none=True),
                "reachable": reachable, "unreachable_reason": detail, **tail}

    def _side_grasp(self, g: np.ndarray) -> dict[str, Any]:
        """The skill's verified side grasp: transit above and behind, pre-grasp 6 cm back along the approach,
        the object centred in the open jaws, then a slide toward the fixed pad. Each pose is solved with the
        approach axis and the deterministic jaw roll (physical site position); the solutions seed the moves."""
        assert self.obj is not None
        try:
            self._validate_pose(Pose(x=float(g[0]), y=float(g[1]), z=float(g[2])))
        except ToolError as e:
            return {"reachable": False, "unreachable_reason": e.message}
        for axis in approaches(g[:2], "side"):
            try:
                open_gap = g + axis * SIDE_TIP_PAST_CENTRE
                q2 = self._ik_axis(open_gap, axis, use_hint=False)
                lateral = self._fk(q2)[1][:, 2]  # site z: the jaws' closing direction with the fixed roll
                slide = open_gap + lateral * SIDE_SLIDE_LATERAL
                q3 = self._ik_axis(slide, axis, seed=q2, use_hint=False)
                pre = open_gap - axis * 0.06
                q1 = self._ik_axis(pre, axis, seed=q2, use_hint=False)
                transit = pre + np.array([0, 0, 0.10]) - axis * 0.04
                q4 = self._ik_axis(transit, axis, seed=q2, tol=0.01, use_hint=False)
                lift = slide + np.array([0, 0, LIFT])
                q5 = self._ik_axis(lift, axis, seed=q3, tol=0.01, use_hint=False)
            except ToolError:
                continue
            poses = {"transit": (transit, q4), "pregrasp": (pre, q1), "grasp": (open_gap, q2), "slide": (slide, q3),
                     "lift": (lift, q5)}
            try:
                for xyz, q in poses.values():
                    self._validate_pose(Pose(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2])))
                    self._check_limits(q)
            except ToolError as e:
                return {"reachable": False, "unreachable_reason": e.message}
            ax = [round(float(v), 4) for v in axis]
            out: dict[str, Any] = {}
            for name, (xyz, q) in poses.items():
                self._q_hint[_key(xyz)] = q  # the agent sends the rounded pose back; same key
                out[name] = {"x": round(float(xyz[0]), 4), "y": round(float(xyz[1]), 4), "z": round(float(xyz[2]), 4),
                             "approach": ax}
            return {"approach": ax, "wrist_roll_deg": round(float(q2[4]), 1),
                    "approach_pitch_deg": round(float(np.rad2deg(np.arcsin(-axis[2]))), 1), **out,
                    "sequence": ["set_gripper(open)", "move_end_effector(transit)", "move_end_effector(pregrasp)",
                                 "move_end_effector(grasp)", "move_end_effector(slide)", "set_gripper(closed)",
                                 "verify_grasp", "move_end_effector(lift)"],
                    "reachable": True, "unreachable_reason": None}
        return {"reachable": False, "unreachable_reason": "no reachable side grasp for this object position "
                "(tried approach tilts 0, 20 and 40 degrees)"}

    def move_end_effector(self, a: MoveArgs) -> dict[str, Any]:
        self._validate_pose(a.target)
        goal = np.array([a.target.x, a.target.y, a.target.z])
        if a.target.approach is not None:
            steps = self._move_axis(goal, _unit(a.target.approach))
        elif np.linalg.norm(goal[:2] - self.ee_pos()[:2]) < VERTICAL_XY_M:
            steps = self._move_vertical(a.target)
        else:
            q = self._ik(a.target)
            target = np.r_[q, self.cmd[5]]
            self._check_path_collisions(target)
            steps = self._drive(target)
        ee = self.ee_pos()
        return {"reached_xyz": ee.round(4).tolist(),
                "error_m": round(float(np.linalg.norm(ee - [a.target.x, a.target.y, a.target.z])), 4),
                "steps": steps, "gripper_opening_0_100": round(self.gripper_opening(), 1)}

    def set_gripper(self, a: GripperArgs) -> dict[str, Any]:
        goal = GRIP_OPEN if a.state == "open" else GRIP_CLOSED
        target = self.cmd.copy()
        target[5] = goal
        steps = self._drive(target, timeout=40, settle=8)
        opening = self.gripper_opening()
        self.holding = a.state == "closed" and opening > HOLDING_MIN
        return {"state": a.state, "measured_opening_0_100": round(opening, 1), "steps": steps,
                "stalled_on_object": self.holding}

    def verify_grasp(self, a: Empty) -> tuple[dict[str, Any], np.ndarray]:
        opening = self.gripper_opening()
        closed_cmd = self.cmd[5] < 1.0
        self.holding = closed_cmd and opening > HOLDING_MIN
        if closed_cmd and not self.holding:
            self.counters["grasp_failures"] += 1
        out = {"gripper_commanded": "closed" if closed_cmd else "open", "measured_opening_0_100": round(opening, 1),
               "holding_object": self.holding,
               "basis": "proprioception: a closed gripper that stalls open is holding something"}
        if self.mode == "privileged":
            key = "privileged_cube_height_m" if getattr(self.env, "info", None) is None else "privileged_object_z_m"
            out[key] = round(float(self.env.cube_pos()[2]), 4)
        return out, self._render("wrist")

    def _zone_xy(self, zone: str, privileged: bool) -> tuple[float, float]:
        zones = self.zones()
        name = next((z for z in zones if _norm_name(z) == _norm_name(zone)), None)
        if name is None:
            raise ToolError("unknown_zone", f"no place zone '{zone}'", zones=list(zones))
        if zones[name] is None:  # the upstream box: no task-spec geometry, its pose is world state
            if not privileged:
                raise ToolError("image_point_required", "camera mode: point at the box in the overhead image")
            return float(self.env.box_pos()[0]), float(self.env.box_pos()[1])
        if privileged and name == self.target_zone():
            return float(self.env.box_pos()[0]), float(self.env.box_pos()[1])  # includes any start-state offset
        (cx, cy), _ = zones[name]  # type: ignore[misc]
        return float(cx), float(cy)

    def place_object(self, a: PlaceArgs) -> dict[str, Any]:
        assert self.obj is not None
        if not self.holding:
            raise ToolError("not_holding", "verify_grasp reports no object in the gripper")
        if a.zone is not None:
            x, y = self._zone_xy(a.zone, privileged=self.mode == "privileged")
            source = ("privileged_ground_truth" if self.mode == "privileged" else
                      "task_spec_zone (nominal centre from the intake, not observed)")
        elif a.target is not None:
            x, y = a.target.x, a.target.y
            source = "explicit_pose"
        elif a.image_point is not None:
            if self.last_overhead is None:
                raise ToolError("no_observation", "call observe_scene(camera='overhead') first")
            blob = self.refine_blob(self.last_overhead, a.image_point.u, a.image_point.v, radius=70)
            # The box is mostly its green floor; use the model's point unless the blob is clearly the box.
            u, v = (blob["u"], blob["v"]) if blob["refined"] and blob["pixels"] > 300 else (a.image_point.u,
                                                                                            a.image_point.v)
            x, y, _ = self.pixel_to_table(u, v, 0.0)
            source = "camera_backprojection"
        elif self.mode == "privileged":
            x, y = (float(v) for v in self.env.box_pos()[:2])
            source = "privileged_ground_truth"
        else:
            raise ToolError("target_required", "give zone, image_point (camera mode) or target")
        if self.obj.mode == "side":
            return {**self._place_side(np.array([x, y])), "source": source, "grasp_mode": "side"}
        h = self.obj.height
        carry = max(self.obj.grasp_z - 0.003 + LIFT, CARRY_Z)  # 0.075 for the cube
        release = h + 0.025  # 0.055 for the cube
        steps = 0
        for z in (carry, release):
            steps += self.move_end_effector(MoveArgs(target=Pose(x=x, y=y, z=z)))["steps"]
        steps += self.set_gripper(GripperArgs(state="open"))["steps"]
        self.holding = False
        out: dict[str, Any] = {"placed_at_xy": [round(float(x), 4), round(float(y), 4)], "source": source}
        try:  # the object is already released; a failed retreat is a warning, not a failed placement
            steps += self.move_end_effector(MoveArgs(target=Pose(x=x, y=y, z=max(RETREAT_Z, release + 0.017))))[
                "steps"]
        except ToolError as e:
            out["retreat_warning"] = e.message
        return {**out, "steps": steps}

    def _place_side(self, r: np.ndarray) -> dict[str, Any]:
        """The skill's side placement: carry at grasp height + LIFT, lower the gripper site to 6 mm above the
        grasp height (the object then stands on the table), open, back off along the approach."""
        assert self.obj is not None
        gz = self.obj.grasp_z
        rz = gz + 0.006
        carry, low = np.r_[r, gz + LIFT], np.r_[r, rz]
        for p in (carry, low):
            self._validate_pose(Pose(x=float(p[0]), y=float(p[1]), z=float(p[2])))
        axis = None
        for ax in approaches(r, "side"):
            try:
                self._ik_axis(carry, ax, tol=0.01, use_hint=False)
                self._ik_axis(low, ax, tol=0.006, use_hint=False)
            except ToolError:
                continue
            axis = ax
            break
        if axis is None:
            raise ToolError("unreachable", "no reachable side placement at the target", target_xy=r.round(4).tolist())
        steps = self._move_axis(carry, axis)
        steps += self._move_axis(low, axis)
        steps += self.set_gripper(GripperArgs(state="open"))["steps"]
        self.holding = False
        out: dict[str, Any] = {"placed_at_xy": [round(float(r[0]), 4), round(float(r[1]), 4)],
                               "approach": [round(float(v), 4) for v in axis]}
        try:  # released; a failed retreat is a warning, not a failed placement
            steps += self._move_axis(low - axis * 0.06, axis)  # straight back out of the jaws, then up
            steps += self._move_axis(low - axis * 0.06 + np.array([0, 0, 0.03]), axis)
        except ToolError as e:
            out["retreat_warning"] = e.message
        return {**out, "steps": steps}

    def go_home(self, a: Empty) -> dict[str, Any]:
        """A validated, recorded joint move to the arm's home pose (collision-swept like every motion)."""
        home = np.asarray(self.env.module.HOME, dtype=np.float64)
        target = np.r_[self._check_limits(home[:ARM].copy()), self.cmd[5] if self.holding else home[5]]
        self._check_path_collisions(target)
        steps = self._drive(target, timeout=240)
        err = float(np.abs(np.asarray(self.env.world.joint_state())[:ARM] - home[:ARM]).max())
        return {"reached_home": err < HOME_TOL_DEG, "max_joint_error_deg": round(err, 2), "steps": steps,
                "joints_deg": np.round(self.obs["agent_pos"], 1).tolist()}  # type: ignore[index]

    def check_task_success(self, a: Empty) -> dict[str, Any]:
        if getattr(self.env, "info", None) is None:
            return {"success": self.env.in_box(), "basis": "simulator ground truth (cube resting inside the box)"}
        assert self.obj is not None
        return {"success": self.env.in_box(),
                "basis": f"simulator ground truth ({self.obj.label} resting upright inside zone "
                         f"'{self.target_zone()}', released)"}

    # ------------------------------------------------------------- dispatch
    TOOLS: dict[str, tuple[type[BaseModel], str]] = {
        "observe_scene": (ObserveArgs, "Render a camera image (overhead by default; moves the arm clear first)."),
        "get_robot_state": (Empty, "Joint angles, gripper opening, end-effector position, step budget."),
        "propose_grasp": (ProposeGraspArgs, "Turn a point on the {object} in the overhead image into validated "
                                            "grasp poses: pregrasp/grasp from above for short objects; for tall "
                                            "ones a side grasp (approach axis, transit, pregrasp, grasp, slide, lift)."),
        "move_end_effector": (MoveArgs, "Move the gripper tip to a validated pose (workspace, reachability, "
                                        "joint limits and collisions are checked). Pass the pose's approach axis "
                                        "for side-grasp poses."),
        "set_gripper": (GripperArgs, "Open or close the gripper; reports whether it stalled on an object."),
        "verify_grasp": (Empty, "Check from proprioception (and a wrist image) whether an object is held."),
        "place_object": (PlaceArgs, "Carry the held {object} to {zones} (or an image point / explicit target) and "
                                    "release it (carry, lower, open, retreat; side placement for tall objects)."),
        "go_home": (Empty, "Return the arm to its home pose (a validated, recorded motion). Do this after placing."),
        "check_task_success": (Empty, "Simulator ground truth: is the {object} resting in {target}?"),
        "finish_episode": (FinishArgs, "End the episode when done or when no further progress is possible."),
    }

    def call(self, name: str, raw: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray | None]:
        """Validate and execute one tool call. Errors come back as structured results."""
        self.counters["tool_calls"] += 1
        t0 = time.time()
        steps0 = len(self.rec)
        if name not in self.TOOLS or name == "finish_episode":
            self.counters["tool_errors"] += 1
            return {"ok": False, "error": {"code": "unknown_tool", "message": f"no tool {name}"}, "sim_steps": 0}, None
        try:
            args = self.TOOLS[name][0].model_validate(raw or {})
        except ValidationError as e:
            self.counters["tool_errors"] += 1
            errs = e.errors(include_url=False, include_context=False)
            msg = "; ".join(f"{'.'.join(str(x) for x in er['loc']) or 'args'}: {er['msg']}" for er in errs)
            return {"ok": False, "error": {"code": "invalid_arguments", "message": msg,
                                           "details": [{"loc": list(er["loc"]), "msg": er["msg"]} for er in errs]},
                    "sim_steps": 0}, None
        img = None
        try:
            res = getattr(self, name)(args)
            if isinstance(res, tuple):
                res, img = res
            out = {"ok": True, **res}
        except ToolError as e:
            self.counters["tool_errors"] += 1
            out = {"ok": False, "error": {"code": e.code, "message": e.message, **e.details}}
        out["sim_steps"] = len(self.rec) - steps0
        out["wall_ms"] = int((time.time() - t0) * 1000)
        return out, img


def annotate_overhead(img: np.ndarray, point_format: str = "pixels") -> np.ndarray:
    """Light grid labelled in the units the planner points in: pixels every 40 px, or
    relative 0-1000 coordinates every 100 (Qwen3-VL's native grounding convention)."""
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    marks = ([(p, str(p)) for p in range(0, PLAN_PX, 40)] if point_format == "pixels" else
             [(round(k * PLAN_PX / 1000), str(k)) for k in range(0, 1000, 100)])
    for p, label in marks:
        dr.line([(p, 0), (p, PLAN_PX)], fill=(255, 255, 255), width=1)
        dr.line([(0, p), (PLAN_PX, p)], fill=(255, 255, 255), width=1)
        dr.text((p + 2, 2), label, fill=(30, 30, 30))
        dr.text((2, p + 2), label, fill=(30, 30, 30))
    return np.asarray(im)


def from_model_point(point: dict[str, Any], point_format: str) -> dict[str, Any]:
    """Convert a model-reported image point to overhead-image pixels."""
    if point_format == "norm1000":
        return {"u": float(point.get("u", 0)) * PLAN_PX / 1000, "v": float(point.get("v", 0)) * PLAN_PX / 1000}
    return point


def to_data_url(img: np.ndarray, quality: int = 85) -> str:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def describe_terms(ctx: dict[str, Any] | None) -> dict[str, str]:
    """How tool descriptions and prompts name the task object and zones (ctx: ManipulationSession.task_context)."""
    if not ctx or ctx.get("scene") != "intake":
        return {"object": "red cube", "zones": "the box", "target": "the box"}
    zones = ctx.get("zones") or []
    return {"object": str(ctx["object"]).replace("_", " "),
            "zones": ("a zone (" + ", ".join(f"'{z}'" for z in zones) + ")") if zones else "the target",
            "target": f"zone '{ctx.get('target_zone')}'"}


def tool_schemas(point_format: str = "pixels", ctx: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """OpenAI-style function tool definitions generated from the Pydantic models. For
    norm1000 planners, image points are described (and bounded) in relative 0-1000 units;
    the planner loop converts them to pixels before validation. ``ctx`` (a session's task_context)
    names the real object and zones in the descriptions and constrains object_id / zone to them."""
    terms = describe_terms(ctx)
    out = []
    for name, (model, desc) in ManipulationSession.TOOLS.items():
        desc = desc.format(**terms)
        schema = model.model_json_schema()
        if ctx and name == "propose_grasp":
            schema["properties"]["object_id"] = {"type": "string", "enum": [ctx["object"]],
                                                 "description": "the task object"}
        if ctx and name == "place_object" and ctx.get("zones"):
            schema["properties"]["zone"] = {"type": "string", "enum": list(ctx["zones"]),
                                            "description": f"a named place zone; the task's target is "
                                                           f"'{ctx.get('target_zone')}'"}
        if point_format == "norm1000" and "ImagePoint" in schema.get("$defs", {}):
            props = schema["$defs"]["ImagePoint"]["properties"]
            props["u"].update(maximum=1000, description="relative x in the overhead image, 0 = left edge, 1000 = right")
            props["v"].update(maximum=1000, description="relative y in the overhead image, 0 = top edge, 1000 = bottom")
        out.append({"type": "function", "function": {"name": name, "description": desc, "parameters": schema}})
    return out
