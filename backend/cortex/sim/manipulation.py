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

from .expert import ARM, solve_ik
from .so101 import PickPlaceEnv

PLAN_CAM, PLAN_PX = "top", 480
OBSERVE_POSE = np.array([0.0, -100.0, 0.0, -90.0, 0.0, 30.0], np.float32)  # arm vertical: clear overhead view
GRIP_OPEN, GRIP_CLOSED = 60.0, 0.0
HOLDING_MIN = 5.0  # gripper stalls above this when closed on an object
WORKSPACE = {"x": (0.08, 0.36), "y": (-0.22, 0.24), "z": (0.006, 0.20)}
CUBE_Z, CARRY_Z, RELEASE_Z, RETREAT_Z = 0.015, 0.075, 0.055, 0.072
MAX_REACH_ERR = 0.012
MOTION_TIMEOUT_STEPS = 150
EPISODE_MAX_STEPS = 900  # 30 s of sim time
ARM_DEG_PER_STEP, GRIP_PER_STEP = 4.0, 8.0


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


class ObserveArgs(BaseModel):
    camera: Literal["overhead", "front", "wrist"] = "overhead"


class ProposeGraspArgs(BaseModel):
    object_id: Literal["red_cube"] = "red_cube"
    image_point: ImagePoint | None = Field(None, description="where the object is in the latest overhead image "
                                                             "(required in camera mode)")


class MoveArgs(BaseModel):
    target: Pose
    reason: str = Field("", max_length=200)


class GripperArgs(BaseModel):
    state: Literal["open", "closed"]


class PlaceArgs(BaseModel):
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

    def __len__(self) -> int:
        return len(self.action)


class ManipulationSession:
    def __init__(self, env: PickPlaceEnv, mode: Literal["camera", "privileged"] = "camera",
                 on_frame: Any = None, video_renderer: Any = None):
        self.env, self.mode = env, mode
        self.on_frame = on_frame  # called with each preview frame (live view)
        self.video_renderer = video_renderer
        self.rec = Recorder()
        self.obs: dict[str, Any] | None = None
        self.cmd = np.zeros(6)
        self._plan_renderer: mujoco.Renderer | None = None
        self.last_overhead: np.ndarray | None = None
        self.holding = False
        self.counters = {"tool_calls": 0, "tool_errors": 0, "grasp_failures": 0}

    # ---------------- episode boundary (owned by the collector, not the planner)
    def reset_environment(self, seed: int, variation) -> dict[str, Any]:
        self.obs = self.env.reset(seed, variation)
        self.cmd = self.obs["agent_pos"].astype(np.float64).copy()
        self.rec = Recorder()
        self.holding = False
        self.last_overhead = None
        return {"seed": seed}

    # ---------------- low level: drive joint targets, record every step
    def _step(self, target: np.ndarray) -> None:
        assert self.obs is not None
        if len(self.rec) >= EPISODE_MAX_STEPS:
            raise ToolError("episode_time_limit", f"episode exceeded {EPISODE_MAX_STEPS} steps (30 s)")
        action = target.astype(np.float32)
        self.rec.front.append(self.obs["pixels"]["front"])
        self.rec.wrist.append(self.obs["pixels"]["wrist"])
        self.rec.state.append(self.obs["agent_pos"])
        self.rec.action.append(action)
        self.obs, reward, _, _, _ = self.env.step(action)
        self.rec.reward.append(float(reward))
        if self.video_renderer is not None:
            self.video_renderer.update_scene(self.env.world.data, camera="front")
            frame = self.video_renderer.render().copy()
            self.rec.video.append(frame)
            if self.on_frame:
                self.on_frame(frame)

    def _drive(self, q_target: np.ndarray, timeout: int = MOTION_TIMEOUT_STEPS, settle: int = 3) -> int:
        steps, held = 0, 0
        while steps < timeout:
            delta = q_target - self.cmd
            step = np.clip(delta, -ARM_DEG_PER_STEP, ARM_DEG_PER_STEP)
            step[5] = np.clip(delta[5], -GRIP_PER_STEP, GRIP_PER_STEP)
            self.cmd = self.cmd + step
            self._step(self.cmd)
            steps += 1
            if np.abs(q_target - self.cmd).max() < 0.5:
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

    def pixel_to_table(self, u: float, v: float, z: float = CUBE_Z) -> np.ndarray:
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

    def refine_blob(self, img: np.ndarray, u: float, v: float, radius: int = 45) -> dict[str, Any]:
        """Camera-only refinement: segment the color region under the point and
        return its centroid and in-plane orientation (image evidence only)."""
        import cv2

        h, w = img.shape[:2]
        u0, v0 = int(round(u)), int(round(v))
        x0, x1, y0, y1 = max(0, u0 - radius), min(w, u0 + radius), max(0, v0 - radius), min(h, v0 + radius)
        hsv = cv2.cvtColor(np.ascontiguousarray(img[y0:y1, x0:x1]), cv2.COLOR_RGB2HSV)
        # Saturated (colored) regions against the low-saturation tabletop; saturation is
        # invariant to the lighting scale, so this works in dim and bright scenes alike.
        mask = ((hsv[..., 1] > 115) & (hsv[..., 2] > 20)).astype(np.uint8)
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
        for axis in ("x", "y", "z"):
            lo, hi = WORKSPACE[axis]
            val = getattr(p, axis)
            if not lo <= val <= hi:
                raise ToolError("outside_workspace", f"{axis}={val:.3f} m outside [{lo}, {hi}]",
                                axis=axis, value=val, limits=[lo, hi])
        if p.z < 0.008 and not allow_low:
            raise ToolError("table_collision", "target is below the 8 mm table clearance", z=p.z)

    def _ik(self, p: Pose) -> np.ndarray:
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
        w = self.env.world
        lo, hi = np.rad2deg(w.lo[:ARM]), np.rad2deg(w.hi[:ARM])
        if np.any(q < lo - 1e-3) or np.any(q > hi + 1e-3):
            raise ToolError("joint_limit", "solution violates joint limits", q=q.round(1).tolist())
        return q

    def _check_path_collisions(self, q_target_deg: np.ndarray) -> None:
        """Kinematic sweep: arm geoms must not penetrate the table or box walls."""
        w = self.env.world
        m = w.model
        d = mujoco.MjData(m)
        d.qpos[:] = w.data.qpos
        arm_bodies = set(range(m.nbody)) - {0, w.box_body, *w.cube_bodies}
        statics = {g for g in range(m.ngeom) if m.geom_bodyid[g] in (0, w.box_body)}
        start = self.cmd[:ARM].copy()
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
            out["privileged_ground_truth"] = {"red_cube_xyz": c.round(4).tolist(), "box_xyz": b.round(4).tolist(),
                                              "cube_yaw_deg": round(float(np.rad2deg(self.env.cube_yaw())), 1)}
        return out, img

    def get_robot_state(self, a: Empty) -> dict[str, Any]:
        ee = self.ee_pos()
        return {"joints_deg": np.round(self.obs["agent_pos"], 1).tolist(),  # type: ignore[index]
                "gripper_opening_0_100": round(self.gripper_opening(), 1),
                "end_effector_xyz": ee.round(4).tolist(), "holding_object": self.holding,
                "steps_used": len(self.rec), "steps_limit": EPISODE_MAX_STEPS}

    def propose_grasp(self, a: ProposeGraspArgs) -> dict[str, Any]:
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
            blob = self.refine_blob(self.last_overhead, a.image_point.u, a.image_point.v)
            c = self.pixel_to_table(blob["u"], blob["v"], CUBE_Z)
            # minAreaRect angle is in image axes; image right = -y, image up = +x in the robot frame.
            yaw = -float(blob.get("rect_deg", 0.0)) % 90.0
            source = "camera_backprojection"
            evidence = {"model_point": [a.image_point.u, a.image_point.v], "refined_point": [blob["u"], blob["v"]],
                        "refined": blob["refined"], "blob_pixels": blob["pixels"]}
        grasp = Pose(x=float(c[0]), y=float(c[1]), z=0.012, yaw_deg=yaw)
        pre = Pose(x=float(c[0]), y=float(c[1]), z=0.055, yaw_deg=yaw)
        reachable = True
        detail = None
        try:
            self._validate_pose(grasp)
            self._ik(grasp)
            self._ik(pre)
        except ToolError as e:
            reachable, detail = False, e.message
        return {"object_id": a.object_id, "source": source, "estimated_xyz": c.round(4).tolist(),
                "pregrasp": pre.model_dump(), "grasp": grasp.model_dump(), "reachable": reachable,
                "unreachable_reason": detail, **({"evidence": evidence} if evidence else {})}

    def move_end_effector(self, a: MoveArgs) -> dict[str, Any]:
        self._validate_pose(a.target)
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
            out["privileged_cube_height_m"] = round(float(self.env.cube_pos()[2]), 4)
        return out, self._render("wrist")

    def place_object(self, a: PlaceArgs) -> dict[str, Any]:
        if not self.holding:
            raise ToolError("not_holding", "verify_grasp reports no object in the gripper")
        if a.target is not None:
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
            raise ToolError("target_required", "give image_point (camera mode) or target")
        steps = 0
        for z in (CARRY_Z, RELEASE_Z):
            steps += self.move_end_effector(MoveArgs(target=Pose(x=x, y=y, z=z)))["steps"]
        steps += self.set_gripper(GripperArgs(state="open"))["steps"]
        self.holding = False
        out: dict[str, Any] = {"placed_at_xy": [round(float(x), 4), round(float(y), 4)], "source": source}
        try:  # the object is already released; a failed retreat is a warning, not a failed placement
            steps += self.move_end_effector(MoveArgs(target=Pose(x=x, y=y, z=RETREAT_Z)))["steps"]
        except ToolError as e:
            out["retreat_warning"] = e.message
        return {**out, "steps": steps}

    def check_task_success(self, a: Empty) -> dict[str, Any]:
        return {"success": self.env.in_box(), "basis": "simulator ground truth (cube resting inside the box)"}

    # ------------------------------------------------------------- dispatch
    TOOLS: dict[str, tuple[type[BaseModel], str]] = {
        "observe_scene": (ObserveArgs, "Render a camera image (overhead by default; moves the arm clear first)."),
        "get_robot_state": (Empty, "Joint angles, gripper opening, end-effector position, step budget."),
        "propose_grasp": (ProposeGraspArgs, "Turn a point on the object in the overhead image into validated "
                                            "pregrasp/grasp poses (yaw from the image)."),
        "move_end_effector": (MoveArgs, "Move the gripper tip to a validated pose (workspace, reachability, "
                                        "joint limits and collisions are checked)."),
        "set_gripper": (GripperArgs, "Open or close the gripper; reports whether it stalled on an object."),
        "verify_grasp": (Empty, "Check from proprioception (and a wrist image) whether an object is held."),
        "place_object": (PlaceArgs, "Carry the held object to a point and release it (carry, lower, open, retreat)."),
        "check_task_success": (Empty, "Simulator ground truth: is the cube resting in the box?"),
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


def annotate_overhead(img: np.ndarray, every: int = 40) -> np.ndarray:
    """Light pixel grid with labels so the planner can point precisely."""
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    for p in range(0, PLAN_PX, every):
        dr.line([(p, 0), (p, PLAN_PX)], fill=(255, 255, 255), width=1)
        dr.line([(0, p), (PLAN_PX, p)], fill=(255, 255, 255), width=1)
        dr.text((p + 2, 2), str(p), fill=(30, 30, 30))
        dr.text((2, p + 2), str(p), fill=(30, 30, 30))
    return np.asarray(im)


def to_data_url(img: np.ndarray, quality: int = 85) -> str:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def tool_schemas() -> list[dict[str, Any]]:
    """OpenAI-style function tool definitions generated from the Pydantic models."""
    out = []
    for name, (model, desc) in ManipulationSession.TOOLS.items():
        schema = model.model_json_schema()
        out.append({"type": "function", "function": {"name": name, "description": desc, "parameters": schema}})
    return out
