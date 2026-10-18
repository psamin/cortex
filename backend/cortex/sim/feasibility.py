"""Can this arm do this task? Measured from the model, before any compute is spent.

Checks, per object:
  * grasp: grasp width + margin must fit the fingertip gap at the tools' "open" command,
    measured by MuJoCo geom distance between the jaws' tip geoms;
  * payload: at each carry pose (above every start and every zone), static joint torque
    (gravity on the arm from ``mj_inverse``, plus the object's weight through the site
    Jacobian) must stay under PAYLOAD_FRACTION of each actuator's force limit;
  * stability: tall objects (height > 2.5x their base) need an upright check and extra
    zone clearance.
Per start and zone: IK reach at grasp and carry height (12 mm tolerance, like the tools).
Per conveyor that carries an object (issue #255): the reachable intercept window along the belt,
the stretch where the arm reaches the object at grasp and hover height, and how long the object
spends in it at the belt speed. An object on a conveyor gets this instead of a start check.

Infeasible objects get a suggestion: the largest catalogue object of the same shape
family that passes.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
from pydantic import BaseModel

from ..intake import catalogue
from ..intake.spec import IntakeSpec, ObjectSpec
from .expert import approach_error_deg, solve_ik
from .grasp import approaches, grasp_height, grasp_mode, ik_seeds
from .scene import SceneInfo
from .scene_env import SceneEnv

GRIP_OPEN_CMD = 60.0  # ManipulationSession's open command (0-100)
GRASP_MARGIN = 0.008  # m of clearance the jaws need around the object
PAYLOAD_FRACTION = 0.7
REACH_TOL = 0.012  # carry and release poses
GRASP_TOL = 0.004  # grasp poses: under half the jaw clearance per side
APPROACH_TOL_DEG = 10.0
TALL_RATIO = 2.5
LIFT = 0.06  # carry height above the grasp height
HOVER_DZ = 0.043  # the intercept's hover above the grasp height (pick_place_skill.HOVER_DZ)
WINDOW_STEP = 0.02  # m between sampled intercept points
WINDOW_END_MARGIN = 0.05  # an intercept this close to the belt's end is too late
MIN_WINDOW_S = 0.5  # the object must spend at least this long in the window




class Check(BaseModel):
    ok: bool
    detail: str
    value: float | None = None
    limit: float | None = None


class ObjectReport(BaseModel):
    name: str
    ok: bool
    grasp: Check
    payload: Check
    tall: bool
    grasp_mode: str = "top"
    suggestion: str | None = None


class PlaceReport(BaseModel):
    name: str
    kind: str  # "start" | "zone"
    ok: bool
    grasp_err_mm: float
    carry_err_mm: float
    suggestion: tuple[float, float] | None = None  # nearest reachable spot on the same bearing


class ConveyorReport(BaseModel):
    """The reachable intercept window on a belt, for the object riding it. ``window_m``: (from, to) metres
    downstream of the belt's start; ``window_xy``: those two points on the object's line (table frame)."""

    name: str
    object: str
    ok: bool
    speed_mps: float
    window_m: tuple[float, float] | None = None
    window_xy: tuple[tuple[float, float], tuple[float, float]] | None = None
    time_in_window_s: float = 0.0
    detail: str = ""


class FeasibilityReport(BaseModel):
    ok: bool
    jaw_opening_m: float
    objects: list[ObjectReport]
    places: list[PlaceReport]
    summary: str
    conveyors: list[ConveyorReport] = []


def jaw_opening(env: SceneEnv, command: float = GRIP_OPEN_CMD) -> float:
    """Fingertip gap (m) at a gripper command, from the model's jaw tip geoms."""
    w = env.world
    m = w.model
    d = mujoco.MjData(m)
    d.qpos[:] = w.data.qpos
    lo, hi = w.lo[5], w.hi[5]
    d.qpos[w.qadr[5]] = lo + command / 100 * (hi - lo)
    mujoco.mj_forward(m, d)
    fixed = [i for i in range(m.ngeom) if m.geom(i).name.startswith("fixed_jaw")]
    tips = [i for i in range(m.ngeom) if m.geom(i).name.startswith("moving_jaw") and "tip" in m.geom(i).name]
    fr = np.zeros(6)
    # The smallest gap over the tips: an object sits deeper than the very tip, where a rotary jaw is narrower.
    return float(min(min(mujoco.mj_geomDistance(m, d, t, f, 0.3, fr) for f in fixed) for t in tips))


def static_torque(env: SceneEnv, q_arm_deg: np.ndarray, payload_kg: float) -> np.ndarray:
    """Joint torques (N·m, 5 arm joints) holding the arm still at q with payload at the gripper site."""
    w = env.world
    m = w.model
    d = mujoco.MjData(m)
    d.qpos[:] = w.data.qpos
    d.qpos[w.qadr[:5]] = np.deg2rad(q_arm_deg[:5])
    d.qvel[:] = 0
    d.qacc[:] = 0
    mujoco.mj_forward(m, d)  # qfrc_bias at zero velocity is the gravity torque (no contact forces)
    dofs = np.array([m.jnt_dofadr[m.joint(n).id] for n in
                     ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")])
    tau = d.qfrc_bias[dofs].copy()
    jp = np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, jp, None, env.site)
    tau += jp[:, dofs].T @ np.array([0.0, 0.0, payload_kg * 9.81])
    return tau


def _ik_err(env: SceneEnv, xyz: np.ndarray, mode: str = "top") -> tuple[np.ndarray, float]:
    """Best IK over seeds and approach tilts. Solutions whose approach axis is off by more than
    APPROACH_TOL_DEG are discarded: a tilted gripper can reach a point it cannot grasp at."""
    best_q, best_e = None, np.inf
    for axis in approaches(xyz[:2], mode):
        for seed in ik_seeds(xyz, np.asarray(env.module.HOME, dtype=np.float64)):
            q, e = solve_ik(env, xyz, seed, approach=axis, iters=150)
            if approach_error_deg(env, q, axis) > APPROACH_TOL_DEG:
                continue
            if e < best_e:
                best_q, best_e = q, e
            if e < 0.002:
                return best_q, best_e
    return best_q, best_e  # type: ignore[return-value]


def check(spec: IntakeSpec, info: SceneInfo, env: SceneEnv | None = None) -> FeasibilityReport:
    own = env is None
    env = env or SceneEnv(info)
    try:
        env.reset(0)
        opening = jaw_opening(env)
        limit = float(np.min(np.abs(env.world.model.actuator_forcerange[env.world.actuators[:5]][:, 1])))
        places: list[PlaceReport] = []
        carry_qs: dict[str, list[np.ndarray]] = {}
        pick = next((o for o in spec.objects if o.name == (spec.task.pick or "")), None)
        objs = spec.objects
        belts = info.belts()
        conveyors: list[ConveyorReport] = []
        for o in objs:
            gz, mode = grasp_height(o.height), grasp_mode(o.height)
            sx, sy = info.starts[o.name]
            if o.on and o.on in belts:  # it rides the belt: where it can be met matters, not where it starts
                rep, qc = _conveyor_window(env, belts[o.on], o, (sx, sy))
                conveyors.append(rep)
                if qc is not None:
                    carry_qs.setdefault(o.name, []).append(qc)
                continue
            qg, eg = _ik_err(env, np.array([sx, sy, gz]), mode)
            qc, ec = _ik_err(env, np.array([sx, sy, gz + LIFT]), mode)
            ok = eg < GRASP_TOL and ec < REACH_TOL
            places.append(PlaceReport(name=o.name, kind="start", ok=ok, grasp_err_mm=round(eg * 1000, 1),
                                      carry_err_mm=round(ec * 1000, 1),
                                      suggestion=None if ok else _nearest_reachable(env, sx, sy, gz, mode)))
            carry_qs.setdefault(o.name, []).append(qc)
        gz = grasp_height(pick.height) if pick else 0.015
        mode = grasp_mode(pick.height) if pick else "top"
        for zn, ((cx, cy), _) in info.zones.items():
            qr, er = _ik_err(env, np.array([cx, cy, gz + 0.01]), mode)
            qc, ec = _ik_err(env, np.array([cx, cy, gz + LIFT]), mode)
            ok = max(er, ec) < REACH_TOL
            places.append(PlaceReport(name=zn, kind="zone", ok=ok, grasp_err_mm=round(er * 1000, 1),
                                      carry_err_mm=round(ec * 1000, 1),
                                      suggestion=None if ok else _nearest_reachable(env, cx, cy, gz + 0.01, mode)))
            if pick:
                carry_qs.setdefault(pick.name, []).append(qc)
        reports = [_object_report(env, o, opening, limit, carry_qs.get(o.name, [])) for o in objs]
        ok = all(r.ok for r in reports) and all(p.ok for p in places) and all(c.ok for c in conveyors)
        bad = [r.name for r in reports if not r.ok] + [p.name for p in places if not p.ok]
        bad += [f"conveyor {c.name}" for c in conveyors if not c.ok]
        # The agent reads this. It must report a physical limit to its human, not edit the world to pass:
        # in a live agent test, it shrank a real mug so the simulated one would fit the jaws.
        summary = "feasible" if ok else (
            "not feasible: " + ", ".join(bad) + ". Tell the user what doesn't fit and why; don't change an object's "
            "measured size or mass to pass. Moving a zone or using a different object needs their OK.")
        return FeasibilityReport(ok=ok, jaw_opening_m=round(opening, 4), objects=reports, places=places,
                                 summary=summary, conveyors=conveyors)
    finally:
        if own:
            env.close()


def _conveyor_window(env: SceneEnv, belt, o: ObjectSpec, start: tuple[float, float]
                     ) -> tuple[ConveyorReport, np.ndarray | None]:
    """The longest stretch of the object's line along the belt where a top grasp and its hover are both
    reachable, between its start and WINDOW_END_MARGIN before the end. Also returns a carry pose in it."""
    s0, lat = belt.coords(np.asarray(start))
    gz = belt.height + grasp_height(o.height)
    if grasp_mode(o.height) == "side":
        return ConveyorReport(name=belt.name, object=o.name, ok=False, speed_mps=belt.speed,
                              detail=f"{o.name} is tall enough to need a side grasp, which has no intercept mode "
                                     "for a moving belt yet"), None
    runs: list[list[float]] = []
    carry: dict[float, np.ndarray] = {}
    for s in np.arange(max(0.0, s0), belt.length - WINDOW_END_MARGIN + 1e-9, WINDOW_STEP):
        p = belt.point(float(s), lat)
        _, eg = _ik_err(env, np.r_[p, gz])
        qc, ec = _ik_err(env, np.r_[p, gz + HOVER_DZ])
        if eg < GRASP_TOL and ec < REACH_TOL:
            if runs and abs(runs[-1][-1] - (s - WINDOW_STEP)) < 1e-6:
                runs[-1].append(float(s))
            else:
                runs.append([float(s)])
            carry[float(s)] = qc
    if not runs:
        return ConveyorReport(name=belt.name, object=o.name, ok=False, speed_mps=belt.speed,
                              detail=f"the arm can't reach {o.name} anywhere along conveyor {belt.name}; "
                                     "move the belt closer to the arm (about 0.15-0.30 m from the base)"), None
    best = max(runs, key=len)
    lo, hi = best[0], best[-1]
    t = (hi - lo) / belt.speed
    ok = t >= MIN_WINDOW_S
    a, b = belt.point(lo, lat), belt.point(hi, lat)
    detail = (f"reachable from {lo:.2f} to {hi:.2f} m along the belt; at {belt.speed:.3f} m/s {o.name} is in "
              f"reach for {t:.1f} s" + ("" if ok else f", under the {MIN_WINDOW_S} s the intercept needs"))
    return ConveyorReport(name=belt.name, object=o.name, ok=ok, speed_mps=belt.speed,
                          window_m=(round(lo, 3), round(hi, 3)),
                          window_xy=((round(float(a[0]), 3), round(float(a[1]), 3)),
                                     (round(float(b[0]), 3), round(float(b[1]), 3))),
                          time_in_window_s=round(t, 2), detail=detail), carry[best[len(best) // 2]]


def _nearest_reachable(env: SceneEnv, x: float, y: float, z: float, mode: str) -> tuple[float, float] | None:
    """Walk along the bearing from the arm base, nearest radius first, for a spot reachable at grasp and
    carry height."""
    r0, th = float(np.hypot(x, y)), float(np.arctan2(y, x))
    for dr in sorted(np.arange(-0.20, 0.21, 0.02), key=abs):
        r = r0 + dr
        if not 0.10 <= r <= 0.42:
            continue
        p = np.array([r * np.cos(th), r * np.sin(th)])
        if _ik_err(env, np.array([*p, z]), mode)[1] < GRASP_TOL and \
                _ik_err(env, np.array([*p, z + LIFT]), mode)[1] < REACH_TOL:
            return (round(float(p[0]), 3), round(float(p[1]), 3))
    return None


def _object_report(env: SceneEnv, o: ObjectSpec, opening: float, limit: float,
                   carry_qs: list[np.ndarray]) -> ObjectReport:
    need = o.grasp_width + GRASP_MARGIN
    grasp = Check(ok=need <= opening, value=round(need, 4), limit=round(opening, 4),
                  detail=f"needs {need * 1000:.0f} mm between the jaws, the gripper opens {opening * 1000:.0f} mm")
    worst = 0.0
    for q in carry_qs:
        worst = max(worst, float(np.max(np.abs(static_torque(env, q, o.mass_kg)))))
    cap = PAYLOAD_FRACTION * limit
    payload = Check(ok=worst <= cap, value=round(worst, 3), limit=round(cap, 3),
                    detail=f"worst holding torque {worst:.2f} N·m vs {cap:.2f} N·m allowed "
                           f"({PAYLOAD_FRACTION:.0%} of the {limit:.2f} N·m actuator limit)")
    ok = grasp.ok and payload.ok
    return ObjectReport(name=o.name, ok=ok, grasp=grasp, payload=payload,
                        tall=o.height > TALL_RATIO * o.grasp_width, grasp_mode=grasp_mode(o.height),
                        suggestion=None if ok else suggest(o, opening))


def suggest(o: ObjectSpec, opening: float) -> str | None:
    """The heaviest catalogue item of the same shape that fits the jaws, as a hint."""
    fits: list[tuple[float, str]] = []
    for key, (shape, size, mass, _) in catalogue.CATALOGUE.items():
        width = min(size[:2]) if shape == "box" else size[0]
        if shape == o.shape and width + GRASP_MARGIN <= opening and mass <= o.mass_kg:
            fits.append((mass, key))
    return f"try a {max(fits)[1]}" if fits else None


def report_dict(r: FeasibilityReport) -> dict[str, Any]:
    return r.model_dump(mode="json")
