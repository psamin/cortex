"""Render-free scene description and body poses for the studio's stylized 3D viewport.

MuJoCo stays the source of truth: the browser loads the arm's Menagerie STL files once, then
places every body at the world pose MuJoCo computed. Three things are served:

* ``describe_model``: for each visual mesh geom its body, mesh file, the transform that places
  the *raw* STL in the body frame, and a colour role (``arm`` or ``motor``); for primitive geoms
  (table, objects, zones, obstacles) their type, MuJoCo size, local pose and rgba.
* ``mesh_path``: the STL file in the pinned EnvHub snapshot, looked up by name in the asset list.
* ``scene_state``: body world poses (xpos, xquat as w x y z), arm joint angles in degrees and the
  TCP position, for a project's built scene at the same reset seed ``capture_frame`` renders.

MuJoCo compiles each mesh into its own inertial frame (``mesh_pos`` / ``mesh_quat``) and folds
that frame into the geom pose. The browser loads the raw STL, so the transform served here is
``geom * inverse(mesh)``.

Live recordings and agent collections run in separate worker processes that publish only JPEG
frames, so ``scene_state`` is the scene's reset state; the studio polls it slowly and keeps the
MJPEG camera view for live motion. Before a project's scene is built, ``scene_state`` describes the
upstream tabletop under a plan condition, matching the legacy ``/sim/frame`` preview.
"""

from __future__ import annotations

import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from ..context import Ctx
from ..models import Project
from ..sim.scene import SceneInfo
from . import intake_service

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
TCP_SITE = "gripperframe"
MOTOR_MATERIAL = "sts3215"  # Feetech servo meshes and materials share this prefix
_lock = threading.Lock()
_models: dict[str, dict[str, Any]] = {}  # scene path -> description
_states: dict[tuple, dict[str, Any]] = {}  # (scene, [condition], seed) -> state
MAX_CACHED = 16
UPSTREAM = "so101_tabletop"  # scene id of the unmodified EnvHub tabletop


def _hub():
    from ..sim.so101 import envhub_module

    return envhub_module()


def assets_dir() -> Path:
    return Path(_hub().repo_root()) / "robotstudio_so101" / "assets"


@lru_cache
def mesh_names() -> frozenset[str]:
    return frozenset(p.stem for p in assets_dir().glob("*.stl"))


def mesh_path(name: str) -> Path:
    """The STL for a mesh name from the asset list. Never builds a path from the request."""
    if name not in mesh_names():
        raise KeyError(f"mesh {name}")
    by_name = {p.stem: p for p in assets_dir().glob("*.stl")}
    return by_name[name]


def joint_label(i: int, name: str) -> str:
    return f"J{i + 1:02d} / {name.upper()}"


def _r(a: Any, n: int = 6) -> list[float]:
    return [round(float(v), n) for v in np.asarray(a).ravel()]


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _quat_mat(q: np.ndarray) -> np.ndarray:
    import mujoco

    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, dtype=float))
    return m.reshape(3, 3)


def _role(model, g: int, scene: bool) -> str:
    """What a primitive is, so the viewer can style it: table, leg, floor, wall, object, zone,
    obstacle, box, parked (the upstream cubes and box a generated scene hides) or motor."""
    import mujoco

    name = model.geom(g).name
    body = model.body(model.geom_bodyid[g]).name
    mat = model.mat(model.geom_matid[g]).name if model.geom_matid[g] >= 0 else ""
    hub = _hub()
    if name.startswith("zone_"):
        return "zone"
    if name.startswith("obs_"):
        return "obstacle"
    if body.startswith("obj_"):
        return "object"
    if body in hub.CUBE_BODIES or body == hub.BOX_BODY:
        return "parked" if scene else ("box" if body == hub.BOX_BODY else "object")
    if body != "world":
        return "motor"  # small visual primitives on the arm (wrist camera board and lens)
    if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
        return "floor"
    if name == "table":
        return "table"
    return {"table_leg": "leg", "wall": "wall"}.get(mat, "prop")


def _describe(scene: Path, digest: str, is_project: bool) -> dict[str, Any]:
    import mujoco

    hub = _hub()
    root = Path(hub.repo_root())
    model = hub.build_model(scene, root)  # render-free: compiles the scene with the arm attached
    paths = model.paths if isinstance(model.paths, bytes) else model.paths.encode()
    meshes, prims = [], []
    for g in range(model.ngeom):
        gtype = int(model.geom_type[g])
        group = int(model.geom_group[g])
        body = model.body(model.geom_bodyid[g]).name
        if gtype == mujoco.mjtGeom.mjGEOM_MESH:
            if group != 2:  # 2 = visual; 3/4 are collision proxies
                continue
            mid = int(model.geom_dataid[g])
            file = paths[int(model.mesh_pathadr[mid]):].split(b"\x00")[0].decode()
            mat = model.mat(model.geom_matid[g]).name if model.geom_matid[g] >= 0 else ""
            # raw STL -> body frame: geom * inverse(mesh frame)
            qg, qm = model.geom_quat[g], model.mesh_quat[mid]
            q = _quat_mul(qg, np.array([qm[0], -qm[1], -qm[2], -qm[3]]))
            r = _quat_mat(q)
            p = model.geom_pos[g] - r @ model.mesh_pos[mid]
            meshes.append({"body": body, "mesh": Path(file).stem, "pos": _r(p), "quat": _r(q),
                           "scale": _r(model.mesh_scale[mid]),
                           "role": "motor" if MOTOR_MATERIAL in (mat or file) else "arm"})
        elif group <= 2:
            prims.append({"name": model.geom(g).name or None, "body": body,
                          "type": mujoco.mjtGeom(gtype).name.removeprefix("mjGEOM_").lower(),
                          "size": _r(model.geom_size[g]), "pos": _r(model.geom_pos[g]),
                          "quat": _r(model.geom_quat[g]), "rgba": _r(model.geom_rgba[g], 3)
                          if model.geom_matid[g] < 0 else _r(model.mat_rgba[model.geom_matid[g]], 3),
                          "role": _role(model, g, is_project)})
    joints = []
    for i, name in enumerate(ARM_JOINTS):
        j = model.joint(name)
        joints.append({"name": name, "label": joint_label(i, name), "body": model.body(j.bodyid[0]).name,
                       "range_deg": _r(np.degrees(model.jnt_range[j.id]), 2)})
    site = model.site(TCP_SITE)
    return {
        "scene": digest,
        "bodies": [model.body(b).name for b in range(model.nbody)],
        "meshes": meshes,
        "primitives": prims,
        "joints": joints,
        "tcp": {"site": TCP_SITE, "body": model.body(site.bodyid[0]).name, "pos": _r(site.pos)},
        "units": {"length": "m", "angle": "deg", "quat": "wxyz", "up": "+z"},
    }


def _cached_model(scene: Path, digest: str, is_project: bool) -> dict[str, Any]:
    key = str(scene)
    with _lock:
        hit = _models.get(key)
    if hit is None:
        hit = _describe(scene, digest, is_project)
        with _lock:
            _models[key] = hit
            while len(_models) > MAX_CACHED:
                _models.pop(next(iter(_models)))
    return hit


async def describe_model(ctx: Ctx, p: Project | None = None) -> dict[str, Any]:
    """The upstream SO-101 tabletop, or a project's built scene when ``p`` is given."""
    if p is None:
        hub = _hub()
        scene, digest, is_project = Path(hub.repo_root()) / hub.SCENE_FILE, UPSTREAM, False
    else:
        info = intake_service.scene_info(ctx, p)
        scene, digest, is_project = info.path, info.digest, True
        if not scene.is_file():
            raise intake_service.SceneNotBuilt("the scene file is missing; call build_simulation again")
    out = await intake_service.on_gl(_cached_model, scene, digest, is_project)
    out = dict(out)
    if p is not None:
        out["project_id"] = p.id
        out["pick"] = p.scene.get("pick") if p.scene else None
    return out


def _read(env, scene: str, seed: int) -> dict[str, Any]:
    import mujoco

    m, d = env.world.model, env.world.data
    bodies = {m.body(b).name: {"pos": _r(d.xpos[b]), "quat": _r(d.xquat[b])} for b in range(1, m.nbody)}
    joints = {n: round(float(np.degrees(d.qpos[m.jnt_qposadr[m.joint(n).id]])), 3) for n in ARM_JOINTS}
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, TCP_SITE)
    return {"scene": scene, "seed": seed, "source": "reset", "time": float(d.time), "bodies": bodies,
            "joints_deg": joints, "tcp": _r(d.site_xpos[sid])}


def _cache(key: tuple, fn) -> dict[str, Any]:
    with _lock:
        hit = _states.get(key)
    if hit is None:
        hit = fn()
        with _lock:
            _states[key] = hit
            while len(_states) > MAX_CACHED:
                _states.pop(next(iter(_states)))
    return hit


def _scene_state(info: SceneInfo, seed: int) -> dict[str, Any]:
    def run():
        with intake_service._render_lock:  # the cached env is shared with capture_frame
            env = intake_service._env(info)
            env.reset(seed)
            out = _read(env, info.digest, seed)
        return {**out, "pick": info.pick, "pick_body": info.objects.get(info.pick) if info.pick else None}

    return _cache((info.digest, seed), run)


def _upstream_state(region: str, lighting: str, seed: int) -> dict[str, Any]:
    """The upstream tabletop under a plan condition: what the legacy /sim/frame preview renders."""
    from ..sim.so101 import PickPlaceEnv, Variation

    def run():
        with intake_service._render_lock:
            # Kept in the scene-env LRU so it is closed on the GL thread, like every other env.
            env = intake_service._envs.get(UPSTREAM)
            if env is None:
                env = intake_service._envs[UPSTREAM] = PickPlaceEnv()
            intake_service._envs.move_to_end(UPSTREAM)
            env.reset(seed, Variation(cube_region=region, lighting=lighting))
            out = _read(env, UPSTREAM, seed)
        return {**out, "pick": "red_cube", "pick_body": "red_cube"}

    return _cache((UPSTREAM, region, lighting, seed), run)


async def scene_state(ctx: Ctx, p: Project, seed: int = 0, cube_region: str = "center",
                      lighting: str = "normal") -> dict[str, Any]:
    """Body world poses after ``reset(seed)``: the project's built scene (what capture_frame shows), or,
    before a scene is built, the upstream tabletop under ``cube_region`` / ``lighting``."""
    from ..sim.so101 import LIGHT_SCALE, REGION_XY

    if p.scene:
        out = await intake_service.on_gl(_scene_state, intake_service.scene_info(ctx, p), seed)
    else:
        if cube_region not in REGION_XY or lighting not in LIGHT_SCALE:
            raise ValueError(f"cube_region must be one of {', '.join(REGION_XY)}; "
                             f"lighting one of {', '.join(LIGHT_SCALE)}")
        out = await intake_service.on_gl(_upstream_state, cube_region, lighting, seed)
    return {**out, "project_id": p.id}
