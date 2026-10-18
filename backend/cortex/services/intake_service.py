"""Spec v2 intake on a project: start, patch, build the sim scene, look at it.

A calling agent (MCP or studio) describes its arm, cameras, objects, zones, environment and
task through ``update_intake``. ``build_simulation`` turns the completed spec into a MuJoCo scene
and runs the feasibility and reach checks. ``capture_frame`` renders a camera so the agent can
check that the scene matches what it described, and patch it if not.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ..context import Ctx
from ..intake import IntakeSpec, apply_patch, start, status
from ..models import CreateProjectRequest, Mode, Project
from ..sim.scene import SceneInfo, build_scene

CAMERAS = ("top", "front", "wrist_cam")
_render_lock = threading.Lock()
_envs: OrderedDict[str, Any] = OrderedDict()  # scene digest -> SceneEnv, most recently used last
MAX_ENVS = 2  # each holds a MuJoCo model and a GL context
# OpenGL contexts belong to the thread that made them (WGL on Windows is strict about it), so every
# scene env is created, rendered and closed on this one thread.
_gl = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cortex-gl")


async def _on_gl(fn, *args):
    return await asyncio.get_running_loop().run_in_executor(_gl, fn, *args)


def _close_all() -> None:
    for env in _envs.values():
        env.close()
    _envs.clear()


def shutdown() -> None:
    """Close cached scene envs on the GL thread (call from the app's shutdown)."""
    _gl.submit(_close_all).result(timeout=30)


_locks: dict[str, asyncio.Lock] = {}


def project_lock(pid: str) -> asyncio.Lock:
    """Serializes read-modify-save of one project's intake and scene, so concurrent patches and
    builds never overwrite each other."""
    return _locks.setdefault(pid, asyncio.Lock())


def intake_digest(spec: IntakeSpec) -> str:
    """Hash of everything that shapes the scene; free-text description and the prompt don't."""
    d = spec.model_dump(mode="json", exclude={"prompt": True, "environment": {"description"}})
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]


def scene_stale(p: Project) -> bool:
    return p.scene is not None and (p.intake is None or p.scene.get("intake_digest") != intake_digest(p.intake))


async def _reload(ctx: Ctx, p: Project) -> Project:
    fresh = await ctx.repo.get("projects", p.id, Project)
    return fresh or p


class SceneNotBuilt(FileNotFoundError):
    """capture_frame and later stages need build_simulation first."""


async def start_intake(ctx: Ctx, prompt: str, mode: str, actor: str) -> tuple[Project, dict[str, Any]]:
    from .agent_orchestrator import create_project  # import here: the orchestrator imports this module

    spec = start(prompt, mode)
    p = await create_project(ctx, CreateProjectRequest(task_prompt=prompt or "Pick and place.", mode=Mode(mode)),
                             actor)
    p.intake = spec
    await ctx.repo.save("projects", p, status="active")
    await ctx.audit(actor, "start_intake", p.id, {"prompt": prompt, "mode": mode}, "ok",
                    f"{len(spec.objects)} objects seeded from the prompt")
    return p, status(spec).model_dump(mode="json")


async def update_intake(ctx: Ctx, p: Project, patch: dict[str, Any], actor: str) -> dict[str, Any]:
    async with project_lock(p.id):
        p = await _reload(ctx, p)
        before = p.intake or start(p.task_prompt, p.mode.value)
        spec = apply_patch(before, patch)
        p.intake = spec
        await ctx.repo.save("projects", p, status="active")
    st = status(spec, before)
    await ctx.audit(actor, "update_intake", p.id, patch, "ok", f"changed {', '.join(st.changed) or 'nothing'}")
    out = st.model_dump(mode="json")
    out["scene_built"], out["scene_stale"] = p.scene is not None, scene_stale(p)
    if out["scene_stale"]:
        out["notes"].append("The intake changed what the scene is built from: call build_simulation again.")
    return out


def _intake(p: Project) -> IntakeSpec:
    if p.intake is None:
        raise ValueError("this project has no intake yet; call start_intake / update_intake first")
    return p.intake


async def build_simulation(ctx: Ctx, p: Project, actor: str) -> dict[str, Any]:
    async with project_lock(p.id):
        p = await _reload(ctx, p)
        spec = _intake(p)
        st = status(spec)
        if not st.complete:
            open_q = [q.path for q in st.questions if q.required]
            raise ValueError(f"intake is incomplete; answer: {', '.join(open_q)}")
        out_dir = ctx.local(f"projects/{p.id}/scene")
        info = await asyncio.to_thread(build_scene, spec, out_dir)
        report = await _on_gl(_feasibility, spec, info)
        # Patches wait on the lock, so the intake saved here is the one the scene was built from.
        p.scene = {**info.to_dict(ctx.data_dir), "feasibility": report, "intake_digest": intake_digest(spec)}
        await ctx.repo.save("projects", p, status="active")
    preview = await capture_frame(ctx, p, "top")
    await ctx.audit(actor, "build_simulation", p.id, {}, "ok", f"scene {info.digest}: {report['summary']}")
    return {"scene": p.scene, "feasibility": report, "preview": preview}


def _feasibility(spec: IntakeSpec, info: SceneInfo) -> dict[str, Any]:
    from ..sim.feasibility import check

    with _render_lock:
        return check(spec, info, _env(info)).model_dump(mode="json")


def _env(info: SceneInfo):
    from ..sim.scene_env import SceneEnv

    env = _envs.get(info.digest)
    if env is None:
        env = _envs[info.digest] = SceneEnv(info)
        env.reset(0)
        while len(_envs) > MAX_ENVS:  # runs on the GL thread, so closing here is safe
            _envs.popitem(last=False)[1].close()
    _envs.move_to_end(info.digest)
    return env


async def on_gl(fn, *args):
    """Run a function that touches MuJoCo rendering on the one GL thread."""
    return await _on_gl(fn, *args)


def scene_info(ctx: Ctx, p: Project) -> SceneInfo:
    if not p.scene:
        raise SceneNotBuilt("no sim scene yet; call build_simulation (after the intake is complete)")
    return SceneInfo.from_dict(p.scene, ctx.data_dir)


def _render(info: SceneInfo, camera: str, seed: int, size: int) -> bytes:
    import mujoco
    from PIL import Image

    with _render_lock:
        env = _env(info)
        env.reset(seed)
        r = mujoco.Renderer(env.world.model, size, size)
        try:
            r.update_scene(env.world.data, camera=camera)
            img = r.render()
        finally:
            r.close()
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=85)
    return buf.getvalue()


async def capture_frame(ctx: Ctx, p: Project, camera: str = "top", seed: int = 0, size: int = 480,
                        inline: bool = True) -> dict[str, Any]:
    """Render a scene camera. MVP: always the sim. (A real camera source plugs in here post-MVP.)"""
    if camera not in CAMERAS:
        raise ValueError(f"camera must be one of {', '.join(CAMERAS)}")
    info = scene_info(ctx, p)
    jpg = await _on_gl(_render, info, camera, seed, size)
    key = f"projects/{p.id}/scene/frame_{camera}_{info.digest}_{seed}.jpg"
    await ctx.blobs.put_bytes(key, jpg, "image/jpeg")
    out = {"camera": camera, "source": "sim", "scene": info.digest, "key": key, "url": await ctx.media_url(key)}
    if inline:
        out["image_b64"] = base64.b64encode(jpg).decode()
    return out
